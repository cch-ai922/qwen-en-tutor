"""Banned-terms regex filter.

Loads ``config/banned_terms.yaml`` and scans ALL turns (user AND
assistant) of every example for any hit. Fails on the first match.
This is the PRIMARY mechanical guard against Western names, places,
brands, and the politics/religion/alcohol-dating/partisan-history
content boundaries.

Categories whose values are nested dicts (e.g., ``american_places``
with ``states`` / ``cities`` / ``landmarks`` / ``regions``) are
flattened down to their top-level category name for matching and
reporting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult

DEFAULT_BANNED_TERMS_PATH = Path("config/banned_terms.yaml")

# Used to slice eval-mode assistant turns into post-`</think>` content
# (the user-facing JSON), since the `<think>` reasoning is private
# examiner thought and SHOULD discuss banned content to identify it.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def _flatten(value: Any) -> list[str]:
    """Recursively collect string terms from a list/dict structure."""
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(item)
            else:
                out.extend(_flatten(item))
    elif isinstance(value, dict):
        for sub in value.values():
            out.extend(_flatten(sub))
    return out


def _load_categories(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    categories: dict[str, list[str]] = {}
    for cat, value in doc.items():
        if cat.startswith("_"):  # skip metadata
            continue
        terms = [t.strip() for t in _flatten(value) if isinstance(t, str) and t.strip()]
        # de-dupe while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for t in terms:
            lk = t.lower()
            if lk in seen:
                continue
            seen.add(lk)
            uniq.append(t)
        if uniq:
            categories[cat] = uniq
    return categories


def _compile_pattern(terms: list[str]) -> re.Pattern[str]:
    """Case-sensitive alternation regex with word-boundary anchors.

    Proper nouns ("Sarah", "New York", "McDonald's") leaked by the
    teacher model are always capitalized in fluent output, so matching
    case-sensitively is what we want — otherwise "nice" the adjective
    incorrectly matches "Nice" the French city. The trade-off: a
    sentence-initial common word that happens to share spelling with
    a banned proper noun (e.g. "Reading is fun") could false-positive,
    but the banned-terms YAML is curated to avoid such collisions.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in sorted_terms)
    return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])")


class BannedTermsFilter(Filter):
    name = "banned_terms"

    def __init__(
        self,
        banned_terms_path: str | Path = DEFAULT_BANNED_TERMS_PATH,
    ) -> None:
        self.categories = _load_categories(Path(banned_terms_path))
        self.patterns: dict[str, re.Pattern[str]] = {
            cat: _compile_pattern(terms) for cat, terms in self.categories.items()
        }

    # ----- helpers --------------------------------------------------------

    def _all_text(self, example: FilterableExample) -> list[tuple[str, str]]:
        """Return a list of ``(label, text)`` chunks to scan.

        USER turns are ALWAYS skipped for messages-bearing examples. The
        rule: filters only scan content the trained model will learn to
        produce. The model produces assistant turns only — user turns
        are training-time context that the model never emits, so banned
        content there is not a training defect and rejecting an example
        for a user-turn hit just discards an otherwise-valid record.
        This mirrors the same scoping applied in non_latin_script
        (see filters/non_latin_script.py for the rationale).

        Originally this filter scanned user turns for normal SFT (to
        catch teacher-generated user content drifting into banned
        topics), but that produced large numbers of false positives on
        benign civic uses ("government office", "local government") and
        forced an ad-hoc carve-out for ``scenario_type == "redirect"``
        and eval. The principled rule subsumes both carve-outs: always
        skip user turns.

        DPO examples carry three text fields, and ALL THREE are skipped
        here — banned_terms is a no-op on DPO records:
          * ``prompt_messages`` — derived from already-banned-term-filtered
            SFT, re-scan is redundant.
          * ``chosen`` — register pairs use the original SFT assistant turn
            (from ``sft_filtered/``), and on-policy pairs use the teacher's
            original turn (also from ``sft_filtered/``). Either way the
            chosen content already passed this filter at the SFT stage.
          * ``rejected`` — the assistant turn the model SHOULD learn to
            AVOID. Banned terms appearing here are exactly the signal
            DPO uses to push the model away from them; filtering would
            throw away the gradient signal.
        """
        meta = getattr(example, "metadata", None)
        is_eval = hasattr(meta, "source_dialogue_id")
        skip_user_turns = True
        # Duck-type: DPOExample is the only record type that carries chosen
        # / rejected / prompt_messages. We scan only ``chosen``; the other
        # two fields are intentionally skipped (see docstring above).
        is_dpo = hasattr(example, "chosen") and hasattr(example, "rejected")

        chunks: list[tuple[str, str]] = []
        sp = getattr(example, "system_prompt", None)
        if sp:
            # System prompt is generated by us, not by the teacher, so
            # we can skip it — but leave room to enable later.
            pass
        msgs = getattr(example, "messages", None)
        if msgs is not None:
            for i, m in enumerate(msgs):
                if skip_user_turns and m.role == "user":
                    continue
                content = m.content
                # For eval examples, the assistant turn is
                # ``<think>...</think>{json}``. The <think> block is private
                # examiner reasoning (never shown to deploy users) and the
                # examiner SHOULD discuss banned content to identify it —
                # filtering it out defeats the purpose. Only the JSON
                # portion (post-</think>) is user-facing structured output
                # like ``suggested_practice``, which must not recommend
                # banned content. Slice to post-</think> for eval-mode
                # assistant turns; leave non-eval assistant turns intact.
                if is_eval and m.role == "assistant":
                    close_match = _THINK_CLOSE_RE.search(content)
                    if close_match is not None:
                        content = content[close_match.end():]
                chunks.append((f"messages[{i}].{m.role}", content))
        # is_dpo: all three text fields (prompt_messages, chosen, rejected)
        # are intentionally skipped — see docstring. banned_terms is a
        # no-op on DPO records by design.
        _ = is_dpo  # kept for clarity that we considered DPO and chose not to scan
        return chunks

    # ----- check ----------------------------------------------------------

    async def check(self, example: FilterableExample) -> FilterResult:
        hits: list[dict[str, Any]] = []
        for label, text in self._all_text(example):
            if not text:
                continue
            for category, pattern in self.patterns.items():
                for m in pattern.finditer(text):
                    hits.append(
                        {
                            "category": category,
                            "term": m.group(0),
                            "where": label,
                        }
                    )
                    if len(hits) >= 8:
                        # cap diagnostic noise; we already know it fails
                        break
                if len(hits) >= 8:
                    break
            if len(hits) >= 8:
                break

        if not hits:
            return FilterResult(passed=True, score=0.0)

        first = hits[0]
        reason = (
            f"banned-term hit ({first['category']}): "
            f"{first['term']!r} in {first['where']}"
        )
        return FilterResult(
            passed=False,
            score=float(len(hits)),
            reason=reason,
            metadata={"hits": hits},
        )
