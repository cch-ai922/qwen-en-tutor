"""Non-Latin-script filter.

English conversational-tutor training data should never contain non-Latin
glyphs (CJK / Cyrillic / Arabic / Devanagari, etc.) - dialogue body is
English, and names / places must be in romanized form. The prompt says so
explicitly, but teacher models occasionally ignore the instruction, so we
add a mechanical guard.

The filter scans these locations in every SFT / DPO / Evaluation example:
  - user_role.name, model_role.name (when present)
  - every message's content
  - DPO chosen / rejected message content
  - SFT system_prompt (usually safe, but checked as a precaution)

Reject decision: any non-Latin character match fails the example. ``score``
is the inverse of the hit count (informational; not used as a threshold -
pass / fail is binary).
"""

from __future__ import annotations

import re

from qwen_tutor.generation.filters.base import (
    Filter,
    FilterableExample,
    FilterResult,
)
from qwen_tutor.schemas import DPOExample, EvaluationExample, SFTExample

# Non-Latin-script Unicode blocks. Starts narrow and can be widened later.
# Excludes Latin-1 / Latin Extended / whitespace / punctuation / digits so
# only scripts that should never appear in English dialogue are matched.
_NON_LATIN_RE = re.compile(
    "["
    "぀-ゟ"   # Hiragana
    "゠-ヿ"   # Katakana
    "ㇰ-ㇿ"   # Katakana Phonetic Extensions
    "　-〿"   # CJK Symbols and Punctuation
    "㐀-䶿"   # CJK Unified Ideographs Extension A
    "一-鿿"   # CJK Unified Ideographs
    "豈-﫿"   # CJK Compatibility Ideographs
    "＀-￯"   # Halfwidth and Fullwidth Forms
    "가-힯"   # Hangul Syllables
    "ᄀ-ᇿ"   # Hangul Jamo
    "㄰-㆏"   # Hangul Compatibility Jamo
    "Ѐ-ӿ"   # Cyrillic
    "Ԁ-ԯ"   # Cyrillic Supplement
    "؀-ۿ"   # Arabic
    "ݐ-ݿ"   # Arabic Supplement
    "ࢠ-ࣿ"   # Arabic Extended-A
    "ﭐ-﷿"   # Arabic Presentation Forms-A
    "ﹰ-﻿"   # Arabic Presentation Forms-B
    "֐-׿"   # Hebrew
    "ऀ-ॿ"   # Devanagari
    "ঀ-৿"   # Bengali
    "਀-੿"   # Gurmukhi
    "฀-๿"   # Thai
    "]"
)

# Used to slice eval-mode assistant turns into post-`</think>` content.
# The `<think>` reasoning is private examiner thought — the examiner
# SHOULD reason about non-Latin content to identify it. Only the JSON
# portion is user-facing and must be script-clean.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def _scan(text: str) -> list[str]:
    """Return a deduplicated list of non-Latin characters found in ``text``.

    Empty / None inputs return an empty list. The list is bounded to the
    first 10 distinct offenders so error messages stay readable.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in _NON_LATIN_RE.finditer(text):
        ch = m.group(0)
        if ch not in seen:
            seen[ch] = None
            if len(seen) >= 10:
                break
    return list(seen.keys())


class NonLatinScriptFilter(Filter):
    """Reject any record whose role names, messages, or system prompt
    contain non-Latin script characters (CJK / Cyrillic / Arabic /
    Devanagari / Hebrew / Thai / Bengali / Gurmukhi).

    English-language tutor data must be in Latin script — including
    romanized forms of foreign names (``Li Na`` not ``李娜``). The
    locale_instruction_block makes this rule explicit to the teacher,
    but small or distilled models sometimes ignore it. This filter is
    the mechanical guard.
    """

    name = "non_latin_script"

    async def check(self, example: FilterableExample) -> FilterResult:
        fields_to_scan: list[tuple[str, str]] = []

        # Metadata role names (when present — only SFT/DPO have them).
        md = getattr(example, "metadata", None)
        user_role = getattr(md, "user_role", None) if md is not None else None
        model_role = getattr(md, "model_role", None) if md is not None else None
        if user_role is not None and getattr(user_role, "name", None):
            fields_to_scan.append(("user_role.name", user_role.name))
        if model_role is not None and getattr(model_role, "name", None):
            fields_to_scan.append(("model_role.name", model_role.name))

        # System prompt (rendered, may contain locale strings — should still be Latin).
        sys_prompt = getattr(example, "system_prompt", None)
        if isinstance(sys_prompt, str):
            fields_to_scan.append(("system_prompt", sys_prompt))

        # ``speaks_l1`` SFT examples are SUPPOSED to have one user turn in
        # the learner's L1 (non-Latin script). The SpeaksL1SanityFilter
        # validates that turn separately, so here we exempt user turns
        # from the script check on these records and only enforce the
        # script rule on assistant + tutor content.
        gen_meta = (md.generation or {}) if md is not None and hasattr(md, "generation") else {}
        is_speaks_l1 = isinstance(gen_meta, dict) and gen_meta.get("language_trigger") == "speaks_l1"

        # Eval examples are built from already-filtered SFT transcripts —
        # the user turn is a verbatim render of an SFT dialogue that
        # already passed this filter at the SFT stage. Re-scanning it
        # here would false-positive on input context the assistant only
        # has to *judge*, not produce. ``source_dialogue_id`` is the
        # unique-to-EvaluationMetadata field we duck-type on.
        is_eval = md is not None and hasattr(md, "source_dialogue_id")

        # Redirect-family SFT (single-shot + 4 persistent 3-strike streams):
        # what the model learns to PRODUCE is the assistant turn; the user
        # turn is training-time CONTEXT that the model never emits. A
        # stray non-Latin glyph the teacher may have leaked into a user
        # turn doesn't teach the model to produce non-Latin — it just
        # discards an otherwise-valid example. Mirror the same scoping
        # banned_terms uses: skip user turns for scenario_type=="redirect".
        scenario_type = getattr(md, "scenario_type", None) if md is not None else None
        is_redirect = scenario_type == "redirect"

        # Messages (SFTExample, EvaluationExample) — list of Message objects.
        msgs = getattr(example, "messages", None) or []
        for i, m in enumerate(msgs):
            if is_speaks_l1 and m.role == "user":
                continue  # L1 user turn is allowed/required for speaks_l1
            if is_redirect and m.role == "user":
                continue  # user turns are training-time context only
            if is_eval and m.role == "user":
                continue  # transcript is pre-filtered
            content = m.content
            # For eval examples, the assistant turn is
            # ``<think>...</think>{json}``. The <think> block is private
            # examiner reasoning — the examiner SHOULD reason about
            # banned/non-Latin content to identify it, so filtering that
            # body defeats the purpose. Only the JSON portion
            # (post-</think>) is the user-facing structured output that
            # the trained student will produce; that's what must be
            # script-clean. Slice to post-</think> for eval-mode
            # assistant turns; leave non-eval assistant turns intact.
            if is_eval and m.role == "assistant":
                close_match = _THINK_CLOSE_RE.search(content)
                if close_match is not None:
                    content = content[close_match.end():]
            fields_to_scan.append((f"messages[{i}].content", content))

        # DPO-specific fields. ALL THREE are skipped here — non_latin_script
        # is a no-op on DPO records by design:
        #   * prompt_messages → derived from already-filtered SFT context.
        #   * chosen → register pairs use the original SFT assistant turn
        #     (from ``sft_filtered/``); on-policy pairs use the teacher's
        #     original turn (also from ``sft_filtered/``). Either way the
        #     chosen content already passed this filter at the SFT stage.
        #   * rejected → the whole point of rejected is to be the bad-pattern
        #     reference. Non-Latin script there is exactly the signal DPO
        #     uses to push the model away from producing it.
        if isinstance(example, DPOExample):
            pass  # see comment above — DPO content is pre-filtered upstream

        offenders: list[tuple[str, list[str]]] = []
        total = 0
        for label, text in fields_to_scan:
            hits = _scan(text)
            if hits:
                offenders.append((label, hits))
                total += len(hits)

        if not offenders:
            return FilterResult(
                passed=True,
                score=1.0,
                reason=None,
                metadata={"scanned_fields": len(fields_to_scan)},
            )

        # Build a short reason naming the first 3 offending fields.
        sample = "; ".join(
            f"{label}: {''.join(chars[:5])}" for label, chars in offenders[:3]
        )
        return FilterResult(
            passed=False,
            score=0.0,
            reason=(
                f"non-Latin script characters detected in {len(offenders)} "
                f"field(s): {sample}"
            ),
            metadata={
                "n_offending_fields": len(offenders),
                "n_distinct_chars": total,
            },
        )


__all__ = ["NonLatinScriptFilter"]
