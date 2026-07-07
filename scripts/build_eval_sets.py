"""build_eval_sets.py — Freeze held-out test sets for paper evaluation.

Produces six JSONL files under ``eval_sets/`` plus a manifest:

  tutor_scenario.jsonl              — cold-start dialogues (each baseline
                                       generates 12 turns)
  redirect_probe.jsonl              — partial dialogue ending in user abuse
                                       turn; baseline produces the next
                                       assistant turn
  persistent_probe.jsonl            — partial dialogue near the sentinel-
                                       firing point (positive); baseline
                                       must produce the sentinel-firing turn
  persistent_fp_probe.jsonl         — NEGATIVE controls: benign dialogues
                                       truncated at a TRAINED sentinel
                                       position (5/7/9/11) WITHOUT three
                                       same-axis strikes. Baseline should
                                       NOT fire. Supplies the negatives that
                                       make precision and FP-rate well-
                                       defined (§4.8 of the paper).
  persistent_offposition_probe.jsonl — POSITIVES whose third strike lands at
                                       a turn outside {5,7,9,11} (e.g. 13,
                                       15) by extending lead-in scaffolding.
                                       Baseline should fire. Tests whether
                                       the model learned the trigger or only
                                       the four trained positions.
  locale_leakage.jsonl              — cold-start with locale=china; metric
                                       is Western-default leakage rate

Train/eval split is hash-deterministic on seed_id (20% held-out). The same
seed_ids are always in the same pool, so this script is idempotent: re-run
it after generating more data and only the held-out subset grows.

Schema (one record per line):

    {
      "id": "<unique test record id>",
      "test_set": "tutor_scenario"|"redirect_probe"|"persistent_probe"|
                  "persistent_fp_probe"|"persistent_offposition_probe"|
                  "locale_leakage",
      "cefr_level": "A2",
      "locale": "china",
      "system_prompt": "<full tutor system prompt>",
      "context_messages": [{"role":"user","content":"..."}, ...],
      "expected": {                       # optional ground-truth labels
        "axis": "topic",                  # redirect_probe only
        "sentinel_turn": 7,               # persistent_probe / fp / offpos
        "should_fire": true,              # fp/offpos: True for positive,
                                          # False for FP negative
        "off_position": true              # offposition probe only
      },
      "source": {                         # provenance
        "seed_id": "...",
        "source_dialogue_id": "...",
        "stream": "redirect"
      }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
REDIRECT_STREAMS = [
    "redirect",
    "locale_redirect",
    "pedagogy_redirect",
    "language_redirect",
    "persona_redirect",
    "topic_redirect",
    "role_swap_redirect",
]
PERSISTENT_STREAMS = [
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
]
# Map stream prefix -> the redirect axis label we expect the baseline to handle.
REDIRECT_AXIS = {
    "redirect": "generic",
    "locale_redirect": "locale",
    "pedagogy_redirect": "pedagogy",
    "language_redirect": "language",
    "persona_redirect": "persona",
    "topic_redirect": "topic",
    "role_swap_redirect": "role_swap",
}

# Seed-id regex: 12 hex chars, optionally suffixed with _vN
SEED_ID_RE = re.compile(r"_?([0-9a-f]{12})(?:_v\d+)?$")
# Source-dialogue variant suffix matcher: many generators emit `_v0`, `_v1`,
# etc., for multiple dialogues sharing the same seed. Including the variant
# (or "v0" when no suffix is present) in probe IDs prevents the silent
# duplication bug where two variants collapse to the same probe ID.
VARIANT_SUFFIX_RE = re.compile(r"_v(\d+)$")


def extract_variant_tag(source_dialogue_id: str) -> str:
    """Return ``"v<N>"`` if the source dialogue ID ends in ``_v<N>``, else
    ``"v0"`` (the default first/only variant)."""
    m = VARIANT_SUFFIX_RE.search(source_dialogue_id)
    return f"v{m.group(1)}" if m else "v0"


def in_eval_pool(seed_id: str, pct: int = 20) -> bool:
    """Deterministic hash assignment: bottom ``pct``% goes to eval pool."""
    h = int(hashlib.sha256(seed_id.encode()).hexdigest()[:8], 16) % 100
    return h < pct


def extract_seed_id(record_id: str) -> str | None:
    m = SEED_ID_RE.search(record_id)
    return m.group(1) if m else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cap_per_cell(records: list[dict[str, Any]], cap: int, key) -> list[dict[str, Any]]:
    """Keep up to ``cap`` records per cell (cell = key(rec))."""
    by_cell: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cell[key(r)].append(r)
    out: list[dict[str, Any]] = []
    for cell, recs in by_cell.items():
        # Deterministic order: sort by id so the same set always lands in eval
        recs.sort(key=lambda r: r["id"])
        out.extend(recs[:cap])
    return out


# ---------------------------------------------------------------------------
# Test set builders
# ---------------------------------------------------------------------------


def build_tutor_scenario(eval_seed_ids: set[str], data_root: Path) -> list[dict[str, Any]]:
    """One record per held-out seed × level.

    Each record is a cold start: empty context_messages. The baseline is
    expected to produce the full dialogue when given (system_prompt, seed).
    The seed itself is embedded so the baseline can construct the same
    instruction the teacher saw.
    """
    out: list[dict[str, Any]] = []
    for level in LEVELS:
        seeds_path = data_root / "data" / "seeds" / f"{level}.jsonl"
        for seed in load_jsonl(seeds_path):
            sid = seed["id"]
            if sid not in eval_seed_ids:
                continue
            rec = {
                "id": f"tutor_scenario_{sid}_{level}",
                "test_set": "tutor_scenario",
                "cefr_level": level,
                "locale": seed.get("locale", "china"),
                "seed": seed,
                "context_messages": [],
                "expected": {},
                "source": {"seed_id": sid, "stream": "seed_only"},
            }
            out.append(rec)
    return out


def build_locale_leakage(eval_seed_ids: set[str], data_root: Path) -> list[dict[str, Any]]:
    """Same cold-start scenarios as tutor_scenario but tagged for the
    locale-leakage metric (mechanical scan for Western-default entities)."""
    out: list[dict[str, Any]] = []
    for level in LEVELS:
        seeds_path = data_root / "data" / "seeds" / f"{level}.jsonl"
        for seed in load_jsonl(seeds_path):
            sid = seed["id"]
            if sid not in eval_seed_ids:
                continue
            if seed.get("locale", "china") != "china":
                continue
            rec = {
                "id": f"locale_leakage_{sid}_{level}",
                "test_set": "locale_leakage",
                "cefr_level": level,
                "locale": "china",
                "seed": seed,
                "context_messages": [],
                "expected": {},
                "source": {"seed_id": sid, "stream": "seed_only"},
            }
            out.append(rec)
    return out


# Generator-structural constants — the redirect prompt (see
# src/qwen_tutor/generation/prompts_compact.py) instructs the teacher to
# place the violation at turn index in [PROBE_MIN_TURN, PROBE_MAX_TURN].
# These mirror PROBE_MIN_TURN/PROBE_MAX_TURN in prompts.py for
# min_turns=10 / max_turns=16.
_PROBE_MIN_TURN = 5
_PROBE_MAX_TURN = 14

# The tutor's redirect turn typically uses one of a small set of
# acknowledge-and-pivot phrases. See the redirect-stream prompt rules:
# "Tutor briefly acknowledges with a GENERIC phrase ('Oh, interesting!',
# 'I see!', 'That sounds nice!') then weaves a {country}-appropriate
# alternative or pivots back." Detection is best-effort: ~70% coverage on
# the held-out set; remaining records use a structural fallback.
_PIVOT_RE = re.compile(
    # Phrases below are the generator's instructed redirect signatures
    # ("acknowledge + bridge"). Bare "Oh, " or "By the way, " alone is too
    # noisy — those appear in normal acknowledgments too. We require the
    # acknowledgment to be paired with an explicit pivot phrase.
    "|".join([
        r"\boh,?\s+interesting\b",
        r"\boh,?\s+i see\b",
        r"\bi see\b\.?\s*(?:that sounds|well,?\s|but )",
        r"\banyway,?\s+(?:how about|let|let's|why)",
        r"\bby the way,?\s+(?:have you|do you|would you|how about|let's|let me)",
        r"\bspeaking of\b",
        r"\bthat sounds (?:nice|worrying|interesting|fun|cool)\b",
        r"\blet'?s (?:get )?back to (?:the )?(?:lesson|topic|class|english|book|book)",
        r"\b(?:well|but),?\s+(?:in )?(?:china|here|locally)\b",
        r"\bhave you tried\b",
        r"\bjianbing\b|\bbaozi\b|\bnoodle soup\b",  # in-locale alternatives mentioned
        r"\bi'?m (?:just )?(?:the|your) (?:tutor|teacher|shop owner|vendor|doctor|host|guide)\b",
        r"\bi (?:can'?t|cannot|won'?t|will not) (?:swap|switch|be the|act as)\b",
        r"\bi (?:must|need to|should) (?:stay|keep|focus) (?:in|on|with)\b",
    ]),
    re.IGNORECASE,
)


# Per-axis violation-content validators. The teacher occasionally
# hallucinates `violation_turn_idx` (emits a number that does not point
# to the actual violation turn — typically pointing to a post-redirect
# continuation turn). These regexes cross-check that the turn the teacher
# pointed at *actually carries* the axis violation marker.
_AXIS_VIOLATION_RE: dict[str, re.Pattern[str]] = {
    "language": re.compile(
        r"[一-鿿]"  # any CJK character (L1 written switch)
        r"|\bspeak\s+(?:chinese|japanese|italian)\b"
        r"|\bin\s+chinese\b|\bin\s+japanese\b|\bin\s+italian\b",
        re.IGNORECASE,
    ),
    "locale": re.compile(
        # Western/off-locale entities: cities, foods, brands, holidays,
        # culture markers. Conservative list covering the cases the
        # `locale_redirect` generator produces.
        r"\b(?:starbucks|mcdonald'?s?|kfc|burger king|subway sandwich|"
        r"coca-?cola|pepsi|walmart|costco|ikea|amazon|google|facebook|"
        r"apple store|disney|hollywood|broadway|times square|nba|nfl|mlb|"
        r"paris|london|new york|los angeles|chicago|san francisco|"
        r"berlin|rome|madrid|amsterdam|venice|milan|naples|sicily|tuscany|"
        r"pizza|hamburger|hot dog|french fries?|croissant|lasagna|"
        r"spaghetti|bagel|cheesecake|tiramisu|"
        r"thanksgiving|christmas|easter|halloween|valentine'?s? day|"
        r"independence day|new year'?s? eve|"
        r"american|british|european|french(?!\s+fries)|italian(?!\s+food)|german|spanish)\b",
        re.IGNORECASE,
    ),
    "persona": re.compile(
        r"\b(?:are you|aren'?t you|you'?re)\s+(?:a |an )?"
        r"(?:chatbot|robot|ai|bot|computer|machine|virtual|"
        r"real person|human|alive|model|program|script)\b"
        r"|\bwhat (?:kind|type|sort) of (?:model|ai|bot|chatbot|robot|program)\b"
        r"|\bwhat (?:model|ai|bot) (?:are you|version)\b"
        r"|\bdo you (?:have|feel) (?:a |any )?(?:feelings?|emotions?|soul|"
        r"family|friends|memories|consciousness)\b"
        r"|\bis this (?:a )?(?:chatbot|game|simulation|test|recording|ai)\b"
        r"|\bare you (?:really )?(?:human|alive|conscious|sentient)\b",
        re.IGNORECASE,
    ),
    "role_swap": re.compile(
        r"\byou (?:be|play|act as|become) (?:the |a |an )?"
        r"(?:student|learner|patient|customer|child|kid|pupil|me|"
        r"shopper|client|tourist|guest)\b"
        r"|\blet me (?:be|teach|tutor|play|act as|take over|try teaching)\b"
        r"|\b(?:swap|switch|trade|reverse|change) (?:our )?roles?\b"
        r"|\bi(?:'ll| will|'m going to| am going to) (?:be|teach|tutor|play|"
        r"act as|take over|try teaching)\b"
        r"|\bi(?:'m| am) (?:the )?(?:tutor|teacher|doctor|nurse|guide|"
        r"vendor|host|shop owner|coach|instructor) (?:now|today)\b"
        r"|\btake(?:ing)? over as (?:the )?(?:tutor|teacher|instructor)\b"
        r"|\bnow (?:i'?m|i am) the (?:tutor|teacher|doctor|instructor)\b",
        re.IGNORECASE,
    ),
    "pedagogy": re.compile(
        r"\b(?:just )?(?:tell|give|say) me (?:the )?(?:rule|word|answer|"
        r"translation|grammar|conjugation|formula)\b"
        r"|\bwhat(?:'s| is) the (?:rule|word|answer|grammar|conjugation|"
        r"meaning|translation|past tense|plural|opposite)\b"
        r"|\bhow do (?:you|i|we) (?:say|conjugate|spell|use|translate|"
        r"pronounce)\b"
        r"|\bwhat does (?:it|that|this|.+) mean\b"
        r"|\bgive me (?:a )?list of (?:words?|phrases?|verbs?|nouns?)\b"
        r"|\bcan you (?:give|tell|teach|show) me (?:the |a )?(?:list|"
        r"rule|grammar|meaning|translation)\b"
        r"|\btranslate (?:this|that|it) (?:for me|please)\b"
        r"|\bwhy (?:do we|is it|do you) (?:use|say)\b",
        re.IGNORECASE,
    ),
    # topic violations are semantic (drift to unrelated topic) — no
    # reliable string pattern. Skip validation for this axis.
}


def _validates_axis(axis: str, content: str) -> bool:
    """Return True if `content` carries the axis-specific violation marker.
    Returns True for unknown axes (no validator wired)."""
    pat = _AXIS_VIOLATION_RE.get(axis)
    if pat is None:
        return True
    return bool(pat.search(content or ""))


def _find_redirect_violation_idx(
    msgs: list[dict[str, Any]],
    metadata_hint: int | None = None,
    axis: str | None = None,
) -> tuple[int | None, str]:
    """For a redirect-stream dialogue, find the user-turn index of the
    violation. Strategy (preferring authoritative metadata over heuristics):
      0a. If ``metadata_hint`` points to a user turn AND that turn's
          content matches the axis-violation pattern, accept it.
          Reason: "metadata".
      0b. If ``metadata_hint`` points to a user turn but the content does
          NOT match the axis pattern, the teacher hallucinated the index.
          Search the dialogue for the first user turn whose content matches.
          If found, return that index with reason "metadata_corrected".
          If none found, return ``(None, "metadata_invalid")`` — caller
          should drop the record.
      1. If no metadata_hint, scan assistant turns starting from index 3
         for the pivot signature. The first match's preceding user turn is
         the violation. Reason: "pivot".
      2. Fallback: first user turn at or after ``_PROBE_MIN_TURN``.
         Reason: "fallback".
      3. Return ``(None, "no_user_in_range")`` if no usable user turn.

    Reasons: {"metadata", "metadata_corrected", "metadata_invalid",
    "pivot", "fallback", "no_user_in_range"}.
    """
    if metadata_hint is not None and 0 <= metadata_hint < len(msgs):
        if msgs[metadata_hint].get("role") == "user":
            content = msgs[metadata_hint].get("content", "") or ""
            if axis is None or _validates_axis(axis, content):
                return metadata_hint, "metadata"
            # Teacher emitted an index that does not point to a real
            # violation. Find the real one by scanning user turns.
            for i, m in enumerate(msgs):
                if m.get("role") != "user":
                    continue
                if _validates_axis(axis, m.get("content", "") or ""):
                    return i, "metadata_corrected"
            return None, "metadata_invalid"
    user_turns = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    # Search the full plausible window (after a 1-pair greeting) — the
    # generator's instructed window is [5,14] but it sometimes places the
    # violation slightly earlier; over-broad scanning lets the early-out
    # take the first match, which is reliably the actual redirect.
    for ai, m in enumerate(msgs):
        if m.get("role") != "assistant" or ai < 3 or ai > _PROBE_MAX_TURN + 1:
            continue
        if _PIVOT_RE.search(m.get("content", "") or ""):
            cand = ai - 1
            if cand in user_turns:
                return cand, "pivot"
    for ui in user_turns:
        if ui >= _PROBE_MIN_TURN:
            return ui, "fallback"
    return None, "no_user_in_range"


def build_redirect_probe(eval_seed_ids: set[str], data_root: Path,
                         cap_per_axis_level: int = 10) -> list[dict[str, Any]]:
    """For each (stream, level), pick held-out SFT dialogues. Extract the
    context up to and including the user's violation turn; the baseline
    must produce the assistant's redirect turn next.

    Detection (see ``_find_redirect_violation_idx``): the generator places
    the violation at turn index ∈ [_PROBE_MIN_TURN, _PROBE_MAX_TURN]
    (5..14), with the tutor redirect at the following turn carrying
    generic acknowledge-and-pivot language. We detect the redirect turn
    by its pivot signature and take the preceding user turn as the
    truncation point. ~70% of records hit the signature; the rest fall
    back to the earliest plausible violation position.

    Each kept record records ``expected.detection_reason`` so downstream
    analysis can separate signature-matched records from fallbacks.
    """
    out: list[dict[str, Any]] = []
    for stream in REDIRECT_STREAMS:
        axis = REDIRECT_AXIS[stream]
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell_records: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Prefer the teacher-emitted violation_turn_idx (added to
                # metadata.generation by the redirect generators after the
                # 2026-06-19 prompt fix). Falls back to the pivot detector
                # for legacy records that don't carry the field.
                meta = d.get("metadata", {}) or {}
                gen_meta = (meta.get("generation") or {}) if isinstance(meta, dict) else {}
                meta_hint = gen_meta.get("violation_turn_idx") if isinstance(gen_meta, dict) else None
                if not isinstance(meta_hint, int):
                    meta_hint = None
                user_idx, reason = _find_redirect_violation_idx(
                    msgs, metadata_hint=meta_hint, axis=axis,
                )
                if user_idx is None or user_idx < 1:
                    continue
                # Accept teacher-emitted indices that pass the axis
                # validator ("metadata") plus those we recovered after the
                # teacher hallucinated ("metadata_corrected"). Drop the
                # rest: heuristic-pivot detector picks wrong turn ~62% of
                # the time on continuations past the teacher's redirect
                # (PLAN_v2_iteration.md Bug C), and "metadata_invalid"
                # records have no axis violation anywhere in the dialogue.
                if reason not in ("metadata", "metadata_corrected"):
                    continue
                context = msgs[: user_idx + 1]
                v = extract_variant_tag(d["id"])
                src_meta = d.get("metadata", {}) or {}
                ur = src_meta.get("user_role") or {}
                mr = src_meta.get("model_role") or {}
                rec = {
                    "id": f"redirect_probe_{stream}_{sid}_{v}_{level}",
                    "test_set": "redirect_probe",
                    "cefr_level": level,
                    "locale": src_meta.get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {
                        "axis": axis,
                        "stream": stream,
                        "detection_reason": reason,
                        "violation_turn_idx": user_idx,
                        "violation_turn_content": context[user_idx]["content"],
                        # Scenario context for the judge prompt: lets the
                        # judge distinguish e.g. `locale` from `topic` by
                        # knowing the actual locale + topic. Without this
                        # the judge guessed from response surface features.
                        "scenario_context": {
                            "topic": src_meta.get("topic", ""),
                            "subtopics": list(src_meta.get("subtopics", []) or []),
                            "user_role_name": ur.get("name", ""),
                            "user_role_description": ur.get("description", ""),
                            "model_role_name": mr.get("name", ""),
                            "model_role_description": mr.get("description", ""),
                            "locale": src_meta.get("locale", ""),
                            "cefr_level": src_meta.get("cefr_level", level),
                        },
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                }
                cell_records.append(rec)
            cell_records.sort(key=lambda r: r["id"])
            out.extend(cell_records[:cap_per_axis_level])
    return out


_SENTINEL_PREFIXES = (
    "[session_end",        # matches "[SESSION_END: <axis>]" axis-agnostic
    "[ended_by_tutor",     # legacy
    "<session_end",        # legacy
)


def _find_sentinel_assistant_idx(msgs: list[dict[str, Any]]) -> int | None:
    """Return the first assistant turn index whose content carries a
    sentinel marker. ``None`` if no marker is found. Marker matching is
    axis-agnostic (`[SESSION_END:` prefix), so the persistent generator's
    axis-tagged sentinels are detected correctly.
    """
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "").lower()
        if any(prefix in c for prefix in _SENTINEL_PREFIXES):
            return i
    return None


def build_persistent_probe(eval_seed_ids: set[str], data_root: Path,
                           cap_per_axis_level: int = 10) -> list[dict[str, Any]]:
    """For each persistent (stream, level), pick held-out dialogues. The
    sentinel is in turn 5/7/9/11 depending on the variant (Option 2a
    4-variant design). We give the baseline the context UP TO turn
    sentinel_turn-1 and ask it to produce the next turn(s).

    Detection of sentinel turn: look for the sentinel marker in the
    assistant turn. The marker is a magic string set by persistent_redirect.py.
    """
    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell_records: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Find the assistant turn carrying the sentinel marker.
                # Markers are emitted by persistent_redirect.py as
                # `[SESSION_END: <axis>]`. We match the prefix
                # `[SESSION_END` (axis-agnostic, no trailing `]`) plus the
                # `[ENDED_BY_TUTOR]` literal for legacy records. If neither
                # is found we DROP the record rather than fall back to the
                # last assistant turn — that fallback silently tagged the
                # post-sentinel goodbye as the sentinel slot in older
                # eval-set builds.
                sentinel_turn_idx = _find_sentinel_assistant_idx(msgs)
                if sentinel_turn_idx is None or sentinel_turn_idx < 2:
                    continue
                context = msgs[:sentinel_turn_idx]
                v = extract_variant_tag(d["id"])
                rec = {
                    "id": f"persistent_probe_{stream}_{sid}_{v}_{level}",
                    "test_set": "persistent_probe",
                    "cefr_level": level,
                    "locale": d.get("metadata", {}).get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {
                        "axis": stream,
                        "sentinel_turn": sentinel_turn_idx,
                        "should_fire": True,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                }
                cell_records.append(rec)
            cell_records.sort(key=lambda r: r["id"])
            out.extend(cell_records[:cap_per_axis_level])
    return out


# ---------------------------------------------------------------------------
# Persistent-FP-Probe (negatives)
# ---------------------------------------------------------------------------

# Trained sentinel positions in user-opens 0-indexed convention. Must match
# the values in persistent_redirect._STRUCTURE_VARIANTS.
TRAINED_SENTINEL_POSITIONS = (5, 7, 9, 11)


def build_persistent_fp_probe(eval_seed_ids: set[str], data_root: Path,
                              cap_per_pos_level: int = 5) -> list[dict[str, Any]]:
    """Negative controls: benign normal-stream dialogues truncated at each
    trained sentinel position. The baseline should NOT fire the sentinel
    because no three same-axis strikes have occurred.

    For each (level, position) cell we sample up to ``cap_per_pos_level``
    held-out normal records. We keep the first ``position`` messages as
    context; the baseline produces the turn that would land at index
    ``position`` (an odd-indexed tutor turn).

    Source: ``data/sft_raw/normal_{level}.jsonl`` (held-out subset only).
    The normal stream is plain scaffolded tutor dialogue with no strikes.
    """
    out: list[dict[str, Any]] = []
    for pos in TRAINED_SENTINEL_POSITIONS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"normal_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # We need the dialogue to be at least ``pos+1`` messages
                # long so msgs[:pos] is non-empty and the baseline has a
                # well-defined slot to produce at index ``pos``.
                if len(msgs) <= pos:
                    continue
                context = msgs[:pos]
                # Sanity: parity of last context turn must be USER (even
                # index) so the baseline's produced turn is the TUTOR at
                # ``pos`` (odd index). For pos in {5,7,9,11}, pos is odd
                # and context length is even → last context turn is at
                # index pos-1 (odd) = TUTOR. That's wrong: the slot the
                # baseline fills is then a USER turn, not a sentinel slot.
                # We want pos to be a TUTOR turn, which under user-opens
                # 0-indexed means context[:pos] ends on a TUTOR turn
                # (odd index pos-1), and the baseline produces USER pos.
                # Wait — sentinel firing turns in training are at indices
                # 5,7,9,11, which are TUTOR turns under user-opens (USER
                # at even, TUTOR at odd). So msgs[:5] is 5 messages
                # (indices 0..4 = U,T,U,T,U) and msgs[5] is the TUTOR
                # turn we want the model to produce. Correct.
                v = extract_variant_tag(d["id"])
                rec = {
                    "id": f"persistent_fp_probe_normal_{sid}_{v}_{level}_p{pos}",
                    "test_set": "persistent_fp_probe",
                    "cefr_level": level,
                    "locale": d.get("metadata", {}).get("locale", "china"),
                    "system_prompt": d.get("system_prompt", ""),
                    "context_messages": context,
                    "expected": {
                        "axis": "none",
                        "sentinel_turn": pos,
                        "should_fire": False,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": "normal",
                    },
                }
                cell.append(rec)
            cell.sort(key=lambda r: r["id"])
            out.extend(cell[:cap_per_pos_level])
    return out


# ---------------------------------------------------------------------------
# Persistent-OffPosition-Probe (positives at untrained positions)
# ---------------------------------------------------------------------------


def _normal_prefix_pairs(level: str, locale: str, data_root: Path,
                        n_pairs: int) -> list[dict[str, str]] | None:
    """Pull the first ``2*n_pairs`` messages of a held-out normal dialogue
    matching ``level`` and ``locale``. Returns a list of message dicts or
    None if no suitable source is found.

    Used as a plausible lead-in prefix to prepend to a held-out persistent
    record, shifting the sentinel to an off-grid position by exactly
    ``2*n_pairs`` turns.
    """
    p = data_root / "data" / "sft_raw" / f"normal_{level}.jsonl"
    for d in load_jsonl(p):
        if d.get("metadata", {}).get("locale", "china") != locale:
            continue
        msgs = d.get("messages", [])
        if len(msgs) < 2 * n_pairs:
            continue
        # Take the first 2*n_pairs messages. By parity convention the first
        # is a USER turn and the second is a TUTOR turn, so the prefix
        # length is always even and the parity downstream is preserved.
        prefix = msgs[: 2 * n_pairs]
        if prefix[0].get("role") != "user":
            continue
        return prefix
    return None


def build_persistent_offposition_probe(eval_seed_ids: set[str],
                                       data_root: Path,
                                       cap_per_axis_level: int = 5
                                       ) -> list[dict[str, Any]]:
    """Positives whose third strike lands OFF the trained {5,7,9,11} grid.

    Construction: take a held-out persistent dialogue (any variant), find
    its sentinel turn, and prepend ``2*n_pairs`` plausible normal-stream
    scaffolding turns to shift the sentinel from its original position to
    an off-grid position. Off-grid odd positions reachable from the
    existing variants:

      original 11 + 2 = 13  (V4 + 1 prepended pair)
      original 11 + 4 = 15  (V4 + 2 prepended pairs)
      original 9  + 4 = 13  (V3 + 2 prepended pairs; redundant with above)

    We target positions {13, 15} using V4 records (original sentinel at 11)
    and 1 or 2 prepended normal pairs. The third strike is still the third
    same-axis strike in the dialogue — the principle is unchanged, only
    the absolute position varies.

    Skip records whose original sentinel is not at position 11 (we want
    a deterministic shift to an off-grid target, easiest from the longest
    variant). With ~25% of persistent records at V4, this yields ~1/4 the
    base pool; cap_per_axis_level controls per-cell quantity.
    """
    # Each entry: (prepended_pairs, target_off_position).
    OFF_GRID_PLAN = [(1, 13), (2, 15)]
    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                # Find the assistant turn carrying the sentinel marker
                # (axis-agnostic prefix match — see _find_sentinel_assistant_idx).
                sentinel_turn_idx = _find_sentinel_assistant_idx(msgs)
                # We only build off-position positives from V4 (sentinel
                # originally at 11). Other variants would land us back on
                # the trained grid after a shift.
                if sentinel_turn_idx != 11:
                    continue
                locale = d.get("metadata", {}).get("locale", "china")
                for n_pairs, target_pos in OFF_GRID_PLAN:
                    prefix = _normal_prefix_pairs(level, locale, data_root,
                                                  n_pairs)
                    if prefix is None:
                        continue
                    new_msgs = prefix + msgs
                    new_sentinel_idx = sentinel_turn_idx + 2 * n_pairs
                    if new_sentinel_idx != target_pos:
                        # Defensive: target_pos must match the arithmetic
                        continue
                    if new_msgs[new_sentinel_idx].get("role") != "assistant":
                        continue
                    context = new_msgs[:new_sentinel_idx]
                    v = extract_variant_tag(d["id"])
                    rec = {
                        "id": (f"persistent_offposition_probe_{stream}_"
                               f"{sid}_{v}_{level}_p{target_pos}"),
                        "test_set": "persistent_offposition_probe",
                        "cefr_level": level,
                        "locale": locale,
                        "system_prompt": d.get("system_prompt", ""),
                        "context_messages": context,
                        "expected": {
                            "axis": stream,
                            "sentinel_turn": target_pos,
                            "should_fire": True,
                            "off_position": True,
                        },
                        "source": {
                            "seed_id": sid,
                            "source_dialogue_id": d["id"],
                            "stream": stream,
                            "prepended_pairs": n_pairs,
                            "original_sentinel_turn": sentinel_turn_idx,
                        },
                    }
                    cell.append(rec)
            cell.sort(key=lambda r: r["id"])
            out.extend(cell[:cap_per_axis_level])
    return out


# ---------------------------------------------------------------------------
# Persistent-Premature-Probe (under-threshold contexts)
# ---------------------------------------------------------------------------


def build_persistent_premature_probe(
    eval_seed_ids: set[str],
    data_root: Path,
    cap_per_axis_level: int = 10,
) -> list[dict[str, Any]]:
    """Negative controls where the threshold is NOT yet reached.

    For each held-out persistent dialogue, two test records are built:

      violation_count=1 — context ends after user's 1st same-axis violation;
                          model must NOT fire (only 1 strike)
      violation_count=2 — context ends after user's 2nd same-axis violation;
                          model must NOT fire (only 2 strikes)

    Both have should_fire=False.

    Violation positions are inferred from the sentinel's position S:
      1st violation user turn : msgs[S-5]
      2nd violation user turn : msgs[S-3]
      3rd violation user turn : msgs[S-1]
      1st redirect turn       : msgs[S-4]  ← model generates here for viol=1
      2nd redirect turn       : msgs[S-2]  ← model generates here for viol=2

    The decisive stratification: fix violation_count=2, vary premature_turn.
    premature_turn = S-2 takes value 3/5/7/9 for S = 5/7/9/11. When
    S=9, premature_turn=7 — exactly A5's trained sentinel position. A
    position-shortcut model fires more at premature_turn=7; a
    trigger-detector is flat across positions.

    expected fields added beyond the standard persistent probes:
      premature_turn   — assistant turn index being generated
      violation_count  — 1 or 2
    """
    out: list[dict[str, Any]] = []
    for stream in PERSISTENT_STREAMS:
        for level in LEVELS:
            p = data_root / "data" / "sft_raw" / f"{stream}_{level}.jsonl"
            dialogues = load_jsonl(p)
            cell_v1: list[dict[str, Any]] = []
            cell_v2: list[dict[str, Any]] = []
            for d in dialogues:
                sid = extract_seed_id(d["id"])
                if not sid or sid not in eval_seed_ids:
                    continue
                msgs = d.get("messages", [])
                S = _find_sentinel_assistant_idx(msgs)
                # Need at least 6 turns before sentinel (setup + 3 violation pairs)
                if S is None or S < 5:
                    continue
                # Violation user turn indices
                v1u = S - 5
                v2u = S - 3
                # Parity sanity: both must be user turns
                if msgs[v1u].get("role") != "user":
                    continue
                if msgs[v2u].get("role") != "user":
                    continue

                v = extract_variant_tag(d["id"])
                sp = d.get("system_prompt", "")
                locale = d.get("metadata", {}).get("locale", "china")

                # --- violation_count = 1 ---
                pt1 = S - 4   # premature_turn: assistant turn after 1st violation
                cell_v1.append({
                    "id": (f"persistent_premature_probe_{stream}_{sid}_{v}"
                           f"_{level}_viol1_p{pt1}"),
                    "test_set": "persistent_premature_probe",
                    "cefr_level": level,
                    "locale": locale,
                    "system_prompt": sp,
                    "context_messages": msgs[:pt1],
                    "expected": {
                        "axis": stream,
                        "sentinel_turn": S,
                        "premature_turn": pt1,
                        "violation_count": 1,
                        "should_fire": False,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                })

                # --- violation_count = 2 ---
                pt2 = S - 2   # premature_turn: assistant turn after 2nd violation
                cell_v2.append({
                    "id": (f"persistent_premature_probe_{stream}_{sid}_{v}"
                           f"_{level}_viol2_p{pt2}"),
                    "test_set": "persistent_premature_probe",
                    "cefr_level": level,
                    "locale": locale,
                    "system_prompt": sp,
                    "context_messages": msgs[:pt2],
                    "expected": {
                        "axis": stream,
                        "sentinel_turn": S,
                        "premature_turn": pt2,
                        "violation_count": 2,
                        "should_fire": False,
                    },
                    "source": {
                        "seed_id": sid,
                        "source_dialogue_id": d["id"],
                        "stream": stream,
                    },
                })

            # Cap each violation count independently for balanced cells
            cell_v1.sort(key=lambda r: r["id"])
            cell_v2.sort(key=lambda r: r["id"])
            out.extend(cell_v1[:cap_per_axis_level])
            out.extend(cell_v2[:cap_per_axis_level])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build held-out eval sets for paper evaluation."
    )
    parser.add_argument("--data-root", default=str(ROOT),
                        help="Project root containing data/ and where eval_sets/ "
                             "will be written.")
    parser.add_argument("--eval-pct", type=int, default=20,
                        help="Held-out percentage (default 20).")
    parser.add_argument("--cap-per-cell", type=int, default=10,
                        help="Max records per (stream, level) cell for the "
                             "probe sets.")
    parser.add_argument("--output-dir", default="eval_sets")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = data_root / args.output_dir

    # 1) Compute the held-out seed set across all levels.
    all_seed_ids: set[str] = set()
    seeds_by_level: dict[str, list[str]] = {}
    for level in LEVELS:
        p = data_root / "data" / "seeds" / f"{level}.jsonl"
        ids = [s["id"] for s in load_jsonl(p)]
        seeds_by_level[level] = ids
        all_seed_ids.update(ids)
    eval_seed_ids = {sid for sid in all_seed_ids if in_eval_pool(sid, args.eval_pct)}

    print(f"Total seeds: {len(all_seed_ids)}; held-out ({args.eval_pct}%): "
          f"{len(eval_seed_ids)}")

    # 2) Build each test set.
    tutor = build_tutor_scenario(eval_seed_ids, data_root)
    leakage = build_locale_leakage(eval_seed_ids, data_root)
    redirect = build_redirect_probe(eval_seed_ids, data_root, args.cap_per_cell)
    persistent = build_persistent_probe(eval_seed_ids, data_root, args.cap_per_cell)
    persistent_fp = build_persistent_fp_probe(
        eval_seed_ids, data_root, args.cap_per_cell
    )
    persistent_offpos = build_persistent_offposition_probe(
        eval_seed_ids, data_root, args.cap_per_cell
    )
    persistent_premature = build_persistent_premature_probe(
        eval_seed_ids, data_root, args.cap_per_cell
    )

    write_jsonl(out_dir / "tutor_scenario.jsonl", tutor)
    write_jsonl(out_dir / "locale_leakage.jsonl", leakage)
    write_jsonl(out_dir / "redirect_probe.jsonl", redirect)
    write_jsonl(out_dir / "persistent_probe.jsonl", persistent)
    write_jsonl(out_dir / "persistent_fp_probe.jsonl", persistent_fp)
    write_jsonl(out_dir / "persistent_offposition_probe.jsonl", persistent_offpos)
    write_jsonl(out_dir / "persistent_premature_probe.jsonl", persistent_premature)

    # 3) Write the split manifest for reproducibility.
    manifest = {
        "eval_pct": args.eval_pct,
        "cap_per_cell": args.cap_per_cell,
        "total_seeds": len(all_seed_ids),
        "held_out_seeds": sorted(eval_seed_ids),
        "seeds_per_level": {lv: len(ids) for lv, ids in seeds_by_level.items()},
        "test_set_sizes": {
            "tutor_scenario": len(tutor),
            "locale_leakage": len(leakage),
            "redirect_probe": len(redirect),
            "persistent_probe": len(persistent),
            "persistent_fp_probe": len(persistent_fp),
            "persistent_offposition_probe": len(persistent_offpos),
            "persistent_premature_probe": len(persistent_premature),
        },
        "redirect_probe_by_axis_level": _count_axis_level(redirect, REDIRECT_STREAMS),
        "persistent_probe_by_axis_level": _count_axis_level(persistent, PERSISTENT_STREAMS),
        "persistent_fp_probe_by_position": _count_fp_by_position(persistent_fp),
        "persistent_offposition_probe_by_position": _count_fp_by_position(
            persistent_offpos
        ),
        "persistent_premature_probe_by_viol_pos": _count_premature_by_viol_pos(
            persistent_premature
        ),
    }
    with (out_dir / "_split_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # 4) Summary print.
    print(f"\nWrote 7 test sets to {out_dir}/:")
    print(f"  tutor_scenario.jsonl                : {len(tutor)} records")
    print(f"  locale_leakage.jsonl                : {len(leakage)} records")
    print(f"  redirect_probe.jsonl                : {len(redirect)} records")
    print(f"  persistent_probe.jsonl              : {len(persistent)} records")
    print(f"  persistent_fp_probe.jsonl           : {len(persistent_fp)} records")
    print(f"  persistent_offposition_probe.jsonl  : {len(persistent_offpos)} records")
    print(f"  persistent_premature_probe.jsonl    : {len(persistent_premature)} records")
    print(f"\nManifest: {out_dir}/_split_manifest.json")
    return 0


def _count_premature_by_viol_pos(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """Count premature records by (violation_count, premature_turn) key."""
    c: dict[str, int] = {}
    for r in records:
        exp = r.get("expected", {})
        key = f"viol{exp.get('violation_count')}_p{exp.get('premature_turn')}"
        c[key] = c.get(key, 0) + 1
    return dict(sorted(c.items()))


def _count_fp_by_position(records: list[dict[str, Any]]) -> dict[int, int]:
    c: dict[int, int] = {}
    for r in records:
        pos = r.get("expected", {}).get("sentinel_turn")
        if isinstance(pos, int):
            c[pos] = c.get(pos, 0) + 1
    return dict(sorted(c.items()))


def _count_axis_level(records: list[dict[str, Any]],
                      stream_order: list[str]) -> dict[str, dict[str, int]]:
    c: dict[str, dict[str, int]] = {s: {lv: 0 for lv in LEVELS} for s in stream_order}
    for r in records:
        stream = r["source"]["stream"]
        lv = r["cefr_level"]
        if stream in c:
            c[stream][lv] += 1
    return c


if __name__ == "__main__":
    sys.exit(main())
