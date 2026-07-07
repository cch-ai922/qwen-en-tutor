"""Automated quality check for the v2 specialized-stream regen.

Run AFTER `python scripts/run_generation.py` writes to data/sft_raw/ +
data/sft_filtered/. Reads the filtered output for each specialized
stream and verifies:

1. **Filter pass rate per stream >= MIN_PASS_RATE**. Below this, the
   teacher prompt is producing too much rejected content.
2. **Per-axis content-pattern checks** on a sample of tutor responses:
   - language_redirect: tutor responses MUST mention L1/switching
     (ACK behavior) and MUST NOT contain L1 script.
   - pedagogy_redirect: tutor responses MUST contain scaffold patterns
     (question marks, "try", "think") and MUST NOT state rules directly.

Exit code 0 = pass, 1 = fail. Writes a JSON diagnostic at
``logs/v2_smoke_failure.json`` on failure.

Conservative thresholds for the v2 final-train smoke gate.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(r"c:\project\conversationFactory\qwen-en-tutor")
SFT_RAW = ROOT / "data" / "sft_raw"
SFT_FILTERED = ROOT / "data" / "sft_filtered"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
FAILURE_REPORT = LOGS / "v2_smoke_failure.json"

# Conservative thresholds (chosen 2026-06-25 for v2 final train)
MIN_PASS_RATE = 0.60        # filter pass-rate floor per stream
MIN_AXIS_BEHAVIOR_RATE = 0.50  # how often the AXIS-positive pattern fires
MAX_AXIS_NEGATIVE_RATE = 0.10  # how often AXIS-negative (forbidden) pattern fires

SPECIALIZED_STREAMS = (
    "locale_redirect", "language_redirect", "pedagogy_redirect",
    "persona_redirect", "role_swap_redirect", "topic_redirect",
)

# CJK / kana / Hangul / Cyrillic. Any L1 script char in a tutor turn
# fails the language_redirect axis.
_L1_SCRIPT_RE = re.compile(
    r"[一-鿿぀-ゟ゠-ヿ가-힯Ѐ-ӿ]"
)
# Pattern that *should* fire on language ACK tutor turns
_LANG_ACK_RE = re.compile(
    r"\b(switch(?:ed|ing)?|your\s+L1|in\s+(?:chinese|japanese|korean|italian|"
    r"russian|german|spanish|french)|noticed|came out|that's okay|try (?:in|again)|"
    r"english|practice)\b",
    re.IGNORECASE,
)
# Pattern that *should* fire on pedagogy scaffolds (counter-question /
# hint / partial scaffold). Loose: any question mark or scaffold cue.
_PED_SCAFFOLD_RE = re.compile(
    r"\?|\b(try|think|what do you|listen|how about|let'?s see|guess|"
    r"your turn|hint|spot the)\b",
    re.IGNORECASE,
)
# Pattern that *should NOT* fire on pedagogy responses (direct rule).
_PED_DIRECT_ANSWER_RE = re.compile(
    r"\bthe rule (?:for|is)\b|"
    r"\bthe answer is\b|"
    r"\brule\s*\d\b|"
    r"\bfirst[,]?\s+(?:we|you)\s+(?:use|say)\b|"
    r"^\s*[-*•]\s",  # bullet list
    re.IGNORECASE | re.MULTILINE,
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _stream_pass_rate(stream: str) -> tuple[int, int, float]:
    """(raw_count, filtered_count, pass_rate). filtered counted across all
    *_passed.jsonl files for the stream."""
    raw = 0
    for p in SFT_RAW.glob(f"{stream}_*.jsonl"):
        raw += sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
    filtered = 0
    for p in SFT_FILTERED.glob(f"{stream}_*_passed.jsonl"):
        filtered += sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
    if raw == 0:
        return 0, filtered, 0.0
    return raw, filtered, filtered / raw


def _tutor_turns_from_filtered(stream: str) -> list[str]:
    """Pull all tutor (assistant) message contents from the filtered jsonl
    of a given stream. Filter pipeline output wraps records as
    ``{"example": {...messages...}, "pipeline": {...}}`` — read messages
    from rec["example"]["messages"], not from the top level."""
    out: list[str] = []
    for p in SFT_FILTERED.glob(f"{stream}_*_passed.jsonl"):
        for rec in _load_jsonl(p):
            ex = rec.get("example") or rec   # tolerate both shapes
            for m in ex.get("messages", []) or []:
                if m.get("role") == "assistant":
                    c = (m.get("content") or "").strip()
                    if c:
                        out.append(c)
    return out


def _post_l1_tutor_turns(stream: str) -> list[str]:
    """Return only the assistant turns that immediately follow a user
    turn containing L1 script. These are the ACTUAL ACK turns we want to
    check — measuring ACK rate over ALL tutor turns dilutes the signal
    (one ACK per ~7-turn conversation = ~14% even when every conversation
    is perfect).
    """
    out: list[str] = []
    for p in SFT_FILTERED.glob(f"{stream}_*_passed.jsonl"):
        for rec in _load_jsonl(p):
            ex = rec.get("example") or rec
            msgs = ex.get("messages", []) or []
            for i, m in enumerate(msgs):
                if m.get("role") != "user":
                    continue
                if not _L1_SCRIPT_RE.search(m.get("content") or ""):
                    continue
                nxt = msgs[i + 1] if i + 1 < len(msgs) else None
                if nxt and nxt.get("role") == "assistant":
                    c = (nxt.get("content") or "").strip()
                    if c:
                        out.append(c)
    return out


def _frac(matches: int, total: int) -> float:
    return matches / total if total > 0 else 0.0


def main() -> int:
    diagnostics: dict = {"per_stream": {}, "failures": []}
    all_ok = True

    # 1. Per-stream filter pass-rate
    print("=== Stream filter pass rates ===")
    print(f'{"stream":<24} {"raw":>6} {"filt":>6} {"pass_rate":>10}  status')
    for stream in SPECIALIZED_STREAMS:
        raw, filt, rate = _stream_pass_rate(stream)
        ok = rate >= MIN_PASS_RATE
        status = "OK" if ok else "LOW"
        print(f"{stream:<24} {raw:>6} {filt:>6} {rate:>10.3f}  {status}")
        diagnostics["per_stream"].setdefault(stream, {})
        diagnostics["per_stream"][stream].update(
            raw=raw, filtered=filt, pass_rate=rate, pass_rate_ok=ok
        )
        if not ok:
            diagnostics["failures"].append(
                f"{stream}: pass rate {rate:.3f} < {MIN_PASS_RATE}"
            )
            all_ok = False

    # 2. Per-axis content patterns on tutor turns
    print()
    print("=== Per-axis content patterns ===")

    # language_redirect: ACK rate measured on the post-L1 assistant turn ONLY
    # (the one that *should* be the ACK), not all assistant turns. L1-script-
    # in-tutor is still checked across ALL tutor turns (any L1 leak is bad).
    lang_all_turns = _tutor_turns_from_filtered("language_redirect")
    lang_ack_turns = _post_l1_tutor_turns("language_redirect")
    if lang_all_turns and lang_ack_turns:
        ack = sum(1 for t in lang_ack_turns if _LANG_ACK_RE.search(t))
        l1 = sum(1 for t in lang_all_turns if _L1_SCRIPT_RE.search(t))
        ack_rate = _frac(ack, len(lang_ack_turns))
        l1_rate = _frac(l1, len(lang_all_turns))
        ack_ok = ack_rate >= MIN_AXIS_BEHAVIOR_RATE
        l1_ok = l1_rate <= MAX_AXIS_NEGATIVE_RATE
        print(f"language: ACK rate {ack_rate:.3f} on {len(lang_ack_turns)} "
              f"post-L1 turns (>= {MIN_AXIS_BEHAVIOR_RATE}); "
              f"L1-script-in-tutor rate {l1_rate:.3f} on {len(lang_all_turns)} "
              f"total turns (<= {MAX_AXIS_NEGATIVE_RATE})")
        diagnostics["per_stream"]["language_redirect"].update(
            tutor_turns=len(lang_all_turns),
            post_l1_tutor_turns=len(lang_ack_turns),
            ack_rate=ack_rate, ack_ok=ack_ok,
            l1_script_in_tutor_rate=l1_rate, l1_script_ok=l1_ok,
        )
        if not ack_ok:
            diagnostics["failures"].append(
                f"language: ACK rate {ack_rate:.3f} on post-L1 turns < "
                f"{MIN_AXIS_BEHAVIOR_RATE} (tutor fails to acknowledge L1 switch)"
            )
            all_ok = False
        if not l1_ok:
            diagnostics["failures"].append(
                f"language: L1-script-in-tutor rate {l1_rate:.3f} > "
                f"{MAX_AXIS_NEGATIVE_RATE} (tutor is using learner's L1)"
            )
            all_ok = False
    else:
        diagnostics["failures"].append("language_redirect: no filtered output found")
        all_ok = False

    # pedagogy_redirect: scaffold present + direct-rule absent
    ped_turns = _tutor_turns_from_filtered("pedagogy_redirect")
    if ped_turns:
        scaffold = sum(1 for t in ped_turns if _PED_SCAFFOLD_RE.search(t))
        direct = sum(1 for t in ped_turns if _PED_DIRECT_ANSWER_RE.search(t))
        scaffold_rate = _frac(scaffold, len(ped_turns))
        direct_rate = _frac(direct, len(ped_turns))
        sc_ok = scaffold_rate >= MIN_AXIS_BEHAVIOR_RATE
        dir_ok = direct_rate <= MAX_AXIS_NEGATIVE_RATE
        print(f"pedagogy: scaffold rate {scaffold_rate:.3f} (>= {MIN_AXIS_BEHAVIOR_RATE}); "
              f"direct-rule rate {direct_rate:.3f} (<= {MAX_AXIS_NEGATIVE_RATE})")
        diagnostics["per_stream"]["pedagogy_redirect"].update(
            tutor_turns=len(ped_turns),
            scaffold_rate=scaffold_rate, scaffold_ok=sc_ok,
            direct_rule_rate=direct_rate, direct_rule_ok=dir_ok,
        )
        if not sc_ok:
            diagnostics["failures"].append(
                f"pedagogy: scaffold rate {scaffold_rate:.3f} < {MIN_AXIS_BEHAVIOR_RATE} "
                f"(tutor responses don't scaffold)"
            )
            all_ok = False
        if not dir_ok:
            diagnostics["failures"].append(
                f"pedagogy: direct-rule rate {direct_rate:.3f} > "
                f"{MAX_AXIS_NEGATIVE_RATE} (tutor is stating rules directly)"
            )
            all_ok = False
    else:
        diagnostics["failures"].append("pedagogy_redirect: no filtered output found")
        all_ok = False

    print()
    if all_ok:
        print("=== SMOKE QUALITY CHECK PASSED ===")
        # Clean any old failure marker
        if FAILURE_REPORT.exists():
            FAILURE_REPORT.unlink()
        return 0

    # Failure: write diagnostic + return 1
    print("=== SMOKE QUALITY CHECK FAILED ===")
    for fail in diagnostics["failures"]:
        print(f"  - {fail}")
    diagnostics["thresholds"] = {
        "MIN_PASS_RATE": MIN_PASS_RATE,
        "MIN_AXIS_BEHAVIOR_RATE": MIN_AXIS_BEHAVIOR_RATE,
        "MAX_AXIS_NEGATIVE_RATE": MAX_AXIS_NEGATIVE_RATE,
    }
    FAILURE_REPORT.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"\nDiagnostic written to {FAILURE_REPORT.relative_to(ROOT)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
