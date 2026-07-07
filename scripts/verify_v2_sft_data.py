"""verify_v2_sft_data.py — sanity-check the v2 SFT datasets BEFORE training.

Verifies that each training condition's persistent records carry the
expected sentinel format AND that the embedded system_prompt's
[persistence] block matches.

Conditions checked:
  A1/A2 main      data/sft_filtered/                  expect: axis-specific
  A3              data/sft_filtered_a3/               expect: NO persistent files
  A4              data/sft_filtered_a4/               expect: NO persistent files
  A5              data/sft_filtered_a5/               expect: axis-specific
  A6              data/sft_filtered_a6/               expect: generic
  A7              data/sft_filtered_a7/               expect: generic
  A5 source       data/sft_filtered_a5_persistent/    expect: axis-specific
  A6 source       data/sft_filtered_a6_persistent/    expect: generic
  A7 source       data/sft_filtered_a7_persistent/    expect: generic

For each dir that has persistent files, prints:
  - record count per stream
  - sentinel format(s) found in messages
  - [persistence] block style found in embedded system_prompts
  - red flag if anything is inconsistent

Run with:
  python scripts/verify_v2_sft_data.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PERSISTENT_AXES = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)

# Patterns we care about.
_AXIS_SENTINEL_RE = re.compile(r"\[SESSION_END:\s*persistent_(?:off_topic|language_violation|persona_break|role_swap)\s*\]")
_GENERIC_SENTINEL_RE = re.compile(r"\[SESSION_END\](?!:)")  # bracket, not followed by colon
# [persistence] block tells (cheap heuristics, not full parse):
_PERSISTENCE_AXIS_TELL = "[SESSION_END: persistent_off_topic]"
_PERSISTENCE_GENERIC_TELL = "Emit this sentinel for ALL four persistent axes"


def _classify_system_prompt(sp: str) -> str:
    if not sp:
        return "missing"
    has_axis = _PERSISTENCE_AXIS_TELL in sp
    has_generic = _PERSISTENCE_GENERIC_TELL in sp
    if has_axis and not has_generic:
        return "axis_specific"
    if has_generic and not has_axis:
        return "generic"
    if has_axis and has_generic:
        return "MIXED (BUG?)"
    return "neither (BUG?)"


def _count_sentinels_in_record(rec: dict) -> tuple[int, int]:
    """Return (axis_specific_count, generic_count) across all assistant turns."""
    ex = rec.get("example") or rec
    msgs = ex.get("messages") or []
    text = "\n".join((m.get("content") or "") for m in msgs if m.get("role") == "assistant")
    n_axis = len(_AXIS_SENTINEL_RE.findall(text))
    n_generic = len(_GENERIC_SENTINEL_RE.findall(text))
    return n_axis, n_generic


def inspect_dir(label: str, dir_path: Path, expected: str) -> None:
    print(f"=== {label}  ({dir_path.relative_to(ROOT)}) ===")
    if not dir_path.exists():
        print(f"  (dir does not exist)")
        print()
        return
    if expected.startswith("none"):
        # Should have no persistent files
        leaks = []
        for axis in PERSISTENT_AXES:
            matches = list(dir_path.glob(f"{axis}_*_passed.jsonl"))
            if matches:
                leaks.append((axis, len(matches)))
        if leaks:
            print(f"  RED FLAG: expected NO persistent files but found:")
            for axis, n in leaks:
                print(f"    {axis}: {n} files")
        else:
            print(f"  OK: no persistent files (as expected)")
        print()
        return

    total = 0
    prompt_classes: Counter = Counter()
    axis_sentinels = 0
    generic_sentinels = 0
    for axis in PERSISTENT_AXES:
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            f = dir_path / f"{axis}_{level}_passed.jsonl"
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ex = rec.get("example") or rec
                sp = ex.get("system_prompt") or ""
                prompt_classes[_classify_system_prompt(sp)] += 1
                a, g = _count_sentinels_in_record(rec)
                axis_sentinels += a
                generic_sentinels += g
                total += 1
    print(f"  records:                {total}")
    print(f"  system_prompt classes:  {dict(prompt_classes)}")
    print(f"  axis-specific sentinels in messages: {axis_sentinels}")
    print(f"  generic [SESSION_END]   in messages: {generic_sentinels}")

    # Verdict
    if expected == "axis_specific":
        ok = (prompt_classes.get("axis_specific", 0) == total
              and generic_sentinels == 0
              and axis_sentinels >= total)  # >= because some recs may have 2
        print(f"  expected: axis-specific. verdict: {'OK' if ok else 'RED FLAG'}")
    elif expected == "generic":
        ok = (prompt_classes.get("generic", 0) == total
              and axis_sentinels == 0
              and generic_sentinels >= total)
        print(f"  expected: generic. verdict: {'OK' if ok else 'RED FLAG'}")
    print()


def main() -> int:
    print()
    print("=" * 78)
    print("v2 SFT data verification")
    print("=" * 78)
    print()
    inspect_dir("A1/A2 main",       DATA / "sft_filtered",                  "axis_specific")
    inspect_dir("A3",               DATA / "sft_filtered_a3",               "none")
    inspect_dir("A4 (deprecated)",  DATA / "sft_filtered_a4",               "none")
    inspect_dir("A5",               DATA / "sft_filtered_a5",               "axis_specific")
    inspect_dir("A6",               DATA / "sft_filtered_a6",               "generic")
    inspect_dir("A7",               DATA / "sft_filtered_a7",               "generic")
    inspect_dir("A5 source",        DATA / "sft_filtered_a5_persistent",    "axis_specific")
    inspect_dir("A6 source",        DATA / "sft_filtered_a6_persistent",    "generic")
    inspect_dir("A7 source",        DATA / "sft_filtered_a7_persistent",    "generic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
