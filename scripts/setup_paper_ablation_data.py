"""setup_paper_ablation_data.py — Materialize per-condition SFT-filtered dirs.

For paper ablations, each training condition needs a different subset of
``data/sft_filtered/*_passed.jsonl``. Rather than modify the loader, this
script creates hardlinks under per-condition directories so the existing
``_iter_filtered_records`` glob just works.

Conditions:
  A1  full system                 — uses data/sft_filtered (no setup needed)
  A2  SFT only                    — uses data/sft_filtered (no setup needed)
  A3  no redirects (specialized   — data/sft_filtered_a3/  (normal + generic
      OR persistent)                redirect only; drops ALL 6 specialized
                                   redirect streams AND all 4 persistent
                                   redirect streams). Clean ablation baseline:
                                   A3→A4 tests "do specialized single-turn
                                   redirects help at all?".
  A4  no persistent streams       — data/sft_filtered_a4/  (normal + generic
                                   redirect + 6 specialized; drops the 4
                                   persistent streams). Tests "does persistence
                                   training add beyond single-turn redirects?"
                                   (A4→A1 comparison).
  A5  fixed-turn-7 persistent     — data/sft_filtered_a5/  (full 12-stream
                                   mix, but the 4 persistent streams come
                                   from data/sft_filtered_a5_persistent/
                                   regenerated with the V2-only override
                                   QWEN_TUTOR_PERSISTENT_FORCED_VARIANT=1).
                                   Used to isolate the decorrelation
                                   contribution: A1-vs-A5 is the direct
                                   test of §3.4. Pre-req: regen persistent
                                   first (see training_a5_fixed_turn_7.yaml
                                   header).
  A6  generic sentinel ablation   — data/sft_filtered_a6/  (full 12-stream
                                   mix like A1, but the 4 persistent streams
                                   come from data/sft_filtered_a6_persistent/,
                                   regenerated with
                                   QWEN_TUTOR_SENTINEL_FORMAT=generic so the
                                   sentinel is ``[SESSION_END]`` with no axis
                                   label). A1-vs-A6 tests whether the
                                   axis-specific sentinel label is
                                   load-bearing for the 0% FP rate, or
                                   whether position-based learning alone is
                                   sufficient.

Run after any change to data/sft_filtered/ — the script clears the per-
condition dir and re-links from scratch, so it's idempotent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source layout: data/sft_filtered/<stream>_<level>_{passed|failed}.jsonl
SRC_DIR = ROOT / "data" / "sft_filtered"

# Stream groups
NORMAL = ["normal"]
GENERIC_REDIRECT = ["redirect"]
SPECIALIZED_REDIRECTS = [
    "locale_redirect",
    "pedagogy_redirect",
    "language_redirect",
    "persona_redirect",
    "topic_redirect",
    "role_swap_redirect",
]
PERSISTENT = [
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
]

# Per-condition stream membership
CONDITIONS: dict[str, list[str]] = {
    # A3: clean "no redirect training" baseline — drops ALL 6 specialized AND
    # all 4 persistent redirect streams. Only normal + generic redirect remain.
    # Rationale: persistent_language_violation / persona_break / role_swap are
    # multi-turn encodings of the same redirect behaviors as the specialized
    # streams. Keeping them in A3 contaminates the A1-vs-A3 comparison with
    # partial redirect signal. Drop both groups for a clean ladder:
    #   A3 (none) → A4 (specialized only) → A1 (specialized + persistent).
    "a3": NORMAL + GENERIC_REDIRECT,
    "a4": NORMAL + GENERIC_REDIRECT + SPECIALIZED_REDIRECTS,  # no persistent
    # A5: same streams as A1, but persistent streams come from a separate
    # fixed-turn-7 regen dir (handled below in setup_condition).
    "a5": NORMAL + GENERIC_REDIRECT + SPECIALIZED_REDIRECTS + PERSISTENT,
    # A6: same streams as A1, but persistent streams come from the
    # generic-sentinel regen dir (sft_filtered_a6_persistent/).
    "a6": NORMAL + GENERIC_REDIRECT + SPECIALIZED_REDIRECTS + PERSISTENT,
    # A7: same streams as A1 and same 4-variant persistent volume as A2,
    # but persistent streams use generic [SESSION_END] (sourced from
    # sft_filtered_a7_persistent/, converted from A1's persistent data
    # by scripts/convert_a1_to_a7.py).
    # Together with A6 this closes the 2x2 design:
    #   A2: 4-variant + axis-specific   A5: fixed-7 + axis-specific
    #   A7: 4-variant + generic         A6: fixed-7 + generic
    "a7": NORMAL + GENERIC_REDIRECT + SPECIALIZED_REDIRECTS + PERSISTENT,
}

# Per-condition stream-specific source overrides. Streams listed here pull
# their _passed.jsonl from the alternate dir instead of data/sft_filtered/.
# Used for A5, where the persistent streams need to come from the V2-only
# regen rather than the standard 4-variant data.
CONDITION_STREAM_SRC: dict[str, dict[str, Path]] = {
    "a5": {
        stream: ROOT / "data" / "sft_filtered_a5_persistent"
        for stream in PERSISTENT
    },
    "a6": {
        stream: ROOT / "data" / "sft_filtered_a6_persistent"
        for stream in PERSISTENT
    },
    "a7": {
        stream: ROOT / "data" / "sft_filtered_a7_persistent"
        for stream in PERSISTENT
    },
}


def link_or_copy(src: Path, dst: Path) -> str:
    """Try hardlink first (Windows + POSIX). Fall back to copy if hardlink fails."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except (OSError, NotImplementedError):
        import shutil
        shutil.copyfile(src, dst)
        return "copy"


def setup_condition(condition: str, streams: list[str], dry_run: bool = False) -> dict:
    """Materialize data/sft_filtered_<condition>/ with only passed files of the
    listed streams. Returns a summary dict.

    Source dir is data/sft_filtered/ by default, but
    CONDITION_STREAM_SRC[condition][stream] overrides it per-stream (used
    by A5 to source persistent streams from the V2-only regen).
    """
    dst_dir = ROOT / "data" / f"sft_filtered_{condition}"
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing files for idempotence
        for p in dst_dir.glob("*.jsonl"):
            p.unlink()

    per_stream_src = CONDITION_STREAM_SRC.get(condition, {})
    summary = {"condition": condition, "dst": str(dst_dir),
               "streams": streams, "files_linked": 0, "missing": []}
    for stream in streams:
        src_dir = per_stream_src.get(stream, SRC_DIR)
        matches = sorted(src_dir.glob(f"{stream}_*_passed.jsonl"))
        if not matches:
            summary["missing"].append(f"{stream} (looked in {src_dir})")
            continue
        for src in matches:
            dst = dst_dir / src.name
            if dry_run:
                print(f"  would link: {src} -> {dst}")
            else:
                link_or_copy(src, dst)
                summary["files_linked"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="a3,a4",
                        help="Comma-separated list of conditions to set up "
                             "(choices: a3, a4, a5, a6, a7).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print(f"ERROR: source dir {SRC_DIR} does not exist; "
              f"run filter_sft first.", file=sys.stderr)
        return 1

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"Source: {SRC_DIR}")
    print(f"Setting up conditions: {conditions}")
    print()
    for c in conditions:
        if c not in CONDITIONS:
            print(f"  skip unknown condition: {c}")
            continue
        summary = setup_condition(c, CONDITIONS[c], dry_run=args.dry_run)
        print(f"[{c}] dst={summary['dst']}")
        print(f"     streams={summary['streams']}")
        print(f"     files linked: {summary['files_linked']}")
        if summary["missing"]:
            print(f"     WARNING: no files for streams: {summary['missing']}")
    print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
