"""setup_paper_ablation_data.py — Materialize per-condition SFT-filtered dirs.

For paper ablations, each training condition needs a different subset of
``data/sft_filtered/*_passed.jsonl``. Rather than modify the loader, this
script creates hardlinks under per-condition directories so the existing
``_iter_filtered_records`` glob just works.

Conditions:
  A1  full system                 — uses data/sft_filtered (no setup needed)
  A2  SFT only                    — uses data/sft_filtered (no setup needed)
  A3  no specialized redirects    — data/sft_filtered_a3/  (normal + generic
                                   redirect + persistent only; drops the 6
                                   specialized redirect streams)
  A4  no persistent streams       — data/sft_filtered_a4/  (normal + 7
                                   single-shot redirects; drops the 4
                                   persistent streams)

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
    "a3": NORMAL + GENERIC_REDIRECT + PERSISTENT,  # no specialized redirects
    "a4": NORMAL + GENERIC_REDIRECT + SPECIALIZED_REDIRECTS,  # no persistent
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
    listed streams. Returns a summary dict."""
    dst_dir = ROOT / "data" / f"sft_filtered_{condition}"
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing files for idempotence
        for p in dst_dir.glob("*.jsonl"):
            p.unlink()

    summary = {"condition": condition, "dst": str(dst_dir),
               "streams": streams, "files_linked": 0, "missing": []}
    for stream in streams:
        for src in SRC_DIR.glob(f"{stream}_*_passed.jsonl"):
            dst = dst_dir / src.name
            if dry_run:
                print(f"  would link: {src.name}")
            else:
                method = link_or_copy(src, dst)
                summary["files_linked"] += 1
        # Sanity: warn if no files matched for this stream
        if not list(SRC_DIR.glob(f"{stream}_*_passed.jsonl")):
            summary["missing"].append(stream)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="a3,a4",
                        help="Comma-separated list of conditions to set up.")
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
