"""assemble_a8_count.py — assemble the full A8 (count-marker, Design B) SFT dirs.

Mirrors build_untrim_a5a6a7.py::assemble: copy a base sft_filtered dir (which
carries the shared non-persistent streams) and OVERWRITE the four
persistent_*_passed.jsonl files with the count-annotated versions.

Produces:
  data/sft_filtered_a8_count/       (untrimmed count persistent)  -> A8
  data/sft_filtered_a8_count_trim/  (trimmed   count persistent)  -> A8-trim

Pre-req:
  data/sft_filtered_a8_count_persistent/       from convert_typed_to_count.py
  data/sft_filtered_a8_count_trim_persistent/  (optional) the trimmed count
      persistent; if absent, --trim is skipped with a note.

CPU only. No training. Idempotent (rewrites destination dirs).

Usage:
  python scripts/convert_typed_to_count.py          # step 1 (untrimmed count)
  python scripts/assemble_a8_count.py               # step 2 (assemble)
  # for the trim cell, first produce the trimmed count persistent, then:
  python scripts/assemble_a8_count.py --trim
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def assemble(base: Path, persist: Path, dst: Path) -> None:
    if not base.exists():
        raise SystemExit(f"base dir missing: {base}")
    if not persist.exists():
        raise SystemExit(f"count persistent dir missing: {persist}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    n = 0
    for f in sorted(persist.glob("*_passed.jsonl")):
        # overwrite the matching persistent file in the copied base
        target_name = f.name.replace("a8_", "")  # ids prefixed but filenames are not
        (dst / f.name).unlink(missing_ok=True)
        shutil.copy(f, dst / target_name)
        n += 1
    print(f"[assemble] {base.name} + {persist.name} -> {dst.name} "
          f"(overwrote {n} persistent files)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_filtered",
                    help="Base dir carrying the shared non-persistent streams.")
    ap.add_argument("--trim", action="store_true",
                    help="Assemble the TRIMMED count cell instead of untrimmed.")
    args = ap.parse_args()

    base = ROOT / args.base
    if args.trim:
        persist = ROOT / "data/sft_filtered_a8_count_trim_persistent"
        dst = ROOT / "data/sft_filtered_a8_count_trim"
        if not persist.exists():
            print(f"NOTE: {persist} not found. Produce the trimmed count "
                  f"persistent first (run trim_persistent_post_sentinel over the "
                  f"count persistent), then re-run with --trim.")
            return 1
    else:
        persist = ROOT / "data/sft_filtered_a8_count_persistent"
        dst = ROOT / "data/sft_filtered_a8_count"

    assemble(base, persist, dst)
    print("DONE. NO configs written, NO training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
