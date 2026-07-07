"""trim_redirect_to_700.py — Truncate redirect records in sft_raw and
sft_filtered to 700 total (proportional first-N from each level file).

No randomness: keeps the first round(n_i / total * 700) lines from each file.
Also trims sft_filtered/redirect_*_passed.jsonl proportionally so filter_sft
does not need to re-run.

Run once. Idempotent if already <= 700.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = 700


def _trim_dir(glob_pattern: str, data_dir: Path, target: int) -> None:
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        print(f"  no files matched {data_dir / glob_pattern}")
        return

    counts: list[tuple[Path, list[str]]] = []
    total = 0
    for f in files:
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        counts.append((f, lines))
        total += len(lines)

    print(f"  {data_dir.name}: {total} records across {len(files)} files", end="")

    if total <= target:
        print(f" — already <= {target}, nothing to do.")
        return

    # Proportional first-N truncation (no randomness)
    allocs = [round(len(lines) / total * target) for _, lines in counts]
    # Fix rounding so sum == target
    diff = target - sum(allocs)
    if diff != 0:
        # Add/remove from the files with the most records
        order = sorted(range(len(counts)), key=lambda i: -len(counts[i][1]))
        step = 1 if diff > 0 else -1
        for i in order[: abs(diff)]:
            allocs[i] += step

    written = 0
    for (f, lines), n in zip(counts, allocs):
        n = max(0, min(n, len(lines)))
        f.write_text("\n".join(lines[:n]) + "\n", encoding="utf-8")
        written += n

    print(f" → trimmed to {written}.")


def main() -> None:
    sft_raw = ROOT / "data" / "sft_raw"
    sft_filtered = ROOT / "data" / "sft_filtered"

    print(f"Trimming redirect to {TARGET} total records...")
    print(f"\nsft_raw:")
    _trim_dir("redirect_*.jsonl", sft_raw, TARGET)

    print(f"\nsft_filtered:")
    _trim_dir("redirect_*_passed.jsonl", sft_filtered, TARGET)

    print("\nDone. Re-run setup_paper_ablation_data.py to refresh condition dirs.")


if __name__ == "__main__":
    main()
