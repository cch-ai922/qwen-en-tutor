"""gen_pedagogy_eval_extra.py — EVAL-ONLY pedagogy_redirect generation.

Generates additional pedagogy_redirect dialogues into eval_sets/pedagogy_extra_raw/
for the purpose of EXPANDING THE PEDAGOGY WITHHOLDING EVAL ONLY.

CRITICAL: this writes to eval_sets/pedagogy_extra_raw/, NOT data/sft_raw/.
It does NOT touch any training data. The dialogues produced here are used
solely as held-out eval probes for the already-frozen A1/A3 adapters and the
prompt-only baselines B1-B4. They must never be added to any SFT corpus.

Generation uses a higher pedagogy_redirect_fraction than the original SFT
run, which (via deterministic_sample's smallest-hash-keep rule) yields a
strict superset of the seeds the SFT run used. The downstream probe builder
(build_pedagogy_extra_probe.py) then EXCLUDES the seed_ids already present in
the existing eval_sets/redirect_probe.jsonl pedagogy probes, so the extra
probes are guaranteed non-overlapping with the current 21.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation import pedagogy_redirect as ped


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.55,
                    help="pedagogy_redirect_fraction; superset of the SFT run's seeds.")
    ap.add_argument("--dialogues-per-seed", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output-dir",
                    default=str(ROOT / "eval_sets" / "pedagogy_extra_raw"))
    ap.add_argument("--config", default=str(ROOT / "config" / "generation.yaml"))
    args = ap.parse_args()

    out = Path(args.output_dir)
    # Guard: refuse to write anywhere under data/sft_raw or data/sft_filtered.
    resolved = out.resolve()
    forbidden = (ROOT / "data" / "sft_raw", ROOT / "data" / "sft_filtered")
    for fb in forbidden:
        if str(resolved).startswith(str(fb.resolve())):
            raise SystemExit(f"REFUSING to write into training data path: {resolved}")

    out.mkdir(parents=True, exist_ok=True)
    print(f"EVAL-ONLY pedagogy generation -> {out}")
    print(f"  fraction={args.fraction} dialogues_per_seed={args.dialogues_per_seed}")
    print("  (training data under data/ is NOT touched)")

    res = await ped.generate_batch(
        cefr_levels=["A1", "A2", "B1", "B2", "C1", "C2"],
        seeds_dir=str(ROOT / "data" / "seeds"),
        output_dir=str(out),
        failures_path=str(out / "_failures.jsonl"),
        config_path=args.config,
        concurrency=args.concurrency,
        pedagogy_redirect_fraction=args.fraction,
        dialogues_per_seed=args.dialogues_per_seed,
    )
    total = sum(res.values())
    print(f"\nWritten this run by level: {res}")
    print(f"Total: {total}")
    # Report total records now in the eval-extra dir
    grand = 0
    for f in sorted(out.glob("pedagogy_redirect_*.jsonl")):
        n = sum(1 for _ in f.open("rb"))
        print(f"  {f.name}: {n}")
        grand += n
    print(f"Eval-extra pool total: {grand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
