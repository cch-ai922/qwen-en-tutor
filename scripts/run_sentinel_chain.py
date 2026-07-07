"""run_sentinel_chain.py — Sentinel-aware DPO chain for one paper condition.

After a condition's SFT adapter is trained, this script runs:

  1. offline sentinel pair generation   (no model, ~seconds)
  2. on-policy sentinel pair generation (student SFT adapter on GPU)
  3. filter_dpo for the sentinel pool   (ensures schema validation)

The teacher is NOT required for either step — both use only the
student (or no model at all for offline). Re-running is safe; both
generators skip ids already on disk.

After this, run the DPO training stage for the condition. The updated
``training/dpo.py`` will pick up the new
``data/dpo_filtered/sentinel_<level>_passed.jsonl`` automatically.

Usage:

    python scripts/run_sentinel_chain.py \\
        --condition a1 \\
        --sft-adapter outputs/paper/a1/sft \\
        --sft-filtered-dir data/sft_filtered

For A3 (no specialized redirects, but persistent kept) use
sft_filtered_dir=data/sft_filtered_a3. A4 has no persistent streams,
so the sentinel chain produces zero pairs there — skip the chain for
A4 entirely.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger("sentinel_chain")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate offline + on-policy sentinel DPO pairs and "
                    "filter them, in one pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--condition", required=True, choices=("a1", "a3", "a5"))
    p.add_argument("--sft-adapter", required=True,
                   help="Path to the trained SFT adapter (outputs/paper/<c>/sft).")
    p.add_argument("--base-model",
                   default="./vendor/models/Qwen_3.5_0.8B-Base",
                   help="Base model dir.")
    p.add_argument("--sft-filtered-dir",
                   help="Where to read persistent_* SFT records from. "
                        "Defaults to data/sft_filtered (A1) or "
                        "data/sft_filtered_a3 (A3).")
    p.add_argument("--levels", default="A1,A2,B1,B2,C1,C2")
    p.add_argument("--output-dir", default=None,
                   help="dpo_raw output dir for sentinel files. Defaults to "
                        "data/dpo_raw (legacy/A1) or data/dpo_raw_<condition> "
                        "for a3/a5 if that dir exists.")
    p.add_argument("--filtered-dir", default=None,
                   help="dpo_filtered output dir. Defaults to data/dpo_filtered "
                        "(legacy/A1) or data/dpo_filtered_<condition> for a3/a5 "
                        "if that dir exists.")
    p.add_argument("--concurrency", type=int, default=2,
                   help="On-policy student inference concurrency. The 3060 "
                        "+ Qwen3.5-0.8B is happy at 2.")
    p.add_argument("--target-max-tokens", type=int, default=320)
    p.add_argument("--target-temperature", type=float, default=0.7)
    p.add_argument("--skip-offline", action="store_true",
                   help="Skip the offline (strip-marker) phase.")
    p.add_argument("--skip-on-policy", action="store_true",
                   help="Skip the on-policy (regen) phase.")
    p.add_argument("--skip-filter", action="store_true",
                   help="Skip the filter_dpo phase.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from qwen_tutor.generation import sentinel_pairs as sm

    if args.sft_filtered_dir is None:
        sft_dir = ROOT / ("data/sft_filtered" if args.condition == "a1"
                          else f"data/sft_filtered_{args.condition}")
    else:
        sft_dir = Path(args.sft_filtered_dir)

    # Per-condition dpo_raw + dpo_filtered. CLI override wins; otherwise
    # auto-pick data/dpo_raw_<condition> when it exists, else fall back to
    # the legacy shared data/dpo_raw (A1 only).
    if args.output_dir is not None:
        sentinel_raw_dir = Path(args.output_dir)
    else:
        per_cond = ROOT / f"data/dpo_raw_{args.condition}"
        sentinel_raw_dir = per_cond if per_cond.exists() else ROOT / "data" / "dpo_raw"
    sentinel_raw_dir.mkdir(parents=True, exist_ok=True)
    sentinel_failures = ROOT / "data" / "sentinel_failures.jsonl"
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]

    print(f"=== sentinel chain for condition {args.condition} ===")
    print(f"  sft_filtered_dir = {sft_dir}")
    print(f"  output_dir      = {sentinel_raw_dir}")
    print(f"  levels          = {levels}")
    print(f"  sft_adapter     = {args.sft_adapter}")

    # ---- Phase 1: offline (deterministic text manipulation) ----
    if not args.skip_offline:
        print("\n--- phase 1: offline sentinel pairs (no model) ---")
        res = sm.generate_offline_batch(
            cefr_levels=levels,
            sft_filtered_dir=sft_dir,
            output_dir=sentinel_raw_dir,
            failures_path=sentinel_failures,
        )
        print(f"  offline pairs written: {res}")

    # ---- Phase 2: on-policy (needs SFT-trained student on GPU) ----
    if not args.skip_on_policy:
        print("\n--- phase 2: on-policy sentinel pairs (student SFT adapter) ---")
        from qwen_tutor.training.eval.run_eval import HFTargetModelClient

        print(f"  loading student: base={args.base_model} adapter={args.sft_adapter}")
        student = HFTargetModelClient.from_pretrained(
            base_model_id=args.base_model,
            adapter_path=args.sft_adapter,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        print("  student loaded; starting on-policy generation")
        res = await sm.generate_on_policy_batch(
            target=student,
            cefr_levels=levels,
            sft_filtered_dir=sft_dir,
            output_dir=sentinel_raw_dir,
            failures_path=sentinel_failures,
            concurrency=args.concurrency,
            target_max_tokens=args.target_max_tokens,
            target_temperature=args.target_temperature,
        )
        print(f"  on-policy pairs written: {res}")

        # Free GPU before filter pass.
        del student
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---- Phase 3: filter_dpo for the sentinel pool ----
    if not args.skip_filter:
        print("\n--- phase 3: filter_dpo (sentinel pool only) ---")
        from qwen_tutor.generation.filters.banned_terms import BannedTermsFilter
        from qwen_tutor.generation.filters.mode_consistency import ModeConsistencyFilter
        from qwen_tutor.generation.filters.naturalness import NaturalnessFilter
        from qwen_tutor.generation.filters.non_latin_script import NonLatinScriptFilter
        from qwen_tutor.generation.filters.pipeline import FilterPipeline
        from qwen_tutor.generation.filters.speaks_l1_sanity import SpeaksL1SanityFilter
        from qwen_tutor.schemas import DPOExample

        filters_list = [
            SpeaksL1SanityFilter(),
            NonLatinScriptFilter(),
            BannedTermsFilter(),
            ModeConsistencyFilter(),
            NaturalnessFilter(),
        ]
        pipeline = FilterPipeline(filters_list, short_circuit=True)
        # Same per-condition auto-pick as for the raw dir.
        if args.filtered_dir is not None:
            dpo_filtered_dir = Path(args.filtered_dir)
        else:
            per_cond = ROOT / f"data/dpo_filtered_{args.condition}"
            dpo_filtered_dir = per_cond if per_cond.exists() else ROOT / "data" / "dpo_filtered"
        dpo_filtered_dir.mkdir(parents=True, exist_ok=True)

        for level in levels:
            in_path = sentinel_raw_dir / f"sentinel_{level}.jsonl"
            if not in_path.exists():
                print(f"  [{level}] skip — {in_path.name} not on disk")
                continue
            examples = list(DPOExample.from_jsonl(in_path))
            base = f"sentinel_{level}"
            passed = dpo_filtered_dir / f"{base}_passed.jsonl"
            failed = dpo_filtered_dir / f"{base}_failed.jsonl"
            for p in (passed, failed):
                if p.exists():
                    p.unlink()
            summary = await pipeline.run_stream(
                examples,
                passed_path=passed,
                failed_path=failed,
                concurrency=2,
                progress_desc=f"sentinel_filter[{base}]",
            )
            print(f"  {base}: {summary}")

    print("\n=== sentinel chain complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
