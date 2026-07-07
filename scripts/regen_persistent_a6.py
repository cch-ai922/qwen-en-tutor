"""regen_persistent_a6.py — Regenerate persistent SFT streams with the
GENERIC sentinel (``[SESSION_END]``, no axis label) for the A6 ablation.

What this does
--------------
1. Sets ``QWEN_TUTOR_SENTINEL_FORMAT=generic`` *before* importing the
   persistent stream module and prompt module, so:
     * persistent_redirect._sentinel_for() returns ``[SESSION_END]``
     * the rendered deployment system prompt embedded in each training
       record carries the matching generic instructions in the
       ``[persistence]`` block
2. Calls ``persistent_redirect.generate_batch`` for each of the four
   persistent axes, writing raw records to
   ``data/sft_raw_a6_persistent/<axis>_<level>.jsonl``.
3. Runs the same filter cascade used by the main pipeline against the new
   raw dir, writing ``data/sft_filtered_a6_persistent/<axis>_<level>_passed.jsonl``.
4. After this completes, run
   ``python scripts/setup_paper_ablation_data.py --conditions a6`` to
   materialize ``data/sft_filtered_a6/`` (the SFT input for A6 training).

Pre-requisites
--------------
* Teacher server up at ``config/generation.yaml::teacher.base_url``.
  (Stop SFT/DPO training first if running — they cannot share the GPU
  on a 12 GB device.)
* ``data/seeds/<LEVEL>.jsonl`` already produced by the seeds stage.

This script is resumable: the underlying generator skips ids already
present in the output JSONL.

Usage
-----
::

  # A1-comparable volume (fraction_multiplier 4.0 + dialogues_per_seed 2)
  python scripts/regen_persistent_a6.py --fraction-multiplier 4.0 --dialogues-per-seed 2

  # Generation only, skip filter:
  python scripts/regen_persistent_a6.py --skip-filter
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Force the generic-sentinel variant. MUST be set before persistent_redirect
# and prompts are imported below.
os.environ["QWEN_TUTOR_SENTINEL_FORMAT"] = "generic"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Project imports (after env var + sys.path).
from qwen_tutor.generation import persistent_redirect  # noqa: E402
from qwen_tutor.generation.teacher import build_teacher_from_config  # noqa: E402

logger = logging.getLogger("regen_persistent_a6")

PERSISTENT_AXES = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

RAW_DIR = ROOT / "data" / "sft_raw_a6_persistent"
FILTERED_DIR = ROOT / "data" / "sft_filtered_a6_persistent"
FAILURES_PATH = ROOT / "data" / "sft_raw_a6_persistent" / "_failures.jsonl"


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


async def _generate_one_axis(axis: str, cfg: dict,
                             config_path: Path,
                             teacher,
                             concurrency: int,
                             max_tokens: int,
                             temperature: float,
                             fraction_multiplier: float = 1.0,
                             dialogues_per_seed: int = 1) -> dict:
    gen = cfg.get("generation", {})
    fraction = float(gen.get(f"{axis}_fraction", 0.05)) * fraction_multiplier
    fraction = min(fraction, 1.0)
    return await persistent_redirect.generate_batch(
        axis=axis,
        cefr_levels=list(LEVELS),
        seeds_dir=ROOT / "data" / "seeds",
        output_dir=RAW_DIR,
        failures_path=FAILURES_PATH,
        config_path=str(config_path),
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        teacher=teacher,
        persistent_fraction=fraction,
        dialogues_per_seed=dialogues_per_seed,
    )


async def _run_filter_stage(config_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_generation import _build_filters, _run_filter  # type: ignore

    from qwen_tutor.schemas import SFTExample

    cfg = _load_yaml(config_path)
    filters_list = _build_filters(cfg, config_path)
    fcfg = cfg.get("filtering", {})
    await _run_filter(
        in_dir=RAW_DIR,
        out_dir=FILTERED_DIR,
        schema=SFTExample,
        file_patterns=[f"{axis}_{{level}}.jsonl" for axis in PERSISTENT_AXES],
        levels=list(LEVELS),
        filters_list=filters_list,
        fcfg=fcfg,
        label="filter_sft_a6",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate persistent SFT streams with the GENERIC "
                    "[SESSION_END] sentinel for the A6 ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/generation.yaml")
    p.add_argument("--axis", default=None,
                   choices=list(PERSISTENT_AXES) + ["all"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--fraction-multiplier", type=float, default=4.0,
                   help="Default 4.0 (with --dialogues-per-seed 2) targets "
                        "~780 records to match A1's persistent volume.")
    p.add_argument("--dialogues-per-seed", type=int, default=2)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-filter", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    print(f"=== A6 persistent regen (generic [SESSION_END]) ===")
    print(f"SENTINEL_FORMAT    : {os.environ.get('QWEN_TUTOR_SENTINEL_FORMAT')!r}")
    print(f"raw output dir     : {RAW_DIR}")
    print(f"filtered output dir: {FILTERED_DIR}")
    print(f"fraction_multiplier={args.fraction_multiplier} "
          f"dialogues_per_seed={args.dialogues_per_seed}")
    print()

    # Sanity: make sure persistent_redirect picked up the env var.
    test_sentinel = persistent_redirect._sentinel_for("persistent_off_topic")
    if test_sentinel != "[SESSION_END]":
        raise SystemExit(
            f"persistent_redirect._sentinel_for returned {test_sentinel!r}, "
            f"expected '[SESSION_END]'. env var didn't take effect.")
    print(f"Sentinel check: persistent_off_topic -> {test_sentinel}")
    print()

    if not args.skip_generation:
        cfg = _load_yaml(config_path)
        teacher = build_teacher_from_config(str(config_path), role="teacher")

        axes = PERSISTENT_AXES if args.axis in (None, "all") else (args.axis,)
        for axis in axes:
            print(f"--- axis: {axis} ---")
            result = await _generate_one_axis(
                axis=axis,
                cfg=cfg,
                config_path=config_path,
                teacher=teacher,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                fraction_multiplier=args.fraction_multiplier,
                dialogues_per_seed=args.dialogues_per_seed,
            )
            print(f"   wrote (per level): {result}")
            print()

    if not args.skip_filter:
        print("--- filter cascade ---")
        await _run_filter_stage(config_path)
        print(f"   filtered output: {FILTERED_DIR}/")

    print()
    print("Next: python scripts/setup_paper_ablation_data.py --conditions a6")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
