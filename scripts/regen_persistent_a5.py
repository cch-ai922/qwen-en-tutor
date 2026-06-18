"""regen_persistent_a5.py — Regenerate persistent SFT streams with the
fixed-turn-7 (V2-only) variant for the A5 ablation.

What this does
--------------
1. Sets ``QWEN_TUTOR_PERSISTENT_FORCED_VARIANT=1`` *before* importing the
   persistent stream module, so every generated record uses the V2
   ("medium") structure with sentinel at turn 7.
2. Calls ``persistent_redirect.generate_batch`` for each of the four
   persistent axes, writing raw records to
   ``data/sft_raw_a5_persistent/<axis>_<level>.jsonl``.
3. Runs the same filter cascade used by the main pipeline against the new
   raw dir, writing ``data/sft_filtered_a5_persistent/<axis>_<level>_passed.jsonl``.
4. After this completes, run
   ``python scripts/setup_paper_ablation_data.py --conditions a5`` to
   materialize ``data/sft_filtered_a5/`` (the SFT input for A5 training).

Pre-requisites
--------------
* Teacher server up at ``config/generation.yaml::teacher.base_url``.
  (Stop A1 DPO first if you are still mid-training — they cannot share
  the GPU on a 12 GB device.)
* ``data/seeds/<LEVEL>.jsonl`` already produced by the seeds stage.

This script is resumable: the underlying generator skips ids already
present in the output JSONL, so a killed-and-resumed run picks up where
it left off.

Usage
-----
::

  # End-to-end (generation + filter):
  python scripts/regen_persistent_a5.py

  # Generation only, skip filter:
  python scripts/regen_persistent_a5.py --skip-filter

  # Only one axis (debug):
  python scripts/regen_persistent_a5.py --axis persistent_off_topic
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Force the V2-only variant. This MUST be set before persistent_redirect is
# imported below — it is read inside _select_structure_idx on every call,
# but setting it here guarantees no import-order surprise.
os.environ.setdefault("QWEN_TUTOR_PERSISTENT_FORCED_VARIANT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Project imports (after env var + sys.path).
from qwen_tutor.generation import persistent_redirect  # noqa: E402
from qwen_tutor.generation.teacher import build_teacher_from_config  # noqa: E402

logger = logging.getLogger("regen_persistent_a5")

PERSISTENT_AXES = (
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
)
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# Where the V2-only raw + filtered output lands. Kept distinct from the
# main pipeline dirs so this regen never clobbers the 4-variant data A1
# was trained on.
RAW_DIR = ROOT / "data" / "sft_raw_a5_persistent"
FILTERED_DIR = ROOT / "data" / "sft_filtered_a5_persistent"
FAILURES_PATH = ROOT / "data" / "sft_raw_a5_persistent" / "_failures.jsonl"


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


async def _generate_one_axis(axis: str, cfg: dict,
                             config_path: Path,
                             teacher,
                             concurrency: int,
                             max_tokens: int,
                             temperature: float) -> dict:
    """Run persistent generation for ONE axis. Same fraction as the main
    pipeline so the A5 persistent volume roughly matches A1's."""
    gen = cfg.get("generation", {})
    fraction = float(gen.get(f"{axis}_fraction", 0.05))
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
        dialogues_per_seed=1,
    )


async def _run_filter_stage(config_path: Path) -> None:
    """Apply the standard filter cascade to RAW_DIR, writing to FILTERED_DIR.

    Reuses ``run_generation._run_filter`` and ``_build_filters`` to stay
    in sync with the main pipeline's filter set. We only run on the four
    persistent file patterns.
    """
    # Local import to avoid pulling the heavy run_generation imports
    # unless the caller actually wants the filter stage.
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
        label="filter_sft_a5",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate persistent SFT streams with V2-only "
                    "(sentinel fixed at turn 7) for the A5 ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/generation.yaml",
                   help="Generation config path.")
    p.add_argument("--axis", default=None,
                   choices=list(PERSISTENT_AXES) + ["all"],
                   help="Restrict to one axis (debug). Default: all four.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip the generation stage (only run the filter).")
    p.add_argument("--skip-filter", action="store_true",
                   help="Skip the filter stage (only run generation).")
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

    print(f"=== A5 persistent regen ===")
    print(f"FORCED_VARIANT     : {os.environ.get('QWEN_TUTOR_PERSISTENT_FORCED_VARIANT')!r}")
    print(f"raw output dir     : {RAW_DIR}")
    print(f"filtered output dir: {FILTERED_DIR}")
    print()

    if not args.skip_generation:
        # Verify variant index range up-front so a typo doesn't waste teacher time.
        idx = int(os.environ["QWEN_TUTOR_PERSISTENT_FORCED_VARIANT"])
        v = persistent_redirect._STRUCTURE_VARIANTS[idx]
        print(f"Using variant {idx}: name={v['name']!r}, "
              f"message_count={v['message_count']}, "
              f"sentinel_turn={v['sentinel_turn']}")
        if v["sentinel_turn"] != 7:
            print(f"WARNING: forced variant sentinel_turn={v['sentinel_turn']}, "
                  f"not 7. A5 expects the fixed-turn-7 design.", file=sys.stderr)
        print()

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
            )
            print(f"   wrote (per level): {result}")
            print()

    if not args.skip_filter:
        print("--- filter cascade ---")
        await _run_filter_stage(config_path)
        print(f"   filtered output: {FILTERED_DIR}/")

    print()
    print("Next: python scripts/setup_paper_ablation_data.py --conditions a5")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
