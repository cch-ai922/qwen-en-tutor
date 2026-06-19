"""regen_redirect_streams.py — Regenerate the 7 single-shot redirect SFT
streams with the new ``violation_turn_idx`` metadata field.

What this does
--------------
1. Forces ``QWEN_TUTOR_PROMPTS=compact`` so the regenerated prompts (which
   ask the teacher for ``violation_turn_idx`` as a top-level JSON field)
   are the ones actually used.
2. Moves the existing ``data/sft_raw/<stream>_*.jsonl`` redirect files
   into a backup dir so the generator's resumability logic doesn't skip
   them all. (We need *new* records with the metadata field, not the
   old ones.)
3. Calls each of the 7 redirect generators' ``generate_batch`` over all
   levels for one variant. Output lands in ``data/sft_raw/``.
4. Does NOT re-run the filter cascade. That means
   ``data/sft_filtered/<redirect>_*_passed.jsonl`` stays unchanged, so
   A5 SFT continues to use the SAME training data A1/A3/A4 used —
   training consistency preserved.

The eval pipeline then reads from ``data/sft_raw/``, picks up the new
records (which carry ``metadata.generation.violation_turn_idx``), and
``build_eval_sets._find_redirect_violation_idx`` uses that authoritative
hint instead of the pivot heuristic.

Pre-requisites
--------------
* Teacher server up at ``config/generation.yaml::teacher.base_url``
  (the Qwen3.5-9B GGUF).
* ``data/seeds/<LEVEL>.jsonl`` produced by the seeds stage.

Usage
-----
::

  # End-to-end (all 7 streams):
  python scripts/regen_redirect_streams.py

  # Single axis (debug):
  python scripts/regen_redirect_streams.py --axis locale_redirect

The script is resumable per-axis (the generator skips ids already in
its output file). Killing and re-running picks up where it left off.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

# Force the compact-prompt variants (which carry the violation_turn_idx
# instruction). Must be set before any qwen_tutor import — read inside
# qwen_tutor.generation._prompt_select at module load.
os.environ.setdefault("QWEN_TUTOR_PROMPTS", "compact")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation import (  # noqa: E402
    language_redirect,
    locale_redirect,
    pedagogy_redirect,
    persona_redirect,
    redirect,
    role_swap_redirect,
    topic_redirect,
)
from qwen_tutor.generation.teacher import build_teacher_from_config  # noqa: E402

logger = logging.getLogger("regen_redirect")

LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
RAW_DIR = ROOT / "data" / "sft_raw"
BACKUP_DIR = ROOT / "data" / "sft_raw" / "_backup_pre_violation_turn_idx"

# Each entry: (axis name, module, default_fraction_param_name)
STREAMS = [
    ("redirect", redirect, "redirect_fraction"),
    ("locale_redirect", locale_redirect, "locale_redirect_fraction"),
    ("pedagogy_redirect", pedagogy_redirect, "pedagogy_redirect_fraction"),
    ("language_redirect", language_redirect, "language_redirect_fraction"),
    ("persona_redirect", persona_redirect, "persona_redirect_fraction"),
    ("topic_redirect", topic_redirect, "topic_redirect_fraction"),
    ("role_swap_redirect", role_swap_redirect, "role_swap_redirect_fraction"),
]


def _backup_existing(axis_filter: list[str] | None = None) -> int:
    """Move existing ``<axis>_<level>.jsonl`` files into a backup dir so
    the generator's per-file skip-if-exists logic doesn't immediately
    short-circuit. Idempotent (skips files already backed up)."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    n_moved = 0
    for axis, _, _ in STREAMS:
        if axis_filter and axis not in axis_filter:
            continue
        for level in LEVELS:
            src = RAW_DIR / f"{axis}_{level}.jsonl"
            if not src.exists():
                continue
            dst = BACKUP_DIR / f"{axis}_{level}.jsonl"
            if dst.exists():
                # Already backed up earlier — keep both: append timestamp
                # to the new backup so we don't clobber.
                import time
                ts = int(time.time())
                dst = BACKUP_DIR / f"{axis}_{level}.jsonl.{ts}"
            shutil.move(str(src), str(dst))
            n_moved += 1
    return n_moved


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


async def _run_axis(axis: str, module, fraction_param: str,
                    config_path: Path, teacher, concurrency: int,
                    max_tokens: int, temperature: float, cfg: dict) -> dict:
    """Dispatch to the axis-specific ``generate_batch``. Each module's
    function takes slightly different kwargs — we call them by name."""
    gen = cfg.get("generation", {})
    fraction = float(gen.get(fraction_param, 0.05))
    failures = RAW_DIR / "_failures.jsonl"
    common = dict(
        cefr_levels=list(LEVELS),
        seeds_dir=ROOT / "data" / "seeds",
        output_dir=RAW_DIR,
        failures_path=failures,
        config_path=str(config_path),
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        teacher=teacher,
    )
    # Each module's generate_batch has its own fraction kwarg name.
    fraction_kwargs = {
        "redirect": "redirect_fraction",
        "locale_redirect": "locale_redirect_fraction",
        "pedagogy_redirect": "pedagogy_redirect_fraction",
        "language_redirect": "language_redirect_fraction",
        "persona_redirect": "persona_redirect_fraction",
        "topic_redirect": "topic_redirect_fraction",
        "role_swap_redirect": "role_swap_redirect_fraction",
    }
    kw = {fraction_kwargs[axis]: fraction}
    return await module.generate_batch(**common, **kw)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate single-shot redirect SFT streams with "
                    "the new violation_turn_idx metadata field.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/generation.yaml")
    p.add_argument("--axis", default=None,
                   choices=[a for a, _, _ in STREAMS] + ["all"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--skip-backup", action="store_true",
                   help="Don't backup existing sft_raw files (use only if "
                        "the previous run was killed mid-way and the files "
                        "are already partially regenerated).")
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

    print(f"=== Redirect-stream regen (compact prompts, violation_turn_idx) ===")
    print(f"raw dir : {RAW_DIR}")
    print(f"backup  : {BACKUP_DIR}")
    print(f"prompts : QWEN_TUTOR_PROMPTS={os.environ['QWEN_TUTOR_PROMPTS']!r}")
    print()

    targets = (
        [s for s in STREAMS if s[0] == args.axis]
        if args.axis and args.axis != "all"
        else STREAMS
    )
    target_names = [t[0] for t in targets]

    if not args.skip_backup:
        n = _backup_existing(axis_filter=target_names)
        print(f"backed up {n} existing files to {BACKUP_DIR}")
        print()

    cfg = _load_yaml(config_path)
    teacher = build_teacher_from_config(str(config_path), role="teacher")

    for axis, module, fraction_param in targets:
        print(f"--- axis: {axis} ---")
        result = await _run_axis(
            axis, module, fraction_param,
            config_path=config_path, teacher=teacher,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens, temperature=args.temperature, cfg=cfg,
        )
        print(f"   wrote (per level): {result}")
        print()

    print("Next: python scripts/build_eval_sets.py "
          "(build_eval_sets._find_redirect_violation_idx prefers "
          "metadata.violation_turn_idx when present)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
