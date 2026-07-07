"""One-off: regenerate redirect for C2 only, targeting ~50 new records.

Calls redirect.generate_batch directly — no generation.yaml changes.
Run while chain step-1 wait loop is active; chain advances automatically
when this process exits.

Usage:
  $env:PYTHONUTF8="1"; python scripts/regen_c2_redirect.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation import redirect
from qwen_tutor.generation.teacher import build_teacher_from_config

CONFIG = str(ROOT / "config" / "generation.yaml")
OUTPUT_DIR = ROOT / "data" / "sft_raw"
FAILURES = ROOT / "data" / "redirect_failures.jsonl"


async def main() -> None:
    teacher = build_teacher_from_config(CONFIG, role="teacher")
    # fraction=0.60: 174 C2 seeds × 0.60 ≈ 104 selected, ~52 already done → ~52 new
    result = await redirect.generate_batch(
        cefr_levels=["C2"],
        output_dir=OUTPUT_DIR,
        failures_path=FAILURES,
        config_path=CONFIG,
        concurrency=8,
        redirect_fraction=0.60,
        dialogues_per_seed=1,
        teacher=teacher,
    )
    print(f"C2 redirect: wrote {result.get('C2', 0)} new records")


if __name__ == "__main__":
    asyncio.run(main())
