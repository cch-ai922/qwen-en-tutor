"""run_diversity.py  -  scan filtered generation output and report distribution.

Walks a directory of filter-pipeline JSONL files (``*_passed.jsonl`` shape
with ``{example, pipeline}`` records, or raw schema records) and feeds them
to ``DiversityTracker``. Reports the top names, cities, foods, and life-
domain categories present in the assistant turns + metadata.

This is the place to look when checking whether seed quota-balancing is
holding up after generation + filtering: if one category dominates here,
either the teacher dropped too many seeds in another category, or filtering
hit a specific bucket hard.

Usage::

    # Scan filtered SFT (default)
    python scripts/run_diversity.py

    # Scan filtered DPO data
    python scripts/run_diversity.py --input-dir data/dpo_filtered

    # Save a JSON report alongside the printed text summary
    python scripts/run_diversity.py --json-out data/diversity_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

# Windows consoles (cp1252/cp949) can't print Unicode by default.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_tutor.generation.diversity import DiversityTracker  # noqa: E402

logger = logging.getLogger("run_diversity")

DEFAULT_INPUT_DIR = Path("data/sft_filtered")


def _iter_passed_examples(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield the inner ``example`` dict from every ``*_passed.jsonl`` line.

    Tolerant of two shapes on disk:
      * filter-pipeline wrapped: ``{"example": {...}, "pipeline": {...}}``
      * raw schema dump:        ``{... schema fields ...}``
    """
    if not directory.exists():
        logger.warning("input directory %s does not exist", directory)
        return
    for path in sorted(directory.glob("*.jsonl")):
        # Skip *_failed.jsonl — we only care about what made it through.
        if path.stem.endswith("_failed"):
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                example = obj.get("example", obj)
                if isinstance(example, dict):
                    yield example


def _extract_category(example: dict[str, Any]) -> str | None:
    """Pull category from either ``metadata.category`` (SFT/DPO) or ``category``
    (raw ScenarioSeed). Returns ``None`` if absent."""
    meta = example.get("metadata")
    if isinstance(meta, dict) and meta.get("category"):
        return str(meta["category"])
    if example.get("category"):
        return str(example["category"])
    return None


def _extract_assistant_text(example: dict[str, Any]) -> Iterator[str]:
    """Yield assistant-turn content strings from the example.

    Supports SFT/EvaluationExample (``messages``) and DPOExample
    (``chosen`` + ``rejected`` + ``prompt_messages``). For DPO the chosen
    side is treated as the assistant text since that's what the model is
    pushed toward.
    """
    msgs = example.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, str):
                    yield content
        return

    chosen = example.get("chosen")
    if isinstance(chosen, dict) and isinstance(chosen.get("content"), str):
        yield chosen["content"]


def scan_directory(directory: Path) -> DiversityTracker:
    tracker = DiversityTracker()
    n = 0
    for example in _iter_passed_examples(directory):
        n += 1
        tracker.track_category(_extract_category(example))
        for text in _extract_assistant_text(example):
            tracker.scan_text(text)
    logger.info("scanned %d examples from %s", n, directory)
    return tracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory of filter-pipeline *_passed.jsonl files (default: data/sft_filtered)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write the report as JSON",
    )
    parser.add_argument("--top-names", type=int, default=20)
    parser.add_argument("--top-cities", type=int, default=20)
    parser.add_argument("--top-foods", type=int, default=20)
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.25,
        help="warn when any top item exceeds this share (default 0.25 = 25%)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tracker = scan_directory(args.input_dir)
    report = tracker.report(
        top_names=args.top_names,
        top_cities=args.top_cities,
        top_foods=args.top_foods,
        warn_threshold=args.warn_threshold,
    )

    print(report.format_text())

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("wrote JSON report to %s", args.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
