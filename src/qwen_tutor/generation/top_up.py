"""One-shot top-up regeneration to rebalance post-filter category distribution.

Background: seeds are generated with a quota-balanced round-robin across
``schemas.CATEGORIES``, but filter pass-rates differ per category. After
``filter_sft`` / ``filter_eval`` / ``filter_dpo`` finish, some life-domain
categories end up under-represented in the surviving data.

This module computes the per-pool deficit per ``(cefr_level, locale,
category)`` triple, takes the worst case across the three pools, and
returns the resulting category quota to ``seeds.generate_batch``. The
downstream stages (sft + redirects + register + eval + filter_*) are then
re-run by the caller (they are resumable, so they only process the new
seeds).

The top-up loop is intentionally **single-shot**: it does not iterate
until convergence. A category that consistently fails filtering needs a
root-cause fix (prompt or filter), not more attempts. ``max_top_up_per_category``
caps the per-(level, locale) regen so a stubborn category cannot
runaway-blow the seed budget.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from qwen_tutor.schemas import CATEGORIES

logger = logging.getLogger(__name__)


PoolKey = str  # "sft" | "eval" | "dpo"
TripleKey = tuple[str, str, str]  # (level, locale, category)


# ---------------------------------------------------------------------------
# Pool readers
# ---------------------------------------------------------------------------


def _iter_passed_examples(directory: Path) -> Iterable[dict[str, Any]]:
    """Yield the inner ``example`` dict from every ``*_passed.jsonl`` line.

    Tolerates both filter-pipeline wrapped (``{example, pipeline}``) and
    raw schema-dump shapes on disk.
    """
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
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


def _example_triple(example: dict[str, Any]) -> TripleKey | None:
    """Pull (level, locale, category) from an SFT / DPO / Eval example.

    Returns ``None`` if any field is missing (legacy data is skipped from
    the count rather than mapped to ``"general"`` so the report shows
    exactly which pool needs to be regenerated to pick up categories).
    """
    meta = example.get("metadata")
    if not isinstance(meta, dict):
        return None
    # EvaluationExample uses ``learner_cefr_target`` instead of ``cefr_level``.
    level = meta.get("cefr_level") or meta.get("learner_cefr_target")
    locale = meta.get("locale")
    category = meta.get("category")
    if not (level and locale and category):
        return None
    if category == "general":
        # Legacy default — not a real life domain. Don't count it.
        return None
    return (str(level), str(locale), str(category))


def count_survivors(directory: Path) -> dict[TripleKey, int]:
    """Count survivors per (level, locale, category) in one filtered pool."""
    counts: dict[TripleKey, int] = defaultdict(int)
    for ex in _iter_passed_examples(directory):
        key = _example_triple(ex)
        if key is not None:
            counts[key] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Deficit math
# ---------------------------------------------------------------------------


def compute_deficits(
    *,
    sft_dir: Path,
    eval_dir: Path,
    dpo_dir: Path,
    target_per_category: int,
    levels: list[str],
    locales: list[str],
    max_top_up_per_category: int,
    categories: list[str] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, Any]]:
    """Compute per-(level, locale, category) seed top-up needs.

    Returns a tuple ``(quotas, report)``:

      * ``quotas`` maps ``(level, locale) -> {category: count_to_generate}``
        suitable to pass straight to ``seeds.generate_batch(category_quotas=...)``.
      * ``report`` is a JSON-friendly dict summarizing per-pool survivors
        and the chosen quota for each triple (useful for logging).

    For each ``(level, locale, category)``:
      1. Read survivor counts in each of the three pools.
      2. ``deficit_per_pool = max(0, target_per_category - survivors)``.
      3. ``deficit = max across pools`` — generate enough new seeds to
         satisfy the worst-off pool. Pools that are already over target
         contribute 0.
      4. Cap at ``max_top_up_per_category`` to prevent runaway when a
         category is being killed by a systematic filter / prompt bug.
    """
    sft_counts = count_survivors(sft_dir)
    eval_counts = count_survivors(eval_dir)
    dpo_counts = count_survivors(dpo_dir)

    quotas: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    triple_reports: list[dict[str, Any]] = []

    cats = categories or list(CATEGORIES)
    for level in levels:
        for locale in locales:
            for category in cats:
                key = (level, locale, category)
                survivors = {
                    "sft": sft_counts.get(key, 0),
                    "eval": eval_counts.get(key, 0),
                    "dpo": dpo_counts.get(key, 0),
                }
                per_pool_deficit = {
                    pool: max(0, target_per_category - n) for pool, n in survivors.items()
                }
                raw_deficit = max(per_pool_deficit.values())
                deficit = min(raw_deficit, max_top_up_per_category)
                if deficit > 0:
                    quotas[(level, locale)][category] = deficit
                triple_reports.append(
                    {
                        "level": level,
                        "locale": locale,
                        "category": category,
                        "survivors": survivors,
                        "deficit_per_pool": per_pool_deficit,
                        "raw_deficit": raw_deficit,
                        "capped_deficit": deficit,
                    }
                )

    report: dict[str, Any] = {
        "target_per_category": target_per_category,
        "max_top_up_per_category": max_top_up_per_category,
        "totals_by_pool": {
            "sft": sum(sft_counts.values()),
            "eval": sum(eval_counts.values()),
            "dpo": sum(dpo_counts.values()),
        },
        "per_triple": triple_reports,
        "seeds_to_generate_by_pair": {
            f"{lvl}/{loc}": sum(cats.values())
            for (lvl, loc), cats in quotas.items()
        },
        "total_seeds_to_generate": sum(
            sum(cats.values()) for cats in quotas.values()
        ),
    }
    return dict(quotas), report


def log_deficit_report(report: dict[str, Any]) -> None:
    """Pretty-print the deficit report at INFO level."""
    total = report["total_seeds_to_generate"]
    logger.info(
        "top_up: target=%d/cat, cap=%d, will generate %d new seeds total",
        report["target_per_category"],
        report["max_top_up_per_category"],
        total,
    )
    logger.info("  current survivors by pool: %s", report["totals_by_pool"])
    if total == 0:
        logger.info("  no deficits — every (level, locale, category) at or above target")
        return
    logger.info("  seeds to generate by (level/locale): %s", report["seeds_to_generate_by_pair"])
    # Surface the worst-off triples so the user can spot pathological cases.
    worst = sorted(
        (t for t in report["per_triple"] if t["raw_deficit"] > 0),
        key=lambda t: t["raw_deficit"],
        reverse=True,
    )[:10]
    if worst:
        logger.info("  worst deficits (top 10):")
        for t in worst:
            logger.info(
                "    %s/%s/%s  survivors=%s  raw=%d  capped=%d",
                t["level"], t["locale"], t["category"],
                t["survivors"], t["raw_deficit"], t["capped_deficit"],
            )
