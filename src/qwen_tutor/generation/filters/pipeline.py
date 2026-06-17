"""Filter-pipeline orchestration.

Runs filters in the canonical order

    speaks_l1_sanity
    non_latin_script
    banned_terms
    mode_consistency
    naturalness
    locale_judge
    naturalness_judge (optional)

and short-circuits on the first failure by default. The locale_judge is
positioned LATE so any example that fails one of the cheap mechanical
filters never reaches the LLM-judge call.

The pipeline can run a single example or stream over an iterable. It
writes one record per example to both a ``passed.jsonl`` and a
``failed.jsonl`` file: each record carries the full per-filter result
list, so downstream auditing can reconstruct exactly why an example was
rejected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qwen_tutor.generation.filters.base import Filter, FilterableExample, FilterResult
from qwen_tutor.utils.runner import append_jsonl, gather_with_concurrency

logger = logging.getLogger(__name__)

CANONICAL_ORDER = (
    # 1) speaks_l1 sanity and non-Latin script checks are the cheapest (plain
    #    regex) and the most specific reject criteria, so they run before the
    #    other mechanical filters - this keeps malformed records out of the
    #    token-counting work the later filters do.
    "speaks_l1_sanity",
    "non_latin_script",
    "banned_terms",
    "mode_consistency",
    "naturalness",
    "locale_judge",
    "naturalness_judge",
)


@dataclass(frozen=True)
class FilterRunRecord:
    name: str
    passed: bool
    score: float | None
    reason: str | None
    skipped: bool = False

    @classmethod
    def from_result(cls, name: str, result: FilterResult) -> "FilterRunRecord":
        return cls(
            name=name,
            passed=result.passed,
            score=result.score,
            reason=result.reason,
        )

    @classmethod
    def skipped_record(cls, name: str) -> "FilterRunRecord":
        return cls(name=name, passed=True, score=None, reason="skipped (short-circuit)", skipped=True)


@dataclass
class PipelineResult:
    example_id: str
    passed: bool
    runs: list[FilterRunRecord] = field(default_factory=list)
    failing_filter: str | None = None
    failing_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _sort_filters(filters: list[Filter]) -> list[Filter]:
    """Sort filters into CANONICAL_ORDER; unknown names go to the end."""

    order = {name: i for i, name in enumerate(CANONICAL_ORDER)}
    return sorted(filters, key=lambda f: order.get(f.name, len(order)))


class FilterPipeline:
    def __init__(
        self,
        filters: list[Filter],
        short_circuit: bool = True,
    ) -> None:
        self.filters = _sort_filters(filters)
        self.short_circuit = short_circuit

    async def run_one(
        self, example: FilterableExample, *, extra_metadata: dict[str, Any] | None = None
    ) -> tuple[PipelineResult, dict[str, FilterResult]]:
        example_id = getattr(example, "id", "<unknown>")
        runs: list[FilterRunRecord] = []
        details: dict[str, FilterResult] = {}
        failing: str | None = None
        failing_reason: str | None = None
        for flt in self.filters:
            if failing and self.short_circuit:
                runs.append(FilterRunRecord.skipped_record(flt.name))
                continue
            try:
                result = await flt.check(example)
            except Exception as exc:  # noqa: BLE001
                logger.exception("filter %s raised on example %s", flt.name, example_id)
                result = FilterResult(
                    passed=False,
                    reason=f"filter raised {type(exc).__name__}: {exc}",
                )
            details[flt.name] = result
            runs.append(FilterRunRecord.from_result(flt.name, result))
            if not result.passed and failing is None:
                failing = flt.name
                failing_reason = result.reason

        passed = failing is None
        pr = PipelineResult(
            example_id=str(example_id),
            passed=passed,
            runs=runs,
            failing_filter=failing,
            failing_reason=failing_reason,
            extra=dict(extra_metadata or {}),
        )
        return pr, details

    async def run_stream(
        self,
        examples: Iterable[FilterableExample],
        passed_path: str | Path,
        failed_path: str | Path,
        concurrency: int = 8,
        extra_per_example: dict[str, dict[str, Any]] | None = None,
        progress_desc: str | None = None,
    ) -> dict[str, int]:
        """Run the pipeline against every example, writing JSONL outputs.

        Returns a summary ``{"total": n, "passed": n, "failed": n}``.
        """
        passed_path = Path(passed_path)
        failed_path = Path(failed_path)
        passed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.parent.mkdir(parents=True, exist_ok=True)

        items = list(examples)
        if not items:
            return {"total": 0, "passed": 0, "failed": 0}

        extra_lookup = extra_per_example or {}

        async def _process(ex: FilterableExample) -> PipelineResult:
            ex_id = getattr(ex, "id", "<unknown>")
            extra = extra_lookup.get(str(ex_id))
            pr, _details = await self.run_one(ex, extra_metadata=extra)
            return pr

        results = await gather_with_concurrency(
            [_process(ex) for ex in items],
            concurrency=concurrency,
            desc=progress_desc,
        )

        n_passed = 0
        n_failed = 0
        for ex, pr in zip(items, results):
            record = {
                "example": ex.model_dump() if hasattr(ex, "model_dump") else ex,
                "pipeline": pr.to_dict(),
            }
            if pr.passed:
                append_jsonl(passed_path, record)
                n_passed += 1
            else:
                append_jsonl(failed_path, record)
                n_failed += 1
        return {"total": len(items), "passed": n_passed, "failed": n_failed}
