"""Side-by-side comparison of two evaluation runs.

Loads each run's ``summary.json``, walks the per-level + overall metric
trees, computes deltas, and flags any metric that regressed beyond a
configurable threshold (default 5 percentage points / 5% relative,
depending on the metric).

Some metrics are "higher is better" (topic_adherence, locale_fidelity,
redirect_success_rate, naturalness_judge_mean, all of the *_rate /
*_mean except above_band_ratio); ``above_band_ratio`` and the
``mean_sentence_length`` are "context-dependent" so we only flag
``above_band_ratio`` as a regression when it *increases* materially.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_THRESHOLD = 0.05


# Direction: +1 means "higher is better"; -1 means "lower is better".
# Metrics not listed are reported but not flagged either way.
METRIC_DIRECTIONS: dict[str, int] = {
    "topic_adherence_mean": +1,
    "locale_fidelity_mean": +1,
    "redirect_success_rate": +1,
    "naturalness_judge_mean": +1,
    "mode_consistency_conversation_rate": +1,
    "mode_consistency_evaluation_rate": +1,
    "eval_json_valid_rate": +1,
    # level_fidelity sub-metrics
    "above_band_ratio_mean": -1,
    "contraction_rate_mean": 0,
    "discourse_marker_rate_mean": 0,
    "mean_sentence_length_mean": 0,
    # eval score means — higher better only if the calibration is
    # consistent across runs; flag any change > threshold either way.
    "fluency": 0,
    "accuracy": 0,
    "vocabulary": 0,
    "interaction": 0,
    "topic_adherence": 0,
}


@dataclass
class MetricDelta:
    path: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    direction: int
    regression: bool
    note: str = ""


@dataclass
class ComparisonReport:
    baseline_run_id: str
    candidate_run_id: str
    threshold: float
    deltas: list[MetricDelta] = field(default_factory=list)
    regressions: list[MetricDelta] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append(
            f"COMPARISON: baseline={self.baseline_run_id}  vs  candidate={self.candidate_run_id}"
        )
        lines.append(f"threshold={self.threshold:.3f}")
        lines.append("=" * 78)
        lines.append("")
        # Group deltas by section (overall / per-level).
        sections: dict[str, list[MetricDelta]] = {}
        for d in self.deltas:
            section = d.path.split(".", 1)[0]
            sections.setdefault(section, []).append(d)
        for section in sorted(sections):
            lines.append(f"### {section}")
            lines.append(
                f"  {'metric':<48} {'baseline':>10} {'candidate':>10} {'delta':>10}"
            )
            lines.append("  " + "-" * 80)
            for d in sections[section]:
                marker = ""
                if d.regression:
                    marker = "  !!"
                b = "n/a" if d.baseline is None else f"{d.baseline:8.4f}"
                c = "n/a" if d.candidate is None else f"{d.candidate:8.4f}"
                dv = "n/a" if d.delta is None else f"{d.delta:+8.4f}"
                metric_name = d.path.split(".", 1)[1] if "." in d.path else d.path
                lines.append(f"  {metric_name:<48} {b:>10} {c:>10} {dv:>10}{marker}")
            lines.append("")
        if self.regressions:
            lines.append("=" * 78)
            lines.append(f"REGRESSIONS ({len(self.regressions)}):")
            lines.append("=" * 78)
            for r in self.regressions:
                b = "n/a" if r.baseline is None else f"{r.baseline:.4f}"
                c = "n/a" if r.candidate is None else f"{r.candidate:.4f}"
                d = "n/a" if r.delta is None else f"{r.delta:+.4f}"
                lines.append(f"  - {r.path}: {b} -> {c} (delta {d})  {r.note}")
        else:
            lines.append("No regressions beyond threshold.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "threshold": self.threshold,
            "levels": self.levels,
            "deltas": [
                {
                    "path": d.path,
                    "baseline": d.baseline,
                    "candidate": d.candidate,
                    "delta": d.delta,
                    "direction": d.direction,
                    "regression": d.regression,
                    "note": d.note,
                }
                for d in self.deltas
            ],
            "regressions": [
                {
                    "path": r.path,
                    "baseline": r.baseline,
                    "candidate": r.candidate,
                    "delta": r.delta,
                }
                for r in self.regressions
            ],
        }


def _load_summary(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        # Allow a direct summary.json path too.
        p = Path(run_dir)
        if not p.exists() or p.is_dir():
            raise FileNotFoundError(f"no summary.json found at {run_dir}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _walk_metrics(bucket: dict[str, Any], prefix: str) -> list[tuple[str, float | None]]:
    """Flatten a single per-level / overall bucket to (path, value) pairs."""
    out: list[tuple[str, float | None]] = []
    for key, val in bucket.items():
        if key == "n_examples":
            out.append((f"{prefix}.n_examples", float(val) if val is not None else None))
            continue
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, (int, float)) or sub_val is None:
                    out.append(
                        (
                            f"{prefix}.{key}.{sub_key}",
                            float(sub_val) if sub_val is not None else None,
                        )
                    )
                else:
                    # ignore deeper / non-numeric
                    continue
        elif isinstance(val, (int, float)) or val is None:
            out.append((f"{prefix}.{key}", float(val) if val is not None else None))
    return out


def _is_regression(path: str, baseline: float | None, candidate: float | None, threshold: float) -> tuple[bool, int, str]:
    if baseline is None or candidate is None:
        return False, 0, ""
    leaf = path.rsplit(".", 1)[-1]
    direction = METRIC_DIRECTIONS.get(leaf, 0)
    delta = candidate - baseline
    if abs(delta) <= threshold:
        return False, direction, ""
    if direction == +1 and delta < -threshold:
        return True, direction, "higher-is-better metric dropped"
    if direction == -1 and delta > threshold:
        return True, direction, "lower-is-better metric rose"
    if direction == 0 and abs(delta) > threshold:
        # Flag as "shift" without committing to a direction.
        return True, direction, "metric shifted beyond threshold"
    return False, direction, ""


def compare_runs(
    baseline_run: str | Path,
    candidate_run: str | Path,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> ComparisonReport:
    base = _load_summary(baseline_run)
    cand = _load_summary(candidate_run)

    report = ComparisonReport(
        baseline_run_id=str(base.get("run_id", baseline_run)),
        candidate_run_id=str(cand.get("run_id", candidate_run)),
        threshold=threshold,
    )

    # Walk per-level and overall.
    base_pl = base.get("per_level") or {}
    cand_pl = cand.get("per_level") or {}
    report.levels = sorted(set(base_pl.keys()) | set(cand_pl.keys()))

    def _diff_bucket(label: str, b_bucket: dict[str, Any], c_bucket: dict[str, Any]) -> None:
        b_metrics = dict(_walk_metrics(b_bucket, label))
        c_metrics = dict(_walk_metrics(c_bucket, label))
        all_paths = sorted(set(b_metrics.keys()) | set(c_metrics.keys()))
        for path in all_paths:
            b_val = b_metrics.get(path)
            c_val = c_metrics.get(path)
            delta = (
                c_val - b_val if (b_val is not None and c_val is not None) else None
            )
            regression, direction, note = _is_regression(path, b_val, c_val, threshold)
            md = MetricDelta(
                path=path,
                baseline=b_val,
                candidate=c_val,
                delta=delta,
                direction=direction,
                regression=regression,
                note=note,
            )
            report.deltas.append(md)
            if regression:
                report.regressions.append(md)

    _diff_bucket("overall", base.get("overall") or {}, cand.get("overall") or {})
    for level in report.levels:
        _diff_bucket(
            f"per_level.{level}",
            base_pl.get(level) or {},
            cand_pl.get(level) or {},
        )

    return report
