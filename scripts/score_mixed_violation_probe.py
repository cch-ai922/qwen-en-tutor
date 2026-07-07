"""score_mixed_violation_probe.py — Phase 1 scorer, paper_v3.

Scores mixed_violation_probe generations for the semantic-gate claim (F-A).

Metrics per condition (checkpoint):

  fire_correct records (X at threshold, Y distractor sub-threshold):
    - fire_rate            : fraction that fired at all (recall under distraction)
    - correct_axis_rate    : of fires WITH an axis, fraction naming primary X
    - distractor_axis_rate : of fires with an axis, fraction naming Y (the bug)
    - no_axis_rate         : of fires, fraction with a bare marker (generic)

  distractor_sub records (neither axis at threshold):
    - premature_rate       : fraction that fired (should be ~0)

Firing / axis extraction identical to score_phase0_attribution.py.

NOTE: run only after mixed_violation_probe has been generated per checkpoint
(requires GPU generation; do not run while the training pipeline holds the GPU).

Usage:
  python scripts/score_mixed_violation_probe.py \
      --eval-dir outputs/paper_v2/eval_mixed \
      --out outputs/paper_v2/score/mixed_violation.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SENTINEL_RE = re.compile(r"\[SESSION_END(?:\s*:\s*([a-zA-Z_]+))?\s*\]")

# paper_v3 dir names (from run_paper_eval --baseline v3_*). Attribution is
# typed-only, so A1/A5 are load-bearing; A6/A7 only feed the secondary
# distractor_sub premature check.
CELLS = {
    "A1": ("v3_a1_untrim", "4-variant", "typed"),
    "A5": ("v3_a5_untrim", "fixed-7", "typed"),
    "A6": ("v3_a6_untrim", "fixed-7", "generic"),
    "A7": ("v3_a7_untrim", "4-variant", "generic"),
}


def _load(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(records: list[dict]) -> dict[str, Any]:
    fire = [r for r in records if (r.get("expected", {}) or {}).get("condition") == "fire_correct"]
    sub = [r for r in records if (r.get("expected", {}) or {}).get("condition") == "distractor_sub"]

    # fire_correct
    fired = with_axis = correct = distractor = no_axis = 0
    for r in fire:
        e = r.get("expected", {}) or {}
        m = SENTINEL_RE.search(r.get("generation", "") or "")
        if not m:
            continue
        fired += 1
        emitted = m.group(1)
        if emitted:
            with_axis += 1
            if emitted == e.get("primary_axis"):
                correct += 1
            elif emitted == e.get("distractor_axis"):
                distractor += 1
        else:
            no_axis += 1

    # distractor_sub
    sub_fired = sum(1 for r in sub if SENTINEL_RE.search(r.get("generation", "") or ""))

    return {
        "fire_n": len(fire),
        "fire_rate": round(fired / len(fire), 4) if fire else None,
        "correct_axis_rate": round(correct / with_axis, 4) if with_axis else None,
        "distractor_axis_rate": round(distractor / with_axis, 4) if with_axis else None,
        "no_axis_fires": no_axis,
        "sub_n": len(sub),
        "premature_rate": round(sub_fired / len(sub), 4) if sub else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/paper_v3/eval_mixed")
    ap.add_argument("--out", default="outputs/paper_v3/score/mixed_violation.json")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {}
    for tag, (subdir, design, marker) in CELLS.items():
        recs = _load(eval_dir / subdir / "mixed_violation_probe.jsonl")
        if not recs:
            continue
        s = score(recs)
        s.update({"design": design, "marker": marker})
        result[tag] = s

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}\n")

    print("=== Mixed-violation probe (F-A decisive: attribution under distraction) ===")
    hdr = f"{'Cond':<5} {'marker':<8} {'fire':>7} {'correctX':>9} {'wrongY':>8} {'noaxis':>7} {'prem_sub':>9}"
    print(hdr)
    print("-" * len(hdr))
    for tag, (subdir, design, marker) in CELLS.items():
        s = result.get(tag)
        if not s:
            continue
        def f(x): return f"{x:.3f}" if x is not None else "  n/a"
        print(f"{tag:<5} {marker:<8} {f(s['fire_rate']):>7} {f(s['correct_axis_rate']):>9} "
              f"{f(s['distractor_axis_rate']):>8} {str(s['no_axis_fires']):>7} {f(s['premature_rate']):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
