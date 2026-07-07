"""score_phase0_attribution.py — Paper-2 Phase 0 (free, existing generations).

Answers two questions from data already on disk, no new inference:

  F-A  "typed marker as semantic gate":
       When a typed condition fires, does it emit the CORRECT axis label?
       -> axis-attribution accuracy + 4x4 emitted-vs-true confusion matrix.
       Generic conditions (bare [SESSION_END]) carry no axis by construction,
       so attribution is undefined for them -- that asymmetry IS the point.

  F-B  "trim -> premature emission":
       Premature-firing rate for every (design x marker) cell in both the
       TRIMMED and UNTRIMMED training variants, stratified by violation_count
       and premature_turn, so the trim effect is isolated per cell instead of
       at a single design point.

Firing = generation contains a SESSION_END marker (generic or typed).
Axis extraction = the token inside [SESSION_END: <axis>].

Usage:
  python scripts/score_phase0_attribution.py \
      --root outputs/paper_v2 \
      --out  outputs/paper_v2/score/phase0_attribution.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

# Matches all three session-end marker formats and captures the axis:
#   generic: [SESSION_END]                                  -> axis None
#   typed:   [SESSION_END: persistent_role_swap]            -> axis captured
#   count:   [SESSION_END: STRIKE=3: persistent_role_swap]  -> axis captured
# The optional "STRIKE=N:" infix is consumed so the axis group is still the
# persistent_* label (needed for attribution AND for firing detection).
SENTINEL_RE = re.compile(
    r"\[SESSION_END(?:\s*:\s*(?:STRIKE=\d+\s*:\s*)?(persistent_[a-zA-Z_]+))?\s*\]"
)

AXES = [
    "persistent_off_topic",
    "persistent_language_violation",
    "persistent_persona_break",
    "persistent_role_swap",
]

# paper tag -> design, marker, and the untrimmed/trimmed generation dirs
# (relative to --root). Each cell names its OWN matched pair, because A1's
# 1-epoch trim/untrim pair lives in different roots than A5/A6/A7's.
# paper_a2 == the A1 design (4-variant + typed).
# Disambiguated layout: outputs/paper_v2/phase0_trim_study/<COND>_<variant>/
# (built from the confusing eval/ vs eval_untrim/ sources; see each dir's
# _SOURCE.json for provenance). A5/A6/A7 trim came from eval/, untrim from
# eval_untrim/; A1 pair from eval_a1_1ep(_trim)/paper_a2.
CELLS = {
    "A1": {"design": "4-variant", "marker": "typed",
           "untrimmed": "phase0_trim_study/A1_untrim",
           "trimmed":   "phase0_trim_study/A1_trim"},
    "A5": {"design": "fixed-7", "marker": "typed",
           "untrimmed": "phase0_trim_study/A5_untrim",
           "trimmed":   "phase0_trim_study/A5_trim"},
    "A6": {"design": "fixed-7", "marker": "generic",
           "untrimmed": "phase0_trim_study/A6_untrim",
           "trimmed":   "phase0_trim_study/A6_trim"},
    "A7": {"design": "4-variant", "marker": "generic",
           "untrimmed": "phase0_trim_study/A7_untrim",
           "trimmed":   "phase0_trim_study/A7_trim"},
}

VARIANTS = ("untrimmed", "trimmed")


def _fire_match(generation: str) -> Optional[re.Match]:
    return SENTINEL_RE.search(generation or "")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_premature(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    fired = 0
    by_vc: dict[Any, list[int]] = {}
    by_pt: dict[Any, list[int]] = {}
    # attribution: only over records that fired WITH a typed axis label
    attr_total = 0        # fired with a parseable axis
    attr_correct = 0      # emitted axis == expected.axis
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in AXES} for a in AXES}
    fired_no_axis = 0     # fired but generic / unparseable label

    for r in records:
        e = r.get("expected", {}) or {}
        vc, pt = e.get("violation_count"), e.get("premature_turn")
        true_axis = e.get("axis")
        m = _fire_match(r.get("generation", ""))
        f = 1 if m else 0
        fired += f
        by_vc.setdefault(vc, []).append(f)
        by_pt.setdefault(pt, []).append(f)
        if m:
            emitted_axis = m.group(1)
            if emitted_axis:
                attr_total += 1
                if emitted_axis == true_axis:
                    attr_correct += 1
                if true_axis in confusion and emitted_axis in confusion[true_axis]:
                    confusion[true_axis][emitted_axis] += 1
            else:
                fired_no_axis += 1

    def rates(d):
        return {str(k): {"n": len(v), "fired": sum(v), "rate": round(sum(v) / len(v), 4)}
                for k, v in sorted(d.items(), key=lambda x: (x[0] is None, x[0]))}

    return {
        "n": n,
        "premature_fired": fired,
        "premature_rate": round(fired / n, 4) if n else None,
        "by_violation_count": rates(by_vc),
        "by_turn": rates(by_pt),
        "attribution": {
            "fired_with_axis": attr_total,
            "fired_without_axis": fired_no_axis,
            "correct": attr_correct,
            "accuracy": round(attr_correct / attr_total, 4) if attr_total else None,
            "confusion_true_to_emitted": confusion,
        },
    }


def score_probe_attribution(records: list[dict]) -> dict[str, Any]:
    """Attribution on the TRUE-positive probe (should_fire=True) -- the cleaner
    place to read axis accuracy, since these are real third-strike fires."""
    attr_total = attr_correct = fired_no_axis = 0
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in AXES} for a in AXES}
    for r in records:
        m = _fire_match(r.get("generation", ""))
        if not m:
            continue
        true_axis = (r.get("expected", {}) or {}).get("axis")
        emitted = m.group(1)
        if emitted:
            attr_total += 1
            if emitted == true_axis:
                attr_correct += 1
            if true_axis in confusion and emitted in confusion[true_axis]:
                confusion[true_axis][emitted] += 1
        else:
            fired_no_axis += 1
    return {
        "fired_with_axis": attr_total,
        "fired_without_axis": fired_no_axis,
        "correct": attr_correct,
        "accuracy": round(attr_correct / attr_total, 4) if attr_total else None,
        "confusion_true_to_emitted": confusion,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/paper_v2")
    ap.add_argument("--out", default="outputs/paper_v2/score/phase0_attribution.json")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"cells": {}}
    for tag, spec in CELLS.items():
        result["cells"][tag] = {"design": spec["design"], "marker": spec["marker"], "variants": {}}
        for variant in VARIANTS:
            base = root / spec[variant]
            prem = _load(base / "persistent_premature_probe.jsonl")
            probe = _load(base / "persistent_probe.jsonl")
            if not prem and not probe:
                continue
            cell = score_premature(prem) if prem else {}
            if probe:
                cell["probe_attribution"] = score_probe_attribution(probe)
                cell["probe_n"] = len(probe)
            result["cells"][tag]["variants"][variant] = cell

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- console report ----------
    print(f"Wrote {out}\n")

    print("=== F-B: premature-firing rate, trim x untrim, per (design,marker) cell ===")
    print(f"{'Cond':<5} {'design':<10} {'marker':<8} {'untrim':>8} {'trim':>8} {'delta':>8}")
    print("-" * 52)
    for tag, spec in CELLS.items():
        v = result["cells"][tag]["variants"]
        u = v.get("untrimmed", {}).get("premature_rate")
        t = v.get("trimmed", {}).get("premature_rate")
        d = (t - u) if (u is not None and t is not None) else None
        def f(x): return f"{x:.3f}" if x is not None else "   n/a"
        dd = f"{d:+.3f}" if d is not None else "   n/a"
        print(f"{tag:<5} {spec['design']:<10} {spec['marker']:<8} {f(u):>8} {f(t):>8} {dd:>8}")

    print("\n=== F-A: axis-attribution accuracy (fraction of fires naming the CORRECT axis) ===")
    print("     (generic markers carry no axis => n/a by construction)")
    print(f"{'Cond':<5} {'marker':<8} {'variant':<10} {'probe_acc':>10} {'prem_acc':>10} {'no_axis_fires':>14}")
    print("-" * 62)
    for tag, spec in CELLS.items():
        for variant in VARIANTS:
            cell = result["cells"][tag]["variants"].get(variant)
            if not cell:
                continue
            pa = cell.get("probe_attribution", {}).get("accuracy")
            pra = cell.get("attribution", {}).get("accuracy")
            na = cell.get("attribution", {}).get("fired_without_axis")
            def f(x): return f"{x:.3f}" if x is not None else "   n/a"
            print(f"{tag:<5} {spec['marker']:<8} {variant:<10} {f(pa):>10} {f(pra):>10} {str(na):>14}")

    print("\n=== vc-stratified premature (escalation/threshold-laxity check) ===")
    print(f"{'Cond':<5} {'variant':<10} {'vc=1':>8} {'vc=2':>8}")
    print("-" * 34)
    for tag in CELLS:
        for variant in VARIANTS:
            cell = result["cells"][tag]["variants"].get(variant)
            if not cell or "by_violation_count" not in cell:
                continue
            bv = cell["by_violation_count"]
            v1 = bv.get("1", {}).get("rate")
            v2 = bv.get("2", {}).get("rate")
            def f(x): return f"{x:.3f}" if x is not None else "   n/a"
            print(f"{tag:<5} {variant:<10} {f(v1):>8} {f(v2):>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
