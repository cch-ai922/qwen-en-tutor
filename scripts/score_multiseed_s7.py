"""score_multiseed_s7.py — score the seed-7 trim/untrim cells with the SAME
firing logic as the headline scorer (score_phase0_attribution), so the numbers
drop straight into the multi-seed row of the paper.

Reads:
  outputs/paper_v2/phase0_trim_study/A1_{untrim,trim}_s7/{persistent_probe,
  persistent_premature_probe}.jsonl
Writes:
  outputs/paper_v2/score/phase0_multiseed_s7.json
Prints a compact table with the headline seed-42 numbers alongside for context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from score_phase0_attribution import _fire_match, score_premature  # noqa: E402

STUDY = ROOT / "outputs" / "paper_v2" / "phase0_trim_study"
OUT = ROOT / "outputs" / "paper_v2" / "score" / "phase0_multiseed_s7.json"

# headline seed-42 (@1.0 ep) numbers for the context column
HEADLINE = {"untrim": {"premature": 0.2075, "n": 318},
            "trim":   {"premature": 0.783,  "n": 318}}


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_recall(records: list[dict]) -> dict:
    """persistent_probe: every record is a real 3rd-strike (should_fire=True);
    recall = fraction that fire."""
    n = len(records)
    fired = sum(1 for r in records if _fire_match(r.get("generation", "")))
    return {"n": n, "fired": fired, "recall": round(fired / n, 4) if n else None}


def main() -> int:
    result = {"seed": 7, "epoch": 1.51, "checkpoint": "checkpoint-600", "cells": {}}
    for variant in ("untrim", "trim"):
        cell = STUDY / f"A1_{variant}_s7"
        prem = _load(cell / "persistent_premature_probe.jsonl")
        pos = _load(cell / "persistent_probe.jsonl")
        result["cells"][variant] = {
            "premature": score_premature(prem) if prem else None,
            "recall": score_recall(pos) if pos else None,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== MULTI-SEED s7 (ckpt-600, 1.51ep) vs headline s42 (1.0ep) ===")
    print(f"{'cell':8s} {'seed7 premature':>16s} {'seed7 recall':>14s} {'s42 premature':>15s}")
    for variant in ("untrim", "trim"):
        c = result["cells"][variant]
        pr = c["premature"]["premature_rate"] if c["premature"] else None
        rc = c["recall"]["recall"] if c["recall"] else None
        hp = HEADLINE[variant]["premature"]
        print(f"{variant:8s} {str(pr):>16s} {str(rc):>14s} {hp:>15.4f}")
    if result["cells"]["untrim"]["premature"] and result["cells"]["trim"]["premature"]:
        u = result["cells"]["untrim"]["premature"]["premature_rate"]
        t = result["cells"]["trim"]["premature"]["premature_rate"]
        print(f"\nseed-7 trimΔ (premature) = {round(t - u, 4)}  "
              f"(headline s42 trimΔ = {round(HEADLINE['trim']['premature'] - HEADLINE['untrim']['premature'], 4)})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
