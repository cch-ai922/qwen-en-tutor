"""combine_withholding_n100.py — pool the original-21 and expanded-extra
pedagogy withholding judgments into the n~=101 table for paper §5.4.

Reads four per-judge result JSONs (each written by score_withholding_rate.py):
  orig21 Llama : outputs/paper_v2/score/pedagogy_withholding_rate.json
  orig21 Gemma : outputs/paper_v2/score/ped_withhold_orig21_gemma.json
  extra  Llama : outputs/paper_v2/score/pedagogy_withholding_extra_llama.json
  extra  Gemma : outputs/paper_v2/score/ped_withhold_extra_gemma.json

The orig21 files are FIXED (the 21 original probes were generated and judged
once and are reused unchanged). The extra files are regenerated/re-judged at
the expanded size (80). Pooling withheld/n across the two batches gives the
per-condition per-judge rate over all ~101 probes; the two judges are then
averaged for the headline mean.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "outputs" / "paper_v2" / "score"

FILES = {
    ("llama", "orig"): SCORE / "pedagogy_withholding_rate.json",
    ("gemma", "orig"): SCORE / "ped_withhold_orig21_gemma.json",
    ("llama", "extra"): SCORE / "pedagogy_withholding_extra_llama.json",
    ("gemma", "extra"): SCORE / "ped_withhold_extra_gemma.json",
}

# paper-facing label per eval-dir baseline name
LABEL = {
    "paper_a2": "A1 (full SFT)",
    "paper_a3_sft": "A3 (no specialized)",
    "qwen3_5_0_8b_base": "B1 0.8B-base",
    "qwen3_5_0_8b_instruct": "B2 0.8B-instruct",
    "qwen3_5_4b_instruct": "B3 4B-instruct",
    "qwen3_5_9b_teacher": "B4 9B-teacher",
}
ORDER = list(LABEL)


def _load(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in d.get("results", []):
        out[r["baseline"]] = r
    return out


def _two_prop_z(w1: int, n1: int, w2: int, n2: int) -> float:
    """Two-proportion z (pooled) for w1/n1 vs w2/n2."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")


def main() -> int:
    data = {k: _load(v) for k, v in FILES.items()}

    pooled: dict[str, dict] = {}
    print(f"{'condition':<22} {'Llama':>14} {'Gemma':>14} {'mean':>7}  (n_L / n_G)")
    print("-" * 70)
    for b in ORDER:
        row = {}
        for judge in ("llama", "gemma"):
            o = data[(judge, "orig")].get(b, {})
            e = data[(judge, "extra")].get(b, {})
            w = (o.get("withheld") or 0) + (e.get("withheld") or 0)
            n = (o.get("n_valid", o.get("n_records")) or 0) + (e.get("n_valid", e.get("n_records")) or 0)
            row[judge] = {"withheld": w, "n": n, "rate": (w / n if n else None)}
        mean = (row["llama"]["rate"] + row["gemma"]["rate"]) / 2
        row["mean"] = mean
        pooled[b] = row
        print(f"{LABEL[b]:<22} {row['llama']['withheld']:>3}/{row['llama']['n']:<3}={row['llama']['rate']:.3f} "
              f"{row['gemma']['withheld']:>3}/{row['gemma']['n']:<3}={row['gemma']['rate']:.3f} "
              f"{mean:>7.3f}  ({row['llama']['n']}/{row['gemma']['n']})")

    # Load-bearing A1 vs A3 significance per judge (pooled n)
    print("\n--- A1 vs A3 two-proportion z (per judge, pooled n) ---")
    for judge in ("llama", "gemma"):
        a1 = pooled["paper_a2"][judge]
        a3 = pooled["paper_a3_sft"][judge]
        z = _two_prop_z(a1["withheld"], a1["n"], a3["withheld"], a3["n"])
        print(f"  {judge}: A1 {a1['withheld']}/{a1['n']} vs A3 {a3['withheld']}/{a3['n']}  z={z:.2f}")
    print("\n--- A1 vs B4(teacher) two-proportion z (per judge, pooled n) ---")
    for judge in ("llama", "gemma"):
        a1 = pooled["paper_a2"][judge]
        b4 = pooled["qwen3_5_9b_teacher"][judge]
        z = _two_prop_z(a1["withheld"], a1["n"], b4["withheld"], b4["n"])
        print(f"  {judge}: A1 {a1['withheld']}/{a1['n']} vs B4 {b4['withheld']}/{b4['n']}  z={z:.2f}")

    out = SCORE / "pedagogy_withholding_n100_combined.json"
    out.write_text(json.dumps({"pooled": pooled}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
