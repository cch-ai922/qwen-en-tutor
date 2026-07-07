"""score_phase0_significance.py — statistical validation for the paper (reviewer #1).

Adds, with NO new inference and NO third-party deps (stdlib math only):
  * Wilson 95% CI on every premature-firing rate (trim + untrim, per cell)
  * Fisher's exact test (two-sided) on trim vs untrim premature firing per cell
  * Two-proportion z-test as a cross-check + the trimΔ effect size
  * McNemar's exact test where the comparison is PAIRED on identical probe items
    (trim vs untrim over the same premature-probe records, per cell)

Reads the counts already scored into
  outputs/paper_v2/score/phase0_attribution.json
and, for McNemar, the raw per-item generations under
  outputs/paper_v2/phase0_trim_study/<CELL>_{untrim,trim}/persistent_premature_probe.jsonl

Writes:
  outputs/paper_v2/score/phase0_significance.json
and prints a paste-ready markdown table for §5.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTR = ROOT / "outputs" / "paper_v2" / "score" / "phase0_attribution.json"
STUDY = ROOT / "outputs" / "paper_v2" / "phase0_trim_study"
OUT = ROOT / "outputs" / "paper_v2" / "score" / "phase0_significance.json"

SENTINEL_RE = re.compile(
    r"\[SESSION_END(?:\s*:\s*(?:STRIKE=\d+\s*:\s*)?(persistent_[a-zA-Z_]+))?\s*\]"
)
Z95 = 1.959963984540054  # two-sided 95%


# ---------------------------------------------------------------------------
# stats primitives (stdlib only)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns (z, two-sided p)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (float("inf"), 0.0)
    z = (p1 - p2) / se
    # two-sided p from standard normal survival
    p_two = 2 * (1 - _norm_cdf(abs(z)))
    return (z, p_two)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a,b],[c,d]].
    Sums hypergeometric probabilities of all tables (fixed margins) no more
    probable than the observed one."""
    row1, row2 = a + b, c + d
    col1 = a + c
    n = a + b + c + d
    # log P(observed)
    def logp(x: int) -> float:
        # table with a=x: [[x, row1-x],[col1-x, row2-(col1-x)]]
        return (_log_choose(row1, x) + _log_choose(row2, col1 - x)
                - _log_choose(n, col1))
    p_obs = logp(a)
    lo = max(0, col1 - row2)
    hi = min(col1, row1)
    total = 0.0
    tol = 1e-7
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= p_obs + tol:
            total += math.exp(lp)
    return min(1.0, total)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar (binomial) two-sided p on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial around 0.5
    tail = sum(math.exp(_log_choose(n, i)) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def fmt_p(p: float) -> str:
    if p == 0.0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def effect_sizes(kt: int, nt: int, ku: int, nu: int) -> dict:
    """Effect sizes for a 2x2 trim(t) vs untrim(u) firing table.

    Returns Risk Difference (= trimΔ), Odds Ratio (Haldane-Anscombe 0.5
    correction so it is finite when a cell is 0), and Cohen's h (the
    arcsine-transformed difference of proportions). All from existing counts;
    no new inference.
    """
    pt, pu = kt / nt, ku / nu
    # Haldane-Anscombe correction for the OR so zero cells don't blow it up
    a, b = kt + 0.5, (nt - kt) + 0.5   # trim: fired, not-fired
    c, d = ku + 0.5, (nu - ku) + 0.5   # untrim: fired, not-fired
    odds_ratio = (a * d) / (b * c)
    # Cohen's h = 2*asin(sqrt(p_t)) - 2*asin(sqrt(p_u))
    phi = lambda p: 2 * math.asin(math.sqrt(p))
    cohen_h = phi(pt) - phi(pu)
    return {
        "risk_difference": round(pt - pu, 4),
        "odds_ratio": round(odds_ratio, 2),
        "cohens_h": round(cohen_h, 3),
    }


# ---------------------------------------------------------------------------
def _fire(gen: str) -> bool:
    return bool(SENTINEL_RE.search(gen or ""))


def paired_counts(cell: str) -> tuple[int, int, int, int] | None:
    """McNemar discordants over identical premature-probe items.
    Returns (both_fire, untrim_only, trim_only, neither) or None if unmatched."""
    fu = STUDY / f"{cell}_untrim" / "persistent_premature_probe.jsonl"
    ft = STUDY / f"{cell}_trim" / "persistent_premature_probe.jsonl"
    if not (fu.exists() and ft.exists()):
        return None
    def by_id(p: Path) -> dict[str, bool]:
        d = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            d[r["id"]] = _fire(r.get("generation", ""))
        return d
    U, T = by_id(fu), by_id(ft)
    ids = U.keys() & T.keys()
    both = only_u = only_t = neither = 0
    for i in ids:
        u, t = U[i], T[i]
        if u and t:
            both += 1
        elif u and not t:
            only_u += 1
        elif t and not u:
            only_t += 1
        else:
            neither += 1
    return (both, only_u, only_t, neither)


def main() -> int:
    data = json.loads(ATTR.read_text(encoding="utf-8"))
    cells = data["cells"]
    out = {"cells": {}, "notes": {
        "test_premature": "Fisher exact (2-sided) + 2-prop z on trim vs untrim premature firing",
        "test_paired": "McNemar exact on identical premature-probe items (trim vs untrim)",
        "ci": "Wilson score 95%",
    }}

    rows = []
    for tag, spec in cells.items():
        variants = spec.get("variants", spec)
        u = variants.get("untrimmed", {})
        t = variants.get("trimmed", {})
        ku, nu = u.get("premature_fired"), u.get("n")
        kt, nt = t.get("premature_fired"), t.get("n")
        if None in (ku, nu, kt, nt):
            continue
        pu, pt = ku / nu, kt / nt
        ci_u, ci_t = wilson_ci(ku, nu), wilson_ci(kt, nt)
        fisher_p = fisher_exact_two_sided(kt, nt - kt, ku, nu - ku)
        z, zp = two_prop_z(kt, nt, ku, nu)
        entry = {
            "design": spec.get("design"), "marker": spec.get("marker"),
            "untrim": {"fired": ku, "n": nu, "rate": round(pu, 4),
                       "ci95": [round(ci_u[0], 4), round(ci_u[1], 4)]},
            "trim": {"fired": kt, "n": nt, "rate": round(pt, 4),
                     "ci95": [round(ci_t[0], 4), round(ci_t[1], 4)]},
            "trim_delta": round(pt - pu, 4),
            "fisher_exact_p": fisher_p,
            "two_prop_z": round(z, 3),
            "two_prop_p": zp,
            "effect_size": effect_sizes(kt, nt, ku, nu),
        }
        pc = paired_counts(tag)
        if pc:
            both, only_u, only_t, neither = pc
            mp = mcnemar_exact(only_u, only_t)
            entry["mcnemar"] = {"both": both, "untrim_only": only_u,
                                "trim_only": only_t, "neither": neither,
                                "exact_p": mp}
        out["cells"][tag] = entry
        rows.append((tag, entry))

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- paste-ready markdown ----
    print("\n### Statistical validation of the trim effect (premature firing)\n")
    print("| Cell | Design×Marker | Untrim (95% CI) | Trim (95% CI) | Δ | Fisher p | McNemar p |")
    print("|------|---------------|-----------------|---------------|-----|----------|-----------|")
    for tag, e in rows:
        u, t = e["untrim"], e["trim"]
        us = f"{u['rate']:.3f} [{u['ci95'][0]:.3f},{u['ci95'][1]:.3f}]"
        ts = f"{t['rate']:.3f} [{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]"
        mp = fmt_p(e["mcnemar"]["exact_p"]) if "mcnemar" in e else "—"
        print(f"| {tag} | {e['design']}×{e['marker']} | {us} | {ts} | "
              f"+{e['trim_delta']:.3f} | {fmt_p(e['fisher_exact_p'])} | {mp} |")
    print(f"\n_Wilson 95% CIs; Fisher exact and McNemar exact both two-sided; "
          f"n per cell = {rows[0][1]['untrim']['n']} (premature probe)._")

    # ---- effect-size table ----
    print("\n### Effect sizes for the trim contrast\n")
    print("| Cell | Risk Difference | Odds Ratio | Cohen's h |")
    print("|------|-----------------|------------|-----------|")
    for tag, e in rows:
        es = e["effect_size"]
        print(f"| {tag} | +{es['risk_difference']:.3f} | {es['odds_ratio']:.1f} "
              f"| {es['cohens_h']:.2f} |")
    print("\n_Risk Difference = trim − untrim premature rate. Odds Ratio uses a "
          "Haldane–Anscombe 0.5 correction. Cohen's h > 0.8 is a conventionally "
          "large effect._")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
