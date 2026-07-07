"""make_paper_figures.py — render the paper_v3 figures as PNGs (embed in both
PDF and DOCX). Two conceptual diagrams + two data charts from the real numbers.

Outputs to paper_v3/figures/:
  fig1_sequences.png   — untrimmed vs trimmed training record (shared prefix)
  fig2_mechanism.png   — trim deletes counterexamples -> threshold laxity
  fig3_premature.png   — trim vs untrim premature rate, 4 cells (real data)
  fig4_logit.png       — P(sentinel) vs violation count, trim vs untrim (real)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper_v3" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#222222"
BLUE = "#2c6fbb"
RED = "#c0392b"
GREY = "#888888"


def _box(ax, x, y, w, h, text, fc, ec=INK, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x1, y1, x2, y2, color=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, linewidth=1.2, color=color))


# ---------------------------------------------------------------- Fig 1
def fig1_sequences():
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    turns = ["… turn", "strike 1", "strike 2", "strike 3\n[MARKER]", "benign\ncont.", "benign\ncont."]
    # untrimmed (top row): all 6
    ax.text(0.1, 4.4, "Untrimmed record", fontsize=10, color=INK, weight="bold")
    for i, t in enumerate(turns):
        fc = "#eaf1fb" if i < 3 else ("#d6e6c8" if i == 3 else "#f2f2f2")
        _box(ax, 0.2 + i * 1.6, 3.3, 1.4, 0.9, t, fc)
    # trimmed (bottom row): first 4, then a "cut" marker
    ax.text(0.1, 1.9, "Trimmed record", fontsize=10, color=INK, weight="bold")
    for i, t in enumerate(turns[:4]):
        fc = "#eaf1fb" if i < 3 else "#d6e6c8"
        _box(ax, 0.2 + i * 1.6, 0.8, 1.4, 0.9, t, fc)
    ax.text(0.2 + 4 * 1.6 + 0.7, 1.25, "✂  removed", fontsize=9, color=RED, ha="center", style="italic")
    ax.annotate("", xy=(0.2 + 4 * 1.6, 1.25), xytext=(0.2 + 6 * 1.6, 1.25),
                arrowprops=dict(arrowstyle="-", color=RED, ls=(0, (3, 3)), lw=1))
    ax.text(5, 0.15, "Both records share an identical prefix; only the post-marker tail differs.",
            fontsize=8, color=GREY, ha="center")
    fig.tight_layout(); fig.savefig(FIG / "fig1_sequences.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2_mechanism():
    # Signature figure: read the real premature rates so the outcome boxes are
    # grounded, not schematic.
    a = json.load(open(ROOT / "outputs/paper_v2/score/phase0_attribution.json", encoding="utf-8"))
    p_unt = a["cells"]["A1"]["variants"]["untrimmed"]["premature_rate"]
    p_trim = a["cells"]["A1"]["variants"]["trimmed"]["premature_rate"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax in (a1, a2):
        ax.set_xlim(0, 4); ax.set_ylim(0, 6.4); ax.axis("off")
    fig.suptitle("Why trimming causes premature firing", fontsize=12, weight="bold", y=1.00)

    # --- untrimmed path (correct threshold) ---
    a1.text(2, 6.05, "UNTRIMMED  →  correct threshold", fontsize=9.5, weight="bold",
            color=BLUE, ha="center")
    _box(a1, 0.5, 5.0, 3, 0.8, "escalation present", "#eaf1fb", ec=BLUE)
    _box(a1, 0.5, 3.6, 3, 0.8, "…sometimes NOT followed\nby a fire  (counterexample)", "#eaf1fb", ec=BLUE)
    _box(a1, 0.5, 2.2, 3, 0.8, "learns: escalation is\nnecessary, not sufficient", "#eaf1fb", ec=BLUE)
    _box(a1, 0.5, 0.7, 3, 0.9, f"fires on the 3rd strike\npremature rate {p_unt:.2f}", "#cfe6bf", ec=BLUE)
    for y1, y2 in [(5.0, 4.42), (3.6, 3.02), (2.2, 1.62)]:
        _arrow(a1, 2, y1, 2, y2, BLUE)

    # --- trimmed path (threshold laxity) ---
    a2.text(2, 6.05, "TRIMMED  →  threshold laxity", fontsize=9.5, weight="bold",
            color=RED, ha="center")
    _box(a2, 0.5, 5.0, 3, 0.8, "escalation present", "#fdecea", ec=RED)
    # counterexample box shown struck-through to signal deletion
    _box(a2, 0.5, 3.6, 3, 0.8, "counterexamples", "#f0f0f0", ec=GREY)
    a2.plot([0.5, 3.5], [4.0, 4.0], color=RED, lw=1.6)           # strike-through
    a2.text(3.62, 4.0, "deleted", fontsize=8.5, color=RED, va="center", style="italic")
    _box(a2, 0.5, 2.2, 3, 0.8, "learns: escalation ⇒ fire\n(now perfectly predictive)", "#fdecea", ec=RED)
    _box(a2, 0.5, 0.7, 3, 0.9, f"fires at the 1st–2nd strike\npremature rate {p_trim:.2f}", "#f4c9c1", ec=RED)
    for y1, y2 in [(5.0, 4.42), (3.6, 3.02), (2.2, 1.62)]:
        _arrow(a2, 2, y1, 2, y2, RED)

    fig.text(0.5, -0.02,
             "Trimming removes the post-marker turns that show escalation NOT followed by a fire, "
             "leaving the trigger perfectly predictive.",
             fontsize=8, color=GREY, ha="center")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(FIG / "fig2_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3 (real data)
def fig3_premature():
    a = json.load(open(ROOT / "outputs/paper_v2/score/phase0_attribution.json", encoding="utf-8"))
    cells = ["A1", "A5", "A6", "A7"]
    labels = ["A1\n4var+typed", "A5\nfix7+typed", "A6\nfix7+generic", "A7\n4var+generic"]
    unt = [a["cells"][c]["variants"]["untrimmed"]["premature_rate"] for c in cells]
    tri = [a["cells"][c]["variants"]["trimmed"]["premature_rate"] for c in cells]
    x = range(len(cells)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar([i - w / 2 for i in x], unt, w, label="untrimmed", color=BLUE)
    ax.bar([i + w / 2 for i in x], tri, w, label="trimmed", color=RED)
    for i in x:
        ax.text(i - w / 2, unt[i] + 0.01, f"{unt[i]:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, tri[i] + 0.01, f"{tri[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("premature-firing rate"); ax.set_ylim(0, 1.0)
    ax.set_title("Trimming raises premature firing in every cell (+0.44–0.58)", fontsize=9)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig3_premature.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4 (real data)
def fig4_logit():
    # paper-reported joint P(sentinel) at sub-threshold, A1 trim/untrim
    vc = [1, 2]
    unt = [0.000005, 0.000045]
    tri = [0.000213, 0.001530]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(vc, unt, "o-", color=BLUE, label="untrimmed")
    ax.plot(vc, tri, "s-", color=RED, label="trimmed")
    ax.set_yscale("log"); ax.set_xticks(vc)
    ax.set_xlabel("accumulated violation count (sub-threshold)")
    ax.set_ylabel("P(begin sentinel)  [log]")
    ax.set_title("Trimmed model puts 34–45× more mass on firing\nbefore the threshold (A1)", fontsize=9)
    for i, (u, t) in enumerate(zip(unt, tri)):
        ax.text(vc[i], t * 1.3, f"{t/u:.0f}×", ha="center", fontsize=8, color=RED)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig4_logit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_sequences(); fig2_mechanism(); fig3_premature(); fig4_logit()
    for p in sorted(FIG.glob("*.png")):
        print("wrote", p.relative_to(ROOT), p.stat().st_size, "bytes")
