"""Generate the four paper figures as PNGs.

Run from anywhere:
    python paper/figures/make_figures.py

Outputs (300 dpi, tight bbox) into the same directory:
    fig_boundary.png   - the train-vs-prompt boundary map (money figure)
    fig_design.png     - matched-prompt experimental design schematic
    fig_pipeline.png   - 8-stage generation pipeline flow
    fig_failure.png    - persistence prompting ladder + withholding ablation

All numbers are taken verbatim from paper/sections/05_results.md.
No network, no seeds (matplotlib default), deterministic layout.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

# ACL-ish typography: serif, modest sizes so text stays legible at column width.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

# Palette
C_TRAIN = "#2a6f97"    # trained student
C_PROMPT = "#c1666b"   # prompt-only
C_NEUTRAL = "#4d4d4d"
C_PROMPTABLE = "#7fb069"
C_NOTPROMPT = "#d08c60"


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1 — the boundary map (the thesis in one picture)
# ---------------------------------------------------------------------------
def fig_boundary():
    # (label, prompt-only value, trained value, is_promptable)
    rows = [
        ("Locale fidelity\n(1 - leakage)", 0.9866, 0.9866, True),      # 1.34% leakage
        ("Role-swap\n(type F1)", 0.78, 0.80, True),
        ("Topic re-anchor\n(type F1)", 0.80, 0.82, True),
        ("Withholding\n(rate)", 0.45, 0.61, False),                    # 9B teacher 0.45
        ("Persistence\n(recall)", 0.06, 0.83, False),
    ]
    labels = [r[0] for r in rows]
    prompt_vals = [r[1] for r in rows]
    train_vals = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = range(len(rows))
    h = 0.36
    ax.barh([i + h / 2 for i in y], prompt_vals, height=h,
            color=C_PROMPT, label="Prompt-only (best)")
    ax.barh([i - h / 2 for i in y], train_vals, height=h,
            color=C_TRAIN, label="Trained 0.8B student")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("score (higher = behavior present)")

    # value annotations
    for i, (p, t) in enumerate(zip(prompt_vals, train_vals)):
        ax.text(p + 0.01, i + h / 2, f"{p:.2f}", va="center", fontsize=7.5, color=C_PROMPT)
        ax.text(t + 0.01, i - h / 2, f"{t:.2f}", va="center", fontsize=7.5, color=C_TRAIN)

    # separator between promptable (top 3) and not-promptable (bottom 2)
    ax.axhline(2.5, color=C_NEUTRAL, ls="--", lw=1)
    ax.text(0.30, 0.98, "PROMPTABLE\n(prompt reaches parity)", ha="center", va="center",
            fontsize=7.5, color=C_PROMPTABLE, fontweight="bold")
    ax.text(0.30, 3.55, "NOT PROMPTABLE\n(training required)", ha="center", va="center",
            fontsize=7.5, color=C_NOTPROMPT, fontweight="bold")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_boundary.png")


# ---------------------------------------------------------------------------
# Figure 2 — matched-prompt experimental design
# ---------------------------------------------------------------------------
def fig_design():
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                           linewidth=1, edgecolor=C_NEUTRAL, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1, color=C_NEUTRAL))

    # the single shared prompt
    box(0.2, 2.4, 2.2, 1.2, "Same fully-specified\ndeployment prompt\n(§3.5)", "#eef2f4", fs=8)

    # conditions
    conds = [
        (4.0, 5.0, "A1  trained 0.8B student", C_TRAIN, "white"),
        (4.0, 4.0, "A3  generic-SFT ablation", "#a9c9d6", "black"),
        (4.0, 3.0, "B2/B3  instruct (0.8B/4B)", "#e8d5d6", "black"),
        (4.0, 2.0, "B4  9B teacher", C_PROMPT, "white"),
        (4.0, 1.0, "B1  0.8B base", "#efe0e1", "black"),
    ]
    for (x, y, t, fc, tc) in conds:
        p = FancyBboxPatch((x, y - 0.32), 3.4, 0.64,
                           boxstyle="round,pad=0.03,rounding_size=0.08",
                           linewidth=1, edgecolor=C_NEUTRAL, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + 1.7, y, t, ha="center", va="center", fontsize=7.8, color=tc)
        arrow(2.4, 3.0, 4.0, y)

    # per-capability instrument
    box(8.0, 2.4, 1.8, 1.2, "Per-capability\ninstrument\n(§5.3-§5.7)", "#eef2f4", fs=8)
    for (_, y, *_rest) in conds:
        arrow(7.4, y, 8.0, 3.0)

    ax.text(1.3, 3.9, "prompt-only\nfailure isolates\npromptability",
            ha="center", va="bottom", fontsize=6.8, style="italic", color=C_NEUTRAL)
    ax.set_title("Matched-prompt protocol: one prompt, a ladder of conditions, one instrument per capability")
    _save(fig, "fig_design.png")


# ---------------------------------------------------------------------------
# Figure 3 — generation pipeline (8 stages)
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        "Seeds\n(CEFR A1-C2)",
        "12 SFT\nstreams",
        "6-filter\ncascade",
        "Yield-aware\ntop-up loop",
        "QLoRA SFT\n(0.8B student)",
        "Freeze 6\neval sets",
        "Score:\nmechanical\n+ 3 judges",
    ]
    n = len(stages)
    w = 1.18
    gap = (10 - n * w) / (n + 1)
    y = 1.1
    xs = []
    for i, s in enumerate(stages):
        x = gap + i * (w + gap)
        xs.append(x)
        fc = "#eef2f4" if i not in (1, 4) else "#dbe7ec"
        p = FancyBboxPatch((x, y), w, 0.9, boxstyle="round,pad=0.04,rounding_size=0.1",
                           linewidth=1, edgecolor=C_NEUTRAL, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + 0.45, s, ha="center", va="center", fontsize=6.6)
        if i > 0:
            xp = xs[i - 1]
            ax.add_patch(FancyArrowPatch((xp + w, y + 0.45), (x, y + 0.45),
                                         arrowstyle="-|>", mutation_scale=9, lw=1,
                                         color=C_NEUTRAL))

    # teacher note
    ax.text(xs[1] + w / 2, y + 1.25,
            "single locally-served 9B teacher generates all streams",
            ha="center", fontsize=6.8, style="italic", color=C_TRAIN)
    ax.annotate("", xy=(xs[1] + w / 2, y + 0.95), xytext=(xs[1] + w / 2, y + 1.18),
                arrowprops=dict(arrowstyle="-|>", color=C_TRAIN, lw=0.8))
    ax.set_title("Locale-aware, yield-aware generation pipeline (all on one RTX 3060 12GB)")
    _save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 4 — failure mechanism (persistence ladder + withholding ablation)
# ---------------------------------------------------------------------------
def fig_failure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))

    # -- left: persistence prompting ladder (9B teacher) vs trained student --
    ladder = [
        ("zero-shot", 0.025),
        ("+ few-shot", 0.013),
        ("+ CoT\nscaffold", 0.057),
        ("+ native\nCoT", 0.63),
    ]
    labels = [l[0] for l in ladder]
    vals = [l[1] for l in ladder]
    xpos = range(len(ladder))
    ax1.bar(xpos, vals, color=C_PROMPT, width=0.6)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.02, f"{v:.2f}" if v >= 0.06 else f"{v:.3f}",
                 ha="center", fontsize=7)
    ax1.axhline(0.83, color=C_TRAIN, ls="--", lw=1.3)
    ax1.text(len(ladder) - 1, 0.85, "trained student 0.83", ha="right",
             va="bottom", fontsize=7.5, color=C_TRAIN)
    ax1.set_xticks(list(xpos))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("sentinel recall")
    ax1.set_title("Persistence: 9B teacher prompting ladder", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # -- right: withholding rate across conditions --
    conds = [
        ("B1 base", 0.087, C_PROMPT),
        ("A3 ablate", 0.119, C_PROMPT),
        ("B3 4B", 0.294, C_PROMPT),
        ("B2 0.8B", 0.310, C_PROMPT),
        ("B4 9B", 0.452, C_PROMPT),
        ("A1 trained", 0.611, C_TRAIN),
    ]
    labels2 = [c[0] for c in conds]
    vals2 = [c[1] for c in conds]
    colors2 = [c[2] for c in conds]
    xpos2 = range(len(conds))
    ax2.bar(xpos2, vals2, color=colors2, width=0.6)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax2.set_xticks(list(xpos2))
    ax2.set_xticklabels(labels2, fontsize=7, rotation=25, ha="right")
    ax2.set_ylim(0, 0.75)
    ax2.set_ylabel("withholding rate")
    ax2.set_title("Withholding: matched prompt (2 judges, n=63)", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig_failure.png")


if __name__ == "__main__":
    fig_boundary()
    fig_design()
    fig_pipeline()
    fig_failure()
    print("all figures written to", OUT)
