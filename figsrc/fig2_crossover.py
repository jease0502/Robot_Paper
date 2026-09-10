"""Figure 2 - The quadrature crossover.

Core conclusion
    Because copper loss adds in quadrature, whether command jitter or posture
    dominates a joint's heat is fixed by that joint's mean static torque, and
    the two shares cross at mean tau = sqrt(Var(tau)) = 3.2 N.m on the Go1.

Archetype
    Quantitative grid, single hero panel (definition curves + measured joint
    families on a rug lane), so the multi-panel alignment gate reports
    NOT APPLICABLE by construction.

All numeric values are transcribed verbatim from the manuscript figure spec.
Run from the repository root:  python figsrc/fig2_crossover.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

mpl.rcParams.update(
    {
        "font.family": ["Source Sans 3", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
    }
)

INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
RIPPLE = "#d1603d"        # filter-reachable variance term
POSTURE = "#33587f"       # unavoidable static term
RIPPLE_BG = "#fdf1ec"
POSTURE_BG = "#eef2f7"

# ---- authoritative numbers ------------------------------------------------
SD_TAU = 3.2              # sqrt(Var(tau)), N.m -> crossover abscissa
JOINTS = [
    ("Hip abduction", 1.24, 1.48, "33–45%"),
    ("Thigh", 2.00, 2.59, "20–32%"),
    ("Calf (knee)", 5.88, 6.88, "14–17%"),
]

FINAL_WIDTH_MM = 180.0      # full journal page width
W_IN, H_IN = FINAL_WIDTH_MM / 25.4, 4.65
Y_LANE = -42.0            # rug lane for the measured joint families


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0.072, 0.108, 0.918, 0.828])

    tau = np.linspace(0.0, 10.0, 1201)
    var = SD_TAU ** 2
    posture = 100.0 * tau ** 2 / (tau ** 2 + var)
    ripple = 100.0 * var / (tau ** 2 + var)

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(Y_LANE, 100.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(0, 11))
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_position(("data", Y_LANE))

    # ---- regime shading ---------------------------------------------------
    ax.axvspan(0.0, SD_TAU, ymin=0.0, ymax=1.0, facecolor=RIPPLE_BG,
               edgecolor="none", zorder=0)
    ax.axvspan(SD_TAU, 10.0, ymin=0.0, ymax=1.0, facecolor=POSTURE_BG,
               edgecolor="none", zorder=0)
    ax.axhline(0.0, color="#c3cbd3", lw=0.6, zorder=1)

    # ---- the two definition curves ---------------------------------------
    # Direct curve labels are used instead of a legend box.
    ax.plot(tau, ripple, color=RIPPLE, lw=1.7, zorder=4)
    ax.plot(tau, posture, color=POSTURE, lw=1.7, zorder=4)

    # ---- crossover --------------------------------------------------------
    # The crossover rule stops at the data rectangle: the regime shading
    # already carries it through the joint-family lane, and extending the line
    # would strike through every lane label.
    ax.plot([SD_TAU, SD_TAU], [0.0, 100.0], color=INK, lw=0.8,
            ls=(0, (3.0, 2.0)), zorder=3)
    ax.plot([SD_TAU], [50.0], marker="o", ms=3.4, mfc="#ffffff",
            mec=INK, mew=0.9, zorder=6)
    ax.text(SD_TAU, 104.0,
            "crossover:  mean τ = √Var(τ) = 3.2 N·m  —  both shares 50%",
            ha="center", va="baseline", fontsize=7, fontweight="bold",
            color=INK, clip_on=False)

    # ---- regime labels ----------------------------------------------------
    ax.text(1.9, 97.5, "ripple dominates", ha="center", va="baseline",
            fontsize=7, fontweight="bold", color=RIPPLE)
    ax.text(1.9, 90.0, "filtering can pay", ha="center", va="baseline",
            fontsize=7, color=RIPPLE)
    ax.text(6.8, 97.5, "posture dominates", ha="center", va="baseline",
            fontsize=7, fontweight="bold", color=POSTURE)
    ax.text(6.8, 90.0, "filtering nearly useless", ha="center", va="baseline",
            fontsize=7, color=POSTURE)

    # ---- direct curve labels (no legend box) ------------------------------
    ax.text(0.45, 78.0, "ripple share", ha="left", va="baseline",
            fontsize=7, fontweight="bold", color=RIPPLE)
    ax.text(9.55, 78.0, "posture share", ha="right", va="baseline",
            fontsize=7, fontweight="bold", color=POSTURE)

    # ---- definition note in the free wedge between the curves -------------
    ax.text(5.40, 48.0,
            "Curves are the definition r = Var / ((mean τ)² + Var), not a fit;",
            ha="left", va="baseline", fontsize=7, color=INK_SOFT)
    ax.text(5.40, 40.0,
            "joint placements and reachable shares are measured.",
            ha="left", va="baseline", fontsize=7, color=INK_SOFT)
    ax.text(5.40, 30.0, "Var(τ) held at (3.2 N·m)² throughout.",
            ha="left", va="baseline", fontsize=7, color=INK_SOFT)

    # ---- measured Go1 joint families, drawn on the rug lane ---------------
    ax.text(0.10, -6.0,
            "Measured Go1 joint families — 8 of Go1's 12 joints sit below the "
            "crossover; the 4 knees sit above.",
            ha="left", va="baseline", fontsize=7, fontweight="bold", color=INK)

    lane_y = [-16.0, -26.0, -36.0]
    for (name, lo, hi, reach), y in zip(JOINTS, lane_y):
        colour = RIPPLE if hi < SD_TAU else POSTURE
        ax.plot([lo, hi], [y, y], color=colour, lw=3.4,
                solid_capstyle="butt", zorder=4)
        for edge in (lo, hi):
            ax.plot([edge, edge], [y - 2.6, y + 2.6], color=colour, lw=0.8, zorder=4)
        text = f"{name}   {lo:.2f}–{hi:.2f} N·m   ·   filter-reachable heat {reach}"
        if name.startswith("Calf"):
            ax.text(lo - 0.20, y - 2.4, text, ha="right", va="baseline",
                    fontsize=7, color=INK)
        else:
            ax.text(hi + 0.20, y - 2.4, text, ha="left", va="baseline",
                    fontsize=7, color=INK)

    ax.set_xlabel("Mean static joint torque (N·m)", fontsize=7.4, color=INK)
    ax.set_ylabel("Share of copper loss (%)", fontsize=7.4, color=INK)
    ax.tick_params(colors=INK_SOFT, labelcolor=INK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#7d8894")
    return fig


def main():
    fig = build()
    fig.canvas.draw()
    out_pdf = os.path.join(HERE, "out", "fig2-crossover.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(HERE, "out", "fig2-crossover.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(os.path.join(ROOT, "fig", "fig2-crossover.svg"))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600)
    plt.close(fig)
    print("wrote fig/fig2-crossover.svg and figsrc/out/fig2-crossover.pdf")


if __name__ == "__main__":
    main()
