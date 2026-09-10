"""Figure 4 - Latency backfires, and only at speed.

Core conclusion
    A smoothing filter that pays its zero phase with real closed-loop delay is
    the best scheme tested at 0.5 m/s and the worst at 1.0 m/s, where it makes
    the joints hotter than doing nothing and drops the robot; the predictive,
    load-gated filter is the only scheme that never falls.

Archetype
    Quantitative grid, single hero panel, so the multi-panel alignment gate
    reports NOT APPLICABLE by construction.

All numeric values are transcribed verbatim from the manuscript figure spec.
Run from the repository root:  python figsrc/fig4_latency.py
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
        "hatch.linewidth": 0.7,
        "legend.frameon": False,
    }
)

INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
ALARM = "#b3261e"
WORSE_BG = "#fbeae7"

# ---- authoritative numbers ------------------------------------------------
VELOCITIES = ["0.5 m/s", "1.0 m/s", "0.3 m/s + turn"]
SERIES = [
    ("Zero-phase via 40 ms delay", "#93a7ba", INK, "#3d4a56",
     [0.883, 1.258, 0.992], [0.00, 0.20, 0.00]),
    ("Butterworth 5 Hz", "#c8d2db", INK, "#4a5661",
     [0.880, 1.295, 0.991], [0.00, 0.70, 0.00]),
    ("Predictive + gate (this work)", "#2f6f9e", "#ffffff", "#dfe9f1",
     [0.952, 0.924, 0.956], [0.00, 0.00, 0.00]),
]

FINAL_WIDTH_MM = 180.0      # full journal page width
W_IN, H_IN = FINAL_WIDTH_MM / 25.4, 4.25
Y_MAX = 1.72
BAR_W = 0.25
OFFSETS = [-0.28, 0.0, 0.28]


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0.075, 0.26, 0.915, 0.725])

    x = np.arange(len(VELOCITIES), dtype=float)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0.0, Y_MAX)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00, 1.25])
    ax.set_xticks(x)
    ax.set_xticklabels(VELOCITIES)
    ax.spines["left"].set_bounds(0.0, 1.25)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#7d8894")
    ax.tick_params(colors="#7d8894", labelcolor=INK)

    # ---- "worse than doing nothing" band -----------------------------------
    ax.axhspan(1.0, Y_MAX, facecolor=WORSE_BG, edgecolor="none", zorder=0)
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3.2, 2.2)), zorder=3)
    ax.text(-0.52, 1.050, "ZOH baseline = 1.00", ha="left", va="baseline",
            fontsize=7, fontweight="bold", color=INK, zorder=5)
    ax.text(2.53, 1.200, "above 1.00 = worse than doing nothing",
            ha="right", va="baseline", fontsize=7, color=ALARM, zorder=5)

    # ---- grouped bars ------------------------------------------------------
    handles = []
    for (name, colour, on_bar, on_bar_soft, values, falls), off in zip(SERIES, OFFSETS):
        for xi, (v, fr) in zip(x, zip(values, falls)):
            failing = fr > 0.0
            ax.bar(xi + off, v, width=BAR_W, facecolor=colour, zorder=2,
                   edgecolor=ALARM if failing else "none",
                   linewidth=0.9 if failing else 0.0,
                   hatch="////" if failing else None)
            if v > 1.0:
                # Above the baseline: label outside so the bar stays legible.
                ax.text(xi + off, v + 0.092, f"{v:.3f}", ha="center",
                        va="baseline", fontsize=7, fontweight="bold",
                        color=ALARM, zorder=5)
                ax.text(xi + off, v + 0.026, f"falls {fr:.2f}", ha="center",
                        va="baseline", fontsize=7, fontweight="bold",
                        color=ALARM, zorder=5)
            else:
                # Below the baseline: label inside the bar so no label can be
                # crossed by the H/H(ZOH) = 1.00 rule.
                ax.text(xi + off, v - 0.072, f"{v:.3f}", ha="center",
                        va="baseline", fontsize=7, fontweight="bold",
                        color=on_bar, zorder=5)
                ax.text(xi + off, v - 0.142, f"falls {fr:.2f}", ha="center",
                        va="baseline", fontsize=7, color=on_bar_soft, zorder=5)
        handles.append(
            mpl.patches.Patch(facecolor=colour, edgecolor="none", label=name)
        )

    # ---- the callout that must be unmissable -------------------------------
    ax.text(1.0, 1.580, "Latency backfires here:", ha="center", va="baseline",
            fontsize=7.4, fontweight="bold", color=ALARM, zorder=5)
    ax.text(1.0, 1.495, "20% and 70% of episodes end in a fall",
            ha="center", va="baseline", fontsize=7, color=ALARM, zorder=5)

    ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=7,
              handlelength=1.3, handleheight=0.85, borderpad=0.35,
              handletextpad=0.5, labelspacing=0.42, labelcolor=INK,
              frameon=True, facecolor="#ffffff", edgecolor="none",
              framealpha=1.0, bbox_to_anchor=(0.0, 1.0))

    ax.set_ylabel("H / H(ZOH)   (higher = hotter)", fontsize=7.4, color=INK)
    ax.set_xlabel("Commanded velocity", fontsize=7.4, color=INK)

    # ---- caption-level takeaway carried inside the figure ------------------
    fig.text(0.075, 0.098,
             "The same delayed filter is the best scheme tested at 0.5 m/s and "
             "falls in a fifth of episodes at 1.0 m/s.",
             ha="left", va="baseline", fontsize=7, color=INK)
    fig.text(0.075, 0.0654,
             "Evaluating a smoothing filter only at low speed records it as a "
             "success and then fails on deployment.",
             ha="left", va="baseline", fontsize=7, color=INK)
    fig.text(0.075, 0.0327,
             "On the policy trained without effort terms the delayed scheme "
             "falls in 37% of episodes overall, and in every episode at 1.0 m/s.",
             ha="left", va="baseline", fontsize=7, color=INK_SOFT, style="italic")
    return fig


def main():
    fig = build()
    fig.canvas.draw()
    out_pdf = os.path.join(HERE, "out", "fig4-latency-backfire.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(HERE, "out", "fig4-latency-backfire.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(os.path.join(ROOT, "fig", "fig4-latency-backfire.svg"))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600)
    plt.close(fig)
    print("wrote fig/fig4-latency-backfire.svg and figsrc/out/fig4-latency-backfire.pdf")


if __name__ == "__main__":
    main()
