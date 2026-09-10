"""Figure 3 - Four-band heat budget for two reward settings.

Core conclusion
    Removing the effort reward terms does not merely make the policy hotter:
    it moves 25 percentage points of copper loss into the frequency band a
    command filter can reach, and the predictive filter recovers a stable
    FRACTION of that reachable heat in both settings.

Archetype
    Quantitative grid, hero panel a (true-scale 100% band bars) plus
    subordinate panel b (recovered heat).  Panels a and b form one row group
    with identical top/bottom edges and heights, so the render-time alignment
    gate is expected to PASS.

All numeric values are transcribed verbatim from the manuscript figure spec.
Run from the repository root:  python figsrc/fig3_band_budget.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

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
        "axes.spines.left": False,
        "xtick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "xtick.labelsize": 7,
        "legend.frameon": False,
    }
)

INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
SIGNAL = "#d1603d"
BAND_DC = "#c9d3dd"
BAND_GAIT = "#8aa8c4"
BAND_JITTER = "#d1603d"
BAND_ZOH = "#f2b263"

BANDS = [
    ("DC", BAND_DC, INK),
    ("0–5 Hz", BAND_GAIT, INK),
    ("5–25 Hz", BAND_JITTER, "#ffffff"),
    (">25 Hz", BAND_ZOH, INK),
]

# ---- authoritative numbers ------------------------------------------------
ROWS = [
    {
        "name": "Full reward",
        "shares": [31.3, 50.6, 16.5, 1.6],
        "hzoh": "H(ZOH) = 255.9",
        "reachable": "18.1%",
        "recovered": 0.915,
        "recovered_note": "47% of 18.1% recovered",
    },
    {
        "name": "No torque + energy",
        "shares": [27.0, 29.7, 40.7, 2.6],
        "hzoh": "H(ZOH) = 376.3  (+47%)",
        "reachable": "43.3%",
        "recovered": 0.765,
        "recovered_note": "54% of 43.3% recovered",
    },
]

# Shared vertical layout so panels a and b read as the same two rows.
Y_TOP, Y_BOT = 10.0, 0.0
ROW_LABEL_Y = [9.30, 5.00]
BAR_TOP = [8.80, 4.50]
BAR_BOT = [7.60, 3.30]
BRACKET_Y = [7.10, 2.80]
BRACKET_LABEL_Y = [6.30, 2.00]
NOTE_Y = [0.95, 0.20]

FINAL_WIDTH_MM = 180.0      # full journal page width
W_IN, H_IN = FINAL_WIDTH_MM / 25.4, 3.05


def panel_label(ax, text):
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(-17.0, 6.0), textcoords="offset points",
                ha="left", va="baseline", fontsize=8, fontweight="bold",
                color=INK, annotation_clip=False)


def build():
    fig = plt.figure(figsize=(W_IN, H_IN))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0],
                          left=0.058, right=0.985, bottom=0.235, top=0.925,
                          wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ================= panel a: true-scale 100% band bars ==================
    ax_a.set_xlim(0.0, 100.0)
    ax_a.set_ylim(Y_BOT, Y_TOP)
    ax_a.set_yticks([])
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.spines["bottom"].set_bounds(0, 100)
    ax_a.spines["bottom"].set_color("#7d8894")
    ax_a.tick_params(colors="#7d8894", labelcolor=INK)
    ax_a.set_xlabel("Share of copper loss (%)", fontsize=7.4, color=INK)

    for i, row in enumerate(ROWS):
        ax_a.text(0.0, ROW_LABEL_Y[i], f"{row['name']}   ·   {row['hzoh']}",
                  ha="left", va="baseline", fontsize=7.4, fontweight="bold",
                  color=INK)
        x = 0.0
        edges = []
        h = BAR_TOP[i] - BAR_BOT[i]
        for share, (name, face, txt_colour) in zip(row["shares"], BANDS):
            ax_a.add_patch(Rectangle((x, BAR_BOT[i]), share, h, facecolor=face,
                                     edgecolor="#ffffff", linewidth=0.8, zorder=2))
            if share >= 5.0:
                ax_a.text(x + 0.5 * share, 0.5 * (BAR_TOP[i] + BAR_BOT[i]) - 0.22,
                          f"{share}", ha="center", va="baseline", fontsize=7,
                          fontweight="bold", color=txt_colour, zorder=3)
            edges.append((x, x + share))
            x += share

        # The >25 Hz slivers (1.6% and 2.6%) are ~5-9 pt wide at final size, so
        # they stay at true scale and are called out with a leader line.
        s0, s1 = edges[3]
        sc = 0.5 * (s0 + s1)
        ax_a.plot([sc, sc], [BAR_TOP[i], ROW_LABEL_Y[i] - 0.20],
                  color=INK_SOFT, lw=0.55, zorder=3)
        ax_a.text(100.0, ROW_LABEL_Y[i], f">25 Hz  {row['shares'][3]}%",
                  ha="right", va="baseline", fontsize=7, color=INK)

        # filter-reachable bracket over the 5-25 Hz and >25 Hz bands
        b0, b1 = edges[2][0], edges[3][1]
        ax_a.plot([b0, b0, b1, b1],
                  [BRACKET_Y[i] + 1.05, BRACKET_Y[i], BRACKET_Y[i],
                   BRACKET_Y[i] + 1.05],
                  color=SIGNAL, lw=0.9, solid_joinstyle="miter", zorder=3)
        ax_a.text(100.0, BRACKET_LABEL_Y[i],
                  f"filter-reachable {row['reachable']}", ha="right",
                  va="baseline", fontsize=7, fontweight="bold", color=SIGNAL)

    ax_a.text(0.0, NOTE_Y[0],
              "Removing the effort reward terms moves 25 percentage points of heat",
              ha="left", va="baseline", fontsize=7, color=INK)
    ax_a.text(0.0, NOTE_Y[1], "into the band a filter can reach.",
              ha="left", va="baseline", fontsize=7, color=INK)
    panel_label(ax_a, "a")

    # ================= panel b: what the filter recovered ==================
    ax_b.set_xlim(0.70, 1.075)
    ax_b.set_ylim(Y_BOT, Y_TOP)
    ax_b.set_yticks([])
    ax_b.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax_b.spines["bottom"].set_bounds(0.70, 1.0)
    ax_b.spines["bottom"].set_color("#7d8894")
    ax_b.tick_params(colors="#7d8894", labelcolor=INK)
    ax_b.set_xlabel("H / H0 after predictive filtering", fontsize=7.4, color=INK)

    ax_b.plot([1.0, 1.0], [BRACKET_Y[1] - 0.2, ROW_LABEL_Y[0] + 0.35],
              color=INK, lw=0.8, ls=(0, (3.0, 2.0)), zorder=3)
    ax_b.text(1.033, 6.05, "H0 baseline = 1.00", rotation=90,
              rotation_mode="anchor", ha="center", va="baseline",
              fontsize=7, color=INK_SOFT)

    for i, row in enumerate(ROWS):
        val = row["recovered"]
        h = BAR_TOP[i] - BAR_BOT[i]
        ax_b.add_patch(Rectangle((val, BAR_BOT[i]), 1.0 - val, h,
                                 facecolor=SIGNAL, edgecolor="none", zorder=2))
        ax_b.text(val - 0.008, 0.5 * (BAR_TOP[i] + BAR_BOT[i]) - 0.22,
                  f"{val:.3f}", ha="right", va="baseline", fontsize=7.4,
                  fontweight="bold", color=INK, zorder=3)
        ax_b.text(0.712, ROW_LABEL_Y[i], row["recovered_note"], ha="left",
                  va="baseline", fontsize=7, color=INK)

    ax_b.text(0.712, NOTE_Y[0], "Raw savings differ by nearly 3×;",
              ha="left", va="baseline", fontsize=7, color=INK)
    ax_b.text(0.712, NOTE_Y[1], "the recovered fraction is stable.",
              ha="left", va="baseline", fontsize=7, color=INK)
    panel_label(ax_b, "b")

    # ---- shared band legend along the bottom of the figure ----------------
    handles = [
        Patch(facecolor=BAND_DC, edgecolor="#ffffff", label="DC (posture)"),
        Patch(facecolor=BAND_GAIT, edgecolor="#ffffff", label="0–5 Hz (gait)"),
        Patch(facecolor=BAND_JITTER, edgecolor="#ffffff",
              label="5–25 Hz (in-band jitter)"),
        Patch(facecolor=BAND_ZOH, edgecolor="#ffffff",
              label=">25 Hz (ZOH harmonics)"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.006), fontsize=7,
               handlelength=1.3, handleheight=0.85, columnspacing=1.5,
               handletextpad=0.5, borderpad=0.0, labelcolor=INK)
    return fig


def main():
    fig = build()
    fig.canvas.draw()
    out_pdf = os.path.join(HERE, "out", "fig3-band-budget.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(HERE, "out", "fig3-band-budget.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(os.path.join(ROOT, "fig", "fig3-band-budget.svg"))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600)
    plt.close(fig)
    print("wrote fig/fig3-band-budget.svg and figsrc/out/fig3-band-budget.pdf")


if __name__ == "__main__":
    main()
