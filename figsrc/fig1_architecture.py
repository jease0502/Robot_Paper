"""Figure 1 - System architecture of the command-filtering intervention.

Core conclusion
    The measured torque stream is produced by a frozen policy driving a
    ZOH + motor-side PD chain; the command filter intervenes at exactly one
    point, and only 18.1% of the resulting copper loss lies in the frequency
    bands that intervention can reach.

Archetype
    Schematic-led composite (block signal chain + true-scale band strip).
    Single drawing axes, therefore the multi-panel alignment gate reports
    NOT APPLICABLE by construction.

All numeric values are transcribed verbatim from the manuscript figure spec.
Run from the repository root:  python figsrc/fig1_architecture.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# --------------------------------------------------------------------------
# Publication rcParams.  svg.fonttype="none" keeps SVG text selectable and the
# file small; pdf.fonttype=42 keeps PDF text editable for the QA audits.
# --------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": ["Source Sans 3", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
    }
)

# Literal colours only: no CSS variables, no dark-mode variant.
INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
BOX_FACE = "#f2f5f8"
BOX_EDGE = "#8c9aa8"
SIGNAL = "#d1603d"          # the intervention / the reachable heat
BAND_DC = "#c9d3dd"         # DC, posture
BAND_GAIT = "#8aa8c4"       # 0-5 Hz, gait
BAND_JITTER = "#d1603d"     # 5-25 Hz, in-band jitter (hero band)
BAND_ZOH = "#f2b263"        # >25 Hz, ZOH staircase harmonics

FINAL_WIDTH_MM = 180.0      # full journal page width
W_PT = FINAL_WIDTH_MM / 25.4 * 72.0
Y_TOP, Y_BOT = 400.0, 52.0
H_PT = Y_TOP - Y_BOT

FS_BODY = 7.0
FS_TITLE = 7.6
LEAD = 9.0

# --------------------------------------------------------------------------
# Signal-chain geometry, in typographic points inside the drawing axes.
# --------------------------------------------------------------------------
BLOCK_TOP, BLOCK_BOT = 322.0, 282.0
FILTER_TOP = 346.0
CHAIN_Y = 302.0
BLOCKS = {
    "policy": (8.0, 72.0),
    "filter": (98.0, 222.0),
    "zoh": (274.0, 344.0),
    "pd": (358.0, 432.0),
    "plant": (446.0, 502.0),
}


def centre(name: str) -> float:
    x0, x1 = BLOCKS[name]
    return 0.5 * (x0 + x1)


def block(ax, name, lines, *, top=BLOCK_TOP, bottom=BLOCK_BOT,
          face=BOX_FACE, edge=BOX_EDGE, dashed=False, lw=0.7):
    x0, x1 = BLOCKS[name]
    ax.add_patch(
        Rectangle(
            (x0, bottom), x1 - x0, top - bottom,
            facecolor=face, edgecolor=edge, linewidth=lw,
            linestyle=(0, (3.2, 2.2)) if dashed else "solid",
            joinstyle="round", zorder=2,
        )
    )
    # Vertically centre the text stack inside the block.
    n = len(lines)
    stack_h = (n - 1) * LEAD
    y = 0.5 * (top + bottom) + 0.5 * stack_h - 2.4
    for text, weight, size, colour in lines:
        ax.text(0.5 * (x0 + x1), y, text, ha="center", va="baseline",
                fontsize=size, fontweight=weight, color=colour, zorder=3)
        y -= LEAD


def chain_arrow(ax, x_from, x_to, y=CHAIN_Y, colour=INK, lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            (x_from, y), (x_to, y),
            arrowstyle="-|>", mutation_scale=6.0,
            linewidth=lw, color=colour, shrinkA=0, shrinkB=0, zorder=3,
        )
    )


def stack(ax, x, y0, lines, *, ha="center", size=FS_BODY, colour=INK_SOFT,
          lead=LEAD, style="normal"):
    y = y0
    for text in lines:
        if text is None:            # blank spacer line
            y -= 0.5 * lead
            continue
        ax.text(x, y, text, ha=ha, va="baseline", fontsize=size,
                color=colour, style=style, zorder=3)
        y -= lead
    return y


def build():
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_axis_off()

    # ---------------- observation feedback loop (drawn first, sits behind) --
    rail_y = 362.0
    ax.plot([centre("plant"), centre("plant")], [BLOCK_TOP, rail_y],
            color=INK_SOFT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax.plot([centre("plant"), centre("policy")], [rail_y, rail_y],
            color=INK_SOFT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            (centre("policy"), rail_y), (centre("policy"), BLOCK_TOP + 1.0),
            arrowstyle="-|>", mutation_scale=6.0, linewidth=0.8,
            linestyle=(0, (3.0, 2.0)), color=INK_SOFT, shrinkA=0, shrinkB=0,
            zorder=1,
        )
    )
    ax.text(257.0, 366.0, "Observation feedback (50 Hz)", ha="center",
            va="baseline", fontsize=FS_BODY, color=INK_SOFT)
    ax.text(257.0, 389.0,
            "The policy is itself a feedback controller: latency added on this",
            ha="center", va="baseline", fontsize=FS_BODY, color=INK)
    ax.text(257.0, 380.0,
            "path is subtracted from its stability margin.",
            ha="center", va="baseline", fontsize=FS_BODY, color=INK)

    # ---------------- the block chain ---------------------------------------
    block(ax, "policy", [
        ("Policy π", "bold", FS_TITLE, INK),
        ("weights frozen", "normal", FS_BODY, INK_SOFT),
        ("50 Hz", "normal", FS_BODY, INK_SOFT),
    ])
    block(ax, "filter", [
        ("Command filter", "bold", FS_TITLE, SIGNAL),
        ("AR(8) predictor, 2 steps ahead", "normal", FS_BODY, INK),
        ("Symmetric Hann kernel", "normal", FS_BODY, INK),
        ("(zero phase)", "normal", FS_BODY, INK),
        ("Load gate: filter only if", "normal", FS_BODY, INK),
        ("mean|τ| < 3.2 N·m", "normal", FS_BODY, INK),
    ], top=FILTER_TOP, face="#fdf1ec", edge=SIGNAL, dashed=True, lw=0.9)
    block(ax, "zoh", [
        ("ZOH", "bold", FS_TITLE, INK),
        ("hold 10 ticks", "normal", FS_BODY, INK_SOFT),
        ("50 → 500 Hz", "normal", FS_BODY, INK_SOFT),
    ])
    block(ax, "pd", [
        ("Motor-side PD", "bold", FS_TITLE, INK),
        ("500 Hz", "normal", FS_BODY, INK_SOFT),
    ])
    block(ax, "plant", [
        ("Joint + motor", "bold", FS_TITLE, INK),
        ("Go1, 12 DoF", "normal", FS_BODY, INK_SOFT),
    ])

    for a, b in (("policy", "filter"), ("filter", "zoh"),
                 ("zoh", "pd"), ("pd", "plant")):
        chain_arrow(ax, BLOCKS[a][1], BLOCKS[b][0])
    ax.text(85.0, 306.0, "q*", ha="center", va="baseline",
            fontsize=FS_BODY, color=INK)
    ax.text(248.0, 306.0, "filtered q*", ha="center", va="baseline",
            fontsize=FS_BODY, color=SIGNAL)

    # ---------------- annotation lane under the chain -----------------------
    stack(ax, centre("filter"), 272.0, [
        "Intervention of this work;",
        "the policy is not retrained.",
        None,
        "No latency paid: the future samples",
        "the symmetric kernel needs are",
        "estimated, not waited for.",
    ], colour=INK)
    stack(ax, centre("zoh"), 272.0, ["staircase → kp·Δq", "transient"])
    stack(ax, centre("pd"), 272.0, ["runs on the drive,", "not in the policy process"])

    # ---------------- torque tap and the two measurement products -----------
    tap_x = centre("plant")
    spine_y = 210.0
    ax.plot([tap_x, tap_x], [BLOCK_BOT, spine_y], color=INK, lw=0.9, zorder=2)
    ax.plot([tap_x], [BLOCK_BOT], marker="o", markersize=2.6, color=INK, zorder=3)
    ax.text(469.0, 245.0, "τ(t)", ha="right", va="baseline",
            fontsize=FS_BODY, color=INK)

    prod = [(8.0, 250.0, [("Four-band decomposition", "bold", FS_TITLE, INK),
                          ("Parseval: the four terms sum exactly", "normal", FS_BODY, INK_SOFT)]),
            (262.0, 502.0, [("Copper-loss proxy", "bold", FS_TITLE, INK),
                            ("H = (1/T) ∫ τ² dt  ~  I²R", "normal", FS_BODY, INK_SOFT),
                            ("rms current I, winding resistance R", "normal", FS_BODY, INK_SOFT)])]
    ax.plot([129.0, tap_x], [spine_y, spine_y], color=INK, lw=0.9, zorder=2)
    for x0, x1, lines in prod:
        xc = 0.5 * (x0 + x1)
        ax.add_patch(Rectangle((x0, 164.0), x1 - x0, 40.0, facecolor="#ffffff",
                               edgecolor=BOX_EDGE, linewidth=0.7, zorder=2))
        ax.add_patch(FancyArrowPatch((xc, spine_y), (xc, 205.0),
                                     arrowstyle="-|>", mutation_scale=6.0,
                                     linewidth=0.9, color=INK, shrinkA=0,
                                     shrinkB=0, zorder=2))
        y = 192.0
        for text, weight, size, colour in lines:
            ax.text(xc, y, text, ha="center", va="baseline", fontsize=size,
                    fontweight=weight, color=colour, zorder=3)
            y -= LEAD

    # ---------------- four-band strip, segments to true scale ---------------
    strip_x0, strip_x1 = 8.0, 502.0
    strip_w = strip_x1 - strip_x0
    strip_bot, strip_top = 110.0, 132.0
    bands = [
        ("DC", 31.3, BAND_DC, INK, ["posture: holding the body up"]),
        ("0–5 Hz", 50.6, BAND_GAIT, INK, ["gait: torque walking actually requires"]),
        ("5–25 Hz", 16.5, BAND_JITTER, "#ffffff",
         ["in-band jitter —", "where the waste actually is"]),
        (">25 Hz", 1.6, BAND_ZOH, INK, []),
    ]
    ax.text(strip_x0, 146.0,
            "Four-band heat budget, full-reward policy (100% of copper loss)",
            ha="left", va="baseline", fontsize=FS_TITLE, fontweight="bold", color=INK)

    x = strip_x0
    edges = []
    for name, share, face, txt_colour, desc in bands:
        w = strip_w * share / 100.0
        ax.add_patch(Rectangle((x, strip_bot), w, strip_top - strip_bot,
                               facecolor=face, edgecolor="#ffffff",
                               linewidth=0.8, zorder=2))
        xc = x + 0.5 * w
        if share > 5.0:                       # label fits inside the segment
            ax.text(xc, 118.2, f"{name}  {share}%", ha="center", va="baseline",
                    fontsize=FS_BODY, fontweight="bold", color=txt_colour, zorder=3)
        stack(ax, xc, 100.0, desc, colour=INK_SOFT)
        edges.append((x, x + w))
        x += w

    # The 1.6% sliver is 7.9 pt wide, so it is drawn to true scale and given an
    # external leader-line callout instead of an in-segment label.
    sliver_x0, sliver_x1 = edges[3]
    sliver_c = 0.5 * (sliver_x0 + sliver_x1)
    ax.plot([sliver_c, sliver_c], [strip_top, 143.0], color=INK_SOFT, lw=0.6, zorder=3)
    ax.text(strip_x1, 146.0, ">25 Hz  1.6%  ZOH staircase harmonics",
            ha="right", va="baseline", fontsize=FS_BODY, color=INK)

    # filter-reachable bracket spans the last two bands (16.5% + 1.6%)
    br_x0, br_x1 = edges[2][0], edges[3][1]
    ax.plot([br_x0, br_x0, br_x1, br_x1], [84.0, 79.0, 79.0, 84.0],
            color=SIGNAL, lw=0.9, solid_joinstyle="miter", zorder=3)
    stack(ax, 0.5 * (br_x0 + br_x1), 70.0,
          ["filter-reachable", "= 18.1%"], colour=SIGNAL)
    return fig


def main():
    fig = build()
    fig.canvas.draw()
    out_pdf = os.path.join(HERE, "out", "fig1-architecture.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(HERE, "out", "fig1-architecture.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(os.path.join(ROOT, "fig", "fig1-architecture.svg"))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600)
    plt.close(fig)
    print("wrote fig/fig1-architecture.svg and figsrc/out/fig1-architecture.pdf")


if __name__ == "__main__":
    main()
