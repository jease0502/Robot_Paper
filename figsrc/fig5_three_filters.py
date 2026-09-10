"""Figure 5 - Three ways to smooth a command, and what each one costs.

Core conclusion
    All three schemes apply the same kernel shape; only its position on the
    time axis differs, and that position is the whole cost. A causal kernel
    buys smoothness with phase lag, a delayed symmetric kernel buys zero phase
    with real closed-loop latency, and the predictive kernel of this work pays
    neither because it estimates the future half instead of waiting for it.

Archetype
    Schematic-led composite. Pure schematic: it contains no measured data, so
    no quantities are printed and no weight values are shown. One drawing axes,
    therefore the multi-panel alignment gate reports NOT APPLICABLE by
    construction.

Palette rationale
    The three rows reuse the same three literal colours the quantitative
    sibling figure assigns to the same three schemes, so a reader tracks one
    scheme across both figures by colour. Their greyscale luminances are well
    separated (about 208 / 163 / 97 on a 0-255 scale), and the proposed scheme
    is the darkest, so it stays the most salient mark in print and in
    grayscale.

Run from the repository root:  python figsrc/fig5_three_filters.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# --------------------------------------------------------------------------
# Publication rcParams, identical to the sibling figures.  svg.fonttype="none"
# keeps SVG text selectable; pdf.fonttype=42 keeps PDF text editable for QA.
# --------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": ["Source Sans 3", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.7,
        "legend.frameon": False,
    }
)

# Literal colours only: no CSS variables, no dark-mode variant.
INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
RULE = "#8c9aa8"          # row baselines and the shared time axis
TICK_SOFT = "#b8c2cb"     # per-row step ticks

# Fill / accent pairs, one per smoothing scheme.
CAUSAL_FILL, CAUSAL_EDGE = "#c8d2db", "#7d8894"
DELAY_FILL, DELAY_EDGE = "#93a7ba", "#4a5661"
PREDICT_FILL, PREDICT_EDGE = "#2f6f9e", "#2f6f9e"

FINAL_WIDTH_MM = 180.0     # full journal page width
W_PT = FINAL_WIDTH_MM / 25.4 * 72.0
H_PT = 444.0

FS_BODY = 7.0
FS_SMALL = 6.5
FS_ROW = 7.8
FS_TITLE = 8.4
LEAD = 9.0

X0, X1 = 8.0, 502.0                    # text / rule margins
TICK_X0, TICK_STEP = 96.0, 63.0        # discrete time axis
BAR_W = 36.0
MAX_BAR = 34.0

AXIS_Y = 33.0                          # shared time axis
TICK_LABEL_Y = 20.0
CAPTION_Y = 8.0

ROW_PITCH = 126.0
ROW3_BASE = 78.0
ROW2_BASE = ROW3_BASE + ROW_PITCH
ROW1_BASE = ROW2_BASE + ROW_PITCH

STEPS = [-4, -3, -2, -1, 0, 1, 2]
STEP_LABELS = ["t-4", "t-3", "t-2", "t-1", "t", "t+1", "t+2"]

# Kernel weights are a design choice, not measurement: they only need to read
# as "decaying", "symmetric about t-2", and "symmetric about t".  No weight
# value is ever printed.
ROWS = [
    dict(
        base=ROW1_BASE,
        title="Causal low-pass",
        sub=["weights decay into the past. The centre of mass sits behind now,",
             "and that is exactly the phase lag."],
        footer="The centre of mass sits about a step behind. "
               "Costs 25 points of motion (Section 5.3).",
        fill=CAUSAL_FILL,
        edge=CAUSAL_EDGE,
        weights={-4: 0.14, -3: 0.26, -2: 0.44, -1: 0.68, 0: 1.00},
        future=(),
        output=-1,
    ),
    dict(
        base=ROW2_BASE,
        title="Zero phase by delaying",
        sub=["symmetric weights, so no phase distortion, but they are centred",
             "on a sample that is already two steps old."],
        footer="40 ms of added latency. Falls at speed, "
               "up to every episode (Section 5.5).",
        fill=DELAY_FILL,
        edge=DELAY_EDGE,
        weights={-4: 0.30, -3: 0.72, -2: 1.00, -1: 0.72, 0: 0.30},
        future=(),
        output=-2,
    ),
    dict(
        base=ROW3_BASE,
        title="Predictive, this paper",
        sub=["symmetric weights centred on now. The right half does not exist yet,",
             "so a small per-joint model estimates it."],
        footer="No lag and no latency. Bounded by how predictable the command is.",
        fill=PREDICT_FILL,
        edge=PREDICT_EDGE,
        weights={-2: 0.30, -1: 0.72, 0: 1.00, 1: 0.72, 2: 0.30},
        future=(1, 2),
        output=0,
    ),
]

NOW_X = TICK_X0 + 4 * TICK_STEP         # the x of step t
NOW_TOP = ROW1_BASE + MAX_BAR + 4.0

# Vertical offsets shared by every row, measured from that row's baseline.
OFF_SUB2 = 46.0
OFF_SUB1 = 55.0
OFF_TITLE = 68.0
OFF_LEADER = -3.0
OFF_ANNOT1 = -15.0
OFF_ANNOT2 = -24.0
# The bold takeaway hugs its own row (10 pt below the marker note) and is
# separated from the next row heading by 24 pt, so it can only be read as
# belonging to the row above it.
OFF_FOOTER = -34.0


HATCH_STEP = 5.0          # spacing of the 45-degree rules inside hollow bars


def step_x(k: int) -> float:
    return TICK_X0 + (k + 4) * TICK_STEP


def hollow_hatched_rect(ax, x, y, w, h, colour, *, zorder=3):
    """A hollow box filled with 45-degree rules drawn as real, clipped lines.

    Matplotlib's ``hatch=`` keyword emits a PDF tiling pattern whose content
    stream extends beyond the patch, which makes rendered-geometry QA report
    phantom strokes far away from the figure element.  Drawing the rules
    explicitly keeps the exported geometry identical to what the reader sees.
    """
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=colour,
                  linewidth=0.8, zorder=zorder)
    )
    offset = -w
    while offset < h:
        u0 = max(0.0, -offset)
        u1 = min(w, h - offset)
        if u1 - u0 > 0.15:
            ax.plot([x + u0, x + u1], [y + u0 + offset, y + u1 + offset],
                    color=colour, lw=0.7, zorder=zorder,
                    solid_capstyle="butt")
        offset += HATCH_STEP


def draw_row(ax, row):
    base = row["base"]
    fill, edge = row["fill"], row["edge"]

    # ---- row baseline and the per-step ticks that carry bar-to-step reading
    ax.plot([X0, X1], [base, base], color=RULE, lw=0.7, zorder=2,
            solid_capstyle="butt")
    for k in STEPS:
        x = step_x(k)
        ax.plot([x, x], [base, base - 3.0], color=TICK_SOFT, lw=0.5, zorder=2,
                solid_capstyle="butt")

    # ---- kernel weights as bars on the baseline --------------------------
    for k, w in row["weights"].items():
        x = step_x(k)
        h = MAX_BAR * w
        if k in row["future"]:
            hollow_hatched_rect(ax, x - 0.5 * BAR_W, base, BAR_W, h, edge)
        else:
            ax.add_patch(
                Rectangle((x - 0.5 * BAR_W, base), BAR_W, h,
                          facecolor=fill, edgecolor=edge, linewidth=0.5,
                          zorder=3)
            )

    # ---- which instant the filtered output actually describes -------------
    out_x = step_x(row["output"])
    lead_y = base + OFF_LEADER
    if row["output"] != 0:
        ax.plot([out_x, NOW_X], [lead_y, lead_y], color=edge, lw=0.7,
                ls=(0, (3.0, 2.0)), zorder=4, solid_capstyle="butt")
    ax.plot([out_x], [lead_y], marker="^", markersize=4.6, color=edge,
            markeredgewidth=0.0, zorder=5, linestyle="none")
    ax.text(out_x, base + OFF_ANNOT1, "the output describes", ha="center",
            va="baseline", fontsize=FS_SMALL, color=INK_SOFT, zorder=5)
    ax.text(out_x, base + OFF_ANNOT2, "this instant", ha="center",
            va="baseline", fontsize=FS_SMALL, color=INK_SOFT, zorder=5)

    # ---- row headings and the bold takeaway ------------------------------
    ax.text(X0, base + OFF_TITLE, row["title"], ha="left", va="baseline",
            fontsize=FS_ROW, fontweight="bold", color=INK, zorder=5)
    ax.text(X0, base + OFF_SUB1, row["sub"][0], ha="left", va="baseline",
            fontsize=FS_BODY, color=INK_SOFT, zorder=5)
    ax.text(X0, base + OFF_SUB2, row["sub"][1], ha="left", va="baseline",
            fontsize=FS_BODY, color=INK_SOFT, zorder=5)
    ax.text(X0, base + OFF_FOOTER, row["footer"], ha="left", va="baseline",
            fontsize=FS_BODY, fontweight="bold", color=INK, zorder=5)


def build():
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H_PT)
    ax.set_axis_off()

    # ---- figure title, two lines ----------------------------------------
    ax.text(0.5 * W_PT, 430.0,
            "Three ways to smooth a command, and what each one costs",
            ha="center", va="baseline", fontsize=FS_TITLE, fontweight="bold",
            color=INK, zorder=5)
    ax.text(0.5 * W_PT, 417.0,
            "The kernel is the same shape throughout. "
            "Only where it sits in time changes.",
            ha="center", va="baseline", fontsize=FS_BODY, color=INK_SOFT,
            zorder=5)

    # ---- the "now" rule, drawn once through all three rows ---------------
    ax.plot([NOW_X, NOW_X], [ROW3_BASE, NOW_TOP], color=INK, lw=0.8,
            ls=(0, (0.8, 2.0)), zorder=6, solid_capstyle="butt")
    ax.text(NOW_X - 4.0, NOW_TOP + 2.0, "now", ha="right", va="baseline",
            fontsize=FS_BODY, color=INK, zorder=6)

    # ---- legend for the hollow, hatched future samples -------------------
    hollow_hatched_rect(ax, 380.0, 395.0, 16.0, 8.0, PREDICT_EDGE, zorder=5)
    ax.text(400.0, 396.0, "sample that has not happened yet", ha="left",
            va="baseline", fontsize=FS_BODY, color=INK, zorder=5)

    for row in ROWS:
        draw_row(ax, row)

    # ---- shared discrete time axis ---------------------------------------
    ax.plot([X0, X1], [AXIS_Y, AXIS_Y], color=RULE, lw=0.7, zorder=3,
            solid_capstyle="butt")
    for k, label in zip(STEPS, STEP_LABELS):
        x = step_x(k)
        ax.plot([x, x], [AXIS_Y, AXIS_Y - 4.0], color=RULE, lw=0.7, zorder=3,
                solid_capstyle="butt")
        ax.text(x, TICK_LABEL_Y, label, ha="center", va="baseline",
                fontsize=FS_BODY, color=INK, zorder=5)
    ax.text(0.5 * (TICK_X0 + step_x(2)), CAPTION_Y, "policy steps, 20 ms apart",
            ha="center", va="baseline", fontsize=FS_BODY, color=INK, zorder=5)
    return fig


def main():
    fig = build()
    fig.canvas.draw()
    out_pdf = os.path.join(HERE, "out", "fig5-three-filters.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(HERE, "out", "fig5-three-filters.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(os.path.join(ROOT, "fig", "fig5-three-filters.svg"))
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=600)
    plt.close(fig)
    print("wrote fig/fig5-three-filters.svg and figsrc/out/fig5-three-filters.pdf")


if __name__ == "__main__":
    main()
