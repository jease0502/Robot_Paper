#!/usr/bin/env python3
"""Add a band ruler under pic/command-spectrum.png.

The original panel shades 15.3 Hz to 25 Hz and labels it the actuator -3 dB
bandwidth. That label is correct -- it is the band rho is defined against in
Section 3.2 -- but the paper's central claim is about the 5-25 Hz filter-reachable
band of Section 5.4, and a reader scanning the figure will read the grey region as
that band. The two are different cuts and must be visibly different.

The underlying curves are measured data and the source rollouts are not in this
repository, so this script does not redraw, retrace, or otherwise reconstruct
them. It extends the canvas downward and draws the ruler in the new strip. Every
pixel of the original figure is copied through untouched; running this on the
output twice would be idempotent only in the sense that it re-reads the original,
so it always reads `SRC` and always writes `DST`.

The x mapping is not assumed. It is measured from the rendered axes: the left and
right spines of each panel are found as the long dark columns, and the 15.3 Hz
dashed rule is used to confirm the mapping before anything is drawn. If that check
fails the script refuses to write.

Run from the repository root:  python figsrc/annotate_command_spectrum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "pic" / "command-spectrum.png"
DST = ROOT / "pic" / "command-spectrum-annotated.png"

F_MAX = 25.0          # both panels span 0..25 Hz linearly
F_DASH = 15.3         # the dashed rule already in the figure, used as the check
F_LO = 5.0            # lower edge of the paper's filter-reachable band
STRIP = 104           # height of the added annotation lane, px

INK = (16, 20, 24)
ACC = (200, 66, 30)       # same orange the vector figures use for "reachable"
MUTED = (110, 120, 126)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("arialbd.ttf", "Arial Bold.ttf") if bold else ("arial.ttf", "Arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def find_panels(arr: np.ndarray) -> list[tuple[float, float]]:
    """Return [(left_spine_x, right_spine_x), ...] for each panel, left to right."""
    dark = arr.sum(axis=2) < 260
    h = dark.shape[0]
    cols = [x for x in range(dark.shape[1]) if dark[:, x].sum() > 0.27 * h]
    # merge adjacent columns (a spine is 1-2 px wide) into single centres
    groups: list[list[int]] = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    centres = [sum(g) / len(g) for g in groups]
    if len(centres) % 3 != 0:
        raise SystemExit(f"error: expected 3 long verticals per panel, got {centres}")
    return [(centres[i], centres[i + 2]) for i in range(0, len(centres), 3)]


def check_mapping(panel: tuple[float, float], dash_centre: float) -> None:
    x0, x1 = panel
    predicted = x0 + (F_DASH / F_MAX) * (x1 - x0)
    if abs(predicted - dash_centre) > 2.0:
        raise SystemExit(
            f"error: axis mapping check failed. The {F_DASH} Hz rule should land at "
            f"x={predicted:.1f} but was found at x={dash_centre:.1f}. Refusing to draw."
        )


def main() -> int:
    if not SRC.exists():
        print(f"error: {SRC} not found", file=sys.stderr)
        return 1
    im = Image.open(SRC).convert("RGB")
    arr = np.asarray(im).astype(int)

    dark = arr.sum(axis=2) < 260
    h = dark.shape[0]
    cols = [x for x in range(dark.shape[1]) if dark[:, x].sum() > 0.27 * h]
    groups: list[list[int]] = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    centres = [sum(g) / len(g) for g in groups]
    if len(centres) != 6:
        raise SystemExit(f"error: expected 6 long verticals (2 panels), got {centres}")

    panels = [(centres[0], centres[2]), (centres[3], centres[5])]
    dashes = [centres[1], centres[4]]
    for p, d in zip(panels, dashes):
        check_mapping(p, d)
    print(f"axis mapping verified on both panels ({F_DASH} Hz rule lands within 2 px)")

    out = Image.new("RGB", (im.width, im.height + STRIP), "white")
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)

    f_lab = _font(15, bold=True)
    f_small = _font(13)
    top = im.height

    # hairline separating the added lane from the figure it annotates
    d.line([(0, top + 1), (im.width, top + 1)], fill=(222, 227, 229), width=1)

    for i, (x0, x1) in enumerate(panels):
        span = x1 - x0
        xlo = x0 + (F_LO / F_MAX) * span
        xhi = x1
        xdash = x0 + (F_DASH / F_MAX) * span
        y = top + 24

        # the paper's reachable band: 5 Hz to Nyquist
        d.line([(xlo, y), (xhi, y)], fill=ACC, width=3)
        for xe in (xlo, xhi):
            d.line([(xe, y - 6), (xe, y + 6)], fill=ACC, width=3)
        d.text(((xlo + xhi) / 2, y + 11), "5-25 Hz: filter-reachable band (Sec. 5.4)",
               font=f_lab, fill=ACC, anchor="ma")

        # The grey region is the same on both panels, so explain it once, under
        # the left one, and right-align so a long label cannot run off the canvas.
        if i == 0:
            y2 = y + 42
            d.line([(xdash, y2), (xhi, y2)], fill=MUTED, width=2)
            for xe in (xdash, xhi):
                d.line([(xe, y2 - 5), (xe, y2 + 5)], fill=MUTED, width=2)
            d.multiline_text(
                (xhi, y2 + 8),
                "grey shading, both panels: 15.3 Hz actuator -3 dB bandwidth\n"
                "-- the band rho is defined against (Sec. 3.2), a different cut",
                font=f_small, fill=MUTED, anchor="ra", align="right", spacing=3)

    out.save(DST, dpi=im.info.get("dpi", (130, 130)))
    print(f"wrote {DST.relative_to(ROOT)}  ({DST.stat().st_size/1024:.0f} KB, "
          f"{out.width}x{out.height})")
    print(f"original {SRC.relative_to(ROOT)} untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
