#!/usr/bin/env python3
"""Stage figures into icra/figs/ and build the ICRA PDF.

main.tex reads only from figs/. Nothing is authored there: the vector figures
come from ../figsrc/out (produced by ../figsrc/fig*.py) and the raster panels
from ../pic. Staging rather than \\graphicspath keeps the LaTeX directory
self-contained, so it can be zipped and handed to a co-author or an Overleaf
project without the rest of the repository.

pic/anim-phase-lag.gif is deliberately absent: a PDF cannot carry an animation,
and the paper's figure set must not depend on one.

Usage:
    python build_icra.py            stage figures, then build
    python build_icra.py --stage    stage only
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = HERE / "figs"

# destination name <- source path
SOURCES = {
    "fig1-architecture.pdf": REPO / "figsrc" / "out" / "fig1-architecture.pdf",
    "fig5-three-filters.pdf": REPO / "figsrc" / "out" / "fig5-three-filters.pdf",
    "fig2-crossover.pdf": REPO / "figsrc" / "out" / "fig2-crossover.pdf",
    "fig3-band-budget.pdf": REPO / "figsrc" / "out" / "fig3-band-budget.pdf",
    "fig4-latency-backfire.pdf": REPO / "figsrc" / "out" / "fig4-latency-backfire.pdf",
    "e1-rho-sweep.png": REPO / "pic" / "e1-rho-sweep.png",
    "command-spectrum-annotated.png": REPO / "pic" / "command-spectrum-annotated.png",
    "e3-command-filters.png": REPO / "pic" / "e3-command-filters.png",
    "e4-closed-loop.png": REPO / "pic" / "e4-closed-loop.png",
    "e4-closed-loop-no-torques-energy.png": REPO / "pic" / "e4-closed-loop-no-torques-energy.png",
}


def stage() -> int:
    FIGS.mkdir(exist_ok=True)
    missing = [str(s.relative_to(REPO)) for s in SOURCES.values() if not s.exists()]
    if missing:
        print("error: missing sources. Run the figure scripts first:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("  (python figsrc/fig1_architecture.py, etc.)", file=sys.stderr)
        return 1
    for name, src in SOURCES.items():
        shutil.copy2(src, FIGS / name)
    print(f"staged {len(SOURCES)} figures into figs/")
    return 0


def build() -> int:
    r = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=HERE, capture_output=True, text=True, errors="replace",
    )
    pdf = HERE / "main.pdf"
    if not pdf.exists():
        print("build failed; last lines of latexmk output:", file=sys.stderr)
        print("\n".join(r.stdout.splitlines()[-25:]), file=sys.stderr)
        return 1
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = next((l.split()[1] for l in info.splitlines() if l.startswith("Pages:")), "?")
    size = next((l for l in info.splitlines() if l.startswith("Page size:")), "")
    print(f"wrote main.pdf -- {pages} pages, {size.split(':',1)[-1].strip()}")
    if pages.isdigit() and int(pages) > 8:
        print(f"WARNING: {pages} pages exceeds the ICRA 8-page limit "
              f"(references included). See README.md.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    rc = stage()
    if rc or "--stage" in sys.argv:
        raise SystemExit(rc)
    raise SystemExit(build())
