#!/usr/bin/env python3
"""Assemble the reading page from the template, the hand-authored SVGs and the pic/ PNGs.

Two outputs from one template:

  paper-page.html   body fragment, for the Artifact tool, which supplies its own
                    <!doctype>/<head> wrapper at publish time. Gitignored.
  index.html        standalone document, for GitHub Pages. Needs the doctype and
                    the head the Artifact host would otherwise have provided --
                    without them the page renders in quirks mode and has no
                    viewport meta, so mobile layout breaks.

SVGs are inlined so they inherit the page's theme colours; PNGs are embedded as
data: URIs because the Artifact CSP blocks external images.
"""
import base64
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
TMPL = ROOT / "paper-page.html.tmpl"
OUT_FRAGMENT = ROOT / "paper-page.html"
OUT_STANDALONE = ROOT / "index.html"

# Mirrors the wrapper the Artifact host injects, so the two outputs render alike.
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex, nofollow">
<style>
html{color-scheme:light dark}
body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif}
img{max-width:100%}
[hidden]{display:none!important}
</style>
</head>
<body>
"""
FOOT = "\n</body>\n</html>\n"


def figure(body: str, label: str, caption: str) -> str:
    return (
        '<figure>\n<div class="fbox">\n' + body + "\n</div>\n"
        f"<figcaption><b>{label}</b>{caption}</figcaption>\n</figure>"
    )


def svg_fig(name: str, label: str, caption: str) -> str:
    src = (ROOT / "fig" / name).read_text(encoding="utf-8")
    src = re.sub(r"^<\?xml[^>]*\?>\s*", "", src).strip()
    # let CSS drive the size
    src = re.sub(r'\s(width|height)="\d+"', "", src, count=2)
    return figure(src, label, caption)


def png_fig(name: str, label: str, caption: str) -> str:
    data = (ROOT / "pic" / name).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    alt = re.sub(r"<[^>]+>", "", caption)[:180]
    img = f'<img alt="{alt}" src="data:image/png;base64,{b64}">'
    return figure(img, label, caption)


FIGS = {
    "FIG1": svg_fig(
        "fig1-architecture.svg",
        "Figure 1 — system architecture",
        "Where the command goes, and where the intervention sits. The policy is frozen; the only thing "
        "added is the dashed block. Two details carry the paper: the torque tap feeding the four-band "
        "decomposition, and the dashed feedback path — the policy is itself a feedback controller, so "
        "latency spent inside that loop is subtracted from its stability margin.",
    ),
    "FIG2": svg_fig(
        "fig2-crossover.svg",
        "Figure 2 — the quadrature crossover",
        "Because the two terms add in quadrature, which one dominates is a property of the joint, not of "
        "the policy. The curves are the definition r = Var/(τ̄²+Var), not a fit; the joint placements and "
        "the reachable shares are measured. Eight of Go1's twelve joints sit left of the line.",
    ),
    "FIG3": svg_fig(
        "fig3-band-budget.svg",
        "Figure 3 — the heat budget, two policies",
        "The &gt; 25 Hz sliver is drawn to scale: that is what 1.6% looks like. Removing the effort reward "
        "terms moves 25 percentage points of heat into the band a filter can reach — and the filter returns "
        "about half of whatever is there, on both policies.",
    ),
    "FIG4": svg_fig(
        "fig4-latency-backfire.svg",
        "Figure 4 — latency backfires, and only at speed",
        "The same delayed filter is the best scheme tested at 0.5 m/s and falls in a fifth of episodes at "
        "1.0 m/s. Bars above the ZOH baseline are worse than doing nothing. This is the figure that argues "
        "against evaluating a smoothing filter at the bottom of the speed range.",
    ),
    "FIG_E1": png_fig(
        "e1-rho-sweep.png",
        "Figure E1 — single-joint load sweep",
        "Left: with no load, heat climbs steeply with out-of-band command power while the joint's motion "
        "does not. Middle: the relationship is exactly linear in ρ/(1−ρ), max residual 0.45%. Right: a "
        "realistic steady load flattens it almost completely.",
    ),
    "FIG_SPEC": png_fig(
        "command-spectrum.png",
        "Figure E2 — command spectra across reward ablations",
        "Left: command power spectra; the unregularized policy is flat across the whole band. Right: the "
        "joint's closed-loop response. Note the shaded region here is the 15.3 Hz actuator bandwidth, not "
        "the 5–25 Hz band the decomposition uses — worth reconciling before submission.",
    ),
    "FIG_E3": png_fig(
        "e3-command-filters.png",
        "Figure E3 — open-loop filter sweep",
        "Up and to the left is better. The zero-phase curve dominates the causal one at every cutoff, and "
        "the gap between them is phase lag alone. At 10 N·m of steady load the whole picture collapses to a "
        "vertical line: there is nothing to win.",
    ),
    "FIG_E4": png_fig(
        "e4-closed-loop.png",
        "Figure E4 — closed loop, full-reward policy",
        "Heat against tracking error with the filter inside the loop. Right panel: which joints the gate "
        "lets the filter touch — the four calves sit above the 3.2 N·m line and are passed through untouched.",
    ),
    "FIG_E4B": png_fig(
        "e4-closed-loop-no-torques-energy.png",
        "Figure E5 — closed loop, effort terms removed",
        "The same matrix on the wasteful policy. The predictive filter moves much further left; the delayed "
        "filter moves off the chart to the right.",
    ),
}


def standalone(fragment: str) -> str:
    """Wrap the fragment as a full document, hoisting its title/link/style into <head>.

    The template opens with <title>, the font <link> and the page <style>; those are
    head content that the Artifact host tolerates inline. A real document should carry
    them in <head>, so split at the first </style> and move that prologue up.
    """
    marker = "</style>"
    idx = fragment.find(marker)
    if idx == -1:
        return HEAD + fragment + FOOT
    head_bits, body = fragment[: idx + len(marker)], fragment[idx + len(marker) :]
    return HEAD.replace("</head>", head_bits + "\n</head>") + body.lstrip("\n") + FOOT


def main() -> int:
    html = TMPL.read_text(encoding="utf-8")
    for key, block in FIGS.items():
        token = "{{" + key + "}}"
        if token not in html:
            print(f"warning: {token} not found in template", file=sys.stderr)
            continue
        html = html.replace(token, block)
    leftover = re.findall(r"\{\{[A-Z_0-9]+\}\}", html)
    if leftover:
        print(f"error: unsubstituted placeholders {leftover}", file=sys.stderr)
        return 1

    OUT_FRAGMENT.write_text(html, encoding="utf-8")
    OUT_STANDALONE.write_text(standalone(html), encoding="utf-8")
    for p in (OUT_FRAGMENT, OUT_STANDALONE):
        print(f"wrote {p.name:<20} ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
