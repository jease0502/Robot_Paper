#!/usr/bin/env python3
"""Assemble the paper page from the template, the generated SVGs and the pic/ PNGs.

Two outputs from one template:

  paper-page.html   body fragment, for the Artifact tool, which supplies its own
                    <!doctype>/<head> wrapper at publish time. Gitignored.
  index.html        standalone document, for GitHub Pages. Needs the doctype and
                    the head the Artifact host would otherwise have provided --
                    without them the page renders in quirks mode and has no
                    viewport meta, so mobile layout breaks.

The page is light-only by design: it is typeset as a paper, and a paper has one
appearance. Nothing here emits a dark palette and the wrapper pins
color-scheme:light so the host cannot impose one.

SVGs are inlined (they are vector line art and stay crisp at any width); PNGs are
embedded as data: URIs because the Artifact CSP blocks external images.

Figures are numbered sequentially in document order by this script, not by hand,
so inserting one does not desynchronise the rest.
"""
import base64
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
TMPL = ROOT / "paper-page.html.tmpl"
REFS = ROOT / "references-60.html"
OUT_FRAGMENT = ROOT / "paper-page.html"
OUT_STANDALONE = ROOT / "index.html"

# Mirrors the wrapper the Artifact host injects, so the two outputs render alike.
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="robots" content="noindex, nofollow">
<style>
html{color-scheme:light}
body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif}
img{max-width:100%}
[hidden]{display:none!important}
</style>
</head>
<body>
"""
FOOT = "\n</body>\n</html>\n"


def _figure(body: str, number: int, caption: str) -> str:
    return (
        f'<figure id="fig{number}">\n<div class="fbox">\n' + body + "\n</div>\n"
        f"<figcaption><b>Fig. {number}.</b> {caption}</figcaption>\n</figure>"
    )


def _svg(name: str) -> str:
    src = (ROOT / "fig" / name).read_text(encoding="utf-8")
    src = re.sub(r"^<\?xml[^>]*\?>\s*", "", src).strip()
    src = re.sub(r"<!DOCTYPE[^>]*>\s*", "", src, flags=re.I).strip()
    # Let CSS drive the size. matplotlib writes width="460.8pt", the old
    # hand-authored files wrote width="900" -- strip either form, but only on the
    # root <svg> tag, and only when a viewBox is there to size it instead.
    root_end = src.find(">") + 1
    root, rest = src[:root_end], src[root_end:]
    if "viewBox" not in root:
        raise SystemExit(f"error: fig/{name} has no viewBox; cannot size it with CSS")
    root = re.sub(r'\s(?:width|height)="[^"]*"', "", root)
    return root + rest


def _raster(name: str, caption: str) -> str:
    """Embed a pic/ raster as a data: URI -- the Artifact CSP blocks external images."""
    mime = {"png": "image/png", "gif": "image/gif"}[name.rsplit(".", 1)[-1].lower()]
    data = (ROOT / "pic" / name).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    alt = re.sub(r"<[^>]+>", "", caption)[:180]
    return f'<img alt="{alt}" src="data:{mime};base64,{b64}">'


# Placeholder -> (kind, filename, caption). Figure numbers are assigned below in
# the order the placeholders appear in the template.
FIGS = {
    "FIG1": (
        "svg",
        "fig1-architecture.svg",
        "Where the command goes, and where the intervention sits. The policy is frozen; the only "
        "addition is the dashed block. Two details carry the paper: the torque tap feeding the "
        "four-band decomposition, and the dashed feedback path. The policy is itself a feedback "
        "controller, so latency spent inside that loop is subtracted from its stability margin.",
    ),
    "FIG_FILTERS": (
        "svg",
        "fig5-three-filters.svg",
        "Three ways to smooth a command, and what each one costs. The kernel is the same shape "
        "throughout; only where it sits in time changes. The causal filter's centre of mass sits "
        "behind now, and that gap <em>is</em> the phase lag. Delaying two steps recovers symmetry but "
        "spends 40 ms of latency out of the policy's stability margin. The predictive filter keeps the "
        "kernel centred on now and estimates the half of it that has not happened yet, which is why "
        "its recovery is bounded by predictability rather than by lag.",
    ),
    "FIG_E1": (
        "png",
        "e1-rho-sweep.png",
        "Single-joint load sweep (E1). Left: with no load, heat climbs steeply with out-of-band "
        "command power while the joint's motion does not. Centre: the relationship is exactly linear "
        "in &rho;/(1&minus;&rho;), maximum residual 0.45%. Right: a realistic steady load flattens it "
        "almost completely.",
    ),
    "FIG2": (
        "svg",
        "fig2-crossover.svg",
        "The quadrature crossover. Because the two terms add in quadrature, which one dominates is a "
        "property of the joint, not of the policy. The curves are the definition "
        "r = Var(&tau;)/(&tau;&#772;&sup2; + Var(&tau;)), not a fit; the joint placements and the "
        "reachable shares are measured. Eight of the Go1's twelve joints sit left of the line.",
    ),
    "FIG_SPEC": (
        "png",
        "command-spectrum-annotated.png",
        "Command spectra across reward ablations. Left: command power spectra; the unregularized "
        "policy is flat across the whole band, that is, white. Right: the joint's closed-loop "
        "response. The grey shading in both panels is the 15.3 Hz actuator bandwidth that \\(\\rho\\) "
        "is defined against in Section 3.2. It is <em>not</em> the 5&ndash;25 Hz filter-reachable band "
        "of Section 5.4, which the rule below the axes marks instead; the two are different cuts and "
        "are easy to conflate. The ruler was added by <code>figsrc/annotate_command_spectrum.py</code>, "
        "which measures the axis mapping from the rendered spines and leaves every data pixel "
        "untouched. Regenerating the panel from the source rollouts is "
        "<span class=\"pending\">pending</span>.",
    ),
    "FIG_E3": (
        "png",
        "e3-command-filters.png",
        "Open-loop filter sweep (E2). Up and to the left is better. The zero-phase curve dominates the "
        "causal one at every cutoff, and the gap between them is phase lag alone. At 10 N&middot;m of "
        "steady load the whole picture collapses to a vertical line: there is nothing to win.",
    ),
    "FIG_ANIM": (
        "gif",
        "anim-phase-lag.gif",
        "Phase lag, animated: one joint's command replayed with each scheme in turn. The causal "
        "low-pass trails the command, the delayed zero-phase filter is smooth but late, and the "
        "predictive filter stays on it. This is the same comparison as Fig. 2 with time running. "
        "Web only &mdash; the submission figure set contains no animations.",
    ),
    "FIG_VARIANTS": (
        "gif",
        "anim-variants.gif",
        "Four reward-ablation policies replayed side by side, left to right: full "
        "reward, no <code>action_rate</code>, no torque and energy, task terms only. "
        "All four were recorded under the same command and the same 400 steps, so the "
        "reward is the only difference. Nothing is re-simulated &mdash; these are the "
        "archived <code>qpos</code> trajectories played back on the Go1 model. The "
        "gaits look alike even at a 138% higher command-delta RMS, which is the point: "
        "joint dynamics low-pass whatever the policy emits, so the waste shows up in "
        "winding current rather than in the motion. Web only.",
    ),
    "FIG3": (
        "svg",
        "fig3-band-budget.svg",
        "The heat budget, both policies. The &gt; 25 Hz sliver is drawn to scale: that is what 1.6% "
        "looks like. Removing the effort reward terms moves 25 percentage points of heat into the band "
        "a filter can reach, and the filter returns about half of whatever is there on either policy.",
    ),
    "FIG_E4": (
        "png",
        "e4-closed-loop.png",
        "Closed loop, full-reward policy (E3). Heat against tracking error with the filter inside the "
        "loop. Right panel: which joints the gate lets the filter touch. The four calves sit above the "
        "3.2 N&middot;m line and pass through untouched.",
    ),
    "FIG4": (
        "svg",
        "fig4-latency-backfire.svg",
        "Latency backfires, and only at speed. The same delayed filter is the best scheme tested at "
        "0.5 m/s and falls in a fifth of episodes at 1.0 m/s. Bars above the ZOH baseline are worse "
        "than doing nothing. This is the figure that argues against evaluating a smoothing filter at "
        "the bottom of the speed range.",
    ),
    "FIG_E4B": (
        "png",
        "e4-closed-loop-no-torques-energy.png",
        "Closed loop, effort terms removed. The same matrix on the wasteful policy. The predictive "
        "filter moves much further left; the delayed filter moves off the chart to the right.",
    ),
}


def standalone(fragment: str) -> str:
    """Wrap the fragment as a full document, hoisting its title/link/style into <head>.

    The template opens with <title>, the font <link>s and the page <style>; those
    are head content the Artifact host tolerates inline. A real document should
    carry them in <head>, so split at the first </style> and move that prologue up.
    """
    marker = "</style>"
    idx = fragment.find(marker)
    if idx == -1:
        return HEAD + fragment + FOOT
    head_bits, body = fragment[: idx + len(marker)], fragment[idx + len(marker) :]
    return HEAD.replace("</head>", head_bits + "\n</head>") + body.lstrip("\n") + FOOT


def main() -> int:
    html = TMPL.read_text(encoding="utf-8")

    if "{{REFS}}" not in html:
        print("error: {{REFS}} not found in template", file=sys.stderr)
        return 1
    html = html.replace("{{REFS}}", REFS.read_text(encoding="utf-8").rstrip("\n"))

    # Number the figures in the order their placeholders appear in the template.
    order = [
        m.group(1)
        for m in re.finditer(r"\{\{(FIG[A-Z_0-9]*)\}\}", html)
    ]
    missing = [k for k in FIGS if k not in order]
    if missing:
        print(f"warning: defined but unused: {sorted(missing)}", file=sys.stderr)

    for number, key in enumerate(order, start=1):
        if key not in FIGS:
            print(f"error: {{{{{key}}}}} in template has no definition", file=sys.stderr)
            return 1
        kind, name, caption = FIGS[key]
        src = ROOT / ("fig" if kind == "svg" else "pic") / name
        if not src.exists():
            print(f"error: {{{{{key}}}}} needs {src.relative_to(ROOT)}, which is missing",
                  file=sys.stderr)
            return 1
        body = _svg(name) if kind == "svg" else _raster(name, caption)
        html = html.replace("{{" + key + "}}", _figure(body, number, caption), 1)

    leftover = re.findall(r"\{\{[A-Z_0-9]+\}\}", html)
    if leftover:
        print(f"error: unsubstituted placeholders {leftover}", file=sys.stderr)
        return 1

    if re.search(r"prefers-color-scheme\s*:\s*dark", html):
        print("error: a dark-mode block survived; this page is light-only", file=sys.stderr)
        return 1

    OUT_FRAGMENT.write_text(html, encoding="utf-8")
    OUT_STANDALONE.write_text(standalone(html), encoding="utf-8")
    for p in (OUT_FRAGMENT, OUT_STANDALONE):
        print(f"wrote {p.name:<20} ({p.stat().st_size/1024:.0f} KB)")
    print(f"figures numbered 1..{len(order)} in document order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
