# ICRA submission build

LaTeX source for the ICRA version of the paper. The reading page (`../index.html`)
and `../paper-v2.md` remain the full-length versions; this directory is the
length-constrained conference variant.

```bash
python build_icra.py      # stage figures from ../figsrc/out and ../pic, then build
latexmk -pdf main.tex     # build only, if figures are already staged
```

Current build: **9 pages**, US Letter, clean compile, no overfull boxes.

---

## Venue rules this build targets

Confirmed against the official ICRA 2027 pages on 2026-09-10; full sourcing with
quotes and URLs is in `../icra-requirements.md`. **These rules change between
editions** — ICRA 2025 used a different "6 pages + unlimited references" scheme —
so re-verify before submitting rather than trusting this file.

| Rule | Value | Consequence of getting it wrong |
|---|---|---|
| Edition | ICRA 2027, Seoul, 24–28 May 2027 | — |
| **Submission deadline** | **15 September 2026** | — |
| **Page limit** | **8 pages, references included** | "returned without review" |
| Document class | `\documentclass[letterpaper, 10 pt, conference]{ieeeconf}` | wrong layout; note it is `ieeeconf`, **not** `IEEEtran` |
| Review | **Double-anonymous** | desk reject if names, affiliations, identifying links or acknowledgements appear |
| Generative-AI content | must be disclosed | disqualification / withdrawal, no fee refund |

No extra-page purchase option was found for this edition.

### Two switches in `main.tex`

```latex
\anonymoustrue     % keep true until the paper is accepted
\showJointTablefalse ...   % one switch per length-optional display item
```

`\anonymoustrue` replaces the author block with `Anonymous submission`. Turn it
off only for the camera-ready version.

---

## The length problem, measured

The paper does not fit. This is a content problem, not a formatting one, so it is
recorded here rather than solved silently.

Page counts are measured, not estimated — each configuration was compiled and the
PDF's page count read back:

| Configuration | Pages |
|---|---|
| Every display item shown (11 figures, 9 tables) | **11** |
| Three raw plots off (the first default) | 10 |
| **All eight optional items off (current build)** | **9** |
| All eight off, plus one full-width core figure removed | 8 |
| Float-placement tuning (`[!tb]`, relaxed float fractions) | no change |

Turning items off **one at a time saved nothing**: floats reflow, so a single cut
rarely crosses a page boundary. Only combinations move the count.

### What is currently switched off

All eight are still in the source; each comes back by flipping its switch.

| Switch | Item | What is lost |
|---|---|---|
| `showJointTable` | Per-joint-family band split | The 14–17% / 33–45% figures **the abstract quotes** |
| `showLoadSweep` | E1 sweep figure | The ρ/(1−ρ) linearity and its 0.45% residual |
| `showNyquistTable` | Above-Nyquist term | Evidence that halving the staircase term *raises* total heat |
| `showClosedLoopScatter` | Closed-loop scatter | Heat-vs-tracking trade-off; which joints the gate touches |
| `showVelocityTable` | Per-velocity breakdown | Exact fall rates behind contribution C3 |
| `showSpectra` | Raw command spectra | Covered by Table II |
| `showOpenLoopSweep` | Raw open-loop sweep | Covered by Table III |
| `showSecondPolicyScatter` | Second-policy scatter | Covered by Table IX |

The first five carry evidence that no table replaces. `showJointTable` is the
uncomfortable one: the abstract quotes numbers the reader can no longer check.

### Getting from 9 to 8

Each option was measured. None is free.

1. **Drop one full-width core figure.** Removing Fig. 1 (architecture), Fig. 4
   (band budget) or Fig. 5 (latency) each saves exactly one page. All three are
   contribution-level evidence — Fig. 5 *is* contribution C3.
2. **Cut prose.** The abstract runs ~270 words where 150–200 is typical, and the
   C1–C5 contribution list restates it almost sentence for sentence. Removing the
   duplication is the cheapest page available and costs no evidence.
3. **Cut a section.** §VI Practical Engineering Guidelines partly restates §V.

Option 2 is the recommendation: it is the only one that does not remove evidence.
It requires an authorial decision, so it has not been done here.

---

## Before submitting

- [ ] **Re-verify the venue rules.** They changed between ICRA 2025 and 2027.
- [ ] Get under 8 pages (currently 9).
- [ ] Fill the `\pending` markers — paper ID, and reference [9]'s author list.
- [ ] **Disclose generative-AI assistance.** ICRA 2027 requires it. The figures in
      `figs/` were produced by scripts written with AI assistance, as was much of
      this LaTeX; disclosure goes in the acknowledgements of the camera-ready, not
      the anonymous submission.
- [ ] Decide the 83 Hz Lite3 policy rate (see `../paper-v2.md`, Appendix C). No
      public source supports it and the only published figure is ~50 Hz.
- [ ] §II calls the 6 kHz GO-M8010-6 figure a *commutation* loop; the datasheet
      says *communication control frequency*. Different things.
- [ ] Run the PaperPlaza PDF compliance checker.
- [ ] Confirm no identifying content survives (`build_icra.py` does not check this;
      the scan was run manually and was clean).

## Files

| Path | What it is |
|---|---|
| `main.tex` | The paper. Both switches are at the top of the preamble. |
| `build_icra.py` | Stages figures from `../figsrc/out` and `../pic`, then builds. Warns if the PDF exceeds 8 pages. |
| `figs/` | Staged figures. Generated — do not edit; regenerate from `../figsrc/`. |
| `main.tex.bak-20260910` | Pre-trim source, before the length switches were added. |

`../pic/anim-phase-lag.gif` is not staged: a PDF cannot carry an animation. It
stays on the reading page only.
