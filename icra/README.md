# ICRA submission build

LaTeX source for the ICRA version of the paper. The reading page (`../index.html`)
and `../paper-v2.md` remain the full-length versions; this directory is the
length-constrained conference variant.

```bash
python build_icra.py      # stage figures from ../figsrc/out and ../pic, then build
latexmk -pdf main.tex     # build only, if figures are already staged
```

Current build: **8 pages**, US Letter, clean compile. Zero errors, zero overfull
boxes, zero undefined references.

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
| All eight optional items off | 9 |
| Float-placement tuning (`[!tb]`, relaxed float fractions) | no change |
| `microtype`, tightened float spacing, `\small` tables, compressed references | 9 |
| **Above, plus the prose restructure below (current build)** | **8** |

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

### How the last page came out — no evidence was removed

The paper reached 8 pages by restructuring prose, not by deleting results. Every
figure and table that was in the 9-page build is still in the 8-page build.

- **§ Practical Engineering Guidelines** stopped being a top-level section and
  became § V-G, one paragraph with the four rules run in rather than four
  headed paragraphs.
- **Related Work** went from four themed paragraphs to two, with grouped
  citations — the reference numbers now render as [9]–[11], [12]–[17],
  [18]–[21], [22]–[24]. No reference was dropped.
- **§ Limitations and Future Work** stopped being a section and became the second
  paragraph of the Conclusion, condensed but with nothing omitted.
- **Equations (1) and (2)** are inline. Neither was referenced by number, so
  nothing points at a number that no longer exists.

Two defects were also repaired, both introduced here rather than inherited:
Tables 5, 6 and 8 were switched off while the prose still cited them, so they
rendered as "Table ??" — the citing text is now inside the same switch as the
float. And five tabulars ran past the column edge, by up to 114 pt, which had
sheared the "Deployable" column off Table III; `\small` and a smaller
`\tabcolsep` bring them inside the column.

---

## Before submitting

- [ ] **Re-verify the venue rules.** They changed between ICRA 2025 and 2027.
- [x] ~~Get under 8 pages.~~ Done: 8 pages, verified by rebuild.
- [x] ~~Fill the `\pending` markers.~~ None render in the PDF. The three left in
      the source are the macro definition, the inactive non-anonymous author
      branch, and the English word "pending" in a sentence about delayed MDPs.
- [ ] At submission, put the PaperPlaza-assigned paper ID back in the author block.
- [x] ~~Decide whether arXiv:2312.17507 is the IEEE RA-M paper.~~ The author
      confirmed they are the same work; the entry now cites IEEE Robot. Autom.
      Mag., vol. 32, no. 2, pp. 49-59, 2025 (DOI 10.1109/mra.2024.3487322).
- [x] ~~**Disclose generative-AI assistance.**~~ Done, in an Acknowledgment
      section that is in the anonymous submission. The two rules do not actually
      conflict: ICRA 2027 says AI content "must be disclosed in the
      acknowledgments section", while the RAS double-anonymous rule withholds only
      acknowledgments "to people or funding agencies". A disclosure naming neither
      satisfies both. **Check the wording matches what you actually used AI for**
      before submitting, and add the human acknowledgments after acceptance.
- [x] ~~Decide the 83 Hz Lite3 policy rate.~~ Removed. No public source gave that
      figure and the only published rate found was ~50 Hz, which contradicts it.
      The 1 kHz state loop next to it is separately sourced and stays.
- [x] ~~§II called the 6 kHz figure a *commutation* loop.~~ Reworded to
      *communication control frequency*, which is the datasheet's own field name.
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
