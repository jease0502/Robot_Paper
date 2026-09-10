#!/usr/bin/env python
"""Appendix A.2: the reward ablation repeated over five training seeds.

Section 5.2 originally reported one seed. The claim it supports is a comparison
between reward terms, so what matters is whether the ordering survives the
variance between training runs.

Two things are reported, and the second is the one to read:

  absolute   mean +- std of each statistic across seeds
  ratio      the same statistic divided by THAT SEED'S OWN baseline, then
             averaged. The percentages in Table 2 are ratios, and pairing them
             within a seed removes the seed-to-seed spread in the baseline
             itself. It is the correct way to put an interval on "+8%".

Statistics come from analyze_command_spectrum.analyse, unchanged, so the seed-0
column reproduces Table 2 exactly.

    python multiseed_ablation.py --roots ~/handson/runs/reward-ablation \\
        ~/handson/runs/reward-ablation-s1 ... --out ../results
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_command_spectrum import CTRL_DT, analyse  # noqa: E402

ORDER = ["baseline", "no_orientation", "no_feet_air_time", "no_action_rate",
         "no_gait_terms", "no_torques_energy", "task_only"]

PRETTY = {"baseline": "Full reward baseline",
          "no_orientation": "No orientation",
          "no_feet_air_time": "No feet air time",
          "no_action_rate": "No action_rate",
          "no_gait_terms": "No gait terms",
          "no_torques_energy": "No torques + energy",
          "task_only": "Task terms only"}

KEYS = ["delta_rms_rad", "tau_step_rms_Nm", "centroid_Hz", "rho_above_5Hz"]
LABEL = {"delta_rms_rad": "dq_rms (rad)",
         "tau_step_rms_Nm": "tau_step RMS (N m)",
         "centroid_Hz": "centroid (Hz)",
         "rho_above_5Hz": "rho > 5 Hz"}


def collect(roots):
    """-> {variant: {seed: stats}}"""
    out = {}
    for seed, root in enumerate(roots):
        root = pathlib.Path(root).expanduser()
        if not root.exists():
            print("missing, skipped:", root)
            continue
        for d in sorted(root.iterdir()):
            f = d / "rollout.npz"
            if not f.exists():
                continue
            a = np.load(f)["actions"].astype(float)
            out.setdefault(d.name, {})[seed] = analyse(a, CTRL_DT)
    return out


def mean_sd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def pct(row, key):
    """Relative change against the paired baseline, with its interval.
    The third value flags an interval that straddles zero, i.e. an effect that
    cannot be separated from seed noise."""
    if row["variant"] == "baseline":
        return None, None, False
    m = 100 * (row[key + "_ratio_mean"] - 1)
    sd = 100 * row[key + "_ratio_sd"]
    return m, sd, abs(m) < sd


def emit_snippets(rows, out, n_seeds):
    """The same table lives in paper-v2.md, paper-page.html.tmpl and
    icra/main.tex. Emitting all three bodies from one place is the only way to
    stop them drifting apart."""
    K = "tau_step_rms_Nm"
    pm_md, pm_html, pm_tex = "±", "&plusmn;", r"$\pm$"
    md, html, tex = [], [], []

    for r in rows:
        name = PRETTY.get(r["variant"], r["variant"])
        m, sd, noisy = pct(r, K)
        vals = [("delta_rms_rad", 4), (K, 2), ("centroid_Hz", 2),
                ("rho_above_5Hz", 3)]

        def cells(pm):
            return [("%.*f %s %.*f" % (d, r[k + "_mean"], pm, d, r[k + "_sd"]))
                    for k, d in vals]

        if m is None:
            rel_md = rel_html = "—"
            rel_tex = "---"
        else:
            rel_md = "%+.0f%% %s %.0f%s" % (m, pm_md, sd, " ns" if noisy else "")
            rel_html = "%+.0f%% %s %.0f%s" % (m, pm_html, sd,
                                              " ns" if noisy else "")
            rel_tex = "%+.0f\\%% %s %.0f%s" % (
                m, pm_tex, sd, "\\,\\textsuperscript{ns}" if noisy else "")

        md.append("| " + " | ".join([name] + cells(pm_md) + [rel_md]) + " |")

        hname = name.replace("action_rate", "<code>action_rate</code>")
        hcells = "".join("<td>%s</td>" % c for c in cells(pm_html))
        html.append("      <tr><td>%s</td>%s<td>%s</td></tr>"
                    % (hname, hcells, rel_html))

        tname = name.replace("action_rate", "\\texttt{action\\_rate}")
        tex.append(" & ".join([tname] + cells(pm_tex) + [rel_tex]) + r" \\")

    # A narrow variant for the ICRA build. IEEE two-column style gives a
    # \begin{table} about 3.4 inches wide; six columns each carrying a "+-"
    # will overflow it. This keeps the two columns the claim rests on and
    # drops the rest, which the full-length versions still carry.
    narrow = []
    for r in rows:
        name = PRETTY.get(r["variant"], r["variant"]).replace(
            "action_rate", "\\texttt{action\\_rate}")
        m, sd, noisy = pct(r, K)
        rel = "---" if m is None else ("%+.0f\\%% $\\pm$ %.0f%s" % (
            m, sd, "\\,\\textsuperscript{ns}" if noisy else ""))
        narrow.append("%s & %.2f $\\pm$ %.2f & %s & %.3f $\\pm$ %.3f \\\\"
                      % (name, r[K + "_mean"], r[K + "_sd"], rel,
                         r["rho_above_5Hz_mean"], r["rho_above_5Hz_sd"]))

    head = ("# Table 2 body, %d seeds, mean +/- sd across seeds.\n"
            "# The last column is the paired change against each seed's own\n"
            "# baseline. 'ns' marks an interval that straddles zero.\n\n"
            % n_seeds)
    for fname, body in (("table2-markdown.txt", md),
                        ("table2-html.txt", html),
                        ("table2-latex.txt", tex),
                        ("table2-latex-narrow.txt", narrow)):
        (out / fname).write_text(head + "\n".join(body) + "\n",
                                 encoding="utf-8")

    print("\nTable 2 body, %d seeds (markdown):" % n_seeds)
    for line in md:
        print("  " + line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = collect(args.roots)
    if "baseline" not in data:
        sys.exit("no baseline found")
    seeds = sorted(set(data["baseline"]))
    variants = [v for v in ORDER if v in data] + \
               [v for v in sorted(data) if v not in ORDER]
    print("seeds: %s   variants: %d" % (seeds, len(variants)))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- raw, one row per (seed, variant) -----------------------------
    with open(out / "ablation-multiseed-raw.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "variant"] + KEYS)
        for v in variants:
            for s in sorted(data[v]):
                w.writerow([s, v] + ["%.6g" % data[v][s][k] for k in KEYS])

    # ---- summary ------------------------------------------------------
    rows = []
    for v in variants:
        common = [s for s in seeds if s in data[v]]
        row = {"variant": v, "n_seeds": len(common)}
        for k in KEYS:
            m, sd = mean_sd([data[v][s][k] for s in common])
            row[k + "_mean"], row[k + "_sd"] = m, sd
            # paired ratio against the same seed's own baseline
            ratios = [data[v][s][k] / data["baseline"][s][k] for s in common
                      if s in data["baseline"] and data["baseline"][s][k]]
            rm, rsd = mean_sd(ratios)
            row[k + "_ratio_mean"], row[k + "_ratio_sd"] = rm, rsd
        rows.append(row)

    with open(out / "ablation-multiseed-summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % x if isinstance(x, float) else x)
                        for k, x in r.items()})

    # ---- printed tables ----------------------------------------------
    for k in KEYS:
        print("\n%s" % LABEL[k])
        print("  %-22s%18s%20s" % ("variant", "mean +- sd", "vs own baseline"))
        print("  " + "-" * 60)
        for r in rows:
            pct = 100 * (r[k + "_ratio_mean"] - 1)
            pct_sd = 100 * r[k + "_ratio_sd"]
            tail = "" if r["variant"] == "baseline" else \
                "%+7.1f%% +- %.1f" % (pct, pct_sd)
            print("  %-22s%10.4f +- %-6.4f%20s"
                  % (PRETTY.get(r["variant"], r["variant"]),
                     r[k + "_mean"], r[k + "_sd"], tail))

    emit_snippets(rows, out, len(seeds))

    # ---- the claim the section actually makes -------------------------
    print("\nThe ordering Section 5.2 asserts, per seed, on tau_step RMS:")
    ok = 0
    for s in seeds:
        try:
            ar = data["no_action_rate"][s]["tau_step_rms_Nm"]
            te = data["no_torques_energy"][s]["tau_step_rms_Nm"]
            base = data["baseline"][s]["tau_step_rms_Nm"]
        except KeyError:
            continue
        holds = te > ar > base
        ok += holds
        print("  seed %d: baseline %.2f < no_action_rate %.2f < "
              "no_torques_energy %.2f   %s"
              % (s, base, ar, te, "holds" if holds else "DOES NOT HOLD"))
    print("  holds in %d of %d seeds" % (ok, len(seeds)))
    print("\nwrote", out / "ablation-multiseed-summary.csv")


if __name__ == "__main__":
    main()
