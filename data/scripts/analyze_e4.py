"""Analyse and plot the closed-loop filtering experiment.

Reads logs/e4-closed-loop-<tag>.json and produces
  logs/e4-summary-<tag>.csv    per-scheme aggregates with effect sizes
  figs/e4-closed-loop.png      the Pareto plot and the per-joint breakdown
"""
import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

JOINT_NAMES = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
               "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]


def cliffs_delta(a, b):
    """Non-parametric effect size. +1 means a is entirely above b."""
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum(int(x > y) for x in a for y in b)
    lt = sum(int(x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def main():
    logs = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    figs = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    tag = sys.argv[3] if len(sys.argv) > 3 else "baseline"

    blob = json.load(open(logs / ("e4-closed-loop-%s.json" % tag), encoding="utf-8"))
    rows = blob["rows"]
    schemes = []
    for r in rows:
        if r["scheme"] not in schemes:
            schemes.append(r["scheme"])

    base = [r for r in rows if r["scheme"] == "zoh"]
    base_H = np.mean([r["H"] for r in base])
    base_V = np.mean([r["vel_err"] for r in base])

    out = []
    for s in schemes:
        sel = [r for r in rows if r["scheme"] == s]
        H = np.array([r["H"] for r in sel])
        V = np.array([r["vel_err"] for r in sel])
        A = np.array([r["steps_alive"] for r in sel])
        F = np.array([float(r["fell"]) for r in sel])
        R = np.array([r["ripple_frac"] for r in sel])
        out.append(dict(
            scheme=s, n=len(sel),
            H=H.mean(), H_sd=H.std(), H_rel=H.mean() / base_H,
            vel_err=V.mean(), vel_sd=V.std(), vel_rel=V.mean() / base_V,
            ripple_frac=R.mean(),
            steps_alive=A.mean(), fall_rate=F.mean(),
            d_H=cliffs_delta(H, np.array([r["H"] for r in base])),
            d_vel=cliffs_delta(V, np.array([r["vel_err"] for r in base])),
        ))

    with open(logs / ("e4-summary-%s.csv" % tag), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})

    hdr = ("%-16s%9s%9s%10s%9s%9s%9s%8s" %
           ("scheme", "H", "H/H0", "vel_err", "vel/v0", "d(H)", "d(vel)", "fall"))
    print("policy = %s   command predictor R2 = %.3f   n = %d per scheme"
          % (tag, blob["pred_r2"], out[0]["n"]))
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        print("%-16s%9.1f%9.3f%10.4f%9.3f%9.2f%9.2f%8.2f"
              % (r["scheme"], r["H"], r["H_rel"], r["vel_err"], r["vel_rel"],
                 r["d_H"], r["d_vel"], r["fall_rate"]))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6),
                           gridspec_kw={"width_ratios": [1.15, 1.0]})

    style = {
        "zoh": ("#1a202c", "o", 95, "ZOH (shipped today)"),
        "linear": ("#2b6cb0", "s", 70, "linear interpolation"),
        "delay_sym": ("#805ad5", "v", 85, "zero-phase via 40 ms delay"),
        "predict": ("#38a169", "D", 95, "predictive filter (ours)"),
        "gated": ("#2f855a", "P", 95, "predictive + per-joint gate (ours)"),
        "predict|raw": ("#68d391", "d", 70, "predictive, policy unaware"),
    }
    butter = sorted([r for r in out if r["scheme"].startswith("butter")],
                    key=lambda r: r["H_rel"])
    ax[0].plot([r["H_rel"] for r in butter], [r["vel_err"] for r in butter],
               "-", color="#c53030", lw=1.4, alpha=0.6, zorder=1)
    ax[0].scatter([r["H_rel"] for r in butter], [r["vel_err"] for r in butter],
                  s=70, marker="^", color="#c53030", zorder=3,
                  label="causal Butterworth (cutoff sweep)")
    for r in butter:
        ax[0].annotate(r["scheme"].replace("butter_lin_", "lin").replace("butter_", "")
                       + " Hz", (r["H_rel"], r["vel_err"]), fontsize=6.5,
                       xytext=(4, 3), textcoords="offset points", color="#c53030")
    for r in out:
        if r["scheme"] in style:
            c, m, sz, lab = style[r["scheme"]]
            ax[0].scatter(r["H_rel"], r["vel_err"], s=sz, marker=m, color=c,
                          zorder=4, label=lab, edgecolors="white", linewidths=0.8)
    ax[0].axhline(base_V, color="0.7", lw=0.8, ls=":")
    ax[0].axvline(1.0, color="0.7", lw=0.8, ls=":")
    ax[0].set_xlabel("heat spent   H / H(ZOH)")
    ax[0].set_ylabel("velocity tracking error (m/s)")
    ax[0].set_title("Closed loop: heat vs tracking, policy frozen", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].grid(alpha=0.25)
    ax[0].annotate("better", xy=(0.86, base_V * 0.92),
                   xytext=(0.95, base_V * 1.35), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=1.1))

    dc = np.asarray(blob["tau_dc_zoh"])
    gate = np.asarray(blob["gate"])
    order = np.argsort(dc)
    colors = ["#38a169" if gate[i] > 0.5 else "#a0aec0" for i in order]
    ax[1].barh(range(len(dc)), dc[order], color=colors)
    ax[1].axvline(blob["gate_tau"], color="#c53030", ls="--", lw=1.2)
    ax[1].set_yticks(range(len(dc)))
    ax[1].set_yticklabels([JOINT_NAMES[i] for i in order], fontsize=7.5)
    ax[1].set_xlabel("steady torque  mean |tau|  (N m)")
    ax[1].set_title("Which joints the filter is allowed to touch\n"
                    "(green = below the %.1f N m crossover)" % blob["gate_tau"],
                    fontsize=10)
    ax[1].grid(alpha=0.25, axis="x")

    fig.suptitle("Filtering the command stream on a frozen Go1 policy, 50 Hz policy "
                 "into a 500 Hz PD loop", fontsize=11)
    fig.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "e4-closed-loop.png", dpi=130)
    print("\nwrote", figs / "e4-closed-loop.png")


if __name__ == "__main__":
    main()
