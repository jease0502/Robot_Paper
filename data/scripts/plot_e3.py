"""Pareto plot for E3: heat spent vs locomotion kept, per steady load.

Up and to the left is better: keep the motion, spend less heat.
The gap between a causal Butterworth and the same filter run zero-phase is
the price of phase lag, and it is the gap a predictive model has to close.
"""
import csv
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

LOGS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")
VARIANT = "baseline"

rows = [r for r in csv.DictReader(open(LOGS / "e3-command-filters.csv", encoding="utf-8"))
        if r["variant"] == VARIANT]
loads = sorted({float(r["load_Nm"]) for r in rows})

STYLE = {
    "zoh": ("#1a202c", "o", "ZOH (shipped today)"),
    "linear": ("#2b6cb0", "s", "linear interpolation"),
    "predict": ("#38a169", "D", "predictive filter (causal)"),
}
CAUSAL = "#c53030"
ZEROPH = "#d97706"

fig, axes = plt.subplots(1, len(loads), figsize=(4.6 * len(loads), 4.3), sharey=True)

for ax, load in zip(axes, loads):
    sel = [r for r in rows if float(r["load_Nm"]) == load]

    for tag, color in (("butter_c", CAUSAL), ("zerophase", ZEROPH)):
        pts = sorted([(float(r["H_rel"]), float(r["motion_kept"]),
                       r["scheme"].split("_")[-1])
                      for r in sel if r["scheme"].startswith(tag)])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                color=color, lw=1.4, alpha=0.6, zorder=1)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42,
                   color=color, marker="^" if tag == "butter_c" else "v",
                   zorder=3,
                   label="Butterworth, causal" if tag == "butter_c"
                   else "same filter, zero-phase (not deployable)")
        for hx, hy, lab in pts:
            ax.annotate(lab + " Hz", (hx, hy), fontsize=6.5,
                        xytext=(3, -8), textcoords="offset points", color=color)

    for name, (color, marker, label) in STYLE.items():
        r = [x for x in sel if x["scheme"] == name]
        if not r:
            continue
        ax.scatter(float(r[0]["H_rel"]), float(r[0]["motion_kept"]), s=80,
                   color=color, marker=marker, zorder=4, label=label,
                   edgecolors="white", linewidths=0.8)

    ax.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax.axvline(1.0, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel("heat spent   H / H(ZOH)")
    ax.set_title("steady joint load %.0f N m" % load, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.08)

axes[0].set_ylabel("locomotion kept (in-band, vs ZOH)")
axes[0].legend(fontsize=7.2, loc="lower right")
axes[0].annotate("better", xy=(0.12, 1.02), xytext=(0.45, 0.86), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=1.1))

fig.suptitle("Filtering the command stream, policy frozen. "
             "Unloaded joints: 46% less heat at 99.5% of the motion. "
             "Loaded joints: nothing to win.", fontsize=10.5)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "e3-command-filters.png", dpi=130)
print("wrote", OUT / "e3-command-filters.png")

for load in loads:
    sel = [r for r in rows if float(r["load_Nm"]) == load]
    zp = [r for r in sel if r["scheme"] == "zerophase_12"][0]
    bc = [r for r in sel if r["scheme"] == "butter_c_12"][0]
    print("load %5.1f N m | zero-phase 12 Hz: H=%.3f kept=%.3f | "
          "causal 12 Hz: H=%.3f kept=%.3f | phase lag costs %.1f pts of motion"
          % (load, float(zp["H_rel"]), float(zp["motion_kept"]),
             float(bc["H_rel"]), float(bc["motion_kept"]),
             100 * (float(zp["motion_kept"]) - float(bc["motion_kept"]))))
