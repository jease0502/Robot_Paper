"""Plot E1 and E1b together.

Left  : unloaded joint -- heat climbs with rho while motion stays flat.
Middle: the closed form H = H0 + kappa * rho/(1-rho).
Right : the correction that matters -- once the joint holds a steady load,
        the same rho buys far less heat, because copper loss is
        H = tau_dc^2 + var(tau) and rho only drives the second term.
"""
import csv
import math
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

LOGS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")

rows = list(csv.DictReader(open(LOGS / "e1-rho-sweep.csv", encoding="utf-8")))
rho = np.array([float(r["rho_measured"]) for r in rows])
heat = np.array([float(r["heat_proxy"]) for r in rows])
motion = np.array([float(r["motion_rms_rad"]) for r in rows])

h0, m0 = heat[0], motion[0]
x = rho / (1 - rho)
kappa = np.polyfit(x[1:], (heat - h0)[1:], 1)[0]
resid = np.abs(h0 + kappa * x - heat) / heat

lrows = list(csv.DictReader(open(LOGS / "e1b-load-sweep.csv", encoding="utf-8")))
loads = sorted({float(r["load_Nm"]) for r in lrows})

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

ax[0].plot(rho, heat / h0, "o-", color="#c53030", lw=1.8, ms=5, label="heat proxy  H/H0")
ax[0].plot(rho, motion / m0, "s-", color="#2b6cb0", lw=1.8, ms=5, label="joint motion  M/M0")
ax[0].set_yscale("log")
ax[0].set_xlabel("out-of-band command fraction  rho")
ax[0].set_ylabel("relative to rho = 0")
ax[0].set_title("Unloaded joint:\nsame movement, much more heat", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

ax[1].plot(x, heat, "o", color="#c53030", ms=6, label="simulated")
ax[1].plot(x, h0 + kappa * x, "-", color="0.3", lw=1.4,
           label="H0 + %.1f rho/(1-rho)" % kappa)
ax[1].set_xlabel("rho / (1 - rho)")
ax[1].set_ylabel("heat proxy  H = mean(tau^2)")
ax[1].set_title("Exactly linear in rho/(1-rho)\n(max residual %.2f%%)"
                % (100 * resid.max()), fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(loads)))
for load, c in zip(loads, colors):
    sel = [r for r in lrows if float(r["load_Nm"]) == load]
    rr = [float(r["rho"]) for r in sel]
    hh = [float(r["H_rel"]) for r in sel]
    ax[2].plot(rr, hh, "o-", color=c, lw=1.8, ms=4,
               label="steady load %.0f N m" % load)
ax[2].set_yscale("log")
ax[2].axhline(1.0, color="0.6", lw=0.8, ls=":")
ax[2].set_xlabel("out-of-band command fraction  rho")
ax[2].set_ylabel("H / H(rho=0)  at the same load")
ax[2].set_title("With a realistic steady load the\nsame roughness costs almost nothing",
                fontsize=10)
ax[2].legend(fontsize=7.5)
ax[2].grid(alpha=0.3, which="both")
ax[2].annotate("stance-phase knee:\n2% effect", xy=(0.20, 1.02), xytext=(0.20, 2.4),
               fontsize=8, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.9))

fig.suptitle("Command roughness is a thermal problem only for lightly loaded joints. "
             "Copper loss is tau_dc^2 + var(tau); rho only drives var(tau).",
             fontsize=10.5)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "e1-rho-sweep.png", dpi=130)
print("wrote", OUT / "e1-rho-sweep.png")
print("kappa = %.3f   max relative residual = %.4f%%" % (kappa, 100 * resid.max()))

# crossover: tau_dc^2 == var(tau)
var_at_02 = float([r for r in lrows
                   if float(r["load_Nm"]) == 0.0 and float(r["rho"]) == 0.20][0]["tau_var"])
print("var(tau) at rho=0.20 is %.2f, so the DC term dominates once "
      "tau_dc > %.2f N m" % (var_at_02, math.sqrt(var_at_02)))
