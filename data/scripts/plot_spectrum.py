"""Plot the policy command spectrum against the joint's closed-loop response.

Labels are English because the WSL matplotlib install carries no CJK font
(same convention as the other handson figures).

Input : ~/handson/runs/reward-ablation/<variant>/rollout.npz  (from lesson 4)
Output: figs/command-spectrum.png
"""
import math
import os
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

# Go1 actuator parameters, verbatim from the MuJoCo Playground model.
KP = 35.0          # position actuator gainprm
DAMPING = 0.5      # dof_damping
ARMATURE = 0.005   # dof_armature  [kg m^2]
ACTION_SCALE = 0.5

WN = math.sqrt(KP / ARMATURE)
FN = WN / (2 * math.pi)
ZETA = DAMPING / (2 * math.sqrt(KP * ARMATURE))
_K = 1 - 2 * ZETA ** 2
F_BW = FN * math.sqrt(_K + math.sqrt(_K ** 2 + 1))

RUNS = pathlib.Path(os.environ.get(
    "ROLLOUT_ROOT",
    pathlib.Path.home() / "handson" / "runs" / "reward-ablation"))
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

SHOW = [
    ("baseline", "#2b6cb0", "baseline (full reward)"),
    ("no_torques_energy", "#d97706", "no torques+energy"),
    ("task_only", "#c53030", "task_only (no regularisation)"),
]


def spectrum(actions, dt):
    n = actions.shape[0]
    q = ACTION_SCALE * actions
    win = np.hanning(n)
    x = np.fft.rfft((q - q.mean(0)) * win[:, None], axis=0)
    freq = np.fft.rfftfreq(n, dt)
    power = (np.abs(x) ** 2).sum(1)
    power[0] = 0.0
    return freq, power / power.sum()


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    for name, color, label in SHOW:
        data = np.load(RUNS / name / "rollout.npz")
        freq, power = spectrum(np.asarray(data["actions"], float), float(data["dt"]))
        ax[0].semilogy(freq, np.maximum(power, 1e-9), color=color, lw=1.4, label=label)

    ax[0].axvspan(F_BW, 25, color="0.85", zorder=0)
    ax[0].axvline(F_BW, color="k", ls="--", lw=1)
    ax[0].annotate("actuator -3 dB bandwidth\n%.1f Hz" % F_BW,
                   xy=(F_BW, 3e-2), xytext=(F_BW + 1.0, 8e-2), fontsize=8, va="top")
    ax[0].annotate("out of band\n(heat only)", xy=(20.5, 2e-6),
                   fontsize=8, ha="center", color="0.35")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("normalised command power")
    ax[0].set_title("Power spectrum of the 50 Hz policy command (Go1)", fontsize=10)
    ax[0].set_xlim(0, 25)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ff = np.linspace(0.1, 25, 500)
    mag = 1 / np.sqrt((1 - (ff / FN) ** 2) ** 2 + (2 * ZETA * ff / FN) ** 2)
    ax[1].semilogy(ff, mag, color="#1a202c", lw=1.8)
    ax[1].axhline(1 / math.sqrt(2), color="0.5", ls=":", lw=1)
    ax[1].axvline(F_BW, color="k", ls="--", lw=1)
    ax[1].axvspan(F_BW, 25, color="0.85", zorder=0)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("|H(f)|   command -> motion")
    ax[1].set_title("Joint closed-loop response   fn=%.1f Hz   zeta=%.2f" % (FN, ZETA),
                    fontsize=10)
    ax[1].set_xlim(0, 25)
    ax[1].grid(alpha=0.3)
    ax[1].annotate("commands arrive but\nthe joint cannot follow",
                   xy=(F_BW + 1.0, 0.12), fontsize=8)

    fig.suptitle("A 50 Hz command stream into a 13 Hz joint: "
                 "power in the grey band can only become heat", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "command-spectrum.png", dpi=130)
    print("wrote", OUT / "command-spectrum.png")


if __name__ == "__main__":
    main()
