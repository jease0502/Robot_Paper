#!/usr/bin/env python
"""Explanatory figures for the paper. These are schematics, not measurements.

Measured results live in handson/10-actuator-thermal/figs/. These go to
docs/figs/ because they belong to the paper's argument rather than to the
experiment, and because an animated GIF cannot live under handson/ (the repo
gitignores handson/**/*.gif).

    python plot_schematics.py <rollout.npz> <out_dir>

Produces
    fig-pipeline-and-budget.png   where the command goes, and where the heat is
    fig-three-filters.png         causal vs delayed vs predictive, the core idea
    fig-crossover.png             why the answer is per joint
    anim-phase-lag.gif            the same three filters, running
"""
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

INK = "#1a202c"
MUTED = "#a0aec0"
RAW = "#2b6cb0"
CAUSAL = "#c53030"
DELAY = "#805ad5"
PRED = "#2f855a"

# measured band split, ZOH baseline, full-reward policy (e5-torque-bands)
BANDS = [("DC\nposture", 31.3, "#4a5568"),
         ("0-5 Hz\ngait", 50.6, "#718096"),
         ("5-25 Hz\nin-band jitter", 16.5, "#dd6b20"),
         (">25 Hz\nZOH harmonics", 1.6, "#c53030")]


# --------------------------------------------------------------- filters

def moving_kernel(half):
    k = np.hanning(2 * half + 3)[1:-1]
    return k / k.sum()


def fit_ar(x, order=8):
    n = len(x)
    rows = np.stack([x[i - order:i][::-1] for i in range(order, n)])
    w, *_ = np.linalg.lstsq(rows, x[order:], rcond=None)
    return w


def causal_kernel(half, tau=1.5):
    """What a causal low-pass actually looks like: heaviest on the newest
    sample, decaying into the past. Reversing a symmetric window would not do,
    because a symmetric window reversed is itself."""
    k = np.exp(-np.arange(2 * half + 1) / tau)[::-1]
    return k / k.sum()


def apply_filters(x, half=2, order=8):
    """Returns causal, delayed-symmetric and predictive outputs."""
    kern = moving_kernel(half)
    w = fit_ar(x, order)
    n = len(x)
    causal = np.zeros(n)
    delayed = np.zeros(n)
    predictive = np.zeros(n)
    ck = causal_kernel(half)
    for i in range(n):
        past = x[max(0, i - 2 * half):i + 1]
        if len(past) < 2 * half + 1:
            past = np.concatenate([np.full(2 * half + 1 - len(past), x[0]), past])
        causal[i] = float(np.dot(ck, past))
        delayed[i] = float(np.dot(kern, past))
        buf = list(x[max(0, i - order + 1):i + 1])
        while len(buf) < order:
            buf.insert(0, x[0])
        fut = []
        for _ in range(half):
            fut.append(float(np.dot(w, np.asarray(buf[-order:])[::-1])))
            buf.append(fut[-1])
        window = np.concatenate([x[max(0, i - half):i + 1], fut])
        if len(window) < 2 * half + 1:
            window = np.concatenate(
                [np.full(2 * half + 1 - len(window), x[0]), window])
        predictive[i] = float(np.dot(kern, window))
    return causal, delayed, predictive


# --------------------------------------------------------------- figure A

def fig_pipeline(out):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.6, 5.4),
                                 gridspec_kw={"height_ratios": [1.25, 1]})
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    boxes = [(0.15, "frozen policy", "50 Hz", "#e2e8f0"),
             (2.35, "command filter", "50 Hz", "#c6f6d5"),
             (4.55, "upsample\n(ZOH or interp)", "50 to 500 Hz", "#e2e8f0"),
             (6.75, "PD on the\nmotor driver", "500 Hz", "#e2e8f0"),
             (8.95, "motor\nwinding", "6 kHz current loop", "#fed7d7")]
    for x, label, rate, colour in boxes:
        edge = PRED if colour == "#c6f6d5" else MUTED
        lw = 2.0 if colour == "#c6f6d5" else 1.0
        bx_ = FancyBboxPatch((x, 1.0), 1.5, 0.85, boxstyle="round,pad=0.06",
                             fc=colour, ec=edge, lw=lw)
        ax.add_patch(bx_)
        ax.text(x + 0.75, 1.52, label, ha="center", va="center", fontsize=8.6,
                color=INK, weight="bold" if colour == "#c6f6d5" else "normal")
        ax.text(x + 0.75, 1.14, rate, ha="center", va="center", fontsize=7.2,
                color="#4a5568", style="italic")
        if x < 8.9:
            ax.add_patch(FancyArrowPatch((x + 1.52, 1.42), (x + 2.13, 1.42),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#4a5568", lw=1.1))

    ax.text(3.1, 2.18, "the only thing this paper changes", fontsize=8.4,
            color=PRED, weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((3.1, 2.06), (3.1, 1.9), arrowstyle="-|>",
                                 mutation_scale=10, color=PRED, lw=1.2))
    ax.text(0.15, 0.62, "Policy weights, PD gains, and the driver are untouched. "
                        "A filter installed here applies to any checkpoint,\n"
                        "including one you cannot retrain.",
            fontsize=8.2, color="#4a5568", va="top")

    # ---- the budget bar
    left = 0.0
    for label, pct, colour in BANDS:
        bx.barh(0, pct, left=left, height=0.52, color=colour, edgecolor="white")
        if pct > 4:
            bx.text(left + pct / 2, 0, "%.1f%%" % pct, ha="center", va="center",
                    color="white", fontsize=8.6, weight="bold")
            bx.text(left + pct / 2, -0.42, label, ha="center", va="top",
                    fontsize=7.8, color=INK)
        else:
            bx.annotate("%s\n%.1f%%" % (label, pct),
                        xy=(left + pct / 2, 0.27), xytext=(left + pct / 2 - 1, 0.78),
                        fontsize=7.6, color=CAUSAL, ha="center",
                        arrowprops=dict(arrowstyle="-", color=CAUSAL, lw=0.9))
        left += pct

    reach = BANDS[2][1] + BANDS[3][1]
    x0 = BANDS[0][1] + BANDS[1][1]
    bx.plot([x0, x0, 100, 100], [-0.86, -1.00, -1.00, -0.86], color=PRED, lw=1.6)
    bx.text((x0 + 100) / 2, -1.10, "reachable by a command filter: %.1f%%\n"
            "measured recovery: 8.5%%, about half" % reach,
            ha="center", va="top", fontsize=8.4, color=PRED, weight="bold")

    bx.set_xlim(-1, 101)
    bx.set_ylim(-1.75, 1.15)
    bx.axis("off")
    bx.set_title("Where the heat is, on a walking Go1 under the shipped "
                 "zero-order hold", fontsize=9.6, color=INK, pad=2)

    fig.suptitle("The command path, and the fraction of actuator heat a filter "
                 "can act on", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "fig-pipeline-and-budget.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- figure B

def fig_three_filters(out):
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.0), sharex=True)
    half = 2
    sym = moving_kernel(half)
    taps = np.arange(-2 * half, half + 1)

    # a causal low-pass weights the recent past most heavily and decays
    decay = np.exp(-np.arange(2 * half + 1) / 1.5)[::-1]
    decay = decay / decay.sum()
    com = float(np.dot(np.arange(-2 * half, 1), decay))   # centre of mass

    rows = [
        ("Causal low-pass", CAUSAL, np.arange(-2 * half, 1), decay, com,
         "weights decay into the past. The centre of mass sits behind now,",
         "and that is exactly the phase lag.",
         "The centre of mass sits about a step behind. "
         "Costs 25 points of motion (Section 5.3)."),
        ("Zero phase by delaying", DELAY, np.arange(-2 * half, 1), sym, -half,
         "symmetric weights, so no phase distortion, but they are centred",
         "on a sample that is already two steps old.",
         "40 ms of added latency. Falls at speed, up to every "
         "episode (Section 5.5)."),
        ("Predictive, this paper", PRED, np.arange(-half, half + 1), sym, 0,
         "symmetric weights centred on now. The right half does not exist yet,",
         "so a small per-joint model estimates it.",
         "No lag and no latency. Bounded by how predictable the command is."),
    ]

    for ax, (name, colour, pos, wts, out_at, sub1, sub2, cost) in zip(axes, rows):
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        for t in taps:
            ax.plot([t, t], [0, 0.02], color=MUTED, lw=0.8, zorder=1)
        # heights are scaled to a common maximum: the figure is about where
        # the weights sit, not how large they are. All three sum to one.
        shown = 0.32 * np.asarray(wts) / float(np.max(wts))
        for t, w in zip(pos, shown):
            predicted = t > 0
            ax.bar(t, w, width=0.44, color="white" if predicted else colour,
                   edgecolor=colour, lw=1.5, hatch="///" if predicted else None,
                   zorder=3)
        ax.axvline(0, color=INK, lw=1.0, ls=":", zorder=2)
        ax.text(0, 0.365, "now", ha="center", fontsize=8.4, color=INK)

        ax.scatter([out_at], [-0.055], marker="^", s=105, color=colour,
                   zorder=4, clip_on=False)
        ax.annotate("", xy=(out_at, -0.055), xytext=(0, -0.055),
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1.2,
                                    ls=(0, (2, 2))))
        ax.text(out_at, -0.115, "the output describes\nthis instant",
                ha="center", va="top", fontsize=7.8, color=colour)

        ax.text(-5.45, 0.575, name, fontsize=11, weight="bold", color=colour)
        ax.text(-5.45, 0.480, sub1, fontsize=8.6, color="#4a5568")
        ax.text(-5.45, 0.405, sub2, fontsize=8.6, color="#4a5568")
        ax.text(-5.45, -0.335, cost, fontsize=8.6, color=INK, weight="bold")

        ax.set_xlim(-5.6, 3.4)
        ax.set_ylim(-0.44, 0.66)
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    axes[-1].set_xticks(taps)
    axes[-1].set_xticklabels(["t%+d" % t if t else "t" for t in taps])
    axes[-1].set_xlabel("policy steps, 20 ms apart", fontsize=9.2)
    axes[0].bar(np.nan, np.nan, color="white", edgecolor=INK, hatch="///",
                label="sample that has not happened yet")
    axes[0].legend(fontsize=8.2, loc="upper right", frameon=False,
                   bbox_to_anchor=(1.0, 1.16))

    fig.suptitle("Three ways to smooth a command, and what each one costs\n"
                 "The kernel is the same shape throughout. Only where it sits "
                 "in time changes.", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.905])
    fig.savefig(out / "fig-three-filters.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- figure C

def fig_crossover(out):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    tau = np.linspace(0, 9, 400)
    var = 10.3                      # measured at rho = 0.2 in E1
    posture = tau ** 2
    ax.plot(tau, posture, color="#4a5568", lw=2.2,
            label=r"posture term  $\bar{\tau}^2$")
    ax.axhline(var, color=CAUSAL, lw=2.2, ls="--",
               label=r"ripple term  Var($\tau$), fixed jitter")
    ax.axvline(3.2, color=INK, lw=1.1, ls=":")
    ax.scatter([3.2], [var], s=70, color=INK, zorder=5)
    ax.annotate("crossover, 3.2 N m", xy=(3.2, var), xytext=(4.0, 26),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.0))

    groups = [("hips", 1.24, 1.48, "#2f855a", 45),
              ("thighs", 2.00, 2.59, "#dd6b20", 32),
              ("calves", 5.88, 6.88, "#c53030", 17)]
    for name, lo, hi, colour, reach in groups:
        ax.axvspan(lo, hi, color=colour, alpha=0.16)
        ax.text((lo + hi) / 2, 74, name, ha="center", fontsize=9,
                color=colour, weight="bold")
        ax.text((lo + hi) / 2, 68, "%d%%" % reach, ha="center", fontsize=8.2,
                color=colour)
    ax.text(0.15, 62, "share of that joint's heat\na filter can reach",
            fontsize=7.8, color="#4a5568")

    ax.text(1.0, 40, "ripple dominates\nfiltering pays", fontsize=9,
            color="#2f855a", ha="center")
    ax.text(7.0, 40, "posture dominates\nnothing to win", fontsize=9,
            color="#c53030", ha="center")

    ax.set_xlabel(r"joint steady torque  $\bar{\tau}$  (N m)")
    ax.set_ylabel(r"contribution to $\mathbb{E}[\tau^2]$")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 82)
    ax.legend(fontsize=8.6, loc="upper left", bbox_to_anchor=(0.02, 0.66))
    ax.grid(alpha=0.22)
    ax.set_title("Whether command jitter matters is a property of the joint, "
                 "not of the policy", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig-crossover.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- animation

def anim_phase_lag(rollout, out, joint=1, n=150, win=34, stride=2):
    """A scrolling window, so the phase relationship is actually visible."""
    d = np.load(rollout)
    x = 0.5 * np.asarray(d["actions"], float)[:, joint][:n]
    causal, delayed, predictive = apply_filters(x)
    t = np.arange(len(x))

    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=82)
    ax.plot(t, x, color=MUTED, lw=0.9, zorder=1)
    ax.scatter(t, x, s=6, color=MUTED, zorder=1)
    lr, = ax.plot([], [], color=RAW, lw=1.7, label="policy command", zorder=3)
    lc, = ax.plot([], [], color=CAUSAL, lw=1.8, label="causal low-pass", zorder=4)
    ld, = ax.plot([], [], color=DELAY, lw=1.8, label="zero phase by delay",
                  zorder=4)
    lp, = ax.plot([], [], color=PRED, lw=2.1, label="predictive (ours)", zorder=5)
    dots = ax.scatter([], [], s=[], c=[], zorder=6)

    pad = 0.16 * float(x.max() - x.min())
    ax.set_ylim(float(x.min()) - pad, float(x.max()) + pad)
    ax.set_xlabel("policy step, 20 ms each", fontsize=8.4)
    ax.set_ylabel("joint target (rad)", fontsize=8.4)
    ax.legend(fontsize=7.2, ncol=4, loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.text(0.5, 0.955, "Red trails the command. Purple is smooth but late. "
             "Green stays on it.", ha="center", fontsize=9.0, weight="bold")

    def update(i):
        k = i * stride + 8
        lo = max(0, k - win)
        sl = slice(lo, k)
        lr.set_data(t[sl], x[sl])
        lc.set_data(t[sl], causal[sl])
        ld.set_data(t[sl], delayed[sl])
        lp.set_data(t[sl], predictive[sl])
        j = k - 1
        dots.set_offsets(np.c_[[j, j, j], [causal[j], delayed[j], predictive[j]]])
        dots.set_sizes([34, 34, 42])
        dots.set_color([CAUSAL, DELAY, PRED])
        ax.set_xlim(lo - 0.5, lo + win + 0.5)
        return lr, lc, ld, lp, dots

    anim = FuncAnimation(fig, update,
                         frames=(len(x) - 8) // stride, blit=False)
    path = out / "anim-phase-lag.gif"
    anim.save(path, writer=PillowWriter(fps=9))
    plt.close(fig)
    shrink_gif(path)
    return path


def shrink_gif(path, colours=48):
    """A line plot needs nothing like 256 colours. Requantising the frames
    cuts the file by roughly two thirds with no visible change."""
    from PIL import Image
    im = Image.open(path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert("RGB").quantize(
            colors=colours, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=im.info.get("duration", 90), loop=0,
                   optimize=True, disposal=2)


def main():
    rollout = pathlib.Path(sys.argv[1]).expanduser()
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    fig_pipeline(out)
    fig_three_filters(out)
    fig_crossover(out)
    p = anim_phase_lag(rollout, out)
    for f in sorted(out.iterdir()):
        print("%-32s %7.0f KB" % (f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()
