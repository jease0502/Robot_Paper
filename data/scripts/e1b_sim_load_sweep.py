"""E1b: the same rho sweep, but with a realistic steady load torque.

Why this exists
---------------
e1_sim_rho_sweep.py ran an UNLOADED joint. There, the in-band tracking error
is tiny, so essentially all of the torque comes from out-of-band content and
the heat proxy explodes with rho (160x at rho=0.8). That is a real effect but
it is not a realistic operating point: a leg joint on a standing robot must
hold a large steady torque against gravity.

Copper loss splits as

    H = mean(tau^2) = tau_dc^2 + var(tau)

and rho only drives the SECOND term. Once tau_dc is large, a given amount of
ripple buys proportionally much less heat. This script measures where the
crossover sits, because that is what decides whether command smoothness is a
first-order or a second-order thermal effect.

The load is applied as a constant external torque the joint must hold, which
is what a stance-phase knee or a balancing ankle actually does.
"""
import csv
import math
import pathlib
import sys

import numpy as np

KP = 35.0
B = 0.5
J = 0.005

POLICY_HZ = 50.0
INTEGRATE_HZ = 5000.0
F0 = 2.0
AMPLITUDE = 0.20
DURATION = 20.0

RHOS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
# N m held steadily. 0 = the unloaded case from e1; 20 N m is roughly a Go1
# knee carrying its share of a 12 kg body in a crouch.
LOADS = [0.0, 2.0, 5.0, 10.0, 20.0]
SEEDS = [0, 1, 2]


def bandwidth():
    wn = math.sqrt(KP / J)
    zeta = B / (2 * math.sqrt(KP * J))
    k = 1 - 2 * zeta ** 2
    return (wn / (2 * math.pi)) * math.sqrt(k + math.sqrt(k ** 2 + 1))


def make_command(rho, f_lo, rng, n, dt):
    t = np.arange(n) * dt
    base = AMPLITUDE * np.sin(2 * math.pi * F0 * t)
    if rho <= 0:
        return base
    freq = np.fft.rfftfreq(n, dt)
    phase = rng.uniform(0, 2 * math.pi, size=freq.shape)
    noise = np.fft.irfft(np.where(freq >= f_lo, 1.0, 0.0) * np.exp(1j * phase), n=n)
    noise /= np.std(noise)
    p_noise = rho / (1.0 - rho) * np.sum(base ** 2)
    return base + math.sqrt(p_noise / np.sum(noise ** 2)) * noise


def simulate(cmd, tau_load):
    dt = 1.0 / INTEGRATE_HZ
    per = int(round(INTEGRATE_HZ / POLICY_HZ))
    n = len(cmd) * per
    q = qd = 0.0
    tau_hist = np.empty(n)
    for i in range(n):
        tau = KP * (cmd[i // per] - q)
        qd += ((tau - B * qd - tau_load) / J) * dt
        q += qd * dt
        tau_hist[i] = tau
    return tau_hist


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    f_bw = bandwidth()
    f_lo = max(f_bw, 16.0)
    dt = 1.0 / POLICY_HZ
    n = int(DURATION * POLICY_HZ)
    k0 = int(INTEGRATE_HZ * 1.0)

    print("Go1 joint, -3 dB bandwidth %.2f Hz, out-of-band noise in [%.0f, 25] Hz"
          % (f_bw, f_lo))
    print("Heat proxy H = mean(tau^2).  H is normalised per load level, so each")
    print("column answers: 'how much extra heat does this rho buy me, given that")
    print("the joint is already holding this much steady torque?'\n")

    grid = {}
    for load in LOADS:
        for rho in RHOS:
            vals = []
            for seed in SEEDS:
                cmd = make_command(rho, f_lo, np.random.default_rng(seed), n, dt)
                tau = simulate(cmd, load)[k0:]
                vals.append((float(np.mean(tau ** 2)), float(np.mean(tau)),
                             float(np.var(tau))))
            a = np.array(vals)
            grid[(load, rho)] = dict(H=a[:, 0].mean(), dc=a[:, 1].mean(),
                                     var=a[:, 2].mean())

    hdr = "%-12s" % "load (N m)" + "".join("%11s" % ("rho=%.2f" % r) for r in RHOS)
    print(hdr)
    print("-" * len(hdr))
    for load in LOADS:
        h0 = grid[(load, 0.0)]["H"]
        print("%-12.1f" % load + "".join("%11.2f" % (grid[(load, r)]["H"] / h0)
                                         for r in RHOS) + "   <- H / H(rho=0)")
    print()
    hdr2 = "%-12s%12s%12s%12s" % ("load", "tau_dc", "tau_dc^2", "var(tau) @ rho=0.20")
    print(hdr2)
    print("-" * len(hdr2))
    for load in LOADS:
        g = grid[(load, 0.20)]
        print("%-12.1f%12.2f%12.2f%12.2f" % (load, g["dc"], g["dc"] ** 2, g["var"]))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e1b-load-sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["load_Nm", "rho", "H", "H_rel", "tau_dc", "tau_var"])
        for load in LOADS:
            h0 = grid[(load, 0.0)]["H"]
            for rho in RHOS:
                g = grid[(load, rho)]
                w.writerow(["%.1f" % load, "%.2f" % rho, "%.4f" % g["H"],
                            "%.4f" % (g["H"] / h0), "%.4f" % g["dc"], "%.4f" % g["var"]])

    print("\nThe unloaded column is the misleading one. Read the 20 N m row:")
    print("that is what a stance-phase joint actually experiences.")


if __name__ == "__main__":
    main()
