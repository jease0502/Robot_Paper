"""E1 in simulation: does out-of-band command power raise the heat proxy
while the joint's actual motion stays the same?

This is the cheapest possible falsification of H1. It needs no robot.

Setup
-----
One joint, Go1 parameters taken verbatim from the MuJoCo Playground model:

    kp       = 35.0    N m / rad   (position actuator gainprm)
    b        = 0.5     N m s / rad (dof_damping, passive joint damping)
    J        = 0.005   kg m^2      (dof_armature; a load inertia can be added)

    J qdd = kp (q_des - q) - b qd

The command is built as

    q_des(t) = A sin(2 pi f0 t)  +  n(t)

where n(t) is band-limited noise confined to [f_lo, 25] Hz, i.e. entirely
above the joint's closed-loop bandwidth. Its amplitude is solved for so that
the out-of-band power fraction rho hits a requested target.

The fundamental amplitude A is held FIXED across every rho, so the useful
motion is matched by construction. Any rise in the heat proxy is therefore
caused by the out-of-band content alone, not by "the joint moved more".

The command is sampled at 50 Hz and zero-order-held onto the 500 Hz
low-level loop, exactly as on hardware. Integration runs at 5 kHz so the
ZOH edges are resolved.

Heat proxy: H = mean(tau^2), proportional to I_rms^2 R for a motor with a
constant torque constant. Copper loss is a square law, which is the whole
reason a zero-mean ripple still costs energy.
"""
import csv
import json
import math
import pathlib
import sys

import numpy as np

KP = 35.0
B = 0.5
J_ARMATURE = 0.005

POLICY_HZ = 50.0
LOWLEVEL_HZ = 500.0
INTEGRATE_HZ = 5000.0

F0 = 2.0          # fundamental of the "gait", Hz -- well inside the bandwidth
AMPLITUDE = 0.20  # rad, fixed across all conditions
DURATION = 20.0   # s
RHO_TARGETS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
SEEDS = [0, 1, 2, 3, 4]


def joint_bandwidth(j_eff):
    """-3 dB bandwidth of  J s^2 + b s + kp  driven by kp."""
    wn = math.sqrt(KP / j_eff)
    zeta = B / (2 * math.sqrt(KP * j_eff))
    k = 1 - 2 * zeta ** 2
    f_bw = (wn / (2 * math.pi)) * math.sqrt(k + math.sqrt(k ** 2 + 1))
    return wn / (2 * math.pi), zeta, f_bw


def make_command(rho_target, f_lo, rng, n_policy, dt_policy):
    """Fundamental sine plus out-of-band noise scaled to hit rho_target."""
    t = np.arange(n_policy) * dt_policy
    base = AMPLITUDE * np.sin(2 * math.pi * F0 * t)
    if rho_target <= 0:
        return base, 0.0

    # Band-limited noise in [f_lo, Nyquist], built in the frequency domain so
    # its support is exact rather than approximate.
    freq = np.fft.rfftfreq(n_policy, dt_policy)
    mask = freq >= f_lo
    phase = rng.uniform(0, 2 * math.pi, size=freq.shape)
    spec = np.where(mask, 1.0, 0.0) * np.exp(1j * phase)
    noise = np.fft.irfft(spec, n=n_policy)
    noise /= np.std(noise)

    # Power of the fundamental (all of it below f_lo by construction).
    p_base = np.sum(base ** 2)
    # rho = p_noise / (p_base + p_noise)  ->  p_noise = rho/(1-rho) * p_base
    p_noise_target = rho_target / (1.0 - rho_target) * p_base
    scale = math.sqrt(p_noise_target / np.sum(noise ** 2))
    return base + scale * noise, scale


def measured_rho(q_des, dt_policy, f_bw):
    n = len(q_des)
    win = np.hanning(n)
    x = np.fft.rfft((q_des - q_des.mean()) * win)
    freq = np.fft.rfftfreq(n, dt_policy)
    p = np.abs(x) ** 2
    p[0] = 0.0
    return float(p[freq > f_bw].sum() / p.sum())


def simulate(q_des_policy, j_eff):
    """ZOH the 50 Hz command onto the fast loop and integrate the joint."""
    dt = 1.0 / INTEGRATE_HZ
    steps_per_cmd = int(round(INTEGRATE_HZ / POLICY_HZ))
    n = len(q_des_policy) * steps_per_cmd

    q = 0.0
    qd = 0.0
    tau_hist = np.empty(n)
    q_hist = np.empty(n)

    for i in range(n):
        target = q_des_policy[i // steps_per_cmd]      # zero-order hold
        tau = KP * (target - q)
        qdd = (tau - B * qd) / j_eff
        qd += qdd * dt
        q += qd * dt
        tau_hist[i] = tau
        q_hist[i] = q
    return tau_hist, q_hist


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    j_eff = J_ARMATURE
    fn, zeta, f_bw = joint_bandwidth(j_eff)
    f_lo = max(f_bw, 16.0)     # noise sits strictly above the bandwidth

    print("Joint:  kp=%.1f  b=%.2f  J=%.4f" % (KP, B, j_eff))
    print("        fn=%.2f Hz  zeta=%.3f  -3dB bandwidth=%.2f Hz" % (fn, zeta, f_bw))
    print("Command: %.1f Hz fundamental, amplitude %.2f rad (FIXED)" % (F0, AMPLITUDE))
    print("         out-of-band noise confined to [%.1f, %.1f] Hz" % (f_lo, POLICY_HZ / 2))
    print("         sampled at %.0f Hz, ZOH onto %.0f Hz, integrated at %.0f Hz\n"
          % (POLICY_HZ, LOWLEVEL_HZ, INTEGRATE_HZ))

    dt_policy = 1.0 / POLICY_HZ
    n_policy = int(DURATION * POLICY_HZ)

    rows = []
    for rho_t in RHO_TARGETS:
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            q_des, _ = make_command(rho_t, f_lo, rng, n_policy, dt_policy)
            rho_m = measured_rho(q_des, dt_policy, f_bw)
            tau, q = simulate(q_des, j_eff)

            # discard the first second (startup transient)
            k0 = int(INTEGRATE_HZ * 1.0)
            tau = tau[k0:]
            q = q[k0:]

            heat = float(np.mean(tau ** 2))
            tau_mean = float(np.mean(np.abs(tau)))
            tau_var = float(np.var(tau))
            motion = float(np.sqrt(np.mean((q - q.mean()) ** 2)))
            # how much of the joint's motion is at the fundamental
            per_seed.append((rho_m, heat, tau_mean, tau_var, motion))

        a = np.array(per_seed)
        rows.append({
            "rho_target": rho_t,
            "rho_measured": float(a[:, 0].mean()),
            "heat_proxy": float(a[:, 1].mean()),
            "heat_proxy_std": float(a[:, 1].std()),
            "tau_mean_abs": float(a[:, 2].mean()),
            "tau_var": float(a[:, 3].mean()),
            "motion_rms_rad": float(a[:, 4].mean()),
            "motion_rms_std": float(a[:, 4].std()),
        })

    base = rows[0]
    hdr = ("%-11s%-11s%12s%9s%12s%13s%9s"
           % ("rho_target", "rho_meas", "heat H", "H/H0", "mean|tau|", "motion rms", "M/M0"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-11.2f%-11.3f%12.3f%9.2f%12.3f%13.5f%9.3f"
              % (r["rho_target"], r["rho_measured"], r["heat_proxy"],
                 r["heat_proxy"] / base["heat_proxy"], r["tau_mean_abs"],
                 r["motion_rms_rad"], r["motion_rms_rad"] / base["motion_rms_rad"]))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e1-rho-sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "e1-rho-sweep-config.json", "w", encoding="utf-8") as fh:
        json.dump({"kp": KP, "b": B, "J": j_eff, "fn_hz": fn, "zeta": zeta,
                   "f_bw_hz": f_bw, "f_lo_hz": f_lo, "f0_hz": F0,
                   "amplitude_rad": AMPLITUDE, "duration_s": DURATION,
                   "policy_hz": POLICY_HZ, "lowlevel_hz": LOWLEVEL_HZ,
                   "integrate_hz": INTEGRATE_HZ, "seeds": SEEDS}, fh, indent=1)

    print("\nRead this as: motion rms stays flat (M/M0 ~ 1) while H/H0 climbs.")
    print("That gap is the claim -- same movement, more heat, purely from")
    print("command content the joint cannot follow.")


if __name__ == "__main__":
    main()
