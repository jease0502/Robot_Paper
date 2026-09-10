"""E3: can a filter on the command stream buy the same motion for less heat?

The policy is FROZEN. Nothing is retrained. The only thing that changes is what
happens to the 50 Hz command between the policy and the 500 Hz PD loop.

Schemes compared
----------------
  zoh        what everybody ships today: hold the last value for 10 ticks
  linear     first-order hold, causal, costs one policy step of lag
  butter_c   causal 2nd-order Butterworth at several cutoffs (has phase lag)
  zerophase  the same Butterworth forwards and backwards. NON-causal, cannot be
             deployed. Included as the ceiling on what lag-free smoothing is worth
  predict    causal and deployable: a small per-joint linear predictor estimates
             the next few commands so a symmetric (zero-phase) kernel can be
             applied without waiting for the future. This is the "model that
             filters the commands": fit offline on the policy's own output,
             no simulator, no retraining

What counts as "the motion we must keep"
----------------------------------------
NOT the raw ZOH trajectory -- that contains the jitter we are trying to remove,
so scoring against it would punish the filter for working. Instead the reference
is the IN-BAND part of the ZOH trajectory: the ZOH joint trajectory low-passed
at F_TASK. That is the locomotion the policy is actually producing. A good
filter reproduces it while spending less heat.

Metrics
-------
  H_rel        mean(tau^2) relative to ZOH -> heat, proportional to I_rms^2 R
  band_err     RMS in-band deviation from the reference trajectory, in rad
  eta          (1 - band_err/band_amp) / H_rel -> in-band motion kept per unit heat
  pred_r2      out-of-sample R^2 of the command predictor (diagnostic)
"""
import csv
import math
import os
import pathlib
import sys

import numpy as np

KP = 35.0
B_DAMP = 0.5
J = 0.005
ACTION_SCALE = 0.5
POLICY_HZ = 50.0
LOWLEVEL_HZ = 500.0
INTEGRATE_HZ = 5000.0

F_TASK = 5.0            # locomotion happens below this; above it is jitter
RUNS = pathlib.Path(os.environ.get(
    "ROLLOUT_ROOT",
    pathlib.Path.home() / "handson" / "runs" / "reward-ablation"))
VARIANTS = ["baseline", "no_torques_energy", "task_only"]
LOADS = [0.0, 2.0, 10.0]
CUTOFFS = [5.0, 8.0, 12.0]
PRED_ORDER = 8
SMOOTH_HALF = 2


def upsample_zoh(cmd, factor):
    return np.repeat(cmd, factor, axis=0)


def upsample_linear(cmd, factor):
    delayed = np.concatenate([cmd[:1], cmd[:-1]])
    ramp = np.arange(factor) / factor
    out = (delayed[:, None, :] * (1 - ramp)[None, :, None]
           + cmd[:, None, :] * ramp[None, :, None])
    return out.reshape(-1, cmd.shape[1])


def butter2_coeffs(cutoff_hz, fs):
    wc = math.tan(math.pi * cutoff_hz / fs)
    k1 = math.sqrt(2.0) * wc
    k2 = wc * wc
    a0 = 1.0 + k1 + k2
    b = np.array([k2, 2 * k2, k2]) / a0
    a = np.array([1.0, (2 * (k2 - 1.0)) / a0, (1.0 - k1 + k2) / a0])
    return b, a


def lfilt(sig, b, a):
    out = np.zeros_like(sig)
    for j in range(sig.shape[1]):
        x1 = x2 = y1 = y2 = 0.0
        for i in range(sig.shape[0]):
            xi = sig[i, j]
            yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            out[i, j] = yi
            x2, x1 = x1, xi
            y2, y1 = y1, yi
    return out


def butter2(x, cutoff_hz, fs, zero_phase=False):
    b, a = butter2_coeffs(cutoff_hz, fs)
    y = lfilt(x, b, a)
    if zero_phase:
        y = lfilt(np.ascontiguousarray(y[::-1]), b, a)[::-1]
    return y


def band_limit(sig, fs):
    """Zero-phase low pass at F_TASK. Used only to DEFINE the reference and to
    measure in-band error, never as a deployed filter."""
    return butter2(sig, F_TASK, fs, zero_phase=True)


def fit_predictor(cmd, order):
    n, nj = cmd.shape
    half = n // 2
    weights = np.zeros((nj, order))
    r2 = np.zeros(nj)
    for j in range(nj):
        rows = np.stack([cmd[i - order:i, j][::-1] for i in range(order, half)])
        targets = cmd[order:half, j]
        w, *_ = np.linalg.lstsq(rows, targets, rcond=None)
        weights[j] = w
        vr = np.stack([cmd[i - order:i, j][::-1] for i in range(half, n)])
        vt = cmd[half:n, j]
        pred = vr @ w
        ss_res = float(np.sum((vt - pred) ** 2))
        ss_tot = float(np.sum((vt - vt.mean()) ** 2))
        r2[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return weights, r2


def predictive_smooth(cmd, weights, half_width):
    """Causal, zero-phase-like smoothing.

    A symmetric kernel needs `half_width` future samples. We predict them, so
    the kernel stays symmetric and the filter keeps (near) zero phase lag in
    the passband, unlike a causal Butterworth.
    """
    n, nj = cmd.shape
    order = weights.shape[1]
    kernel = np.hanning(2 * half_width + 3)[1:-1]
    kernel /= kernel.sum()
    out = np.empty_like(cmd)
    for j in range(nj):
        col = cmd[:, j]
        for i in range(n):
            lo = max(0, i - order + 1)
            past = col[lo:i + 1]
            if len(past) < order:
                past = np.concatenate([np.full(order - len(past), col[0]), past])
            buf = list(past)
            fut = []
            for _ in range(half_width):
                nxt = float(np.dot(weights[j], np.asarray(buf[-order:])[::-1]))
                fut.append(nxt)
                buf.append(nxt)
            lo2 = max(0, i - half_width)
            hist = list(col[lo2:i + 1])
            if len(hist) < half_width + 1:
                hist = [col[0]] * (half_width + 1 - len(hist)) + hist
            out[i, j] = float(np.dot(kernel, np.asarray(hist + fut)))
    return out


def simulate(stream, tau_load):
    dt = 1.0 / INTEGRATE_HZ
    sub = int(round(INTEGRATE_HZ / LOWLEVEL_HZ))
    n, nj = stream.shape
    q = np.zeros(nj)
    qd = np.zeros(nj)
    tau_sq = np.zeros(nj)
    q_hist = np.empty((n * sub, nj))
    k = 0
    for i in range(n):
        target = stream[i]
        for _ in range(sub):
            tau = KP * (target - q)
            qd += ((tau - B_DAMP * qd - tau_load) / J) * dt
            q += qd * dt
            tau_sq += tau * tau
            q_hist[k] = q
            k += 1
    return tau_sq / (n * sub), q_hist


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    factor = int(round(LOWLEVEL_HZ / POLICY_HZ))
    rows = []

    for variant in VARIANTS:
        d = np.load(RUNS / variant / "rollout.npz")
        cmd = ACTION_SCALE * np.asarray(d["actions"], float)
        n = cmd.shape[0]
        weights, r2 = fit_predictor(cmd, PRED_ORDER)
        eval_from = (n // 2) * factor * int(INTEGRATE_HZ / LOWLEVEL_HZ)

        streams = {
            "zoh": upsample_zoh(cmd, factor),
            "linear": upsample_linear(cmd, factor),
            "predict": upsample_linear(
                predictive_smooth(cmd, weights, SMOOTH_HALF), factor),
        }
        for fc in CUTOFFS:
            streams["butter_c_%.0f" % fc] = upsample_zoh(
                butter2(cmd, fc, POLICY_HZ), factor)
            streams["zerophase_%.0f" % fc] = upsample_zoh(
                butter2(cmd, fc, POLICY_HZ, zero_phase=True), factor)

        for load in LOADS:
            base_h, base_q = simulate(streams["zoh"], load)
            ref = band_limit(base_q, INTEGRATE_HZ)[eval_from:]
            band_amp = ref.std(axis=0)
            for name, st in streams.items():
                h, qh = simulate(st, load)
                got = band_limit(qh, INTEGRATE_HZ)[eval_from:]
                err = (got - ref).std(axis=0)
                keep = np.clip(1.0 - err / np.maximum(band_amp, 1e-9), 0.0, 1.0)
                heat_rel = h / base_h
                rows.append({
                    "variant": variant,
                    "load_Nm": load,
                    "scheme": name,
                    "H_rel": float(np.mean(heat_rel)),
                    "band_err_rad": float(np.mean(err)),
                    "motion_kept": float(np.mean(keep)),
                    "eta": float(np.mean(keep / heat_rel)),
                    "pred_r2": float(np.mean(r2)),
                })

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "e3-command-filters.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Reference = ZOH joint trajectory low-passed at %.0f Hz (the locomotion)."
          % F_TASK)
    print("A good scheme keeps motion_kept near 1.0 while H_rel drops.\n")
    for variant in VARIANTS:
        r2v = [r["pred_r2"] for r in rows if r["variant"] == variant][0]
        print("=" * 66)
        print("%s    command predictor out-of-sample R2 = %.3f" % (variant, r2v))
        for load in LOADS:
            sel = [r for r in rows if r["variant"] == variant and r["load_Nm"] == load]
            print("  steady load %.0f N m" % load)
            print("    %-16s%9s%12s%13s%8s"
                  % ("scheme", "H/H_zoh", "band err", "motion kept", "eta"))
            for r in sorted(sel, key=lambda z: -z["eta"]):
                print("    %-16s%9.3f%12.4f%13.3f%8.2f"
                      % (r["scheme"], r["H_rel"], r["band_err_rad"],
                         r["motion_kept"], r["eta"]))
    print("\nzerophase_* is NOT deployable (needs the future); it is the ceiling.")
    print("predict is causal and deployable. Compare it against butter_c at the")
    print("same cutoff: same smoothing, but without the phase lag.")


if __name__ == "__main__":
    main()
