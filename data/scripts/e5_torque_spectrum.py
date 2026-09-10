#!/usr/bin/env python
"""E5: where in frequency does the heat actually sit?

e4's three-way split used the policy step as the only boundary, so everything
between the gait fundamental and the 25 Hz command Nyquist landed in the "gait"
bucket. But a 12 Hz low-pass reaches into that bucket. This script measures the
full band decomposition of the heat so the claim can be stated correctly.

Method: run the frozen policy under ZOH, keep the per-policy-step mean torque as
a 50 Hz time series, and split its power spectrum into

    DC          the posture term, mean(tau)^2
    0-5 Hz      gait: the torque modulation walking requires
    5-25 Hz     in-band jitter: reachable by a low-pass filter
    >25 Hz      fast: the ZOH staircase harmonics, from the within-step variance

The four add up to E[tau^2] exactly.
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground import registry

from e4_closed_loop import (ENV_NAME, PRED_ORDER, build_rollout, make_policy,
                            scheme_spec)


def band_split(tau_mean, within, dt):
    """tau_mean (T, nj) 50 Hz series; within (nj,) mean within-step variance."""
    t, nj = tau_mean.shape
    dc = tau_mean.mean(axis=0)
    x = tau_mean - dc
    win = np.hanning(t)
    scale = (win ** 2).sum()
    spec = np.abs(np.fft.rfft(x * win[:, None], axis=0)) ** 2 / scale
    freq = np.fft.rfftfreq(t, dt)
    # Parseval for the one-sided spectrum of a real signal
    spec[1:-1] *= 2.0
    total_var = spec.sum(axis=0)
    actual_var = x.var(axis=0)
    spec *= (actual_var / np.maximum(total_var, 1e-12))

    gait = spec[(freq > 0) & (freq <= 5.0)].sum(axis=0)
    mid = spec[freq > 5.0].sum(axis=0)
    return dc ** 2, gait, mid, within


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--sim-dt", type=float, default=0.002)
    args = ap.parse_args()

    cfg = registry.get_default_config(ENV_NAME)
    cfg.sim_dt = args.sim_dt
    env = registry.load(ENV_NAME, config=cfg)
    n_sub = int(round(env._config.ctrl_dt / env._config.sim_dt))
    with open(os.path.expanduser(args.params), "rb") as f:
        params = pickle.load(f)
    inference_fn = make_policy(params, env, ENV_NAME)
    nj = env.action_size
    jrollout = jax.jit(build_rollout(env, inference_fn, args.steps, n_sub))

    spec = scheme_spec("zoh", nj, jp.zeros((nj, PRED_ORDER)))
    spec = {k: (jp.asarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in spec.items()}

    commands = [np.array(c, dtype=np.float32) for c in
                ([0.5, 0, 0], [1.0, 0, 0], [0.3, 0, 0.5])]
    dt = float(env._config.ctrl_dt)
    acc = []
    for ci, cmd in enumerate(commands):
        for e in range(args.episodes):
            out = jrollout(jax.random.PRNGKey(2000 + e), jp.asarray(cmd), spec)
            alive = np.asarray(out["alive"]) > 0.5
            if alive.sum() < 100:
                continue
            tm = np.asarray(out["tau_mean"], dtype=np.float64)[alive]
            ts = np.asarray(out["tau_sq"], dtype=np.float64)[alive]
            within = np.maximum(ts - tm ** 2, 0.0).mean(axis=0)
            acc.append(band_split(tm, within, dt))

    dcs = np.mean([a[0] for a in acc], axis=0)
    gait = np.mean([a[1] for a in acc], axis=0)
    mid = np.mean([a[2] for a in acc], axis=0)
    fast = np.mean([a[3] for a in acc], axis=0)
    tot = dcs + gait + mid + fast
    T = tot.sum()

    print("policy = %s   n = %d rollouts" % (args.tag, len(acc)))
    print("total heat proxy E[tau^2] summed over joints = %.1f" % T)
    print()
    print("%-28s%10s%10s" % ("band", "absolute", "share"))
    print("-" * 48)
    for name, v in [("DC  (posture)", dcs), ("0-5 Hz  (gait)", gait),
                    ("5-25 Hz (in-band jitter)", mid),
                    (">25 Hz  (ZOH staircase)", fast)]:
        print("%-28s%10.2f%9.2f%%" % (name, v.sum(), 100 * v.sum() / T))
    print("-" * 48)
    print("%-28s%10.2f%9.2f%%" % ("reachable by a filter", (mid + fast).sum(),
                                  100 * (mid + fast).sum() / T))

    names = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
             "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
    print()
    print("%-11s%9s%9s%9s%9s%11s" % ("joint", "DC%", "gait%", "mid%", "fast%",
                                     "reach%"))
    for i in np.argsort(-(mid + fast) / tot):
        print("%-11s%8.1f%%%8.1f%%%8.1f%%%8.1f%%%10.1f%%"
              % (names[i], 100 * dcs[i] / tot[i], 100 * gait[i] / tot[i],
                 100 * mid[i] / tot[i], 100 * fast[i] / tot[i],
                 100 * (mid[i] + fast[i]) / tot[i]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e5-torque-bands-%s.json" % args.tag), "w") as f:
        json.dump({"tag": args.tag, "n_rollouts": len(acc), "joints": names,
                   "dc": dcs.tolist(), "gait": gait.tolist(),
                   "mid_5_25": mid.tolist(), "fast_gt25": fast.tolist(),
                   "shares": {"dc": float(dcs.sum() / T),
                              "gait": float(gait.sum() / T),
                              "mid_5_25": float(mid.sum() / T),
                              "fast_gt25": float(fast.sum() / T),
                              "reachable": float((mid + fast).sum() / T)}}, f, indent=1)
    print("\nwrote", os.path.join(args.out, "e5-torque-bands-%s.json" % args.tag))


if __name__ == "__main__":
    main()
