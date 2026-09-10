#!/usr/bin/env python
"""E4: closed-loop command filtering on a frozen Go1 policy.

Unlike e3_command_filters.py, which replayed a recorded action sequence, here
the filter sits inside the loop: the policy sees the joint states that the
FILTERED commands produced, and reacts to them. That is the deployment
situation, and it is where an open-loop win can evaporate.

What is held fixed
------------------
  policy weights, initial states, observation-noise RNG stream, velocity
  command, episode length, PD gains, physics.

What varies
-----------
  only how the 50 Hz action becomes the 500 Hz actuator target stream.

Loop structure (per policy step, 20 ms)
---------------------------------------
  obs  ->  frozen policy  ->  action a[t]
        ->  FILTER (this is the only thing under study)
        ->  target = default_pose + action_scale * a_filtered
        ->  n_sub substeps of mjx.step, each with its own ctrl
            (ZOH holds the target; linear ramps towards it)

Measured
--------
  H          mean(actuator_force^2) summed over joints -> the heat proxy
  tau_dc     mean |actuator_force| per joint
  tau_var    variance of actuator_force per joint
  vel_err    ||local linear velocity[:2] - command[:2]||
  fell       up-vector z went negative
  alive      steps survived

Run
---
  python e4_closed_loop.py --params ~/handson/runs/reward-ablation/baseline/params.pkl \
      --out handson/10-actuator-thermal/logs
"""
import argparse
import functools
import json
import math
import os
import pickle
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ENV_NAME = "Go1JoystickFlatTerrain"
PRED_ORDER = 8
PRED_HORIZON = 2


# --------------------------------------------------------------- policy

def make_policy(params, env, env_name):
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    factory = functools.partial(ppo_networks.make_ppo_networks,
                                **ppo_params.get("network_factory", {}))
    net = factory(env.observation_size, env.action_size,
                  preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


# --------------------------------------------------------------- filters

def butter2_coeffs(cutoff_hz, fs):
    """2nd-order Butterworth via the bilinear transform."""
    wc = math.tan(math.pi * cutoff_hz / fs)
    k1 = math.sqrt(2.0) * wc
    k2 = wc * wc
    a0 = 1.0 + k1 + k2
    b = np.array([k2, 2 * k2, k2]) / a0
    a = np.array([1.0, (2 * (k2 - 1.0)) / a0, (1.0 - k1 + k2) / a0])
    return b, a


PASSTHROUGH = (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def scheme_spec(name, nj, pred_w, gate=None):
    """Static config for a filtering scheme.

    kind 0 = IIR (or passthrough)
    kind 1 = predictive symmetric smoothing: causal, zero added latency,
             the future samples the symmetric kernel needs are PREDICTED
    kind 2 = delayed symmetric smoothing: the same kernel, but the future
             samples are REAL. Truly zero-phase and truly deployable, at the
             price of `half` policy steps of added latency. This is the
             honest ceiling for kind 1.
    """
    g = np.ones(nj) if gate is None else np.asarray(gate, dtype=np.float64)
    lar = 1.0 if str(name).endswith("|raw") else 0.0
    name = str(name).split("|")[0]
    if name == "zoh":
        b, a = PASSTHROUGH
        return dict(kind=0, b=b, a=a, interp=0, gate=g, pred_w=pred_w, last_act_raw=lar)
    if name == "linear":
        b, a = PASSTHROUGH
        return dict(kind=0, b=b, a=a, interp=1, gate=g, pred_w=pred_w, last_act_raw=lar)
    if name.startswith("butter"):
        fc = float(name.split("_")[-1])
        b, a = butter2_coeffs(fc, 50.0)
        interp = 1 if name.startswith("butter_lin") else 0
        return dict(kind=0, b=b, a=a, interp=interp, gate=g, pred_w=pred_w, last_act_raw=lar)
    if name.startswith("delay"):
        b, a = PASSTHROUGH
        return dict(kind=2, b=b, a=a, interp=1, gate=g, pred_w=pred_w, last_act_raw=lar)
    if name.startswith("predict") or name.startswith("gated"):
        b, a = PASSTHROUGH
        return dict(kind=1, b=b, a=a, interp=1, gate=g, pred_w=pred_w, last_act_raw=lar)
    raise ValueError(name)


def kernel_weights(half):
    k = np.hanning(2 * half + 3)[1:-1]
    return k / k.sum()


# --------------------------------------------------------------- rollout

def build_rollout(env, inference_fn, n_steps, n_sub, half=PRED_HORIZON):
    """Returns a jitted function (rng, command, spec) -> metrics."""
    default_pose = jp.asarray(env._default_pose)
    action_scale = float(env._config.action_scale)
    mjx_model = env.mjx_model
    kern = jp.asarray(kernel_weights(half))

    def one_policy_step(carry, _):
        data, info, last_act, iir_x, iir_y, hist, prev_target, alive, rng = carry

        obs = env._get_obs(data, info)
        rng, key = jax.random.split(rng)
        act, _ = inference_fn(obs, key)
        act = jp.clip(act, -1.0, 1.0)

        # ---- IIR branch (direct form II transposed, order 2) ----
        b, a = info["filt_b"], info["filt_a"]
        y_iir = b[0] * act + b[1] * iir_x[0] + b[2] * iir_x[1] \
            - a[1] * iir_y[0] - a[2] * iir_y[1]
        new_x = jp.stack([act, iir_x[0]])
        new_y = jp.stack([y_iir, iir_y[0]])

        # ---- predictive branch: roll the AR model forward, smooth symmetrically ----
        buf = jp.concatenate([hist[1:], act[None]], axis=0)   # (PRED_ORDER, nj)
        w = info["pred_w"]                                    # (nj, PRED_ORDER)

        def roll(c, _):
            b_ = c
            nxt = jp.sum(w * b_[::-1].T, axis=1)
            return jp.concatenate([b_[1:], nxt[None]], axis=0), nxt

        _, fut = jax.lax.scan(roll, buf, None, length=half)    # (half, nj)
        window = jp.concatenate([buf[-(half + 1):], fut], axis=0)  # (2*half+1, nj)
        y_pred = jp.sum(kern[:, None] * window, axis=0)

        # ---- delayed symmetric FIR: same kernel, real future, costs latency ----
        y_delay = jp.sum(kern[:, None] * buf[-(2 * half + 1):], axis=0)

        y = jp.where(info["kind"] == 1, y_pred,
                     jp.where(info["kind"] == 2, y_delay, y_iir))
        y = info["gate"] * y + (1.0 - info["gate"]) * act
        y = jp.clip(y, -1.0, 1.0)

        target = default_pose + action_scale * y

        # ---- substeps: ZOH holds `target`, linear ramps from prev_target ----
        def substep(c, i):
            d, tsq, tsum = c
            frac = (i + 1.0) / n_sub
            ctrl = jp.where(info["interp"] == 1,
                            prev_target + (target - prev_target) * frac,
                            target)
            d = mjx.step(mjx_model, d.replace(ctrl=ctrl))
            f = d.actuator_force
            return (d, tsq + f * f, tsum + jp.abs(f)), f

        (data, tsq, tsum), forces = jax.lax.scan(
            substep, (data, jp.zeros(env.action_size), jp.zeros(env.action_size)),
            jp.arange(n_sub))

        up_z = env.get_upvector(data)[-1]
        still = jp.logical_and(alive > 0.5, up_z > 0.0)
        alive_f = still.astype(jp.float32)

        vel = env.get_local_linvel(data)
        vel_err = jp.linalg.norm(vel[:2] - info["command"][:2])

        # last_act semantics: during training this is the action that was
        # actually applied to the actuators, so with a filter in the loop the
        # faithful choice is the FILTERED action. Feeding the raw action
        # instead keeps the policy completely unaware of the filter.
        info = dict(info)
        info["last_act"] = jp.where(info["last_act_raw"] > 0.5, act, y)
        new_carry = (data, info, y, new_x, new_y, buf, target, alive_f, rng)
        out = dict(tau_sq=tsq / n_sub, tau_abs=tsum / n_sub,
                   tau_mean=forces.mean(axis=0), action=y, raw_action=act,
                   vel_err=vel_err, alive=alive_f, up_z=up_z)
        return new_carry, out

    def rollout(rng, command, spec):
        state = env.reset(rng)
        info = dict(state.info)
        info["command"] = command
        info["filt_b"] = spec["b"]
        info["filt_a"] = spec["a"]
        info["gate"] = spec["gate"]
        info["pred_w"] = spec["pred_w"]
        info["kind"] = spec["kind"]
        info["interp"] = spec["interp"]
        info["last_act_raw"] = spec["last_act_raw"]
        nj = env.action_size
        carry = (state.data, info, jp.zeros(nj), jp.zeros((2, nj)), jp.zeros((2, nj)),
                 jp.zeros((PRED_ORDER, nj)), default_pose, jp.float32(1.0), rng)
        _, out = jax.lax.scan(one_policy_step, carry, None, length=n_steps)
        return out

    return rollout


# --------------------------------------------------------------- driver

def summarize(out, n_steps):
    alive = np.asarray(out["alive"])
    n_alive = int(alive.sum())
    mask = alive > 0.5
    if n_alive < 5:
        mask = np.ones_like(alive, dtype=bool)
        n_alive = len(alive)
    tau_sq = np.asarray(out["tau_sq"])[mask]      # per policy step: mean_sub(f^2)
    tau_mean = np.asarray(out["tau_mean"])[mask]  # per policy step: mean_sub(f)
    vel_err = np.asarray(out["vel_err"])[mask]

    # Exact three-way split of the heat proxy:
    #   E[f^2] = dc^2  +  Var_step(step means)  +  E_step[within-step variance]
    # dc^2   posture: the torque needed to hold the body up
    # gait   the locomotion's own torque modulation, below 25 Hz
    # fast   everything faster than the policy step -- the ZOH transient and
    #        the jitter. THIS is the only part a command filter can remove.
    dc = tau_mean.mean(axis=0)
    within = np.maximum(tau_sq - tau_mean ** 2, 0.0).mean(axis=0)
    across = tau_mean.var(axis=0)
    total = tau_sq.mean(axis=0)
    tot = max(float(total.sum()), 1e-9)
    return dict(
        H=float(total.sum()),
        H_per_joint=[float(x) for x in total],
        tau_dc=[float(x) for x in dc],
        tau_var=[float(x) for x in np.maximum(total - dc ** 2, 0.0)],
        ripple_frac=float(np.maximum(total - dc ** 2, 0).sum() / tot),
        dc_frac=float((dc ** 2).sum() / tot),
        gait_frac=float(across.sum() / tot),
        fast_frac=float(within.sum() / tot),
        fast_abs=float(within.sum()),
        fast_per_joint=[float(x) for x in within],
        vel_err=float(vel_err.mean()),
        steps_alive=int(n_alive),
        fell=bool(alive[-1] < 0.5),
    )


def fit_predictor(actions, order):
    """Least-squares one-step-ahead AR per joint, fitted on ZOH rollouts."""
    n, nj = actions.shape
    w = np.zeros((nj, order))
    r2 = np.zeros(nj)
    for j in range(nj):
        rows = np.stack([actions[i - order:i, j][::-1] for i in range(order, n)])
        tgt = actions[order:, j]
        cut = int(0.7 * len(tgt))
        ww, *_ = np.linalg.lstsq(rows[:cut], tgt[:cut], rcond=None)
        w[j] = ww
        pred = rows[cut:] @ ww
        ss_res = float(np.sum((tgt[cut:] - pred) ** 2))
        ss_tot = float(np.sum((tgt[cut:] - tgt[cut:].mean()) ** 2))
        r2[j] = 1.0 - ss_res / max(ss_tot, 1e-12)
    return w, float(r2.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sim-dt", type=float, default=0.002)
    ap.add_argument("--schemes", default=("zoh,linear,butter_5,butter_8,butter_12,"
                                          "butter_lin_8,butter_lin_12,"
                                          "delay_sym,predict,gated"))
    ap.add_argument("--gate-tau", type=float, default=3.2,
                    help="filter only joints whose mean |tau| is below this")
    ap.add_argument("--commands", default="0.5,0,0|1.0,0,0|0.3,0,0.5")
    args = ap.parse_args()

    cfg = registry.get_default_config(ENV_NAME)
    cfg.sim_dt = args.sim_dt
    env = registry.load(ENV_NAME, config=cfg)
    n_sub = int(round(env._config.ctrl_dt / env._config.sim_dt))
    print("ctrl_dt %.4f  sim_dt %.4f  n_sub %d  -> low-level %.0f Hz"
          % (env._config.ctrl_dt, env._config.sim_dt, n_sub, 1.0 / env._config.sim_dt))

    with open(os.path.expanduser(args.params), "rb") as f:
        params = pickle.load(f)
    inference_fn = make_policy(params, env, ENV_NAME)

    nj = env.action_size
    rollout = build_rollout(env, inference_fn, args.steps, n_sub)
    jrollout = jax.jit(rollout)

    commands = [np.array([float(x) for x in c.split(",")], dtype=np.float32)
                for c in args.commands.split("|")]
    schemes = args.schemes.split(",")

    # --- pass 1: ZOH rollouts to fit the AR predictor -------------------
    zero_w = jp.zeros((nj, PRED_ORDER))
    spec0 = scheme_spec("zoh", nj, zero_w)
    spec0 = {k: (jp.asarray(v) if isinstance(v, np.ndarray) else v)
             for k, v in spec0.items()}
    acts, dcs = [], []
    t0 = time.time()
    for e in range(6):
        out = jrollout(jax.random.PRNGKey(1000 + e),
                       jp.asarray(commands[e % len(commands)]), spec0)
        alive = np.asarray(out["alive"]) > 0.5
        a = np.asarray(out["raw_action"], dtype=np.float64)
        t = np.asarray(out["tau_abs"], dtype=np.float64)
        keep = alive if alive.sum() > 50 else np.ones_like(alive, dtype=bool)
        acts.append(a[keep])
        dcs.append(t[keep].mean(axis=0))
    print("warmup + 6 calibration rollouts in %.1f s" % (time.time() - t0))

    # The AR predictor is fitted OFFLINE on the frozen policy's own ZOH output.
    # No simulator interaction beyond these calibration rollouts, no retraining.
    raw = np.concatenate(acts, axis=0)
    pred_w, pred_r2 = fit_predictor(raw, PRED_ORDER)
    print("AR predictor fitted on %d ZOH steps, out-of-sample R2 = %.3f"
          % (len(raw), pred_r2))

    # Per-joint steady torque, measured on the ZOH baseline. E1b says the
    # ripple term only matters where this is small, so the gate switches the
    # filter off on the load-bearing joints.
    dc = np.mean(dcs, axis=0)
    gate = (dc < args.gate_tau).astype(np.float64)
    print("per-joint mean |tau| (N m): %s" % np.round(dc, 2).tolist())
    print("gate at %.1f N m -> filtering %d of %d joints: %s"
          % (args.gate_tau, int(gate.sum()), nj, np.where(gate > 0)[0].tolist()))

    results = []
    for name in schemes:
        spec = scheme_spec(name, nj, pred_w,
                           gate=gate if name.startswith("gated") else None)
        spec = {k: (jp.asarray(v) if isinstance(v, np.ndarray) else v)
                for k, v in spec.items()}
        for ci, cmd in enumerate(commands):
            t0 = time.time()
            for e in range(args.episodes):
                out = jrollout(jax.random.PRNGKey(2000 + e), jp.asarray(cmd), spec)
                r = summarize(out, args.steps)
                r.update(scheme=name, command=ci, episode=e,
                         cmd=[float(x) for x in cmd])
                results.append(r)
            print("  %-14s cmd%d  %5.1f s  H=%.1f  vel_err=%.3f  alive=%.0f  fall=%.2f"
                  % (name, ci, time.time() - t0,
                     np.mean([r["H"] for r in results[-args.episodes:]]),
                     np.mean([r["vel_err"] for r in results[-args.episodes:]]),
                     np.mean([r["steps_alive"] for r in results[-args.episodes:]]),
                     np.mean([r["fell"] for r in results[-args.episodes:]])))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "e4-closed-loop-%s.json" % args.tag)
    with open(path, "w") as f:
        json.dump({"env": ENV_NAME, "tag": args.tag, "steps": args.steps,
                   "sim_dt": args.sim_dt, "n_sub": n_sub,
                   "pred_r2": pred_r2, "episodes": args.episodes,
                   "tau_dc_zoh": [float(x) for x in dc],
                   "gate_tau": args.gate_tau,
                   "gate": [float(x) for x in gate],
                   "rows": results}, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
