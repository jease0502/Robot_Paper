#!/usr/bin/env python3
"""Fig. 6 -- does the thermal difference show in the behaviour, or only the command?

Top row: each reward-ablation variant's recorded pose at the same instant, drawn
on the Unitree Go1 model. Bottom row: the command stream that variant's policy
emitted for one calf joint over the same two seconds.

Nothing here is re-simulated. data/rollouts/<variant>/rollout.npz stores qpos at
(400, 19) -- the free joint's position and quaternion followed by the twelve
joint angles -- so the frames are recorded states replayed through MuJoCo's
renderer. All seven rollouts share the command [0.5, 0, 0] and 400 steps, so the
reward is the only difference between panels.

The two rows disagree, and that is the point: the gaits look alike while the
command streams separate cleanly. Joint dynamics low-pass whatever the policy
emits, which is why the heat appears in the winding current and not in the
motion -- and why a policy can look healthy while its actuators overheat.

Outputs
    fig/fig6-variants.png, figsrc/out/fig6-variants.pdf   the paper figure
    pic/anim-variants.gif                                 the reading page's animation

Requires the Go1 model from mujoco_menagerie. Point GO1_SCENE at its scene.xml,
or let the script fetch unitree_go1 into figsrc/go1/ on first run.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("needs mujoco:  python -m pip install mujoco")

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "data" / "rollouts"
GO1 = pathlib.Path(os.environ.get("GO1_DIR", ROOT / "figsrc" / "go1"))
MENAGERIE = ("https://api.github.com/repos/google-deepmind/mujoco_menagerie/"
             "contents/unitree_go1")

VARIANTS = [("baseline", "full reward"),
            ("no_action_rate", "no action\\_rate"),
            ("no_torques_energy", "no torque+energy"),
            ("task_only", "task terms only")]
JOINT = 2            # a calf joint, index into the twelve
T0, T1 = 100, 200    # two seconds at 50 Hz


def fetch_model() -> pathlib.Path:
    """Download unitree_go1 from mujoco_menagerie if it is not already here."""
    scene = GO1 / "scene.xml"
    if scene.exists():
        return scene
    print("fetching unitree_go1 from mujoco_menagerie ...")
    (GO1 / "assets").mkdir(parents=True, exist_ok=True)

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "fig6"})
        return urllib.request.urlopen(req, timeout=120).read()

    for name in ("go1.xml", "scene.xml"):
        (GO1 / name).write_bytes(
            get(f"https://raw.githubusercontent.com/google-deepmind/"
                f"mujoco_menagerie/main/unitree_go1/{name}"))
    for f in json.loads(get(MENAGERIE + "/assets")):
        (GO1 / "assets" / f["name"]).write_bytes(get(f["download_url"]))
        print(f"  {f['name']}")
    return scene


def load(variant):
    d = np.load(ROLLOUTS / variant / "rollout.npz")
    a = d["actions"]
    return d["qpos"], a, float(np.sqrt((np.diff(a, axis=0) ** 2).mean()))


def make_renderer(w, h):
    model = mujoco.MjModel.from_xml_path(str(fetch_model()))
    model.vis.global_.offwidth = w
    model.vis.global_.offheight = h
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 1.6, -10, 125
    return model, mujoco.MjData(model), mujoco.Renderer(model, h, w), cam


def pose(model, data, renderer, cam, qpos, i):
    data.qpos[:] = qpos[min(i, len(qpos) - 1)]
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.qpos[:3]
    renderer.update_scene(data, cam)
    return renderer.render()


def paper_figure(rolls):
    model, data, renderer, cam = make_renderer(420, 340)
    fig, ax = plt.subplots(2, 4, figsize=(13.5, 5.4),
                           gridspec_kw=dict(height_ratios=[1.55, 1],
                                            hspace=0.30, wspace=0.16))
    base = rolls[0][2]
    for k, ((q, a, j), (_, nice)) in enumerate(zip(rolls, VARIANTS)):
        ax[0, k].imshow(pose(model, data, renderer, cam, q, 150))
        ax[0, k].set_axis_off()
        ax[0, k].set_title(nice, fontsize=11, fontweight="bold", pad=6)

        t = np.arange(T0, T1) / 50.0
        ax[1, k].plot(t, a[T0:T1, JOINT], lw=0.9, color="#2f6f9e")
        ax[1, k].set_xlabel("time (s)", fontsize=9)
        ax[1, k].tick_params(labelsize=8)
        ax[1, k].grid(alpha=0.25, lw=0.5)
        tag = "baseline" if k == 0 else f"{(j / base - 1) * 100:+.0f}\\%"
        ax[1, k].set_title(f"$\\Delta q_{{rms}}$ {j:.3f}   {tag}",
                           fontsize=9, pad=4)
        if k:
            ax[1, k].set_yticklabels([])

    lo = min(a_.get_ylim()[0] for a_ in ax[1])
    hi = max(a_.get_ylim()[1] for a_ in ax[1])
    for a_ in ax[1]:
        a_.set_ylim(lo, hi)
    ax[1, 0].set_ylabel("commanded angle,\none calf joint (rad)", fontsize=9)

    out_png = ROOT / "fig" / "fig6-variants.png"
    out_pdf = ROOT / "figsrc" / "out" / "fig6-variants.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    for p in (out_png, out_pdf):
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"wrote {p.relative_to(ROOT)}")
    plt.close(fig)


def web_animation(rolls):
    """A small GIF for the reading page, which inlines it as a data URI."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping the animation")
        return
    w, h, step, nframes = 184, 156, 4, 160
    model, data, renderer, cam = make_renderer(w, h)
    frames = []
    for i in range(0, nframes, step):
        row = np.hstack([pose(model, data, renderer, cam, q, i)
                         for q, _, _ in rolls])
        frames.append(Image.fromarray(row).quantize(
            colors=32, method=Image.Quantize.MEDIANCUT))
    out = ROOT / "pic" / "anim-variants.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 * step / 50), loop=0, optimize=True)
    print(f"wrote {out.relative_to(ROOT)}  "
          f"{len(frames)} frames, {out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    rolls = [load(v) for v, _ in VARIANTS]
    print("command:", np.load(ROLLOUTS / "baseline" / "rollout.npz")["command"])
    paper_figure(rolls)
    web_animation(rolls)
