"""量「策略指令流」裡有多少功率落在致動器頻寬之外。

輸入：handson 實作 4 的獎勵消融所產生的 rollout.npz（每組一個訓練好的策略）。
輸出：每組的指令抖動、頻寬外功率比、以及 PD 暫態力矩統計。

為什麼只看指令端：頻寬外功率比 rho 只需要指令串流本身，不需要真機、
不需要電流計。這讓它可以在部署前就算出來，也讓模擬與真機用同一個定義。

致動器參數逐字取自 MuJoCo Playground 的 Go1 模型（見 logs/go1-actuator-params.txt）：
  kp = 35 N·m/rad（position actuator gainprm）
  joint damping = 0.5 N·m·s/rad（dof_damping）
  armature = 0.005 kg·m^2（dof_armature）
  action_scale = 0.5 rad，ctrl_dt = 0.02 s（50 Hz），sim_dt = 0.004 s（250 Hz）
"""
import csv
import math
import os
import pathlib
import sys

import numpy as np

KP = 35.0
DAMPING = 0.5
ARMATURE = 0.005
ACTION_SCALE = 0.5
CTRL_DT = 0.02

RUNS = pathlib.Path(os.environ.get(
    "ROLLOUT_ROOT",
    pathlib.Path.home() / "handson" / "runs" / "reward-ablation"))
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# 只用 armature 當關節慣量 -> 這是頻寬的「上界」，因為真正的等效慣量還要加連桿。
WN = math.sqrt(KP / ARMATURE)              # rad/s
ZETA = DAMPING / (2.0 * math.sqrt(KP * ARMATURE))
FN = WN / (2 * math.pi)                    # Hz

# 二階系統 -3 dB 頻寬的閉式解（低通型 1/(1+2*zeta*s/wn+(s/wn)^2)）
_k = 1 - 2 * ZETA ** 2
F_BW = FN * math.sqrt(_k + math.sqrt(_k ** 2 + 1))


def second_order_mag(f_hz):
    """|H(jw)| of the closed-loop position response."""
    r = f_hz / FN
    return 1.0 / np.sqrt((1 - r ** 2) ** 2 + (2 * ZETA * r) ** 2)


def analyse(actions, dt):
    """actions: (T, nj) in [-1,1]. 回傳一個 dict 的統計量。"""
    T, nj = actions.shape
    q_des = ACTION_SCALE * actions          # rad，相對預設姿態

    # --- 1. 階梯跳幅與它造成的 PD 暫態力矩 ---
    dq = np.diff(q_des, axis=0)             # 每個控制步的目標跳幅
    tau_step = KP * np.abs(dq)              # 跳的瞬間，PD 的 P 項多出來的力矩
    # --- 2. 指令頻譜 ---
    w = np.hanning(T)
    X = np.fft.rfft((q_des - q_des.mean(0)) * w[:, None], axis=0)
    f = np.fft.rfftfreq(T, dt)
    P = (np.abs(X) ** 2).sum(axis=1)        # 所有關節相加的功率譜
    P[0] = 0.0
    tot = P.sum()

    def frac_above(fc):
        return float(P[f > fc].sum() / tot)

    # --- 3. 有多少指令功率真的變成運動 ---
    H = second_order_mag(f)
    P_moved = (P * H ** 2).sum()
    # --- 4. 頻譜重心 ---
    centroid = float((f * P).sum() / tot)

    return {
        "steps": T,
        "joints": nj,
        "action_rms": float(np.sqrt((actions ** 2).mean())),
        "delta_rms_rad": float(np.sqrt((dq ** 2).mean())),
        "delta_p95_rad": float(np.percentile(np.abs(dq), 95)),
        "delta_max_rad": float(np.abs(dq).max()),
        "tau_step_rms_Nm": float(np.sqrt((tau_step ** 2).mean())),
        "tau_step_p95_Nm": float(np.percentile(tau_step, 95)),
        "tau_step_max_Nm": float(tau_step.max()),
        "centroid_Hz": centroid,
        "rho_above_bw": frac_above(F_BW),
        "rho_above_fn": frac_above(FN),
        "rho_above_5Hz": frac_above(5.0),
        "motion_efficiency": float(P_moved / tot),
    }


def main():
    print(f"致動器：kp={KP} damping={DAMPING} armature={ARMATURE}")
    print(f"  自然頻率 fn = {FN:.2f} Hz   阻尼比 zeta = {ZETA:.3f}  ({'欠阻尼' if ZETA<1 else '過阻尼'})")
    print(f"  -3 dB 頻寬 = {F_BW:.2f} Hz")
    print(f"  策略 {1/CTRL_DT:.0f} Hz -> Nyquist {0.5/CTRL_DT:.0f} Hz，是頻寬的 {(0.5/CTRL_DT)/F_BW:.1f} 倍")
    peak = second_order_mag(FN * math.sqrt(max(1 - 2 * ZETA ** 2, 1e-9)))
    print(f"  共振尖峰增益 = {peak:.2f}x（zeta<0.707 才有）\n")

    variants = sorted(p.name for p in RUNS.iterdir() if (p / "rollout.npz").exists())
    rows = []
    for v in variants:
        d = np.load(RUNS / v / "rollout.npz")
        st = analyse(np.asarray(d["actions"], dtype=np.float64), float(d["dt"]))
        st["variant"] = v
        rows.append(st)

    cols = ["variant", "delta_rms_rad", "tau_step_rms_Nm", "tau_step_p95_Nm",
            "tau_step_max_Nm", "centroid_Hz", "rho_above_bw", "rho_above_5Hz",
            "motion_efficiency"]
    hdr = f"{'變體':<20}{'Δq_rms':>9}{'τ_rms':>8}{'τ_p95':>8}{'τ_max':>8}{'重心Hz':>9}{'ρ>BW':>8}{'ρ>5Hz':>8}{'運動效率':>9}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['variant']:<20}{r['delta_rms_rad']:>9.4f}{r['tau_step_rms_Nm']:>8.2f}"
              f"{r['tau_step_p95_Nm']:>8.2f}{r['tau_step_max_Nm']:>8.2f}{r['centroid_Hz']:>9.2f}"
              f"{r['rho_above_bw']:>8.3f}{r['rho_above_5Hz']:>8.3f}{r['motion_efficiency']:>9.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "command-spectrum.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"\n寫出 {OUT/'command-spectrum.csv'}")

    print("\n讀法：")
    print("  Δq_rms  每個控制步目標角跳多少（rad）。這是階梯的高度。")
    print("  τ_rms   跳的瞬間 P 項多出來的力矩 = kp*Δq。它不做淨功，只發熱。")
    print("  ρ>BW    指令功率有多少落在致動器 -3 dB 頻寬之外 —— 這些永遠變不成運動。")
    print("  運動效率 指令功率經過二階致動器後還剩多少。1.0 = 全部變成運動。")


if __name__ == "__main__":
    main()
