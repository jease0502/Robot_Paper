"""從 unitree_rl_mjlab 的公開常數，推出 G1 每組致動器的 PD 增益與頻寬。

資料來源（逐字取自原始碼，非本檔杜撰）：
  unitreerobotics/unitree_rl_mjlab  src/assets/robots/unitree_g1/g1_constants.py
  mujocolab/mjlab                   src/mjlab/utils/actuator.py

重點：增益不是手調的，是從「目標閉迴路自然頻率 10 Hz、阻尼比 2.0」反推的。
這給了我們一個明確的、可引用的致動器頻寬上限，用來跟 50 Hz 策略的指令頻譜比較。
"""
import math

WN = 10 * 2.0 * 3.1415926535   # NATURAL_FREQ，rad/s
ZETA = 2.0                     # DAMPING_RATIO
POLICY_HZ = 50.0               # 課程與 mjlab G1 任務的策略頻率
LOWLEVEL_HZ = 500.0            # unitree_sdk2 lowcmd 迴路


def reflected_inertia_two_stage(I, G):
    """mjlab utils/actuator.py 的 reflected_inertia_from_two_stage_planetary。"""
    assert G[0] == 1
    return I[0] * (G[1] * G[2]) ** 2 + I[1] * G[2] ** 2 + I[2]


# (名稱, 轉子慣量, 減速比, 速度上限 rad/s, 力矩上限 N·m, 幾個致動器, 對應關節)
MOTORS = [
    ("5020",    (0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46/18, 1 + 56/16), 37.0,  25.0, 1,
     "shoulder p/r/y, elbow, wrist_roll"),
    ("7520_14", (0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48/22),       32.0,  88.0, 1,
     "hip_pitch, hip_yaw, waist_yaw"),
    ("7520_22", (0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5),               20.0, 139.0, 1,
     "hip_roll, knee"),
    ("4010",    (0.068e-4, 0.0,      0.0),      (1, 5, 5),                 22.0,   5.0, 1,
     "wrist_pitch, wrist_yaw"),
    ("ANKLE (2x5020)", (0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46/18, 1 + 56/16), 37.0, 25.0, 2,
     "ankle_pitch, ankle_roll  <-- 四連桿，幾何未知，原始碼自承假設 1:1"),
    ("WAIST (2x5020)", (0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46/18, 1 + 56/16), 37.0, 25.0, 2,
     "waist_pitch, waist_roll"),
]

# 過阻尼二階系統的主導極點： wn * (-zeta + sqrt(zeta^2 - 1))
p_dom = WN * (ZETA - math.sqrt(ZETA ** 2 - 1))      # 取絕對值，rad/s
p_fast = WN * (ZETA + math.sqrt(ZETA ** 2 - 1))
f_dom = p_dom / (2 * math.pi)
f_fast = p_fast / (2 * math.pi)

print(f"設計參數  wn = {WN:.3f} rad/s ({WN/2/math.pi:.1f} Hz)   zeta = {ZETA}")
print(f"閉迴路極點  {-p_dom:.2f} rad/s ({f_dom:.2f} Hz) 與 {-p_fast:.2f} rad/s ({f_fast:.2f} Hz)")
print(f"→ 主導頻寬約 {f_dom:.2f} Hz；策略 Nyquist = {POLICY_HZ/2:.0f} Hz；比值 {(POLICY_HZ/2)/f_dom:.1f}x")
print(f"→ 低階迴路 {LOWLEVEL_HZ:.0f} Hz，每個策略指令被 ZOH 保持 {LOWLEVEL_HZ/POLICY_HZ:.0f} 拍\n")

hdr = f"{'致動器':<16}{'armature':>11}{'kp':>9}{'kd':>8}{'力矩上限':>9}{'action_scale':>14}{'滿幅反轉力矩':>13}"
print(hdr)
print("-" * len(hdr.encode('utf-8')) // 2 * "-" if False else "-" * 80)
for name, I, G, vlim, elim, n, joints in MOTORS:
    arm = reflected_inertia_two_stage(I, G) * n
    kp = arm * WN ** 2
    kd = 2.0 * ZETA * arm * WN
    e = elim * n
    scale = 0.25 * e / kp                 # G1_ACTION_SCALE = 0.25 * effort_limit / stiffness
    tau_step = kp * 2 * scale             # 動作從 -1 跳到 +1 的瞬間 PD 力矩
    print(f"{name:<16}{arm:>11.5f}{kp:>9.2f}{kd:>8.3f}{e:>9.1f}{scale:>14.4f}{tau_step:>13.1f}")
    print(f"{'':<16}關節：{joints}")

print("\n注意：滿幅反轉力矩 = kp * 2 * (0.25*e/kp) = 0.5*e —— 與關節無關，")
print("      恆等於力矩上限的一半。這是 mjlab 自己的 action_scale 慣例直接推出的。")
