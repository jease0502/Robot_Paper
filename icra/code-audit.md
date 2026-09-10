# Code audit: `icra/main.tex` against `data/scripts/*.py`

Read-only audit performed 2026-09-11. Scope: manuscript Sections II–V, all 13
released scripts, and every file in `data/results/`. Line numbers are as of the
files audited; no file was modified.

Where paper and code agree, nothing is reported. The following **were** checked
and **do** agree, and are listed once here only so the reader knows they were not
skipped: AR order 8 and horizon 2; least-squares fit; symmetric-Hann kernel;
E1 `f_n = 13.3 Hz`, `zeta = 0.5976`, `-3 dB = 15.33 Hz` recomputed from
`k_p=35, b=0.5, J=0.005`; E3 `n=30` per scheme, 10 episodes x 3 commands x 500
steps, `n_sub=10`, `sim_dt=0.002`, matched PRNG keys; E4 30 rollouts, cuts at
5 Hz and Nyquist, intra-step variance as the `>25 Hz` term; every number in
Tables I, II, III, IV, V, VII, VIII, IX; `kappa = 38.384`, max residual 0.4451%;
open-loop R^2 0.679 / 0.385 / 0.059; closed-loop R^2 0.609 / 0.402; the -4.5% to
+11.6% per-seed range; return 25.79 +/- 0.65 and 26.31 +/- 0.30; all per-velocity
fall rates including the 100% figure on the unregularized policy.

---

## (A) CONTRADICTION — the paper says something the code does not do

### A1. The 3.2 N.m crossover is neither analytical nor measured on the Go1; it is `sqrt(Var)` read off one hard-picked row of a synthetic sweep, inside a plotting script

The manuscript calls it analytical three times — abstract (l.102) "an analytical
crossover at $3.2\Nm$"; contributions (l.151) "We identify an analytical
crossover at $\bar{\tau} \approx \sqrt{\mathrm{Var}(\tau)} \approx 3.2\Nm$ on a
Unitree Go1"; Sec. V-A (l.375) "The crossover ... emerges at $3.2\Nm$."

No script solves anything. The only place 3.2 is produced is a print statement at
the bottom of a figure script:

`data/scripts/plot_e1.py:88`
```python
var_at_02 = float([r for r in lrows
                   if float(r["load_Nm"]) == 0.0 and float(r["rho"]) == 0.20][0]["tau_var"])
```
`data/scripts/plot_e1.py:90`
```python
print("var(tau) at rho=0.20 is %.2f, so the DC term dominates once "
      "tau_dc > %.2f N m" % (var_at_02, math.sqrt(var_at_02)))
```

`Var(tau)` in `e1b-load-sweep.csv` is a property of the **synthetic** E1 command
(2 Hz sine + noise in [16,25] Hz through a single-joint model), and it is a
function of the chosen `rho`, which is swept:

| rho | Var(tau) | sqrt(Var) = "crossover" |
|---|---|---|
| 0.05 | 2.917 | **1.71** |
| 0.10 | 5.105 | **2.26** |
| **0.20** | **10.301** | **3.21** |
| 0.40 | 25.890 | **5.09** |

The paper never states that the crossover is conditioned on `rho = 0.20`, and
`rho = 0.20` is not justified anywhere. Pick 0.10 and the crossover is 2.3 N.m,
which moves four thigh joints across the line. Nothing in `e1b_sim_load_sweep.py`
computes a crossover at all, contrary to `data/README.md` ("`e1b_sim_load_sweep.py`
| **§5.1**, the load sweep and the 3.2 N.m crossover").

3.2 then propagates as a hard-coded default into the closed-loop gate:

`data/scripts/e4_closed_loop.py:299`
```python
ap.add_argument("--gate-tau", type=float, default=3.2,
```

**Risk:** this is the number the abstract, one contribution bullet, Fig. 2, the
gate, and the "essential next steps" (l.808, "a dynamometer confirmation of the
$3.2\Nm$ crossover") all rest on. A reviewer who opens `plot_e1.py` finds it is
`sqrt` of one cell of a table.

### A2. The load-gate rule the paper states is not the rule the code implements

Sec. V-G (l.742): "\emph{Gate by load}: restrict smoothing to joints below
$\bar{\tau} \approx \sqrt{\mathrm{Var}(\tau)}$".

The code compares each joint's **mean absolute** torque against a single global
constant. It never computes any joint's `Var(tau)`:

`data/scripts/e4_closed_loop.py:352`
```python
gate = (dc < args.gate_tau).astype(np.float64)
```
where `dc` is mean `|tau|` averaged over the calibration rollouts
(`e4_closed_loop.py:351`, `dc = np.mean(dcs, axis=0)`, fed from
`tau_abs = tsum / n_sub` at `:204`).

Two separate substitutions are hidden here:
1. `sqrt(Var(tau))` per joint is replaced by a fixed 3.2 imported from E1 (A1);
2. `\bar{\tau}` (the DC/posture term, a **signed** mean, which is what Eq. (1)
   squares) is replaced by mean `|tau|`. On this policy the two differ a lot:
   signed per-joint means are `[0.48, -1.04, 4.70, -0.44, -1.42, 4.35, 0.68,
   0.42, 4.32, -0.89, 0.21, 3.85]` against mean `|tau|` of `[1.24, 2.39, 6.88,
   1.35, 2.59, 6.27, 1.46, 2.26, 6.39, 1.48, 2.00, 5.88]`. For the thighs
   `\bar{\tau}^2` is 0.04-2.0 while `(mean|tau|)^2` is 4.0-6.7.

The paper uses one symbol for both: Eq. (1) and the crossover use `\bar{\tau}`,
while l.378 says "measured mean absolute joint torques $|\bar{\tau}|$". See B4 —
the honest fix is cheap, because the stated rule happens to give the same mask.

### A3. Fig. 2's ripple line is a hard-coded synthetic constant, but the caption says the curves are "the definition"

Fig. 2 caption (l.390): "The curves are the definition $r = \mathrm{Var}/
(\bar{\tau}^2+\mathrm{Var})$, not a fit; the joint placements and reachable
shares are measured."

`data/scripts/plot_schematics.py:256`
```python
    var = 10.3                      # measured at rho = 0.2 in E1
```

A single flat `Var = 10.3` is drawn as `ripple term Var(tau), fixed jitter`
(`:260-261`) across all twelve joints, and it is the E1 synthetic value from A1,
not a Go1 measurement. The Go1's own per-joint `Var(tau)` under ZOH, recomputed
from `e4-closed-loop-baseline.json`, is **2.4-3.1 for hips, 6.7-8.9 for thighs,
31.7-37.1 for calves** — it varies by 15x across the joints the figure places on
one horizontal line. The joint placements (`:270-272`) and reachable shares are
also hard-coded rather than read from `e5-torque-bands-baseline.json`:

`data/scripts/plot_schematics.py:270`
```python
    groups = [("hips", 1.24, 1.48, "#2f855a", 45),
```

The placements and shares do currently match the data; the ripple line does not
come from Go1 data at all.

### A4. "Table II reports the five most representative" — Table II has all seven rows

`icra/main.tex:400`
```latex
Table~\ref{tab:ablation} reports the five most representative.
```
Table II body (`main.tex:416-422`) lists seven variants: Full reward, No
orientation, No feet air time, No `act_rate`, No gait terms, No torque+energy,
Task terms only. `multiseed_ablation.py:32` (`ORDER = [...]`, seven entries)
emits all seven, and `results/table2-latex-narrow.txt` has seven body lines. A
reviewer counts rows before reading anything else.

### A5. Hip reachable share is 32%, not 33%

Sec. V-D (l.532): "lightly loaded hip actuators reach 33--45\%". Table VI
(l.548) repeats `33--45\%` for both the 5--25 Hz column and the Reach. column.

Recomputed from `results/e5-torque-bands-baseline.json`, the four hips give
5--25 Hz shares of 37.5 / 44.7 / 32.6 / **31.3**% and reachable shares of
37.8 / 45.0 / 32.9 / **31.6**%. The lower bound rounds to 31 and 32, not 33.
Thighs and calves are correct. The sentence at l.532 is in the always-visible
body text, not behind a `\ifshow` switch.

### A6. The printed thermal-proxy formula does not produce 255.9

Sec. III-C (l.254): "$\mathcal{H} = \frac{1}{T}\int_0^T \tau^2\,\mathrm{d}t$".
As printed this is a single-joint scalar. Every absolute `H` the paper quotes
(255.9, 376.3, Table IX) is a **sum over the twelve joints**:

`data/scripts/e4_closed_loop.py:254`
```python
        H=float(total.sum()),
```
with `total = tau_sq.mean(axis=0)` (`:251`) and `tau_sq` accumulated per substep
then divided by `n_sub` (`:184`, `tsq + f * f`; `:204`, `tau_sq=tsq / n_sub`).
So the code computes mean(tau^2) per joint (normalisation correct) and then
adds twelve of them.

The same symbol `\mathcal{H}` also denotes a strictly single-joint quantity in
E1/E1b/E2 (`e1_sim_rho_sweep.py:154`, `heat = float(np.mean(tau ** 2))`;
`e3_command_filters.py:180`, `return tau_sq / (n * sub), q_hist`). The formula
needs a `\sum_j`, or 255.9 needs a unit and a definition.

### A7. Cliff's delta is described as paired; the implementation is the standard unpaired dominance statistic

Sec. III-D (l.269): "\emph{Paired} comparisons between schemes are reported with
Cliff's $\delta$".

`data/scripts/analyze_e4.py:28`
```python
    gt = sum(int(x > y) for x in a for y in b)
```
This is the textbook definition, `delta = (#(x>y) - #(x<y)) / (mn)`, over all
30x30 = 900 cross pairs. The implementation is **correct as Cliff's delta**; the
word "Paired" is wrong. The data genuinely *are* paired — every scheme is run at
`PRNGKey(2000 + e)` for the same `e` and the same three commands
(`e4_closed_loop.py:366`) — so a reviewer who reads "paired" will expect the
30 matched differences and will not find them. Either drop "Paired", or report a
matched-pairs statistic alongside.

### A8. The predictive filter's numbers include first-order-hold upsampling; the Butterworth rows it is compared against do not

Sec. IV describes the compensator as AR(8) rolled forward two steps enabling a
symmetric Hann window (l.304-308). Nothing about substep interpolation. In the
code the predictive scheme is the Hann smoother **composed with** linear
upsampling, while every Butterworth and zero-phase row keeps ZOH substeps:

`data/scripts/e3_command_filters.py:198` (open loop, Table III)
```python
            "predict": upsample_linear(
```
against `:202`
```python
            streams["butter_c_%.0f" % fc] = upsample_zoh(
```
and `:204`, `streams["zerophase_%.0f" % fc] = upsample_zoh(`.

Table III (l.468-472) therefore compares Predictive 0.428 against Zero-phase
0.540 and Causal Butterworth 0.588 without disclosing that only the first gets
first-order hold — and the paper's own `linear` row shows first-order hold alone
is worth 0.696. The same asymmetry is in the closed loop:

`data/scripts/e4_closed_loop.py:119`
```python
        return dict(kind=1, b=b, a=a, interp=1, gate=g, pred_w=pred_w, last_act_raw=lar)
```
(`interp=1` for `predict`/`gated`, and `:116` `interp=1` for `delay`), against
`:113` where plain `butter_*` gets `interp = 1 if name.startswith("butter_lin")
else 0` — i.e. 0. So in Table IV, Predictive 0.915 and Butterworth 12 Hz 0.935
differ in two things, not one. `butter_lin_12` (0.942) is the matched control and
is in the table, but the paper never says what it controls for.

**Note this cuts the honest way for the Butterworth pair only:** zero-phase vs
causal at 12 Hz *are* matched (both `upsample_zoh`), so the 99.5%-vs-74.5%
phase-lag claim at l.490-493 is clean.

### A9. E1's "ZOH to 500 Hz" stage does not exist in the code

Sec. IV, E1 (l.285): "Commands are sampled at 50\,Hz, held via ZOH to 500\,Hz,
and integrated at 5\,kHz."

`e1_sim_rho_sweep.py` defines `LOWLEVEL_HZ = 500.0` (`:49`) and then never uses
it except to print it (`:134`) and store it in the config JSON (`:193`). The hold
goes straight from 50 Hz to the 5 kHz integrator:

`data/scripts/e1_sim_rho_sweep.py:105`
```python
    steps_per_cmd = int(round(INTEGRATE_HZ / POLICY_HZ))
```
(= 100), and `:114` `target = q_des_policy[i // steps_per_cmd]`. The torque law
at `:115` is then evaluated every 0.2 ms, i.e. a **continuous** P law, not a
500 Hz discrete one. `e1b_sim_load_sweep.py` does not even define `LOWLEVEL_HZ`
(`:67-78`). For contrast, `e3_command_filters.py:164` does implement the stage
(`sub = int(round(INTEGRATE_HZ / LOWLEVEL_HZ))`), which shows the omission in E1
is not a modelling convention but an inconsistency.

The ZOH value itself is unchanged by the missing stage, so no E1 number moves;
the sentence is nonetheless describing a stage that is not there, and it is in
the Experimental Setup where a replicator will look.

### A10. The 20 N.m "realistic steady load" is 3x the paper's own measured knee torque

Sec. V-A (l.372): "a joint supporting a $20\Nm$ static load exhibits merely a 2\%
thermal increase under the same command roughness", set against the unloaded
10.9x as the realism correction. Sixteen lines later (l.380) the paper reports
its own measurement: calves 5.88--6.88 N.m.

At 5 N.m the same sweep row gives `H/H0 = 1.3605` and at 10 N.m `1.0927`
(`results/e1b-load-sweep.csv`, rho=0.20). So the real Go1 knee sits at roughly
**+25 to +36%**, not +2%. The 20 N.m column is an extrapolation past every joint
on the robot. The code asserts the opposite in a comment:

`data/scripts/e1b_sim_load_sweep.py:42`
```python
# N m held steadily. 0 = the unloaded case from e1; 20 N m is roughly a Go1
```
and the figure annotates it as such:

`data/scripts/plot_e1.py:74`
```python
ax[2].annotate("stance-phase knee:\n2% effect", xy=(0.20, 1.02), xytext=(0.20, 2.4),
```

Table I bolds `1.02` at 20 N.m (l.353) as the counterpart to the bolded `10.86`
at 0 N.m, so the framing is deliberate. The 5 N.m row is the one that matches the
robot, and it is already in the table.

---

## (B) UNDER-CLAIMED — the code is more careful than the paper says

### B1. Both R^2 figures are strict temporal hold-outs, and the paper only claims it for one of them

Sec. V-C (l.494) says "Out-of-sample" for the open-loop 0.679/0.385/0.059.
Table IX's `Command predictor $R^2$` row (l.705, 0.609/0.402) and its caption
(l.692, "the closed-loop fit") say nothing about holding anything out. Both are
held out, by different splits:

`data/scripts/e3_command_filters.py:120`
```python
        vr = np.stack([cmd[i - order:i, j][::-1] for i in range(half, n)])
```
(fit on the first half at `:116-118`, score on the second half)

`data/scripts/e4_closed_loop.py:279`
```python
        ww, *_ = np.linalg.lstsq(rows[:cut], tgt[:cut], rcond=None)
```
with `cut = int(0.7 * len(tgt))` (`:278`) and scoring on `rows[cut:]` (`:281`).

**Sentence to add** (Table IX caption or Sec. V-C): "Both predictor $R^2$ values
are out-of-sample: the open-loop fit is scored on the held-out second half of the
rollout, and the closed-loop fit on the held-out final 30\% of the calibration
rollouts."

### B2. The predictor and the gate are calibrated on rollouts that share no seed with any evaluated episode, and are collected under pure ZOH

This is the strongest no-leakage guarantee in the paper and it is stated nowhere.
Calibration uses one seed family, evaluation another:

`data/scripts/e4_closed_loop.py:331`
```python
        out = jrollout(jax.random.PRNGKey(1000 + e),
```
against evaluation at `:366`
```python
                out = jrollout(jax.random.PRNGKey(2000 + e), jp.asarray(cmd), spec)
```

and the calibration pass runs the unfiltered scheme only (`:325`,
`spec0 = scheme_spec("zoh", nj, zero_w)`), so neither the AR weights nor the gate
mask has seen a filtered trajectory. The paper says only "fitted by least squares
on baseline rollouts" (l.305) and "measured once per joint on the same baseline
rollouts that fit the predictor" (l.311-313), which leaves open the obvious
reviewer question of whether the fit saw the test episodes.

**Sentence to add** (Sec. IV, after l.313): "Calibration uses six rollouts drawn
from a seed family disjoint from the thirty evaluation episodes, and is collected
under unfiltered ZOH, so neither the predictor weights nor the gate mask has
observed an evaluated trajectory or a filtered one."

### B3. The Go1's own per-joint statistics reproduce the gate mask exactly, which would repair A2 for free

The rule the paper states in Sec. V-G — filter joint $j$ iff
$\bar{\tau}_j < \sqrt{\mathrm{Var}(\tau_j)}$ — was never run, but it can be
evaluated from `e4-closed-loop-baseline.json` and it gives the identical 8/4
split the shipped constant gives:

| family | mean `|tau|` | `sqrt(Var(tau))` | rule | shipped gate |
|---|---|---|---|---|
| hips | 1.24, 1.35, 1.46, 1.48 | 1.56, 1.68, 1.77, 1.72 | filter | filter |
| thighs | 2.39, 2.59, 2.26, 2.00 | 2.94, 2.98, 2.69, 2.59 | filter | filter |
| calves | 6.88, 6.27, 6.39, 5.88 | 6.09, 5.70, 5.63, 5.67 | pass | pass |

**Sentence to add** (Sec. IV, replacing the bare threshold, or Sec. V-G): "In
implementation the gate is a fixed $3.2\Nm$ threshold on mean $|\tau|$ calibrated
from E1; applying the per-joint rule $\bar{\tau}_j < \sqrt{\mathrm{Var}(\tau_j)}$
directly to the measured Go1 statistics selects the same eight joints, so the
mask does not depend on the imported constant."

### B4. A policy-awareness ablation exists and is a result, not a table row

Table IV's last row, "Predictive, policy-unaw." (l.596, 0.927 / 1.236), is never
explained. It ablates whether the `last_act` observation the policy sees reports
the filtered or the raw action:

`data/scripts/e4_closed_loop.py:202`
```python
        info["last_act"] = jp.where(info["last_act_raw"] > 0.5, act, y)
```
with the reasoning in the comment above it (`:198-200`). This is a real question
about deploying a filter under a frozen policy — whether the observation must be
made consistent with the intervention — and the answer here is that it barely
matters (0.915 vs 0.927 heat, delta(H) -0.33 both ways).

**Sentence to add** (Sec. V-E): "Whether the policy's previous-action observation
reports the filtered or the raw command changes little ($\mathcal{H}/
\mathcal{H}_0$ 0.915 against 0.927, $\delta = -0.33$ in both cases), so the
filter does not need to be made visible to a frozen policy's observation vector."

### B5. E2's reference is chosen so the filter cannot be rewarded for the jitter it removes, and the paper does not say why

Sec. IV (l.290): "Preserved motion is benchmarked against the ZOH trajectory
low-passed at 5\,Hz." The code states the defence explicitly:

`data/scripts/e3_command_filters.py:105`
```python
    """Zero-phase low pass at F_TASK. Used only to DEFINE the reference and to
```
with the argument at `:21-25` — scoring against the raw ZOH trajectory "would
punish the filter for working." The reference low-pass is itself zero-phase
(`:107`, `butter2(sig, F_TASK, fs, zero_phase=True)`), so the reference carries
no lag of its own.

**Sentence to add** (Sec. IV, E2): "The reference is deliberately the in-band
part of the ZOH trajectory rather than the raw one: scoring against the raw
trajectory would penalise a filter for removing the jitter it is meant to remove.
The reference low-pass is applied zero-phase so it contributes no lag."

### B6. E1's "same motion, more heat" is matched by construction, not observed to be matched

Sec. V-A (l.371) reports the outcome ("altering joint motion by only 1.8\%") but
not that the design forces it: the fundamental's amplitude is a constant and only
the out-of-band noise is scaled to hit each `rho`.

`data/scripts/e1_sim_rho_sweep.py:53`
```python
AMPLITUDE = 0.20  # rad, fixed across all conditions
```
with the noise solved for at `:87-88` and the rationale at `:24-26`.

**Sentence to add** (Sec. IV, E1): "The fundamental's amplitude is held constant
across every $\rho$ and only the out-of-band component is scaled, so the useful
motion is matched by construction rather than by observation."

### B7. The "ns" marks in Table II are a mechanical criterion, not a judgement

`data/scripts/multiseed_ablation.py:80`
```python
    return m, sd, abs(m) < sd
```
flags any paired interval whose magnitude is smaller than its own s.d., and the
ratios are paired within seed first (`:187`,
`ratios = [data[v][s][k] / data["baseline"][s][k] ...]`). The caption (l.406)
says only "$^{\textrm{ns}}$ marks an interval straddling zero". Saying the rule
out loud costs nothing and pre-empts "how did you decide what counts as null".

### B8. The Butterworth cutoff sweep is wider than shown, and `butter_lin_*` is a real control

`e4_closed_loop.py:296-298` runs `butter_5, butter_8, butter_12, butter_lin_8,
butter_lin_12` and `e3_command_filters.py:56` sweeps 5/8/12 Hz causal **and**
zero-phase at every cutoff. Table III shows one cutoff; Table IV shows one
`butter_lin`. The `butter_lin_*` pair is exactly the interpolation control that
A8 says is missing from the prose — it exists, it just is not named as one.

---

## (C) UNDER-SPECIFIED — a reviewer would ask; the code answers

### C1. Heat for a scheme that falls is measured only over the pre-fall steps

`data/scripts/e4_closed_loop.py:234`
```python
    mask = alive > 0.5
```
and `:238-240` restrict `tau_sq`, `tau_mean` and `vel_err` to it. So the 5 Hz
Butterworth's `H/H_ZOH = 1.098` (Table IV) is averaged over ~451 of 500 steps
(`steps_alive = 450.67` in `e4-summary-baseline.csv`) and the delayed filter's
1.082 over ~480. The falling schemes' heat and tracking are therefore measured on
a truncated, survivorship-selected window against the ZOH baseline's full 500.
The paper reports `Fall` and `H/H_ZOH` in the same row without saying they are
computed over different horizons.

There is also an untriggered fallback that would silently include post-fall
torques: `:235-237`, `if n_alive < 5: mask = np.ones_like(alive, dtype=bool)`.

### C2. The policies were trained at 250 Hz physics and evaluated at 500 Hz

Sec. IV, E3 (l.294): "500\,Hz physics rate, ten substeps per policy step". The
evaluation does exactly that (`e4_closed_loop.py:295`, `--sim-dt` default 0.002;
`:307`, `n_sub`; JSON confirms `n_sub: 10`). But the training environment is
documented as half that:

`data/scripts/analyze_command_spectrum.py:13`
```python
  action_scale = 0.5 rad，ctrl_dt = 0.02 s（50 Hz），sim_dt = 0.004 s（250 Hz）
```

`e4_closed_loop.py:305` (`cfg.sim_dt = args.sim_dt`) overrides the registry
default, so the frozen checkpoints are being replayed under a physics rate they
were not trained at. *Confidence: medium* — this comes from a docstring, not
executable code, and the registry default is not visible in this repository. The
paper should state the training `sim_dt` either way.

### C3. Sec. II's `zeta = 2.0`, `f_n = 10 Hz` and 2.68 Hz pole are G1 humanoid numbers; every experiment uses a Go1 plant with `zeta = 0.60` and `f_n = 13.3 Hz`

`derive_g1_gains.py:12-13` sets `WN = 10 * 2.0 * pi`, `ZETA = 2.0` from
`unitree_rl_mjlab`'s **G1** constants (`:25-38` are G1 motor part numbers:
5020, 7520, 4010, ankle, waist). The dominant pole at `:41`,
`p_dom = WN * (ZETA - math.sqrt(ZETA ** 2 - 1))`, gives 2.679 Hz — matching
l.213 — for an **overdamped** plant.

Everything measured runs on a Go1 plant that is **underdamped and resonant**:
`zeta = 0.5976`, `f_n = 13.32 Hz`, and `plot_spectrum.py:79` draws its resonance
peak. Sec. II frames the 25 Hz-vs-2.68 Hz mismatch as "the core focus of this
study" (l.215) using the G1 ratio (9.3x) while the studied joint's ratio is 1.6x.
The paper never reconciles the two, and a reviewer who recomputes 25/13.3 will
notice.

### C4. The E1 plant has no derivative feedback term

Sec. II describes "proportional-derivative (PD) tracking loops" and E1 is
introduced as "A second-order joint model ... ($k_p = 35$, damping 0.5 ...)".
The 0.5 is passive plant damping, not a `k_d`:

`data/scripts/e1_sim_rho_sweep.py:115`
```python
        tau = KP * (target - q)
```
and `:116`, `qdd = (tau - B * qd) / j_eff` — the damping acts on the plant and is
absent from the commanded torque, so the measured `tau^2` is a pure P-term
square. `e1b_sim_load_sweep.py:74` and `e3_command_filters.py:174` are identical.
This is a defensible modelling choice (`dof_damping` really is passive in the
MuJoCo model) but it means the heat proxy in E1/E1b/E2 excludes whatever a real
`k_d` would contribute, and the paper does not say so.

### C5. Paper experiment labels are off by one against the released filenames

Paper E1 -> `e1_sim_rho_sweep.py` + `e1b_sim_load_sweep.py`; paper **E2** ->
`e3_command_filters.py`; paper **E3** -> `e4_closed_loop.py`; paper **E4** ->
`e5_torque_spectrum.py`. `data/README.md` sidesteps this by referring to table
numbers instead of E-labels, so nothing in the repository maps the paper's four
protocol names onto the four scripts. Cheap to fix with one column in the README
table; expensive if a reviewer concludes E4's numbers came from `e4_closed_loop.py`.

### C6. Figures 1-5 have no producing script in the released code

`icra/build_icra.py:30-34` stages all five vector figures from
`REPO / "figsrc" / "out" /`, which is not part of `data/`. `plot_schematics.py`
produces PNG ancestors of three of them (`fig-pipeline-and-budget.png`,
`fig-three-filters.png`, `fig-crossover.png`) and `plot_spectrum.py` produces
`command-spectrum.png`, but the manuscript includes
`figs/command-spectrum-annotated.png`, and its caption (l.432-434) describes "the
rule below the axes" marking 5--25 Hz, which `plot_spectrum.py` does not draw
(`:65`, `ax[0].axvspan(F_BW, 25, ...)` is the only band it draws). `data/README.md`
claims `plot_schematics.py` produces "the `pic/` figures", which is true of the
PNGs and not of the PDFs the paper compiles. Fig. 3 (both policies) and Fig. 4
(latency backfire) have no producer in `data/scripts/` at all.

### C7. The four-band sum against 256.2 is a consistency check between two scripts, not independent evidence

Table VIII caption (l.507): "The four components sum to 255.9 against an
independently computed heat proxy of 256.2." Within `e5_torque_spectrum.py` the
four terms sum to `E[tau^2]` by construction — the spectrum is rescaled onto the
measured variance before splitting:

`data/scripts/e5_torque_spectrum.py:46`
```python
    spec *= (actual_var / np.maximum(total_var, 1e-12))
```
and the script's own docstring says so (`:17`, "The four add up to E[tau^2]
exactly"). 256.2 is the ZOH `H` from `e4-summary-baseline.csv` — the same
quantity, over the same 30 rollouts at the same PRNG keys, computed by
`e4_closed_loop.py:254`. The 0.1% gap is a numerical-path difference, not an
independent measurement. The check is worth keeping (it confirms the two scripts'
episode filtering agrees) but "independently computed" oversells it.

### C8. The `>25 Hz` term is intra-step force variance, which interpolation *raises*

`e4_closed_loop.py:249`, `within = np.maximum(tau_sq - tau_mean ** 2, 0.0).mean(axis=0)`.
Under linear substep ramping the actuator force varies *within* the step, so
`fast_abs` goes **up**, not down: ZOH 4.024, predictive 4.021, gated 4.772,
linear **5.762**. Table VII (l.631-635) shows ZOH, three schemes that lower it,
and predictive at 4.02 — and omits `linear` (5.76) and `gated` (4.77), the two
schemes that raise it. The claim at l.617 ("the predictive filter leaves
staircase harmonics untouched") is exactly right for `predict`; the table's
selection makes the term look monotone in filtering when it is not. Labelling
this term "ZOH staircase harmonics" is safe for the baseline decomposition
(E4 is ZOH-only) and is loose for Table VII.

### C9. The E1 closed-form fit discards a fitted intercept and scores the residual on the anchor point

`data/scripts/plot_e1.py:32`
```python
kappa = np.polyfit(x[1:], (heat - h0)[1:], 1)[0]
```
`polyfit` with `deg=1` fits slope **and** intercept; only the slope is kept and
the intercept is replaced by the measured `h0` (the fitted intercept is -0.0375,
so the substitution is harmless here). The residual at `:33` is then evaluated
over all eight `rho` including `rho = 0`, where it is zero by construction. The
"maximum residual 0.45\%" (l.375) is therefore over seven fitted points plus one
anchor. Reported for completeness; the number does not move.

### C10. The AR design matrix crosses rollout-concatenation boundaries

`e4_closed_loop.py:343`, `raw = np.concatenate(acts, axis=0)`, then
`fit_predictor` builds rows with `actions[i - order:i, j]` (`:276`) over the
concatenation. Seven rows per boundary (five boundaries) mix the tail of one
rollout with the head of the next. 35 contaminated rows out of ~3000; negligible,
but a careful reviewer will look for the mask and not find one.

### C11. `rho` in E1 is injected above 16 Hz but measured above 15.33 Hz

`e1_sim_rho_sweep.py:128`, `f_lo = max(f_bw, 16.0)` (= 16.0, since
`f_bw = 15.33`), while `measured_rho` at `:99` integrates above `f_bw`. The
paper defines `rho` against "the joint's $-3$\,dB closed-loop bandwidth,
15.3\,Hz" (l.247) and describes the excitation as "bandpass noise above
bandwidth" (l.284). The two cuts differ by 0.7 Hz. `rho_measured` tracks
`rho_target` to within 1.5% in `e1-rho-sweep.csv`, so nothing moves; the paper
should say the noise floor is 16 Hz.

---

## Summary

| bucket | count |
|---|---|
| (A) contradiction | 10 |
| (B) under-claimed | 8 |
| (C) under-specified | 11 |

The load-bearing items are **A1/A2/A3**, which are one issue seen three times:
the 3.2 N.m crossover is a synthetic constant read off a hard-picked row of the
E1 sweep, and it is presented as analytical, as a Go1 measurement, and as a
per-joint rule. **B3** shows the repair is cheap — the per-joint rule the paper
already states reproduces the shipped gate mask exactly on the Go1's own measured
statistics, so the constant can be demoted from claim to implementation detail
without changing a single result.
