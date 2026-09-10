# Where the Heat Actually Is: Decomposing Actuator Load in Learned Legged Locomotion, and What Command Filtering Can and Cannot Fix

*Draft. Simulation only. No hardware results yet; see §8 for exactly what is
and is not claimed.*

Artefacts: [`handson/10-actuator-thermal/`](../handson/10-actuator-thermal/)
· Literature: [`topics/actuator-thermal-load-in-learned-locomotion`](../topics/actuator-thermal-load-in-learned-locomotion.md)
· Vendor control stacks: [`docs/legged-rl-control-stack-reference`](legged-rl-control-stack-reference.md)

---

## Abstract

Reinforcement-learning locomotion policies are repeatedly reported to overheat
the motors of commercial quadrupeds and humanoids, in cases where the
manufacturer's own controller does not. The community's standard explanation is
command jitter, and the standard remedy is a smoothness penalty in the reward.
Neither the explanation nor the remedy has been measured against the alternative
explanation, which is that the learned policy simply chooses postures that
demand more steady torque.

We separate them. Copper loss decomposes exactly into a posture term and a
ripple term, and which dominates depends on the joint's steady torque. We locate
the crossover, and it falls **below** the steady torque of a Unitree Go1's
load-bearing joints and **above** that of its hips. Measured on the robot, the
ordering holds: the hip joints carry 33–45% of their heat in the band a filter
can reach, the knees 14–17%. The two camps in the literature are describing
different joints.

We then ask what can be recovered by filtering the command stream with the
policy weights frozen, the only intervention that applies to a checkpoint one
cannot retrain. A band decomposition of the joint torque gives the budget: for a
well-regularized policy, 31% of the heat is posture, 51% is the gait-band torque
walking requires, and **18% lies in bands a filter can reach**. For a policy
trained without effort terms the reachable share is **43%**. Notably, the
zero-order hold's own harmonics account for only 1.6%, so the staircase everyone
points at is not the waste; 5–25 Hz command content is.

A causal predictive filter, which estimates the samples a symmetric kernel would
need instead of waiting for them, recovers **about half of the reachable budget
on both policies**, 8.5% and 23.5% of total heat, with no falls. Open-loop replay
suggests far more is available, 46%, but that ceiling is unreachable in closed
loop and the standard route to it is actively harmful: obtaining zero phase by
delaying the command two policy steps *raises* heat and makes the robot fall, in
20% of episodes at 1.0 m/s on the well-regularized policy and in every episode on
the wasteful one. The policy is itself a feedback controller, and latency spent
on smoothing comes out of its stability margin. That failure is invisible at low
speed, where the same filter is the best scheme we tested.

We report two negative results that matter for practice. The `action_rate`
reward term, used across the field on the stated grounds that jitter burns
motors, is not the term that controls command-side thermal load in our
measurements; the effort terms are. And the joints that carry the body, which
dominate the heat budget in absolute terms, are the ones with the least to gain
from filtering.

---

## 1. Introduction

**The failure is real and it is documented.** On a Unitree G1 running a research
whole-body controller, both ankle-roll motors reach roughly 90 °C within five to
ten minutes, including while the robot is standing idle, whereas the
manufacturer's controller keeps the same joints at 30–45 °C on the same
machine.[1] On a Unitree A1 carrying 3 kg, a baseline learned policy
trips thermal protection in five to seven minutes.[2], [3] The reverse
also happens: on a Go2-W, a learned policy plateaus at 60 °C and walks for 51
minutes where the manufacturer's controller diverges to 84 °C and shuts down at
34 minutes.[4]

So "learned policies run hot" is not a true statement. What is true is that
**the policy's behaviour decides the thermal load**, and the field does not agree
on which property of the behaviour is responsible.

**Two explanations are in circulation, each with evidence.**

The *posture* explanation says the heat is set by the steady torque a posture
demands. Its strongest evidence is the Go2-W result, where both controllers run
at 50 Hz on the same robot at the same ambient temperature and the 24 °C
difference is attributed to the learned policy periodically lifting the feet, so
that hip angles return and torque varies instead of accumulating.[4]

The *jitter* explanation says the heat comes from high-frequency content in the
command, which raises RMS current without contributing net torque. Its
strongest evidence is the G1 ankle case above: at idle, the steady torque needed
to hold a posture is fixed by gravity, so a 45–60 K difference between two
controllers appears to require a difference in RMS current.[1] The only
quantitative support in the literature is CAPS, which reports a near-80%
reduction in power on a quadrotor after adding smoothness
regularization.[5]

**Neither camp has measured the two terms separately**, and the physics says
they must be separated, because copper loss adds them in quadrature:

$$
\mathcal{H} \;\propto\; \mathbb{E}[\tau^2] \;=\; \underbrace{\bar{\tau}^2}_{\text{posture}} \;+\; \underbrace{\mathrm{Var}(\tau)}_{\text{ripple}} .
$$

A zero-mean ripple contributes nothing to the mean torque and its full share to
the heat. But because the two add in quadrature, a large $\bar\tau$ drowns the
ripple term. Whether jitter matters is therefore not a property of the policy;
it is a property of the joint.

**This paper.** We (i) locate the crossover between the two regimes, (ii) measure
what a command-stream filter can recover with the policy frozen, in open and in
closed loop, and (iii) report where the intervention pays and where it does not.

**Contributions.**

1. **A band decomposition of actuator heat** into posture, gait, in-band jitter
   and staircase harmonics, exact by construction and computable from one
   baseline rollout. It gives the budget for any command-stream intervention:
   **18.1%** for a well-regularized Go1 policy and **43.3%** for one trained
   without effort terms. The zero-order hold's own harmonics are 1.6% of it.
2. **A load crossover, confirmed per joint on the robot.** Under fixed motion
   amplitude the heat penalty of out-of-band command power follows
   $\mathcal{H} = \mathcal{H}_0 + \kappa\,\rho/(1-\rho)$ to within 0.45%, and
   posture overtakes ripple at $\bar\tau \approx \sqrt{\mathrm{Var}(\tau)}$,
   which is 3.2 N·m for a Go1 joint. The joints below that line carry two to
   three times the filter-reachable heat share of the joints above it.
3. **The first measurement of what command filtering buys a frozen legged
   policy**, open loop and closed loop: a 46% ceiling that becomes 8.5%, a
   *small* effect by Cliff's delta.
4. **You cannot buy smoothness with latency.** The standard route to zero phase
   is counterproductive because the policy is itself a feedback controller. It
   is the best scheme at 0.5 m/s, falls in 20% of episodes at 1.0 m/s on the
   well-regularized policy, and in every episode on the wasteful one.
5. **A stable recovery ratio.** The predictive filter returns 47% and 54% of the
   reachable budget on two policies whose raw savings differ by a factor of
   three. Measuring the split therefore predicts what a filter will return before
   one is built.
6. **Two negative results.** `action_rate` is not the thermally relevant reward
   term in our measurements, and load-bearing joints are not worth filtering.

---

## 2. Background: where the command actually goes

Both major vendors place the PD controller on the motor driver, not in the
policy process. The policy emits a joint position target and the gains, and a
separate thread ships them down the bus. Unitree's own deployment code runs the
policy at 50 Hz and a `low_cmd_write` thread at 500 Hz; DeepRobotics' Lite3
deployment runs the policy at about 83 Hz over a 1 kHz state loop. Motor-side
control frequencies are 6 kHz for the Unitree GO-M8010-6 and 1 kHz for the
DeepRobotics J60, per their datasheets.[6]

Between the policy and the PD loop sits a zero-order hold. At 50 Hz into 500 Hz,
every target is held for ten ticks and then steps discontinuously. The
proportional term responds to that step immediately, producing a torque
transient of $k_p \Delta q$ before the joint has moved.

The magnitude of that transient is not small, and it can be derived from the
simulator configuration rather than assumed. In `unitree_rl_mjlab`, the G1 gains
are not hand-tuned; they are derived from a target closed-loop natural frequency
of 10 Hz and a damping ratio of 2.0, and the action scale is set to
$0.25\,e/k_p$ where $e$ is the effort limit. Substituting,

$$\tau_{\text{step}} = k_p \cdot 2 \cdot \frac{0.25\,e}{k_p} = 0.5\,e ,$$

so **a full-scale action reversal between two consecutive control steps commands
a transient equal to half the joint's torque limit, independently of the
joint.** With $\zeta = 2$ the dominant closed-loop pole sits at 2.68 Hz, while
the 50 Hz command stream carries content to 25 Hz.

---

## 3. Metrics

All are computed from quantities a deployed system already has.

**Band decomposition of the heat.** The central metric. Writing the joint torque
as a mean plus a zero-mean signal and using Parseval,

$$ \mathbb{E}[\tau^2] \;=\; \bar\tau^2 \;+\; \int_{0}^{f_g} S_\tau(f)\,\mathrm{d}f \;+\; \int_{f_g}^{f_{\text{Nyq}}} S_\tau(f)\,\mathrm{d}f \;+\; \mathbb{E}_k\!\left[\mathrm{Var}_{\text{sub}}(\tau)\right] , $$

with $f_g = 5$ Hz separating the gait band from in-band jitter and
$f_{\text{Nyq}} = 25$ Hz the command Nyquist. The last term collects everything
faster than one policy step, which is the zero-order hold's own harmonics. The
four terms sum to the heat proxy exactly, and **the last two are the entire
budget available to a command-stream filter.** We call their sum the *reachable
share*.

**Ripple fraction.** A coarser version used where the spectrum is unavailable:

$$ r \;=\; \frac{\mathrm{Var}(\tau)}{\bar{\tau}^2 + \mathrm{Var}(\tau)} . $$

It is computed per joint, because the answer differs per joint.

**Out-of-band command power.** Computable before deployment, from the command
stream alone, with no robot:

$$ \rho \;=\; \frac{\int_{f_c}^{f_{\text{Nyq}}} S_q(f)\,\mathrm{d}f}{\int_{0}^{f_{\text{Nyq}}} S_q(f)\,\mathrm{d}f} , $$

where $f_c$ is the joint's closed-loop $-3$ dB bandwidth.

**Heat proxy.** $\mathcal{H} = \frac{1}{T}\int_0^T \tau^2\,\mathrm{d}t$,
proportional to $I_{\text{rms}}^2 R$ for a constant torque constant. We use the
square, not the absolute value, because copper loss is a square law. This choice
matters: on the one motor for which we have a manufacturer figure, the winding
thermal time constant is 3.12 s against 777 s for the motor as a whole, a factor
of 249.[7] A driver-reported temperature therefore lags the winding badly, and
current is the better primary signal. Thermal characterization of legged
quasi-direct-drive actuators exists,[8] but we have not verified per-winding time
constants for the actuators used here, and §8 records that.

---

## 4. Experiments

Four experiments, in increasing order of realism and decreasing order of
control.

**E1 (single joint, synthetic command).** A second-order joint with Go1
parameters taken from the MuJoCo Playground model: $k_p = 35$, joint damping
0.5, armature 0.005 kg·m², giving $f_n = 13.3$ Hz, $\zeta = 0.60$ and a $-3$ dB
bandwidth of 15.3 Hz. The command is a 2 Hz sine of fixed amplitude plus
band-limited noise confined above the bandwidth, sampled at 50 Hz, zero-order
held onto 500 Hz, integrated at 5 kHz. Because the fundamental amplitude is
fixed, useful motion is matched by construction and $\rho$ can be swept alone.
A second axis sweeps a constant load torque.

**E2 (recorded commands, open loop).** Seven Go1 policies from a reward-ablation
study provide real command streams. We replay them through the same joint model
under different upsampling and filtering schemes. The reference for "motion that
must be kept" is the ZOH joint trajectory low-passed at 5 Hz, not the raw ZOH
trajectory, which would penalise the filter for removing the jitter it is
supposed to remove.

**E3 (closed loop, full robot).** A frozen Go1 policy in MuJoCo, policy at
50 Hz, physics at 500 Hz, ten substeps per policy step, with the filter inside
the loop so the policy observes the states the filtered commands produce.
Schemes: ZOH; linear interpolation; causal Butterworth at three cutoffs, with
and without linear interpolation; a symmetric kernel applied with two steps of
delay, which is genuinely zero-phase and genuinely deployable at the cost of
40 ms of latency; a predictive filter that estimates the samples the symmetric
kernel needs; and the predictive filter restricted by a per-joint gate.
Ten episodes of 500 steps per condition, three velocity commands, identical
seeds and observation-noise streams across schemes.

**E5 (band decomposition).** The same closed-loop setup under ZOH only, keeping
the per-policy-step mean torque as a 50 Hz series. Its power spectrum is split at
5 Hz and 25 Hz, and the within-step variance supplies the band above the command
Nyquist. The four bands sum to $\mathbb{E}[\tau^2]$ exactly, which we verify
against the independently computed heat proxy.

**Predictive filter.** An order-8 linear autoregressive model per joint,
fitted by least squares on the frozen policy's own ZOH rollouts, rolled forward
two steps so a symmetric Hann kernel can be applied without waiting. It needs no
simulator interaction beyond the calibration rollouts, no reward, and no
retraining. It is deliberately the simplest model that can close the phase gap,
so that its out-of-sample $R^2$ becomes the baseline a larger model must beat.

**Gate.** Joints whose measured mean $|\tau|$ on the ZOH baseline falls below the
E1 crossover are filtered; the rest are passed through untouched.

---

## 5. Results

### 5.1 The crossover exists and it is low

With motion amplitude held fixed, the heat penalty of out-of-band command power
depends almost entirely on how loaded the joint is.

| steady load | $\rho{=}0.02$ | $0.05$ | $0.10$ | $0.20$ | $0.40$ |
|---|---|---|---|---|---|
| 0 N·m | 1.81 | 3.08 | 5.38 | **10.86** | 27.30 |
| 2 N·m | 1.15 | 1.40 | 1.84 | 2.89 | 6.04 |
| 5 N·m | 1.03 | 1.08 | 1.16 | 1.36 | 1.96 |
| 10 N·m | 1.01 | 1.02 | 1.04 | 1.09 | 1.25 |
| 20 N·m | 1.00 | 1.00 | 1.01 | **1.02** | 1.06 |

*Heat relative to $\rho = 0$ at the same load. Three seeds.*

![E1 load sweep](pic/e1-rho-sweep.png)

*Left: with no load, heat climbs steeply with out-of-band command power while the
joint's motion does not. Middle: the relationship is exactly linear in
$\rho/(1-\rho)$. Right: a realistic steady load flattens it almost completely.*

At $\rho = 0.2$ an unloaded joint spends 10.9 times the heat while moving 1.8%
more; a joint holding 20 N·m spends 2% more. Unloaded, the relationship is
exactly $\mathcal{H} = \mathcal{H}_0 + \kappa\,\rho/(1-\rho)$ with
$\kappa = 38.4$ and a maximum residual of 0.45%. The crossover, where
$\bar\tau^2 = \mathrm{Var}(\tau)$, sits at 3.2 N·m.

**Measured on the robot, this line falls between the joint types.** On the
frozen Go1 policy the mean $|\tau|$ per joint is 1.24–1.48 N·m at the hip
abduction joints, 2.00–2.59 N·m at the thighs and 5.88–6.88 N·m at the calves.
Eight of twelve joints sit below the crossover; the four knees sit above it.
§5.4 confirms the ordering directly: the joints below the line carry two to three
times the filter-reachable heat share of the joints above it.

> This resolves the contradiction in §1. The posture camp and the jitter camp are
> both right, about different joints. The G1 case that motivates the jitter
> explanation is an **ankle roll**, one of the least statically loaded joints in
> a humanoid leg.

### 5.2 `action_rate` is not the thermally relevant reward term

Command-side statistics of seven ablated Go1 policies, computed from the command
stream alone:

| variant | $\Delta q_{\text{rms}}$ (rad) | $\tau_{\text{step}}$ rms (N·m) | spectral centroid (Hz) | $\rho_{>5\,\mathrm{Hz}}$ |
|---|---|---|---|---|
| full reward | 0.0668 | 2.34 | 3.29 | 0.051 |
| no `action_rate` | 0.0723 | 2.53 (+8%) | 3.43 | 0.119 |
| no gait terms | 0.0747 | 2.62 (+12%) | 4.98 | 0.146 |
| no `torques`+`energy` | 0.1267 | **4.43 (+89%)** | 5.44 | 0.363 |
| task only | 0.1586 | **5.55 (+137%)** | 10.44 | 0.898 |

![command spectrum](pic/command-spectrum.png)

*Left: command power spectra. The unregularized policy is flat across the whole
band, that is, white. Right: the joint's closed-loop response, which cannot
follow the grey region.*

Removing `action_rate` alone raises the staircase transient by 8%. Removing the
effort terms raises it by 89%. The field uses `action_rate` and `dof_acc` with
the stated justification that jitter burns motors; on this evidence they are not
the terms doing that work. *Single seed, 8 s per variant; this is a direction,
not a hypothesis test.*

### 5.3 Open loop, the ceiling is high and the cost is phase

Replaying the full-reward policy's recorded commands through the unloaded joint:

| scheme | heat $H/H_{\text{ZOH}}$ | motion kept | deployable |
|---|---|---|---|
| ZOH | 1.000 | 1.000 | — |
| linear interpolation | 0.696 | 0.819 | yes |
| causal Butterworth 12 Hz | 0.588 | 0.745 | yes |
| predictive filter | 0.428 | 0.741 | yes |
| **zero-phase 12 Hz** | **0.540** | **0.995** | no |

The last two rows are the point. Zero-phase filtering removes 46% of the heat
while reproducing the in-band trajectory to 0.0004 rad. The causal version of
**the same filter at the same cutoff** keeps only 74.5%. The difference is phase
lag alone, and it costs 25.0, 24.8 and 22.9 percentage points of motion at 0, 2
and 10 N·m respectively.

![E3 open loop](pic/e3-command-filters.png)

*Up and to the left is better. The zero-phase curve dominates the causal one at
every cutoff, and the gap between them is phase lag alone.*

The predictor's out-of-sample $R^2$ is 0.679 on the full-reward policy, 0.385
without the effort terms and 0.059 with no regularization at all. **The policies
whose commands most need filtering are the ones whose commands are least
predictable**, which bounds how much any causal scheme can recover.

### 5.4 How much of the heat is a filter even allowed to touch?

Before asking how well a filter works, we ask what its budget is. Decomposing
the joint torque of the frozen policy under ZOH into a mean and a power
spectrum, and adding the within-policy-step variance as the band above the
command Nyquist:

| band | meaning | share of heat |
|---|---|---|
| DC | posture: holding the body up | 31.3% |
| 0–5 Hz | gait: the torque modulation walking requires | 50.6% |
| **5–25 Hz** | **in-band jitter, reachable by a low-pass** | **16.5%** |
| **> 25 Hz** | **the ZOH staircase's own harmonics** | **1.6%** |
| | **total reachable by a command filter** | **18.1%** |

*Thirty rollouts, three velocity commands. The four bands sum to
$\mathbb{E}[\tau^2]$ exactly, and the total, 255.9, matches the independently
computed heat proxy of 256.2 to 0.1%.*

Two things follow immediately.

**The zero-order hold is not the problem.** The staircase's own harmonics carry
1.6% of the heat. The intuition that motivated this investigation, that holding
a target for ten ticks and then stepping discontinuously wastes energy, is
correct in mechanism and small in magnitude. **The waste is in the 5–25 Hz
command content instead**, which is ten times larger and which no amount of
interpolation will remove.

**The budget is joint-dependent, and the crossover predicts the ordering.**

| joint | mean $|\tau|$ (N·m) | DC | gait | 5–25 Hz | > 25 Hz | **reachable** |
|---|---|---|---|---|---|---|
| hips | 1.2–1.5 | 7–21% | 47–54% | 33–45% | 0.3–0.4% | **33–45%** |
| thighs | 2.0–2.6 | 2–19% | 56–78% | 16–29% | 3.0–4.0% | **20–32%** |
| calves | 5.9–6.9 | 32–38% | 45–54% | 13–16% | 1.1–1.6% | **14–17%** |

The hips, which sit furthest below the 3.2 N·m crossover, have the largest
reachable share; the calves, which sit above it, have the smallest. **This is the
ordering §5.1 predicts**, and it is what the per-joint gate in §5.5 exploits.

### 5.5 Closed loop: the ceiling is not reachable, and the obvious route to it backfires

Frozen Go1 policy, filter inside the loop, ten episodes of 500 steps at each of
three velocity commands, identical seeds and observation-noise streams
across schemes ($n = 30$ per scheme).

| scheme | $H/H_{\text{ZOH}}$ | vel err $/$ ZOH | $\delta(H)$ | $\delta(\text{vel})$ | fall rate |
|---|---|---|---|---|---|
| ZOH | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| linear interpolation | 0.956 | 1.048 | −0.31 | 0.13 | 0.00 |
| Butterworth 5 Hz | 1.098 | 2.594 | −0.04 | 1.00 | **0.23** |
| Butterworth 8 Hz | 0.946 | 1.444 | −0.28 | 0.93 | 0.00 |
| Butterworth 12 Hz | 0.935 | 1.160 | −0.28 | 0.52 | 0.00 |
| Butterworth 12 Hz + linear | 0.942 | 1.284 | −0.26 | 0.75 | 0.00 |
| **zero-phase via 40 ms delay** | **1.082** | **2.493** | −0.03 | 1.00 | **0.07** |
| **predictive filter** | **0.915** | 1.279 | −0.33 | 0.71 | 0.00 |
| **predictive + per-joint gate** | 0.940 | **1.157** | −0.33 | **0.50** | 0.00 |
| predictive, policy unaware | 0.927 | 1.236 | −0.33 | 0.64 | 0.00 |

*$\delta$ is Cliff's delta against ZOH; negative means lower. Fall rates are
noisy at $n = 30$: across two independent runs of the whole matrix the two
aggressive schemes fell between 7% and 30% of the time and no other scheme ever
fell. Everything else reproduces to about 0.3%, limited by GPU
non-determinism.*

![closed loop](pic/e4-closed-loop.png)

**The open-loop ceiling does not survive.** The best closed-loop scheme removes
8.5% of the heat, against 46% open loop. Cliff's delta of −0.33 is a *small*
effect by the usual convention, and we report it as such.

**Set against the budget, it is about half.** §5.4 puts the filter-reachable
heat at 18.1%; the predictive filter recovers 8.5%, which is 47% of what is
available. That is a more useful way to state the result than the raw
percentage, and §5.6 shows the ratio is stable across policies.

**The ZOH staircase is not where the saving comes from.** Tracking the
above-Nyquist term separately:

| scheme | > 25 Hz term, absolute | total $H/H_0$ |
|---|---|---|
| ZOH | 4.02 | 1.000 |
| Butterworth 5 Hz | **1.96** | **1.098** |
| zero-phase via delay | **2.34** | **1.082** |
| Butterworth 12 Hz | 2.93 | 0.935 |
| **predictive filter** | **4.02** | **0.915** |

The two schemes that halve the staircase term have the *highest* total heat, and
the scheme with the lowest total heat leaves it untouched. The saving comes from
the 5–25 Hz band, not from the staircase, which is consistent with §5.4 putting
only 1.6% of the heat there. **Interpolating between commands, the intervention
the zero-order hold most obviously invites, is not worth doing.**

**The scheme that wins open loop is the worst closed loop.** Zero-phase
filtering obtained by delaying the command two policy steps was the best
open-loop scheme in §5.3. In closed loop it *increases* heat by 8.5% and makes
the robot fall in 13% of episodes. The 5 Hz causal Butterworth fails the same
way. Broken out by command:

| scheme | 0.5 m/s | 1.0 m/s | 0.3 m/s + turn |
|---|---|---|---|
| zero-phase via delay, $H/H_0$ | 0.883 | **1.258** | 0.992 |
| zero-phase via delay, fall rate | 0.00 | **0.20** | 0.00 |
| Butterworth 5 Hz, $H/H_0$ | 0.880 | **1.295** | 0.991 |
| Butterworth 5 Hz, fall rate | 0.00 | **0.70** | 0.00 |
| predictive + gate, $H/H_0$ | 0.952 | **0.924** | 0.956 |
| predictive + gate, vel err $/$ ZOH | 1.298 | **0.995** | 1.250 |

At 0.5 m/s the delayed filter looks excellent, removing 12% of the heat with no
falls at all. At 1.0 m/s it falls in a fifth of episodes, and the 5 Hz
Butterworth in seven tenths of them. Falling is expensive: the heat proxy rises
26 to 30% above baseline. **A filter evaluated only at low speed will be
reported as a success and will fail on deployment.**

The explanation is that the policy is itself a feedback controller. Adding 40 ms
of latency inside its loop degrades the feedback it depends on, and the failure
appears where the loop gain matters most, which is at speed. Open-loop replay
cannot show this, because there is no loop.

**Prediction is the only route we found that does not pay latency.** The
predictive filter gives the largest heat reduction with no falls. Gating it by
joint load gives up a little heat for a much better tracking trade: at 1.0 m/s
it removes 7.5% of the heat while tracking *slightly better* than ZOH.

**Whether the policy sees the filtered or the raw action barely matters**
(0.917 vs 0.927 heat, 1.249 vs 1.214 tracking). The filter can be installed
without touching the observation pipeline.

### 5.6 A wasteful policy has more to recover, and the same fraction of it comes back

We repeated the whole closed-loop matrix on a second frozen policy, trained
identically except that the `torques` and `energy` reward terms were removed.
It is the variant §5.2 identified as the jittery one.

| | full reward | no effort terms |
|---|---|---|
| ZOH heat | 255.9 | **376.3** (+47%) |
| DC / gait / 5–25 Hz / > 25 Hz | 31.3 / 50.6 / **16.5** / 1.6 | 27.0 / 29.7 / **40.7** / 2.6 |
| **reachable by a filter** | **18.1%** | **43.3%** |
| command predictor $R^2$ | 0.609 | 0.402 |
| predictive filter, $H/H_0$ | 0.915 | **0.765** |
| **recovered / reachable** | **47%** | **54%** |
| predictive + gate, $H/H_0$ | 0.940 | **0.879** |
| predictive + gate, vel err $/$ ZOH | 1.157 | 1.149 |
| zero-phase via delay, $H/H_0$ | 1.082 | 1.053 |
| zero-phase via delay, fall rate | 0.07 | **0.37** |

Three things follow.

**Dropping the effort terms moves 25 percentage points of heat into the
filter-reachable band**, from 18.1% to 43.3%, and raises the total by 47%. This
is the clearest quantitative statement we can make about the thermal cost of
reward design, and it is the effort terms that buy it. §5.2 showed `action_rate`
does not.

**The filter recovers about half of whatever is reachable, on both policies**:
47% and 54%. The raw reduction differs by a factor of nearly three, 8.5% against
23.5%, but the ratio is stable. That makes the §5.4 budget a usable predictor:
measure the band split on a baseline rollout, halve the reachable share, and you
have an estimate of what a filter will return before building one.

**Predictability is not the binding constraint.** The jittery policy is harder to
predict, $R^2 = 0.402$ against 0.609, and yet more is recovered from it, because
it has more to give. Our earlier expectation that the predictor's accuracy would
bound the benefit was wrong, and we report it as such.

![second policy](pic/e4-closed-loop-no-torques-energy.png)

*The same closed-loop matrix on the policy trained without the effort terms. The
predictive filter moves much further left; the delayed filter moves off the chart
to the right.*

Gating remains the best operating point on both policies: on the wasteful one it
removes 12.1% of the heat for a 15% tracking cost, against the ungated filter's
23.5% for a 59% tracking cost. And the delayed filter's failure gets worse, not
better, on the policy that most needs help: it falls in 37% of episodes overall
and in **every** episode at 1.0 m/s.

---

## 6. Discussion

**The community's causal story survives, with the numbers attached.** The story
is: the policy's commands jitter, jitter raises RMS current, current is heat,
therefore smooth the commands. Every link holds. What was missing was the size of
the prize, and it turns out to be strongly dependent on the policy: 18.1% of the
heat sits in the filter-reachable band for a well-regularized policy and 43.3%
for one trained without effort terms. About half of that is recoverable in closed
loop without falls.

**But the part everyone points at is the wrong part.** The zero-order hold is the
visible defect: a target held for ten ticks, then a discontinuous step, and a
proportional term that responds to the step before the joint moves. That
staircase carries 1.6% of the heat. Interpolating it away, which is the obvious
fix and the one several deployed systems implement, buys almost nothing, and our
linear-interpolation condition confirms it at 4% of total heat. **The waste is
one decade lower in frequency**, in 5–25 Hz command content that a low-pass
reaches and an interpolator does not.

**Reward design is the larger lever, and not through the term people use for
it.** Removing the effort terms moved 25 percentage points of heat into the
reachable band and raised the total by 47%. Removing `action_rate` did neither in
any measurement we made. If the goal is to protect motors, our evidence says
penalise effort, not action difference.

**The 80% figure from the quadrotor literature does not transfer, and we can say
why.** A rotor has almost no steady holding torque, so its $\bar\tau^2$ term is
small, its gait-band term does not exist, and the ripple term dominates by
default. A stance-phase knee is the opposite case in every one of those three
respects.

**For a learned policy you cannot buy smoothness with latency.** This is the
most transferable finding. Every deployed smoothing filter we found in the
literature is causal, and the standard way to remove a causal filter's phase lag
is to delay. Our closed-loop result says that trade is not available here: the
policy is a feedback controller, and the latency you spend comes out of its
stability margin. It shows up as falls at speed, not as degraded tracking at low
speed, so it is easy to miss.

**We expected predictability to bound the benefit, and it does not.** The
predictive filter must estimate the samples a symmetric kernel would need, and
the out-of-sample $R^2$ of an order-8 linear model falls from 0.609 on the
well-regularized policy to 0.402 on the wasteful one and 0.059 on a completely
unregularized one. We predicted that the benefit would fall with it. It rose
instead, from 8.5% to 23.5%. This is consistent with the dissociation above: if
the saving does not come from reconstructing the command accurately, then the
accuracy of the reconstruction is not what limits it. We do not have a
mechanistic account of what does, and we think obtaining one is the most useful
next piece of analysis.

**Practical guidance, in order of confidence.**

1. **Measure the band split before doing anything.** It costs one baseline
   rollout, it tells you the size of the prize, and halving the reachable share
   estimates what you will actually get. If the reachable share is small, no
   command filter will save your motors and the effort belongs on the reward or
   the posture instead.
2. **Do not obtain zero phase by delaying**, and do not use aggressive cutoffs.
   Evaluate any filter at the top of the speed range, not the bottom, because
   that is the only place the failure appears.
3. **Gate by joint load** if you filter at all. The threshold is
   $\bar\tau \approx \sqrt{\mathrm{Var}(\tau)}$. Gating cost us 2.5
   percentage points of heat and bought back most of the tracking.
4. **Expect single-digit percentage gains** on a loaded quadruped, and do not
   attribute them to jitter removal without checking the split.
5. If you add a smoothness term to the reward hoping to protect the motors, our
   measurements suggest the effort terms do that work, not `action_rate`.

**Where this should pay much better.** The regime the decomposition says is
favourable, very low steady torque, is not represented on a walking quadruped.
It is exactly the condition of the reported G1 failure: an ankle-roll joint on a
standing humanoid. That is where the hardware experiment should look first, and
it is a prediction this work makes rather than a result it demonstrates.

---

## 7. Related work

**Thermal-aware learned locomotion.** Two 2026 papers add motor temperature to
the observation and thermal constraints to the reward,[2] then move the
correction into a residual policy conditioned on thermal state.[3] Both
retrain, both report runtime before overheating as the metric, and neither
separates $\bar\tau^2$ from $\mathrm{Var}(\tau)$. Earlier, thermal
management by posture was posed as a contact-constrained inverse-kinematics
problem on a humanoid.[9] Our intervention is complementary: it does
not retrain, and it asks a different question, namely how much of the command
stream is waste.

**Interfaces between a slow policy and a fast actuator.** Actuator Reality
Shaping equips each joint with a two-degree-of-freedom controller that shapes
the physical actuator to match the idealized second-order reference dynamics of
simulation;[10] high-frequency linear feedback at up to 40 kHz has been used
to interpolate learned torque policies;[11] and commanded accelerations
have been constrained to a voltage-realizable set.[12] All three occupy the
architectural position we use. None of them targets electrical loss, and all
three are designed rather than learned. Post-optimization of chunked action
sequences exists for manipulation, combining overlapping-chunk scheduling, linear
blending and jerk-minimizing optimization,[13] as does a B-spline action
representation that builds smoothness into the policy output rather than
filtering it afterwards.[14] On cost-constrained quadrupedal hardware, a measured
transport delay above 50 ms is handled with a forward model of the average
actuator delay and a time-aware network, which is a compensation strategy rather
than a filtering one.[15]

**Smoothness regularization.** CAPS,[5] Lipschitz-constrained
policies[16] and spectral normalisation[17] all reduce
high-frequency content; a benchmark study reports that its best hybrid method
improves control smoothness by 26.8% over the baseline for a worst-case
performance degradation of 2.8%.[18] None reports motor current or temperature
on a legged robot.

**Control rate.** Robust locomotion has been demonstrated at 8 Hz on ANYmal C,
with low-rate policies shown to be *less* sensitive to actuation latency;[19]
Q-learning is known to degenerate as the time step shrinks;[20] and
policies that choose their own action durations reduce average control frequency
without losing reward.[21] None of this literature reports a thermal or
electrical quantity as a function of control rate. To our knowledge no paper
reports motor temperature or current as a function of policy control rate.

---

## 8. Limitations

**Eleven of the eighteen cited papers are represented in our notes by their
abstract only.** References [8], [9], [10], [12], [13], [14], [15], [16], [17],
[20] and [21] have not been read in full by the authors, so every statement we
make about them is restricted to what their abstracts assert. We caught two
errors this way during preparation, one of which had a specific numeric claim
attached, and we cannot rule out others. Related-work claims in §7 should be
read as abstract-level summaries, not as verified accounts of method detail.

**No hardware.** Everything here is simulation. The heat proxy $\mathbb{E}[\tau^2]$
is proportional to copper loss only under a constant torque constant, and it
ignores iron loss, inverter loss and every thermal path. The claims are about a
*proxy*, and the reader should read them that way until the bench experiment in
§9 is done.

**The single-joint model has no contact.** E1 and E2 use a second-order joint
with a constant load. Real stance-phase torque is neither constant nor
independent of the command.

**One robot, one gait family, one policy family.** All policies come from the
same training setup on flat terrain.

**The 5 Hz band split is a modelling choice, and the headline number depends on
it.** "Gait" versus "jitter" is defined by a low-pass at 5 Hz, chosen because the
gait fundamental is near 2 Hz. A lower boundary would make the reachable share
larger and a higher one smaller. The 25 Hz boundary, by contrast, is not a
choice: it is the command Nyquist. A companion analysis that swept the 5 Hz line
would make the budget claim considerably more solid, and we have not done it.

**"Reachable" is an upper bound, not an achievable target.** Some 5–25 Hz torque
content is the robot's genuine response to contact, not command jitter, and
removing it costs stability. The recovery ratio of about a half may be exactly
this: the fraction of the band that is truly discretionary.

**The reward-ablation evidence is single-seed.** §5.2 reports a direction, not a
test.

**Success is measured as tracking, not as task completion over long horizons.**
A filter that preserves in-band trajectory can still remove the high-frequency
response a robot needs at touchdown, and 500-step episodes may not expose that.

---

## 9. What would settle this

Ordered by cost.

1. **Measure the driver's current telemetry rate and resolution.** One
   afternoon. If the reported rate cannot resolve the ripple band, the central
   hardware measurement is impossible and everything downstream must change.
2. **Single-joint thermal bench.** One motor, a thermocouple, an adjustable
   weight, a current probe. Sweep $\rho$ against steady load and check that the
   crossover appears where E1 puts it. This is the cheapest falsification of the
   central claim.
3. **Posture-matched current comparison on hardware.** Hold a learned policy and
   the manufacturer's controller at the *same* posture and compare
   $I_{\text{rms}}/\bar{I}$ per joint. This is the experiment the literature is
   missing, and it decides the argument in §1. A null result is publishable: it
   would show the reported overheating is posture, not jitter.
4. **Long-horizon closed-loop evaluation** over rough terrain and perturbations,
   to test whether preserving in-band motion is sufficient for task success.

---

## References

[1] NVIDIA Research, "Ankle roll motors overheating on G1," GR00T-WholeBodyControl, GitHub issue #174, 2026. [Online]. Available: https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174. Not peer reviewed.

[2] L. Qian, Y. Wan, S. Wang, and X. Luo, "Learning thermal-aware locomotion policies for an electrically-actuated quadruped robot," arXiv:2603.01631, 2026.

[3] Y. Wan et al., "Learning to balance motor thermal safety and quadrupedal locomotion performance with residual policy," arXiv:2605.27046, 2026.

[4] T. Matsuzawa, K. Irie, T. Yoshida, T. Suzuki, Y. Hara, and M. Tomono, "Long-distance real-world navigation of the legged-wheeled robot Go2-W using deep reinforcement learning," arXiv:2606.21387, 2026.

[5] S. Mysore, B. Mabsout, R. Mancuso, and K. Saenko, "Regularizing action policies for smooth control with reinforcement learning," in *Proc. ICRA*, 2021. arXiv:2012.06644.

[6] Vendor control-stack reference, this repository, docs/legged-rl-control-stack-reference.md. Collects the source lines and datasheet entries behind the frequencies quoted in Section 2.

[7] maxon motor ag, "EC max 30 Ø30 mm, brushless, 60 W," datasheet, 2024. Winding thermal time constant 3.12 s; motor thermal time constant 777 s.

[8] K. Urs, C. E. Adu, E. J. Rouse, and T. Y. Moore, "Design and characterization of 3D printed, open-source actuators for legged locomotion," arXiv:2202.12395, 2022.

[9] S. J. Jorgensen, J. Holley, F. Mathis, J. S. Mehling, and L. Sentis, "Thermal recovery of multi-limbed robots with electric actuators," arXiv:1902.00187, 2019.

[10] S. Yamamori et al., "Actuator reality shaping for zero-shot sim-to-real robot learning," arXiv:2607.02205, 2026.

[11] R. Kourdis et al., "Very high frequency interpolation for direct torque control," arXiv:2509.24175, 2025.

[12] L. Zhang et al., "VRA: Grounding discrete-time joint acceleration in voltage-constrained actuation," in *Proc. RSS*, 2026. arXiv:2605.10696.

[13] D. Son, and S. Park, "LiPo: A lightweight post-optimization framework for smoothing action chunks generated by learned policies," arXiv:2506.05165, 2025.

[14] X. Han et al., "B-spline policy: Accelerating manipulation policies via B-spline action representations," arXiv:2607.09648, 2026.

[15] J. C. Weddington, B. P. Ölveczky, and S. A. Baccus, "Reinforcement learning on cost-constrained quadrupedal hardware," arXiv:2607.26434, 2026.

[16] Z. Chen et al., "Learning smooth humanoid locomotion through Lipschitz-constrained policies," arXiv:2410.11825, 2024.

[17] J. Shin, W. Cha, D. Kim, J. Cha, and J. Park, "Spectral normalization for Lipschitz-constrained policies on learning humanoid locomotion," arXiv:2504.08246, 2025.

[18] G. Christmann, Y. Luo, H. Mandala, and W. Chen, "Benchmarking smoothness and reducing high-frequency oscillations in continuous control policies," in *Proc. IROS*, 2024. arXiv:2410.16632.

[19] S. Gangapurwala, L. Campanaro, and I. Havoutis, "Learning low-frequency motion control for robust and dynamic robot locomotion," in *Proc. ICRA*, 2023. arXiv:2209.14887.

[20] C. Tallec, L. Blier, and Y. Ollivier, "Making deep Q-learning methods robust to time discretization," arXiv:1901.09732, 2019.

[21] A. Sukhija, L. Treven, J. Cheng, F. Dörfler, S. Coros, and A. Krause, "TARC: Time-adaptive robotic control," arXiv:2510.23176, 2025.
