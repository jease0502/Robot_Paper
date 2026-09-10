# Single-joint thermal bench: protocol

Closes **Appendix A.4**, and is the cheapest falsification of the paper's central
claim. Everything below is written to be executed by one person in a room with a
door that shuts.

**What is being tested.** §5.1 predicts that the heat penalty of out-of-band
command power collapses as the joint's steady load rises, crossing over at
$\bar\tau \approx 3.2$ N·m. That prediction comes from a second-order joint model
with Go1 parameters. It has never been checked against a real motor.

**The prediction is asymmetric, and that is what makes it testable cheaply.**

| steady load | predicted heat at $\rho = 0.2$, relative to $\rho = 0$ |
|---|---|
| 0 N·m | **10.9×** |
| 2 N·m | 2.89× |
| 5 N·m | 1.36× |
| 10 N·m | 1.09× |
| 20 N·m | **1.02×** |

A tenfold effect is visible on almost any instrument. A 2% effect is not. So the
experiment is really two claims with very different measurement demands: *find
the large effect where it is predicted*, and *fail to find it where it is
predicted to be absent*. The second half is the one that can falsify the paper,
and it needs a stated detection floor rather than a null result by default.

---

## 1. Instrumentation, and which signal is primary

| Signal | Role | Why |
|---|---|---|
| **PSU input power** (or clamp current × bus voltage) | **primary** | Instantaneous. Copper loss is what the paper is about, and electrical input measures it without waiting for heat to move |
| Thermal camera | endpoint validation + spatial | Slow, but it is the quantity anyone actually cares about, and it shows *where* the heat appears |
| Joint angle feedback | **the control variable** | Proves the motion is matched across conditions. Without it the experiment means nothing |

**Temperature is not the primary metric, and this is not a compromise.** The
winding thermal time constant is seconds while the housing is minutes; a surface
measurement is a heavily lagged, heavily smoothed version of the thing we want.
Electrical input power has no such lag. Temperature enters at the end, to confirm
that the power differences we measure do become the temperature differences that
matter.

### Thermal camera: the one thing that will ruin the data

Bare aluminium has an emissivity around 0.05–0.10. A camera set to $\varepsilon =
0.95$ pointed at a bare motor can read tens of degrees wrong, and worse, the error
*changes with temperature and viewing angle*, so it does not cancel in a
comparison.

**Before any measurement**: apply matte black tape or high-emissivity paint to
three patches, and take every reading from those patches only.

- **P1 stator can** — the outside of the motor body, over the windings
- **P2 gearbox housing** — over the reduction stage
- **P3 output bearing / flange**

Set the camera to the tape's stated emissivity, note it in the log, and never
change it. Keep the camera on a tripod at a fixed distance and angle; move
nothing between conditions.

Recording all three patches separately is worth the effort. If the paper's
mechanism is right, the ripple-driven heat appears at **P1 first** and conducts
outward. If P2 leads P1, the heat is coming from the gearbox, not the windings,
and the copper-loss framing needs revisiting. That is a result either way.

---

## 2. Mechanical setup

```
     motor fixed to bench
          |
          |==========[ lever arm, length L ]==========o  hanging mass m
          |
     angle zero = arm horizontal
```

**Put the oscillation centre at the horizontal.** Gravity torque is
$\tau = m g L \cos\theta$, so at $\theta = 0$ the torque is at maximum *and*
$\mathrm{d}\tau/\mathrm{d}\theta = 0$. Over a $\pm 0.2$ rad swing the load varies
by only 2%, which keeps $\bar\tau$ genuinely constant across the motion. Centring
anywhere else makes the load swing with the command and confounds the whole load
axis.

**Load levels.** Choose $m$ and $L$ to hit $\bar\tau \in \{0, 2, 5, 10, 20\}$ N·m.
Verify each by a second route: command the joint to hold still at $\theta = 0$
and read the steady current. $\bar\tau = k_t I$ gives an independent check on
$mgL$. Log both. If they disagree by more than 10%, stop and find out why before
collecting anything.

**Guarding.** A 20 N·m load on a swinging arm is a hazard. Cage it, or run it
inside a box, and never stand in the swing plane.

---

## 3. Stimulus

**Do not use a policy.** A policy changes its posture when you change its reward
or its noise, which moves $\bar\tau$ and destroys the comparison. Inject synthetic
commands so that $\rho$ can be swept with everything else held.

Generate the command **at 50 Hz and zero-order hold it to the bench loop rate**,
exactly as the simulation does, so the bench and §5.1 are measuring the same
object.

$$q_{\text{des}}[k] = q_0 + A \sin(2\pi f_0 t_k) + n[k]$$

- $f_0 = 2$ Hz, $A = 0.20$ rad — the useful motion, **held constant everywhere**
- $n[k]$ — noise band-limited to $[f_{\text{bw}}, 25]$ Hz, where $f_{\text{bw}}$ is
  the joint's measured $-3$ dB bandwidth (§4 below). Scale its variance to hit the
  target $\rho$
- $\rho \in \{0, 0.05, 0.10, 0.20, 0.40\}$

$$\rho = \frac{\int_{f_{\text{bw}}}^{f_{\text{Nyq}}} S_q(f)\,\mathrm{d}f}{\int_0^{f_{\text{Nyq}}} S_q(f)\,\mathrm{d}f}$$

Use the **same noise realisation** (same random seed) at every load level, so the
load axis and the noise axis are independent.

`data/scripts/e1_sim_rho_sweep.py` already generates exactly this signal. Reuse
its generator rather than writing a second one, so the bench stimulus and the
simulated stimulus cannot drift apart.

---

## 4. Before the sweep: three checks that can save you a day

**4.1 Measure the actual bandwidth.** The simulation assumed 15.3 Hz from armature
alone, and the paper flags that as an *upper bound* because it ignores link
inertia. With the arm and mass attached, sweep a small-amplitude chirp and find
the $-3$ dB point from the angle feedback. **Use the measured value to define the
noise band**, not 15.3 Hz. If the real bandwidth is much lower, every $\rho$ in
the simulation was understated, which is itself worth reporting.

**4.2 Measure the noise floor of your power measurement.** Hold the joint still at
$\theta = 0$ under each load, log input power for 60 s, and compute the standard
deviation. **This number is your detection floor.** It decides whether the
1.02× prediction at 20 N·m is testable at all. Write it down before you look at
any experimental condition — otherwise a null result is indistinguishable from an
insensitive instrument.

**4.3 Pilot the extremes.** Unloaded, run $\rho = 0$ and $\rho = 0.40$ only. The
prediction is a factor of roughly 27 in the ripple term. If you cannot see a large,
obvious difference in input power here, something is wrong with the rig and the
full sweep will waste a day. Fix it first.

---

## 5. The sweep

A 5 × 5 grid: $\rho \in \{0, 0.05, 0.10, 0.20, 0.40\}$ crossed with
$\bar\tau \in \{0, 2, 5, 10, 20\}$ N·m.

**Per condition, 3 minutes:**

1. Confirm the motor has returned to baseline: P1 within **±1 °C** of the
   pre-session ambient. This is the slowest part of the protocol and it cannot be
   skipped.
2. Log ambient temperature.
3. Run the stimulus for 180 s. Log continuously: input power, joint angle, and a
   thermal frame every 10 s.
4. Stop. Record the peak P1, P2, P3.
5. Cool to baseline before the next condition.

**Order matters more than you expect.** Room temperature drifts, and the motor
accumulates heat across a session. Two defences, use both:

- **Randomise the condition order** within each session.
- **Counterbalance in ABBA blocks** for the pairs you care about most, so a linear
  drift cancels: run $\rho{=}0$, $\rho{=}0.2$, $\rho{=}0.2$, $\rho{=}0$ back to
  back at a fixed load.

**Three repeats per condition, on different days if possible.** Report mean ± s.d.
Three is the minimum that gives an interval at all; five is better if the cooldown
budget allows.

### Extracting the metrics

- $P_{\text{net}}$ = mean input power over the run, **minus the holding power at
  the same load** measured in 4.2. The subtraction removes the driver's quiescent
  draw and the gravity-holding term, leaving the motion-and-ripple term the paper
  is about.
- $A_{\text{meas}}$ = the amplitude of the 2 Hz component of the *measured* angle.
  **This must match across conditions.** If it drifts by more than 5%, the motion
  is not matched and that condition is void. Report it in every table; it is the
  control that makes "same motion, different heat" a legitimate claim.
- $\Delta T_{180}$ = P1 rise over the 180 s run.
- $\mathrm{d}T/\mathrm{d}t$ = slope of P1 fitted over 60–180 s. In this window the
  housing rise is close to linear in dissipated power, which makes it a usable
  fast proxy for what a 52-minute steady-state run would tell you.

---

## 6. The steady-state runs

The 3-minute runs give the shape of the effect. They do not give a temperature
anyone would quote. Pick **four corner conditions** and run each to steady state,
about 50 minutes, or until P1 changes by less than 0.5 °C in 10 minutes:

| | $\rho = 0$ | $\rho = 0.4$ |
|---|---|---|
| **0 N·m** | corner A | corner B |
| **20 N·m** | corner C | corner D |

The paper's claim reduces to: **B − A is large, D − C is small.** Four runs, one
afternoon, and it is the headline figure.

---

## 7. Pre-registered predictions

Write these down before collecting, and report the outcome against them whatever
it is.

1. $P_{\text{net}}(\rho)$ rises with $\rho$ at every load, and the **slope falls
   monotonically as load rises**.
2. Fitting $P = P_0 + \kappa\,\rho/(1-\rho)$ at zero load holds to within a few
   percent. This closed form fitted the simulation to 0.45%; on hardware, iron
   loss and friction will degrade it, and by how much is a result.
3. The crossover, where the $\rho$-sensitivity has fallen by half, lands between
   **2 and 5 N·m**.
4. Corner B exceeds corner A by a large, unambiguous margin. Corner D exceeds
   corner C by less than the detection floor from 4.2.
5. In the thermal frames, **P1 leads P2**.

## What falsifies the paper

- **$\rho$-sensitivity does not fall with load.** The quadrature argument is the
  spine of the paper. If loading a joint does not suppress the ripple penalty, §5.1
  is wrong and everything resting on it goes with it.
- **The crossover is far from 3.2 N·m** — say below 0.5 or above 15. The mechanism
  might survive but every per-joint recommendation in §6 would need redoing.
- **D − C is large.** That would mean heavily loaded joints *do* suffer from
  command jitter, which would invalidate the gating rule and make the paper's
  practical advice actively wrong.
- **P2 leads P1.** The heat would be coming from the gearbox, not the windings.
  The copper-loss framing would need to become a friction-and-churning framing.

**A null result is publishable** and should be reported as such. If the effect is
absent at every load, that is a clean negative that the literature currently
lacks.

---

## 8. Logging

One CSV per condition, one row per sample:

```
t_s, cond_id, rho, tau_bar_Nm, seed,
v_bus_V, i_bus_A, p_in_W,
q_meas_rad, q_des_rad,
T_amb_C, T_P1_C, T_P2_C, T_P3_C
```

Plus one session header recording: date, ambient at start, camera emissivity
setting, camera distance, tape type, lever length $L$, masses used, bench loop
rate, measured $-3$ dB bandwidth from 4.1, and the detection floor from 4.2.

Keep the raw thermal frames. They are the only part of this that cannot be
reconstructed later, and the spatial question in §1 may turn out to matter more
than the scalar one.

## 9. What this cannot tell you

Stated here so it goes into the paper's limitations rather than being discovered
by a reviewer.

- One motor is not a population. Iron loss, friction and gear churn differ between
  units and between designs.
- Surface temperature is not winding temperature. Nothing here measures the
  winding directly, so no absolute thermal claim can be made — only comparisons
  under matched duration and matched starting temperature.
- A synthetic sine plus band-limited noise is not a gait. It isolates $\rho$
  cleanly, which is the point, but it means the bench validates the *mechanism*,
  not the closed-loop numbers of §5.5.
- The bench has no contact events. Stance-phase impact torque is absent.
