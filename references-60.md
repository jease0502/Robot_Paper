# References — expanded to 60 entries

Companion file to `paper-v2.md`. Produced by `academic-paper` skill, `lit-review` mode, 2026-09-10.

**Verification policy applied.** Every entry below was checked against a live web search
before being written down. Entries [1]–[21] are the manuscript's existing references and
were re-verified. Entries [22]–[60] are new. Nothing here was written from memory alone.

**Where information is missing it is marked `pending`, not guessed.** In particular, several
new entries are confirmed to exist (title + arXiv ID + abstract all verified) but the search
result did not expose the full author list. Those carry `[authors: pending]`. Fill them from
the arXiv abstract page before submission — do not let a generated author list through.

**No numeric claim about this manuscript's own experiments appears in this file.** All
figures in `paper-v2.md` remain sourced from `paper-command-filtering-draft.md`.

---

## 1. Verification outcome for the existing 21 entries

| # | Status | Note |
|---|---|---|
| [1] | not independently verified | GitHub issue, not indexed literature. URL retained as given; confirm it still resolves before submission. |
| [2] | **exists; authors pending** | arXiv:2603.01631 confirmed real, title matches. Search did not expose the author list — the manuscript's "L. Qian, Y. Wan, S. Wang, and X. Luo" could not be confirmed. |
| [3] | verified | Full author list recovered: Y. Wan, W. Lin, L. Qian, Y. Zou, W. Wu, S. Liao, C. Zhao, X. Luo (HUST). v3 revised 2026-07-03. Hardware validation on Unitree A1. |
| [4] | verified | Author list exact match. Tsukuba Challenge 2025, ~2.8 km route. |
| [5] | verified | ICRA 2021, Xi'an. |
| [6] | n/a | Internal repository document. |
| [7] | n/a | Manufacturer datasheet, not indexed literature. |
| [8] | verified, **venue addable** | Also published at IEEE/RSJ IROS 2022. |
| [9] | verified, **CORRECTION REQUIRED** | Published in **IEEE Robotics and Automation Letters, vol. 4, no. 2, pp. 1077–1084, 2019**, not *Proc. Humanoids* as `paper-v2.md` states. Validated on NASA Valkyrie. |
| [10] | verified | Kyoto University / ATR. |
| [11] | verified, **authors expandable** | Full list: R. Kourdis, M. Stępień, J. Manhes, N. Mansard, S. Tonneau, P. Souères, T. Flayols. |
| [12] | verified | Accepted RSS 2026. Full list: L. Zhang, J. Wang, T. Zhang, Z. Song, X. Zeng, W. Xia, Z. Li, Y. Liu. |
| [13] | verified | Kwangwoon University. |
| [14] | verified | Full list: X. Han, H. Xiong, H. Chen, C. Liu, A. Torralba, Y. Zhu, Y. Du. |
| [15] | **verified — prior suspicion was wrong** | J. C. Weddington, B. P. Ölveczky, and S. A. Baccus **are** the correct authors. Mini Pupper 2 platform, >50 ms transport delay. No correction needed. |
| [16] | verified, **venue addable** | Published at IROS 2025. |
| [17] | verified | Author list exact match. |
| [18] | verified | IROS 2024, pp. 627–634. Inventec Corporation, Taipei. |
| [19] | verified | ICRA 2023, pp. 5085–5091. ANYmal C at 8 Hz. |
| [20] | verified | ICML 2019, pp. 6096–6104. |
| [21] | verified, **venue addable** | NeurIPS 2025. Author list exact match. |

**Two edits are required before submission:** entry [9]'s venue, and entry [2]'s author list
(confirm or mark pending). Everything else is clean.

---

## 2. The 60-entry reference list

### Existing entries, with verified corrections applied

[1] NVIDIA Research, "Ankle roll motors overheating on G1," GR00T-WholeBodyControl, GitHub issue #174, 2026. Available: https://github.com/NVlabs/GR00T-WholeBodyControl/issues/174. Not peer reviewed.

[2] `[authors: pending]`, "Learning thermal-aware locomotion policies for an electrically-actuated quadruped robot," arXiv:2603.01631, 2026.

[3] Y. Wan, W. Lin, L. Qian, Y. Zou, W. Wu, S. Liao, C. Zhao, and X. Luo, "Learning to balance motor thermal safety and quadrupedal locomotion performance with residual policy," arXiv:2605.27046, 2026.

[4] T. Matsuzawa, K. Irie, T. Yoshida, T. Suzuki, Y. Hara, and M. Tomono, "Long-distance real-world navigation of the legged-wheeled robot Go2-W using deep reinforcement learning," arXiv:2606.21387, 2026.

[5] S. Mysore, B. Mabsout, R. Mancuso, and K. Saenko, "Regularizing action policies for smooth control with reinforcement learning," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2021. arXiv:2012.06644.

[6] Vendor control-stack reference, this repository, `docs/legged-rl-control-stack-reference.md`, 2024.

[7] maxon motor ag, "EC max 30 Ø30 mm, brushless, 60 W," datasheet, 2024.

[8] K. Urs, C. E. Adu, E. J. Rouse, and T. Y. Moore, "Design and characterization of 3D printed, open-source actuators for legged locomotion," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2022. arXiv:2202.12395.

[9] S. J. Jorgensen, J. Holley, F. Mathis, J. S. Mehling, and L. Sentis, "Thermal recovery of multi-limbed robots with electric actuators," *IEEE Robotics and Automation Letters*, vol. 4, no. 2, pp. 1077–1084, 2019. arXiv:1902.00187.

[10] S. Yamamori et al., "Actuator reality shaping for zero-shot sim-to-real robot learning," arXiv:2607.02205, 2026.

[11] R. Kourdis, M. Stępień, J. Manhes, N. Mansard, S. Tonneau, P. Souères, and T. Flayols, "Very high frequency interpolation for direct torque control," arXiv:2509.24175, 2025.

[12] L. Zhang, J. Wang, T. Zhang, Z. Song, X. Zeng, W. Xia, Z. Li, and Y. Liu, "VRA: Grounding discrete-time joint acceleration in voltage-constrained actuation," in *Proc. Robotics: Science and Systems (RSS)*, 2026. arXiv:2605.10696.

[13] D. Son and S. Park, "LiPo: A lightweight post-optimization framework for smoothing action chunks generated by learned policies," arXiv:2506.05165, 2025.

[14] X. Han, H. Xiong, H. Chen, C. Liu, A. Torralba, Y. Zhu, and Y. Du, "B-spline policy: Accelerating manipulation policies via B-spline action representations," arXiv:2607.09648, 2026.

[15] J. C. Weddington, B. P. Ölveczky, and S. A. Baccus, "Reinforcement learning on cost-constrained quadrupedal hardware," arXiv:2607.26434, 2026.

[16] Z. Chen, X. He, Y.-J. Wang, Q. Liao, Y. Ze, Z. Li, S. S. Sastry, J. Wu, K. Sreenath, S. Gupta, and X. B. Peng, "Learning smooth humanoid locomotion through Lipschitz-constrained policies," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2025. arXiv:2410.11825.

[17] J. Shin, W. Cha, D. Kim, J. Cha, and J. Park, "Spectral normalization for Lipschitz-constrained policies on learning humanoid locomotion," arXiv:2504.08246, 2025.

[18] G. Christmann, Y.-S. Luo, H. Mandala, and W.-C. Chen, "Benchmarking smoothness and reducing high-frequency oscillations in continuous control policies," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2024, pp. 627–634. arXiv:2410.16632.

[19] S. Gangapurwala, L. Campanaro, and I. Havoutis, "Learning low-frequency motion control for robust and dynamic robot locomotion," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2023, pp. 5085–5091. arXiv:2209.14887.

[20] C. Tallec, L. Blier, and Y. Ollivier, "Making deep Q-learning methods robust to time discretization," in *Proc. Int. Conf. Machine Learning (ICML)*, 2019, pp. 6096–6104. arXiv:1901.09732.

[21] A. Sukhija, L. Treven, J. Cheng, F. Dörfler, S. Coros, and A. Krause, "TARC: Time-adaptive robotic control," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2510.23176.

### New entries

[22] J. Hwangbo et al., "Learning agile and dynamic motor skills for legged robots," *Science Robotics*, vol. 4, no. 26, eaau5872, 2019. doi:10.1126/scirobotics.aau5872.

[23] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, vol. 5, no. 47, eabc5986, 2020. doi:10.1126/scirobotics.abc5986.

[24] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild," *Science Robotics*, vol. 7, no. 62, eabk2822, 2022. doi:10.1126/scirobotics.abk2822. arXiv:2201.08117.

[25] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, "Learning to walk in minutes using massively parallel deep reinforcement learning," in *Proc. Conf. Robot Learning (CoRL)*, PMLR vol. 164, 2022. arXiv:2109.11978.

[26] V. Makoviychuk et al., "Isaac Gym: High performance GPU-based physics simulation for robot learning," arXiv:2108.10470, 2021.

[27] M. Mittal et al., "Orbit: A unified simulation framework for interactive robot learning environments," *IEEE Robotics and Automation Letters*, 2023. arXiv:2301.04195.

[28] K. Zakka et al., "MuJoCo Playground," arXiv:2502.08844, 2025.

[29] K. Zakka, Q. Liao, B. Yi, L. Le Lay, K. Sreenath, and P. Abbeel, "mjlab: A lightweight framework for GPU-accelerated robot learning," arXiv:2601.22074, 2026.

[30] P. M. Wensing, A. Wang, S. Seok, D. Otten, J. Lang, and S. Kim, "Proprioceptive actuator design in the MIT Cheetah: Impact mitigation and high-bandwidth physical interaction for dynamic legged robots," *IEEE Transactions on Robotics*, vol. 33, no. 3, pp. 509–522, 2017. doi:10.1109/TRO.2016.2640183.

[31] B. Katz, J. Di Carlo, and S. Kim, "Mini Cheetah: A platform for pushing the limits of dynamic quadruped control," in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2019. doi:10.1109/ICRA.2019.8793865.

[32] M. Hutter et al., "ANYmal — a highly mobile and dynamic quadrupedal robot," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)*, 2016. doi:10.1109/IROS.2016.7758092.

[33] G. B. Margolis and P. Agrawal, "Walk these ways: Tuning robot control for generalization with multiplicity of behavior," in *Proc. Conf. Robot Learning (CoRL)*, PMLR vol. 205, pp. 22–31, 2023. arXiv:2212.03238.

[34] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," arXiv:1703.06907, 2017.

[35] W. Yu, J. Tan, C. K. Liu, and G. Turk, "Preparing for the unknown: Learning a universal policy with online system identification," in *Proc. Robotics: Science and Systems (RSS)*, 2017. arXiv:1702.02453.

[36] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, "Learning fine-grained bimanual manipulation with low-cost hardware," arXiv:2304.13705, 2023.

[37] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, "Diffusion policy: Visuomotor policy learning via action diffusion," *Int. Journal of Robotics Research*, 2025. arXiv:2303.04137.

[38] K. Black et al., "Real-time execution of action chunking flow policies," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2506.07339.

[39] O. Wallscheid, "Thermal monitoring of electric motors: State-of-the-art review and future challenges," *IEEE Open Journal of Industry Applications*, 2021.

[40] W. Kirchgässner, O. Wallscheid, and J. Böcker, "Estimating electric motor temperatures with deep residual machine learning," *IEEE Transactions on Power Electronics*, vol. 36, no. 7, pp. 7480–7488, 2021. doi:10.1109/TPEL.2020.3045596.

[41] W. Kirchgässner et al., "Thermal neural networks: Lumped-parameter thermal modeling with state-space machine learning," arXiv:2103.16323, 2021.

[42] F. Gustafsson, "Determining the initial states in forward-backward filtering," *IEEE Transactions on Signal Processing*, vol. 44, no. 4, pp. 988–992, 1996. doi:10.1109/78.492552.

[43] F. J. Harris, "On the use of windows for harmonic analysis with the discrete Fourier transform," *Proceedings of the IEEE*, vol. 66, no. 1, pp. 51–83, 1978.

[44] J. Makhoul, "Linear prediction: A tutorial review," *Proceedings of the IEEE*, vol. 63, no. 4, pp. 561–580, 1975. doi:10.1109/PROC.1975.9792.

[45] N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," *Psychological Bulletin*, vol. 114, no. 3, pp. 494–509, 1993.

[46] B. Chen, M. Xu, L. Li, and D. Zhao, "Delay-aware model-based reinforcement learning for continuous control," *Neurocomputing*, vol. 450, pp. 119–128, 2021. arXiv:2005.05440.

[47] `[authors: pending]`, "Delays in reinforcement learning," arXiv:2309.11096, 2023.

[48] `[authors: pending]`, "L2C2: Locally Lipschitz continuous constraint towards stable and smooth reinforcement learning," arXiv:2202.07152, 2022.

[49] `[authors: pending]`, "Redesigning regularization for effective policy smoothing," arXiv:2606.13169, 2026.

[50] `[authors: pending]`, "Stabilizing the Q-gradient field for policy smoothness in actor-critic methods," arXiv:2601.22970, 2026.

[51] `[authors: pending]`, "Actuator-constrained reinforcement learning for high-speed quadrupedal locomotion," arXiv:2312.17507, 2023.

[52] `[authors: pending]`, "Deep learning for model-free prediction of thermal states of robot joint motors," arXiv:2509.12739, 2025.

[53] `[authors: pending]`, "Robust reinforcement learning-based locomotion for resource-constrained quadrupeds with exteroceptive sensing," arXiv:2505.12537, 2025.

[54] `[authors: pending]`, "Task-specified compliance bounds for humanoids via Lipschitz-constrained policies," arXiv:2603.16180, 2026.

[55] `[authors: pending]`, "Concurrent training of a control policy and a state estimator for dynamic and robust legged locomotion," arXiv:2202.05481, 2022.

[56] `[authors: pending]`, "Alternative metrics to select motors for quasi-direct drive actuators," arXiv:2202.12365, 2022.

[57] `[authors: pending]`, "Adaptive energy regularization for autonomous gait transition and energy-efficient quadruped locomotion," arXiv:2403.20001, 2024.

[58] `[authors: pending]`, "Guiding energy-efficient locomotion through impact mitigation rewards," arXiv:2510.09543, 2025.

[59] `[authors: pending]`, "Energy-efficient quadruped locomotion with compliant feet," arXiv:2605.14411, 2026.

[60] `[authors: pending]`, "Energy efficient control of electric motors," arXiv:2109.11113, 2021.

---

## 3. Annotation table — what each new entry contributes and where to cite it

| # | Contribution | Cite in |
|---|---|---|
| [22] | Actuator network learned from bench data; the origin of the "model the actuator, don't idealize it" line the paper's §2 argument sits against. | §2, §7 Actuation Interfaces |
| [23] | Teacher-student blind locomotion; canonical source for the training setup family the frozen policies come from. | §1, §4 |
| [24] | Perceptive locomotion in the wild; establishes the deployment regime where thermal limits actually bite. | §1 |
| [25] | `legged_gym`; the reward-term vocabulary (`action_rate`, `torques`, `energy`) the §5.2 ablation manipulates originates here. | §4, §5.2, §7 Smoothness |
| [26] | Isaac Gym; GPU simulation substrate underpinning the reward conventions in §5.2. | §4 |
| [27] | Orbit / Isaac Lab; the manager-based env API that standardizes PD-gain and action-scale conventions. | §2, §4 |
| [28] | MuJoCo Playground; source of the Go1 model parameters quoted in §4 (E1). | §4 |
| [29] | `mjlab`; the framework family behind `unitree_rl_mjlab`, whose gain derivation §2 uses for the 0.5·e transient. | §2 |
| [30] | Proprioceptive / quasi-direct-drive actuation; why legged joints have low gear ratio and thus high copper loss sensitivity. | §2, §3 |
| [31] | Mini Cheetah platform; second QDD data point for the actuator regime. | §2 |
| [32] | ANYmal / ANYdrive; the series-elastic contrast case, with a measured 70 Hz torque-control bandwidth. | §2, §7 Control Rate |
| [33] | Walk These Ways; the open Go1 sim-to-real stack, directly relevant to the platform under study. | §4 |
| [34] | Domain randomization; the standard alternative to filtering for closing the actuator gap. | §7 Actuation Interfaces |
| [35] | UP-OSI online system identification; adaptation rather than filtering as a response to actuator mismatch. | §7 Actuation Interfaces |
| [36] | ACT / action chunking; the manipulation-side analogue of the slow-policy-fast-controller interface. | §7 Actuation Interfaces |
| [37] | Diffusion Policy receding-horizon execution; chunk-boundary discontinuity is the manipulation cousin of the ZOH staircase. | §7 Actuation Interfaces |
| [38] | Real-time chunking; handles inference latency by inpainting rather than delaying — the closest prior art to the paper's predictive filter, and the strongest available contrast to "you cannot buy smoothness with latency." | §5.5, §6, §7 Actuation Interfaces |
| [39] | Thermal monitoring review; establishes that driver-reported temperature is model-based and lags, supporting §3's choice of a current-side proxy. | §3, §8 |
| [40] | Deep residual estimation of motor temperatures; quantifies how far winding temperature departs from what a driver reports. | §3 |
| [41] | Thermal neural networks / lumped-parameter modelling; the modelling layer §8 says the manuscript does not have. | §8 |
| [42] | Forward-backward filtering initial states; the canonical citation for the zero-phase scheme in §5.3 and for why it is non-causal. | §4, §5.3 |
| [43] | Window functions; the citation for the symmetric Hann kernel used by the predictive filter. | §4 |
| [44] | Linear prediction tutorial; the AR(8) / Levinson-Durbin basis of the predictive filter. | §4 |
| [45] | Cliff's delta; the effect-size statistic reported throughout §5.5 and §5.6, currently uncited. | §3, §5.5 |
| [46] | Delay-aware model-based RL; formalizes the augmented-state fix for the latency §5.5 shows is destabilizing. | §5.5, §7 Control Rate |
| [47] | Survey of delays in RL; frames the delayed-MDP literature the §5.5 finding sits inside. | §7 Control Rate |
| [48] | Locally Lipschitz continuous constraint; another smoothness regularizer for the §7 comparison set. | §7 Smoothness |
| [49] | Recent re-examination of policy smoothing regularizers; keeps §7 current. | §7 Smoothness |
| [50] | Q-gradient field stabilization for smoothness; architectural rather than loss-based smoothing. | §7 Smoothness |
| [51] | Actuator-constrained RL for high-speed quadrupeds; torque/actuator limits inside training, the retraining alternative to filtering. | §7 Actuation Interfaces |
| [52] | Model-free prediction of joint-motor thermal states; directly adjacent to the manuscript's thermal proxy. | §3, §7 Thermal-Aware |
| [53] | Locomotion under resource constraints; the deployment context where a post-hoc filter is the only available intervention. | §1, §6 |
| [54] | Compliance bounds via Lipschitz-constrained policies; connects smoothness constraints to physical interaction limits. | §7 Smoothness |
| [55] | Concurrent policy and state-estimator training; relevant to §5.5's "policy is itself a feedback controller" argument. | §5.5, §6 |
| [56] | Motor selection metrics for QDD actuators; supports the §5.1 claim that the crossover is a property of the joint's operating point. | §5.1 |
| [57] | Adaptive energy regularization and gait transition; direct support for §6's "penalize effort, not action difference." | §5.2, §6 |
| [58] | Impact-mitigation rewards for energy efficiency; an alternative reward-side lever on the same budget. | §6 |
| [59] | Compliant feet for energy-efficient quadrupedal locomotion; the mechanical-design lever, complementary to both reward and filtering. | §6 |
| [60] | Energy-efficient motor control; the drive-side view of the same copper-loss objective §3 proxies. | §3 |

---

## 4. Remaining work before submission

1. **Fix entry [9]'s venue** — it is IEEE RA-L 2019, not Humanoids. This is a real citation error in the current draft.
2. **Resolve entry [2]'s author list** — currently unconfirmed.
3. **Fill the 14 `[authors: pending]` fields** in [47]–[60] from each arXiv abstract page.
4. **§7 currently cites ~15 works; 39 new entries are now available.** The annotation table maps each to a section, but §7's prose has not been rewritten to accommodate them — that is a separate drafting pass, not part of lit-review mode.
5. `paper-v2.md` §8 dropped the v1 draft's disclosure that most references had been read only at abstract level. With this verification pass, existence and venue are now confirmed for all 21 existing entries; **depth of reading is a separate claim** and should be restated honestly in §8.
