# Next experiment design: from interface to a useful training comparison

**Completed action study (September 6):** [all 12 new runs](ACTION_LABEL_COMPLETION_V1_RESULT.md)
complete 16/16 matched 0.20 s action-label pairs. In the earlier-beam development case,
seed 8042 passes at 0.30 s but fails at 0.20 and 0.40 s. The [v2 learning contract](LEARNING_CONTRACT_V2.md)
therefore selects one common 0.30 s decision for new data in every arm; the old labels
remain at 0.20 s and cannot be reused as 0.30 s outcomes. The matched no-contrast generator
baseline is fitted, the shared outcome learner is implemented, and robot-data selector
fits remain zero. Next acquire the comparative corpus and evaluate learning utility.


**Learning-utility priority (September 6):** [the revised comparison plan](LEARNING_UTILITY_PLAN_V1.md)
freezes the current generator, makes Motion2Scene versus strong analytic data the
primary question, and uses complete outcomes for explicit encounter commands.
The [registered 12-cell label/timing study](ACTION_LABEL_COMPLETION_V1.md) is the
only added diagnostic prerequisite. Continuous certificate expansion, new skills
and new generator architectures are deferred. The plan also corrects the evaluation
cost and current ICRA submission/video schedule.


**Previous completed panel (retained):** the [42-cell Isaac Lab stress
study](OVERHANG_VARIATION_V1_RESULT.md) is complete: reactive and oracle each pass
9/10 pose trials, so the registered pose-robustness prediction fails. The earlier
beam in seed 8042 causes a 429.159 N contact for both. All ten blind trials contact;
100 ms and realized 260 ms delays pass 2/2 each, while 500 ms refuses and fails 2/2.
All six negative controls reject overhang detection; blocked passages still fail.
Measurement and reference-bank audits pass all 42 cells.

The [full-batch native/time audit](FRESH_NATIVE_TEMPORAL_V1_RESULT.md) retains
384/384 sampled passes and independently verifies 384/384 conditional interval bounds
at nominal placements, with a 21.623 mm worst lower bound. The original capped-bound
attempt remains unresolved 0/384. These are authored outer enclosures and a declared
reference interpolant, not cooked-mesh CCD or actual dynamic guarantees.

The [policy-data readiness audit](POLICY_READINESS_V1_RESULT.md) packages 144 sensor
features with object identity and outcome labels excluded. Only 10/16 scene–seed
groups have both skill labels; one has neither tested skill passing. Six negative
controls lack crouch comparators. No policy is trained on this development-only set.
Next: close those six labels, diagnose the earlier-beam failure without altering this
panel, then freeze source splits and the matched four-arm learning-data comparison.
See the [completion assessment](PROJECT_COMPLETION_STATUS.md) and
[latest demos](index.html#next-stage). The downstream learning benefit remains unproved.



**Previous completed stage (retained):** the [overhang/guard follow-up](OVERHANG_INTERFACE_V2_RESULT.md)
completes fourteen evaluation-bank Isaac trials. All ten normal reactive/oracle
requests pass contact-free traversal; both critical-beam reactive trials pass, and
all four absent/raised reactive controls reject wall observations without switching.
Both deliberately late requests are denied and remain failed avoidance outcomes.
The earlier fourteen-cell batch is retained: a training-augmented alternate reference
froze in one seed and stopped the robot. The repair loads both references for evaluation
and independently verifies the realized route and bank arrays in every run.
Next: [height, position and observation-delay variation](OVERHANG_VARIATION_PANEL_DESIGN.md),
then the matched downstream learning comparison. No learning-benefit claim is made.


Planning record, 2026-09-06. This is not a registered learning run or a measured
result. The reactive-interface protocols and their reported failures remain fixed.

The immediate contribution target is a controlled test of whether motion-conditioned
scene generation improves the *training data* for a humanoid traversal selector.
Retain the release SONIC controller and its verified reference bank. The changed
component is a small policy that receives overhead range observations, proprioception
and legal approach phase, then chooses a feasible reference. At deployment it must
not receive authored beam pose, scene identity, generator arm or outcome labels.

The [completed interface result](REACTIVE_INTERFACE_V3_RESULT.md) clears the critical
beam in both seeds but falsely switches in all four absent/raised controls on the
far wall. Before more physics or learning, design upper-occupancy plus lower-free-space
observations and an explicit legal-transition check. Retain wall/raised/empty negative
controls. Do not use hit identity or hide failures with a timing cutoff alone.
Register new tests before execution; preserve this failed specificity prediction.

First close the interface's restricted scope. The current 41002 rule switches early
and returns at a fixed motion phase. Test a frozen panel of beam heights, longitudinal
positions and observation delays before claiming support for later decisions. Report
missed detections and transition failures separately from geometry rejection. Include
empty, raised, critical and infeasible scenes. Do not count refusal as traversal.
A simulator collision-ray sensor is an ideal lidar observation, not rendered depth;
a depth-camera experiment and sensor-noise transfer remain separate extensions.

Then freeze the sensing/command API and acquire an independent development pool for
training. Keep source ancestry intact: a neutral motion, derivatives and all scene
jitter share one split. Sources 43001–43008 remain permanently excluded from fitting
and model selection; the previously observed 410xx/420xx studies are not fresh tests.
Freeze a new source-level test bank before policy fitting and never tune on its results.

Compare Motion2Scene learned initialization + fixed search/rejection, uniform placement,
the strong analytic/grid generator, and an unconstrained version of the same scene
representation. Specify the actual unconstrained model before registration; do not
claim a diffusion baseline without implementing and matching one. Use the same feasible
command vocabulary, sensor, policy architecture and common supervised objective.
Physics labels must test all legal alternatives; d040 and d055 may both be feasible.
Do not collapse a failed source qualification into a missing row or assign a positive
label from kinematic clearance alone.

Use five optimizer seeds per arm and equal downstream optimizer steps and execution
label budgets. Report acquisition, synthesis, rejection, verification, physics labeling,
training and evaluation costs separately and together. Random placement's infeasible
fraction is an outcome, not a reason to refill only that arm. Match scene-family support
and source access; retain an oracle observer/selector to measure the interface ceiling.

Primary outcomes: held-out first-episode contact-free passage, falls, refusal rate and
learning curves at frozen data/step checkpoints. Secondary outcomes: time to traverse,
tracking error, 200 Hz beam-force peak/duration/impulse, sensing misses and switch timing.
A common loss and disclosed label thresholds avoid arm-specific reward shaping; they
do not establish that the task contains no manual design. Source groups are the
experimental unit for transfer; seeds and beam jitter are repeated observations.

Only launch learning after writing a concrete manifest with acquired source hashes,
fixed scene lists, measured interface qualification, sensor calibration, architecture,
loss, budgets, optimizer seeds, stopping rules and predictions. If a strong analytic
baseline ties or wins, report it and revise the learned-generator contribution claim.
