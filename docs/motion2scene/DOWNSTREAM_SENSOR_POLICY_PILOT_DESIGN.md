# Next experiment design: from interface to a useful training comparison

**Current stage (September 7):** [the selector breakpoint study](SELECTOR_BREAKPOINT_RESULT.md)
completes twelve matched command executions and twenty-four shared linear-control
executions in Isaac Lab. Walking fails at the middle-height seed-8512 encounter;
d040 passes. The analytic-trained linear control observes the scene, requests d040
legally and realizes that rescue with 0 N recorded beam force (walking: 57.947 N).
Its passage is 4/6 conditions; the other three controls pass 3/6. This is development
evidence on one observed carrier, not Motion2Scene superiority or source transfer.

The original twenty MLPs remain frozen at 120/600 admitted evaluations; **480 assigned
runs are operationally paused**. Input diagnosis finds normalized state magnitudes
up to 1952.75. Exact observation groups have zero measured action-ambiguity gap;
the uniform BCE is near its empirical floor. An empty-transition geometry forecast
catches thirteen reference false-clear commands but misses two additional
contact-qualified commands. Neither forecast replaces physical labels.

**Next main experiment:** [transition-aware construction on additional development
carriers](TRANSITION_AWARE_NEXT_STAGE.md), with the same achieved transition information
for analytic and learned generation. Stabilize the common learner, then freeze the
five-arm 24/48/96 comparison and genuinely new-source evaluation. No second carrier
is yet qualified for this switching interface. The learned generator's advantage
and a complete ICRA contribution remain unproved.

[All 24 new execution replays](assets/selector-breakpoint.mp4) ·
[Complete result and retained failures](SELECTOR_BREAKPOINT_RESULT.md) ·
[Source archive](evidence/selector-breakpoint-source-20260907/research-source.tar.gz).
Isaac Lab supplies physical dynamics and measured contacts; MuJoCo renders recorded
states only. The common control changes scaling, capacity and regularization jointly.

**Historical first-wave snapshot (September 6; superseded above):** [the independent-layout evaluation](INDEPENDENT_LAYOUT_V1_RESULT.md)
has completed 120/600 assigned Isaac Lab executions, with
120 admitted after input, command, geometry and contact audits.
The first wave covers one reserved station, three heights and two physics seeds
for all twenty frozen learners. The remaining 480 assignments stay pending.
This is a partial evaluation on observed source 41002, not source-held-out transfer.
Training remains eleven complete encounters per arm. The learned generator's
advantage over analytic training data remains unproved. All four arms tie on these
first six blocks, and none requests d040; the remaining panel is still pending.

**Reproducibility and demos:** all twenty original selectors are now
[downloadable with their exact training inputs](evidence/selector-bundle-20260906/selectors.tar.gz);
CPU refitting reproduces every weight, normalization value and final loss exactly.
The [new replay](assets/independent-layout-wave1.mp4) shows a fixed twenty-four-run subset
covering all four data arms, both physics seeds and all three first-station heights. Isaac Lab supplies
the dynamics and measured contacts; MuJoCo renders recorded states only.
The [post hoc readout diagnostic](LAYOUT_READOUT_DIAGNOSTIC_V1.md) tests sensor
sensitivity without adding physics or changing the frozen policies.

**Historical sequencing (superseded by the breakpoint study):** finish the remaining frozen layout and control assignments,
then evaluate shared scripted comparators. Preserve every failure and keep controls
separate from traversal. Larger 24/48/96 data budgets require new corpora and
separate fits; the reserved final-source candidates still require acquisition and
qualification under the same switching contract. A complete ICRA learning claim
also needs source-grouped effects and full acquisition cost accounting. See the
[methods draft](LEARNING_COMPARISON_METHODS_DRAFT.md) and
[focused prior-work comparison](RELATED_WORK_POSITIONING_20260906.md).

**Historical corpus stage (September 6; superseded above):** the [comparative command corpus](COMPARATIVE_CORPUS_STAGE_RESULT.md)
contains 76 completed Isaac Lab executions and 38 paired scene outcomes, captured at
one common 0.30 s decision. Both-fail outcomes and the analytic test-neighborhood
rejection remain in the record. All 38 pairs pass the direct-input/bank audits, and all twenty matched learners
are fitted (eleven complete encounters per arm, five optimizer seeds). All nine
Motion2Scene generated encounters are both-fail; four of eight analytic encounters
make d040 useful. This is a development data-yield finding, not a policy ranking.
Eight final-transfer candidate IDs are now [reserved against explicit acquisition
provenance](FINAL_SOURCE_PROVENANCE_RESERVATION_V1.md), including canonical fingerprints
of 212 referenced motion files. They are not yet generated or qualified; the narrower
provenance scope does not establish global metadata absence or source independence.

**Learned command check:** all twelve actual Isaac Lab integrations match the registered
features, model requests, full trajectories and outcomes. These are observed development
scenes, not independent performance evidence. [Integration protocol](LEARNED_COMMAND_CHECK_V1.md).

**Historical next step:** [evaluate the frozen policies on the twelve independently specified layouts](INDEPENDENT_LAYOUT_EVALUATION_V1_PLAN.md).
Keep the analytic arm as the primary comparator. Report source-specific contact-qualified
passage/recovery, unnecessary crouching and blocked-scene detection separately. This
single-source, eleven-encounter development comparison is not the final 24/48/96
acquisition-budget study or fresh-source transfer experiment.

**Historical prelaunch snapshot (September 6; superseded by the current stage above):** [the four-arm d040-bound pipeline](COMPARATIVE_ACQUISITION_V1_RESULT.md)
now records 64 outputs and 36 assigned generated slots, plus a shared background quota.
The common checks admit 9/9 uniform, 8/9 analytic, 9/9 no-contrast and 9/9 Motion2Scene
slots; the analytic test-neighborhood rejection is retained. A twenty-cell first-slice
manifest is frozen, with direct 0.30 s pre-command state capture. **0/20 runs have
started because GPU memory is below the registered 7500 MiB floor.** No new paired
labels, robot-data selector fit or learning benefit is claimed. Exact d040 input binding
is complete within its declared numerical tolerance. Final source reservation remains
unresolved after the bounded inventory timed out.

**Earlier action-study snapshot (September 6; counts below are historical):** [all 12 new runs](ACTION_LABEL_COMPLETION_V1_RESULT.md)
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
