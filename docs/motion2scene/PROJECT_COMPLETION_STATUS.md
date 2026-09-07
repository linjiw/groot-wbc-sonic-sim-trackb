# Motion2Scene: progress toward the complete project

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


Assessment dated 2026-09-06. **We have an embodied research prototype, but not yet a
complete ICRA experimental contribution.** Geometry generation and simulator execution
have evidence. The central claim—that generated environments improve policy learning—
has a partial independent-layout comparison but no demonstrated advantage. Counting finished scripts or rollouts would overstate
completion, so the status below uses evidence gates instead of a percentage.

| Workstream | Current evidence | Completion assessment |
| --- | --- | --- |
| Inverse scene generation | 384/384 fresh-source requests accepted by proxy screening; conditioning/search ablations; explicit rejection | Strong offline foundation. Learned value against the strong analytic baseline remains to be earned downstream. |
| Collision and time realism | 384/384 nominal authored-native interval bounds, minimum 21.623 mm; separate static proxy pose domains | Bounded progress. Joint pose/time uncertainty, cooked-shape and numerical enclosure guarantees remain open. |
| Embodied scene validation | 52-cell generated-source pilot; 6/8 source pairs qualify; 18/24 requested slots separate | Demonstrated in Isaac Lab within the development pool. Source refusals remain; walk contacts but can still cross. |
| Sensing and legal commands | Complete 42-cell stress panel: reactive/oracle 9/10 poses each, 6/6 negative specificity, 500 ms delay fails 2/2 | One observed source. Earlier-beam robustness prediction fails; noisy sensing, later decisions and source transfer remain open. |
| Downstream learning benefit | 38 paired development encounters; twenty frozen fits; 120/600 independent-layout executions admitted | Partial single-source evaluation; the training-data advantage remains unproved. |
| Paper and release | Public notebook, protocols, complete result tables, failed experiments, source snapshots and recorded-state demos | Research documentation exists. Final comparative figures, claim audit and a reproducible end-to-end benchmark remain. |

Two of the three intended contribution pillars—generation and closed-loop feasibility—
have bounded evidence. That is **not “two-thirds complete”**: the remaining learning
comparison could reject the motivating thesis. The project is beyond an offline geometry
demo and still well short of submission readiness. A strong analytic method tying or
beating the learned generator must change the contribution statement, not be hidden.

Eight candidate IDs are reserved under the explicit acquisition-provenance ledger;
no final ancestors have been generated or qualified. The earlier broad inventory
failures remain historical records. The twelve layouts remain frozen; the first station is now measured and the remaining assignments are pending.
[Reservation failure record](evidence/transfer-reservation-failure.json).

## Remaining experiments, in decision order

1. **Finish the frozen independent-layout panel.** Complete the remaining 480 assignments under the per-block budget and admission gates. Keep traversal, absent/raised adaptation and blocked refusal outcomes separate. Do not revise learners or the test panel from the first-wave results.
2. **Measure shared scripted comparators.** Execute the same legal command interface with the existing sensor rule and privileged baseline. Use paired action labels to distinguish unsupported transitions from wrong predictions; an unexecuted alternative is unknown.
3. **Run the larger learning-data comparison.** The present eleven-example fits are a development study. Acquire the declared 24/48/96 complete-encounter budgets, fit each arm and optimizer seed separately, and account for fitting, proposals, rejections, verification and paired physics labeling. Avoid a phase or timing shortcut with matched controls and observation diagnostics.
4. **Qualify and evaluate final source transfer.** The eight reserved candidate IDs are not qualified sources. Acquire their motions and legal switching bank; keep every derivative and physics repeat with its ancestor. Previously inspected 430xx sources remain excluded from fitting and are not newly untouched tests.
5. **Finish the evidence-driven manuscript and release.** Use source-grouped effects, tail failures and cost measurements to decide whether the learned generator earns its claim against analytic generation. Complete the fixed-learner and generator ablations, operating limits and reproducible benchmark. Hardware requires a separately qualified deployment study if included in the final claim.

The decisive missing result is learning utility beyond the analytic data generator.
Completed software, exact CPU reproduction and contact-qualified passage in easy
encounters cannot establish that advantage. If the full comparison ties or loses,
report it and narrow the contribution.

## What the demos mean

The newest videos reconstruct measured Isaac Lab joint/root states with the repository's
MuJoCo visual robot model and the registered beam transform. Contact readouts come from
the recorded 200 Hz Isaac sensor. Rendering uses `mj_forward`, never `mj_step`: these
are visual replays, not independent MuJoCo dynamics validation. The visual mesh differs
from the Isaac collision asset; numerical validity comes from the measurements and
audits, not apparent pixel clearance. Older reference-only animations remain labeled.

See the [ICRA gate ledger](ICRA_COMPLETION_PLAN.md), [latest completed interface
result](OVERHANG_INTERFACE_V2_RESULT.md), [variation registration](OVERHANG_VARIATION_V1.md),
and [downstream design](DOWNSTREAM_SENSOR_POLICY_PILOT_DESIGN.md).
