# Motion2Scene: progress toward the complete project

Read the [September 9 research brief](RESEARCH_BRIEF_20260909.md) for the dated acquisition snapshot, full paper link, method explanation, evidence assessment, and prioritized ICRA experiment plan.

**Completed M4 acquisition:** all twelve original corpora and the independent
raw-record audit are complete. The [M4 checkpoint report](M4_ACQUISITION_CHECKPOINT.md)
records 468 acquisition episodes and 557,856 measured physics steps including
bootstrap. At 139,464 steps per arm, adaptation-required yield is 2/12 for
uniform, 4/12 for target-only and 12/12 for executed contrast and replay. Every
assigned task is bank-solvable, but every individual corpus remains solvable
by one fixed schedule. Across constructors, the bank covers all 34 unique
geometry/seed conditions and the best fixed schedule covers 32. This supports
a task-yield difference at matched cost, not held-out policy performance.

**First cross-corpus learning result:** the completed
[M1 readout](EXPANDED_DEVELOPMENT_STUDY.md#cross-corpus-decision-learning-at-m1)
uses eight unique acquired scenes from other seeds. Executed contrast improves
the recorded passage count over uniform for one seed (5/5 versus 2/5), while
the other two seeds tie. Target-only gives 4/5 on the differing seed; replay adds
no gain. A fixed late sustained schedule covers all eight scenes, so the result
does not establish better perceptive selection than a constant response.
The completed M2 cross-corpus readout contains 17 unique tasks. Uniform,
executed contrast and observation replay tie at 11/11, 11/12 and 10/11 branch
proxies across the three training seeds; target-only gives 10/11, 11/12 and 10/11.
A fixed late sustained schedule covers all 17 tasks. The M2 useful-task yield
advantage has not yet translated into improved policy selection. Separately,
all three seven-way replay-control comparisons tie at 5/6 recorded development
passages, missing the complementary context. No M2 replay advantage is measured.

At M4, uniform, executed contrast and replay still tie in recorded-input
cross-corpus readouts: 21/22, 21/23 and 22/23; target-only gives 16/22, 21/23
and 16/23. The tied arms select a fixed prior within each corpus. These folds
reuse conditions and add no policy executions. The separate
[seven-way M2/M4 replay controls](REPLAY_CONTROLS_M2_M4.md) all give 5/6 on the
same development recordings, with no changes in selected schedules. No replay
benefit is established through M4. The three reference bootstrap prefixes are
complete and the existing expanded worker is now collecting its acquisition
encounters; no M8 or reserved policy result is available yet.

**Execution sequence:** original M4 → five-arm M8 → matched development policy
panel → independent extension capability panel → six equivalent-teaching fits
and 24 matched arm-policy trials → M16/M32. The revised
[staged worker](/home/linjiw/research-data/groot-wbc/m2s-expanded-and-capability-stages-20260909-v3/sequence.json)
is live. Its predecessor was stopped while waiting for M4, before any expanded
acquisition output existed; no active rollout was interrupted. The
[M8 panel](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-panel-20260909-v1/study.json)
assigns 138 episodes: all 15 acquired policies, the strong script, and all seven
fixed schedules on six development contexts at the same physics seed. This will
separate matched capability from policy selection. Reserved evaluation remains
untouched.

**Repeatable option extraction:** the [matched extension study](OPTION_EXTENSION_STUDY.md)
has completed four fresh neural samples and four authored alternatives, applying
the same repair to all eight. All 17 physical executions have completed: both
procedures qualify 8/8 schedules and 4/4 candidates, with 20,264 measured physics
steps including the shared neutral. This establishes repeatable qualification,
not a generated-motion advantage. Independently fixed task coverage is next.
The current curriculum bank and reserved evaluation are unchanged.
The candidate-level coverage analysis is now implemented and tested alongside
the bank scorer (14 focused tests). It groups the two entry schedules by motion
reference and reports every candidate's incremental and nonredundant coverage;
the 204 assigned task outcomes remain unmeasured.
The equivalent-teaching core now recomputes WAIT within each arm's available
motions and projects the common ridge interface without changing sensing.
Twenty-three combined tests pass; the recorded-data reconstruction reproduces
all 18 original teacher tables. Three whole-longitudinal-center folds are fixed
before new task outcomes. Their native integration and 24 learned-policy trials
remain to be completed after the fixed-task recordings become available.
The equivalent-teaching CPU fitter is now declared and running in wait mode:
it will fit six models after the twelve-task capability panel completes, using
only each fold's eight training tasks. Thirty combined focused tests pass.
No new policy fit or physical result is claimed from this waiting worker.
The arm-specific simulator adapter and model loader are implemented with 40
combined focused tests. The adapter retains the full physical bank and records
the arm-projected inputs separately. Its episode preparation/scoring wrapper is
now implemented: 54 combined tests pass, with the same physical criteria and
explicit held-task/model binding. The 24-episode batch is not yet prepared or
executed; runtime tests use simulator doubles until the fitted models are ready.
The complete batch is now declared and queued after capability and fitting,
before continuation through M16/M32. All 64 combined focused tests pass. The
previous continuation parent and child were replaced while both were waiting
for M4; no expanded output existed and no active physical rollout was interrupted.
The held-center policy analysis now compares each arm against its own bank and
training-selected constant, and reports cross-arm passage differences and
mutually successful timing. All 78 combined extension tests pass. The analysis
does not treat overlapping folds as independent corpus replicates; no policy
outcomes are available yet. All 168 queued panel sources remain unchanged.

**Decision learner:** the fixed depth-two feasibility/time model now has a
NumPy-only native deployment path. Its exported decisions exactly match all 18
training and 18 whole-context-holdout predictions. The completed six-context
closed-loop comparison gives ridge 6/6 passages and the tree 4/6: the tree loses
the long and complementary passages and is 0.12 s slower over the four mutually
successful contexts. All six captures are complete (7,152 physics steps).
Ridge remains the common acquisition learner. Seventy-eight combined focused
tests pass. Primary acquisition resumed after this bounded comparison.

**Sensor sensitivity:** the [separate intervention pipeline](SENSOR_SENSITIVITY_STUDY.md)
now supports channel dropout, hit-range noise and causal packet latency. Its
recorded-input check reproduces all 18 nominal vectors exactly. The subsequent
four-episode native short-passage study passes all four conditions with zero
measured beam force: nominal/dropout/range-noise take 4.10 s, while 100 ms latency
changes commitment to the prior and takes 3.60 s. All 1,192 control-row inputs
reconstruct exactly across 4,768 physics steps. Twenty-two focused sensor tests
and 53 combined method/experiment tests pass. The primary dispatcher has resumed
after the bounded study. Reserved evaluation remains untouched.

**Action-relevant gate:** the completed
[finite continuation comparison](INFORMATION_CONSISTENT_TEACHING.md#action-relevant-continuation-gate)
admits five early nonempty-context decisions when scene features first arrive at
1.00 s, because a common WAIT has successful distinguishable continuations.
One original immediate teacher action must change. With a 1.40 s reveal, no
common all-passing early policy exists. Under nominal sensing, the original cue
gate already admits all 18 phases and the action rule admits 17, so no nominal
admission gain is measured. All 32 combined gate/information tests pass. This
is a completed recorded-data method comparison, not a new replay or rollout gain;
the active acquisition's gate and teacher remain unchanged.

**Information-consistent teaching:** a completed
[recorded-data information ablation](INFORMATION_CONSISTENT_TEACHING.md) shows
that revealing scene summaries at 1.00 s permits a common WAIT followed by six
passing continuations, while revealing them at 1.40 s limits any finite
observation-consistent policy to five. The scene-wise bank remains 6/6.
The extension now constructs the passage-first optimal shared policy and trains
the existing ridge equations with group-consistent regret targets. A controlled
three-scene test corrects an unrealizable WAIT preference, improving the common
policy from one to two passages. On the actual six-context recorded tables,
both teachers tie at 6/6 or 5/6 depending on information timing; nominal fitted
models are identical. This is a completed fixed-data method comparison, not new
physical policy performance. Seven
replay controls also fit the first acquired checkpoint successfully; all tie at
5/6 recorded development proxies. Their M2/M4/M8/M16/M32 worker is live and
waiting for completed corpora. Physical acquisition continues separately.

**Expanded construction experiment:** the completed
[3,840-candidate comparison](EXPANDED_DEVELOPMENT_STUDY.md) finds that 169/645
reference-selected pairs retain the executed geometric contrast. This is a
construction-mechanism result, not physical passage. The five-arm, three-seed
8/16/32-encounter plan is fixed and its continuation worker is waiting for the
original M4 corpora. The original acquisition completed all twelve first-round
models at 16:33 UTC and began M2 acquisition at 16:34 UTC on September 9.
Reserved evaluation remains untouched.

**September 9 method update:** completed [decision-learning and replay
experiments](DECISION_LEARNING_RESULTS_20260909.md) on the 42-branch development
corpus. A separate feasibility/time learner reduces training regret but ties
ridge at 4/6 whole-context holdout branch proxies. Four replay controls also tie
at 4/6. Exact student features form six distinct groups at every phase; the
observation-consistent teacher therefore finds no nontrivial collision in these
data. The [revised paper](submission/traversal_method_v2.pdf) now defines the
continuation extension and coverage weights, explains passage versus recovery
time, and moves initialization history to the supplement. Larger independent
training corpora and actual final-policy evaluation remain necessary.

The [project guide](PROJECT_GUIDE.md) and
[current execution status](TRAVERSAL_V2_STATUS.md) describe the present seven-schedule
traversal interface. The [42-episode teacher archive](PORTABLE_SIX_CONTEXT_DATASET_V1.md)
and its NumPy quickstart are complete. The separate
[36-attempt actual policy archive](PORTABLE_POLICY_PANEL_V1.md) retains 35 complete
captures and one partial verified failure: 10,619 physical rows, 10,618 sensor
packets and 42,476 physics steps. Original scorer versions remain separate, with
the partial attempt's missing sensor packet and null cost preserved explicitly.
These are development releases; held-out gains and curriculum superiority remain
unestablished. The adopted continuation completed its first primary bootstrap
model at 04:34:26 UTC on September 9. It uses two consequential decisions and
one exact measured-tie initializer from the original seven captures, with no
new physics from reuse or fitting. The [bounded status receipt](PRIMARY_ACQUISITION_TIE_CONTINUATION_V1.md)
records adoption, preserved failures and V8 accounting; curriculum checkpoint
results remain pending.

The reserved handoff was [stopped at 06:33 UTC](/home/linjiw/research-data/groot-wbc/m2s-reserved-handoff-hold-v1/decision.json)
to finish development choices before evaluation. The primary acquisition
dispatcher remains active. Its current outputs are in the
[dispatch directory](/home/linjiw/research-data/groot-wbc/m2s-primary-acquisition-dispatch-tie-proposed-v5).
Reserved outcomes remain unexecuted. These model counts are acquisition progress,
not traversal-performance results.

## Historical snapshots

**September 7 result snapshot:** [82/82 labeling commands and 366/540 policy
executions are admitted](M2S_ICRA_366_RESULT.md). All four primary fits and 24 registered
refits are complete. Analytic-trained passage is 30/61 conditions, versus 21/61 for
uniform, target-only and background-only Motion2Scene; scripted rays pass 36/61.
Analytic provides nine additional contact-qualified passages across three inspected
carriers. Motion2Scene acquires zero generated groups, so its learner uses only six
shared backgrounds. The equal-24-label goal fails in three arms. This is an ordered
partial panel, not a learned-generator advantage or fresh-source result.

**Remaining:** 66 traversal and 108 background executions, final grouped comparison
and full cost accounting, then the September 10 claim decision and manuscript finish.
The supervisor remains active under unchanged memory and rolling-budget limits.
[Working manuscript](ICRA_MANUSCRIPT.md) · [All outcomes](assets/icra-366-outcomes.pdf) ·
[Records and saved models](evidence/icra-results-366-20260907/exports.json).
No generator refit, extra data budgets or final-source acquisition is added. The
original 480 MLP assignments remain paused.

## Historical material before the fable scope revision

**Current stage (September 7):** [the transition-construction pilot](TRANSITION_CONSTRUCTION_RESULT.md)
adds twelve matched empty-scene executions, qualifying carriers 41001/41002/41003 in
both physics seeds. Only **1/12 assigned scene slots** survives screening, from the
analytic arm on 41001; all learned slots are refused. Its four physical executions
produce walk-fail/d040-pass contrasts in both seeds. d040 records 0 N beam contact
and returns, while the separate endpoint-tracking metric still rejects both runs.

A finite support map finds 67 nominal witnesses after basic/visibility/reservation
checks, but only one under the inherited perturbation audit. The frozen initializer's
trust boxes cover none of the robust grid witnesses. This identifies both limited
geometric support and limited proposal reach. It does not prove infeasibility or
establish a learned-generator advantage. Original v1 remains 120/600 with 480 paused.

**Next stage:** [a separately registered nominal-task acquisition study](NOMINAL_TRANSITION_NEXT_STAGE.md),
with perturbation results reported separately, transition-conditioned initializer
supervision and equally informed analytic construction. Complete physical labels
before the next shared-learner comparison. The final five-arm 24/48/96 curves and
actual new-source acquisition/evaluation remain outstanding.

[All four new physical replays](assets/transition-construction.mp4) ·
[Full funnel, failed predictions and costs](TRANSITION_CONSTRUCTION_RESULT.md) ·
[Source archive](evidence/transition-stage-source-20260907/research-source.tar.gz).
These are forced-command development experiments; no new selector is trained.

**Previous breakpoint snapshot (September 7; superseded above):** [the selector breakpoint study](SELECTOR_BREAKPOINT_RESULT.md)
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
