# Motion2Scene: completion gates for an embodied ICRA project

**Current stage (September 6):** the [comparative command corpus](COMPARATIVE_CORPUS_STAGE_RESULT.md)
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

**Next research stage:** [evaluate the frozen policies on the twelve independently specified layouts](INDEPENDENT_LAYOUT_EVALUATION_V1_PLAN.md).
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



Updated 2026-09-06. Registered historical experiments retain their original predictions and failure outcomes.

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

## The intended contribution and the present gap

Test whether motion-conditioned, counterfactual scene generation supplies more useful
training environments for humanoid overhead traversal than random generation, an
analytic curriculum, and unconstrained generation, at matched downstream compute.
Retain the release SONIC controller initially; change environment generation and a
perception-conditioned high-level traversal policy. The learned generator must earn
its cost against the strong analytic solver already implemented in this repository.

The proposed contribution set is:

1. A generator/search/verifier interface that proposes overhead constraints from a
   target motion and declared alternatives, with explicit acceptance and refusal.
   Test conditioning, verified unique yield, coverage and complete synthesis cost.
2. A measured connection between those generated constraints and closed-loop G1
   behavior under a frozen controller. Test source-level execution, removal controls,
   contact measurement, intermediate alternatives and tracking robustness.
3. A controlled demonstration that generated training environments improve an
   exteroceptive traversal policy's held-out success or sample efficiency. This is
   still a hypothesis. A geometry result or oracle skill selector cannot establish it.

These are intended contributions, not a current novelty claim. Check current primary
literature before freezing submission positioning. No venue deadline, paper length,
hardware requirement or acceptance prediction is assumed here.

## Evidence ledger: what is already known

| Claim | Evidence and experimental unit | Status / limitation |
| --- | --- | --- |
| Learned initialization helps bounded geometric search | [Fresh audit](FRESH_SOURCE_V1_RESULT.md): 384/384 vs uniform 30/384 over eight source groups | Static proxy checks; gradient also 384/384; output requests are not independent sources |
| Distillation removes search cost without loss | [Distillation](DISTILLATION_V1_RESULT.md): hybrid5 377/384 vs original17 384/384 | Strong claim fails; 70.6% reduction is search only; total online query reduction is 47.4% including audit/crosscheck |
| Analytic generation is a serious competitor | [Counterfactual CPU results](COUNTERFACTUAL_STAGE_V1_RESULT.md): 127/128 unique accepted, 99.0% reference station coverage | One preserved rejection; learned pattern has 37.3% coverage at the same online budget |
| Imported geometry retains this pair's margins | [Native audit](NATIVE_BEAM_AUDIT_V1_RESULT.md): six empty-scene recordings, 81 offsets each | Native primitive subset plus mesh outer enclosures; sampled poses; one carrier |
| A physical beam distinguishes walk from d055 | [14-run intervention](BEAM_EXECUTION_V3_RESULT.md): absent 3/3 each; present d055 3/3, upright contact-free 0/3 | Upright still crosses with contact; this is contact-free separation, not impassability; one analytic beam |
| d055 is the minimum required adaptation | [New d040 physics](BEAM_D040_EXECUTION_V1_RESULT.md): d040 clears the selected beam 3/3 with zero sampled force | Necessity relative to d040 is contradicted; old effect-threshold failures remain unchanged |
| Generated constraints separate motions in closed loop | [Source execution](SOURCE_EXECUTION_V1_RESULT.md): 6/8 pairs qualify, 18/24 requested slots separate | All 18 executed targets pass; one seed, observed development sources; no analytic execution comparison yet |
| Between-frame geometry agrees for a declared interpolant | [Temporal result](TEMPORAL_NATIVE_AUDIT_V1_RESULT.md): all d055 repeats retain >=10 mm interval bounds at 81 offsets | One carrier; actual unrecorded dynamics remain separate |
| Accepted outputs have nonzero local full-pose support | [Local pose audit](FRESH_LOCAL_POSE_CERTIFICATE_V1_RESULT.md): 384/384 conditional local domains | Static proxy model; smaller than original uncertainty domain; raw-score cap mismatch explicitly audited |
| Overhead sensing can select an executed adaptation | [Typed-ray interface](REACTIVE_INTERFACE_V3_RESULT.md): critical beam 2/2, blind contact 2/2; absent/raised specificity fails 4/4 | Scripted ideal lidar, one carrier; late wall-triggered switches, plus two retained integration failures |
| Upper/lower sensing rejects wall false positives | [Overhang follow-up](OVERHANG_INTERFACE_V2_RESULT.md): 4/4 negative controls, 2/2 critical-beam reactive passages; all 10 normal positive requests pass | One carrier, ideal sparse rays, narrow legal phases; late refusals remain task failures |
| Loaded references match the frozen source | [Bank audit](evidence/overhang-reference-bank-audit.json): all 14 banks reproduce source root XY and agree across cells | Prior training-loader freeze failure retained; this is reference fidelity, not physical accuracy |
| Authored outer-envelope crossing preserves the interface outcomes | [Native crossing audit](NATIVE_CROSSING_AUDIT_V1_RESULT.md): all 14 classifications unchanged, completion 20–40 ms later | 45 outer shapes at 50 Hz; not exact cooked meshes or continuous collision detection |
| Full-batch nominal native/time clearance | [Native batch](FRESH_NATIVE_TEMPORAL_V1_RESULT.md): 384/384 conditional bounds, minimum 21.623 mm | Declared reference interpolant and authored outer enclosures; capped attempt retained 0/384 unresolved |
| Pose and observation-delay tolerance | [42-cell stress panel](OVERHANG_VARIATION_V1_RESULT.md): reactive/oracle 9/10 each; 500 ms fails 2/2 | Pose prediction fails; one observed source and two seeds, ideal rays |
| Generated data improves traversal learning | No downstream policy comparison exists | Central unfinished gate |

The user-supplied diagnosis's “zero obstacle-present executions” is superseded by
the 14-run intervention. Its structural warning remains valid: one selected pair is
insufficient evidence for a generated-data robotics contribution.

## Gate 1: align the collision and time contracts

**Completed bounded step:** audit every recorded walk/d040/d055 repeat at 50 and 200 Hz
with both proxy and native geometry. Keep all 81 offsets, query timings and conservative
interval bounds. Report unresolved intervals rather than equating them with collision.
Do not interpolate contact forces or claim that body-pose interpolation recovers physics.

**Batch extension completed:** all 384 original accepted requests retain sampled
native clearance and pass the independently registered uncapped temporal bound at
nominal placements. The initial capped-bound attempt remains unresolved 0/384.
See [full report](FRESH_NATIVE_TEMPORAL_V1_RESULT.md). No beam was adjusted.
Report the full 384 denominator, unique placements, every rejection, and costs against
the proxy baseline. Reusing the old 430xx audit for this contract check cannot make it
a new held-out learning test; keep it excluded from fitting and method selection.

The runtime asset currently declares 26 capsules, one sphere and 18 meshes below
collision transforms. Audit the intended official G1 asset and the deployed asset
explicitly. Do not claim to have replaced these with “official convex decomposition”
without obtaining, hashing and checking that asset and its simulator cooking settings.
Keep outer enclosures for clearance and actual/inner geometry for interference.

Expand the placement branch-and-bound driver to every frozen output with a fixed
per-output budget; preserve conditional passes, violations and unresolved cells.
The existing domain is x/y/z plus yaw: nonzero four-dimensional volume in
R^3 × SO(2), not nonzero volume in six-dimensional R^3 × SO(3). A full SO(3) claim
requires nonzero roll/pitch intervals and a verified rotation-displacement bound.
Likewise, 200 Hz sampling alone is not continuous collision detection.

The new local certificate supplies such full-rotation neighborhoods for all 384
outputs using a box-corner displacement bound in a rotation-vector chart. It does
not discharge the original larger uncertainty domain; keep that branch-and-bound
task distinct. Likewise, its static fresh-source model and the separate 41002 native
temporal result cannot be combined into a single joint physical guarantee.

**Exit evidence:** imported-asset inventory, complete batch table, temporal and
placement scopes, numerical allowances, unresolved rates and end-to-end query cost.
100% validity is a prediction that can fail, not a result to preserve by filtering.

## Gate 2: demonstrate generated constraints in closed loop

1. The frozen eight-source pilot is complete: 16 absent + 36 present cells, 18/24
   requested slots separate and P1/P2/P3 pass. Preserve the infrastructure recovery
   and every source refusal in subsequent studies.
2. If source qualification fails, diagnose reference-to-execution tracking and motion
   construction before more generator fitting. If qualified targets hit beams,
   compare reference versus achieved native geometry at the frozen scene poses.
   If upright clears, diagnose missing counterfactual interference separately.
3. Carrier 41002's d040 diagnostic is complete: 3/3 absent and 3/3 present passes.
   The necessity of the deeper reference is contradicted for this scene. Retain
   both feasible crouch references in any downstream labels.
4. Register a height-response panel before execution. Start with five fixed underside
   heights around the existing beam (h0 + {-20,-10,0,10,20} mm), all three motions,
   three seeds and shared absent controls. Describe sampled transitions and seed
   variation; a finite sweep cannot prove a continuous dynamic separation interval.
5. After a successful pilot, freeze the generator and selection procedure before
   source acquisition and the larger three-seed comparison. Include analytic and
   random-generation execution arms with matched budgets. Do not repeat only
   successful physics seeds or count repeated carrier 41002 as new sources.

**Measurement hardening:** the interface pilot now captures beam-specific forces at each 200 Hz physics step
and validates 796 substeps against each 199-frame control trace with exact last-substep
agreement. Its blind/absent controls provide positive/negative contacts. Extend this
measurement contract beyond the selected carrier. The remaining broader task is to capture forces
with synchronization tests and positive/negative controls; keep the old 50 Hz results
as their own measurement tier. Score full native-collider crossing, contact through
stabilization, falls, resets, route progress and tracking error separately. Existing
scores reject >1 N; any new strict <1 N rule must be named and frozen prospectively.
Report peak force, contact duration and impulse using the actual sensor cadence.

**Exit evidence:** generated-scene source funnel, paired beam removal effects,
all-run force/progress plots, intermediate-motion table and measured height responses.
Do not call contact a fall, a block, penetration depth, or proof of ordinal minimality.

## Gate 3: establish downstream utility

The prior structured-geometry walk/crouch/refuse selector remains a useful diagnostic
and oracle baseline. The ICRA target in this plan additionally requires **exteroceptive
closed-loop traversal**. Use depth or a representation that stores overhead occupancy
(for example floor and ceiling channels); a single ground-height channel does not
represent both free space beneath a beam and its overhead surface.

Start with a small policy mapping local depth/overhead geometry plus proprioception
and approach state to feasible motion commands for frozen SONIC. Define legal command
switches and timing before training. Validate the interface in empty scenes and with
an oracle scene observer before attributing failures to perception or generation.
If the skill interface cannot support online switching, repair and separately validate
it before calling a one-time trajectory choice a reactive traversal policy.

The early interface trial failed wall specificity; its upper-height band confused a
wall with a traversable overhang. The [follow-up](OVERHANG_INTERFACE_V2_RESULT.md)
adds lower free-space queries and explicit legal-transition checks. Four absent/raised
controls now have zero raw overhang-positive frames over the full capture, independently
of the timing guard. Deliberately late requests are denied but remain failed avoidance.

A new-seed trial also exposed a training-loader augmentation in the alternate library:
its reference froze after frame 23. The repaired implementation explicitly reloads that
library for evaluation, preserves RNG streams and checks the realized bank against its
source route. All fourteen banks are saved and compared across runs. Keep the earlier
failed batch and lifecycle cost supplement in the evidence ledger.

Next register the [finite variation panel](OVERHANG_VARIATION_PANEL_DESIGN.md). Keep
wall and blocked-underpass controls; do not hide missed detections behind a time cutoff.
The default downward height map and reset-based resampling remain unsuitable for this
interface. Use the opt-in evaluation-bank implementation and freeze its exact hashes
before learning. No arbitrary-transition, noisy-sensor or transfer claim is supported.

Use imitation from verified scene–motion outcomes for the first downstream pilot;
retain RL as the next experiment if online adaptation requires it. Both can test the
generated-data hypothesis. Fix losses/rewards across scene-generation arms. “Without
manual reward sculpting” requires an explicit common training objective and disclosure
of every task-specific threshold or label construction; do not imply absence of design.

| Training environment arm | Controlled comparison |
| --- | --- |
| Motion2Scene learned + fixed search/rejection | Proposed data mechanism; charge generation, audit, physics labeling and failures |
| Uniform random placements | Same beam family, parameter support and downstream observations; record infeasible fraction |
| Analytic/grid curriculum | Use the strong distinct analytic solver; curriculum schedule frozen on training data |
| Unconstrained generative arm | Same scene representation/support and training data; remove counterfactual feasibility conditioning; name the actual model before execution |
| Oracle scene/analytic selector | Separates perception and command-interface failures from environment-generation failures |

A diffusion model should be selected only after checking its scene representation and
training-data compatibility; an arbitrary furnished-room model is not a matched beam
baseline. An unconstrained model using the same decoder is also a useful mechanism
ablation, but must not be mislabeled as a reproduced external diffusion method.

**Freeze before downstream spend:** source-family splits, scene seeds, sensor model,
action interface, policy architecture, five optimizer seeds, common loss/reward,
training-step checkpoints, budgets, success rule and stopping rule. Use a separate
generator-independent test bank spanning empty/high beams, separating beams, infeasible
beams and varied approach states. Hold out source ancestry, beam configurations and
approach variation; derivatives and scene jitter stay with their parent group. Never
let a failure observed in the test bank tune generation or controller parameters.

**Primary comparisons:** held-out contact-free traversal success and area under the
success-versus-training-transitions curve. Report transitions to a preregistered target
success rate, treating unreached thresholds as censored. Show all five training seeds
and uncertainty grouped by independent source/scene family. Report both equal downstream
transitions and equal total generation-plus-training cost; equal accepted-scene counts
alone conceal expensive refusal/labeling rates.

**Required secondary metrics:** beam contacts, falls, unnecessary crouches in easy
scenes, refusal on feasible/infeasible scenes, time to traverse including failures,
training wall/GPU time, generation queries, and scene-support coverage. Always refusing
must not score as success. Keep infeasible-scene decisions separate from feasible-scene
traversal. If confidence intervals remain inconclusive, report that result and narrow
the claim; do not prescribe a statistically significant gain in advance.

**Exit evidence:** reproducible learning curves, a held-out traversal table against
the above arms, observations/actions from actual closed-loop evaluation, and failure
analysis. This gate has no completed experiment yet.

## Paper assembly and branch discipline

I–III: define the operational need, the inverse constraint problem, alternatives,
and refusal. Define `y(S)=min{k:F(M_k,S)=1}` with an explicit no-feasible-member value;
name whether F is reference geometry or a closed-loop execution criterion. In a
stochastic controller setting report seed-dependent outcomes rather than silently
treating F as a deterministic physical law.

IV: local-event proposal, bounded correction and independent checker; explain
training/deployment information and costs. V: source-held-out geometric audits,
analytic baseline, ablations and conditional certificates. VI: generated-scene
physics and the full acquisition funnel. VII: downstream perception-policy learning
and its controlled comparisons. Limitations retain the weakest subgroup and every
failed preregistered prediction.

Review gates weekly; time estimates are planning targets, not promises. With the source,
temporal and intermediate pilots complete, prioritize beam-visible sensing and stable
online command transitions before downstream training. Freeze the height-response and
multi-seed comparison prospectively; coverage distillation v2 remains a separate CPU
study against the analytic baseline. No full-paper
completion claim is justified until the downstream gate is met or the user explicitly
chooses a narrower paper scope.
