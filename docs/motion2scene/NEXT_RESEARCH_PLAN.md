# Research plan: robust motion-conditioned inverse scene generation

**Latest acquisition preparation (September 6):** [the four-arm d040-bound pipeline](COMPARATIVE_ACQUISITION_V1_RESULT.md)
now records 64 outputs and 36 assigned generated slots, plus a shared background quota.
The common checks admit 9/9 uniform, 8/9 analytic, 9/9 no-contrast and 9/9 Motion2Scene
slots; the analytic test-neighborhood rejection is retained. A twenty-cell first-slice
manifest is frozen, with direct 0.30 s pre-command state capture. **0/20 runs have
started because GPU memory is below the registered 7500 MiB floor.** No new paired
labels, robot-data selector fit or learning benefit is claimed. Exact d040 input binding
is complete within its declared numerical tolerance. Final source reservation remains
unresolved after the bounded inventory timed out.

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


**Previous simulation stage (retained):** the [Isaac sensing/interface pilot](REACTIVE_INTERFACE_V3_RESULT.md)
now completes all 12 corrected condition/seed cells. A sparse collision-ray observer
selects d040 and returns to neutral without state/clock resets; both critical-beam
reactive trials pass with zero recorded 200 Hz beam force. Blind walking contacts the beam in both seeds. However, all four absent/raised
controls switch unnecessarily on the far wall at 2.58–2.64 s: specificity P1 fails. The initial disabled
scene-query batch and interrupted typed-callback attempt remain in the failure ledger.
This supports contact-free adaptation on one beam, but does not validate selector
specificity or a trained perception policy. Next distinguish walls from overhead
free space and enforce legal switch timing, then test height/position/delay variation
before executing the [matched learning-data
comparison](DOWNSTREAM_SENSOR_POLICY_PILOT_DESIGN.md).


**Current direction (2026-09-06):** follow the [ICRA completion plan](ICRA_COMPLETION_PLAN.md).
The [generated-source pilot](SOURCE_EXECUTION_V1_RESULT.md) completes 52 physics cells
and separates 18/24 requested slots, with two source refusals retained. The
[d040 intervention](BEAM_D040_EXECUTION_V1_RESULT.md) shows d040 also clears the selected
41002 beam 3/3, so d055 necessity is contradicted. The [native temporal audit](TEMPORAL_NATIVE_AUDIT_V1_RESULT.md)
and [384-output local full-pose certificates](FRESH_LOCAL_POSE_CERTIFICATE_V1_RESULT.md)
advance separate geometric contracts without establishing a joint physical guarantee.
Next validate beam-visible sensing and stable online motion commands, then compare
downstream policy learning against random, analytic/grid and unconstrained generation.
Keep complete source refusals, costs and the strong analytic comparator. The remaining
text preserves the prior planning record.

The [training-source distillation pilot](DISTILLATION_V1_RESULT.md) now completes a teacher bank, twelve student/control fits and 3,456 independently checked outputs. Hybrid raw accepts **334/384**, original raw **292/384**, and the stronger query-budget control **310/384**. Hybrid five-evaluation search accepts **377/384**, versus **384/384** for original seventeen-evaluation search. These are results on observed development sources; the 430xx pool remains excluded.

The recipe misses at least one registered source-wise raw-proposal or reduced-search criterion. Prioritize diagnosing teacher coverage, source-specific geometry and proposal concentration on development data before another fresh acquisition. Do not treat pooled gains or cheaper search alone as preserved performance. See the [research deep dive](RESEARCH_DEEP_DIVE.md) for the method, evidence and collision contract.

The next development experiment should isolate **teacher quality and retained support**
before increasing network size or acquiring another audit pool. The current teacher
retains 58 placements across 24 training cases, and its earliest-per-bin rule ignores
which retained witness has more clearance slack. Register a comparison of earliest
versus highest-slack accepted witnesses within the SAME bins and teacher-query budget;
keep the number/weighting of bins explicit so an apparent robustness gain cannot hide
concentration. Keep the student architecture, optimizer and geometric objective fixed.
This is a proposed ablation, not a measured remedy.

Report the known difficult development source 42007 explicitly: hybrid raw improves
only 7/48→9/48 there, while reducing search loses two outputs. Also retain the three-output
regression versus the query control on 41007 and the four reduced-search losses on 42008.
Use these as descriptive diagnostics; do not tune on the permanently excluded 430xx pool.
A claim that distillation replaces search requires both source-wise acceptance and full
teacher/training/verification cost accounting. The current 334/384 raw gain is real in
this panel, but it does not satisfy that stronger claim.

The working direction is a **local event-conditioned scene distribution trained through
fixed geometric feedback, followed by bounded correction and independent rejection**. The
[event/scale experiment](EVENT_SCALING_V1_RESULT.md) now supports useful conditioning:
110/144 and 133/144 valid proposals on its two test sources, versus 2/144 and 4/144 with
globally pooled features. Swapping the input event reduces both local counts to zero.

The [source/phase study](SOURCE_PHASE_V1_RESULT.md) now tests 4/6/8 training parents,
two six-parent subsets and two unseen event locations. At equal query budgets, eight
parents improve unseen-phase test yield from 95/192 (49.5%) to 119/192 (62.0%). Equal
visits give 123/192 (64.1%), but source 42007 regresses, and all three predictions of
improvement on every test parent fail. The two six-parent subsets score 83/192 and
118/192 at equal queries: source composition matters, and scale is not a uniform remedy.

The [matched-query refinement study](REFINEMENT_V1_RESULT.md) now supports a hybrid
inference pipeline. Learned initialization plus 16 bounded correction steps reaches
189/192 (98.4%) valid test outputs, versus raw 123/192, ranked learned samples 152/192,
ranked uniform samples 66/192 and uniformly initialized refinement 26/192. It rescues
66 failures without losing a raw success. Ranking and refinement tie on three test
parents; the refinement benefit is concentrated on the previously difficult 42007.
These are finite reference checks on observed development sources, not execution proof.

The [station-search follow-up](STATION_SEARCH_V1_RESULT.md) now removes the remaining
five original-gradient failures in a fresh-draw panel: probe/refinement, multiple starts
and pattern search each accept 192/192 test outputs versus 187/192 for original gradient,
with zero paired losses. All fix the separate known 5/8 replay to 8/8. Pattern search
uses 35.19 s versus 114.23 s of main-panel search time at the same query count. These
methods tie on observed acceptance; this does not establish a unique winner or prove
reliability on new sources. Accepted-bin counts also differ.

The [fresh-source audit](FRESH_SOURCE_V1_RESULT.md) now completes eight new source motions and 16 derivatives, with all seven methods frozen before acquisition. Learned pattern accepts **384/384**, uniform pattern **30/384**, original gradient **384/384** and the raw model **296/384**. Learned pattern improves over uniform on every source, but its prediction of higher acceptance than gradient fails because the counts tie. Probe/refinement retains one failure. The complete source-wise predicates and costs are retained. These are finite reference checks on CPU-generated straight walking, not physical execution or a collision guarantee.

The acquisition comparison is complete. The completed learning pilot follows [train-only search distillation](TRAIN_ONLY_DISTILLATION_DESIGN.md): test whether synthetic geometric targets improve raw proposals and reduce online query cost. Preserve all 43001–43008 sources and derivatives as excluded from training and selection. Another confirmatory transfer comparison requires a separately registered source pool. Develop imported-body and temporal clearance contracts alongside the learning work.

The completed pilot studies distillation from TRAINING-source refinements to reduce
inference cost. Keep fresh audit sources excluded from fitting, normalization, target
construction for training and method selection. Broader scene families require joint
whole-motion checks; imported body geometry, time between frames and continuous
placement uncertainty remain open contracts. The [data plan](DATA_SCALING_DESIGN.md)
keeps lineage and compute contracts explicit, and the [source notes](SCALING_SOURCE_NOTES.md)
connect the work to LfLH and critical-point hallucination.
The [framework](LEARNED_GENERATOR_FRAMEWORK.md) contains the LfH/LfLH source reading and
full mathematical design. The [first inverse experiment](INVERSE_LEARNING_V1_RESULT.md)
establishes a one-carrier optimization mechanism. The [placement uncertainty protocol](UNCERTAINTY_LEARNING_V1.md)
and its separate result determine which objective should enter the next study.

## 1. Align both geometric margins with the training objective

The [explicit-margin experiment](MARGIN_LEARNING_V1_RESULT.md) is complete. Without KL,
adding the interference barrier improves the preference model from 11/35/6 to 56/54/34
joint-valid proposals out of 64 across seeds. With KL, counts change from 17/20/0 to
9/38/42, including a regression. The constraint-only ablation does not consistently
replace preference across settings. Retain **preference plus both margins, without KL**
as the next development candidate and keep all ablations as controls.

This candidate still rejects 48/192 proposals, including seven target-clearance
failures. Its geometry queries are finite, its source carrier is singular, and accepted
bin counts do not establish support coverage. The full-support mixture cannot guarantee
all raw draws are valid. Keep independent rejection in the system, report its cost,
and do not spend the next study merely tuning weights on this carrier.

The grouped experiment now tests excluded source carriers under these margins and the
same uncertainty domain, with constant/shuffled-input controls and fixed-budget per-motion
search. Its failed conditioning result motivated the now-completed local-feature comparison,
without changing the margin definitions. Separate proposal quality from accepted-distribution quality. The next geometric
experiment should establish numerical/imported-body/temporal contracts for the bounded
placement checker described below before claiming executable scene guarantees.

Use the KL arm to investigate conditional coverage and the arm without KL as a
concentration baseline. Robust acceptance and diversity must both be measured before
selecting a production sampler. Keep the direct per-motion optimizer and analytic beam
sampler as competitors. The learned network has not yet shown an amortization benefit.

## 2. Establish independent carriers before testing generalization

The first reference-only registry contains eight source parents and 24 early/middle/late
crouch cases, split 4/2/2 by parent before construction. Every target and neutral passes
Q0/Q1. This establishes a reproducible grouped geometry diagnostic, not execution
qualification or Q4 admission. Previously inspected parents also cannot serve as fresh
confirmatory evidence. The completed expansion contains 16 sources and 80 targets,
with 8/4/4 source roles and two withheld event phases; all reference screens pass.
It retains the same distinction between reference diagnostics and execution qualification.

The audited timing and repeatability studies provide one repeatedly successful
upright/deep-crouch development pair, carrier 41002. The original timing trials on
41001 and 41003 do not pass the crouch execution/route gates. Repeating 41002 with more
physics seeds supplies execution variation, not more independent examples. The audit
covers these studies; it is not an assertion that every motion elsewhere in the
repository has been screened.

Build a carrier registry with source-generation identity, parent lineage, reference
hashes, desired route/event, transformations, controller/configuration hashes, and every
qualification attempt. Preserve rejection reasons. Split by the root carrier before
fitting the generator, prior, normalization, thresholds, or event extractor. All crops,
time warps, edited depths, execution seeds, and generated scenes stay with their parent.
Treat existing 41002 and previously inspected development examples as development data.

For each new carrier, construct or retrieve a same-task upright counterpart using the
existing frozen motion operators. Apply the existing reference and tracking checks,
then measure executed event magnitude and a shared-world geometric separation witness.
A reference-only separation cannot substitute for the recorded executed pair. For an
ordinal claim, additionally qualify intermediate levels; the existing 1/3 intermediate
result cannot be counted as a successful ordered ladder.

The next acquisition registration must fix candidate count, source seeds, train/
validation/test assignment, execution repeats, budget, stopping rule, and all gate
thresholds before spending physics. A useful pilot design target is 12 independent
admitted pairs split 6/3/3, but that is a proposed capacity target, not a promised
qualification yield or an adequate final publication sample. Plan and report the full
candidate funnel needed to obtain it. Keep the existing Q4 admission policy distinct
from a development-only model study; do not relabel one as the other.

## 3. Test whether the input motion actually matters

The first grouped diagnostic is implemented with station/height beams, nine generator
runs and 36 per-motion searches. Keep this family for the representation comparison to
isolate learning from scene-family complexity. Build the target feature from whole-body geometry and
root-relative motion with an explicit shared scene transform. Decode scenes back into
the same world frame used by every candidate; never align alternatives independently
inside the collision evaluator.

The next comparison should retain the completed controls and add the missing analytic
competitor:

- The same architecture receiving a constant motion input.
- Motion inputs shuffled among training cases, with geometry targets unchanged, plus
  event swaps within each excluded carrier. Record the exact permutation; the current
  cyclic derangement changes events and sometimes carrier identity.
- A per-motion mixture optimized to a preregistered convergence or query budget.
- Prior sampling with rejection and the analytic beam search.

Report raw robust yield, independent checked yield after rejection, feasible-region
coverage, and total time/geometry queries per accepted scene on each held-out carrier.
Keep preprocessing, optimization, verification, and rejection cost in the accounting.
For the simple beam family, use dense independent geometry to estimate support coverage.
Do not reward diversity generated outside the valid support. Report failures separately
by target collision, insufficient alternative interference, and out-of-family motion.

The decisive outcome is an advantage on previously unseen carriers over an input-agnostic
model and a competitive search method. Three optimizer seeds on one carrier cannot
supply that evidence. If conditioning fails this test, retain the optimizer as a useful
scene-search tool and diagnose representation or data coverage before scaling the model.

The first grouped pooled model failed its conditioning criteria. The local-feature
follow-up now passes both input-control predictions. Keep its small 12,118-parameter
architecture and compare 600/2,400 updates in the next data acquisition. The short snapshot
matches pooled test yield at lower fitting cost, but the individual sources change in
opposite directions. This does not replace the completed study's 2,400-update endpoint.

Prioritize repeated nested 4/8/16-parent curves with fresh excluded sources and withheld
event phases. Include the early-event weakness explicitly: current carrier 41007 accepts
only 18/48 local proposals for that event, and the fixed fresh sampler accepts 0/8 because all eight are
insufficiently discriminating against the upright alternative. The full local test audit
still rejects 45/288 draws, including two target-clearance failures. Add analytic search
and bounded geometric refinement as comparators before claiming end-to-end amortization.
Keep data-source acquisition costs and verifier costs visible. The reference-only data
scale target is separate from the execution-qualified pilot capacity proposed above.

## 4. Turn sampled clearance into a precise collision contract

Separate three outstanding contracts:

| Contract | Implementation needed | Acceptance evidence |
|---|---|---|
| Body geometry | Audit actual imported collider transforms and shapes, including hands; certified outer enclosures for clearance and actual/inner geometry for interference | Independent checks over all relevant links; no undocumented omitted collision geometry |
| Time | Specify motion interpolation; use validated continuous collision detection or conservative interval bounds on surface motion | Every time interval and forbidden body–obstacle pair covered, with tolerances and unresolved intervals recorded |
| Placement | Bound clearance variation over the continuous translation/yaw uncertainty set, or conservatively subdivide it | Every placement cell certified or explicitly rejected/unresolved; a finite jitter grid is not enough |

The [adaptive placement checker](PLACEMENT_CERTIFICATE_V1_RESULT.md) is now implemented.
For the two protocol-selected no-KL representatives, 1419 and 1623 queries discharge
710 and 812 cells covering the full four-dimensional placement domain. An independent
trace audit verifies the partition and checks every query against PyTorch. The result
is conditional on the assumed 1e-8 m numerical allowance and static capsule geometry.
It covers neither the imported robot nor motion between frames.

The checker uses the box displacement bound `||delta_translation|| + 2 R sin(|delta_yaw|/2)`
to bound fixed-segment distance changes over a pose cell. It subdivides unresolved cells
and requires every leaf to satisfy both margins. Budget/precision exhaustion remains
unresolved. Next establish rigorous numerical error/enclosure bounds and report verifier
cost across unfiltered proposals and distinct carriers. The selected 2/2 demonstration
is not a distribution success rate. Add the temporal contract before physical admission.

For a fixed beam pose, all target frames/body parts must clear. Each alternative needs
at least one verified interference witness; the witness time/body part may vary across
placement cells. Testing the actual geometry is necessary before interpreting proxy
intersection as an execution-relevant collision. Contact-support tasks require a
separate permitted-contact contract.

## 5. Extend the scene family only with matching motion evidence

Add lateral gaps next, with lateral body-envelope adaptation and qualified wider-body
alternatives. Then add step-over obstacles, with swing-foot clearance and foot support
phases. Parameterize actual 3D objects: dimensions, pose, support and object type. A
larger latent vector is not a substitute for checking all objects against the full
sequence and every candidate motion.

For composed scenes, use event-conditioned object proposals followed by a joint
whole-sequence check. Record which object explains which alternative's exclusion.
Objects that constrain no qualified alternative are background decoration; they do not
establish motion preference. Add a semantic furniture prior only as a separately sourced
prior, since motion-only data cannot identify real-world scene frequencies or semantics.

## 6. Validate execution and downstream usefulness

After the geometric contracts are satisfied, register matched target/alternative trials
in the same generated scene, obstacle-removal controls, and controlled placement shifts.
Record cause, body, obstacle, time, route/event retention and fall/contact outcomes.
Use the frozen controller. A recorded empty-scene motion with an obstacle overlay
cannot predict what happens after contact changes the motion.

Only then evaluate a fixed downstream scene-to-skill learner trained with competing
scene-generation methods and tested on independent scenes/carriers. Keep downstream
model capacity and training budget fixed. This experiment decides whether inverse
learning creates useful training data, beyond producing pleasing scene samples.
