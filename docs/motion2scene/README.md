# Motion2Scene visual research report

**Current stage (September 6):** [the independent-layout evaluation](INDEPENDENT_LAYOUT_V1_RESULT.md)
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

**Next research stage:** finish the remaining frozen layout and control assignments,
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


**Latest (2026-09-06):** the [generated-source execution pilot](SOURCE_EXECUTION_V1_RESULT.md)
completes 52 physics cells: six of eight source pairs qualify and all their three
frozen learned beams separate upright from d055, **18/24 requested slots** with
source refusals retained. All 18 target runs record zero sampled beam force;
upright contacts but still crosses. The [six new d040 runs](BEAM_D040_EXECUTION_V1_RESULT.md)
also pass 3/3 with the selected 41002 beam, contradicting d055 necessity for that scene.

The [native 200 Hz temporal audit](TEMPORAL_NATIVE_AUDIT_V1_RESULT.md) retains a
10.075 mm worst interval lower bound for the declared d055 interpolant. The
[384-output local pose audit](FRESH_LOCAL_POSE_CERTIFICATE_V1_RESULT.md) supplies
conditional nonzero six-dimensional domains under the static proxy model, with a
preserved and explained clearance-cap reproduction failure. These are separate
contracts, not a combined native continuous-time guarantee. The
[ICRA completion plan](ICRA_COMPLETION_PLAN.md) makes downstream exteroceptive
traversal learning the next central evidence gate; that comparison remains untested.

The following entries preserve the earlier experiment history.

The [fixed-controller beam intervention](BEAM_EXECUTION_V3_RESULT.md) now completes **14 physics runs**. Both motions pass 3/3 without the beam; with it, crouch passes 3/3 and upright fails contact-free passage 3/3. Upright still crosses but contacts the beam at **73.6–762.1 N**; crouch records zero beam force. Contact controls pass. This is one selected source pair and one frozen analytic development beam, not learned-generator execution yield. The registered separation criterion passes at **0.13057 contended GPU-hours**.

The [September 6 counterfactual stage](COUNTERFACTUAL_STAGE_V1_RESULT.md) also measures teacher support against 40 independent reference maps and adds a serious analytic baseline. Distinct analytic generation accepts **127/128** unique placements at **99.0% reference station coverage**, versus historical learned pattern search at **384/384** and **37.3%** coverage, with the same online query budget. Its exact-yield prediction fails on one preserved audit rejection. The [current plan](COUNTERFACTUAL_STAGE_V1.md) now advances to source-level execution and coverage-aware learning against that stronger competitor.

The [training-source distillation pilot](DISTILLATION_V1_RESULT.md) now completes a teacher bank, twelve student/control fits and 3,456 independently checked outputs. Hybrid raw accepts **334/384**, original raw **292/384**, and the stronger query-budget control **310/384**. Hybrid five-evaluation search accepts **377/384**, versus **384/384** for original seventeen-evaluation search. These are results on observed development sources; the 430xx pool remains excluded.

The recipe misses at least one registered source-wise raw-proposal or reduced-search criterion. Prioritize diagnosing teacher coverage, source-specific geometry and proposal concentration on development data before another fresh acquisition. Do not treat pooled gains or cheaper search alone as preserved performance. See the [research deep dive](RESEARCH_DEEP_DIVE.md) for the method, evidence and collision contract.

This static GitHub Pages report covers the fresh-machine shared-seed Kimodo motion qualification
study as of 2026-09-05. It does not pool outcomes with the earlier SweepCF or parallel learned
geometry-only hallucination track.

The [fresh-source audit](FRESH_SOURCE_V1_RESULT.md) now completes eight new source motions and 16 derivatives, with all seven methods frozen before acquisition. Learned pattern accepts **384/384**, uniform pattern **30/384**, original gradient **384/384** and the raw model **296/384**. Learned pattern improves over uniform on every source, but its prediction of higher acceptance than gradient fails because the counts tie. Probe/refinement retains one failure. The complete source-wise predicates and costs are retained. These are finite reference checks on CPU-generated straight walking, not physical execution or a collision guarantee.

The acquisition comparison is complete. Move the learning work toward [train-only search distillation](TRAIN_ONLY_DISTILLATION_DESIGN.md): test whether synthetic geometric targets improve raw proposals and reduce online query cost. Preserve all 43001–43008 sources and derivatives as excluded from training and selection. Another confirmatory transfer comparison requires a separately registered source pool. Develop imported-body and temporal clearance contracts alongside the learning work.

The earlier [fixed-budget station-search study](STATION_SEARCH_V1_RESULT.md) compares
three local alternatives on fresh draws. All reach 192/192 valid test outputs versus
187/192 for original gradient refinement, rescue five failures with zero paired losses,
and improve the separate known replay from 5/8 to 8/8. Gradient-free pattern search
uses 69.2% less measured search time. The [fresh-source acquisition](FRESH_SOURCE_V1_RESULT.md) follows this comparison; these earlier results remain
finite reference checks on observed development sources.

The [matched-query refinement study](REFINEMENT_V1_RESULT.md) adds a bounded
correction layer and a reusable checked sampler. Test acceptance rises from 123/192
(64.1%) raw to 189/192 (98.4%), versus 152/192 for ranking extra learned samples and
26/192 for uniformly initialized refinement. The hard case improves from 0/24 to
21/24, but a fixed fresh batch still rejects 3/8. The subsequent station study targets stalled station
proposals under a fixed query budget. These remain reference-only development results.

The [source diversity and unseen-phase study](SOURCE_PHASE_V1_RESULT.md) adds
80 targets from 16 sources and 12 fits. Doubling training sources raises unseen-location
yield from 49.5% to 62.0% at equal compute, or 64.1% with equal visits per motion.
One test source regresses, and the two six-source subsets differ sharply. The updated
plan prioritizes geometric feasibility/search and source composition before larger runs.
All 7,680 proposals receive independent reference checks; there is no execution guarantee.

The [paired timing follow-up](TIMING_DIAGNOSTIC_V1_RESULT.md) adds 12 separate development
executions and a complete held-out route measurement audit. It rejects the tested shared-clock
slowdown and identifies original-clock carrier 41002 for repeatability testing. No Q4 or learned
generator result is admitted. Its reproduction commands, evidence and hashes are separate from
the original visual-report bundle below.

The [repeatability and analytic-teacher follow-up](TEACHER_V1_RESULT.md) adds nine completed
physics runs, a new analytic capsule–box query, 27 finite-beam placement attempts, discrete jitter
checks and an intermediate-motion audit. The original-clock pair repeats at 3/3 seeds; the
intermediate level retains its required effect at 1/3. The beam remains a binary geometric
proposal and cannot establish ordinal minimality. No learning-eligible scenes are admitted.

The [learned-generator framework](LEARNED_GENERATOR_FRAMEWORK.md) reads the original LfH/LfLH
mechanisms against the humanoid problem and specifies a motion-only inverse objective, a first
model, distribution priors, collision-proof conditions, and discriminating experiments. It also
records a constructed counterexample to the archived sampled decoder's clearance claim. This is
a research design; it does not add trained-model or obstacle-present execution results.

The [first inverse-learning implementation and results](INVERSE_LEARNING_V1_RESULT.md) now add
12 CPU training runs with saved checkpoints, a motion-only mixture generator, loss ablations,
independent geometry checks, a sampling command, and distribution/validity plots. Preference
learning improves geometric yield on the selected development carrier; KL trades some yield
for broader placement coverage. Generalization and obstacle-present execution remain untested.

The [placement-uncertainty experiment](UNCERTAINTY_LEARNING_V1_RESULT.md) adds six CPU
training runs and checks all 768 comparator/new-model proposals at 113 placements.
Robustness improves in two of three seeds per KL family, but both regress on the third.
The rejection audit exposes a preference-loss/interference-margin mismatch. The
[updated research plan](NEXT_RESEARCH_PLAN.md) prioritizes an explicit interference
constraint, independent-carrier tests, and continuous geometry contracts.

The [explicit-margin experiment](MARGIN_LEARNING_V1_RESULT.md) adds twelve CPU runs.
Without KL, the preference model improves from 11/35/6 to 56/54/34 valid proposals out
of 64 across seeds; the KL setting still has a regression. A separate
[continuous-placement checker](PLACEMENT_CERTIFICATE_V1_RESULT.md) conditionally verifies
two selected scenes over their full translation/yaw domains. Numerical-error, imported
body geometry and between-frame assumptions still prevent physical certification.

The [grouped conditioning experiment](CARRIER_LEARNING_V1_RESULT.md) adds eight source
parents, 24 constructed early/middle/late crouch cases, nine generator runs and 36
per-motion searches. The conditioned model produces zero all-placement-valid proposals
on its two excluded test carriers and barely responds when the input event moves.
The [next model design](EVENT_CONDITIONED_GENERATOR_DESIGN.md) preserves event location
and specifies controls for the location prior introduced by that representation. This
is a reference-only development result; it does not qualify the original motion bank
or establish obstacle-present execution.

The [event and scale follow-up](EVENT_SCALING_V1_RESULT.md) adds 18 fits with two fixed
budget snapshots. Local features achieve 110/144 and 133/144 valid test proposals,
versus 2/144 and 4/144 for matched pooled features; swapping the event drops both to
zero. Doubling training parents helps in this comparison, while extra updates and
parameters do not consistently improve the small full-data model. The fixed fresh
sampler still accepts 0/8 on an early event, so rejection remains necessary. The
[data-scaling plan](DATA_SCALING_DESIGN.md) prioritizes source diversity and withheld
phases; [source notes](SCALING_SOURCE_NOTES.md) record the LfLH/LfH-CP recheck.

## Rebuild

Run from the repository root with the trusted local research bundle and standalone Motion2Scene
research repository available:

```bash
MUJOCO_GL=egl .venv_research/bin/python scripts/research/render_motion2scene_report.py \
  --data-root /path/to/research-data/groot-wbc \
  --research-repo /path/to/motion2scene
```

Dependencies: NumPy, MuJoCo, Matplotlib, Pillow, ImageIO/FFmpeg, and the repository's G1 meshes.
`--preview` renders only midpoint poster frames. Full generation produces three MP4 replays,
three posters, three SVG figures, portable evidence snapshots, and a hash manifest.

The first two videos replay recorded Isaac states. The third replays shared-clock reference CSVs
with a manually placed illustrative beam. No `mj_step` is called; none of these renders creates
a physics verdict. Existing recorder metadata supplies the verdicts for the first two videos.
The beam scene is not an admitted critical scene and must not be represented as a physical
preference-reversal result. Videos use a shared camera and elapsed time across panels.

`assets/manifest.json` records the selected source hashes and output hashes. Original absolute
machine paths are shortened in public JSON copies; source hashes and public snapshot hashes are
therefore separate. The full non-git motion bundle is needed to rebuild the media.

## Validation

```bash
.venv_research/bin/ruff check --select E,F,I scripts/research/render_motion2scene_report.py
.venv_research/bin/black --check scripts/research/render_motion2scene_report.py
git diff --check
.venv_research/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Browser validation: Chrome headless via Playwright, 1440×1000 desktop and 390×844 mobile;
no JavaScript errors or horizontal overflow. All three video sources decode and play; durations
are 3.967, 3.967 and 4.767 seconds, respectively. The synthetic beam slider is tested at both
endpoints (no feasible motion / neutral feasible). Public links, evidence hashes and media
metadata are checked before publication. No new controller experiment is performed by the original media renderer; the separate timing follow-up records its own physics runs.
