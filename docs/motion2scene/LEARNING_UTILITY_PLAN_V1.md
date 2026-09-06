# Motion2Scene: learning utility is the next claim-bearing experiment

Working title: **Motion2Scene: Learning Task-Relevant Environments from Humanoid
Motion Contrasts**. Updated 2026-09-06 in response to the user's research guidance.
This is a prospective comparison design; it is not a completed learning result or
a physics-spend manifest. Generator freeze and interface tests have separate records.

The thesis to test is that executable motion contrasts supervise environments that
teach a perception-conditioned policy when adaptation is useful. The changed component
is training-scene construction. SONIC, the reference bank, command legality, learner
and evaluation suite remain shared. No new generator architecture, skill family,
distillation improvement or complete collision certificate is a prerequisite.

## Immediate contract

The [12-cell action study](ACTION_LABEL_COMPLETION_V1.md) completes six missing controls
and tests 0.20/0.30/0.40 s commands in the earlier-beam scene. The old privileged rule
already requests at the earliest legal time, 0.20 s; it is not an optimal timing oracle.
It is a scripted privileged baseline. Its shared failure with the reactive rule
suggests a transition/execution limit, but does not establish the cause.

Use one encounter decision initially. A0 commits to walking; A1 requests d040 now and
returns at the shared recovery phase. Labels are Y(scene, pre-decision state, phase,
recorded history, actual command), never whole-motion or wait-then-switch labels.
Retain both binary outcomes, including both-fail. A missing or mismatched comparator
is masked. Exact recorded prefix agreement is checked across state, velocity, actions,
motion tokens and references; this is replay matching, not an internal-state snapshot.
If the new explicit command path does not reproduce the prior 0.20 s behavior, resolve
that integration issue before a dependent policy fit or deployment.

The current 16 scene–seed pairs are interface development only, not a training corpus.
In particular, 144 sensor values are input dimensions. The shared outcome learner now
has 214 inputs: those 144 values plus phase, current skill, realized observation age,
projected gravity, root velocities and 29 joint positions/velocities. Its two heads
predict the two supported command outcomes. Architecture: 214→64→32→2, ReLU hidden
layers; train-only normalization; masked binary cross entropy; Adam 1e-3; 1000 full-batch
updates; fixed endpoint, no test-driven checkpoint selection. Five optimizer seeds
[8501,8502,8503,8504,8505] are paired across arms. CPU unit tests with synthetic labels
are software validation, not fits on robot data or evidence of learning utility.

At deployment, the learner makes the request. The shared manager checks only phase
and reference continuity. The old upper/lower-ray rule remains a separate baseline;
it cannot override the learner based on scene geometry. Prefer walking if both predicted
outcomes exceed 0.5; otherwise choose predicted-feasible d040. Neither positive yields
a refusal plus a disclosed neutral fallback, not a physical stop. Report refusal
classification and subsequent actual passage separately.

Bind features to the state at packet/decision capture, not a later trajectory frame.
The legacy packet index and recorded qpos index differ by one in existing replays;
verify the recorder phase before extracting proprioception for fitting. Requested
250 ms delay is 13 capture steps: records must show 0.260 s observation age for every
delivered packet. Warmup packets remain unavailable, not future observations.

## Four arms, one learner

| Arm | Complete construction pipeline | Role |
| --- | --- | --- |
| Uniform | Same station [0.1,0.9], underside [1.1,1.45] m and beam family; common basic validity checks; no criticality filter or free retries | Untargeted data |
| Analytic | Existing event-aware envelope construction, global station search and distinct-output selection with matched motion access | Claim-bearing comparator |
| No-contrast model | Same EventMixture architecture, training sources, initialization and fitting updates; remove selection/preference and neutral-interference terms; target-only bounded correction and rejection | Contrast mechanism ablation |
| Motion2Scene | Frozen all8/8421/1200 local model, pattern17 search and declared independent verification/rejection | Full method |

The no-contrast arm must not regain neutral interference through search, ranking or
rejection. Its entire pipeline sees target clearance for optimization; neutral geometry
may be measured afterwards for audit but cannot select its outputs. Match basic asset,
finite-parameter, supported-scene and initial-penetration checks across all arms.
The existing 127/128 analytic result is the baseline to retain, not a weaker random
local search renamed analytic. No diffusion baseline is claimed.

For a d040 command bank, construct generator inputs from that exact target reference,
with shared-origin route and DOF/frame binding. The historical generator was fitted on
local 55 mm contrasts; do not silently feed its old d055 scenes as if conditioned on
the deployed d040 target. This is an input binding step, not generator refitting.

Every arm uses the same fixed background quota: one quarter of assigned encounters,
equally absent, raised and blocked. Add beam-removed paired observations at the same
phase, with their acquisition cost shared and disclosed. Retain every assigned proposal,
rejection and physics failure. No arm gets uncharged retries or positive-only refills.

## Primary comparison and independent examination

Primary question: **at the same number of complete labeled encounters, does Motion2Scene
improve the fixed learner over the strong analytic generator?** This isolates data
quality. A second separately reported acquisition-budget analysis includes generator
fitting, teacher acquisition, proposal/search/rejection, verification and both action
executions. Equal observed rates do not establish noninferiority; that requires a new
prespecified margin and interval analysis.

The main data budgets are proposed as 24, 48 and 96 complete encounter groups per arm,
with independent fits at each size (4×5×3 = 60 fits). Checkpoints of one full-data fit
are not three data budgets. A group is a scene with complete command outcomes; sensor
dimensions, frames and optimizer seeds do not create independent scene/source samples.
Freeze the exact acquired lists, nested subsets and accounting before fitting. The
largest budget is the primary endpoint; smaller sizes are data-efficiency endpoints.
A failed acquisition quota is reported and charged, never silently replaced.

Construct test layouts independently of every generator, with disjoint station/height
combinations and fixed absent/raised/blocked suites. First use a common qualified bank
for all arms; label the result unseen-layout transfer on that bank. Reserve genuinely
new ancestors for source transfer with the same qualified switching contract. All
motion derivatives, scenes, physics repeats and sensor traces stay in their ancestor
split. 43001–43008 remain excluded from fitting and selection, and are already inspected,
not a newly untouched confirmatory test bank. Source identifiers and acquisition seeds
for the final transfer panel must be registered before generation, with full refusals.

Main endpoint: source-averaged contact-qualified passage plus the existing recovery
criterion on the assigned traversal suite; for the one-bank first stage, report layout-
averaged outcomes explicitly. Report unnecessary crouch on absent/raised and infeasibility
classification on blocked separately. Keep >1 N per-body sampled contact violations,
force traces, resets, falls and refusals. Pair optimizer seeds across methods; interval
resampling must keep source groups together. Do not demand every source improve; the
final registration must set the aggregate effect and acceptable tail-regression limit
before outcomes. Historical all-source predicates remain unchanged.

Add a phase-only diagnostic and a fixed observation-swap test. One-shot decisions from
a shared prehistory limit learner-induced shift within an encounter; any later sequential
extension needs a fixed, matched data-aggregation round for every arm and new labels.

## Resource-feasible schedule

The completed 42-cell panel cost 0.400278 GPU h: 34.310 s/cell on this machine, including
startup. The suggested 2880 final evaluations alone project **27.448 GPU h**, exceeding
the standing 24 h/week before labels and controls. This is a projection, not a spend
request or power calculation. Do not launch that matrix unchanged.

Before final registration, price training-label acquisition and every rollout. One
feasible reduced planning option is 20 primary policies × 12 independent encounters ×
2 physics seeds = 480 common-bank executions (~4.575 h); transfer of one prespecified
optimizer seed per arm over 144 assigned source encounters = 576 (~5.490 h). This gives
one fit seed per arm in transfer and must be described that way; it cannot substitute
for 20-policy source-held-out evaluation. Fit all 60 models cheaply on CPU, then choose
which endpoints receive physics before inspecting fit results. Shared outcome-table
selection can be a separately labeled counterfactual evaluation, not invented policy
executions. Final source count and power remain unsettled until acquisition is costed.

September 6–7: labels, timing and input/command binding; freeze generation and learner.
September 8–9: small development comparison, independent test reservation and costed
main manifest. September 10–12: fixed learning comparison and transfer. September 13–14:
freeze evidence, rebuild four principal figures and finish the eight-page paper.
No schedule entry turns an unfinished hypothesis into a paper claim.

The official call, checked September 6, gives September 15, 2026 (11:59 PST) and eight
pages including references and supplementary content. Video is a separate attachment;
upload windows are August 5–September 9 and September 17–22, with uploads unavailable
September 10–16. Confirm the submission portal's clock when preparing delivery.
[Official ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

## Positioning and claim–evidence map

1. **Method hypothesis:** local motion contrasts identify useful constraints. Existing
   held-out proxy results and conditioning controls support geometric utility; analytic
   generation remains strong. The new experiment tests its value as training data.
2. **Embodied evidence:** 18/24 requested generated-scene slots separate the declared
   contact outcomes, with refused sources retained. Upright still crosses; d055 is not
   proved necessary when d040 passes. The 42-cell panel characterizes a scripted interface.
3. **Learning hypothesis:** complete transition-conditioned outcomes train a shared
   perceptive selector better or more cheaply than analytic data. No result yet. If
   only uniform loses, distinguish useful critical scenes from a learned-generator gain.

LfLH already learns obstacle distributions and trains planning from them; inverse
motion-to-environment direction alone is not our novelty claim.
[LfLH](https://arxiv.org/html/2108.09793v1). Perceptive Humanoid Parkour demonstrates
perceptive skill chaining, and HumanoidPF couples traversal learning with hybrid scene
generation. Our planned comparison isolates training-data construction rather than
reproducing those whole systems. [Parkour](https://arxiv.org/abs/2602.15827),
[HumanoidPF](https://arxiv.org/abs/2601.16035). SONIC is the retained tracking foundation.
[SONIC](https://arxiv.org/html/2511.07820v3). Sequential learner-induced distribution shift
motivates matched aggregation if the one-shot contract is extended.
[DAgger](https://proceedings.mlr.press/v15/ross11a.html). Grouped uncertainty and paired
comparisons, not just point estimates, guide the evaluation plan.
[Statistical evaluation](https://agarwl.github.io/rliable/).
