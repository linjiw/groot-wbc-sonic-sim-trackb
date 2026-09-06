# Motion2Scene: counterfactual constraint synthesis, next stage

Registered development plan, 2026-09-06. This supersedes the next-action recommendations
in NEXT_RESEARCH_PLAN.md; previous protocols and results retain their original meaning.

**Goal:** generate scenes that change which declared humanoid motion is feasible under
a fixed controller, and improve verified yield and station coverage per total compute.
The immediate milestone is a measured development intervention on original-clock 41002,
an independent teacher-coverage diagnostic, and an event-aware analytic competitor.
The two-week program below is a research target, not work already completed.

## Argument and evidence

| Proposed contribution | Existing evidence | Decisive next test | Limit |
| --- | --- | --- | --- |
| Counterfactual feasible-set formulation for a target and declared alternatives | Two-margin geometric evaluator and event-conditioning controls | Fixed beam versus beam removal, two matched motions | Finite capsule checks do not prove native-collider or continuous-time feasibility |
| Coverage-aware amortized synthesis with a small retained local network | Local 243/288 versus pooled 6/288; fresh learned pattern 384/384 versus uniform pattern 30/384 | Original, hybrid, cost-matched geometry continuation, coverage hybrid and analytic baseline | Eight fresh sources from one construction/backend; existing distillation does not preserve every source or coverage |
| Fixed-controller execution and downstream utility evaluation | Original-clock 41002 repeats in an empty scene | Four intervention conditions, then eight source groups; structured skill selector | These are proposed experiments, not current contributions established by results |

Evidence: EVENT_SCALING_V1_RESULT.md, FRESH_SOURCE_V1_RESULT.md,
DISTILLATION_V1_RESULT.md, evidence/repeatability.json. Existing results are inherited
records; reading them is not an independent rerun. The new scripts retain raw arrays,
hashes, query counts, runtime and failed cases.

Define F(M+, A) by minimum target clearance over placement uncertainty >= 0.01 m,
and maximum alternative clearance <= -0.01 m for every declared alternative.
Here those extrema range over the recorded finite poses and placement offsets.
Binary upright/d055 separation is primary. A valid d040 alternative remains in any
later minimality experiment even when its achieved reduction misses a semantic threshold.

Code inspection: event_features encodes 29 target capsules × 7 values, progress and time
(205 channels), interpolated to 64 route stations. It does not encode alternatives or
construction event IDs. Keep this restricted target-only model for the fixed upright
construction; independently varying alternatives will require pair/set conditioning.

## Workstream A: execution gets the first available simulation slot

Use the existing selected analytic beam_021, median interval height 1.2652671813964844 m,
centre (2.696616322673983, 0.22645592215640215), yaw 0.07179232999993775 rad,
depth 0.10 m, width 1.20 m, thickness 0.10 m. Selection is frozen from existing
whole-world geometric evidence, including previous achieved trajectories. Thus this is
development qualification of an analytic beam, not learned-generator execution yield.
Never adjust its height after observing intervention outcomes.

Keep original-clock references, release checkpoint, physics settings, start pose and
shared world frame. Build a collision-enabled beam; the old illustration USD has no
PhysicsCollisionAPI. Record imported collision prims, transforms, collision schemas,
filter relations, contact/rest offsets and physics settings from the running stage.
Beam-specific filtered forces must distinguish scene contact from net body forces.

First run upright sensor controls: the same beam raised to 2.0 m (clear) and lowered
to 1.10 m (intersecting), seed 7910. These test observation infrastructure, not the
counterfactual hypothesis. Clear requires no measured beam force >1 N; intersecting
requires at least one >1 N observation. Stop the inference branch if either fails.
Missing instrumentation is infrastructure failure, never zero contact.

Then run original walk and d055 with beam enabled/disabled at matched seeds 7911–7913:
12 runs, all retained, no dependency skips and no scientific retries. Disabled beam
keeps the same floor, scene and sensors; only collision/visibility is disabled.
Contact sampling frequency and thresholds are explicit numerical detection limits.
Record all positive forces and thresholds at 0.1/1/10 N as diagnostics; primary >1 N
is frozen. No claim of continuous zero contact follows from sampled zero forces.

Local passage requires every recorded robot body origin to cross the downstream beam
plane plus 0.10 m in the original shared frame, remain downstream for 0.30 s, and
remain upright (root z >=0.50 m and gravity alignment >=0.5 during stabilization).
Require no measured prohibited beam contact through stabilization. Report native-body
extents separately; body-origin crossing is explicitly a local passage proxy.
Never translate each achieved route separately. Preserve separate flags for contact,
incomplete crossing, falling, and post-crossing tracking failure; the last is diagnostic
in addition to the frozen passage criterion. Trim at the first motion reset, never join
two episodes to obtain a crossing. Report all four rates and their difference-in-differences.
Require both absent motions 3/3, present target 3/3, present upright 0/3 with observed
beam contact for strict development separation. A partial outcome is still a result.

Serial runs, 375 s per run, <=14 runs / 1.46 contended GPU-hours, 9,000 MiB free-memory
gate, standing 8/day and 24/week envelope. Resource preflight yields without stopping
another workload. Each launch needs the completed hash-pinned manifest and stop rules.
Infrastructure retries require a separate incident record and new output directory.

If native execution consumes the narrow geometric interval, stop generator tuning on
this pair and return to achieved behavioral separation/motion acquisition. If controls
and pair pass, freeze generated beams before expanding to eight selected source pairs,
three placements and three seeds: 144 present plus 48 shared absent runs = 192.
Report the acquisition funnel; sources, not runs, are the independent units.

## Workstream B: coverage and an analytic competitor on CPU

First diagnostic (bounded at 1,800 seconds, two CPU threads, no fitting): load the
24 training cases and 16 observed development cases from the frozen distillation
registration. Exclude all 430xx sources. Independently evaluate a fixed 20×18 grid
of station/height centres over [0.10,0.90]×[1.10,1.45] at all 113 audit offsets.
Station-bin width is 0.04; height-bin width is 0.35/18. Retain both constraint scores,
passing centres, failures and unresolved cells. These are passing centres, not certified
cells; a zero map can miss a thin region. No map evaluation supplies optimization feedback.

Compare existing 58 teacher points to the TRAIN maps and original/hybrid raw and search
outputs to the DEV maps. Never compare a development case to a different training case
as if their feasible sets were the same. Reference-relative station coverage is the
fraction of independently passing reference station bins reached by accepted outputs,
at eight requests per fitting-seed/case job. Also report accepted bins outside the grid
support, empty-map cases, and valid yield. Existing teacher counts are not fixed-eight
sample coverage; keep their support diagnostic separate from proposal comparisons.

Analytic competitor: infer promising stations using only permitted target/upright capsule
geometry and the route. At each of 20 fixed stations, derive horizontal-overlap top
envelopes, propose the midpoint between target+0.02 m and upright-0.02 m, clipped to
the declared height domain. Rank by envelope separation, retain eight distinct stations,
and apply exactly the same 17-offset bounded pattern search and independent 113-offset
audit. Envelope overlap is a proposal approximation, not an interference witness.
Charge preprocessing, all search calls and verification. Existing global station grid
diagnostic is NOT free preprocessing for this baseline. No event IDs or development
map feedback. A later bounded global solver arm can use explicit charged exploration.

Decision: distinguish omitted teacher stations, student concentration, height failures
and grid-resolution uncertainty. Only then freeze a training-only, query-capped teacher
with coarse exploration and boundary refinement. Weight cases equally, and within case
weight independently passing station bins equally; keep failures and unresolved regions.
Retain width 32, normalization, optimizer and existing preference/two-margin objective.
Use a weighted version of the existing energy-distance set loss unless a correct bounded
mixture density is implemented and tested; the current loss is not a log likelihood.

Next fitting comparison: original, existing hybrid, geometry continuation matched to
teacher+fitting query cost, coverage hybrid; three seeds. Budgets 0/5/9/17 plus adaptive
5→9→17 using search-side failure only. Independent audit remains outside feedback.
Primary paired source summaries: yield, reference-relative station coverage and total
cost. Report source bootstrap 95% intervals (10,000 resamples, fixed seed), full source
table, and a declared worst-source guardrail of at most 2/48 fewer accepted outputs.
The guardrail is a development selection choice, not a power calculation. A candidate
must improve cost at matched yield/coverage, or yield/coverage at matched cost. Ties at
ceiling may win on cost. If the analytic method dominates, retain it for this beam family.
If coverage distillation fails, retain the existing learned pattern system.

One geometry query = one whole-motion clearance at one beam placement. Eight outputs
cost 1,808 audit queries and 452 additional Torch crosscheck queries in the existing
implementation. Thus 17/5 search evaluations cost 6,884/3,620 online queries including
the current audit AND crosscheck, a 47.4% reduction before acceptance/offline costs.
The 6,432/3,168 illustration excludes crosschecking. Record acquisition, teacher,
fitting, proposal, search, audit and crosscheck separately, with median/p90/p95 latency.

## Bounded program and branch endpoints

| Stage | Deliverable | Decision |
| --- | --- | --- |
| Days 1–2 | Collision contract + paired intervention; grid/teacher diagnostic; analytic baseline | Is the current pair discriminative in simulation? Which coverage mechanism is missing? |
| Days 3–6 | Three-seed coverage comparison and fixed/adaptive search frontier; more qualified pairs | Keep coverage hybrid, existing system, or analytic synthesis |
| Days 7–10 | Freeze method, then acquire ~16 fresh reference source groups; eight-group execution experiment | Transfer under the same backend/construction, source uncertainty and refusals |
| Final stage | Small structured scene skill selector and evidence package | Does generated data improve useful motion choice? |

Selector inputs: beam geometry and approach state; choices walk/crouch/refuse. Fixed
controller and motion bank. Compare random, analytic, learned-search and selected
distilled generation at equal total cost and equal accepted example count. Independently
generate tests containing empty/high beams, distinguishing beams and infeasible scenes.
Measure passage, beam contact, unnecessary adaptation and refusal; keep an analytic
selector. This is not a perception or end-to-end RL study.

Defer furnished rooms, dynamic obstacles, new controllers and architecture sweeps.
First expansion is varied beam geometry and motion events; lateral gaps need qualified
pairs. Do not start every later stage automatically before resolving its branch decision.

## Positioning

The intended distinction is declared counterfactual alternatives, motion-feature transfer,
a fixed controller and measured verified-synthesis cost. INFERACT jointly optimizes its
controller and scene generator; adding physics alone is not a novelty claim
([primary paper](https://arxiv.org/html/2405.12460v1)). Contact generation and force/penetration
are distinct; inspect the actual simulator settings rather than assuming contact means
penetration ([PhysX documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html)).
The supplied LfLH/LfH-CP, MIME/SUMMON, ReGen and learned-search comparisons inform the
working plan; verify their primary sources before writing a final novelty claim.
