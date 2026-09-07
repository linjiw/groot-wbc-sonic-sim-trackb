# Shared linear-control execution v1

This is a development follow-up to the [P0/P1 study](SELECTOR_BREAKPOINT_STUDY_V1.md),
registered after its outcomes. Original v1 remains 120/600 with 480 paused assignments.
No original checkpoint, test assignment, runtime or source bank changes.

P0 fitted exactly four deterministic regularized logistic controls from the original
fixed eleven-example subsets, using common physical scales. No diagnostic P1 labels
entered fitting. All fitting rules are in P0; this study performs no additional fit.
P1 measured one missed useful adaptation: height 1.27 m, seed 8512. Low height is
both-fail in both seeds; the other three conditions are both-success.

## Runtime representation and before-run validation

Use the existing frozen learned-command runtime by embedding each linear control
in its existing 214-64-32-2 checkpoint container. For each of the two linear logits,
the first layer computes its positive and negative signed forms; four ReLU channels
pass through an identity second layer; the output subtracts each signed pair.
All other weights are zero. This is the same logistic function, not another trained
MLP or a new optimization result. Store mean zero and the exact physical scales.

Before launch, hash-bind original linear weights, derived checkpoints, the P0 result,
P1 pairs and v1 blocks. Compare original linear and embedded-container readouts on
all 38 corpus observations and 120 recorded evaluation inputs: probability tolerance
1e-6 and exact action/refusal. Do not change the threshold to achieve equivalence.

## Assigned physics and predictions

Four controls (uniform, analytic, no contrast, Motion2Scene) × six matched original
layout/physics-seed conditions = 24 new serial Isaac Lab executions. Use the same
source 41002, 0.30 s one-shot decision, neutral/d040 bank, jump/phase guards, 3.3 s
return request, sensor queries and frozen scorer. Retain every assigned condition.

The recorded-input prediction is analytic: refuse at low height, request d040 at
middle height, walk at high height. Uniform and Motion2Scene refuse low and walk at
middle/high; no contrast walks throughout. These are now development predictions,
not independent test or source-transfer claims.

Registered execution predictions: the actual feature vector and model request match
the recorded-input prediction; paired pre-decision states/histories match P1; issued
commands follow the unchanged guard; and each actual outcome matches its P1 command
comparator. In particular, test the analytic control's useful request at 1.27 m,
seed 8512. Keep all failed predicates and scientific outcomes. No rescoring or refill.

Use direct feature/state/ray/bank and 200 Hz contact audits before admission. Report
contact-qualified body-origin passage and first-episode 0.3 s stability separately
from reset, return and later recovery. Refusal still commits to walking; it is not
physical stopping or successful avoidance on the low beam.

The maximum reservation is 24 × 375 s = 2.5 contended GPU h. Respect 7500 MiB free
memory, serial physics, 375 s cell timeout and the 8 h/day, 24 h/week envelope with a
fresh latest-activity budget check. A resource stop preserves pending assignments.

## Interpretation

This tests a common low-capacity/physical-scaling repair jointly. It does not isolate
normalization from capacity or regularization, and it does not validate Motion2Scene's
data-generation advantage. Any favorable analytic result strengthens that baseline.
There is one observed source, one deterministic control per arm, and a diagnostic
six-condition panel. These layouts cannot serve as untouched confirmation of a
revised method. The next method study must add execution-aware construction and more
qualified development carriers before the final 24/48/96 and new-source comparison.
