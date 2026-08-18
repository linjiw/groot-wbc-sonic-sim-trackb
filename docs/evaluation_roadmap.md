# What this dataset can evaluate, and what it would take to close the loop

Two things are worth separating: the evaluation contribution that is **already available** from the
data we have, and the perception work that would let a G1 policy be trained and measured in the
environments we design. The first is cheap and mostly unclaimed. The second is the real paper.

## 1. The controller is an evaluation instrument, not just an executor

SONIC answers a question no kinematic checker can: *is this motion physically executable by this
embodiment under closed-loop control.* Used deliberately, that makes it a **feasibility oracle** —
cheap relative to human review, reproducible, and grounded in the same physics the dataset is about.

Four labels that are usually collapsed into one "success rate":

| label | question | cost | current value |
|---|---|---|---|
| `embodiment_feasible` | can the body reach the pose? | CPU, ms | 96% of 150 |
| `controller_trackable` | can the controller execute it? | 1 rollout, empty scene | 96% of 47 joined |
| `semantically_valid` | does the *execution* contain the named behaviour? | predicate on execution | 32% of 75 |
| `scene_compatible` | does it succeed in *this* room? | 1 rollout per scene | the counterfactual itself |

The honest headline for a corpus is the **smallest** of these, not the largest. Reporting 96% when
the semantic rate is 32% is how synthetic humanoid datasets overstate themselves, and the fix costs
nothing but discipline.

**Three uses of the oracle that are available now.**

*Graded difficulty, not pass/fail.* Reference-tracking drift rises monotonically with how far an
adaptation moves a joint — 0.048, 0.104, 0.140, 0.223, 0.224 m/s at 0.000, 0.420, 0.619, 0.980,
1.000 rad within one motion. Drift is therefore a continuous difficulty score, usable for curriculum
ordering and for reporting how hard a clip is rather than only whether it survived.

*Retargeting quality, measured rather than asserted.* The gap between a reference clip and its
execution is exactly what a retargeter is supposed to minimise. We already record root drift,
tracking error and contact decomposition per episode; reported per behaviour class, that is a
retargeting-quality table for a generator-plus-controller stack, which is a thing the field lacks.

*Generator diagnosis.* Joining generation-side screens to controller verdicts localised the failure
precisely: crouch is the only behaviour with a kinematic failure rate (44% against 100%), and the
mechanism is the generator folding a waist the G1 cannot fold. That is a reusable protocol for
anyone building on a text-to-motion generator.

## 2. Why a *fixed* controller makes the learning experiment unusually clean

The selector chooses among prescribed motions; SONIC executes whichever is chosen. The controller
does not learn, does not adapt, and is deterministic under a fixed configuration — a property we
have measured, with two rollouts of the same clip diverging only after the frame an obstacle touched
one of them.

So any difference in outcome between two candidate choices is attributable to **the choice**, not to
control. Most humanoid work cannot separate those: a policy that both perceives and controls will
improve for reasons its authors cannot decompose. Here the control is held constant by construction,
which is what makes "does counterfactual supervision improve *selection*" a question with a clean
answer.

That is worth stating as a design property rather than an accident.

## 3. The perception path, concretely

**What exists.** Every rollout already records an ego camera on the head — 640×480, one frame per
control step, and it moves with the body. The scenes carry exact obstacle geometry. The perception
timing gate is built: `t_first_visible`, `t_adaptation_onset`, `t_bottleneck`.

**What is missing, in order of cost.**

1. *Depth.* The recorder writes RGB; the geometry a selector needs is depth or a local point cloud.
   Isaac can render it in the same pass. Cost: a configuration change plus a re-render of the
   families, not new physics.

2. *A route-aligned clearance profile.* The bridge between pixels and the quantity that decides
   outcomes: for each point along the remaining route, the free height and the free left/right
   width. This is the privileged representation the smoke-stage selector already consumes, computed
   from geometry; the deployable version computes it from depth. Having both is what turns
   "does perception work" into a measurable gap rather than a yes/no.

3. *The observability gate, enforced.* A clip that crouches from frame 0 supports only
   map-conditioned selection, because the obstacle was not yet visible when the behaviour began.
   The gate exists; it must become a release criterion, not a diagnostic.

## 4. The closed-loop experiment, once the dataset is large enough

Three ascending versions, each answerable with the apparatus that exists:

**Open-loop selection (claims 5 and 6).** The selector sees the scene, picks a candidate, and the
outcome is looked up from the verified family. No new physics. This is the experiment scoped for
6 September and it is the paper's core.

**Closed-loop execution.** The selector picks, SONIC executes in the actual scene, and the verdict
comes from physics rather than a table. Costs one rollout per decision instead of a lookup, and
answers a stricter question: does the choice survive when the robot actually walks it.

**Scene-first generalisation.** The same, on the 30 frozen scenes that were sampled before any
operator existed and were never fitted to a trajectory. This is the only cell that speaks to
deployment, and it is deliberately sealed.

## 5. What could make this fail, stated in advance

**The selector may not need perception.** If the candidate bank is small and the regimes are
visually distinct, a model may separate them on scene identity rather than geometry. The controls
are built for exactly this — no-scene, scene-shuffle, nominal-motion holdout — and they must be
reported whatever they show.

**Depth may not carry the deciding millimetres.** The windows that separate behaviours are 20-100 mm
at 2-3 m range. If ego depth at that range cannot resolve them, the privileged-versus-deployable gap
will be large, and the honest result is to measure and report that gap rather than to tune around it.

**Scale may not arrive.** 24-30 verified families over 8+ motions is the bar for treating the
learning comparison as evidence. At 3 verified families the correct move is to say so, not to run
the comparison anyway and present a result the sample cannot support.

## 6. What this does *not* commit to

Floor-level adaptation, photorealism, manipulation and an end-to-end VLA are all out of scope until
claims 5 and 6 are answered. They are each a plausible extension and each would consume the time the
decisive experiment needs.
