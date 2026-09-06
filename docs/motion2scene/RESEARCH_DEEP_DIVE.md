# Motion2Scene: what we are learning, what the experiments establish, and what comes next

Research status, 2026-09-06. This is an evidence-backed account of a developing method.
The public report separates reference geometry experiments from recorded controller
executions. No obstacle-present learned-generator or downstream-policy result is admitted.

## The central question

A humanoid crouches while walking. Could a model learn where to place a beam so that
the crouch clears it and a matched upright walk does not? We want a distribution of
useful constraints, not one obstacle coordinate copied from a labeled scene.

The same crouch could occur in an empty room, beneath a shelf, or as an exercise.
Motion therefore does not uniquely identify the original scene or the frequency of
obstacles in the real world. Our geometric assumptions, alternative motions and scene
prior define the inverse problem. Learning must be judged against that contract.

The most promising tested architecture is **local motion features → learned proposal
→ bounded geometric correction → independent acceptance or refusal**. The next
learning question is whether a network can absorb more of the correction work, so
that better proposals require fewer geometry queries at inference.

## What LfH and LfLH contribute

LfH uses motion collected in open space to construct imaginary obstacle settings and
train navigation behavior. LfLH replaces handcrafted hallucination with a learned
obstacle distribution. Its encoder produces scene parameters, and a fixed planner
decoder supplies a reconstruction signal from the motion; obstacle priors and collision
regularization also matter. [Original LfH](https://arxiv.org/html/2007.14479v4),
[original LfLH, §III](https://arxiv.org/html/2108.09793v1).

Our proposed humanoid adaptation keeps fixed geometric feedback but changes the
question to whole-body feasibility and preference among declared motion alternatives.
This is a research adaptation, not a reproduction of the authors' planner or evidence
that mobile-robot results transfer automatically to a humanoid. The
[full framework](LEARNED_GENERATOR_FRAMEWORK.md) records the source reading and equations.

## The implemented input and output

The current Unitree G1 references contain 29 joint coordinates plus a floating root,
recorded as 36-column position/quaternion/joint arrays. Forward kinematics supplies a
29-capsule approximation attached to 13 links. The encoder retains 64 route-progress
samples, each with 205 features: capsule midpoint/axis/radius descriptors plus progress
and elapsed-time fraction. Horizontal geometry is relative to the root; vertical
geometry remains available. The independent checker uses all recorded frames, not
just the encoder's resampled frames.

A temporal model with local route anchors produces an eight-component distribution.
Its output is currently **beam route station and underside height**. Beam width,
depth, thickness and orientation convention are fixed. This model does not yet learn
arbitrary object count, mesh shape, scene semantics or joint multi-object layouts.

The supervised target in the original inverse objective is the observed motion's
selection under a fixed evaluator. A separate geometric loss enforces both sides of
the relation: the target clears the obstacle; its cheaper neutral alternative is
blocked. No human supplies a correct obstacle placement. Nevertheless, the experiments
use explicitly constructed target/neutral pairs; arbitrary unpaired mocap would need
a justified way to obtain alternatives and intended-contact labels.

## Why two constraints are needed

Let c(M,S,u) be the minimum capsule–box clearance over a complete sampled motion M
at placement perturbation u. Positive means separation in this proxy; negative means
overlap of its capsule model. Current acceptance requires, for every tested u:

    c(target, scene, u) ≥ +0.010 m
    c(neutral, scene, u) ≤ −0.010 m

The first inequality prevents obstacles that hit the crouch. The second rejects an
irrelevant beam that the neutral also clears. A preference loss alone can miss this
second margin; our explicit-margin experiment exposed that mismatch. The negative
proxy value is not a measurement of physical penetration depth.

Training, search and final checking have different jobs. The original training loss
samples geometry. Search evaluates a fixed candidate budget at 17 perturbations.
The final audit checks every requested output at 113 perturbations using an independent
NumPy implementation, with Torch agreement checks. A sampled distribution may produce
failures even when its loss is low. Refusal is part of the output contract.

## The sequence of evidence

| Experiment | Result | What it changes in the research direction |
|---|---|---|
| Globally pooled conditioning | Grouped model produced zero jointly valid outputs on its two excluded test carriers | Preserve the location of the event in the representation |
| Local event features | 110/144 and 133/144 on two test sources; pooled control 2/144 and 4/144 | Local timing/location information matters; input-swap controls are necessary |
| Training-source scaling | At equal queries, 4 to 8 sources raises unseen-phase yield from 95/192 to 119/192; one source regresses | Data composition matters; scale is not a uniform remedy |
| Bounded correction | 189/192 versus 123/192 raw on its development panel | Search can turn useful but imperfect proposals into a strong hybrid |
| Station search | Three alternatives reach 192/192 on fresh draws over observed sources | Cheaper search is plausible, but this is not a fresh-source test |
| Frozen fresh-source audit | Learned pattern 384/384 versus uniform pattern 30/384 across eight newly generated sources | Learned initialization helps this bounded search on new sources within the beam/crouch setting |
| Fresh-source gradient comparison | Gradient also reaches 384/384; the predicted strict pattern gain fails | Equal acceptance does not establish a unique winning search method |
| Search distillation | See the separately registered development result below | Test whether the network can reduce inference queries, counting its teacher cost |

The new distillation pilot improves raw acceptance **292/384→334/384**, with gains
on all eight development sources. The stronger geometry-training control reaches
310/384, but hybrid loses three outputs against it on source 41007. Reduced search
reaches **377/384 versus 384/384**, and mean accepted bins decrease **2.125→2.021**.
Thus the raw-proposal prediction passes, while the budget-control source criterion,
reduced-search criterion and bin-diversity criterion fail. The next useful ablation
is teacher witness quality within a fixed coverage and query budget, not a claim that
the present network has already replaced geometric search.

The datasets, random seeds and denominators differ between rows; do not treat this
as one controlled learning curve or pool the counts. Detailed evidence:
[conditioning](CARRIER_LEARNING_V1_RESULT.md), [event model](EVENT_SCALING_V1_RESULT.md),
[source scaling](SOURCE_PHASE_V1_RESULT.md), [refinement](REFINEMENT_V1_RESULT.md),
[station search](STATION_SEARCH_V1_RESULT.md), [fresh sources](FRESH_SOURCE_V1_RESULT.md),
[distillation](DISTILLATION_V1_RESULT.md).

## The new learning experiment

The [distillation protocol](DISTILLATION_V1.md) freezes a common teacher, all source
roles, budgets, final checkpoints and predictions before spending on teacher search.
It uses the 24 ORIGINAL TRAINING derivatives only. All 43001–43008 fresh-audit sources
and descendants remain excluded from training, normalization and selection.

The teacher starts from a fixed mixture of learned, uniform and stratified proposals,
corrects them, independently checks them and retains several distinct accepted bins.
It preserves failures and empty sets instead of searching until a success quota is
filled. Students compare geometry-only continuation, set-only imitation and a hybrid.
The set loss matches weighted samples to the teacher distribution; it is not an
implemented likelihood for the decoder. A longer geometry-only control receives at
least the hybrid student's teacher-plus-training query budget.

The inference comparison measures raw proposals and fixed correction budgets. The
five-evaluation method actually makes five evaluations; it does not run seventeen
and hide twelve. Reducing 4,624 to 1,360 search queries is a 70.6% online search-query
reduction, but a useful claim also requires preserved acceptance and accounting for
teacher generation, fitting, sampling and verification. Results on reused development
sources cannot serve as another fresh-source confirmation.

## How the collision claim can become stronger

The current 113-placement, sampled-frame audit is finite. More samples can find more
failures; they cannot by themselves certify an entire continuous domain.

For target clearance, an outer approximation must contain the imported collision
body. If that outer body clears, the contained body clears too. Proving interference
requires the opposite direction: a valid inner-body witness or direct imported-geometry
intersection. An outer capsule may overlap while the actual body does not. The same
approximate proxy does not automatically justify both claims.

For time, first define the interpolation between reference frames. If every point on
link i has a proved speed bound V_i within an interval of duration Δt, the nearest
endpoint's clearance minus V_i Δt/2 is a conservative temporal lower bound. Subtract
placement uncertainty and numerical allowances as well. Obtain the speed bound from
kinematics and bounded root/joint motion; an observed maximum at sampled frames is
insufficient. Adaptive subdivision can reduce excessive conservatism.

The [existing placement certificate](PLACEMENT_CERTIFICATE_V1_RESULT.md) covers two
selected scenes over a continuous placement domain under its stated static-proxy
and numerical assumptions. It does not yet cover imported bodies or motion between
frames. Tracking errors require their own uncertainty model and obstacle-present
execution evidence. A certificate that runs out of budget returns unresolved, not
an invented safety verdict.

## How to grow beyond beams

Preserve full-body time, local frames and contact state when extending the encoder.
Start with one additional obstacle family and suitable qualified alternatives before
jointly learning a set of oriented primitives or meshes. The present crouch operator
does not establish side-step, step-over or support-contact competence.

For a scene containing several objects, the target must clear every forbidden-contact
object throughout the motion. An alternative may be blocked by any object, but scene
validity also includes object overlap, access, support and legitimate robot contacts.
Per-object acceptance does not certify a joint scene. Scene completion must be checked
along with the critical obstacle, rather than treated as harmless decoration.

Scale should add independent sources and new tested axes, not just more edits of the
same motion. Compare multiple nested source subsets under both equal-query and
equal-visits budgets. Study teacher coverage and representation before increasing
network size indiscriminately. Keep a new source pool untouched until the full next
comparison is registered.

The next publishable contribution must connect a mechanism to measured benefit:
less online geometric work at preserved source-level acceptance, a stronger explicit
collision contract, or downstream policy improvement in qualified obstacle-present
scenes. Each requires its own evidence. The [next research plan](NEXT_RESEARCH_PLAN.md)
and [distillation design](TRAIN_ONLY_DISTILLATION_DESIGN.md) keep those decisions separate.
