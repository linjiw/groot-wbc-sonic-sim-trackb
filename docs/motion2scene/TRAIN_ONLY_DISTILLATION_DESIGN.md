# Next learning experiment: amortize geometric search using training motions only

Design, 2026-09-05. No teacher acquisition, training, new source selection or inference
result is launched by this document. The [fresh-source protocol](FRESH_SOURCE_V1.md)
is already registered and must finish under its own rules. This proposal does not
change it. The [main framework](LEARNED_GENERATOR_FRAMEWORK.md) contains the original
LfH/LfLH reading and the broader humanoid formulation.

## The problem this experiment would address

A learned initializer followed by search is useful only if its benefit justifies its
training, acquisition and inference cost. Our present assisted methods still spend
4,624 clearance queries per eight-output job. The concrete learning objective is to
move more of that work into a reusable conditional distribution, then retain an
independent acceptance check. Scaling parameters or motion count alone does not
establish that we have amortized search.

Proposed contribution: train a conditional scene proposal on sets of geometrically
refined placements derived from TRAINING motion pairs, while retaining both direct
clearance objectives and diversity measurements. Test whether that proposal achieves
the frozen hybrid baseline's acceptance with fewer inference queries on separately
registered new sources. This is an untested contribution statement, not a result.

## What the network learns from motion alone

Let M be the target motion, A(M) its registered alternative set, z proposal noise and
S a parameterized scene. Our current instance is a beam described by route station
and bottom height. The target motion conditions q_theta(S | M); the alternatives
supply training-time geometric feedback and verification, as in the existing pipeline.
Inference that omits alternatives cannot claim the interference relation was checked.

A teacher T searches on training pairs using the frozen proxy, margins and uncertainty
set. It returns a SET of retained placements plus rejected placements, their clearance
margins and provenance. These are synthetic supervisory targets constructed from
motion geometry, without manually labeled scene placements. They are not observations
of the natural scene distribution and should never be labeled as such.

For each training source and event, preserve several accepted placements across station
and height bins, their pre-search proposals and failure types. Fix teacher candidate
and query budgets before acquisition, including learned, uniform and stratified starts.
Do not keep searching until an arbitrary acceptance quota is filled. Empty teacher
sets remain visible in the funnel and direct geometric training continues on them.
Stratification is an explicit proposal prior, not evidence of natural scene frequency.

A candidate objective is

    L = lambda_set * mean_M mean_(S in teacher_set(M)) [-log q_theta(S | M)]
        + lambda_geom * existing_target_and_alternative_margin_loss
        + lambda_pref * existing_motion_preference_loss.

Use the existing differentiable density only when its normalization and bounded-domain
handling support the stated likelihood. Otherwise use an explicitly specified sample
matching loss; do not report an unnormalized score as a likelihood. Set weights, teacher
selection, normalization and optimizer schedules using training and development data
before a new transfer registration. Retain an objective-only baseline with the same
architecture and a training-compute-matched baseline. No single best placement target:
that would teach teacher concentration without testing conditional support.

## Source roles and comparisons

The old TRAINING pool is 41001–41004 and 42001–42004. Every teacher derivative stays
with its source. The 43001–43008 acquisition and every derivative remain excluded
from teacher targets, normalization, training and selection permanently, whether or
not its acquisition gates pass. Old validation/test sources are development evidence;
they cannot become a new confirmatory test through renaming.

Use old development data for implementation and budget choices. For a final transfer
comparison, register another source pool BEFORE generation and freeze all candidate
models and query schedules. Do not repeatedly choose variants by checking the 430xx
pool. A later descriptive comparison there must be labeled reuse of an observed audit,
not an independent confirmation.

Compare, at minimum:

| Comparison | What it isolates |
|---|---|
| Original raw vs distilled raw | Amortized proposal gain at zero search queries |
| Original pattern vs distilled pattern, same query budget | Initialization gain with identical correction |
| Distilled proposals with reduced search budgets | Acceptance versus online query cost |
| Uniform pattern with matched queries | Whether learned initialization is necessary |
| Original objective with equal total training queries | Extra supervision versus extra compute |

Use fixed output quotas; do not compare a rejected-only accepted subset against an
unfiltered baseline. Count refusals against the requested-output denominator. Report
source-level results, both event phases, all fitting seeds, target and alternative
failures, paired changes where initial proposals correspond, accepted-bin diversity,
and nearest-proposal sensitivity to moving the input event. Accepted-bin count alone
is not feasible-volume coverage, and a higher acceptance rate can hide mode collapse.

## Make scaling and amortization measurable

Initially hold source count and architecture fixed. Compare teacher target count and
teacher-query budgets before increasing model size. Then vary training sources using
multiple predeclared nested subset orders, with equal-total-query and equal-visits
contracts reported separately. A larger teacher set from one source is more placement
coverage, not more independent motion data.

Charge teacher search, source generation, normalization/feature construction, fitting,
checkpoint setup, proposal generation, online correction and final verification.
For an online deployment assumption of N requested jobs, compare

    C_total(N) = C_acquisition + C_teacher + C_fit
                 + N * (C_proposal + C_online_search + C_verification).

Report the break-even N only when the measured online saving is positive and under
an explicitly stated reuse model. A cheaper model forward pass is insufficient if
verification dominates. Keep geometry query counts alongside local wall time; more
parallelism or a faster backend does not demonstrate a better learning algorithm.

## Extend to complex 3D only after the collision contract is explicit

The input should retain time, whole-body geometry, contact state and local route frames;
root trajectories alone cannot distinguish arm/torso/leg interactions. A later scene
output may be an unordered set of oriented primitives with existence probabilities,
followed by object geometry. This is a separate representation experiment. Start from
one additional family with suitable registered motion alternatives before jointly
learning arbitrary scenes or adding many objects.

For a union of obstacles, target noncollision requires clearance from EVERY object.
An alternative may be blocked by ANY object, but the full alternative set and the
intended contact policy must be evaluated jointly. Independent object sampling cannot
establish scene feasibility: support, obstacle–obstacle overlap, access and intended
robot contacts can impose additional constraints. Distinguish collision avoidance
from legitimate foot/support or hand/contact interactions.

A learned distribution or a low collision loss cannot guarantee noncollision. The
acceptance layer must state which body geometry, continuous-time interpolation,
placement uncertainty and numerical error it bounds. The present checker evaluates
proxy capsules at recorded frames and finite perturbations. Extend the existing
conditional placement certificate to imported-body containment and conservative
between-frame bounds before asserting a stronger guarantee. Tracking uncertainty
needs its own bound or obstacle-present execution evidence; it cannot be inferred
from reference clearance alone.

A useful guarantee requires two different geometric directions. For target clearance,
certify that the true imported collision body lies inside an outer approximation;
clearance of the outer approximation then implies clearance of that body. For
alternative interference, an outer approximation can give false collisions. Use a
certified inner approximation or direct imported-geometry intersection witnesses.
A single approximate capsule representation does not automatically satisfy both roles.

For time, define the reference interpolation first. If every point on link i moves
at speed at most V_i within an interval of length delta_t, the distance to a fixed
obstacle changes by at most V_i * delta_t / 2 from its nearest endpoint. Require
endpoint clearance minus that bound and numerical allowance to exceed the margin,
or subdivide the interval. Obtain V_i from bounded root/joint motion and the kinematic
chain, not from an empirical maximum over sampled frames. High DOF increases the
bound's conservatism and motivates adaptive subdivision, not omission of arm/leg links.
Placement-cell displacement adds another conservative allowance. A failed certificate
means unresolved or infeasible under that contract; it is not automatically a collision.
These are proposed sufficient conditions, not bounds established by the current audit.

The next research decision should therefore use two separate outcomes: can learning
reduce geometric query cost on new motions, and can the checker support a stronger
scene–motion contract? Neither outcome substitutes for the other.
