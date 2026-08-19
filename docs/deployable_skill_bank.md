# Two artifacts from one pair: the causal reference and the deployable skill

A matched counterfactual pair and a motion a robot can be sent are not the same object, and the
project has been asking one clip to be both. The overhead band is where that stopped working.

## The conflict, stated plainly

`local_adaptation` holds root XY, yaw, duration, frame count and gait phase fixed between the
nominal and the adapted clip. That is not fussiness — it is the entire reason the family is
attributable. Two clips that differ only by the adaptation cannot differ *because* they were
different journeys, which is the objection every mined pair is open to.

The price is that the adapted clip is internally inconsistent. It asks for a crouch and for
undiminished forward progress at the same time, and a crouched G1 cannot walk at an upright pace.
On `n_013_ceiling_overhead_left` the adapted reference cleared its ceiling at 49 N against the
nominal's 1543.6 N strike, and was rejected anyway — `reference_endpoint_tracking_error`, 0.524 m
against a 0.35 m budget, where the nominal it was bent from already spent 0.217 m. Since a ceiling
can only be relieved by lowering the robot, this bounds the whole overhead band rather than one
family.

Two responses were available and both were consequential: change the pair, or change the gate.
Neither is right, because they are answers to two different questions that were being asked of one
clip:

* *Why must the behaviour change here?* — answered by holding everything except the adaptation.
* *How does the robot reliably execute the changed behaviour?* — answered by a clip that is
  physically coherent, whatever it costs in duration.

So the project keeps both, and never confuses them.

## What each one is for

| | causal reference | deployable skill |
|---|---|---|
| built by | `local_adaptation.local_crouch` / `local_arm_tuck` | `deployable_retiming.deployable_clip` |
| root path | identical to the nominal, frame for frame | the same line through the room, resampled along it |
| duration | identical to the nominal | longer, by what the slowdown costs |
| gait phase | aligned with the nominal | continuous, stretched inside the adaptation |
| start and goal | identical | identical |
| joints | minimum edit | the same edit, at the same route position |
| used for | the 2×2, attribution, the paper's claims | `motion_lib`, SONIC fine-tuning, sim2sim, hardware |
| what it proves | that geometry reverses the preferred behaviour | that the robot can hold the preferred behaviour |

The deployable clip is **derived from** the causal one and never replaces it. Nothing reported on
the 2×2 comes from a retimed clip, and the pre-registration is untouched, because the pair it
describes is not the artifact being changed.

## How the pace is chosen

Measured, not modelled. If the executed adapted clip arrived `d` metres short of where its own
nominal arrived, and the adaptation was active over an alpha-weighted arclength `W`, the stretch is
commanded at `1 − safety · d / W` of nominal pace. That is the same delivery correction
`local_adaptation.delivery_corrected_target` applies to the geometric target, applied to the
transport one.

Nothing infers walking speed from knee angle. Four attempts to predict trackability from a clip
rather than measure it are recorded as wrong in [the register](prediction_register.md), and the
prediction this correction makes is registered there as P9 rather than assumed.

The slowdown is a **time warp**: root pose and joint angles are resampled against one warped clock,
so the legs still swing the distance the root travels. Slowing the root alone would reintroduce
exactly the inconsistency the retiming exists to remove, with the reference feet skating.

## What the bank needs, and what it has

The minimum set a closed loop can be built on:

| skill | state |
|---|---|
| nominal walk | **exists**, verified, many clips |
| local arm tuck | **exists** as a causal reference; tracks at 88–105% of the nominal, so it needs no retiming |
| speed-consistent local crouch | **now buildable**; one clip retimed, none rolled out yet |
| stop / abstain | **missing** — no clip, and the selector has nothing to choose when nothing is feasible |
| phase-aligned walk → adapt → walk transition | **partly**: the retiming ramps pace with the adaptation inside one clip, but nothing yet composes two clips at a chosen gait phase |

The four beyond the minimum — graded crouch depths, left/right tuck, sidestep, step-over — are all
reachable with the existing operators and none of them is the blocker.

**The transition is the one that cannot be skipped.** A real robot never enters a crouch from frame
zero of a clip. Every closed-loop episode has to decide when to begin adapting, from which gait
phase, when to recover, and how to switch between two obstacles without a discontinuity in the
commanded action. Retiming makes each clip internally coherent; it does not yet make two clips
joinable.

## What the retiming has been checked against, on CPU

Before any rollout, on `n_013_ceiling_overhead_left`:

| check | result |
|---|---|
| `screen_reference` on the retimed clip | no notes: reachable, no self-intersection, no saturation |
| peak silhouette under the shelf footprint | 1.2091 m against the causal clip's 1.2093 m — **0.18 mm** |
| pointwise silhouette across the footprint | at most 1.65 mm from the causal clip |
| route deviation | 0.00 mm; start and goal unmoved |
| pace and duration | window at 0.751 of nominal, 120 → 131 frames |

The first two rows matter more than they look. The family plan predicted `nominal_reach_m` 1.2996
and `adapted_reach_m` 1.2093; measuring the clips directly returns 1.2995 and 1.2093, so the
silhouette computed here is the same quantity the placement used, and the retimed clip presents the
shelf the same body at the same place. The geometry the counterfactual is about survives.

What does change is exposure in time: the deployable clip spends 69 frames below the hard shelf face
against the causal clip's 58, at the same clearance. That is the open half of the registered risk.

## What is not claimed

That a retimed clip tracks. One clip has been built and none has been rolled out. The prediction
that it will, and the sharper prediction about *how far* the endpoint error falls, are registered as
P9 before the measurement, along with the named risk that a slower crouch spends longer under the
shelf and may buy tracking at the price of clearance.
