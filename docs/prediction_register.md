# Predictions recorded before the result

Written before the measurement, so the record cannot be rewritten afterwards. Four of the
predictions below were wrong, which is the reason this file exists: without it, a refuted guess
quietly becomes a "finding we always suspected".

## Open

### P9: retiming the adapted clip pays back the transport cost, and pays back the right amount

**Registered 2026-08-19, before the retimed clip has been rolled out.**

[The crouch cannot pay its own transport cost](#the-crouch-cannot-pay-its-own-transport-cost) left
two responses open and took neither mid-batch: retime the adapted reference so a crouched robot is
asked for a crouched pace, or judge adapted clips on progress ratio rather than endpoint error. The
first is now taken, and taken in a way that does not touch the thing the gate is measured on: the
causal reference is unchanged and remains what the 2x2 is reported from, and a *second* artifact --
the deployable skill -- is written beside it by `gear_sonic/dataset_generation/deployable_retiming.py`.
The gate is not moved. The pre-registration is not amended, because the matched pair it describes is
not the clip being changed.

The correction is arithmetic on a measurement, not a model. `n_013_ceiling_overhead_left`'s adapted
cell missed its endpoint by 0.524 m where its own nominal missed by 0.217 m; the adaptation was
active over 1.478 m of alpha-weighted arclength; so the window is commanded at
`1 - 1.2 * 0.307 / 1.478 = 0.751` of nominal pace and the clip runs 120 -> 131 frames. Route, start,
goal, joint angles at each route position and the shelf's station in route progress are all held.

1. **The adapted cells pass the tracking gate.** `adapted_easy` and `adapted_hard` both record
   endpoint error below 0.35 m. Falsified by either staying above it.
2. **The correction lands where its arithmetic says, not merely on the right side of the gate.**
   `adapted_easy`'s endpoint error falls to within 0.06 m of the nominal's 0.217 m. Falsified by an
   error that clears 0.35 m but sits above 0.28 m -- outcome right, mechanism wrong.
3. **The 2x2 survives the retiming.** `nominal_hard` still strikes the ceiling and `adapted_hard`
   still clears it, so the family verifies as a minimum-edit reversal.

Prediction 2 is the sharp one, and it is the one I expect to be least safe. Predictions 1 and 3
could both come out right while the linear reading of the shortfall -- that it accrues in proportion
to how active the adaptation is, and that commanding that stretch slower returns it one for one --
is wrong; a slowdown that overshoots would satisfy both and teach nothing.

**The named risk, registered rather than discovered later.** A slower crouch spends *more* frames
under the shelf than the causal reference did, so the obstacle sees a longer exposure. If
`adapted_hard` now fails on contact where it previously cleared at 49 N, the retiming has bought
tracking at the price of clearance, and the honest reading is that the deployable clip needs a
deeper crouch as well as a slower one -- not that retiming failed. `delivered_window_m` is expected
to change for the same reason and is not evidence either way.

**What refutation would mean.** If 1 fails, transport is not what rejected the clip and the
diagnosis below is wrong. If 1 holds and 2 fails, retiming is a usable engineering fix whose
magnitude cannot be predicted, which puts every future deployable clip on a sweep rather than a
single measured correction.

**Amendment, 2026-08-19, same day, before any rollout: the geometric half of the risk is closed and
the prediction is not amended.** The risk above has two halves — that the retimed clip presents a
different silhouette to the shelf, and that a longer exposure lets the controller drift out from
under it. The first is now measured on CPU and does not occur. Within the shelf's actual footprint
(0.5 m of a 5.296 m route, so 0.094 of route progress centred at 0.553) the peak silhouette is
1.2995 m nominal, 1.2093 m causal adapted, 1.2091 m deployable — the first two reproducing the
family plan's own `nominal_reach_m` and `adapted_reach_m` to four decimals, and the retiming moving
the peak by **0.18 mm**, 1.65 mm pointwise. `screen_reference` returns no notes on the retimed clip.

What remains of the risk is therefore only the second half: the deployable clip spends 69 frames
below the hard face against the causal clip's 58 (2.30 s against 1.93 s), at the same clearance. If
`adapted_hard` now fails on contact, it is drift over a longer crouch, not a shallower one, and that
distinction is worth more than the verdict.

### P10 — BONES-SEED is much cleaner than the AMASS bank, and the screen will mostly find nothing

**Registered 2026-08-19, before the dynamic-feasibility screen has been run over BONES-SEED and
before any arm of `plan_feasibility_hygiene_v1.md` has been trained.**

The sibling project measured 22.8% of a 10,705-clip AMASS-derived bank as dynamically infeasible
for more than 10% of frames. BONES-SEED reaches SONIC through a different retargeter and through a
filtering step that is named in the directory itself (`robot_filtered`, from
`filter_and_copy_bones_data.py`).

Prediction: **under 10% of `bones_seed_official_headline_scale4950` will exceed
`infeasible_frac > 0.10`**, and the flagged clips will concentrate in a few source subsets rather
than spreading evenly — the sibling measurement ranged 0.1% to 100% per source.

If the rate comes in at or above 10%, the "already filtered" premise is wrong and the release
filter is passing references that no controller can track. That is the more interesting outcome and
the one that justifies the rest of the pipeline. A low rate means arms B and C are near-identical
to A, and the ablation should be descoped rather than run at full cost.

### P11 — the per-motion cap is a cheap substitute for data hygiene, and mostly wins

**Registered 2026-08-19, before any arm has run. This predicts against work I have already built,
which is the reason to write it down now.**

`max_prob_per_motion` already exists (`motion_lib_base.py:2461-2462`) and is `None` in every shipped
yaml, so the constraint block early-returns. Setting it to 5x fair share brings a maximally-failing
clip from 43x its fair share down to 5.2x, measured by driving the real code path over the real
BONES-SEED length inventory (1,006 motions, 7,863 bins). That costs
nothing: no screen, no repair, no bank rebuild.

Prediction: **arm E (raw bank + per-motion cap) recovers at least half of arm C's worst-decile
survival gain over arm A**, at zero data-pipeline cost.

If it does, the honest recommendation to this codebase is a one-line config change, and the screen
and the repair operator are worth keeping only for what a cap cannot do. The falsifier that would
save the data pipeline is specific: **C well above E on the ground-contact/kneel/crawl stratum**,
because capping exposure to a bad clip limits waste but does not recover a pose the bank never
contained in trackable form. If C and E are equal there too, the data-side work has not earned its
cost and I should say so in those words.

### P12 — repair beats pruning exactly where the discarded clips were rare

**Registered 2026-08-19, before any arm has run.**

Pruning removes flagged clips; repair replaces them, keeping N, the clip names, the durations and
the fps identical.

Prediction: **C exceeds B on the ground-contact/kneel/crawl stratum by more than C and B differ on
the easy stratum**, and the easy-stratum difference stays within +/-0.02.

If C is indistinguishable from B everywhere, repair does nothing that deletion does not, and the
operator is unjustified complexity — the 65.8% recovery rate the sibling project measured would
then be a statement about file counts rather than about anything the policy learns.


## P6, refuted — and it exposed a worse error

**P6 — the arm tuck's collisions are the wrist reaching the floor (registered 2026-08-18, before
measuring).** The tuck fails by `disallowed_robot_contact` at *low* drift, on `pelvis` (x000) and
`right_wrist_yaw_link` (x002). These are bare-plane rollouts, so the only surface available is the
ground: a wrist registering external contact means the arm came down far enough to touch it.

Prediction: the tuck lowers the wrist's minimum ground clearance, and the two rejected motions lose
more of it than the accepted one (x003).

*Why this one is worth another attempt after P4 failed.* P4 asked a **self**-clearance question
(wrist against hip) about a failure the gate scores as **external**, which was the wrong currency
from the start. This asks a ground-clearance question about a ground collision. It is also the
right *kind* of predictor by the distinction the crouch work established: the tuck fails by
collision, and collisions are governed by geometry, which is exactly what a swept volume can see.

If refuted, then geometry does not predict the tuck's failures either, and per-clip rollout
screening is confirmed as the only method for the lateral regime — which is a usable answer, just an
expensive one.

**Refuted, and on a premise that was false.** The wrist's minimum ground clearance is 0.62–0.67 m in
every clip and the tuck changes it by at most 0.4 mm, in the wrong direction. But the reasoning
rested on "these are bare-plane rollouts, so the only surface is the floor" — and the screening scene
`cf_005_056_easy` in fact holds a shelf at 1.3906 m and four walls.

Chasing that down was worth more than the prediction. `x001`'s nominal, which I had excluded as
untrackable, was rejected for a force of exactly (0.0, 277.6, 0.0) N with its root 0.21 m from a wall
— it **walked into the wall**, a room-sizing artifact, not a trackability failure. The tuck's yield
of 1 of 3 valid nominals therefore rests on a denominator that may be 4.

The lesson is not about prediction at all: **I asserted a property of the apparatus — "bare plane" —
without checking it, and then reasoned from it repeatedly across documents and commit messages.**
The registered prediction is what forced the check.

## P5, confirmed — after I got the intermediate reading wrong twice

The registered claim was that knee excursion governs crouch trackability. It does. Getting there
took two wrong intermediate readings, both recorded here because the sequence is the lesson.

**Reading 1 (too strong).** Four clips across three motions gave a monotone drift-versus-excursion
table with the boundary between 0.994 and 1.000 rad, and I reported trackability as predictable from
the clip.

**Reading 2 (too weak).** Rebuilding the failing clip at 0.980 rad left it rejected with drift 0.223
against 0.224 — twenty mrad changed nothing — so I withdrew the causal claim and said excursion
correlates across motions without controlling within one.

**Reading 3, from a full within-motion sweep.** Excursion *does* control it, monotonically, inside a
single motion:

| excursion | window | outcome | drift |
|---|---|---|---|
| 0.000 rad | — | accepted | 0.048 m/s |
| 0.420 rad | 24.0 mm | accepted | 0.104 m/s |
| 0.619 rad | 43.1 mm | **accepted** | 0.140 m/s |
| 0.980 rad | 92.3 mm | rejected | 0.223 m/s |
| 1.000 rad | 95.6 mm | rejected | 0.224 m/s |

The threshold is real, **motion-specific** — x000's acceptance boundary lies in (0.619, 0.980) where
motion 005 holds at 0.994 — and the drift curve **saturates** near the top: 0.223 against 0.224 for
the last 20 mrad.

That saturation is the whole explanation for reading 2. I sampled two points on the flat part of the
curve, saw no difference, and concluded the curve did not exist. The lesson is narrower than "test
within as well as between": it is that a null result from a single small step near a plateau says
nothing about the relationship, and a sweep costs three rollouts where a wrong retraction costs a
published claim.

Incidentally the gate's own drift threshold is bracketed by the same sweep: accepted at 0.140 m/s,
rejected at 0.223.

### What this means operationally, which is the useful part

x000 is not a failure of the operator. It is caught in a **two-sided squeeze**, and the two sides
are measured in different currencies:

| excursion | predicted window | trackable? |
|---|---|---|
| 0.619 rad | 43.1 mm | **yes** |
| 0.980 rad | 92.3 mm | no |
| 1.000 rad | 95.6 mm | no |

The strengths that buy a usable window are untrackable, and the strength that tracks buys a window
too narrow to trust — 43 mm predicted is 10 to 35 mm real, given that measured windows run 23–82% of
predicted, against 82–98 mm for the three families that work.

**So the crouch's yield is set by whether a motion's trackability threshold sits above the excursion
its window needs.** That is a screen worth one rollout per motion rather than a bisection: compute
the excursion a 60 mm window requires (CPU only), then test trackability at exactly that excursion.
Motions where the two do not overlap are dropped before any family is attempted.

### The original registration, unedited

**P5 — crouch trackability is set by knee excursion (registered 2026-08-18, 2 of 3 cells pending).**
Motion 005's crouch at 0.994 rad is accepted; x000's at 1.000 rad — sitting exactly on the cap — is
rejected for pure reference drift with zero contact, at 0.225 m/s against the accepted crouch's
0.100. If excursion is the governing variable, then **x002 (0.929 rad) and x003 (0.936 rad) should
both be accepted**, and the crouch's boundary sits just under 1.00 rad.

*Confidence: moderate, and deliberately stated because the analogous prediction failed for the arm
tuck.* Excursion did **not** predict tuck trackability there. What differs is the failure mode: the
tuck's rejections are collisions at low drift (0.051 and 0.003 m/s), while the crouch's is a
tracking failure with no contact at all — and a tracking failure is the kind of thing excursion
plausibly governs, where a collision is governed by geometry. If x002 or x003 is rejected, excursion
is not the variable for the crouch either, and per-clip rollout screening is the only method for
both operators.

## Parked under the timebox rule

**mf_x002_c08 — parked at 10 rollouts (2026-08-19).** The rule is five rollouts or half a day on one
motion; x002 has taken ten across three attempts, and the third attempt succeeded only in narrowing
the question rather than answering it.

| shelf | nominal | crouch |
|---|---|---|
| 1.3478 m (easy) | accepted | accepted |
| 1.3000 m | accepted | accepted |
| 1.2567 m | rejected | rejected |

So both boundaries lie inside a 43 mm band, and a family needs the shelf between them. Two more
probes would bisect it. That is cheap and it is still the wrong call: the same two rollouts spent on
a fresh screened nominal buy a whole configuration rather than one contested family, and the point
of the timebox is that this trade is invisible while you are inside the rabbit hole.

The first four rollouts were void for the coordinate-frame bug, so only six carried information.
That does not change the decision — the rule counts rollouts spent, not rollouts that worked, because
otherwise every failure buys an extension.

**Not a finding about x002.** Its window may well be real and merely narrow. If the family count
falls short at the end of Workstream A, this is the cheapest place to return to, and the band is
recorded here so the return costs two rollouts rather than ten.

## The operator delivers about a third of its predicted window

*Found 2026-08-19 on the first wall family; eight probes, then stopped under the timebox.*

`n_013_wall_chest_left` came within one cell of holding: both easy cells accepted, `nominal_hard`
struck the aperture at 110.3 N. The adapted cell struck it too, at 91.4 N — a 17% softer collision,
not a clearance.

The planner sized that scene from a predicted window. Measured against what the executed body
actually did, the operator under-delivers by a factor of nearly three:

| quantity | chest band, `n_013` |
|---|---|
| predicted nominal reach | 0.2632 m |
| predicted adapted reach | 0.2394 m |
| **predicted window** | **23.8 mm** |
| executed nominal surface | 0.2551 m |
| executed adapted surface | 0.2460 m |
| **delivered window** | **9.1 mm** |

Two details keep this from being a placement bug. The nominal exceeds the planned face by 3.8 mm and
duly strikes, so the hard scene is doing its job. And the contact force is (−86.6, 0.0, −29.1) N —
purely frontal — so these walls are apertures the robot passes through and the shoulder catches the
jamb, not side panels it brushes. The binding surface is the upper-arm capsule, which `shoulder_yaw`
and `elbow` share; the tuck retracts it 9.1 mm while retracting the wrist 14.7 mm, because the
operator acts distally and the jamb is caught proximally.

Predicted windows differ by band by more than a factor of four, and the batch was gated on predicted
width at `min_window_m = 0.02`:

| band | configs | predicted window | at 38% delivery |
|---|---|---|---|
| overhead | 3 | 90–105 mm (median 92) | ~35 mm |
| waist | 6 | 51–110 mm (median 58) | ~22 mm |
| chest | 5 | 21–52 mm (median 22) | **~8 mm** |

**A 20 mm gate on predicted width admits configurations with 8 mm of real window.** That is below
the run-to-run variation of the executed body, so a scene placed inside it is not reliably placeable
at all — which is what the chest family shows.

### It is the minimum-edit rule, not the excursion cap

The obvious suspect for an under-powered tuck is `MAX_TUCK_EXCURSION_RAD = 0.40`. It is not the
cause. The chest configuration commands **0.0635 rad**, an order of magnitude below the cap, so the
cap never binds.

The cause is the objective. `m*(S) = argmin_m D(m, m₀)` asks for the smallest edit that clears the
obstacle *as predicted*, and the prediction over-states what reaches the binding surface by about
2.6×. A minimum computed against an optimistic model is itself optimistic by the same factor, so the
rule reliably specifies an edit too small to work — and does so most severely exactly where the
window is narrowest and the margin for error least.

Joint-space survival and clearance delivery are not the same ratio: the chest tuck retains 64% of
its commanded joint amplitude while delivering 38% of its predicted window. Both numbers are
consistent with the operator acting distally on a constraint that binds proximally, and the gap
between them is the part a joint-space cap could never fix.

## Retraction: a day of family conclusions came from the wrong verdict function

*Found 2026-08-19, reconciling the pitch page against the scorer.*

`score_family_batch.py` called `evaluate_locomotion_trajectory` directly. That function runs every
gate. Whether a *reference* gate should bind is not its decision — `gate_policy.py` makes that call
per episode, and these rooms were built around the executed corridor, so under `SCENE_AROUND_MOTION`
the reference is a diagnostic and not the label. `classify_episode` applies the policy; the scorer
bypassed it.

The correction runs in both directions, which is how it was caught:

| family | scored (raw) | correct (policy) |
|---|---|---|
| `duck_002` | not verified | **verified** |
| `duck_003` | not verified | **verified** |
| `n_013_ceiling_overhead_left` | verified | **not verified** |

So the project has **2 verified families, not 0 and not 1** — the number the release index reported
all along, and the reason the pitch page shows `duck_003` clearing its obstacle with three cells
accepted. The pitch was right and the scorer was wrong.

### What this retracts

**"The crouch cannot pay its own transport cost", as a claim about gating.** Endpoint error never
gated these episodes; the policy demotes it. The register entry, the paper subsection and the
preflight check built on it all assumed a gate that was not binding.

**P8 in its entirety.** It argued that the path gate charges a crouch for lag. Under the policy
these episodes ran with, the path gate is demoted too, so it was not charging them anything.
P8.1 was falsified on its own terms as well — 11 cells flipped against a predicted 4–8, because I
counted family cells and forgot the sweep corpora.

**The three-miss-mode taxonomy**, which named a structural transport cost as one of three. Two modes
survive: an under-calibrated edit, and a nominal that fails its own easy scene.

### What survives, and why it is worth separating

The *measurements* are untouched, because none of them asked a gate anything:

* Endpoint lag rises with crouch depth, 0.257 → 0.509 m. Still true, still a real property of the
  controller, and still the right thing to report as a quality column. It is simply not a rejection.
* Operator survival by body region, 46–67% against 88–105%.
* Delivery ratio by band, 30–70%.
* The room-centring defect and its 418 mm.

### The real blocker for the banded ceiling family

Under the correct policy `n_013_ceiling_overhead_left` fails on **`unstable_reference_drift`** in
all three non-nominal cells — a gate nothing today had looked at. That is the thing to investigate
next for the overhead band, and it is not what any of today's analysis was about.

### Why this went unnoticed for a day

Every number was internally consistent, so nothing looked wrong. The scorer, the preflight and the
paper all agreed with each other because they all shared the same wrong assumption, and the release
index — which used `classify_episode` and reported two verified families — was the one artefact
disagreeing. I read that disagreement as the release being stale rather than the scorer being
wrong, and only checked when a published page contradicted a fresh score.

**The rule this earns:** one verdict function, reached one way. `classify_episode` is that function.
Anything calling `evaluate_locomotion_trajectory` for a pass/fail is asking a question it does not
have the standing to answer.

## P8: the path gate is charging a crouch for being behind, not for leaving the route

*Registered 2026-08-19, after measuring cross-track error and before computing which cells flip.*

`path_error_p95` computes ‖executed(t) − reference(t)‖: the distance between two positions at the
same timestamp. That is one number covering two different failures — leaving the route, and being
behind on it — and a crouched robot commits only the second. Measured as distance to the reference
*polyline* instead, the two separate cleanly:

| clip | same-timestamp p95 | cross-track p95 | share that is phase |
|---|---|---|---|
| `w_nominal` | 0.222 m | 0.210 m | 5% |
| `w_crouch05` | 0.284 m | 0.182 m | 36% |
| `w_crouch08` | 0.379 m | 0.211 m | 44% |
| `w_crouch11` | 0.487 m | 0.228 m | 53% |
| `n_013` adapted_hard | 0.507 m | 0.241 m | 53% |

The nominal is 5% phase and every crouch is 36–53%, rising monotonically with depth. **Every
cross-track value sits under the existing 0.25 m threshold.** The crouches never leave the route.

This is the four-label argument the paper already makes, applied to a metric instead of a corpus:
one number covering two questions answers neither. Route adherence, schedule adherence and goal
attainment are three separate facts about a traversal, and a counterfactual family's claim concerns
obstacle clearance, which happens mid-route.

**Registered before computing the consequence**, because a metric change that rescues previously
rejected cells is exactly the change that must not be adopted because it helped:

1. **Between 4 and 8 of the currently-rejected cells flip to accepted** when path error is measured
   cross-track and endpoint error is reported rather than gated. Falsified outside that range.
2. **No cell that currently passes flips to rejected.** Cross-track error is bounded above by
   same-timestamp error, so this should be arithmetically impossible; if any cell flips, the
   implementation is wrong rather than the idea.
3. **At least one overhead family becomes verifiable**, since the ceiling cells fail on tracking
   alone while clearing their obstacle at 49 N. Falsified if none does.

**What is not proposed.** The contact gates do not move. Progress ratio does not move. Upright and
height checks do not move. Endpoint error stops being a rejection reason and becomes a reported
quality attribute, because a clip that follows its route and clears its obstacle while finishing
0.5 m short has traversed — it has simply traversed slowly, and the release should say so in a
column rather than by deletion.

**The alternative considered and not taken.** Retiming the reference so a crouched robot is asked
for a crouched pace would also close the gap, and it is the more physically honest fix. It is
rejected here because it changes the journey that the counterfactual holds fixed: nominal and
adapted would then differ in timing as well as posture, and the single-cause attribution that the
whole dataset rests on would be gone. Retiming stays available as a future operator variant where
both cells are retimed identically, which preserves the contrast; it is not a fix for this gate.

## The room is sized from the route and centred on the origin

*Found 2026-08-19. The batch was stopped on this.*

`n_064` fails its easy scene identically in two different scenes — 183.5 N lateral, 0.355 m lag, the
same rejection set, in a ceiling scene and a wall scene. Identical numbers across different
obstacles mean the contact is with neither. It is with the room.

`build_graded_scene.py` computes `room = (span_x + 4.0, max(span_y + 4.0, 5.0))` and passes it as
`room_size_xy` — **a size with no centre**. The room is therefore built about the origin, while a
route has no obligation to be centred there. `n_064`'s route runs x = −2.83 … −2.00, so a 4.83 m
room spans −2.42 … +2.42 and the robot begins 0.42 m outside its own room. The measured overrun is
418 mm and the peak force is +183.5 N in pure +x, pushing it back in.

Screening all eight screened nominals against this criterion costs nothing and settles the batch:

| nominal | forward travel | origin-centred room fit |
|---|---|---|
| `n_001` | 1.87 m | yes |
| `n_004` | 2.12 m | yes |
| `n_013` | 4.21 m | yes |
| `n_014` | 6.22 m | yes |
| `n_126` | 3.50 m | yes |
| `n_064` | 1.80 m | **no, by 906 mm** |
| `n_065` | 0.82 m | **no, by 414 mm** |
| `n_122` | 3.98 m | **no, by 1981 mm** |

`n_064` and `n_065` were the only nominals left in the queue, and both fail. Every remaining cell —
ten configurations, forty rollouts — would have been void: not failed, void, because a baseline that
cannot survive its easy scene makes its whole family uninterpretable. **The batch loop was stopped
after 21 of 56 cells**, with the rollout already in flight left to finish.

Two things this is not. It is not a screening threshold that needs loosening: the screen asks
whether a clip tracks, which `n_064` does. And it is not a property of short clips — `n_001` travels
1.87 m against `n_064`'s 1.80 m and fits, because it happens to start near the origin. The criterion
is *where* the route sits, not how far it goes, which is why no amount of looking at the clips
predicts it and one line of arithmetic does.

The fix is to centre the room on the route rather than the origin, which is a change to the scene
builder alone. Five nominals pass and a corrected batch can use them, so the pilot is not blocked —
it is delayed by the rollouts already spent.

## A screened nominal can still fail its own easy scene

*Found 2026-08-19 on the first `n_064` family.*

`n_064` passed the nominal screens. Its first family is nonetheless void: the nominal fails the
**easy** scene, which is supposed to be the cell that always works.

| cell | verdict | external | overhead | lag |
|---|---|---|---|---|
| `n_013` nominal_easy | accepted | 0.0 N | 0.0 N | 0.217 m |
| `n_064` nominal_easy | **rejected** | 183.5 N | 0.0 N | 0.355 m |
| `n_064` nominal_hard | rejected | 113.6 N | 43.1 N | 0.477 m |

The contact is **lateral, with no overhead component at all**, in a scene whose only intended
obstacle is a ceiling — and it carries `disallowed_foot_non_ground_contact`, a foot touching
something that is not the floor. Neither is possible from the ceiling the scene was built to place.
The route is meeting geometry that is not the obstacle, most likely the room's own walls, which
`build_graded_scene.py` sizes as the nominal's span plus 4 m.

**This is a screening gap, not a placement gap.** The nominal screen asks whether a clip tracks;
it does not ask whether the clip fits the room that will be built around it. `n_013` clears its easy
scene at exactly 0.0 N and `n_064` does not, and nothing in the screen distinguishes them.

Its consequence for this batch is large. If the four remaining `n_064` configurations share the
fault, twenty rollouts produce void families — not failed ones, void, because a baseline that cannot
survive the easy scene makes every other cell in its family uninterpretable. `n_065` is untested and
may or may not follow.

The scorer now names this case first for exactly that reason. It previously reported this family as
"adapted motion still struck the obstacle", which is true and useless: it describes a symptom three
cells downstream of a baseline that never worked.

## The recalibrated tuck is already known to track

*Checked 2026-08-19 against clips already run, no new rollouts.*

The wall repair asks for 0.091 rad at chest and 0.211 rad at waist, against the 0.064 rad the
minimum-edit rule currently commands. Whether that is safe is answerable from the sweep, which
already spans an order of magnitude of tuck amplitude:

| clip | commanded | endpoint lag | tracking |
|---|---|---|---|
| `w_tuck06win18` | 0.072 rad | 0.257 m | accepted outright |
| `w_tuckcap30` | 0.282 rad | 0.218 m | passes |
| `w_tuckcap40` | 0.314 rad | 0.195 m | passes |
| `w_tuck14win18` | 0.741 rad | 0.112 m | passes |
| `w_tuck10win30` | 0.869 rad | 0.084 m | passes |

**No tuck fails tracking at any amplitude tested**, and endpoint lag *falls* monotonically as the
tuck grows — 0.257 m down to 0.084 m, against a 0.35 m threshold. The recalibrated targets sit in
the middle of a range already demonstrated to track, and they land on the better side of it.

This is the exact opposite of the crouch, where lag rises with depth and the gate binds at about
5 cm. The two operators differ in sign on every axis measured so far: survival, transport cost, and
now trackability headroom. An adaptation that moves the arms is nearly free to the controller; one
that moves the legs is charged for.

The clips above are rejected, but for `disallowed_robot_contact` rather than tracking — they were
run in a scene with obstacles, so contact is expected and says nothing about whether the amplitude
is executable. What is *not* established is whether a larger tuck creates self-contact by pressing
the arm into the torso; the sweep cannot separate that from obstacle contact, and the recalibrated
batch would need to.

## The reference-side planning is not what misaligns

*Screened 2026-08-19 across all fourteen configurations, no GPU.*

The obstacle station is a constant: `build_graded_scene.py --station` defaults to **0.55**, the same
fraction of the route for every configuration, while each configuration's adaptation window is its
own. That looked like the cause of the misalignment found on `n_013_wall_waist_right`, and it is
cheap to check on the reference clips alone.

It is not the cause. Every one of the fourteen configurations places 0.55 inside its own window:

| nominal | window (route fraction) |
|---|---|
| `n_013` | 0.43–0.69, and 0.48–0.63 for the ceiling |
| `n_064` | 0.39–0.70, and 0.46–0.61 for the ceiling |
| `n_065` | 0.40–0.68, and 0.46–0.62 for the ceiling |

**0 of 14 configurations place the obstacle outside their own adaptation window.** So the claim in
the previous entry — that this is the old "shelf placed where the duck had not yet begun" failure —
is wrong at the level it was stated. On the reference clips the two coincide everywhere.

What remains true is the measurement: at the place its obstacle binds, `n_013_wall_waist_right` has
its adapted arm 24.9 mm further out than the nominal's, and the same tuck retracts 10.6 mm over the
episode. The discrepancy is therefore between the reference window and where the *executed* body
actually is when it meets the solid — a gap the reference-side screen cannot see, and one this
entry does not explain. Parked under the timebox at eight probes on one family, with the screen
recorded so the reference-side explanation does not get proposed again.

## Negative delivery is a diagnostic, not noise

*Found 2026-08-19 on the fourth family, after two measurement bugs of my own.*

`n_013_wall_waist_right` reports **−58% delivery**: at the place its obstacle binds, the adapted
arm sits 24.9 mm *further out* than the nominal's. Two false explanations were checked and
discarded first, and both were mine:

* **Not an operator sign error.** Over the whole episode the right tuck retracts the right wrist by
  10.6 mm, against the left tuck's 18.9 mm. Both pull inward.
* **Not the measurement's sign.** Lateral extent does need the obstacle's side — measuring +y on a
  right-side configuration measures the *left* arm, which that tuck never touches — but fixing it
  made the number more negative, not less.

The cause is alignment. The left family's obstacle binds at x = −0.27 m, inside the tuck's active
window, and the arm is 11.2 mm retracted there. The right family's binds at x = +0.40 m, after the
window has closed, where the arm has already swung back out. The operator did its job in the wrong
place.

The external forces say the same thing from the other side: the left family's nominal strikes at
**608.3 N**, the right family's at **31.2 N**. One is an obstacle in the route; the other is a graze
the robot half-misses on its way past.

This is the failure `build_family_report.py` already names — *"a shelf placed where the duck had not
yet begun"* — recurring in the banded planner, which sizes windows from reach without checking that
the reach and the obstacle occur at the same point along the route. Both bands measure the right
quantity at the wrong place.

**So a negative delivery ratio is worth reporting rather than clipping.** It means the adaptation and
the obstacle do not coincide, which no amount of scaling the edit will fix — and which the
repairability arithmetic would silently mis-answer, since dividing by a negative ratio yields a
negative required command.

### Superseded by the automated measurement

The hand figures below (30%, 59%, 25%) were point samples. `score_family_batch.py --plans` now
measures the same quantity reproducibly — on the binding body, over the frames where each run is
within 10 cm of the binding position — and reports **43%, 70% and 30%**. The automated numbers are
authoritative because they are what will run over all fourteen families; the hand ones are kept
because the reasoning that produced them is what identified the metric in the first place.

Aligning by position rather than frame index is the part that matters. Matched frames compare a
blocked nominal at x = 1.32 m against an adapted run already at x = 1.92 m — two different places,
one of which has no obstacle in it — and that error alone reported the overhead crouch at −21%.

The repairability conclusions are unchanged under the new numbers: chest needs 0.091 rad and waist
0.211 rad against a 0.40 rad cap, and the ceiling needs 1.419 rad against 0.98 — still over cap,
now by 1.4× rather than 2.1×. Both conclusions survive the revision, which is the only reason they
were worth stating before it.

### Three bands measured: delivery is 25–59%, and the metric has to be local

With three families complete, delivered window measured on the binding body the plan names:

| family | binding body | predicted | delivered | ratio |
|---|---|---|---|---|
| `n_013_ceiling_overhead_left` | `torso_link` | 90.2 mm | 27.5 mm | 30% |
| `n_013_wall_chest_left` | `left_elbow_link` | 23.8 mm | 14.0 mm | 59% |
| `n_013_wall_waist_left` | `left_wrist_yaw_link` | 51.4 mm | 12.9 mm | 25% |

P7.3 predicted a third within a factor of two, meaning 17–67%. All three fall inside it, though on
three families that is weak support rather than confirmation.

**The overhead row needed correcting twice, and the corrections are the lesson.** Measured as the
tallest surface anywhere on the robot, delivery came out at −1.4 mm — the adapted body appearing
*taller* than the nominal. Measured on `torso_link` alone but still as a maximum over the whole
episode, it stayed at −1.4 mm. Only when measured at the place the obstacle sits does the crouch
appear at all: the torso is 27.5 mm lower at x = 0.60 m and 22.0 mm lower at x = 0.90 m, and 10 mm
*higher* by x = 1.10 m, because the adaptation window has closed by then.

A local edit cannot be measured by a global extreme. The maximum over an episode is dominated by
whatever the robot does furthest from the obstacle, which is precisely the part the operator did not
touch. Every window figure quoted from here on is measured on the named binding body at the
obstacle's own location.

The waist family failed the same way as the chest family — its adapted motion struck the aperture —
despite a predicted window more than twice as wide. That is the outcome P7.2 did not expect, and it
weakens the reading that narrow windows alone explain the chest failure.

### The shortfall was already known, and the gate is the wrong lever

`MIN_WINDOW_M` carries its own comment: *"the realised window ran 23-82% of the predicted one."* The
delivery shortfall was measured before this batch was planned, and the gate was set to 20 mm anyway.
The three measurements here — 25%, 30%, 59% — fall inside that range and confirm it rather than
discover it. What is new is the consequence, which the earlier note did not draw.

That makes the fix I suggested under P7 the wrong one. Raising the gate so predicted × worst-case
delivery clears 20 mm would require about 87 mm of predicted window, and of the fourteen planned
configurations only the three overhead ones clear that — the same three the transport cost already
disqualifies. A gate strict enough to be honest selects nothing this batch can use.

The other lever is the edit itself. The minimum-edit rule asks for exactly what an optimistic model
says will clear the obstacle; asking instead for that divided by the delivery ratio makes the
*delivered* edit the minimum, which is what the rule was always meant to mean. The required commands
are then 0.108 and 0.253 rad for chest and waist, both far inside the 0.40 rad cap, and the gate can
stay where it is because the window it protects is now real.

This is one line of arithmetic in the operator, not a change to the cap, the gate, the journey, or
the acceptance thresholds — which is what makes it the right lever. It is still not applied while
the batch that would test the current behaviour is running.

### The walls are fixable; the ceiling is not

Inverting the delivery ratio gives the edit each configuration would have needed to clear its own
hard scene, which is the number that decides whether the method is repairable or the band must go:

| config | commanded | delivered | needed | required command | operator cap | reachable |
|---|---|---|---|---|---|---|
| ceiling / overhead | 0.614 rad | 27.5 mm | 90.2 mm | **2.013 rad** | 0.98 rad | **no — 2.1× over** |
| wall / chest | 0.064 rad | 14.0 mm | 23.8 mm | 0.108 rad | 0.40 rad | yes |
| wall / waist | 0.064 rad | 12.9 mm | 51.4 mm | 0.253 rad | 0.40 rad | yes |

**The wall families are a calibration error.** Both need edits comfortably inside the tuck's cap —
1.7× and 4× what the minimum-edit rule asked for. Correcting the delivery model would make them
reachable without touching the cap, the gate, or the journey. The tuck also tracks better than the
nominal it modifies, so there is trackability headroom to spend.

**The ceiling families are not repairable by scaling.** Clearing that hard scene needs 2.013 rad of
commanded crouch against a 0.98 rad cap, and the crouch already fails the endpoint gate at 0.614 rad
by consuming 0.524 m of a 0.35 m budget. Both limits bind, and neither is an artefact of
calibration. An overhead family in this batch cannot be made to hold by asking for a deeper crouch.

That leaves the overhead band with two honest options, both scene-side rather than operator-side:
place its hard face at a margin the crouch can actually deliver, or drop the band. Deciding that
needs the remaining `n_064` and `n_065` overhead configurations, which will show whether 2× over cap
is typical or particular to `n_013`.

One caveat holds this whole table together and should not be silently assumed: it treats delivery as
proportional to commanded amplitude. Survival is not quite linear — crouch survival rises from 46%
to 58% as amplitude grows, and small tucks survive better than large ones — so the required commands
above are estimates, not solutions. For the walls the conclusion is robust because the required
edits sit far inside the cap; for the ceiling it is robust because 2.1× over cap does not close
under any plausible curvature.

### P7, registered before the remaining twelve configurations report

1. **No chest-band configuration produces a verified family.** Falsified by any one of the five.
2. **Waist-band configurations verify at a higher rate than chest-band ones.** They have the widest
   windows among wall obstacles and the tuck tracks better than the nominal, so this is where the
   method should work if it works anywhere. Falsified by chest matching or beating waist.
3. **Delivered window stays near a third of predicted across bands**, within a factor of two.
   Falsified by any band delivering more than two thirds or less than a sixth.

The third is the one worth having, because it converts `min_window_m` from a guess into a number:
if delivery holds near a third, the gate must sit near 0.06 m to admit only configurations with a
real 20 mm window. That change is not made here — it is a design parameter, and changing it while
the batch that would test it is still running would leave nothing to test it against.

## P6: the transport cost predicts family yield by band

*Registered 2026-08-19, with 2 of 56 cells seen and 12 of 14 configurations unrolled.*

The batch splits by operator exactly along the band: the three ceiling/overhead configurations are
relieved by `local_crouch`, the eleven wall chest/waist configurations by `local_arm_tuck`. The
transport-cost finding above says the crouch spends more forward progress than the endpoint gate
allows, and the survival measurement says arm departures reach the body nearly intact while leg
departures do not. If both are right, yield should separate cleanly by band. Registered before the
results exist:

1. **No ceiling/overhead configuration produces a verified family.** All three adapted cells fail
   `reference_endpoint_tracking_error`. Falsified by any one of the three verifying.
2. **At least 8 of the 11 wall configurations have adapted cells that pass tracking** — endpoint
   error below 0.35 m in both easy and hard scenes. Falsified at 7 or fewer.
3. **The adapted cell's endpoint error is lower for tuck configurations than for crouch
   configurations**, with no overlap between the two groups. Falsified by any overlap.

Prediction 3 is the sharp one. Predictions 1 and 2 could both come out right for reasons unrelated
to transport cost — a wall is simply an easier obstacle than a ceiling, and that alone would
separate the groups. Only 3 tests the mechanism, because it asks about the specific quantity the
mechanism is about, and it fails if the two groups' endpoint errors interleave even where both
verify.

What would make all three uninformative: if the wall configurations fail for contact rather than
tracking, the tuck's transport advantage is never exercised and the comparison says nothing. That
outcome is recorded as such rather than reinterpreted.

## The crouch cannot pay its own transport cost

*Found 2026-08-19, diagnosing the first completed banded family.*

`n_013_ceiling_overhead_left` produced the counterfactual it was designed for: `nominal_hard` walks
its torso into the ceiling at **1543.6 N** on a pure −z vector, and `adapted_hard` clears the same
ceiling at 49 N. Both adapted cells are nonetheless **rejected**, and not for contact —
`reference_endpoint_tracking_error`, missing the commanded endpoint by 0.524 m against a 0.35 m
threshold.

The cause is in the operator, not the controller. `local_crouch` lowers the reference root by about
0.14 m and leaves the forward schedule untouched: the adapted reference still commands x from −2.00
to +2.44 over the same four seconds. A crouched G1 cannot walk that fast, so it arrives short. The
reference is internally inconsistent — it asks for a crouch and for undiminished progress.

Endpoint lag rises monotonically with crouch depth, which is a five-point sweep and not an anecdote:

| clip | endpoint lag | verdict |
|---|---|---|
| `w_nominal` | 0.257 m | accepted |
| `w_tuckcap30` | 0.218 m | rejected (contact, not tracking) |
| `w_crouch05` | 0.313 m | rejected (path) |
| `w_crouch08` | 0.399 m | rejected (endpoint + path) |
| `w_crouch11` | 0.509 m | rejected (endpoint + path) |

The decisive number is the first row. **The nominal already spends 0.257 m of the 0.35 m budget**,
leaving roughly 90 mm for the adaptation to consume. Every crouch tested deeper than about 5 cm
trips the gate. The arm tuck has the opposite sign — it tracks *better* than the nominal — which is
consistent with [operator survival](operator_survival.md), where arm departures pass through at
88–105% and leg departures at 46–67%.

**This bounds the overhead band structurally.** A ceiling can only be relieved by lowering the
robot, so overhead families need a crouch, and a crouch deep enough to clear a ceiling costs more
progress than the gate allows. Three of the fourteen queued configurations are ceiling/overhead; the
remaining eleven are wall obstacles at chest and waist, which the tuck relieves.

**Not yet a decision to change anything.** Two responses are available and both are consequential:
retime the adapted reference so a crouched robot is asked for a crouched pace, which changes the
journey the family holds fixed; or judge adapted clips on progress ratio rather than endpoint error,
which is a gate change and must go through the pre-registration rather than be adopted because it
helps. Neither is taken mid-batch, and the batch is left running because the nominal cells are
producing exactly the strikes the design predicts.

## Resolved

| # | prediction | outcome |
|---|---|---|
| P1 | The arm tuck will track better than the crouch because it moves less mass | **wrong** — it holds the route (19.6 mm) but loses the effect, and yields 1 of 3 |
| P2 | Lower joint excursion improves retention | **wrong** — the smallest excursion gave the worst retention |
| P3 | Clip duration dominates trackability | **wrong** — refuted alongside P2 |
| P4 | The tuck fails by pressing the wrist into the hip | **wrong** — the gap is negative on every nominal and its change anti-correlates with the verdict; the failures are external contact, not self-contact |

The pattern in P1–P4 is one thing: each was an attempt to *infer* trackability from the clip instead
of measuring it. That is also what forced `family_eligible` out of the Stage 2 report after four
estimators gave four answers. P5 is the same species of claim, which is why it is registered rather
than assumed.
