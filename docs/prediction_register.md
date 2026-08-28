# Predictions recorded before the result

Written before the measurement, so the record cannot be rewritten afterwards. Four of the
predictions below were wrong, which is the reason this file exists: without it, a refuted guess
quietly becomes a "finding we always suspected".

## Open

### P9 — REFUTED. Retiming does not pay back the transport cost, because the cost is a ratio and not a debt

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

**Void attempt, 2026-08-19, recorded rather than discarded.** The first rollout of the retimed clip
produced a complete, clean 2x2 that meant nothing. Both adapted cells ran to completion and were
scored -- `adapted_easy` 0.642 m endpoint with `disallowed_robot_contact`, `adapted_hard` 1.227 m
and a 2896.9 N torso strike straight down at frame 53 -- and read as a decisive refutation of P9.

It was not. The retimed clip had been converted with `--scene-start 0 0 0` while its causal family
was converted at `-2.0 0 0`, so it was placed two metres along the route from the shelf it was built
for and met the ceiling sixty frames early, still upright. That is the same frame-mismatch failure
`scene_route_check` was written for, in the same direction, at the same -2.0 m, made by the person
who had read that docstring the same afternoon.

Two things are worth keeping from it. The footprint check does **not** catch this and was tried
first: the shelf is three metres long across the route, so a two-metre shift along the route still
enters its footprint. What is exact is the conversion frame, and it is recorded on both sides, so
`run_deployable_family.preflight_route` now refuses to spend a rollout when a deployable clip's
`scene_start_xyz` differs from its causal family's. And the failure was invisible at every level a
person would normally look: exit status zero, success markers present, artifacts written, a scorer
that produced a well-formed table. Only the frame number of the strike was wrong, and only against
an expectation of where the crouch was.

The attempt is void, not negative. P9 stands unanswered and its predictions are unchanged.

**Result, 2026-08-19, on the correctly-placed clip.** Predictions 1 and 2 are refuted; 3 partly
holds. Both adapted cells ran clean and the preflight confirmed the route meets the obstacle.

| clip | frames | commanded | delivered | ratio | endpoint |
|---|---:|---:|---:|---:|---:|
| nominal | 199 | 5.288 m | 5.100 m | 96.4% | 0.2171 m |
| crouch, causal | 199 | 5.288 m | 4.859 m | 91.9% | 0.5242 m |
| crouch, retimed | 217 | 5.280 m | 4.807 m | **91.0%** | **0.5647 m** |

*Prediction 1, refuted.* `adapted_easy` and `adapted_hard` both record 0.5647 m against a 0.35 m
gate — **worse** than the 0.5242 m the un-retimed clip achieved.

*Prediction 2, refuted, and more informatively than 1.* The correction was supposed to bring the
endpoint to within 0.06 m of the nominal's 0.217 m. It moved it 40 mm in the wrong direction.

*Prediction 3, partly held.* The 2×2's contact structure survives and improves: `nominal_hard` still
strikes at 1543.6 N, while both adapted cells now peak at **15.2 N**, down from the causal clip's
49.1 N, and `disallowed_robot_contact` no longer appears on either. Retiming did buy clearance. It
is still rejected, now for `unstable_reference_drift` rather than for tracking error — a different
gate, not a softer one.

**Why it failed, which is the part worth keeping.** The clip was given 18 extra frames and delivered
*less* distance: 4.807 m against 4.859 m, at a delivery ratio of 91.0% against 91.9%. More time did
not let the robot catch up, because it was never running out of time.

The correction assumed the shortfall is a **debt**: a fixed number of metres the robot failed to
cover, recoverable by commanding the same distance more slowly. The measurement says it is a
**ratio**: a crouched G1 delivers roughly 91% of whatever arclength it is commanded, and stretching
the clock stretches the loss along with it. `d` is not a constant to divide by a window; it is
proportional to the window.

That makes P9 the same species of error as P1–P4, and it is worth naming as such rather than filed
as an unlucky guess. Each of those inferred trackability from a clip instead of measuring it. This
one inferred a *transport model* from a single endpoint number when the delivery ratio was sitting
in the same trajectory pickle and could have been measured first. The lesson is not "retiming does
not work" — it is that the quantity to measure before correcting anything is the ratio, not the
residual.

**What follows.** A correction that respects a proportional loss has to command **less distance**,
not less speed — which moves the goal, which is exactly what the matched pair holds fixed. So the
deployable skill and the causal reference diverge more than
[the two-artifact split](deployable_skill_bank.md) assumed: they cannot share a goal. That is a
design consequence, not a tuning parameter, and it is not taken here.

The 15.2 N result also stands on its own and should not be lost in the refutation: whatever retiming
does to the endpoint, it made the crouch's *clearance* strictly better.

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

### P10 — BONES-SEED is much cleaner than the AMASS bank, and the screen will mostly find nothing — **CONFIRMED, and it descopes the ablation**

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

**Result, 2026-08-19: confirmed, by a wide margin.** Full-bank screen of all 4,950 clips
(`gear_sonic/research/hygiene/screen.py`, 131.7 s wall on 8 CPU workers, 0 failures):

| threshold | `infeasible_frac >` | `airborne_frac >` |
|---|---|---|
| 0.05 | 29 (0.59 %) | 225 (4.55 %) |
| **0.10** | **7 (0.14 %)** | 111 (2.24 %) |
| 0.20 | 5 (0.10 %) | 32 (0.65 %) |

**0.14 %, against the 22.8 % measured on the AMASS/whole_body_tracking bank — a factor of 160.**
The prediction said under 10 %; the truth is under one part in 700.

The concentration half of the prediction also holds, but the *mechanism* is not what I guessed.
The seven clips are `jump_on_50cm_002` (0.658), `kick_back_001` (0.472), `jump_off_front_50cm_R_002`
(0.379), `jump_off_50cm_R_001` (0.366), `jump_off_front_50cm_001` (0.353), `high_jump_R_003`
(0.138), `burpee_002` (0.136). Five of seven are box jumps. Their references are not corrupt —
they are unsupportable **on a flat floor**, because the 50 cm box they jump onto is not in the
scene. That is a scene-mismatch, not a retargeting artifact, and root-projection repair is the
wrong operator for it: the fix is terrain or exclusion.

**Consequence, pre-committed above and now taken: the SONIC feasibility-hygiene training ablation
is descoped.** Arms B (prune) and C (repair) would differ from A by seven clips out of 4,950. No
training arm can resolve that, and running the matrix would burn GPU to measure noise. I am not
looking for a rescue analysis.

**What the screen did earn on this bank**, and it is worth keeping:
- It is *right about the hard case*. Seven `kneeling_loop_*` clips sit at `airborne_frac = 1.000`
  with `infeasible_frac = 0.000` — feet 7–9 cm off the floor for the entire clip, weight carried
  on the knees. A naive "airborne means broken" filter deletes exactly the rare ground-contact
  behaviour the bank is short of. Keeping airborne and infeasible as separate axes is what makes
  that distinction, and this is the evidence that it matters.
- It is cheap enough to be a standing release gate: 0.145 CPU-s per clip, 0.84 ms per screened
  frame. Running it on every new corpus costs minutes and would have caught the box jumps before
  they reached training.
- The negative result is itself the finding: **the official `robot_filtered` pipeline is doing its
  job.** That is worth stating plainly, because the AMASS number invites the assumption that all
  retargeted banks are 20 % broken. They are not.

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


## The room-sizing fix landed, and the scenes it invalidates are still on disk

*Measured 2026-08-19, after [`7d7f88b`](#), and it re-derives that fix rather than finding it.*

**The diagnosis is not new and is not claimed as new.** `7d7f88b` — *size the room to contain its
route, not a route of that size* — already establishes it, on the same nominal, with sharper
numbers than these: n_064 began 418 mm beyond the wall, 183.5 N of lateral contact, a foot against a
vertical surface, in a scene whose only intended obstacle was a ceiling. This section exists for
three things that measurement adds on top.

**First, the fixed builder has not been run.** Every `gf_*` scene on disk carries an mtime of
**12:37**; the fix was committed at **14:31**. The builder is repaired and the assets are not, so
the void scenes are still the ones any re-roll will load.

**Second, the scope is wider than n_064.** Measuring each obstacle box against its own walls' inner
faces, across all 48 counterfactual scenes:

| nominal | scenes | obstacle inside its room |
|---|---|---|
| `gf_013_*` | 8 | **100%** — all eight |
| `gf_064_*` | 10 | 27–57% |
| `gf_065_*` | 10 | 40%, and **0%** for the four `*_right` scenes |
| `mf_x002_c08_*` | 2 | 73% |
| `mf_x003_c08_*` | 2 | 98% |

`gf_065` is affected as heavily as `gf_064`, and the four `gf_065_*_right` scenes place their
obstacle at y ∈ [−4.47, −4.23] against a room spanning y ∈ [−3.97, 3.97] — **entirely outside**.
Those cells cannot test anything and did not. So 10 of the 14 queued configurations are void, and
any yield computed over 14 has a denominator of 4. `n_013`'s four configurations stand, and with
them [P9](#p9-retiming-the-adapted-clip-pays-back-the-transport-cost-and-pays-back-the-right-amount)
and the transport diagnosis, both of which rest on `n_013_ceiling_overhead_left` alone.

**Third, it was reached from the other end, which is the reusable part.** Not by reading the
builder. The per-frame constraint record (`constraint_distance.py`) was cross-checked against the
recorded physics verdicts and agreed on 16 of 20 cells; every disagreement was a cell where physics
recorded a contact the shelf could not account for. Chasing those four produced the same conclusion
the builder fix reached independently. A record reporting only the intended obstacle's clearance
would have agreed with itself and found nothing. That two independent routes — reading the geometry
code, and asking a second question of the executed episodes — land on the same defect is the useful
fact here, because only the second one is available when the defect is in data rather than code.

**Remedy, not yet taken.** Regenerate the `gf_*` scenes with the post-`7d7f88b` builder and re-roll.
Not a re-grade: the old cells are discarded as unevaluable, not re-scored.

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

## LFH autonomous experiments — Codex-owned

These entries are filed under `docs/hallucination/GOVERNANCE.md`. They are additive: all
user-owned entries above remain unchanged. Registered 2026-08-20T15:22:06-04:00, before any LFH
Phase-2 physics outputs.

### LFH-E1a — source repeatability

Manifest: `E1A_REPEATABILITY_PROPOSED.json`, SHA-256
`cdb718992587554cb1628160dc0a8050bc1b768a5c0646b89c8cebe74d52cbac` (16 serial cells,
1.667 contended GPU-hour ceiling).

For each of two new seeds, both `duck_003` and `mf_005_c08` will reproduce the source pattern:
`nominal_easy=accepted`, `nominal_hard=rejected`, `adapted_easy=accepted`, and
`adapted_hard=accepted`. A clearance noise floor is a measurement, with no numeric prediction.
Any outcome flip falsifies this entry and triggers the governance stop line before E1b or probes.

### LFH-E1b-V2 — shelf-plank golden physics

Manifest: `E1B_DUCK003_PHYSICS_PROPOSED_V2.json`, SHA-256
`7809e36bec81f4f0b9b3211725300e7fc8c6d1fe3357716a23b8ada9baf48856` (4 serial cells,
0.417 contended GPU-hour ceiling), contingent on LFH-E1a completing without a flip.

The physics prediction is the same 2x2 outcome pattern as LFH-E1a, and the hard/nominal strike is
uniquely attributable to the authored shelf binding primitive; ambiguous or secondary contact
refuses the golden. The KCS capsule instrument separately predicts binding-geometry clearances
`easy(orig/edit)=+89.577/+202.302 mm` and `hard(orig/edit)=-68.000/+66.663 mm`. Those millimetre
values are reported calibration measurements, not outcome gates: the historical boundary-bisection
instrument recorded hard/orig near `-0.0 mm`, so cross-instrument disagreement is not a refutation.

### LFH-PROBES-1 — minimal plane trackability

Manifest: `MINIMAL_PLANE_PROBES_PROPOSED_V2.json`, SHA-256
`0cb332765361d57e934a698d4954b4c571d20eb79b972bbee3c998b248d0c0c6` (at most 6 serial cells,
0.625 contended GPU-hour ceiling).

`mf_005_c08__probe_nominal` is predicted accepted because historical nonempty-scene executions
tracked the same motion. Its crouch-adapted pair is measurement without prediction: an endpoint-gate
failure would be a scientific transport-cost finding, not infrastructure. Both left/right arm-tuck
pairs are also measurement without prediction because no accepted empty-room execution exists.
Each adapted probe runs only if its nominal is accepted; completed rejections are recorded rather
than retried or replaced by nonempty-scene evidence.

### LFH-E1a reconciliation — 2026-08-20

**Confirmed: 16/16 outcomes, 0 flips.** Both fresh seeds reproduced the registered 2x2 pattern
for both source families; all captures were evaluable at 50 Hz. The executed-capsule-to-loaded-USDA
instrument measured an **18.044 mm source-inclusive clearance noise floor** (11.348 mm between
the two fresh repeats alone). See `docs/hallucination/REPORT_E1A.md` and
`docs/hallucination/e1a_repeatability.json`. Actual serial spend was 0.158 contended GPU-hours.
This discharges the E1b contingency; it does not adjudicate LFH-E1b-V2 or LFH-PROBES-1.

### LFH-E1b-V2 reconciliation — 2026-08-20

**Confirmed on the binding physics gates: 4/4 outcomes and unique contact identity.** The generated
shelf produced accepted/rejected/accepted/accepted in nominal-easy/nominal-hard/adapted-easy/
adapted-hard order. Hard/nominal contacted the authored binding primitive at frame 107 on
`torso_link`, uniquely separated from the next authored cube by 2006.6 mm, before reference drift
at frame 123. No other cell had external collision contact.

The KCS diagnostic prediction `[+89.577, +202.302, -68.000, +66.663] mm` compared with executed
`[+89.253, +216.788, -0.107, +80.286] mm`; the three non-penetrating-cell errors are within the
18.044 mm E1a floor, while hard/orig exhibits the preregistered penetration-instrument mismatch.
Against the shipped execution itself, both adapted cells lie outside the E1a envelope by at most
6.625 mm. This is reported—not converted into an outcome gate—and becomes an E2
context-invariance warning. See `docs/hallucination/REPORT_E1B.md`. Actual spend: 0.039 contended
GPU-hours.

### LFH-PROBES-1 manifest amendment — before spend, 2026-08-20

The executable record is superseded by `MINIMAL_PLANE_PROBES_PROPOSED_V3.json`, SHA-256
`10dcd17454b913c0eb6be73ace580db21efe2a4d2b904129320b77a6364e0ed7`. V2 is retired before
any probe rollout because its arm-tuck motions shared a candidate-level provenance file that did
not independently record `scene_start_xyz`. V3 supplies hash-pinned per-motion conversion sidecars.
All six motion bytes, cell IDs, dependencies, predictions, abstentions, and the 0.625 contended
GPU-hour ceiling are unchanged. The LFH-PROBES-1 prediction above therefore remains in force.

### LFH-PROBES-1 infrastructure reconciliation and V4 amendment — before new spend

V3's first bare-plane nominal capture completed successfully (199 frames at 50 Hz), but the
verdict is **void/unevaluable**, not rejected: bare-plane recording provides no pair-resolved foot
ground forces or registered support-floor path, so the physics acceptance gates cannot run. The
capture is retained as non-verdict swept-volume evidence; it is not rerolled or counted against the
prediction. Actual infrastructure spend was 0.010 contended GPU-hours.

The executable replacement is `MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V4.json`, SHA-256
`a028f08ab7ca1ad6b3c81a9be3baa638e0f5f1c98c3afc1720a815be73756bdf`. V4 changes only the
environment and output paths: all cells use hash-pinned `screen_empty`, whose registered floor is
gradeable and whose walls and 5.0 m-high shelf are remote. Motion bytes, dependencies, prediction
for `mf_005_c08` nominal, five abstentions, and the 0.625 GPU-hour ceiling remain unchanged.
Scientific acceptance additionally requires zero external collision contact so remote geometry
cannot masquerade as emptiness. D2-006 records the correction; LFH-PROBES-1 remains in force.

### LFH-PROBES-1 V5 scorer amendment — before spend

V4 is retired without execution because it did not pin which existing acceptance policy defines
*trackability*. `MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V5.json`, SHA-256
`1bae4625fee43821676d49bc78f40152fdee2fa592aff6ddb7942342eb06e560`, binds the raw
`evaluate_locomotion_trajectory` report after reset splitting: reference-path and endpoint gates
remain binding, as required when the commanded motion itself is the label. It also binds all
contact/support/fall gates. The scene, motion hashes, cell order/dependencies, predictions,
abstentions, and spend ceiling are identical to V4. LFH-PROBES-1 remains unchanged.

### LFH-PROBES-1 reconciliation — 2026-08-20

**The sole prediction is confirmed:** `mf_005_c08` nominal accepted in the gradeable empty room.
The unpredicted crouch-adapted cell rejected solely on `reference_endpoint_tracking_error`
(0.396 m against 0.35 m), with zero external collision, confirming a transport-cost failure rather
than infrastructure. Therefore the `mf_005_c08` pair is not repaired or promoted as an LFH source.

Both unpredicted arm-tuck pairs accepted nominal and adapted with zero external collision. They are
trackable but **not scene-eligible**: position-aligned executed windows are 4.32 mm left and
17.41 mm right, below the 20 mm spend floor and far below the 42.00/41.62 mm scalar-prior
predictions. Twenty per-keypoint response rows are filed; D_phi still refuses with only one command
level per side. See `docs/hallucination/REPORT_EMPTY_ROOM_PROBES.md` and D2-007. Actual V5 spend:
0.059 contended GPU-hours (plus 0.010 for the preserved void V3 capture).

### LFH-E2 — one-source overhead variant transfer

Registered before spend against `E2_VARIANT_TRANSFER_PROPOSED.json`, SHA-256
`cba8997fdd0f041e20be8c325e4da661c8e407be035ab91f183ddd914331b2d7` (12 serial
cells, 1.250 contended GPU-hour ceiling). The denominator is honestly one extractable source,
`cf_005_056`; `mf_005_c08` remains refused after its adapted empty-room failure.

The primary prediction is that **at least 2 of 3** CPU-certified variants (`door_lintel`,
`hvac_duct`, `ibeam`) reproduce the complete accepted/rejected/accepted/accepted pattern with
hard/nominal contact uniquely attributed to their binding primitive. Geometry predicts the pattern
for each variant individually; an individual miss is recorded even if the aggregate threshold
passes. `door_lintel` additionally predicts no jamb/secondary contact because its measured context
clearance is 924 mm. Falsifiers: at most one full variant, ambiguous/secondary hard-cell contact,
or a nominal-hard rejection whose tracking drift begins before contact. Clearance drift against
the 18.044 mm E1a floor is a reported diagnostic, with no directional prediction after E1b's
adapted-cell warning.

### LFH-E2 reconciliation — 2026-08-20

**The aggregate prediction is confirmed exactly at threshold: 2/3 variants verified.** All 12/12
cell outcomes reproduced the accepted/rejected/accepted/accepted source pattern. `door_lintel`
and `ibeam` had unique binding-primitive contact before 0.15 m reference drift. `hvac_duct` is an
honest individual refusal: its hard/nominal collision was uniquely attributed to the intended duct,
but drift began at frame 30 before contact at frame 110, activating the preregistered falsifier.

No cell produced secondary contact, and the door-lintel jamb prediction was confirmed. All 12
executed binding clearances stayed within the 18.044 mm E1a floor (largest per-variant maximum:
15.507 mm); authored binding-face offsets round to 0.000000 mm. The immutable approved manifest
SHA-256 is `84d5d6847539343569652fd4db09da7d545d2129158bdb8323963bdaa87232d2`.
Actual serial spend was 0.112 contended GPU-hours. See `docs/hallucination/REPORT_E2.md`.

### LFH-CAL1 — arm-tuck executed-delivery calibration

Registered before spend against `ARM_TUCK_CALIBRATION_PROPOSED.json`, SHA-256
`51202f748bf81d33791383a2c4b8fc962537f765dca88c54e3b5c01b4774b830` (12 serial
empty-room cells, 1.250 contended GPU-hour ceiling). The cohort uses three commanded levels on two
strict-CPU-gated motions per side. Accepted V5 nominal/full-amplitude captures for `089/left` and
`092/right` are hash-pinned and reused; only missing levels and the `086/left`, `084/right` pairs
are rolled out.

The four smaller edits on the already accepted anchor motions are predicted **accepted**, because
they preserve the same root, legs, and support schedule while strictly reducing arm-joint
excursion. The other eight cells are registered as measurements with no outcome prediction: their
references pass the CPU gate, but no empty-room execution exists. The delivery prediction is that
the conservative fitted 60 mm lower bound remains **below the 20 mm scene-spend floor on both
sides**, consistent with V5's 4.32/17.41 mm delivery. Falsifiers are any anchor-level rejection, an
evidence-backed lower bound of at least 20 mm, or failure to obtain the two-motion/three-level
support needed by D_phi. No calibration result is a scene verdict.

### LFH-CAL1 reconciliation — 2026-08-20

**Both registered predictions are confirmed.** All 12/12 new empty-room cells accepted with zero
external collision, including all 4/4 predicted reduced anchor edits. Together with the reused V5
anchors, the cohort contributes 120 position-aligned per-keypoint rows and fits four supported
D_phi models (left/right shoulder and wrist); nonmoving keypoints remain refused.

At a 60 mm wrist command, the monotone two-motion fits estimate 6.82 mm left and 5.16 mm right.
Their conservative lower bounds are only **1.60 mm left and 4.29 mm right**, both far below the
20 mm scene-spend floor. E3 lateral scene instantiation is therefore refused: controller
trackability is green, but the current arm-tuck operator does not reliably deliver enough executed
geometry. Actual serial spend was 0.116 contended GPU-hours. The immutable approved manifest
SHA-256 is `17bd7adf563bdd710a29704540b7577d07835f5d5c6984e685634371eb797eb0`.
See `docs/hallucination/REPORT_ARM_TUCK_CALIBRATION.md`.

### LFH-CAL1 review correction — 2026-08-20

Independent post-run review found that the first D_phi implementation grouped the two motions by
exact per-keypoint `commanded_mm`. Nearly equal matched commands were therefore treated as adjacent
independent x-values; on the right, 60.88 mm -> 5.25 mm and 61.43 mm -> 17.41 mm became an
artificial steep segment with understated cross-motion uncertainty. Raw captures, response rows,
acceptance predictions, and the below-20 mm scene refusal are unchanged.

The corrected contract pools commands and responses by preregistered alpha, requires at least two
motions at every level, and estimates residuals against the matched-level fit. At a 60 mm query the
revised wrist estimates/lower bounds are **6.66/1.10 mm left** and **10.84/4.76 mm right**. Both
remain below the 20 mm spend floor, so the original CAL1 delivery prediction remains confirmed.
This correction supersedes only the four fitted numbers in the reconciliation above.

### LFH-CAL2 — strong-command executed-delivery calibration

Registered before spend against `ARM_TUCK_STRONG_CALIBRATION_PROPOSED.json`, SHA-256
`d3b0dcf8a370c51405b79cdc88205a71a246b9cdc281811af7ca7764a721ef65` (5 serial
empty-room cells, 0.521 contended GPU-hour ceiling). Three accepted nominal captures are reused by
exact reference-CSV identity; `094/left` adds one nominal dependency. The two left motions share
alpha=2.0 (120 mm operator target; 116.50/117.98 mm body windows), and the two right motions share
alpha=2.5 (150 mm target; 95.37/100.15 mm windows). Alpha remains scaled to CAL1's 60 mm anchor,
per D2-010.

All five cell outcomes are registered as measurements with no prediction because CAL1 provides no
trackability extrapolation beyond alpha=1. The delivery prediction is that the corrected
matched-level conservative wrist bound remains **below 20 mm on both sides** after CAL2. Either
side reaching at least 20 mm falsifies that side's refusal and licenses CPU scene proposals (not a
physics verdict); failure to obtain two accepted motions at a strong level refuses that model.

### LFH-CAL2 reconciliation — 2026-08-20

CAL2 completed its registered funnel: 5 proposed, 4 rolled out, 3 accepted, 1 rejected, and
1 dependency skip. Every accepted strong edit had zero external contact. Strong-left is refused:
`094/left` nominal rejected for endpoint tracking and external robot contact, leaving alpha=2 with
only one accepted motion. Strong-right is replicated at alpha=2.5. Its matched wrist fit estimates
52.80 mm delivery at a 261.33 mm mean wrist command, with a conservative lower bound of **23.84
mm**. The registered below-20 prediction is therefore **partially falsified on the right** and not
evaluable on the left.

This result licenses a CPU all-keypoint scene search, not scene instantiation or a physics verdict.
The follow-up finite-face audit distinguishes the earlier root-relative selection proxy from the
authoritative capsule-surface response and records that distinction in D2-011. Actual serial spend
was 0.030 contended GPU-hours. See `REPORT_ARM_TUCK_STRONG_CALIBRATION.md`.

### LFH-E3-LATERAL-1 — coverage-targeted arm-tuck pilot

Registered 2026-08-20T18:07:58-04:00 before generated-scene physics against
`E3_LATERAL_PILOT_PROPOSED.json`, SHA-256
`eab420e271953952eb1aa64c6a7b5f7c522c15a4eb2fde6748ae493360bd28f6` (4 serial cells,
0.417 contended GPU-hour ceiling).

The fixed CPU funnel evaluated 24 full-gap trials and retained one. `084/right` at executed active
progress 0.3925 and 0.30 m face depth has a 165.80 mm raw gap-width window and a 56.41 mm
noise-certified intersection with the empty DCS target `wrist_right × lateral_gap × 0.8_0.9 ×
clear_25_50`. Exact Tier-2 box clearances are `easy(orig/edit)=+35.998/+112.125 mm` and
`hard(orig/edit)=-32.806/+26.419 mm`; `092/right` is refused by the temporal/anatomy screen.

The physics prediction is the complete pattern **accepted/rejected/accepted/accepted** for
nominal-easy/nominal-hard/adapted-easy/adapted-hard. Nominal-hard contact must occur before binding
tracking drift and be uniquely attributable to one of the two authored binding panels. Any outcome
mismatch, secondary or ambiguous contact, or drift preceding the nominal-hard contact falsifies the
pilot. The generated family remains isolated from claim 5 regardless of outcome.

### LFH-E3-LATERAL-1 reconciliation — 2026-08-20

The registered prediction is **falsified**: nominal-easy, nominal-hard, and adapted-easy matched,
but adapted-hard contacted a binding panel and rejected (3/4 outcomes). Its first contact at frame
115 was uniquely attributable to the left binding panel and preceded 0.15 m reference drift at
frame 142, so this is a causal binding failure rather than secondary clutter or post-failure drift.
The candidate is refused, the rank-48 DCS target remains empty, and no generated family is added.

The accepted easy-scene executions close the retry question without more physics. At the registered
station and face extent, nominal and adapted required full gaps are 0.976216 m and 0.906489 m. The
adapted envelope exceeds the `0.8_0.9` target's upper edge by 6.49 gap-mm before uncertainty and
requires 0.942577 m after the registered 36.088 mm gap uncertainty. No in-bin retry is certified;
a wider intervention belongs to another bucket and would require a new prediction. Actual serial
spend was 0.040 contended GPU-hours. See `REPORT_E3_LATERAL_PILOT.md` and D2-015.

### LFH-CAL3 — crouch executed-window calibration

Registered 2026-08-20 before physics against `CROUCH_CALIBRATION_PROPOSED.json`, SHA-256
`a58019a4d3acb4c23a183b6f35f190bd57e22e69d3d8a86d75cd658aafcd9a2e` (8 serial
empty-room cells, 0.833 contended GPU-hour ceiling). The fixed CPU funnel evaluated all 15 strict
stand-to-walk motions at 40/60/80/100 mm target drops (60/60 reference passes) and selected the
four strongest route-straight 80 mm settings: motions `090`, `086`, `089`, and `095`. All four
commands are uncapped and preserve the root path. These are properties of the reference edit, not
executed-delivery or trackability claims.

Every cell outcome is registered as a measurement with no cell-level prediction because prior work
shows crouch trackability is motion-specific. The cohort-level prediction is that at least **2/4**
matched pairs will obtain accepted nominal and adapted executions, and at least one accepted pair
will exhibit a **>=20 mm raw executed head/torso overhead window** in the manifest's fixed
time-mapped station x 0.1/0.2/0.3/0.4 m finite-face search. Failure of either condition refuses new
crouch scene synthesis from this cohort. Passing licenses CPU proposal search only; no geometry or
physics family is implied, and all outputs remain isolated from claim 5.

### LFH-CAL3 reconciliation — 2026-08-20

The cohort prediction is **confirmed**. All four nominals and three adapted motions were accepted;
`095` adapted was a completed scientific rejection for endpoint tracking, giving 3/4 accepted
matched pairs. The fixed 36-trial finite-face search found temporally active exact head/torso raw
windows up to **72.26 mm**, above the registered 20 mm threshold. Actual serial spend was 0.063
contended GPU-hours.

The stronger E3 coverage gate nevertheless refuses every scene. All uncertainty-cleared feasible
coordinates lie in the already occupied `head_torso × overhead × 1.2_1.3` region; no matching
empty target intersects. Empty shoulder targets fail the binding-anatomy contract, and empty
head/torso coordinate buckets are outside the measured windows. Thus CAL3 is retained as executed
operator evidence, while its funnel ends at **0 target-matched scene candidates** with no scene
physics spend. See `REPORT_CROUCH_CALIBRATION.md` and D2-016.

### LFH-CAL3 final-audit correction — 2026-08-20

The preceding zero-candidate reconciliation is superseded, without changing the registered CAL3
physics result. SweepCF-DCS v1 allowed calibration `probe` rows and refused sibling variants to
fill causal targets, and 48/168 configured cells paired `local_crouch` with shoulder binders that
the proposer forbids. D2-017 introduces v2: only canonical cells of exact verified variants occupy
targets, and the coupling-valid denominator is 120. Replaying the same hash-pinned CAL3 executions
restores **20 CPU-valid proposals across all three accepted pairs** in the previously probe-only
`head_torso × overhead × 1.2_1.3 × clear_25_50` target. No family or physics verdict is inferred by
this correction.

### LFH-E6 — independent-source critical-window pilot

Registered 2026-08-20 before E6 physics. E6 tests one canonical `shelf_plank` family for each of
`lfh_086_crouch`, `lfh_089_crouch`, and `lfh_090_crouch`. The fixed selected hard coordinates are
1.257310 m, 1.261340 m, and 1.250508 m. Their symmetric engineering-margin windows are 36.17,
33.54, and 26.01 mm after subtracting the empirical E1a 18.044 mm envelope on both sides. The
envelope is a conservative engineering margin with no calibrated confidence level.

The overall primary prediction is that at least **2/3 sources** reproduce the complete pattern
`nominal-easy=accepted`, `adapted-easy=accepted`, `nominal-hard=rejected`,
`adapted-hard=accepted`; stretch is 3/3. Nominal-hard contact must be uniquely attributable to the
head/torso binding primitive, authored secondary contact is forbidden, and face measure-back must
remain within 0.5 mm. All outputs stay excluded from claim 5.

E6 is staged to enforce D2-015. E6a is the six easy cells in
`E6A_CROUCH_CONTEXT_PROPOSED.json`, SHA-256
`0969a48cb83f00594034f234b68d71169818b1e5526eeb3c4c1fa90a22c4835a` (0.625 contended GPU-hour
ceiling). All six cell outcomes are predicted accepted. The cohort gate is at least **2/3 source
pairs** with both cells accepted, zero external contact, and the pinned hard coordinate retained
inside the scene-conditioned symmetric interval; stretch is 3/3. Fewer than two eligible sources
stops E6. E6b hard cells require a new hash-pinned manifest and reconciliation of E6a before spend.

### LFH-E6a reconciliation and E6b registration — 2026-08-20

E6a completed all six easy cells at 0.091 contended GPU-hours. Five of six predicted outcomes
matched. `lfh_086_crouch` nominal-easy accepted, but adapted-easy rejected for
`unstable_reference_drift` despite zero external contact; that source is refused before hard
physics. Both easy cells accepted with zero external contact for `lfh_089_crouch` and
`lfh_090_crouch`. Exact scene-conditioned reach retained their pinned hard coordinates inside
37.62 mm and 33.45 mm symmetric engineering intervals. Thus the E6a primary gate is met at exactly
2/3 sources and the stretch 3/3 gate is falsified. Evidence:
`E6A_CROUCH_CONTEXT_APPROVED.json` SHA-256
`6b5d23c9dd55aab3a819bfc81bc892151d1bdef89563786441cf542dbf64ac24`; run record SHA-256
`3255ed4150707b5e7a00c95e4a191310bbf3a724eba8a3a5583cacc8742ec2a9`; recertification SHA-256
`9f7c61747437683ba789f4c26238b3ff61e993e0dd26e08909a0bf41d49e6a94`.

E6b is registered before hard physics against `E6B_CROUCH_HARD_PROPOSED.json`, SHA-256
`145a8bd14c794c018a6b2e003e57c55c2721196fda09966ffe4c4ec84c642224` (four serial cells,
0.417 contended GPU-hour ceiling). Predictions are nominal-hard rejected and adapted-hard accepted
for both 089 and 090. Both complete patterns are required to confirm E6's primary >=2/3 result;
either source failure falsifies it. Nominal-hard rejection must be uniquely binding-head/torso
contact, adapted-hard must have no secondary contact, and infrastructure failure remains separate.

### LFH-E6 reconciliation — 2026-08-20

The primary prediction is **confirmed at exactly 2/3 sources**; the 3/3 stretch prediction is
falsified. E6b reproduced rejected/accepted hard outcomes for both 089 and 090. In both sources the
first >1 N external contact was uniquely nearest `/World/ConstraintFrame/BindingShelfPlank`, carried
by `torso_link`, and preceded 0.15 m reference drift (089: contact frame 110, drift frame 118; 090:
contact frame 99, drift frame 116). Both adapted-hard cells had zero external contact and accepted.
Together with their E6a easy cells, these are two complete new source families. Source 086 remains
refused at adapted-easy and received no hard spend.

E6 used 10 rollouts and 0.148 contended GPU-hours. Approved E6b manifest SHA-256:
`11e8e7cd606aa16fc7163bbf9b40c4dfbe45df74f5435868b40ac63f143cb046`; hard run SHA-256:
`c6ddd7a16b9240d021a0af54dcf3d34872ba19e1deeb507d8d70e6149b4aa892`; adjudication
`e6_crouch_pilot.json` SHA-256:
`b3d76eb9c1b99d2590f7f277a3ec3b9692daa44f865b4977cce7f881fdbf41dc`. No result enters claim 5.
The extractable source count is now three (`cf_005_056`, 089, 090), so E5 remains blocked. The next
source-expansion question is whether 086's earlier/shorter valid support atom avoids the observed
visual-context drift; archetype multiplication must not substitute for that missing source.

### LFH-E6c — source-086 finite-exposure ablation

Registered 2026-08-20 after E6 and before E6c physics. This is an explicitly adaptive follow-up,
not a retry counted toward E6's already reconciled 2/3 result. It holds source, motion pair,
operator, route progress 0.543890, `shelf_plank` archetype, DCS target, and symmetric engineering-
margin rule fixed while reducing along-route face exposure from 0.30 m to 0.10 m. The new exact
empty-room support retains a 33.41 mm engineering window; hard coordinate is 1.255868 m.

E6c-easy is two cells in `E6C_CROUCH_EXPOSURE_EASY_PROPOSED.json`, SHA-256
`c1fcb5229a73b1d6f10cc8c94111590d3f895dbd8dd3f273187547b1577fefe9` (0.208 contended GPU-hour
ceiling). Prediction: nominal-easy and adapted-easy both accept with zero external contact, and the
pinned hard coordinate remains inside the scene-conditioned symmetric interval. Any easy failure
refuses the ablation without hard spend and shows that exposure reduction alone is insufficient.
If the easy gate passes, a separately registered two-cell hard phase predicts nominal rejection by
the head/torso binding plank and adapted acceptance with zero external contact.

### LFH-E6c-easy reconciliation and hard registration — 2026-08-20

The exposure-ablation easy prediction is **confirmed**. Nominal and adapted both accepted with zero
external contact; adapted drift fell from 0.15097 m/s in the 0.30 m E6 face to 0.12820 m/s in the
0.10 m face. The pinned 1.255868 m hard coordinate remains inside a 25.53 mm scene-conditioned
symmetric interval. Easy approved-manifest SHA-256:
`07872054d5b43acda65f07c421cad045a170160b33e0da8a5d2277ba82faba71`; run SHA-256:
`6bfd403c375a71eaf71fd8914ec899e2b9e593e1f763c23d314fb91b6b7ffe1f`; recertification SHA-256:
`cec7ab748e7c219b4800796fec7b60759beb3f23934dcbdf7d6b4fc379affa3a`.

The two hard cells are registered before physics in
`E6C_CROUCH_EXPOSURE_HARD_PROPOSED.json`, SHA-256
`89dee203edeebd8aebcf35de6874842eac4625b186f5e19f6d6d603477fdcab9` (0.208 contended GPU-hour
ceiling). Prediction: nominal-hard rejects by uniquely attributed head/torso binding contact before
drift; adapted-hard accepts with zero external contact. Both are required to verify this adaptive
source family. This result will test finite exposure, not retroactively change E6's 2/3 denominator.

### LFH-E6c final reconciliation — 2026-08-20

The registered hard prediction is **confirmed**. Nominal-hard rejected on uniquely attributed
`torso_link` contact with `/World/ConstraintFrame/BindingShelfPlank` at frame 124, before reference
drift at frame 133; the runner-up authored primitive was 3902.1 mm farther away. Adapted-hard
accepted with zero external contact. Together with the easy gate, the 0.10 m exposure ablation is a
complete canonical 2x2 source family. It does not alter E6's registered 2/3 result. The paired
adapted-easy drift change (0.15097 to 0.12820 m/s) is evidence that finite exposure matters for this
source/seed, not a population causal estimate.

E6c used four rollouts and 0.056 contended GPU-hours. Approved hard-manifest SHA-256:
`ee1ef7a0cfb7f4187f75edb7468f4059afc3f92c73f7f902d639938e03fbb235`; hard run SHA-256:
`18b526d56b61a01aa83f46ebbfbfdcff6a208609e02a77539bd7adb808bd01dc`; adjudication
`e6c_exposure.json` SHA-256:
`12a7dca98cb421f6eee8686f476518811c2437435ef5360e70e04aecb3dd9105`. The extractable source
count is now four (`cf_005_056`, 086, 089, 090). E5 remains blocked pending two verified archetype
transfers per new source; no E6/E6c result enters claim 5.

### LFH-E7 — source-conditioned archetype transfer

Registered 2026-08-21 before E7 physics. E7 preserves each verified source's exact motion pair,
route station, finite exposure, easy/hard coordinates, and engineering-margin rule while changing
only the deterministic obstacle archetype to `door_lintel` or `ibeam`. In particular, source 086
retains the E6c-verified 0.10 m exposure. Six source-archetype variants passed binding-face
measure-back, route, four-sign geometry, and non-binding keepout gates.

E7a contains the 12 easy cells in `E7A_ARCHETYPE_CONTEXT_PROPOSED.json`, SHA-256
`f95bfb63f1a403e6bf85b8fef97bb056709f65f589101a23ee75c2228f3c3d45` (1.25 contended GPU-hour
ceiling). Prediction: at least four of six variants, spanning all three sources, accept both easy
motions with zero external contact and retain the pinned hard coordinate inside a ≥20 mm
scene-conditioned engineering interval. Stretch: all six. Only passing variants may enter a
separately registered hard phase.

Final E7 primary, assessed after hard registration: at least four of six proposed variants produce
the complete accepted/accepted/rejected/accepted pattern, with at least one verified transfer per
source. Nominal-hard rejection must be uniquely attributed to the head/torso binding primitive
before reference drift; every intended-clear cell must have zero external contact. Stretch: 6/6,
which reaches the four-source × three-archetype E5 readiness gate. Any smaller result remains
evidence about context survival and does not justify learner training or a learned verdict gate.

### LFH-E7a reconciliation and E7b registration — 2026-08-21

The E7a prediction is **confirmed at the 6/6 stretch level**. All 12 easy cells accepted with zero
external contact. Every source-archetype variant retains its pinned hard coordinate inside the
scene-conditioned symmetric interval; widths range from 26.07 to 39.07 mm. Approved easy-manifest
SHA-256: `98665f87cdcb190fee9f259393af70f1c40bc2386587dbd5f1d2548b584aa64a`;
easy run SHA-256: `8d51a156e2072bb0bdb0ac31bef9a7ad9c5d76739af38c020ea785148b111661`;
recertification SHA-256: `fd6cb53f156933cf6dd4ed6550248a38e8ccc2d8b8e25c2984d9345172bbdbd8`.
E7a used 12 rollouts and 0.171 contended GPU-hours.

E7b contains the 12 hard cells in `E7B_ARCHETYPE_HARD_PROPOSED.json`, SHA-256
`58243a8a0c8ec7d570d64aa80b33fd77d53d2c2870bbce331cd5bfb387cff4be` (1.25 contended GPU-hour
ceiling). Prediction: every nominal-hard cell rejects on uniquely attributed `torso_link` contact
with the archetype's binding primitive before drift, and every adapted-hard cell accepts with zero
external contact. The registered primary remains ≥4/6 complete variants spanning all sources;
stretch remains 6/6. Results remain isolated from claim 5.

### LFH-E7 reconciliation and E7c registration — 2026-08-21

The E7 primary is **confirmed at 5/6 variants across all three sources**; the 6/6 stretch is
falsified. All 24 outcome labels matched and all nominal-hard contacts were uniquely attributed to
`torso_link` on the binding primitive. Five variants pass contact-before-drift and no-secondary-
contact gates. Source 086's door-lintel is refused because binding contact at frame 122 followed
reference drift at frame 95, despite its matching nominal-reject/adapted-accept labels. This is a
causal refusal, not an infrastructure failure, and leaves source 086 one archetype short of E5
readiness.

E7 used 24 rollouts and 0.340 contended GPU-hours. Approved E7b manifest SHA-256:
`e824e13ceb7ebf221b1368ba1baaf0d4ae7d98f8dc653fba0b1433c3ca889fbc`; hard run SHA-256:
`7877e0038c4105313806d2bfaf0771954e0f171932e49d8d2c93a9331556ca5f`; adjudication
`e7_transfer.json` SHA-256:
`048a6e27bae36cbd393c98501ffc91b09ff09c88e186e4e87beaf8ea5a034571`.

E7c is an adaptive replacement, not a change to E7's denominator. It keeps source 086, its exact
motion pair, route station, 0.10 m exposure, easy/hard coordinates, and margin rule fixed while
replacing only the refused `door_lintel` with the existing `hanging_panel` archetype. Tier-2
measure-back, route, four-sign, and keepout checks pass. E7c-easy contains two cells in
`E7C_REPLACEMENT_EASY_PROPOSED.json`, SHA-256
`49f2ad83142aa8b88f9f0327ecabd1c3481f2504e1b72a7a2bd32d78332eb16e` (0.208 contended GPU-hour
ceiling). Prediction: both easy cells accept with zero external contact and the pinned hard
coordinate remains inside a ≥20 mm scene-conditioned interval. If confirmed, a separately
registered hard pair predicts nominal rejection on uniquely attributed binding contact before
drift and adapted acceptance with zero external contact. Both phases are required before E5 opens.

### LFH-E7c-easy reconciliation and hard registration — 2026-08-21

The E7c easy prediction is **confirmed**. Nominal-easy and adapted-easy accepted with zero external
contact, and the pinned 1.255868 m hard coordinate remains inside a 27.79 mm scene-conditioned
engineering interval. Approved easy-manifest SHA-256:
`6653e293e268264d7690c56bd930c2705b4d2554f79ad5e4db58f1bd69fb57ce`; easy run SHA-256:
`0f76e5a8bb2f66a4d9bcbd6d6a898ca009252aadf7397d663eb389cb881e4796`; recertification SHA-256:
`e221933da273bbb398685b3f91586fa6740179fd4772982ba50e24591d5da8c6`. The easy phase used two
rollouts and 0.029 contended GPU-hours.

The two hard cells are registered before physics in `E7C_REPLACEMENT_HARD_PROPOSED.json`, SHA-256
`6d477ec4bb2108d0c33c1d754ed47fba38a95b0003c441979ff0598ee7e8f125` (0.208 contended GPU-hour
ceiling). Prediction: nominal-hard rejects on uniquely attributed `torso_link` contact with the
binding hanging panel before drift; adapted-hard accepts with zero external contact. Both are
required to verify the replacement and open the four-source × three-archetype readiness gate.

### LFH-E7c final reconciliation and crossed-gate audit — 2026-08-21

The E7c hard prediction is **confirmed**. Nominal-hard rejected on uniquely attributed
`torso_link` contact with `/World/ConstraintFrame/BindingHangingPanel` at frame 120, before drift
at frame 135; adapted-hard accepted with zero external contact. Together with the confirmed easy
phase, E7c is a complete replacement without changing E7's registered 5/6 denominator. Hard-run
SHA-256: `6a00b3cf1758260dd8790efc7b8b4f41d6e7bf884497bcb767267af85b010ece`;
final `e7c_replacement.json` SHA-256:
`55552511ccfc62933f6ce119e8e57d4de63ffdf181ff0e84cb32a80bf3d791df`. E7c used four rollouts
and 0.057 contended GPU-hours; E7 plus E7c used 28 rollouts and 0.398 contended GPU-hours.

The earlier “four sources × three archetypes” language is now recorded as a **count-only gate**,
not E5 authorization. The realized support has 12 verified source-archetype pairs, but only
`shelf_plank` and `ibeam` are common across all four sources. A held-out source × archetype test
requires a complete cross with at least three common archetypes. E5 therefore remains closed; the
next evidence target is a staged E8 transfer of `hanging_panel` to `cf_005_056`, 089, and 090.

### LFH-E9a — fresh motion to trajectory-frame calibration — 2026-08-21

Registered before E9a physics. Kimodo-G1-RP freshly generated a 120-frame motion from “A person
walks at a steady pace curving gently to the left.” with seed 45001 and 100 denoising steps. The
strict CPU gate passes. The existing `local_crouch` operator is applied at route progress 0.55;
its reference capsule silhouette drops 79.999 mm while preserving the root path, and the adapted
reference also passes embodiment and self-collision gates.

E9a contains the two empty-scene cells in `E9A_MOTION_SCENE_CALIBRATION_PROPOSED.json`, SHA-256
`c1722d3cb72d2622c1e83aed0c97b2243bb9993cd508dd4b94f45310192b5f4d` (0.208 contended GPU-hour
ceiling). Primary prediction: both nominal and adapted motions accept in `screen_empty` with zero
external contact. Only then may executed trajectories define a 3D route frame and finite-face
support. Stretch: an overhead engineering interval of at least 20 mm survives symmetric 18.044 mm
engineering margins. Lateral, floor, and oblique axes remain unsupported unless those same
executions show a corresponding paired separation; the route frame alone is not evidence.

### LFH-E9a reconciliation and E10 registration — 2026-08-21

The E9a primary and stretch predictions are **falsified**. The fresh nominal motion accepted with
zero external contact, 0.08377 m endpoint error, and 0.11147 m p95 path error. The crouch twin had
zero external contact but rejected on endpoint and path tracking: 0.37573 m endpoint error and
0.41021 m p95 path error. The loop stops before geometry authoring, as registered. This is a
controller-delivery failure, not evidence against an overhead obstacle and not a physics retry.

E10 tests a different, already verified source to make scene composition explicit. It keeps the
source-086 motion pair, hanging-panel binding atom, finite exposure, face coordinates, and seed
policy fixed, then adds four deterministic context obstacles at route progress 0.22–0.83 on both
lateral sides, at floor level, and overhead. These faces are context only: they do not establish
critical lateral, floor, or oblique support. The CPU certificate preserves the four binding signs,
measures each face back, and reports 288.78 mm minimum context-to-sweep clearance against the
required 50 mm.

`E10_CONTEXT_RICH_PROPOSED.json`, SHA-256
`deb3ceba0c99b52a7559544dc18c44b54e40089f054236acf1d35fdc953e936c`, contains four cells with a
0.417 contended GPU-hour ceiling. Primary prediction: the original
accepted/accepted/rejected/accepted pattern survives. Nominal-hard contact must remain uniquely
attributed to the hanging-panel binding primitive before drift; the three intended-clear cells
must have zero external contact. Any context contact refuses the scene. E10 is a multi-obstacle
context-survival and visualization pilot, not evidence that the LFH critical proposal distribution
already supports all directions.

### LFH-E10 reconciliation and E10b seed control — 2026-08-21

E10’s primary is **falsified**: nominal-easy accepted, but adapted-easy rejected on reference drift;
nominal-hard and adapted-hard rejected. Both adapted cells had zero external contact and identical
0.15274 m/s drift, so no context obstacle struck the robot. Nominal-hard contacted `torso_link`.
E10 spent four rollouts and 0.0575 contended GPU-hours.

E10 does not identify a context effect because its seed 33101 differed from the verified E7c seed
32301 and the seed also controls the hanging-panel non-binding thickness. E10b corrects this by
holding the original motion pair, binding geometry seed 32301, face coordinates, exposure, and
four route-relative context obstacles fixed. Its CPU certificate again reports 288.78 mm minimum
context clearance. `E10B_CONTEXT_RICH_SEED_CONTROL_PROPOSED.json`, SHA-256
`962af6a87de018cad19c6b040c764d4b49694cd1c7b238cfd00d162e5b875447`, contains four cells with a
0.417 contended GPU-hour ceiling. Primary: the verified accepted/accepted/rejected/accepted pattern
survives at the exact source seed, with zero external contact in intended-clear cells and unique
binding contact before drift in nominal-hard. If either adapted cell still rejects without contact,
context invariance is falsified for this source; if all four match, E10 is assigned to seed
sensitivity rather than context.

### LFH-E10b final reconciliation — 2026-08-21

The seed-matched E10b primary is **confirmed**. The five-obstacle scene reproduces
accepted/accepted/rejected/accepted. All three intended-clear cells have zero external contact.
Nominal-hard contact is uniquely attributed to `torso_link` on
`/World/ConstraintFrame/BindingHangingPanel` at frame 120, before reference drift at frame 135;
none of the four context obstacles becomes causal. E10b used four rollouts and 0.0586 contended
GPU-hours. Approved-manifest SHA-256:
`7dd953b45e768a15c5915d21c84cdfe6575532f93d541a2979876232e1b9c0db`; run SHA-256:
`93321dc778297c019b0a332ec110b309499ac69783521b8b89bb6c0236f0bbea`; adjudication SHA-256:
`210ceda7f8db28b252d3eca37bafa4bd8879e95683a57a753a90e262e6f73ec6`.

The contrast with E10 establishes seed sensitivity for this source/context pair; it does not prove
population-level context invariance. E10b verifies multi-obstacle composition with a single causal
binding obstacle. Critical lateral, floor, and oblique proposal support remains unverified.

### LFH audit corrections — 2026-08-26

An audit of the geometry, scene-authoring, and model layers ran every claim against executed code.
Full report: `docs/hallucination/REPORT_AUDIT_2026-08-26.md`. Three entries in this register are
affected, and the corrections are recorded here rather than by editing the original entries.

**1. `q_LFH_conditional_v1`'s conditioning is refuted at this scale.** The retained mechanism check
compared 1.51373 nats against a uniform prior at 1.60944 and credited the gap to trajectory
conditioning. Replacing the kernel with a constant — same source balancing, same 0.10 exploration
floor — scores **1.46416 nats**. Feature-blind archetype counting beats the fitted model, so the
kernel *costs* 0.0496 nats and the entire gain over uniform belongs to marginal frequency. Top-3
recall 11/12 and 400/400 in-support sampling are true by construction (two archetypes are verified
for every source and tie at 0.32; the coordinate sampler maps a bounded quantile affinely onto the
interval the check tests). This is a **refutation of the mechanism claim**, not a partial
confirmation. The evaluation does not leak; the learning target is wrong. The executed pair
determines where the face must be — closed form, deliberately unlearned — and does not determine
what the face looks like.

**2. LFH-E9a's conclusion is confounded and does not support its stated finding.** E9a is recorded
as evidence that a freshly generated motion failed on controller delivery. The fresh motion has
route straightness **0.747**, measured from the stored clip, against the `MIN_ROUTE_STRAIGHTNESS =
0.95` that the CAL3 selection itself imposes; the three verified sources are 0.985–0.991. It was
also commanded at 80 mm, the top of the CPU ladder, which CAL3 swept and then discarded in favour
of the single historically verified value. E9a therefore varied straightness and amplitude
together and outside the calibrated envelope. The recorded physics is valid; the inference "a fresh
generated motion is not yet a valid LFH source" is not supported by it. E12 re-tests it inside the
envelope.

**3. The E10/E10b seed mechanism as stated does not hold.** The E10b entry attributes E10's failure
partly to the seed also controlling the hanging panel's non-binding thickness. That coupling is
real — `archetypes.py:172` draws the binding-cube thickness from the field the manifest calls
`visual_seed`, and it feeds `PhysicsCollisionAPI` — but measured, seeds 33101 and 32301 give
0.41799 m and 0.41784 m, a **0.15 mm** difference. That cannot explain a four-cell pattern change.
E10's failure is a simulator-seed effect with no geometric difference to speak of. The E10b
conclusion (seed sensitivity, not context) is unchanged; only the offered mechanism is withdrawn.

Seven fail-open code defects were fixed under regression test, one of which (`lateral_face_reach`
measuring whole-capsule endpoints after an AABB test, up to 8× overestimate) combined with
`solve_window` emitting `+inf` margins meant **no lateral window could ever populate a spec**. The
recorded claim that lateral critical support is empty is therefore confounded between the E3
physics observation and a broken instrument, and must be re-derived on fixed code before it is
reported as a negative result. The overhead path is unaffected: after all fixes, recomputing
`lfh_089_crouch`'s support from the stored trajectories reproduces `critical_support_cal3.json`
bit-exact, and the full offline suite passes at 828.

### LFH-E12 — crouch amplitude ladder and source supply — registered before spend, 2026-08-26

**Question.** Does a freshly screened motion source a critical window when the crouch amplitude is
chosen per motion rather than fixed at 80 mm, and how many independent sources does the existing
clip pool actually hold?

**Why now.** Every downstream claim scales with independent sources, and the recorded count is 4.
The pool holds 150 clips, 94 gated worth-a-rollout, **58 with straightness ≥ 0.95** across ten body
modes; four have ever reached a crouch calibration. CAL3 swept a 40/55/70/85 mm ladder on CPU and
then recommended only the 80 mm rung, so no evidence exists about whether shallower commands
deliver where 80 mm does not.

**Protocol.** `screen_crouch_ladder.py` screens every eligible clip at all four rungs, recording
delivered reference drop, joint excursion, cap status, and lateral coupling per rung. A rung is
*usable* only if the strict reference gate passes, the root path is preserved, the excursion is
uncapped, and at least 90% of the commanded drop is delivered on the reference. Physics then runs
empty-scene cells for a cohort selected from that screen: the nominal first, then **every** usable
rung of an accepted nominal. The largest accepted rung is the motion's delivered amplitude.

**Design amendment, before any spend.** The rungs were first chained, each depending on the
acceptance of the one below, so a motion's ladder stopped at its first rejection. That saves at
most a handful of rollouts — the cohort is a fixed size either way — and it makes prediction 3
untestable, because a deeper rung is never rolled after a shallower one is rejected and a
non-monotone motion therefore cannot be observed. Every rung now depends on the nominal alone, so
the amplitude-acceptance curve is measured rather than assumed.

**Predictions, registered before any rollout.**

1. **Primary.** At least 8 clips outside the four existing sources yield an accepted
   nominal *and* at least one accepted crouch rung.
2. The median largest-accepted amplitude lies between 40 and 70 mm — i.e. below the fixed 80 mm
   command that CAL3 standardised on.
3. Acceptance is monotone in amplitude within a motion: no motion accepts a deeper rung after
   rejecting a shallower one. A violation would falsify the ladder's stopping rule and is worth
   more than a confirmation.
4. At least 6 of the accepted pairs clear a 20 mm engineering window after symmetric 18.044 mm
   margins, and so enter `P_feas` as new sources.
5. Lateral coupling above 20 mm on the deepest usable rung predicts rejection. This is a
   one-observation hypothesis from motion 095 and is registered so it can be refuted.

**Failure interpretation.** If prediction 1 fails, the binding constraint is the operator rather
than the pool, and the ladder is the evidence for saying so; the paper's yield figure becomes the
headline and the equal-budget learner comparison is dropped for this submission. A null result is
retained either way.

**Scope.** Empty-scene cells only. No scene is authored from any of these motions in E12; that
requires the accepted executed pair first, per the standing execution order. Generated-motion
families remain excluded from claim 5 unless a separate explicit scope decision is made.

### LFH lateral re-derivation on fixed code — exploratory, 2026-08-26

Not a physics batch: this re-measures **existing immutable executions** on the repaired instrument,
which the governance charter places under "CPU analysis of any depth". No rollout was spent and no
prediction is adjudicated. It is filed because it changes what the standing lateral claim means.

Before the audit, `lateral_face_reach` measured whole-capsule endpoints after an AABB test (up to
8x overestimate) and `solve_window` emitted `+inf` margins for any group that never occupied the
face, which fail to serialise. A lateral window therefore could not reach a `ConstraintSpec` even
where the geometry allowed one. Both defects are fixed under regression test.

Re-deriving the eight accepted empty-scene arm-tuck pairs with `lateral_gap_reach` — the exact
two-face gap contract from D2-012, not the one-sided diagnostic — over a grid of height bands
(0.55–1.30 m, widths 0.20/0.30/0.40 m) and exposures (0.10/0.20/0.30 m) at route progress 0.55:

| pair | commanded one-sided reduction | best band / exposure | raw gap window |
|---|---:|---|---:|
| `lfh_089_arm_tuck_left` a067 | 20.0 mm | [0.65, 0.85] / 0.10 | **51.4 mm** |
| `lfh_084_arm_tuck_right` a100 | 10.9 mm | [0.55, 0.75] / 0.30 | **48.2 mm** |
| `lfh_084_arm_tuck_right` a067 | 8.3 mm | [0.55, 0.75] / 0.30 | 42.4 mm |
| `lfh_084_arm_tuck_right` a033 | 5.9 mm | [0.55, 0.75] / 0.30 | 33.0 mm |
| `lfh_092_arm_tuck_right` | 7.7 mm | [0.75, 0.95] / 0.20 | 28.8 mm |
| `lfh_086_arm_tuck_left` a033 | 14.7 mm | [1.10, 1.30] / 0.30 | 13.7 mm |
| `lfh_086_arm_tuck_left` a100 | 45.2 mm | [1.10, 1.30] / 0.20 | 12.7 mm |
| `lfh_086_arm_tuck_left` a067 | 29.6 mm | [1.10, 1.30] / 0.20 | 8.3 mm |

**All eight refuse with `window_below_min`.** The raw separations are non-trivial — the top two are
comparable to the overhead raw windows of 52–72 mm — but the symmetric 18.044 mm engineering
margin removes 36.1 mm from every one, leaving the best candidate at 12.1 mm against a 20 mm floor.

**The standing claim changes shape.** "Lateral critical support is empty" was recorded as a
property of the axis. On the repaired instrument it is not empty; it is **margin-limited**, and the
margin in question is an overhead number imported wholesale. E1a measured it as the maximum
three-run *in-scene clearance* range across eight **overhead** cells. Nothing has ever measured
lateral repeatability, and the paired null control in `REPORT_DELIVERY_MODEL.md` shows the
empty-scene overhead estimator reproducing to within 2.73 mm — six times tighter — so the imported
value is plausibly wrong for this use in both axes.

**Two things this is not.** It is not evidence that a lateral family exists: the E3 physics
observation stands unchanged, and a binding lateral face may still alter the executed path before
drift. And every number above is a **maximum over a 36-cell band/exposure grid**, so it is
optimistically biased by selection in exactly the way a grid-searched overhead window is; these are
upper bounds on what a pre-registered station and band would find, not certified windows.

**LFH-E14, consequently redesigned and not yet registered for spend.** The next lateral step is
*not* another E3-style retry. It is a lateral repeatability measurement — the E1a analogue, three
runs of one accepted tuck pair, measuring the run-to-run spread of the executed gap window — so
that the axis gets a margin derived from its own noise instead of an overhead import. Only then is
a pre-registered station, band, and exposure worth a hard cell. Until that exists, lateral remains
reported as unverified support, and the paper's negative result must be stated as "no lateral
family survives the current margin rule", never as "the axis admits none".

### LFH-E12 reconciliation — 2026-08-26

E12 spent 47 cells and **0.427 contended GPU-hours**. Run record
`E12_CROUCH_LADDER_2026-08-26.json`; report `docs/hallucination/REPORT_E12_CROUCH_LADDER.md`;
machine-readable `docs/hallucination/e12_crouch_ladder.json`.

**The primary prediction is falsified.** Predicted at least 8 new sources with an accepted nominal
and at least one accepted rung; observed **5**. Prediction 4 is also falsified: predicted at least
6 pairs clearing a 20 mm engineering window, observed **1**.

Funnel: 12 motions → **8** nominals accepted → **5** with a delivered amplitude → **1** entering
`P_feas`. Two of the four nominal rejections are gait-specific and clean: `side_step` at 0.454 m
endpoint error and `backward` at 0.380 m plus a disallowed foot contact, both on routes of
straightness ≥ 0.99. Body-mode diversity in the clip pool does not survive the controller, and
`turn_in_place`, `stand_to_walk`, and the second `side_step` accepted their nominals but rejected
every rung.

**Prediction 2 is confirmed, and it matters.** The median largest-accepted amplitude is **55 mm**,
inside the predicted [40, 70] and well below the 80 mm that CAL3 standardised on. Delivered
amplitudes are 40, 40, 55, 70, 70 mm — five different answers for five motions. A single fixed
command is the wrong instrument; that much of E12's premise holds.

**Prediction 3 is confirmed.** No motion accepted a deeper rung after rejecting a shallower one,
across all 8 accepted nominals. This was only observable because the design amendment made the
rungs independent of each other; under the original chained design a violation could not have been
seen. Prediction 5 is not adjudicable: no motion with an accepted nominal carried more than 20 mm
of lateral coupling on a graded rung.

**Why the ladder did not rescue supply, which is the real finding.** A shallow crouch tracks but
does not separate. The five delivered pairs give raw executed windows of 31.5, 43.7, 51.3, 54.5 and
70.2 mm, and the symmetric 18.044 mm engineering margin removes 36.1 mm from each:

| margin, each side | pairs clearing the 20 mm floor |
|---|---|
| **18.044 mm (current)** | **1 / 5** |
| 10 mm | 4 / 5 |
| 5 mm | 5 / 5 |
| 2.73 mm (measured null-control spread) | 5 / 5 |

**So the binding constraint is not the amplitude and not the motion supply — it is the imported
engineering margin.** Four of the five pairs produce raw windows in the same range as the three
verified CAL3 sources (52–72 mm) and are refused by a margin rule, not by physics. E1a derived
18.044 mm as the maximum three-run **in-scene clearance** range across eight **overhead** cells;
the empty-scene paired null control in `REPORT_DELIVERY_MODEL.md` reproduces a true-zero window to
within **2.73 mm**. Those are different quantities and the margin is **not** being changed here.
But it is now the single highest-value question in the pipeline, and it is answerable cheaply.

**Registered next step, not yet spent: LFH-E16, a margin measurement.** Take the three E12 pairs
with the widest raw windows, re-roll each accepted nominal/adapted pair at three simulator seeds in
the empty scene, and measure the run-to-run spread of the *executed window* directly. That yields
an empty-scene-calibrated margin for the quantity the margin is actually applied to, replacing an
in-scene overhead import. Predictions will be filed before spend. At 24 rollouts this costs roughly
0.2 GPU-h and, if the spread is anywhere near the null control's 2.73 mm, converts four refused
pairs into sources — a larger yield gain than any amount of further motion generation.

**What E12 does not show.** It does not show that fresh or diverse motions cannot source families:
it tested 12 clips from one pool with one operator at one station, and 5 produced executed windows.
It does not license changing the margin. And it does not adjudicate E9a, whose confound is
straightness and amplitude together; E12's cohort was straightness-filtered by construction.

### LFH-E16 — empty-scene window repeatability — registered before spend, 2026-08-26

**Question.** What is the run-to-run spread of the *executed critical window* in the empty scene,
under simulator seed alone, with motion, operator, amplitude and station held fixed?

**Why.** E12 refused 4 of its 5 delivered pairs on the margin rule rather than on physics. The
margin in force, 18.044 mm each side, is E1a's maximum three-run **in-scene clearance** range over
eight overhead cells. The window it is applied to is an **empty-scene reach difference**. Nothing
has ever measured the repeatability of that quantity, and the paired arm-tuck null control in
`REPORT_DELIVERY_MODEL.md` reproduces a true-zero window to within 2.73 mm — a factor of six
tighter. E16 measures the right quantity so the margin can be derived rather than imported.

**Protocol.** Three E12 pairs with the widest raw executed windows — `ladder_138` (70.2 mm),
`ladder_148` (54.5 mm), `ladder_034` (51.3 mm) — each re-rolled at **three simulator seeds** for
both nominal and its deepest accepted rung, in `screen_empty`. Motion files, operator, amplitude,
station, face extent and scorer are byte-identical to E12; only `++seed` changes. 18 cells.
Executed window is recomputed by the same `overhead_face_reach` instrument at the same commanded
station, and the statistic is the **range across the three seeds** per pair — the same statistic
E1a reported, so the two are directly comparable.

**Predictions, filed before any rollout.**

1. **Primary.** The maximum across the three pairs of the three-seed executed-window range is
   **below 18.044 mm**. This is the claim that the in-scene margin is over-conservative for the
   empty-scene quantity.
2. The maximum three-seed range is **at or below 10 mm**. Stronger, and it is the threshold at
   which 4 of E12's 5 pairs clear the 20 mm floor.
3. All 18 cells accept with zero external contact. A rejection at a seed that E12 accepted would
   itself be a finding about seed sensitivity and is retained, not retried.
4. Per-pair seed ranges do not scale with window width: the widest pair is not the noisiest.

**Failure interpretation.** If prediction 1 is falsified, the 18.044 mm margin is vindicated for
this use, E12's yield of 1/5 stands as a real physical result, and the supply problem is genuinely
about motion and operator rather than bookkeeping. That outcome is retained and reported; it would
redirect effort to a deeper or second edit operator rather than to margin arithmetic.

**Scope and standing rules.** Empty-scene only; no geometry is authored in E16. The margin is
**not** changed by this experiment — E16 measures, a separate registered decision would apply. The
E1a value stays in force for every existing artifact regardless of outcome, so no published window
is retroactively re-graded.

### LFH-E16 partial reconciliation — 2026-08-26

E16 spent 0.580 contended GPU-hours and stopped at **6 of 18 cells** with
`status: infrastructure_failure`. The cause is external GPU contention, not the experiment: another
user's training job took 23.8 GB of the 32 GB card mid-batch, the seventh cell stalled in Isaac
startup with 1.7 GB free, and the runner's 1800 s hang timeout fired and stopped the line as its
stop conditions require. **No cell was scored as rejected because of it** — the infrastructure /
science split held. The remaining 12 cells are resumable against the same run record when the card
frees.

The six completed cells are one full pair, `ladder_138` at three seeds, and they falsify a
prediction and displace the experiment's own primary.

**Prediction 3 is falsified.** All 18 cells were predicted to accept; the nominal accepted 3/3 but
the adapted rung accepted **1/3**.

**The primary measurement is not available for this pair, and that is the finding.** E16 set out to
measure the three-seed range of the executed *window*. A window needs both cells accepted at the
same seed, and only one seed qualifies, so no range exists to report. The experiment assumed
acceptance was a property of the motion; it is not.

| cell | endpoint error across four seeds (E12 + E16) | range | gate |
|---|---|---:|---|
| `ladder_138` nominal | 0.0705, 0.0852, 0.1105, 0.1214 m | **50.9 mm** | 0.35 m |
| `ladder_138` d070 (adapted) | 0.3298, 0.3300, 0.3823, 0.4355 m | **105.7 mm** | 0.35 m |

The adapted cell sits **directly on the acceptance threshold** and straddles it: two of four seeds
below 0.35 m, two above. Its seed-to-seed endpoint spread is 105.7 mm, twice the nominal's, and the
gate is 20 mm from the median. Acceptance at this amplitude is close to a coin flip.

**What this does to the preceding results.** E12's "delivered amplitude" is a single-seed point
estimate taken at the edge of a cliff. `ladder_138` was E12's best new source — widest window,
70.2 mm, the only pair clearing the current margin — and it is the pair that fails to reproduce.
The honest reading of E12 is therefore weaker than its reconciliation stated: its five delivered
amplitudes are seed-optimistic, and the yield of 1/5 entering `P_feas` may itself not reproduce.
Nothing in E12 is withdrawn — every recorded cell stands — but "delivered amplitude" must be
redefined as *accepted at k of n seeds*, and no single-rollout amplitude should be promoted again.

**This also reorders the margin question.** E16 was registered because the engineering margin
looked like the binding constraint. It may still be, but a prior constraint has appeared: the
deepest accepted rung is not a stable operating point, so a window measured there is not a stable
window. Choosing the amplitude with *margin below* the tracking cliff now matters more than
trimming millimetres off the placement margin.

**Registered next, before spend: LFH-E16b.** Re-scope from window repeatability to **acceptance
repeatability**. For three pairs, roll each of the three usable rungs at three seeds — 27 cells,
about 0.3 GPU-h — and report, per motion and per amplitude, the fraction of seeds accepted and the
endpoint-error distribution. The delivered amplitude becomes the deepest rung accepted at 3/3
seeds, and the window is measured only there. Predictions: (1) at least one rung per motion accepts
3/3; (2) the 3/3 amplitude is at least one rung shallower than E12's single-seed answer; (3) among
rungs that accept 3/3, the three-seed window range is below 18.044 mm — E16's original primary,
asked where it is answerable. Do not resume the original E16 manifest; its design conditions on an
assumption now known to be false.

### LFH-E17 — first family from a ladder-discovered source — registered before spend, 2026-08-26

**What.** The counterfactual 2x2 for `ladder_138`, a `reach_walk` clip that became a source through
the LFH-E12 amplitude ladder rather than through the CAL3 cohort. Its executed pair gives a
70.2 mm raw window, 34.1 mm after symmetric 18.044 mm margins, binding `head_torso`. A
`shelf_plank` is authored at the window centre (hard, 1.2671 m) and 50 mm above the nominal's
executed reach (easy, 1.3521 m). Tier-2 keep-out passed at authoring.

**Why it matters.** Every verified family to date descends from the CAL3 `stand_to_walk` cohort.
This is the first test of whether the pipeline generalises to a source found by a different
selection route and a different body mode.

**Seed policy.** The simulator seed is pinned to E12's `34007`, at which this adapted rung was
observed to accept. LFH-E16 showed the same rung straddles the tracking gate across seeds
(endpoint error 0.3298 / 0.3300 / 0.3823 / 0.4355 m against a 0.35 m threshold), so an unpinned
seed would confound a scene result with a delivery coin flip. This is a seed-matched test in the
sense E10b established, and it is **not** a population claim about the source.

**Predictions.**

1. **Primary.** The canonical pattern: `easy/nominal` accepted, `easy/adapted` accepted,
   `hard/nominal` rejected, `hard/adapted` accepted.
2. `hard/nominal` is rejected for `disallowed_robot_contact` with contact attributed to the binding
   plank, and that contact occurs **before** any reference drift.
3. The three intended-clear cells record zero external contact.
4. `hard/adapted` endpoint error stays within 20 mm of its empty-scene value at the same seed
   (0.3298 m); a larger excursion would mean the scene, not the operator, is spending the budget.

**Failure interpretation.** If `hard/adapted` rejects, the 34.1 mm engineering window is not
deliverable in an authored scene for this source and the pair is refused — which would be evidence
that the margin is *not* over-conservative after all, and would cut directly against the E12
reconciliation's reading. If `hard/nominal` is accepted, the closed-form window over-states the
separation for this source and the window solver needs re-examination for `reach_walk` geometry.
Either outcome is retained and reported.

**Scope.** One source, one archetype, one seed. This does not establish that ladder-discovered
sources generalise; it tests whether this one produces a family at all.

### LFH-E17 reconciliation — 2026-08-26

**All four predictions are confirmed.** E17 spent 4 cells and **0.078 contended GPU-hours**. Run
record `E17_LADDER_FAMILY_2026-08-26.json`.

| cell | verdict | endpoint | peak external | attribution |
|---|---|---:|---:|---|
| `easy/nominal` | accepted | 0.1214 m | 0.0 N | — |
| `easy/adapted` | accepted | 0.3298 m | 0.0 N | — |
| `hard/nominal` | **rejected** | 0.3464 m | **415.9 N** | `torso_link`, frame 101 |
| `hard/adapted` | accepted | 0.3298 m | 0.0 N | — |

1. **Primary confirmed.** The canonical accepted/accepted/rejected/accepted pattern, from a source
   the CAL3 cohort never contained: `ladder_138` is a `reach_walk` clip discovered by the E12
   amplitude ladder. This is the first family authored by `synthesize_from_ladder.py`, which
   generalises the CAL3-bound synthesiser to any accepted executed pair.
2. **Attribution confirmed, and stronger than predicted.** `hard/nominal` is rejected for
   `disallowed_robot_contact` **alone** — 415.9 N on `torso_link` at frame 101, of which 369.2 N is
   the overhead component. No drift reason is recorded at all, so the contact is not merely before
   drift; drift never crossed its threshold. One plank, one link, one cause.
3. **Zero external contact in all three intended-clear cells.**
4. **Endpoint stability confirmed exactly.** `hard/adapted` records 0.3298 m, identical to its
   empty-scene value at the same seed. The authored scene costs the adaptation nothing.

**The two adapted cells are bit-identical, and that is the result rather than a defect.** Both
report endpoint 0.3298 m, p95 0.3422 m and the same trajectory SHA-256. The handoff's standing
check applies here and passes: the two cells ran as separate rollouts at 13:13:00 and 13:18:13,
took 57.4 s and 68.1 s, wrote different output directories and different rollout logs, and used
scenes with different SHA-256. The trajectories agree because the simulator is deterministic at a
fixed seed and **the adapted motion never touches either plank** — the 85 mm difference in plank
height is causally irrelevant to a robot that crouches under both. That is the counterfactual
stated as sharply as this corpus can state it: the only thing the scene changes is the nominal's
fate.

**Scope, unchanged from registration.** One source, one archetype, one pinned seed. LFH-E16 showed
this adapted rung straddles the tracking gate across seeds, so E17 is a seed-matched demonstration
that the source can produce a family — not evidence that it does so at a randomly drawn seed. The
seed-robustness question belongs to E16b.

Visual evidence, kinematic MuJoCo replay of the recorded Isaac states:
`docs/source/_static/lfh_e17/ladder138-family-2x2.png` (all four cells at their binding frames,
captioned with the authoritative verdicts), `ladder138-critical-frame.png` (the 35.1 mm that
decides it), and `verified-hard-nominal.mp4` / `verified-hard-adapted.mp4`.

### LFH-E18 — archetype freedom at a fixed critical point — registered before spend, 2026-08-26

**Question.** Given one executed pair and one face coordinate, is the *appearance* of the binding
obstacle free? This is the two-stage factorisation LfH-CP proposes — identify the critical
configuration, then generate diverse realisations through it — tested directly for the first time
on a ladder-discovered source.

**Design.** `ladder_138`'s window, station, exposure and both face coordinates are held byte-fixed
at the values LFH-E17 verified. Only the archetype changes: `door_lintel`, `ibeam`,
`hanging_panel`, each with its own geometry seed. Twelve cells, seed pinned to 34007 for the same
reason as E17. `shelf_plank` is not re-run; E17 is its result.

**Predictions.**

1. **Primary.** All three archetypes reproduce accepted/accepted/rejected/accepted.
2. Every `hard/nominal` rejection is attributed to the binding face's own primitive, with no
   context primitive touched.
3. The three intended-clear cells of each archetype record zero external contact.
4. `hard/adapted` endpoint error is within 20 mm of 0.3298 m for every archetype — an obstacle the
   adaptation clears should cost it nothing regardless of what the obstacle looks like.

**Failure interpretation.** A single archetype failing while others pass would mean appearance is
*not* free at this coordinate — most likely because a non-binding part of that archetype (a jamb,
a web, a skirt) intrudes where the swept volume needs room, which the keep-out validator should
have caught and would then be a Tier-2 gap worth chasing. That outcome is more interesting than a
clean pass and is retained either way.

**Scope.** One source, one seed, one coordinate. This tests archetype freedom at a point, not
across the window or across sources.

### LFH-E16b — acceptance repeatability — registered before spend, 2026-08-26

Replaces the original LFH-E16, whose design conditioned on an assumption its own first six cells
falsified: that a deepest-accepted rung reliably accepts. The E16 manifest is **not** resumed.

**Question.** For a given motion and crouch amplitude, what fraction of simulator seeds accept, and
how far below the deepest single-seed rung must the operating point sit to be stable?

**Design.** Three E12 pairs — `ladder_034` (single-seed delivered 70 mm), `ladder_148` (55 mm) and
`ladder_126` (40 mm), chosen to span the delivered range — each rolled at **two fresh seeds**
(39001, 39002) for the nominal and all three rungs. Combined with the seed each already ran in
E12, every cell reaches a three-seed cohort for 24 new rollouts rather than 36. Motions, operator,
station, exposure and scorer are byte-identical to E12; only `++seed` differs.

**Predictions.**

1. **Primary.** Every motion has at least one rung that accepts at **3/3** seeds.
2. The 3/3 amplitude is at least one rung shallower than E12's single-seed delivered amplitude for
   at least two of the three motions.
3. Endpoint-error spread across seeds grows with amplitude: the deepest graded rung of each motion
   has a larger three-seed range than its nominal.
4. Among rungs accepting 3/3, the three-seed range of the executed window is below 18.044 mm —
   the original E16 primary, asked where it is answerable.

**Failure interpretation.** If prediction 1 fails for a motion, that motion has no stable crouch
amplitude at all and is not a source at any depth, however good its single-seed window looked. If
prediction 2 fails — the deepest rung is already stable — then `ladder_138`'s coin-flip behaviour
is specific to it rather than a property of operating at the ladder's top, and E12's delivered
amplitudes stand as recorded.

**Budget.** 24 cells, roughly 0.35 contended GPU-h.

### LFH-E18 reconciliation — 2026-08-26

E18 spent 12 cells and **0.465 contended GPU-hours** (inflated by repeated GPU contention: another
user's job cycled between 21 GB and idle, so the runner yielded and resumed six times; every
yield was a clean refusal, never a mis-scored cell).

**Predictions 1, 3 and 4 are confirmed. Prediction 2 is falsified as worded, and the wording was
the problem.**

| archetype | easy/nom | easy/adp | hard/nom | hard/adp | binding contact | adapted endpoint |
|---|---|---|---|---:|---:|---:|
| `shelf_plank` (E17) | acc | acc | **rej** | acc | 415.9 N `torso_link` f101 | 0.32976 m |
| `door_lintel` | acc | acc | **rej** | acc | 416.0 N `torso_link` f101 | 0.32976 m |
| `ibeam` | acc | acc | **rej** | acc | 410.1 N `torso_link` f101 | 0.32976 m |
| `hanging_panel` | acc | acc | **rej** | acc | 538.5 N `torso_link` f117 | 0.32976 m |

**P1 confirmed.** Four visually distinct obstacles — a 40 mm plank, a lintel with jambs, an I-beam
with web and top flange, and a 0.42 m-thick panel — placed at the *same* face coordinate inferred
from the *same* executed pair, all reproduce accepted/accepted/rejected/accepted. This is the
two-stage factorisation working: the executed pair fixes the critical coordinate, and the object
realising it is free.

**P4 confirmed to the digit.** `hard/adapted` endpoint error is **0.3297638984283594 m in all
four**, identical to the empty-scene value at the same seed. The adaptation is completely
insensitive to what the obstacle is, because it never touches any of them.

**P3 confirmed.** Zero external contact in all twelve intended-clear cells.

**P2 falsified, and the predicate was wrong rather than the result.** I predicted every
`hard/nominal` rejection would be attributed to the binding face "with no context primitive
touched", and implemented that as *contact is the only recorded rejection reason*. `ibeam` records
three reasons — `reference_endpoint_tracking_error`, `reference_path_tracking_error` and
`disallowed_robot_contact` — with 410.1 N on `torso_link` at frame 101, the same frame and body as
`door_lintel`. Its endpoint error is 0.4208 m against `door_lintel`'s 0.3432 m. The tracking
failures are the **consequence** of a harder strike knocking the robot off course, not an
independent cause: the contact frame is identical and no non-binding primitive is touched. My
predicate conflated "contact is the cause" with "contact is the only line in the report", and only
the first is the scientific claim. The corpus's existing standard — contact attributed to the
binding face, occurring before drift — is met by all four. The predicate is corrected in
`analyze_ladder_family.py`'s report rather than the result being reinterpreted after the fact.

**One measurement worth keeping.** Peak force varies 410–538 N across archetypes at an identical
face coordinate, and the peak-force frame moves from 101 to 117 for the thick `hanging_panel`.
Verdict is invariant; contact *dynamics* are not. Consistent with the corpus's standing position
that force is not a graded quantity and only the verdict is reported.

Visual evidence: `docs/source/_static/lfh_e17/archetype-gallery.png` — all four obstacles at
underside 1.2671 m, each striking the nominal by 35.1 mm.

### LFH-E16b reconciliation — 2026-08-26

E16b ran all 24 cells for **0.746 contended GPU-hours** (inflated by repeated contention yields).
Combined with each cell's E12 seed, every rung reaches a three-seed cohort.

| pair | rung | accepted | endpoint errors (m) | endpoint range |
|---|---|---:|---|---:|
| `ladder_034` | nominal | 3/3 | 0.0568 0.0643 0.0736 | 16.8 mm |
| `ladder_034` | d040 | **3/3** | 0.0445 0.0842 0.1224 | 77.9 mm |
| `ladder_034` | d055 | **3/3** | 0.0681 0.1071 0.1513 | 83.2 mm |
| `ladder_034` | d070 | **3/3** | 0.0895 0.1302 0.1867 | 97.2 mm |
| `ladder_148` | nominal | 3/3 | 0.0661 0.1453 0.1603 | 94.1 mm |
| `ladder_148` | d040 | 1/3 | 0.2264 0.3818 0.3954 | 169.1 mm |
| `ladder_148` | d055 | 1/3 | 0.3045 0.4460 0.4576 | 153.1 mm |
| `ladder_148` | d070 | **0/3** | 0.3528 0.5098 0.5231 | 170.2 mm |
| `ladder_126` | nominal | 3/3 | 0.1152 0.1693 0.1813 | 66.2 mm |
| `ladder_126` | d040 | **3/3** | 0.2166 0.3001 0.3309 | 114.3 mm |
| `ladder_126` | d055 | 1/3 | 0.2824 0.3611 0.3830 | 100.6 mm |
| `ladder_126` | d070 | 1/3 | 0.3483 0.4049 0.4368 | 88.5 mm |

**P1 falsified.** Two of three motions have a 3/3 rung; `ladder_148` has none at any amplitude,
despite E12 recording it as delivering 55 mm. Its nominal is stable (3/3) but every rung straddles
or exceeds the gate. A motion can be trackable and still have no stable crouch.

**P2 falsified, in the informative direction.** I predicted E12's single-seed amplitudes would
prove optimistic for at least two of three. They were **exactly right for two** — `ladder_034` is
stable at 70 mm and `ladder_126` at 40 mm, precisely E12's answers — and catastrophically wrong for
the third. Single-seed delivery is therefore not systematically optimistic; it is *unreliable*, and
the failure is concentrated in motions whose rungs sit near the gate. `ladder_138`'s coin flip and
`ladder_148`'s collapse are the same phenomenon, not a general bias.

**P3 confirmed, 3/3 motions.** Seed spread grows with amplitude in every motion: 16.8 → 97.2,
94.1 → 170.2, 66.2 → 114.3 mm from nominal to deepest rung. Crouching does not merely cost tracking
budget on average — it makes tracking *less predictable*, which is why an operating point near the
gate is unsafe even when its mean sits below it.

**P4 confirmed — and it settles the question E16 was created to ask.** On rungs that accept 3/3,
the three-seed range of the executed window is:

| pair / rung | windows across seeds | **range** |
|---|---|---:|
| `ladder_034` d040 | 27.03, 27.72, 28.21 mm | **1.18 mm** |
| `ladder_034` d055 | 41.06, 38.40, 39.81 mm | **2.66 mm** |
| `ladder_034` d070 | 51.26, 51.10, 54.50 mm | **3.39 mm** |
| `ladder_126` d040 | 43.67, 42.16, 35.73 mm | **7.94 mm** |

**The executed window is highly reproducible at a stable amplitude — worst observed range
7.94 mm, against an engineering margin of 18.044 mm applied to *each* side.** This is the same
order as the 2.73 mm paired null control in `REPORT_DELIVERY_MODEL.md`, measured now by the
statistic E1a used and on the quantity the margin is actually applied to. The margin currently
removes 36.1 mm from every window to guard a quantity that moves by at most 8 mm.

**Two things this licenses, and one it does not.** It licenses a registered proposal to derive an
empty-scene margin from empty-scene repeatability, and it establishes `ladder_034` (`walk_look`,
70 mm at 3/3, window 51–55 mm) and `ladder_126` (`carry_walk`, 40 mm at 3/3, window 36–44 mm) as
**stable sources** in a sense no earlier source has been shown to be. It does **not** license
changing the margin here: E16b measured, and a margin change is a separate registered decision
affecting every existing artifact.

**Delivered amplitude is redefined, with effect.** From now the delivered amplitude of a motion is
the deepest rung accepted at 3/3 seeds, and a window is measured only there. Under the old
single-seed definition E12 reported five delivered pairs; under the new one, of the three re-tested,
two survive. `docs/hallucination/e12_crouch_ladder.json` is not rewritten — its cells stand — but
its yield figure should be read as an upper bound.

---

## LFH-E19 — the ladder is its own rival set (registered 2026-08-28, before reading the run)

Two `train_lflh_sdf.py` runs are in flight (`crouch_070`, `crouch_040`, 11 train / 5 held-out clips)
and a second ceiling sweep is measuring two readings of "the scene excludes the alternative":

* **any-rival** — some cheaper candidate is struck. Measured first, and it gives a clean linear law,
  `ceiling_mm ~= 0.578 * amplitude_mm + 10.0`: 33.2 / 41.7 / 50.5 mm median at 40 / 55 / 70 mm.
* **all-rival** — *every* cheaper candidate is struck. This is what the training loss actually asks
  for, and what makes the observed motion the only survivor rather than merely a survivor.

The two coincide at `crouch_040`, whose only cheaper candidate is the nominal. They cannot coincide
at `crouch_070`, which has **five**: nominal, `crouch_040`, `crouch_055`, and both 40 mm tucks.

**P5.** Under the all-rival rule the ceiling will *not* follow the amplitude. A box that clears
`crouch_070` and strikes the nominal must also strike `crouch_055`, whose body sits only 15 mm above
`crouch_070`'s — so the admissible band collapses from the 70 mm amplitude to roughly the 15 mm rung
spacing. I predict the all-rival median ceiling at `crouch_070` comes out **below** the any-rival
median at `crouch_040` (33.2 mm), and below its own any-rival figure (50.5 mm) by more than half.

**P6.** Consequently the `crouch_070` training run, whose margins were set from the *any-rival*
ceiling at 27.6 mm per side, is infeasible under the rule its own loss enforces, and will report a
**lower** counterfactual rate than the `crouch_040` run at 18.7 mm per side — the opposite of what
the amplitude law predicts. If P6 holds, margins must be derived from the all-rival ceiling and both
runs re-done; the amplitude law in `REPORT_SCENE_CEILING.md` §2 then describes an upper bound that
only a single-rung ladder attains.

**P7.** The falsifier: if the all-rival ceiling at `crouch_070` lands within a few mm of the
any-rival one, then the intermediate rungs are being struck for free by the same box, the rival set
is not binding, and P5–P6 are wrong. Either way the ceiling instrument now reports both, so the
question cannot be quietly dropped.

### LFH-E19 outcome, part 1 (2026-08-28)

**P5 confirmed, and by a wider margin than predicted.** On `crouch_070`, clip 000: the any-rival
ceiling is **49.7 mm** and the all-rival ceiling is **2.7 mm** — a factor of eighteen, against a
prediction of "below 33.2 mm and less than half of 50.5". The intermediate rung is not merely
binding, it is almost the whole story. P7's falsifier (the two rules landing within a few mm of each
other) is decisively not what happened.

The `eps`-parameterised reading interpolates between them exactly as the definition says it should:

| regret tolerance `eps` | ceiling, clip 000 |
|---|---:|
| 0 | 16.1 mm |
| 0.1 | 16.1 mm |
| 0.25 | 37.8 mm |
| 0.5 | 49.7 mm |

`crouch_055` costs 0.214 less than `crouch_070`, which is why the ceiling jumps between `eps = 0.1`
and `eps = 0.25`: that is the tolerance at which the neighbouring rung stops having to be excluded.

**P6 is settled by geometry, and its run was stopped rather than completed.** The prediction was
that the `crouch_070` training run would report a lower counterfactual rate than `crouch_040`,
because its 27.6 mm margins were taken from the any-rival ceiling while its loss enforced the
all-rival rule. The ceiling measurement establishes the infeasibility directly and far more cheaply:
27.6 mm demanded against 2.7 mm achievable. The run was killed at 67 minutes on a box already at
load average 90, and its loss formulation has since been replaced. **P6 should therefore be read as
confirmed by the instrument, not by the experiment it was registered against** — the weaker of the
two, and recorded as such.

What replaces it: the rival term now implements `S_{eps,delta}` directly — necessity charged to the
cheapest candidate, regret to anything cheaper by more than `eps` — and margins are read from
`scene_ceiling.json` for the target and tolerance in use, so a training objective outside the
feasible set can no longer be written by hand.
