# Predictions recorded before the result

Written before the measurement, so the record cannot be rewritten afterwards. Four of the
predictions below were wrong, which is the reason this file exists: without it, a refuted guess
quietly becomes a "finding we always suspected".

## Open

*(none)*

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
