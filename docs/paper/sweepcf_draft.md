# SweepCF — working draft

**Status: a draft of an argument, not of a paper.** Claim levels are marked throughout, and
the central learning experiment has not been run. Nothing here should be quoted as a result
without checking the status line beside it.

---

## The thesis, in one sentence

**Counterfactual supervision is what turns scene variation into scene-conditioned behaviour.**

A dataset can vary its scenes without varying anything a policy could learn from. The corpus
this work started from does exactly that: the same motion, replayed through different clutter,
yields different images and a bit-identical state trajectory. Scene appearance changed;
behavioural supervision did not. A policy trained on it can ignore the scene entirely and lose
nothing.

The repair is not more scenes. It is that **for the same task, geometry must change which
executable whole-body behaviour is preferred.**

## Claim ladder

Claim 4 is stated as a *minimum-edit* reversal deliberately. Under the lexicographic rule
`m*(S) = argmin_m D(m, m₀) s.t. y(S,m) = 1` with `D(m₀,m₀) = 0` and `D(T(m₀),m₀) > 0`, the easy
scene selects the nominal at zero edit and the hard scene selects the adapted motion because the
nominal is infeasible. That is a strict reversal and it needs no weights over knee angle, energy or
joint travel — so no calibrated scalar cost is required, and none blocks the claim. The continuous
cost is retained for comparing two *different* adaptations later.

| # | claim | status |
|---|---|---|
| 1 | Two executable motions' full-body swept volumes separate, so a geometric window exists | **established** |
| 2 | The same controller succeeds or fails as that geometry changes | **established** (3 verified families, 1 matched) |
| 3 | The 2×2 *outcome* survives a jittered start pose | **established** (12/12 cells, 3/3 jitters) |
| 4 | The scene reverses the **minimum-edit feasible behaviour** | **established** on the matched overhead family |
| 5 | A learner given counterfactual data uses scene geometry when it could otherwise ignore it | **not started — this is the decisive experiment** |
| 6 | That behaviour generalises to scenes never fitted to a trajectory | **not started** |

Levels 5 and 6 are the paper. Everything below them is apparatus.

## What is established

### The counterfactual, and its attribution

Same task, same start, same goal, same corridor. One shelf moves 179 mm.

| | easy (1.391 m) | hard (1.212 m) |
|---|---|---|
| nominal walk | accepted, 0.0 N | **rejected** |
| adapted duck | accepted, 0.0 N | accepted, 0.0 N |

The rejection is attributed, not merely counted. Contact is on `torso_link` at frame 113
against a swept-volume prediction of frame **112** — one frame, 20 ms. Reference drift begins
at frame 120, *after* the contact, so the drift is the collision's consequence rather than a
tracking failure that coincided with it. Every other cell records no lateral contact at all.

Forces are four separate quantities and are reported as such: 55.4 N at first contact,
137.2 N peak, 19.8 N·s impulse, 0.32 s duration. Under start-pose jitter the peak ranges
95.5–658.3 N while the verdict never moves, which is why the established claim is
**start-pose outcome robustness** and not robustness in general.

### Obstacle placement is part of the algorithm

Over every compatible pair in the corpus, with a 50 mm window as the usability bar:

| overhead, 55 pairs | pairs ≥ 50 mm |
|---|---|
| random station | **0 / 55** |
| route midpoint | **0 / 55** |
| maximal envelope separation | **5 / 55** |

Lateral: 4 → 6 → **25 of 88**. In the overhead regime neither baseline yields a single family,
so station selection is not an optimisation but the step that makes the method exist. The
chosen station sits a median 0.50 m from the midpoint — further than a 0.5 m shelf is deep, so
the two placements do not overlap.

*Caveat: this is geometric yield. A 50 mm window is necessary for a family, not sufficient,
and the bar is calibrated on one family.*

### Physical validity is not behavioural validity

Over the frozen corpus, 132 of 156 evaluable episodes are accepted — the robot tracked its
reference safely. Where a predicate exists to ask whether it did the thing its label claims,
only **17 of 37** pass. Three body modes score zero and one scores 3/7.

Graded on the references directly by forward kinematics, across all 150 prompts: **24 of 75**
carry their behaviour, in the seven modes that have a predicate. The split is clean —
whole-clip *styles* come back at 100%, *events at a specified moment* at 0%.

## The bottleneck, and what it forced

Mining the corpus for pairs gives the number the plan needs, and it is not the pair count:

| regime | viable pairs | distinct adapted motions | **independent families** |
|---|---|---|---|
| overhead | 5 | 2 | **2** |
| lateral | 3 | 3 | **1** |
| floor | 0 | 0 | **0** |

Three independent families. The ceiling is the supply of semantically valid adapted motions,
not the method.

Prompting did not lift it. Seven new modes at two seeds each: every phrasing that lowers the
body pins the waist at its limit on 100% of frames, the one phrasing that does not is 8 mm
*taller* than a walk, and an explicit arm-tuck request produced motions 64 mm *wider*.
*Scope: one set of templates, one checkpoint, one sampling configuration. This closed an
engineering path; it is not a claim about what the generator can do.*

## Matched adaptation operators

The fix is to construct the adapted motion from the nominal rather than pair two independently
generated clips. That also removes a fair objection to the existing family — that a shelf
separated two different journeys rather than two behaviours.

Both operators preserve root XY and yaw, duration, gait phase, start and goal, and act only
over a window in **route-progress** coordinates centred on the obstacle station:

| | `local_crouch` | `local_arm_tuck` |
|---|---|---|
| regime | overhead | lateral |
| touches | legs + root height | arms only |
| waist change | **0.000 rad** | untouched |
| leg change | — | **0.000 rad** |
| effect at the station | 1.305 → 1.207 m silhouette | 0.276 → 0.231 m half-width |
| active window | 35% of clip | 35% of clip |
| kinematic feasibility | **6/6** | 6/6 reachable |
| SONIC accepts it | **2 of 3** valid nominals | 1 of 3 valid nominals |
| how it fails | tracking drift, no contact | collision at low drift |

Locality is not tidiness. A clip crouched from frame zero cannot demonstrate a decision made
from what the robot sees, because the obstacle is not visible when the crouch begins; such a
pair can only support map-conditioned selection.

### The overhead operator clears all three gates

Family `mf_005_c08`. One journey, two behaviours: the two clips have identical root XY frame for
frame, identical duration, and differ in exactly six joints — knees 0.994 rad, hip and ankle pitch
0.497 rad each, waist 0.000 rad.

| | easy scene (1.355 m) | hard scene (1.257 m) |
|---|---|---|
| **nominal walk** | accepted, 0.0 N | **rejected, 3253.5 N** |
| **local crouch** | accepted, 0.0 N | **accepted, 0.0 N** |

Attribution: contact on `torso_link` at frame 90, root-height deficit at frame 97 — the drift
follows the collision by 140 ms. The crouch's drift is 0.100 m/s in *both* scenes, so the shelf
never reaches it.

### The predicted window is optimistic on both sides

> **Note.** Two later families appeared to show a far larger optimism, above 40 mm. They were void:
> their shelves sat in the reference motion's frame while the rollout offset the motion by −2.0 m,
> so the robot never reached them. Those numbers are withdrawn; the figures below come from the one
> family whose obstacle was verifiably in the path.

Probing both boundaries rather than assuming them changed how families must be placed.

| shelf | motion | predicted | observed |
|---|---|---|---|
| 1.2971 m | nominal | fails | **accepted** |
| 1.2801 m | nominal | fails | rejected — 85.0 N overhead, 0.0 lateral |
| 1.2574 m | nominal | fails | rejected |
| 1.2574 m | crouch | clears | accepted |
| 1.2154 m | crouch | clears | **rejected** |

The nominal survives 8 mm below where it is predicted to fail; the crouch fails 8 mm above where it
is predicted to clear. Both errors shrink the usable window: the real one is between **22.7 and
81.7 mm** wide against a predicted 97.7, so in the worst case the prediction over-states it more
than fourfold. It does contain 1.2574 m, verified from both sides.

The two errors have different causes. Swept capsules are conservative outer approximations, so they
should make a motion look taller than it is and predict interference early, which is what happened
to the nominal. The crouch failing *higher* than predicted is not geometric at all: at 1.2154 m the
shelf pressed `torso_link` down at 1017.4 N and the controller lost its reference. A swept volume
knows where the robot went; it does not know that a controller squeezed into a gap stops being able
to track. **So placement targets the window's centre**, and the margin cannot be replaced by a
predicted-boundary offset.

### Measuring this found a gate defect, and corrected three published numbers

`disallowed_robot_contact` tested only the *horizontal* component of external contact, because the
settling load a dropped robot puts on its hips is vertical and had to be excluded. Excluding all
vertical force also excluded every overhead collision. The crouch at 1.2154 m carried 1017.4 N
straight down on `torso_link` with a horizontal component of exactly 0.0, so the gate stayed
silent; only the ensuing drift caught it, meaning a milder jam would have been **accepted** — in
precisely the regime this corpus exists to supply. Sign separates the two cases: the floor holding a
knee up pushes +z, an obstacle overhead pushes −z.

Auditing all 230 evaluable episodes found 10 carrying an overhead push above the lateral one, all
already rejected on other grounds — so the corpus held no false accepts. **That was luck, and the
next rollout proved it:** the 1.2801 m probe is rejected for `disallowed_robot_contact` alone, its
drift below threshold, and regraded with the lateral-only gate it comes out **accepted** — a walk
whose torso is pressed down 85 N, recorded as a clean traversal.

The audit also corrected three earlier claims, all of which had read the smaller component:

- the first family's "3.0 N graze" was a **409.3 N** push on `torso_link`; that negative was never
  marginal
- penetration does **not** track force: 3.3× penetration buys **1.15×** overhead force, not the 18×
  the lateral figures implied, so the graze-versus-crash tuning story is withdrawn
- the start-pose jitter spread is **1.35×** on overhead against 6.9× on lateral, so the force is far
  steadier than reported

### The two operators fail in different ways, and only one is predictable

Cross-motion yield over nominals SONIC accepts. One caveat travels with these numbers: the screen
was run in a *furnished* room rather than on a bare plane, and `x001`'s nominal was excluded on the
strength of a 277.6 N rejection that turns out to be a wall strike — its root passed 0.21 m from the
wall face carrying a force of exactly (0.0, 277.6, 0.0) N. **So the tuck's denominator may be 4, and
a re-screen on a true bare plane is running.** An audit of all 47 graded cells found only x001's two
within 0.35 m of a wall, so no verified family is affected.

| | crouch | arm tuck |
|---|---|---|
| yield | **2 of 3** | 1 of 3 |
| failure mode | reference drift, **zero** external contact | `disallowed_robot_contact` at 0.051 and 0.003 m/s drift |

The distinction matters more than the counts. A collision is governed by geometry, which the
swept-volume machinery already models. A tracking failure is governed by how hard a clip is to
hold — and for the crouch that turns out to be predictable from the clip alone:

| knee excursion | motion | outcome | \|drift\| |
|---|---|---|---|
| 0.929 rad | x002 | accepted | 0.015 m/s |
| 0.936 rad | x003 | accepted | 0.025 m/s |
| 0.994 rad | 005 | accepted | 0.100 m/s |
| 1.000 rad | x000 | **rejected** | 0.225 m/s |

Monotone across four clips and three nominals. The threshold is **motion-specific**, not global: a
within-motion sweep on the failing clip accepts at 0.619 rad and rejects at 0.980, where another
motion holds 0.994. So crouch feasibility is not predictable from motion identity or a universal
cap — it is **cheaply screenable with one targeted rollout**, at exactly the excursion a usable
window requires. This prediction
was **registered in writing before the last two rollouts returned**, after four earlier attempts to
infer trackability from a clip had all been withdrawn; the stated reason for expecting it to hold
this time — that excursion should govern a drift failure where geometry governs a collision — is
what distinguished it. The operator's cap is now set from this measurement rather than from the arm
tuck's unrelated failure.

For the tuck, no such predictor exists. Two were built and refuted, so its yield is a **budget
line** — roughly three rollouts per usable lateral clip, plus one to screen each nominal.

### The lower-body operator cannot pay its own transport cost

The two operators differ on a second axis, and this one bounds which obstacles the method can
address at all. `local_crouch` lowers the reference root by about 0.14 m and leaves the forward
schedule untouched: the adapted reference still commands the same displacement over the same four
seconds. A crouched G1 cannot walk that fast, so it arrives short, and the acceptance gate reads the
shortfall as a tracking failure.

Endpoint lag rises monotonically with crouch depth:

<!-- generated:transport-table -->
| clip | endpoint lag | verdict |
|---|---|---|
| `w_nominal` | 0.257 m | accepted |
| `w_tuckcap30` | 0.218 m | rejected on contact, not tracking |
| `w_crouch05` | 0.313 m | rejected |
| `w_crouch08` | 0.399 m | rejected |
| `w_crouch11` | 0.509 m | rejected |
<!-- /generated:transport-table -->

The decisive row is the first. **The nominal already spends 0.257 m of the 0.35 m budget**, leaving
roughly 90 mm for an adaptation to consume, and every crouch tested deeper than about 5 cm exceeds
it. The arm tuck has the opposite sign — it tracks *better* than the nominal it modifies — which is
the same asymmetry survival measures from the other direction.

The consequence is structural rather than incidental. Only lowering the robot relieves a ceiling, so
overhead-band families require a crouch, and a crouch deep enough to clear a ceiling costs more
forward progress than the gate allows. The first completed banded family shows exactly this: its
nominal walks the torso into the ceiling at 1543.6 N and its adapted motion clears the same ceiling
at 49 N, and the adapted cells are still rejected — for missing the endpoint by 0.524 m.

Two repairs are available and neither is adopted here. Retiming the adapted reference so a crouched
robot is asked for a crouched pace would make the reference self-consistent, but it changes the
journey the counterfactual holds fixed. Judging adapted clips on progress ratio rather than endpoint
error would admit them, but it is a gate change and belongs in the pre-registration rather than in
whichever patch happens to rescue the result. The limitation is reported as a limitation.

### The lateral operator's yield is a measured cost

The arm tuck is accepted on **1 of 3** valid nominals. Two cheap predictors of *which* one were
built and both refuted: wrist-to-hip clearance (already negative on every nominal, and its change
anti-correlates with the verdict at both extremes) and lateral CoM excursion (two cells 0.1 mm
apart landing on opposite sides of the gate). The verdict is set entirely by **external** contact;
self-contact magnitude is not severity — the accepted tuck carries the highest
self-contact of eight cells, 192.1 N, and the rejected nominal the lowest, 10.9 N.

So trackability is measured per clip, and the 1-in-3 yield is a budget line — roughly three
rollouts per usable lateral clip, plus one to screen each nominal — not a defect awaiting a fix.
This is the fourth time on this operator that an inferred quantity had to be withdrawn in favour of
a rollout.

## A tracking controller is a feasibility oracle, and that is a second contribution

Synthetic humanoid datasets report a *success rate*. The number is almost always a conflation of
four different questions, which have different answers, different costs, and different failure
modes. Separating them is cheap, and the gaps between them are where a corpus overstates itself.

| label | question | cost | measured here |
|---|---|---|---|
| `embodiment_feasible` | can the body reach this pose at all? | CPU, milliseconds | **96%** of 150 clips |
| `controller_trackable` | can the controller execute it? | one rollout, empty scene | **96%** of 47 joined |
| `semantically_valid` | does the executed motion contain the named behaviour? | predicate on the execution | **32%** of 75 |
| `scene_compatible` | does it succeed in *this* room? | one rollout per scene | the counterfactual signal itself |

The four are not interchangeable. A clip can be kinematically fine and untrackable; trackable and
semantically empty; semantically correct and infeasible in the room it is needed for. Reporting one
rate hides which surface a corpus is failing on, and the honest headline is the *smallest* of them —
here 32%, not 96%.

**The tracking controller is doing evaluation work, not just execution work.** SONIC answers
"is this motion physically executable by this embodiment under closed-loop control", which is
exactly the question a kinematic checker cannot answer and a human reviewer answers slowly and
inconsistently. Used deliberately it is a *feasibility oracle*: cheap relative to human review,
reproducible, and grounded in the same physics the dataset claims to be about.

It also grades continuously, not just pass/fail. Reference-tracking drift rises monotonically with
how far an adaptation moves a joint — measured within one motion at 0.048, 0.104, 0.140, 0.223 and
0.224 m/s for knee excursions of 0.000, 0.420, 0.619, 0.980 and 1.000 rad — so drift is a graded
difficulty score, not merely a threshold. That makes it usable for curriculum ordering and for
reporting *how hard* a clip is rather than only whether it survived.

### It also says how much of an edit actually happened

A clip named `crouch18` asserts a crouch. What the corpus can honestly claim is that a crouch was
*commanded*; the frozen controller decides how much of it occurs. The difference is measurable from
artefacts every rollout already writes, because `reference_g1_qpos` and `dof_pos` land on the same
frame grid:

> survival = mean |executed_adapted − executed_nominal| ÷ mean |reference_adapted − reference_nominal|

taken over the joints the operator moves and the frames it is active. Over 23 adapted clips with
matched nominals, survival separates by body region rather than by how much was asked for:

<!-- generated:survival-table -->
| operator | n | survival | median |
|---|---|---|---|
| crouch (lower body) | 11 | 46–67% | **58%** |
| combo (hip + waist) | 2 | 65–65% | 65% |
| tuck (upper body) | 10 | 57–116% | **89%** |
<!-- /generated:survival-table -->

At matched commanded amplitude near 0.3 rad the groups diverge by a factor of two — `x000_crouch015`
survives at 46% and `x000_crouch030` at 51%, against 105%, 91% and 88% for three tucks commanding
0.28–0.33 rad. **The controller preserves arm departures and resists leg departures**, which is the
expected shape for a policy whose legs carry the load and hold balance, and which had not previously
been measured. The arm's fidelity has its own ceiling: both uncapped large tucks, commanding 0.741
and 0.869 rad, fall back to 57%.

Two rollouts of one journey drift apart on their own, so every ratio is reported against a drift
floor measured on the joints the operator never touches; the operator signal stands 4.7×–13.6× clear
of it. One clip, `w_tuck06win18`, commands only 0.072 rad and survives at 61% where the amplitude
trend predicts near-total survival; it carries the thinnest margin in its group and is recorded
rather than explained.

This matters for what a label may claim. A lower-body label overstates the executed departure by
roughly 40%, so the release carries executed amplitude beside commanded amplitude rather than
letting a consumer inherit the label's assertion. It also gives semantic validity a numeric partner:
survival separates a clip whose behaviour the controller discarded from a clip whose behaviour is
present and merely hard to see — a distinction human review cannot make from a contact sheet, and
one that has already corrected a reviewer's reading in this project.

## Generated motion is not free, and it fails where the dataset needs it most

Joining the generator's own kinematic screen to the controller's verdicts on the same clips gives a
pipeline that looks healthy: 96% embodiment-feasible, 96% of the joined subset trackable. Broken
down by behaviour it is not.

| behaviour | generated | embodiment-feasible | tracked |
|---|---|---|---|
| **crouch** | 9 | **4 (44%)** | 2 |
| side / narrow | 3 | 2 (67%) | 2 |
| arms | 6 | 6 (100%) | 1 |
| turn | 67 | 67 (100%) | 33 |
| stop / start | 23 | 23 (100%) | 5 |
| walk | 42 | 42 (100%) | 2 of 4 |

Crouch is the **only** behaviour with a kinematic failure rate at all — 44% against 100% everywhere
else — and every rejection is joint saturation. The mechanism is specific: the generator lowers the
torso by folding the waist, pinning `waist_pitch_joint` at its limit on 100% of frames, and the G1
cannot fold that far. Meanwhile 73% of the corpus is unconstrained locomotion, 67 turn variants and
42 plain walks, which is precisely the diversity that does not help. The behaviours a counterfactual
needs are 12 of 150, and they are the ones that fail.

Tracking quality agrees independently. Over 238 evaluable episodes, crouch carries a median
reference drift of **0.100 m/s** against a turn's 0.026 — roughly four times — and is accepted 13 of
21 against 49 of 50. The two stages could easily have disagreed: a generator can emit clips that are
kinematically awkward but easy to follow, or smooth ones impossible to balance. Here the difficulty
is intrinsic to the behaviour rather than to either component, which is the more useful finding,
because no amount of prompt engineering or controller tuning removes it.

This is the quantitative case for constructing adapted motions with a **local operator** rather than
requesting them from a generator. An operator applied to an already-accepted walk inherits that
walk's feasibility and changes only what one obstacle requires, which is why the adapted clip is
matched to its nominal by construction rather than paired after the fact.

*Scope: one checkpoint, one set of prompt templates, 4-second clips, 47 of 150 clips joined because
only part of the corpus has been rolled out. This closes an engineering path under a tested
protocol; it is not a claim about what the generator can do.*

## Scene difficulty is a design parameter, not a search outcome

Every scene here was originally built by lowering a shelf until something touched. That finds a
boundary but cannot say *what* is being tested, and a full-height shelf can only bind the tallest
capsule — `torso_link`, on 199 of 199 frames of a walk. Every overhead scene tested the same part.

Because the executed trajectory gives every collision capsule's pose per frame, an obstacle confined
to a height *band* binds whatever passes through that band. On one verified nominal:

A band admits two obstacles, stopped by different parts: a **wall** from the side meets whatever
reaches furthest sideways, a **ceiling** from above meets the highest point in the band.

| band | wall binds | reach | ceiling binds | height |
|---|---|---|---|---|
| overhead 1.15–1.60 m | `torso_link` | 0.079–0.097 m | `torso_link` | 1.301 m |
| chest 0.85–1.15 m | `elbow_link` | 0.235–0.252 m | `torso_link` | 1.296 m |
| waist 0.55–0.85 m | `wrist_yaw_link` | 0.284–0.327 m | `shoulder_yaw_link` | 1.044 m |
| knee 0.25–0.55 m | `wrist_yaw_link` | 0.283–0.304 m | `hip_roll_link` | 0.701 m |
| floor 0.00–0.25 m | `ankle_roll_link` | 0.245 m | `knee_link` | 0.423 m |

**Eight wall configurations are constructible**, binding three distinct parts, plus two ceiling ones;
the rest meet a shoulder, hip, knee or ankle that no operator relieves. Left and right reaches differ by up to 43 mm because the arms swing out of phase, so
the two sides pose genuinely different problems rather than mirrored ones.

Difficulty then follows from `obstacle face = reach + margin`, so a margin ladder on one
configuration yields a graded series — clear, near-threshold, marginal, infeasible — all with the
same binding part, and therefore comparable. This also changes the cost of scale: 24–30 families is
not 24–30 motion pairs but roughly four screened nominals at eight configurations each.

A binding part is **not** a counterfactual. Nothing relieves the ankle, so floor obstacles produce a
negative with no matching positive; the map marks that before rollouts are spent rather than after.

## The decisive experiment (not yet run)

Three datasets, identical in motions, scene count, rendering budget, learner and training
steps. Only the **construction** differs:

- **A, decorated** — one trajectory through many scenes. The failure mode being repaired.
- **B, random obstacles** — more visual diversity, obstacles placed without regard to which
  behaviours they separate.
- **C, SweepCF** — same task, same start and goal, geometry chosen so the preferred feasible
  behaviour reverses.

Train a small **behaviour selector**, not a policy — and score scene–motion *compatibility*
rather than scene identity, so a model cannot pass by learning "room 7 wants the crouch".
Selection is the lexicographic rule: among candidates predicted to survive, take the cheapest.

The harness and its controls are built and validated on synthetic families, so no verified family
is spent proving the apparatus works. The privileged geometry model reaches 0.963 choice accuracy
against a 0.25 always-nominal baseline; hiding the scene drops it to 0.126, pairing families with
the wrong scene to 0.593, and permuting candidate order changes nothing at all, as an
order-invariant rule must. Holding out nominal motions rather than families gives 0.958, so the
model is not memorising what each nominal usually needs.

One negative result from that validation is worth reporting: a linear model given only the four
geometric margins scored 0.224 — *below* the baseline, choosing a failing candidate 47% of the
time — because survival is an AND over margins and that is not linearly separable. The limiting
margin is therefore supplied explicitly, which is what makes this model privileged and bounds what
an ego-depth model must recover from pixels.
Evaluate on the **frozen scene-first test set** — 30 scenes sampled independently of any
motion, SHA-256 fingerprinted, frozen before either operator existed.

Metrics: counterfactual choice accuracy (walk in the easy scene *and* crouch in the hard one),
unsafe-choice rate, **unnecessary-adaptation rate**, and realised success minus adaptation cost.

The last two matter because a policy that always crouches is safe and has understood nothing.

**Go/no-go.** If C beats A and B on counterfactual choice accuracy, scale the motion bank. If
all three are equal, the question becomes *when does counterfactual supervision add anything
beyond geometric collision reasoning* — which is still a paper, and a more interesting one.

## Anticipating the three objections

**Why n = 30 suffices.** The statistical unit is the scene, and thirty scene-first scenes were
frozen before any operator existed. Exact Clopper–Pearson intervals at n = 30 are wide but not
uninformative: 24/30 gives [0.61, 0.92] and 27/30 gives [0.73, 0.98]. A one-sided exact test rejects
0.50 at ≥20/30 with power 0.97 against 0.80 and 0.73 against 0.70. So a 0.30 difference is
comfortably detectable and a 0.20 difference is not, which is why the pre-registration commits in
advance to claiming nothing below about 0.25. Intervals are shown on every reported proportion
rather than point estimates alone.

**Why a selector rather than a policy.** Because the controller is frozen, deterministic under a
fixed configuration, and does not learn, any difference in outcome between two candidate choices is
attributable to **the choice** rather than to control. A policy that both perceives and acts improves
for reasons its authors cannot decompose; here control is held constant by construction, which is
what makes *does counterfactual supervision improve selection* a question with a clean answer. The
cost is that this work claims nothing about continuous visuomotor control, and says so.

**Why the random-obstacle baseline is fair.** Dataset B draws obstacles from the same parameter
support as C and receives **honest labels**: predictions from the swept-volume model, with mandatory
physics audit of any scene whose predicted margin falls within ±25 mm of a boundary — the calibrated
error of that predictor on the one family where both boundaries were bracketed. Predictor–physics
agreement is reported on the audited slice, so the reader can see how much of B's labelling is
trusted rather than measured. Example count, motion identity and rendering budget are equalised
across A, B and C, and **informative-label density is reported per dataset** rather than assumed
equal — a comparison that wins only because one arm has more informative examples has explained
nothing.

## What this draft deliberately leaves out

- The contact-attribution detail beyond the one number that ties geometry to physics
  (predicted 112, observed 113).
- The prompt-generation audit as a headline; it belongs as a motion-source ablation.
- The adaptation cost as a reportable ratio. It is a sign check until its weights are frozen,
  and an operator that minimises a cost cannot use that cost to prove it is minimal.
- Any claim that the robot perceives the scene and chooses. Both motions are prescribed
  references; what changes is which one physics permits.

## Figure 1, as intended

Left: three rooms, one trajectory — *scene changes, behaviour label does not.*
Centre: the 2×2, with one cell failing.
Right: success probability against shelf height for both behaviours, the counterfactual window
between the two boundaries, and `b*(g)` stepping from walk to crouch.
