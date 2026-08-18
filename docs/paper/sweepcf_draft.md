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

| # | claim | status |
|---|---|---|
| 1 | Two executable motions' full-body swept volumes separate, so a geometric window exists | **established** |
| 2 | The same controller succeeds or fails as that geometry changes | **established** (one family, verified) |
| 3 | The 2×2 *outcome* survives a jittered start pose | **established** (12/12 cells, 3/3 jitters) |
| 4 | The scene *reverses the preference* between two feasible behaviours | **partial** — the reversal is demonstrated; the cost defining "preferred" is not calibrated |
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

Cross-motion yield over nominals SONIC accepts on a bare plane — `x001` is excluded because its own
nominal is rejected at 277.6 N on `left_knee_link`, so adapting it never tested an operator:

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

Monotone across four clips and three nominals, with the boundary in a 6 mrad gap. This prediction
was **registered in writing before the last two rollouts returned**, after four earlier attempts to
infer trackability from a clip had all been withdrawn; the stated reason for expecting it to hold
this time — that excursion should govern a drift failure where geometry governs a collision — is
what distinguished it. The operator's cap is now set from this measurement rather than from the arm
tuck's unrelated failure.

For the tuck, no such predictor exists. Two were built and refuted, so its yield is a **budget
line** — roughly three rollouts per usable lateral clip, plus one to screen each nominal.

### The lateral operator's yield is a measured cost

The arm tuck is accepted on **1 of 3** valid nominals. Two cheap predictors of *which* one were
built and both refuted: wrist-to-hip clearance (already negative on every nominal, and its change
anti-correlates with the verdict at both extremes) and lateral CoM excursion (two cells 0.1 mm
apart landing on opposite sides of the gate). On bare-plane rollouts the verdict is set entirely by
**external** contact; self-contact magnitude is not severity — the accepted tuck carries the highest
self-contact of eight cells, 192.1 N, and the rejected nominal the lowest, 10.9 N.

So trackability is measured per clip, and the 1-in-3 yield is a budget line — roughly three
rollouts per usable lateral clip, plus one to screen each nominal — not a defect awaiting a fix.
This is the fourth time on this operator that an inferred quantity had to be withdrawn in favour of
a rollout.

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
