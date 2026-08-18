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
| waist change | **0.0000 rad** | untouched |
| leg change | — | **0.0000 rad** |
| effect at the station | 1.257 → 1.089 m silhouette | 0.276 → 0.231 m half-width |
| active window | 35% of clip | 35% of clip |
| kinematic feasibility | **6/6** | 6/6 reachable |

Locality is not tidiness. A clip crouched from frame zero cannot demonstrate a decision made
from what the robot sees, because the obstacle is not visible when the crouch begins; such a
pair can only support map-conditioned selection.

*Status: stage 1 (kinematic) passes. Stage 2 (trackability) is queued. No hard scene has been
built for these, deliberately.*

## The decisive experiment (not yet run)

Three datasets, identical in motions, scene count, rendering budget, learner and training
steps. Only the **construction** differs:

- **A, decorated** — one trajectory through many scenes. The failure mode being repaired.
- **B, random obstacles** — more visual diversity, obstacles placed without regard to which
  behaviours they separate.
- **C, SweepCF** — same task, same start and goal, geometry chosen so the preferred feasible
  behaviour reverses.

Train a small **behaviour selector**, not a policy: scene → which of {walk, crouch, tuck, …}.
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
