# Guidance, 2026-08-18: stop generating clips, start reversing preferences

Saved verbatim in intent from the review of `cf_005_056`. The one-line instruction:

> **Do not blindly generate more humanoid clips. Generate a stable set of whole-body motion
> options, then automatically construct counterfactual families in which the scene changes
> the ranking of those options, and finally train the robot to select the minimal but
> necessary adaptation from vision.**

## The claim ladder now has six rungs

| Level | Claim | Status |
|---|---|---|
| 1 | Two executable motions' full-body swept volumes differ | established |
| 2 | A geometry window exists where nominal fails and adapted succeeds | established |
| 3 | That prediction produces the expected 2×2 in re-run physics | established |
| 4 | The 2×2 outcome survives small start-pose jitter | established |
| 5 | The robot *selects* a suitable motion from scene observation | **not established** |
| 6 | That selection generalises to scenes not fitted to a trajectory | **not established** |

## Three corrections to how results are worded

**"Robust" is too broad.** What was established is **start-pose outcome robustness**, not
dynamics robustness and not collision-severity robustness — peak force on the failing cell
ranged 95.5–658.3 N across jitters. *Whether* it collides is stable; *how hard* is not. Say
`outcome-robust counterfactual`.

**Force is not one number.** `first_contact_force_n`, `peak_contact_force_n`,
`contact_impulse_ns` and contact duration are four different quantities and must be named
separately. Reporting 55.4 N and 137.2 N without naming them reads as a contradiction.

**Every height needs a body and a frame.** "0.849 m" means nothing until it says which body,
measured how, in which frame.

## The deeper result: scene-induced preference reversal

The 2×2 is more than a matched negative. In the easy scene both motions succeed, but the walk
is the better choice — lower cost, less deviation from a nominal gait. In the hard scene only
the duck survives. So the environment **reverses the ranking**:

```
S_easy :  walk  ≻  duck
S_hard :  duck  ≻  walk
```

That makes the research question a decision problem rather than a classification:

```
m* = argmin_m  C(m)     subject to   P(success | S, m) ≥ τ
```

with `C(m)` covering deviation from nominal gait, pelvis lowering, joint travel, an
energy proxy, duration, tracking difficulty. A robot should not crouch in an open room; it
should crouch when the shelf requires it. **Scene-conditioned minimal adaptation.**

## Obstacle position is part of the algorithm, not a scene parameter

Moving the shelf from the route midpoint to the station of maximum envelope separation took
the window from 53 mm to 178 mm. Formalised as **pairwise discriminative envelope mining**:
align two task-compatible motions by route progress `s`, and for body group `g`

```
x*  =  argmax_x [ d_g(m_a, x) − d_g(m_n, x) ]
```

then search at `x*` for an interval where `d_g(m_n) < −δ_f` and `d_g(m_a) > +δ_s`.

**Do not pick the hard margin by targeting a force.** Force is sensitive to centimetre-scale
perturbation. Choose the *minimum sufficient* intervention: the smallest hard parameter such
that, over registered perturbations, the nominal fails with probability ≥ 1−ε and the adapted
succeeds with probability ≥ 1−ε.

## The generator is a mode producer, not an event scheduler

The FK screen over 150 prompts shows a clean split: whole-clip styles come back reliably,
events at a specified moment do not. Build a **whole-route motion mode bank** and let geometry
select between modes, rather than asking text to schedule adaptation in time:

| regime | nominal mode | adapted mode |
|---|---|---|
| overhead | normal walk | knee-driven crouch walk |
| lateral arm clearance | natural arm swing | arm-tucked walk |
| narrow body passage | forward-facing walk | shoulder-turned walk |
| narrow route orientation | forward walk | side-step |
| floor obstacle | normal gait | persistent high-knee gait |

Mid-clip `walk → duck → recover` is a later extension via explicit keyframes, foot/hand
constraints, or segment stitching — not more free-form text.

## Four failure stages, never one accepted/rejected flag

| stage | question | current example |
|---|---|---|
| prompt → reference | does the reference express the prompt | `step_over`: no |
| reference → embodiment | joint limits, self-collision, proportions | `crouch_walk`: waist saturated |
| reference → execution | does the tracker preserve the semantics | drift, pose tracking |
| motion × scene | is the execution compatible with geometry | `walk × hard shelf` |

Each motion needs `reference_semantic_valid`, `embodiment_feasible`, `physics_trackable`,
`scene_compatible` as separate fields.

## Headline metrics change

Stop leading with episode count, scene count and acceptance rate. Lead with: verified
counterfactual families, unique motion pairs, scene-induced preference reversals,
discriminative-window width, physics verification yield, intended-failure purity, outcome
robustness rate, and **unnecessary-adaptation rate**.

## Splits

- **Train** — motion-fitted counterfactual families.
- **Validation** — held-out fitted families, split at family level by motion seed, route,
  obstacle template and appearance. Model selection only, never generalisation evidence.
- **Test** — **scene-first** families: sample the scene parameters independently, *freeze
  them before training*, then run every candidate motion to obtain physics labels. Nothing
  about a test scene may be optimised against a test motion's swept volume.

## The learning benchmark to build first

Not a large VLA. A **scene–motion compatibility and minimal-adaptation ranker**: from ego
RGB-D, robot state, goal and a candidate motion, predict success probability, minimum
clearance, failure body and failure time, plus adaptation cost.

Baselines that make it meaningful: no-scene, fixed-cylinder heuristic, analytic full-body
oracle with privileged geometry, RGB-D learned scorer, and a **scene-shuffled control**.

Metrics: selected-motion task success, false-safe rate, unnecessary-adaptation rate, cost
regret against the oracle, contact-body accuracy, contact-time error, calibration, and
**preference-reversal accuracy** — walk in the easy scene, duck in the hard one.

## Family count gates

| gate | target |
|---|---|
| method pilot | 4 verified families per regime, 12 total |
| learning pilot | 10–12 per regime, 30–36 total |
| strong corpus | 60+, across multiple motion seeds and route variants |

Never inflate the headline by expanding one motion pair across thirty near-identical shelf
scenes. Report unique motion pairs, unique route families and unique obstacle templates
alongside the family count.

## Execution order

- **P0.5** — clean up the first artifact: name the force quantities, say "start-pose
  outcome-robust", state the rollout-count and prompt-count denominators, name the body and
  frame for every height, and freeze `cf_005_056` as a checked-in fixture that tests cannot
  overwrite.
- **P1** — a cheap reference prefilter: semantic predicate, joint-saturation fraction,
  joint-limit margin, root-path compatibility, body-envelope profile, self-collision
  estimate, functional separation from nominal. No GPU until all pass.
- **P2** — the pairwise station optimiser over motion pair, body group, obstacle station,
  obstacle parameter and hard margin — with random-station, midpoint-station and
  maximal-separation ablations.
- **P3** — scale physics verification, reporting contact-time MAE, failure-body accuracy,
  verification yield, outcome-robust rate and severity variance.
- **P4** — freeze the scene-first test set *before* training anything.
- **P5** — train the minimal-adaptation selector: privileged geometry first, then RGB-D.

## Paper shape

1. Pairwise discriminative geometry mining.
2. Physics-verified counterfactual motion families.
3. Scene-conditioned minimal-adaptation selection.

Working title: **SweepCF: Physics-Verified Counterfactual Motion Families for
Scene-Conditioned Humanoid Traversal**.
