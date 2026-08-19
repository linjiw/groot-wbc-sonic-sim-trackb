# Pre-registration — perceptive whole-body traversal

Committed **before any physics label exists for the frozen scene-first test set**. Nothing below may
be revised after labels are produced; revisions must be added as dated amendments that state what
changed and why, and any result reported under an amended plan must say so.

The task is **perceptive whole-body traversal** throughout. It is not navigation, and no claim of
continuous visuomotor control is made: the controller is frozen and the learner selects among
prescribed references. If the hardware stretch lands, the ladder's top row rises only to
*perception-conditioned selection*.

## Hypotheses

**R1 (claims 5–6).** A learner trained on counterfactual supervision uses scene geometry when it
could ignore it, and that behaviour transfers to scenes never fitted to any trajectory.

**R2.** The same learner, given ego pixels or depth instead of privileged geometry, selects
references that succeed closed-loop in physics.

R2 is R1's learner with a different input and a different readout — one experimental frame, two
readings — not a separate workstream.

## Primary and co-primary metrics

**Primary — counterfactual choice accuracy.** The fraction of test scenes where the selector picks
exactly the minimum-edit feasible reference under the lexicographic rule
`m*(S) = argmin_m D(m, m₀) s.t. y(S,m) = 1`, with `D(m₀,m₀) = 0`.

**Co-primary — closed-loop traversal success.** The fraction of test scenes where the *chosen*
reference is accepted by the full gate set when executed in that scene.

## Decision rule

Dataset **C (SweepCF)** wins if it beats **both** A (decorated) and B (random obstacles) on the
primary metric, with exact binomial confidence intervals over **n = 30** scenes, at **≥10 seeds**.
The statistical unit is the scene, never the frame.

* **C wins** — both CIs exclude the comparator's point estimate.
* **Ambiguous zone** — declared in advance as any outcome where C's CI overlaps either comparator's
  point estimate while C's point estimate is higher. Only in this case is the second 30-scene batch
  labelled, and both vintages are then disclosed with their separate results.
* **Tie or loss** — reported as such, under the framing *when does counterfactual supervision exceed
  geometric collision reasoning*.

**Minimum detectable effect.** At n = 30 and α = 0.05, an exact binomial test distinguishes 0.50
from 0.80 with power ≈ 0.85. Differences smaller than about 0.25 in proportion are not detectable at
this n and will not be claimed, whatever the point estimates show.

## Baselines, all reported

| baseline | what it tests |
|---|---|
| always-nominal | never adapt; scores the easy-scene fraction |
| always-adapt | safe and scene-blind |
| random choice | chance |
| blind (no scene input) | must collapse to always-nominal level |
| privileged geometry oracle | the ceiling perception must be measured against |

**Expected structure, stated in advance:** dataset A's selector sees only nominal labels, so its
closed-loop success should approximate the easy-scene fraction of the test set. Observing that
collapse is a prediction, not a post-hoc explanation.

## Selector inputs

Four, all reported: privileged geometry (the 0.963 synthetic ceiling), ground-truth ego depth, ego
RGB, and a blind control. Frozen DINOv2/SigLIP features with a small MLP scoring scene–motion
survival; lexicographic minimum-edit selection.

**If RGB fails while depth succeeds, that is a result and will be reported as one** — *geometry is
learnable from depth; RGB needs visual realism* — not a failure to be tuned away.

## Also reported, whatever they show

Unsafe-choice rate, unnecessary-adaptation rate, abstention rate, cost regret, scene contact force,
and scene-shuffle degradation. Counts are reported separately for mined pairs, operator-built
families, and scene instances; five shelf heights on one nominal are never five families.

## The shortcut-free property

Decision frames are drawn from the approach segment, with the obstacle in view and at least 30
frames before predicted contact. Pre-window reference frames are **identical across labels by
construction**, so the only label-discriminating signal in the observation is the scene itself. A
learner cannot pass by reading the motion.

## Second test batch

A second 30-scene batch is generated and SHA-fingerprinted at the same time as this document, and
remains unlabelled. It is labelled only if the first result falls in the ambiguous zone declared
above. Both vintages are disclosed in any reported result.

## Validity conditions

Every rollout must pass `scene_route_check.py`; a scene the robot never reaches measures nothing
whatever its outcome reads. Success is explicit markers plus expected artifacts, never exit codes.
No null conclusion is drawn without a sweep of at least three points spanning the range.

---

## Amendment 1 — 2026-08-19, before any label

**What changed.** The stated power for the primary comparison. The original text read *"an exact
binomial test distinguishes 0.50 from 0.80 with power ≈ 0.85"*. Computing the Clopper–Pearson
intervals and the exact one-sided test gives **0.97**, not 0.85.

**Why.** The original figure was an estimate written from memory rather than computed. It understated
the test's power, which is the conservative direction, but a pre-registration that carries an
uncomputed number is not doing its job.

**The corrected numbers**, exact and reproducible:

| successes / 30 | rate | 95% CI | width |
|---|---|---|---|
| 15 | 0.50 | [0.31, 0.69] | 0.37 |
| 21 | 0.70 | [0.51, 0.85] | 0.35 |
| 24 | 0.80 | [0.61, 0.92] | 0.31 |
| 27 | 0.90 | [0.73, 0.98] | 0.24 |

One-sided at α = 0.05: reject H₀ = 0.50 at **≥20/30**; power against 0.80 is **0.97**, against 0.70
is **0.73**.

**What does not change.** The minimum detectable effect stands as originally registered: a 0.30
difference is detectable at 0.97, a 0.20 difference only at 0.73 — below the 0.80 convention — so
the threshold sits near 0.25 and differences below it will not be claimed. The decision rule, the
n, the seed count and the ambiguous zone are all unchanged.
