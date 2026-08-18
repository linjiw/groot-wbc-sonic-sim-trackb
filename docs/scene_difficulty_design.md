# Designing scene difficulty instead of discovering it

Every scene so far was built by lowering a shelf until something touched. That finds *a* boundary,
but it has two limits worth naming: it cannot say **what** is being tested, and a full-height shelf
can only ever bind the tallest part of the robot. On a plain walk that is `torso_link`, on 199 of
199 frames. Every overhead scene we have built tests the same body part.

Because the executed trajectory gives every collision capsule's pose per frame, difficulty can
instead be **chosen**, along two axes: *which body part* an obstacle binds, and *how much margin* it
leaves.

## Two obstacle types, bound by different parts

A band admits two obstacles and they are stopped by different things. A **wall** — a rack, a cabinet
edge, a door frame — comes from the side and is stopped by whatever reaches furthest sideways within
the band. A **ceiling** — a shelf, a beam, a lintel — descends and is stopped by the highest point
in the band, usually a different capsule entirely.

Measured on one verified nominal walk:

| band | side | wall binds | reach | family? | ceiling binds | height | family? |
|---|---|---|---|---|---|---|---|
| overhead 1.15–1.60 | left | `torso_link` | 0.097 | yes | `torso_link` | 1.301 | yes |
| overhead | right | `torso_link` | 0.079 | yes | `torso_link` | 1.301 | yes |
| chest 0.85–1.15 | left | `left_elbow_link` | 0.252 | yes | `torso_link` | 1.296 | yes |
| chest | right | `right_elbow_link` | 0.235 | yes | `torso_link` | 1.296 | yes |
| waist 0.55–0.85 | left | `left_wrist_yaw_link` | 0.327 | yes | `right_shoulder_yaw_link` | 1.044 | **no** |
| waist | right | `right_wrist_yaw_link` | 0.284 | yes | `right_shoulder_yaw_link` | 1.044 | **no** |
| knee 0.25–0.55 | left | `left_wrist_yaw_link` | 0.304 | yes | `right_hip_roll_link` | 0.701 | **no** |
| knee | right | `right_wrist_yaw_link` | 0.283 | yes | `right_hip_roll_link` | 0.701 | **no** |
| floor 0.00–0.25 | left | `left_ankle_roll_link` | 0.245 | **no** | `right_knee_link` | 0.423 | **no** |
| floor | right | `right_ankle_roll_link` | 0.245 | **no** | `right_knee_link` | 0.423 | **no** |

**Eight wall configurations are constructible**, binding three distinct parts — torso, elbow, wrist.
Ceilings are constructible only in the two upper bands, because lower down they meet a shoulder, hip
or knee and no operator relieves any of those.

Left and right reaches differ by up to 43 mm because the arms swing out of phase, so the two sides
pose genuinely different problems rather than mirrored ones.

> **Correction.** The first version of this map reported a single reach per band and gave the
> overhead row 0.097 m — the torso's *sideways* extent, which a ceiling never touches. The overhead
> ceiling constraint is the peak height, 1.301 m. One number cannot serve both obstacle types, and
> conflating them would have placed shelves a metre from where they were meant to go.

## Why this is the answer to the scale problem

Reaching 24–30 verified families has been costed as 24–30 separate motion pairs. It is not. Each
nominal motion supports **eight constructible configurations**, so four screened nominals cover the
target with margin, and every configuration inherits a nominal that is already known to track.

The margin is then a parameter rather than a search:

```
wall face   = lateral reach  + margin
ceiling face = vertical reach + margin
```

A ladder of margins on one configuration produces a graded difficulty series — comfortably clear,
near-threshold, marginal, infeasible — all with the *same* binding part, which is what makes them
comparable. That is a far better structure for a curriculum than a set of unrelated shelves.

## What this does not do

**A binding part is not a counterfactual.** A family needs an adaptation that relieves the part the
obstacle binds. The crouch relieves the torso; the arm tuck relieves wrists and elbows. Nothing
relieves the ankle, so floor obstacles produce a negative with no matching positive — a scene, not a
family. The table above marks that explicitly rather than leaving it to be discovered after the
rollouts are spent, and it is consistent with the standing instruction not to pursue floor
adaptation.

**Geometry still only proposes.** The reach is computed from the executed trajectory, so it is exact
for *that* execution. Whether the controller still tracks when an obstacle is 20 mm away is a
physics question, and the measured window calibration says the geometric prediction is accurate to
about 8 mm at the reference and 2 mm at the execution — good, but not zero, and not a substitute for
the rollout.

**One motion's map is not universal.** These reaches are for one walk. A motion with a wider arm
swing moves the wrist bands; a faster gait changes which frames occupy which band. The map is per
trajectory and cheap to recompute, which is the point.

## Where this leads

The same structure is what a navigation or traversal dataset needs, and it is why this is worth
building now rather than after the paper. A scene labelled *"the binding constraint is the left
elbow at 30 mm of margin"* carries far more than *"there is a shelf here"*: it states which part of
the body the geometry is about, how close it is, and which behaviour would resolve it. That is a
usable supervision target for a policy that must decide **what to change**, not merely whether an
obstacle exists — and it extends to manipulation-adjacent traversal without changing the machinery,
because the map does not care why a capsule is where it is.
