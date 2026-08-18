# Designing scene difficulty instead of discovering it

Every scene so far was built by lowering a shelf until something touched. That finds *a* boundary,
but it has two limits worth naming: it cannot say **what** is being tested, and a full-height shelf
can only ever bind the tallest part of the robot. On a plain walk that is `torso_link`, on 199 of
199 frames. Every overhead scene we have built tests the same body part.

Because the executed trajectory gives every collision capsule's pose per frame, difficulty can
instead be **chosen**, along two axes: *which body part* an obstacle binds, and *how much margin* it
leaves.

## An obstacle in a height band binds whatever passes through it

Measured on one verified nominal walk:

| band | side | binding part | reach from root | operator that relieves it | family? |
|---|---|---|---|---|---|
| overhead 1.15–1.60 | left | `torso_link` | 0.097 m | `local_crouch` | yes |
| overhead | right | `torso_link` | 0.079 m | `local_crouch` | yes |
| chest 0.85–1.15 | left | `left_elbow_link` | 0.252 m | `local_arm_tuck` | yes |
| chest | right | `right_elbow_link` | 0.235 m | `local_arm_tuck` | yes |
| waist 0.55–0.85 | left | `left_wrist_yaw_link` | 0.327 m | `local_arm_tuck` | yes |
| waist | right | `right_wrist_yaw_link` | 0.284 m | `local_arm_tuck` | yes |
| knee 0.25–0.55 | left | `left_wrist_yaw_link` | 0.304 m | `local_arm_tuck` | yes |
| knee | right | `right_wrist_yaw_link` | 0.283 m | `local_arm_tuck` | yes |
| floor 0.00–0.25 | left | `left_ankle_roll_link` | 0.245 m | **none** | **no** |
| floor | right | `right_ankle_roll_link` | 0.245 m | **none** | **no** |

**Four distinct body parts from one walk** — torso, elbow, wrist, ankle — and eight of the ten
configurations have an operator that can answer them. The reach values differ left to right by up to
43 mm because arms swing out of phase, so the two sides are genuinely different problems rather than
mirror images.

## Why this is the answer to the scale problem

Reaching 24–30 verified families has been costed as 24–30 separate motion pairs. It is not. Each
nominal motion supports **eight constructible configurations**, so four screened nominals cover the
target with margin, and every configuration inherits a nominal that is already known to track.

The margin is then a parameter rather than a search:

```
obstacle face = reach + margin
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
