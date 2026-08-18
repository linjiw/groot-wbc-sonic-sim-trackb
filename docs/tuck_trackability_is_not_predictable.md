# The arm tuck's trackability cannot be predicted from the clip

The local arm tuck is accepted by SONIC on **1 of 3 valid nominal motions**. This records two
attempts to predict *which* one from the clip alone, both of which failed, and the decision to
stop trying.

## The evidence

Bare-plane rollouts, no obstacles. The gate's verdict is set entirely by **external** contact:

| cell | outcome | self N | external N | external body |
|---|---|---|---|---|
| x000_nominal | accepted | 63.7 | 0.0 | — |
| x000_tuck | **rejected** | 76.0 | 44.7 | `pelvis` |
| x001_nominal | **rejected** | 10.9 | 277.6 | `left_knee_link` |
| x001_tuck | **rejected** | 71.4 | 138.1 | `left_knee_link` |
| x002_nominal | accepted | 79.1 | 0.0 | — |
| x002_tuck | **rejected** | 152.5 | 26.2 | `right_wrist_yaw_link` |
| x003_nominal | accepted | 6.3 | 0.0 | — |
| x003_tuck | accepted | 192.1 | 0.0 | — |

Every accepted cell has external contact of exactly 0.0 N; every rejected cell has more. With no
obstacles in the scene, the only external body available is the floor, so each rejection is a
limb reaching the ground that should not — a stumble, not a collision with furniture.

**Self-contact force is not the signal.** `x003_tuck` carries the highest self-contact of all
eight cells, 192.1 N, and is accepted. `x001_nominal` carries the lowest, 10.9 N, and is
rejected. The gate has always decomposed contact by Newton's third law and counted only the
external part; that is correct and it means self-contact magnitude cannot be read as severity.

`x001`'s nominal is itself rejected, so it cannot test an operator at all. **Nominals must be
screened for trackability before anything is adapted from them** — otherwise the operator is
blamed for a motion the controller could not hold to begin with.

## Two refuted predictors

**Wrist-to-hip clearance.** The rejected cells' self-contact bodies repeat a suggestive triple
(`left_hip_roll_link`, `left_wrist_yaw_link`, `pelvis`), which reads as the tuck pressing the
wrist into the hip. It is not the mechanism. Measured on MuJoCo collision geoms, the gap is
*already negative* on every nominal — the arms rest against the hips, so the absolute distance
saturates exactly as `self_clearance` did before it. The change from nominal to tuck
anti-correlates with the verdict at both extremes:

| motion | wrist–hip change | verdict |
|---|---|---|
| 003 | **−94.7 mm** (worst) | accepted |
| 005 | −81.6 mm | rejected |
| 002 | −70.2 mm | rejected |
| 010 | −19.4 mm | accepted |
| 000 | **−12.5 mm** (best) | rejected |

A bound on this quantity was added to the operator and then reverted: it cut tuck strength
roughly eightfold to enforce a constraint that does not predict the outcome.

**Lateral centre-of-mass excursion.** If the failure is balance, the tuck should be shifting
mass. It barely does — the arms are light relative to the body:

| cell | lateral CoM shift vs nominal | peak lateral CoM | verdict |
|---|---|---|---|
| x000 | 1.1 mm (smallest) | 10.1 mm | rejected |
| x002 | 1.8 mm | 16.0 mm | rejected |
| x003 | 1.2 mm | 15.9 mm | accepted |

`x002` and `x003` are 0.1 mm apart on peak lateral CoM and land on opposite sides of the gate.

## What follows

Trackability is measured, not predicted. Both cheap CPU-side screens failed, and this is the
fourth time on this operator that an inferred quantity has had to be withdrawn in favour of a
rollout — the same lesson that removed `family_eligible` from the Stage 2 report.

So the tuck's ~1/3 yield is a **budget fact, not a bug**: roughly three rollouts per usable
lateral clip, on top of one to screen each nominal. That is the number to plan the lateral
column of the 3×3 around, and the reason the overhead crouch — if the matched 2×2 holds — is the
cheaper anchor for the first paper-level family set.
