# The arm tuck's trackability cannot be predicted from the clip

The local arm tuck is accepted by SONIC on **1 of 3 valid nominal motions**. This records two
attempts to predict *which* one from the clip alone, both of which failed, and the decision to
stop trying.

> **Correction (2026-08-18).** This document described these as "bare-plane rollouts, no obstacles"
> and reasoned from it that every external contact must be the floor. That is wrong. The screening
> scene `cf_005_056_easy` contains a shelf at 1.3906 m spanning x ∈ [2.424, 2.924] and four walls
> in a 12.882 × 5.000 m room. One motion passes under the shelf and another comes within 0.4 m of a
> wall, so the screen was never obstacle-free and the floor was never the only surface available.
> The consequences are set out at the end.

## The evidence

The gate's verdict is set entirely by **external** contact:

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

`x001`'s nominal is rejected, so it cannot test an operator in this scene. **Nominals must be
screened before anything is adapted from them** — otherwise the operator is blamed for a clip that
was going to fail regardless. But the reason matters, and here it is not the one first recorded:
see below.

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


## Correction: what the rejections actually hit

Locating each rejection's contact in the scene changes two conclusions.

**`x001`'s nominal walked into a wall.** Contact at frame 198 with the root at y = −2.24 m — 0.21 m
from the wall face — carrying a force of exactly (0.0, 277.6, 0.0) N: purely lateral in y, pushing
the robot back toward the room's centre. That is a **room-sizing artifact**, the same failure mode
already recorded against an earlier family whose room was sized from one motion while another
travelled further. The controller tracked x001 perfectly well; the room was too narrow for its path.

So describing x001 as a nominal "the controller could not hold" was wrong, and the arm tuck's yield
of **1 of 3 valid nominals** rests on a denominator that may be 4.

**The two tuck rejections are not explained by the scene either.** `x000_tuck` contacts `pelvis` at
44.7 N with the root under the shelf's x-span but at z = 0.75 m, far below the 1.39 m underside;
`x002_tuck` contacts `right_wrist_yaw_link` at 26.2 N two metres from any wall and nowhere near the
shelf. Neither is a floor strike — the wrist never comes within 0.6 m of the ground in any of these
clips. Both may be self-contact whose opposing pair the decomposition failed to match, which at
26–45 N is plausible; that is untested.

**What this does not change.** The crouch results are unaffected: x000's crouch was rejected for
pure reference drift with *zero* external contact, and x002's and x003's were accepted with zero
external contact, so no obstacle was ever involved. The finding that self-contact magnitude is not
severity also stands, since it compares cells within the same scene.

**How far the damage goes: two cells of forty-seven.** Auditing every graded rollout for how close
its root came to a wall bounds this precisely.

| wall gap | outcome | cell |
|---|---|---|
| **0.20 m** | rejected | `x001_tuck` |
| **0.21 m** | rejected | `x001_nominal` |
| 0.42 m | accepted | `x002_nominal` |
| 0.44 m | rejected | `x002_tuck` |
| ≥1.18 m | accepted | every `mf_005_c08` cell |

Only x001's two cells come within 0.35 m, and **no verified family is affected** — every family cell
sits at least 0.54 m clear. So the correction is confined to one motion, and the arm tuck's
denominator is the only number in doubt.

**What it changes about method.** A trackability screen must run in a genuinely empty scene.
Screening in a furnished room cannot separate "the controller cannot hold this clip" from "this clip
does not fit this room", and those demand opposite responses — discard the motion, or resize the
room. `scripts/research/audit_wall_proximity.py` now runs this check on demand and exits non-zero
when any cell is too close, so a yield number can be quoted only after it passes.
