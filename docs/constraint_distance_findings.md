# What the corpus looks like once the approach is kept, not just the verdict

*Measured 2026-08-19 over the 22 rolled-out cells of the banded batch.*

Every episode in the corpus already knows how close the body came to the obstacle. It knows it at
one frame, as one number, because that is what the gate needs. `constraint_distance.py` keeps the
whole series instead — signed arclength to the obstacle station, planar distance to its footprint,
and capsule-to-box clearance, per frame, on the **executed** trajectory.

This costs no new physics. Each series is the un-reduced form of a number the pipeline already
trusts, taken from the same function, so a record and the verdict it explains cannot disagree.

## The corpus, per cell

| family | nominal_hard | adapted_hard | gained | verdict on geometry |
|---|---:|---:|---:|---|
| `n_013_ceiling_overhead_left` | −0.0001 | **+0.0335** | +33.6 mm | reversal |
| `n_064_ceiling_overhead_left` | −0.0033 | **+0.0043** | +7.6 mm | reversal |
| `n_013_wall_chest_left` | +0.0000 | −0.0001 | −0.1 mm | adapted still strikes |
| `n_013_wall_waist_right` | −0.0103 | −0.0019 | +8.4 mm | adapted still strikes, by 1.9 mm |
| `n_013_wall_waist_left` | −0.0253 | −0.0189 | +6.4 mm | adapted still strikes, by 18.9 mm |

Metres of clearance from the nearest collision capsule to the obstacle box; negative is overlap.

Two things fall out that a pass/fail column cannot express.

**The tuck is close, not wrong.** `n_013_wall_waist_right`'s adapted cell misses clearing its wall
by **1.9 mm**, and `n_013_wall_chest_left`'s by 0.1 mm. These are not failures of the operator's
principle, they are a delivery shortfall with a size, and the size is small enough to be worth one
more turn of `DELIVERY_RATIO` rather than a redesign. `n_013_wall_waist_left` at 18.9 mm is the one
genuinely short of its target.

**Both ceiling families reverse on geometry.** The crouch does clear a ceiling the walk strikes, in
both nominals that produced a complete 2×2 — including `n_064`, whose cells are void for an
unrelated reason. Geometry was never the blocker in the overhead band; transport was, which is what
[P9](prediction_register.md) is about.

## The verified family reproduces, independently

Run over the 26 cells of `matched/`, the one operator-built verified family comes back as its own
2×2 — measured from the executed trajectories by a geometric proxy that knows nothing about the
gate that graded them:

| `mf_005_c08` | easy | hard |
|---|---:|---:|
| nominal walk | +0.0562 | **−0.0000**, overlapping from frame 99 |
| crouch (0.8) | +0.1017 | +0.0108 |

Metres of capsule-to-box clearance. Only the cell the paper reports as rejected goes negative.

The overlap begins at frame **99**, before the recorded contact at frame **113** and before the
swept-volume prediction of 112. The ordering is the right way round — capsule surfaces meet before
a disallowed force builds — and the 14-frame gap is the capsule envelope being conservative against
the real mesh. It is reported rather than tuned to agree: this is a proxy, and a proxy that matched
a force record to the frame would be suspicious rather than reassuring.

The same run independently confirms the audit's reading of `mf_x003_c08`. Its `z1228` and `z1239`
cells report a minimum clearance of **0.77–0.80 m** against a binding body of
`left_shoulder_yaw_link`, and reach their station only at frame ~187 of 199. The robot does not come
near that obstacle, which is what makes those cells void rather than accepted.

## Where the record disagrees with the gate, and why that was the useful part

Cross-checked against the recorded physics verdicts, geometric overlap agrees with
`disallowed_robot_contact` on **16 of 20** cells. All four disagreements are the same direction —
physics recorded a contact the shelf cannot account for — and they split into two causes:

* `n_013_wall_chest_left/nominal_hard` clears the shelf by **10 µm**. At that separation a capsule
  approximation of the collision mesh cannot resolve the verdict, and the honest reading is that
  the geometric proxy has run out of resolution, not that the physics is wrong.
* All four `n_064_ceiling_overhead_left` cells touch something that is **not the shelf**. Chasing
  that led to the room-sizing defect recorded in the register — independently re-deriving a fix the
  builder had already received hours earlier.

That second case is the argument for the record. A measurement that only reported the intended
obstacle's clearance would have agreed with itself on every cell and surfaced nothing.

## What it is for

The plan's sentence is that a model trained on the scalars learns *that* a crouch is needed and
never *when*. `remaining_to_station_m` is the field that carries the when, and its sign is what
makes "not yet" and "too late" different states rather than the same distance. `seconds_to_station`
converts it at each frame's own speed, and is NaN where the robot is stopped rather than a large
finite number that would read as safe.

## What is not claimed

That these clearances are the gate. They are a geometric proxy measured on collision capsules; the
verdicts remain the physics gate's, and where the two disagree the physics wins and the disagreement
is reported. Nothing here re-grades any cell.
