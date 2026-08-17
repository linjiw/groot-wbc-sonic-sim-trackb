# Where the scene contact actually comes from

Scene contact became the dominant rejection after the gate rework — 18 of 24 rejections
carry it — which sat in tension with the corpus's claim that clutter is built so the robot
threads through it. Two hypotheses were on the table: that scenes were built from the
*reference* corridor and drift ate the margin, or that layouts were reused across motions.

Neither is the answer. The join between how each scene was made and what its episode did
splits the problem cleanly in a third way.

## The measurement

| Scene construction | Episodes | With scene contact | Rate |
|---|---|---|---|
| Motion **placed into** a fixed hand-authored scene | 67 | 14 | **21%** |
| Scene **built around** that motion's reference | 97 | 4 | **4%** |

Placement into fixed geometry is **five times** more likely to produce contact. The clutter
builder is not the problem; the placement search is.

And of the four contacts in built-around scenes, **three are the same broken motion**:
`clutter_147`, the squat whose reference is self-intersecting (thigh through pelvis), which
pushes limbs outside the envelope the margin was sized for. The fourth is `clutter_033` at
2.8 N against a 1.0 N threshold. So the built-around failure rate attributable to margin
sizing is **one marginal episode out of 97**.

Drift is not the mechanism either. On built-around episodes with contact, maximum drift is
0.083–0.202 m against an 0.85 m clearance. On the placed episodes it reaches **2.564 m**
(`05_root_path__factory_aisle`, 1849.9 N) — but that is a tracking collapse, not a margin
being eaten.

## One premise I had wrong

The gate-policy argument for re-anchoring labels to the executed trajectory was written as
*"the room is built around the corridor the robot executed, so the scene explains the path
the robot actually took."* That is false. The executed corridor does not exist before a
rollout, so it cannot be an input: `build_clutter_scenes.py` reads
`canonical_path_xy(np.loadtxt(csv))` — the **reference**.

The conclusion survives, but on different evidence and more weakly stated. Re-anchoring is
not justified by traversability-by-construction. It is justified by measurement: episodes in
built-around scenes touch the scene at 4%, and every one of the 38 episodes the re-grade
promoted has **0.000 N** of contact. **The contact gate is what guarantees collision-freedom
per episode. The construction only makes it likely.**

Both the docstring and the review page said the stronger thing and now say the weaker one.

## What follows

- **The placement path needs the attention, not the clutter builder.** `route_placement`
  searches yaw and translation so a *reference* path clears obstacles at a fixed margin;
  with diverse motions that margin is being chosen against the wrong body model, the same
  0.45 m pelvis radius already corrected in the clutter builder to a measured 0.664 m.
- **Prefer built-around scenes for scale.** At 4% against 21%, generating a room per motion
  is both cheaper in rejected rollouts and better labelled than fitting motions into a fixed
  library of rooms.
- **The squat is one motion, not a class.** Excluding `clutter_147` leaves a single marginal
  contact episode in 97. The reference-feasibility prefilter removes that motion before it
  reaches the GPU at all.
- **Record nearest-obstacle margin per episode.** It is a free difficulty label and the
  natural axis for a benchmark's hard split, and it is already computed inside the builder.
