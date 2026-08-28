# Self-report: motion-to-scene generation, for external review

**Date:** 2026-08-28 · **Author:** the agent that wrote this code · **Status:** written to be audited

This describes what the pipeline actually does, in the order data flows through it, with the
constants and file locations needed to check each claim. It is written for a reviewer who has not
seen the code and who should assume nothing here is correct until verified.

**Two defects found while writing this document** are in §11. One of them undercuts a claim I made
earlier today, and a reviewer should read §11 before §7 so they know what to look for.

Everything below is CPU-side geometry and a learned generator. **No physics is involved anywhere in
this pipeline.** The frozen SONIC controller in Isaac Lab is the only thing that decides whether a
motion actually executes, and it has not been run on any scene in this report.

---

## 1. What the system is supposed to do

Given an executed motion `tau` — a humanoid walking, and at some point crouching — infer a scene
`S` that *explains* it: a set of obstacles under which crouching is necessary and sufficient.

The target set, which the project committed to before this code was written:

```
S_{eps,delta}(tau) = { S : Feasible = 1, Regret <= eps, Necessity >= delta, Realism >= r0 }
```

* **Feasible** — `tau` fits in `S`.
* **Necessity** — the motion the robot would have executed in an empty scene (the cheapest
  candidate, i.e. the un-edited nominal) is blocked by at least `delta`.
* **Regret** — no candidate cheaper than `tau` by more than `eps` is also feasible; otherwise the
  robot wasted more than `eps` of edit cost and the scene does not explain *this* motion.
* **Realism** — the obstacle looks like a thing (a lintel, a beam), not an arbitrary box.

## 2. Pipeline, end to end

```
Kimodo text-to-motion            150 qpos CSVs, (T, 36), 30 fps        [§3]
        |
   reference gate                94 of 150 "worth a rollout"           [§4]
        |
   edit operators                per clip: nominal + 3 crouches         [§5]
   local_crouch / local_arm_tuck              + 4 one-sided arm tucks
        |
   candidate set                 labels, edit costs, extent profiles    [§6]
        |
        +---------------> capsule point clouds (29 capsules)            [§7]
        |                        |
   MultiObstacleHallucinator      |    <- the only learned component    [§8]
   (extent profiles -> K boxes)   |
        |                        |
   ObstacleGeometry.decode -> to_world -> world boxes
        |                        |
        +----> SdfChoiceDecoder (fixed, parameter-free) -> choice       [§9]
                                 |
                          losses / metrics                              [§10]
```

## 3. Where the motions come from — they are generated, not captured

`/data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/motions/clips/` — **150 CSVs, not in git.**

They are produced by **Kimodo**, a text-to-motion diffusion model, from natural-language prompts,
then retargeted to the Unitree G1. Filenames carry the prompt:
`000_a_person_walks_slowly_in_a_straight_line_s0.csv`,
`001_a_person_walks_slowly_curving_gently_to_the_left_s0.csv`,
`003_a_person_walks_slowly_then_turns_sharply_left_an_s0.csv`.

Each file is `(T, 36)` float: **7 root** (xyz + wxyz quaternion) + **29 joint positions**, 30 fps,
typically 120 frames (4 s). Prompts cover: straight walks, gentle left/right curves, sharp turns,
turn-in-place-then-walk, sideways stepping, stop/start, backward walking, carrying, at slow /
steady / brisk pace.

**These are reference trajectories, not executed ones.** Nothing in the LfLH pipeline has been
through the controller. `docs/generation_quality_vs_tracking.md` records that of 150 generated
clips, 144 are kinematically feasible and, of 47 rolled out, 45 tracked — but crouches specifically
were only 4 of 9 kinematically feasible, which is why crouches are *constructed* by editing walks
(§5) rather than generated from a crouch prompt.

Generation environment and its traps are in the `kimodo-generation-env` note, not here.

## 4. The reference gate

`docs/hallucination/coverage/reference_gate.json`, 150 rows, `worth_a_rollout` true for **94**.
Screens on `embodiment_feasible`, `reference_semantic_valid`, `self_collision_free`,
`saturated_cell_fraction`. `build_candidate_sets.py` iterates the gate and skips any clip that
fails it, then re-runs `screen_reference` on the clip itself.

## 5. How motions are edited — this is the heart of it

`gear_sonic/dataset_generation/local_adaptation.py`. The key design decision: **a counterfactual
pair is one motion and a local deviation of it**, never two independently generated clips. Pairing
two separate clips cannot distinguish "the shelf separated the behaviours" from "these were two
different journeys", and the construction cannot tell them apart.

Both operators hold fixed: **root XY and yaw** (so both walk the same line), duration and frame
count (so gait phase stays aligned), start and goal. `root_path_preserved` is asserted in the
report and checked by `build_candidate_sets.py`.

### 5.1 Where the edit is applied

The edit is localised in **route progress** — cumulative path length normalised to `[0, 1]`
(`route_progress`) — not frame index, because an obstacle sits at a place, not a time.

`adaptation_profile` builds a smooth `0 -> 1 -> 0` bump centred on `station_fraction`, of
half-width `window`, with `ramp = 0.45` of it spent ramping, shaped by **smoothstep**
(`a^2(3-2a)`) so onset has no velocity discontinuity for the tracker to fight.

Constants actually used by the candidate builder: `STATION_FRACTION = 0.55`,
`WINDOW_FRACTION = 0.30` (overriding the module default of 0.18).

**Locality is a claim about causality**, per the module docstring: a clip crouched from frame zero
cannot demonstrate a decision made from what the robot sees, because the obstacle is not visible
when the crouch begins.

### 5.2 `local_crouch`

Joints: `hip_pitch`, `knee`, `ankle_pitch`. Applied as **one coupled parameter**, not per joint:

```
coupling = {hip_pitch: -1.0, knee: +2.0, ankle_pitch: -1.0}
out[joint] = qpos[joint] + alpha * scale * gain[joint]
```

The coupling is built in rather than discovered because moving any single leg joint alone *raises*
the body — measured, all six gave a negative drop — since flexing one tilts the body or lifts a foot
and root compensation puts it back.

* **Additive, not multiplicative.** Scaling existing angles ties depth to gait phase; measured, the
  drop oscillated 0.033–0.157 m inside a fully-active window.
* **Root follows the legs per frame**: `out[:, 2] += nominal_soles - sole_height(out)`, so the feet
  stay where the nominal put them.
* **`scale` is found by bisection (12 iterations) on the measured capsule silhouette**, not from
  joint angles — the leg-flexion-to-body-height relation depends on pose and is not modelled.
* **The bisection targets the *minimum* drop inside the fully-active window** (`alpha > 0.9`), not
  the maximum anywhere. A shelf must clear the robot on every frame it passes under.
* **Excursion cap `MAX_CROUCH_EXCURSION_RAD = 0.98`**, set from a measured trackability boundary
  between 0.994 rad (accepted, 0.100 m/s drift) and 1.000 rad (rejected, 0.225 m/s). Clips that hit
  the cap report `excursion_capped` and are **dropped** by the candidate builder.
* **Waist pitch is off by default** (`WAIST_USE_FRACTION = 0.0`) although it is the most efficient
  lever (109 mm/rad vs the squat's 64), because the crouch that physics verified moves the waist by
  exactly 0.000 rad and enabling it meant the repo no longer reproduced that clip.
* Joints are clipped to `centre +/- 0.94 * half_range` (`DEFAULT_RANGE_KEEP`), leaving headroom so
  the reference does not ride a limit.

Amplitudes offered: **40, 55, 70 mm** of silhouette drop (`CROUCH_DROPS_M`).

### 5.3 `local_arm_tuck`

Joints: `shoulder`, `elbow`, `wrist`, on one side (`left` / `right`). Legs, root and contact
schedule are untouched, which is why it is the easier operator.

The subtle part: **which direction narrows the robot is measured, not assumed.** Each mirrored
joint group is probed with `+/- 0.15 rad` under up to four sign hypotheses, and the sign that
reduces mean half-width over the active window is kept. An earlier version blended toward the arm
pose at the clip's own narrowest frame and made two clips **64 mm wider** than the walk they came
from. Mirrored joints are probed **together**, because half-width is a maximum over capsules and in
a symmetric pose both wrists attain it at once, so a per-joint probe sees a flat objective.

A one-sided tuck is scored on its **signed half-width on its own side**, not the symmetric maximum,
or a left-side reduction is hidden whenever the right arm is wider — half the gait cycle.

Excursion cap `MAX_TUCK_EXCURSION_RAD = 0.40`, from a failure where an unbounded bisection applied
1.300 rad, destabilised the robot and drove it into a wall at 114 N.

Amplitudes offered: **40, 70 mm** of half-width reduction, each on left and right.

### 5.4 Edit cost — and an asymmetry a reviewer should check

`build_candidate_sets.py`, per candidate, normalised so a 70 mm crouch and a 70 mm tuck cost ~1:

```python
costs.append(report.silhouette_drop_m / 0.070)   # crouch: the ACHIEVED drop
costs.append(reduction / 0.070)                  # tuck:   the REQUESTED reduction
```

**These are not the same kind of quantity.** Crouch cost is measured from the operator's report;
tuck cost is the target that was asked for, regardless of what the bisection delivered. I have not
audited whether tuck delivery is close enough for this to be harmless. It matters because the
decoder's choice rule is `cost + penalty * collision`, so a mis-stated cost directly changes which
candidate wins.

## 6. What a "candidate set" contains, per motion

One per clip, written to `docs/hallucination/lflh_candidates.json` (24 sets) and
`lflh_candidates_large.json` (64). Built by `build_candidate_sets.py`; sets with fewer than 4
surviving candidates are dropped.

| field | meaning |
|---|---|
| `motion_index`, `source_csv`, `body_mode` | provenance back to the Kimodo clip |
| `labels` | `["nominal", "crouch_040", "crouch_055", "crouch_070", "tuck_left_040", "tuck_left_070", "tuck_right_040", "tuck_right_070"]` |
| `costs` | `[0.0, 0.571, 0.786, 1.000, 0.571, 1.000, 0.571, 1.000]` — see §5.4 |
| `stations` | 16 |
| `fractions` | `linspace(0.25, 0.85, 16)` — route-progress positions of the station grid |
| `station_xy_m` | world XY of each station, from the **nominal** clip |
| `yaw_rad` | executed heading at each station, from the **nominal** clip |
| `extents` | `(candidates, 3, 16)` — up / left / right body extent per station, per candidate |

**`extents` is the only thing the learned model ever sees.** Not the trajectory, not the joint
angles: three numbers per station per candidate, and only the nominal's and the observed's rows are
used (§8).

`extents` comes from `motion_envelope.extract_envelope`. At each station it takes every capsule
endpoint of every frame, projects into that station's route frame `[tangent, lateral, z]`, keeps
points whose along-route offset is within `0.20/2 + radius`, and reports
`max(z + radius)`, `max(across + radius)`, `max(-across + radius)`. Radii are added, so the extents
describe the **body surface**, not capsule axes. A station no body part occupies reports `0.0`
rather than NaN.

Note that `station_xy_m` and `yaw_rad` are taken from the nominal only. This is consistent — the
operators preserve root path — but it means the route frame is a property of the clip, not of the
candidate.

## 7. Body geometry

`gear_sonic/dataset_generation/swept_volume.py` — `G1_COLLISION_CAPSULES`: **14 bodies, 29
capsules** (pelvis, torso, both hip_roll, knee, ankle_roll, shoulder_yaw, elbow, wrist_yaw, ...).

`sdf_decoder.candidate_cloud(payload, frame_stride, samples_per_capsule=3)` produces the point
cloud the decoder scores: forward-kinematics the qpos, place capsules in world, subsample frames by
`frame_stride`, then place 3 points along each capsule axis and carry each point's radius.

Sampling **on the axis** means the reported distance is never smaller than the true one, so the
error is conservative in the safe direction. `frame_stride = 8` throughout; a stride mismatch
between two stages caused a real bug (§11.2).

## 8. The scene generator — the only learned component

`gear_sonic/dataset_generation/hallucination/lflh.py::MultiObstacleHallucinator`.

**Input:** `(batch, 2, 3, 16)` — the **nominal** and the **observed** extent profiles. Nothing else.

**Encoder:** three 1-D convolutions over the station axis (`k=5, 5, 3`, 64 channels, ReLU), applied
with shared weights to each of the two profiles, then mean- and max-pooled over stations.

**Fusion:** `[Z_nom, Z_obs, Z_obs - Z_nom, |Z_obs - Z_nom|]` — pair-conditioned, so the *edit* is
handed to the model explicitly rather than having to be inferred. A single-motion encoder was
measured to be beaten by its own input ablation.

**Head:** MLP `8*64 -> 64 -> 64 -> 2*K*6`, giving a mean and a log-sigma per obstacle parameter.
`log_sigma` is clamped to `[-20, 2]`; the floor is deliberately far away because the retracted
result's headline number *was* its clamp.

**Output:** `K = 4` obstacles x 6 parameters, sampled as `mean + sigma * eps`.

### 8.1 Latent to metres — `ObstacleGeometry.decode`

```python
station     = sigmoid(z0) * (stations - 1)        # 0 .. 15
lateral_m   = tanh(z1) * 0.60                     # -0.60 .. +0.60 m, + is left
height_m    = 1.30 + tanh(z2) * 0.45              # 0.85 .. 1.75 m, box CENTRE
half_along  = 0.03 + sigmoid(z3) * (0.80 - 0.03)
half_lateral= 0.04 + sigmoid(z4) * (1.20 - 0.04)
half_vert   = 0.02 + sigmoid(z5) * (0.80 - 0.02)
```

None of these anchors is derived from a feasibility answer — that was the retracted version's fatal
flaw, where the coordinate was anchored on `adapted.mean()`, which *is* the window's lower edge.

### 8.2 Route frame to world — `train_lflh_sdf.to_world`

```python
index    = clamp(round(station).long(), 0, 15)
yaw      = clip.yaw[index];  base = clip.station_xy[index]
centre_x = base_x - sin(yaw) * lateral
centre_y = base_y + cos(yaw) * lateral
centre_z = height_m
```

**`round().long()` is where a serious defect lives — see §11.1.**

## 9. The decoder — fixed, parameter-free, differentiable

`sdf_decoder.SdfChoiceDecoder`. It is not learned and has no trainable parameters. Its job is to
decide, given a scene, which candidate motion the robot would choose.

**Clearance.** Exact signed distance from a point to an oriented box, minus the capsule radius:

```python
q      = |local| - half_extents
signed = ||max(q, 0)|| + min(max(q), 0) - radius
```

Two reductions over the cloud, and the distinction matters:

* `clearance()` — **soft** minimum, `softmax(-signed / 0.01)` weighted average. Used for gradients
  only, so they reach the nearest few points rather than one.
* `exact_clearance()` — **hard** minimum. Used for every reported number. A soft minimum is an
  average and so never below the true minimum, which means it can call a penetrating scene clear.
  `tests/dataset_generation/test_sdf_exact_clearance.py` exhibits a case where it does.

**Choice rule:**

```python
collision = softplus(-(clearance - 0.0) / 0.02) * 0.02
objective = cost + 60.0 * collision.sum(over obstacles)
weights   = softmax(-objective / 0.15)
```

So a candidate is chosen on edit cost unless obstacles block it. `blocked_penalty = 60` against
costs in `[0, 1]` means roughly **17 mm** of penetration is enough to overcome the full cost range
— a reviewer should check whether that scale is deliberate; I set it by hand and have not swept it.

## 10. Losses

`train_lflh_sdf.train_sdf`, per step, averaged over `samples * clips`:

| term | form | weight |
|---|---|---|
| reconstruction | `-log weights[observed]` | 1 |
| **feasibility** | `relu(m_clear - clearance(observed))^2` summed over obstacles | 40 |
| **necessity** | `relu(m_hit - struck(cheapest))^2`, softmin over obstacles | 20 |
| **regret** | `relu(-blocked(rival))^2` for each rival cheaper by more than `eps` | 20 |
| realism | `size_penalty` toward `PlausibleShape`, annealed in after 33% of training | 6 |
| KL | `0.5(mu^2 + sigma^2 - 1) - log sigma`, unit prior | 0.004 |

`m_clear` and `m_hit` are read from `scene_ceiling.json` for the target and `eps` in use, at
`0.6 x ceiling`. `eps` defaults to `0.25`. The KL is a genuine Gaussian KL carrying `-log sigma`,
so it opposes collapse; without it collapse is a theorem, not a finding.

`set_regret_rivals` decides which candidates count: everything cheaper than the observed by more
than `eps`, excluding the observed and the cheapest (the latter is charged to necessity instead).
For `crouch_070` at `eps = 0.25` that is `{crouch_040, tuck_left_040, tuck_right_040}` — and
`crouch_055` is deliberately tolerated, being only 0.214 cheaper.

## 11. Defects found while writing this report

### 11.1 The station parameter receives exactly zero gradient — and the prior hides it

`to_world` does `round().long()` and uses the result as an **index**. Rounding has zero derivative
almost everywhere and `.long()` discards the graph entirely, so `z0` — *where along the route the
obstacle goes* — gets no gradient at all. Measured on the trained model:

```
d|loss|/d station      = 0.000000e+00      <- exactly zero
d|loss|/d lateral      = 2.08e-09          <- effectively zero
d|loss|/d height       = 3.21e-01
d|loss|/d half_along   = 1.07e-02
d|loss|/d half_lateral = 1.19e-16          <- effectively zero
d|loss|/d half_vert    = 4.39e-01
```

Only **height and half_vertical** are meaningfully learned. `lateral` and `half_lateral` are near
zero because the emitted slabs are already wider than the body, so the clearance minimum is set by
the underside height and is insensitive to both.

**The 6-parameter, 4-obstacle generator is in effect a 2-parameter model of one underside height.**

**Worse, the prior supplies the right answer for the dimension that is not learned.** With no
gradient, `z0` is driven only by the KL toward 0. The learned means are
`[0.042, 0.019, -0.005, -0.021]`, decoding to stations `[7.66, 7.57, 7.48, 7.42]` — i.e. **7.5 of
15**, which on the `linspace(0.25, 0.85, 16)` grid is route fraction **0.550**. And
`STATION_FRACTION = 0.55` is exactly where every edit operator centres its adaptation.

So the obstacle lands at the right place along the route **by construction**, not by inference.
This is the same class of error as the retracted result — where the parameterisation was anchored
on the answer — and I did not notice it until auditing for this document.

**Consequence for the amortisation claim.** I reported that the model beats a random-search control
5/5 at equal budget and at 43x budget. That control randomises all six parameters, including the
station the model never had to solve, so much of the gap may be the control wasting draws on a
dimension the prior gave the model for free. A station-matched control is in
`compare_amortisation.py --station-matched`, and its result is below.

### 11.1a The station-matched control — the amortisation claim shrinks by 3.6x but survives

Same model, same clips, same budget; the only change is that the control's `z0` is set to the prior
mean instead of drawn, so it is handed the station the model never had to learn.

| clip | model, best of 24 | control, best of 24 | control, best of 512 |
|---|---:|---:|---:|
| 011 | **+13.1 mm** | -133.8 | -1.2 |
| 012 | **+16.3 mm** | +10.4 | +10.4 |
| 013 | **+15.8 mm** | -22.4 | +10.6 |
| 014 | **+17.4 mm** | +5.2 | +13.2 |
| 030 | **+12.9 mm** | -115.4 | -10.7 |

| | all-random control | station-matched control |
|---|---:|---:|
| model wins at equal budget (24 draws) | 5 / 5 | **5 / 5** |
| median draws to match the model's median | 121 | **34** |
| clips where the control never matched within budget | 0 / 5 | 1 / 5 |

**What I claimed and what is true.** I reported "one forward pass is worth ~121–182 random draws".
Once the control is given the station, that falls to **34** — so roughly **3.6x of the reported
advantage was the prior, not the model**, exactly as §11.1 predicts.

**What survives, and it is not nothing.** The model's best of 24 still beats the station-matched
control's best of 24 on **5 clips out of 5**, and still beats its best of **512** on all five
(16.3 vs 10.4, 15.8 vs 10.6, 17.4 vs 13.2, and two clips where the control stays negative). A 21x
compute advantage does not close the gap on the parameters the model *does* learn.

So the honest statement is: **conditioned on being handed the station, the model still places a
better underside height than random search finds in 21x the compute — but the station itself is a
property of the parameterisation, not a thing this model infers.** Any paper claim must say so, and
must not present a 6-parameter generator when four of the parameters are inert.

### 11.2 Stride mismatch flipped a scene's verdict (fixed)

The ceiling search ran at `frame_stride = 4` and everything else at `8`. On clip 030 the binding
rival's clearance is `-0.7 mm` at stride 4 and `+5.3 mm` at stride 8 — excluded versus feasible,
which inverts the scene's verdict. Compounding it, regret was applied as a **gate** ("no cheaper
candidate is feasible"), which puts the optimum exactly on a rival's zero crossing, the least
robust place available. Regret is now a **margin** and stride defaults to 8 everywhere; the
reported `crouch_070` ceiling fell from 32.6 mm to 20.8 mm as a result.

## 12. The ceiling instrument — the one model-free measurement

`scripts/research/hallucination/measure_scene_ceiling.py`. Brute-forces **one** box per clip over
16 stations x 9 lateral offsets (`-0.4 .. +0.4 m`) x 140 underside heights (`0.80 .. 2.20 m`, 1 cm),
with a **fixed slab shape** `0.20 x 0.90 x 0.30 m`, and reports the best achievable

```
margin(S) = min( clearance(observed), min over excluded candidates of penetration )
```

This is an upper bound no hallucinator can exceed, and it exists so a learned margin is never
reported without a denominator. Measured, 16 clips, stride 8, `eps = 0.25`:

| target | rivals | any-rival | all-rival | **eps = 0.25** |
|---|---:|---:|---:|---:|
| `crouch_040` | 1 | 38.2 | 38.2 | **38.2 mm** |
| `crouch_070` | 5 | 58.6 | 9.8 | **20.8 mm** |

The headline finding: **the ceiling is set by the spacing of the edit ladder, not the amplitude of
the edit.** A deeper crouch buys room against the nominal and spends it on the rungs it passes
through — `crouch_070` acquires `crouch_040`, `crouch_055` and both 40 mm tucks as candidates it
must out-argue. The shallowest edit is the most explainable one.

**A reviewer should check whether the fixed slab shape biases this.** The ceiling is a bound for
*that shape*; the model may emit shapes that do better or worse, and I have not tested it. A model
scoring above 100% efficiency would be the signal, and that did happen once — for a different
reason (§11.2).

## 13. Metrics, and which of them are traps

| metric | definition | trustworthy? |
|---|---|---|
| selection rate | decoder's argmax is the observed | **no, on its own** |
| robot clear | `exact_clearance(observed) > 0` for every obstacle | yes |
| **counterfactual rate** | selection **and** clear | yes — report this |
| margin efficiency | achieved margin / per-clip ceiling | yes, if denominators match (§11.2) |

**Selection rate is a trap and I nearly reported it as a result.** The random control scores
**58.8%** on it while clearing the robot **0%** of the time at a median worst clearance of
**-320 mm**: boxes drawn from the prior engulf every candidate, and the deepest crouch wins by
being swallowed least. Any metric a scene which buries the robot can score well on is not measuring
explanation.

## 14. Current numbers

`crouch_070`, `eps = 0.25`, 11 clips fitted / 5 held out, margins 18.8 mm, 24 draws per clip.

| | model | random from prior |
|---|---:|---:|
| held-out selection | 81.2% | 58.8% |
| held-out robot clear | 50.0% | 0.0% |
| **held-out counterfactual** | **32.5%** | **0.0%** |
| margin efficiency, best of 24 | **72% median** | — |
| admissible scenes | 36 of 120 draws | — |

`crouch_040` — larger ceiling, cheaper edit, more useful — **fails completely**: 0.0% counterfactual
out of sample, *below* the random control on selection, reconstruction oscillating 3.46–6.83 and
never converging. A shallow target leaves the deeper crouches as escape routes: any box low enough
to strike the nominal is within 40 mm of striking `crouch_040` itself, and the decoder retreats to a
costlier rung that clears easily and that the regret rule does not require excluding.

Also measured: retraining `crouch_070` at strictly feasible margins (12.5 mm) made it **worse**
(clear 50.0% -> 25.0%). The barrier is `relu(m - gap)^2`, so the margin sets the gradient's scale
everywhere below it, not merely where it stops firing.

## 15. What I want reviewed, in priority order

1. **§11.1, the station gradient and the prior coincidence.** Is my reading of `to_world` right?
   Does the station-matched control (§11.1a) leave anything of the amortisation claim? If not, the
   honest conclusion is that this model learns an underside height and nothing else, and the paper
   should say so.
2. **Is the whole thing a 1-D problem in disguise?** If only height and half-vertical are learned,
   the learned generator is solving the same scalar the closed-form overhead window already solves
   in one line. That would make the learned component redundant rather than merely weak.
3. **§5.4, the cost asymmetry.** Crouch cost is achieved, tuck cost is requested. Does tuck
   delivery differ enough to change which candidate the decoder picks?
4. **`blocked_penalty = 60` and `choice_temperature = 0.15`** (§9). Set by hand, never swept.
   17 mm of penetration overcomes the entire cost range — is that the right scale?
5. **The fixed slab in the ceiling search** (§12). Does it bound the model fairly?
6. **`crouch_040`'s failure** (§14). Is my escape-route explanation right, or is it an optimiser
   problem I gave up on too early? Only 1000–1500 steps were ever run.
7. **The realism term.** `PlausibleShape` is an annealed penalty toward plank-like boxes. It is the
   weakest-justified part of the loss and nothing measures whether the output is realistic.
8. **`n = 5` held-out clips.** Every out-of-sample number rests on five clips and 120 draws.
   `lflh_candidates_large.json` has 64 sets built and unused.

## 16. Claim ledger

**Stands.** The ceiling instrument and the ladder-spacing finding (model-free, exhaustive,
replicated at 12 and 16 clips). The exact/soft clearance split. That a feasible objective trains
where an infeasible one does not. That selection rate alone is not a valid metric here.

**Weak.** Held-out counterfactual 32.5% vs 0.0% — real but two draws in three still express
nothing. Margin efficiency 72% — depends on the fixed-slab ceiling being a fair bound.

**Unproven.** Conditioning: the model is genuinely conditional (blinding the motion moves output by
84% of its across-clip spread, against the retracted model's 0.15 mm) but beats its motion-blind
control on only **2 of 5** clips.

**Reduced, not withdrawn (§11.1a).** Amortisation. One forward pass is worth ~34 station-matched
random draws, not the ~121–182 first reported; 3.6x of that figure was the prior handing the model
a station it never learned. The model still beats a station-matched control at equal budget 5/5 and
at 21x budget 5/5, so the advantage on the parameters it *does* learn is real.

**Previously retracted, do not cite.** `REPORT_LFLH_COMPARISON.md` in full;
`REPORT_LFLH_MULTI_OBSTACLE.md` addenda 2–3; `lflh_sdf.json`'s 18.8% robot-clear (infeasible
objective, and measured with the soft minimum).

## 17. Reproduce

```bash
PY="env -u PYTHONPATH LFH_TORCH_THREADS=2 $HOME/miniconda3/envs/env_isaaclab/bin/python"

$PY scripts/research/hallucination/build_candidate_sets.py --clips 24
$PY scripts/research/hallucination/measure_scene_ceiling.py \
    --clips 16 --targets crouch_070 crouch_040 --epsilon 0.25 --stride 8
$PY scripts/research/hallucination/train_lflh_sdf.py --target crouch_070 --epsilon 0.25 \
    --clips 16 --holdout 5 --steps 1000 --stride 8 --model-out /tmp/sdf.pt
$PY scripts/research/hallucination/compare_amortisation.py --model-in /tmp/sdf.pt --draws 24
$PY scripts/research/hallucination/compare_amortisation.py --model-in /tmp/sdf.pt --draws 24 \
    --station-matched                                   # the control that decides section 11.1
$PY scripts/research/hallucination/render_lflh_sdf_scenes.py --target crouch_070 --epsilon 0.25 \
    --clips 16 --holdout 5 --stride 8 --draws 24 --render-top 3 --with-oracle \
    --model-in /tmp/sdf.pt --out-dir docs/source/_static/lflh_ceiling

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  $HOME/miniconda3/envs/env_isaaclab/bin/python -m pytest tests/dataset_generation -q \
  -p no:cacheprovider          # 881 passing; read the count, not the exit status
```

The one hard dependency is `/data/robotixx/groot-wbc-kimodo-m0/` (466 MB, not in git). For the LfLH
path only `sweepcf_release/motions/clips/` is needed. See `docs/lfh/REPLICATION.md`.
