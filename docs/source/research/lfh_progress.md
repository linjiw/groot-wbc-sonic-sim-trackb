# LFH Trajectory-Conditioned Scene Research

<div class="lfh-stats">
  <div><strong>5</strong><span>obstacles in latest scene</span></div>
  <div><strong>4/4</strong><span>E10b physics pattern</span></div>
  <div><strong>91.7%</strong><span>conditional top-3 LOO</span></div>
  <div><strong>400/400</strong><span>samples inside support</span></div>
</div>

LFH does not estimate natural obstacle frequency and never predicts a physics verdict. It builds a
source-balanced proposal distribution over obstacles that are feasible for exact executed nominal
and adapted trajectories. Deterministic geometry proposes; the unchanged physics scorer labels.

## Motion → LFH → physics loop

The implemented loop now begins with a freshly sampled Kimodo motion. A pair is allowed to reach
scene design only after both the original and deterministic adaptation track in empty-scene Isaac
physics. Failed motion pairs stop; LFH does not invent an obstacle around an invalid controller
response.

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/fresh-motion-wide-poster.png">
  <source src="../_static/lfh_progress/fresh-motion-wide.mp4" type="video/mp4">
</video>

This is actual Isaac Lab rendering of the new curved walking sample. Nominal accepted. Its 80 mm
crouch twin rejected with 0 N external contact: endpoint error rose from
0.084 m to
0.376 m and p95 path error from
0.111 m to
0.410 m. No scene was generated for this
pair.

## Completed multi-obstacle scene

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e10b-isaac-wide-poster.png">
  <source src="../_static/lfh_progress/e10b-isaac-wide.mp4" type="video/mp4">
</video>

This seed-matched Isaac replay shows the completed five-obstacle E10b hard pair. LFH places one
blue hanging panel as the causal binding obstacle and four route-relative context obstacles on the
left, right, floor level, and overhead. Nominal strikes the binding panel at frame
120; adapted crouches and clears. The three intended-clear physics cells
have zero external contact, and no context object becomes causal. The wide videos are deterministic
presentation reruns; verdicts come from the hash-pinned experiment trajectories.

## MuJoCo scene and motion cross-render

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e10b-mujoco-cross-render.png">
  <source src="../_static/lfh_progress/e10b-mujoco-cross-render.mp4" type="video/mp4">
</video>

The same hard USDA scene and the two validated SONIC/Isaac trajectories are independently
cross-rendered above with MuJoCo EGL. Blue is the binding hanging panel; orange objects are
keep-out-certified context. The bridge converts the declared IsaacLab 29-DoF order to the audited
MuJoCo order and compiles USDA cube centres within `2.22e-16 m` and full sizes exactly. The nominal
view carries the recorded Isaac contact/rejection verdict; the adapted crouch carries the recorded
Isaac accepted verdict.

This is a kinematic `mj_forward` replay, not a MuJoCo physics rollout. It verifies scene and motion
materialization in a second renderer without replacing Isaac as the physics label source.

## Scene and trajectory maps

![Top-down route and multi-obstacle map](../_static/lfh_progress/route-map-2d.png)

The measured executed paths, binding footprint, and four context footprints are all shown in world
coordinates. Context placement is rejected against both swept bodies before physics.

![Isometric multi-obstacle scene and torso paths](../_static/lfh_progress/scene-map-3d.png)

The isometric view exposes the full 3D relationship rather than reducing the scene to one height.
The current v1 scene has one causal obstacle so the training label remains identifiable; context
objects increase scene complexity without receiving the causal label.

![Route-frame direction support](../_static/lfh_progress/direction-support.png)

LFH's representation can address route tangent, lateral, vertical, and oblique face normals. The
evidence is narrower: only overhead `−z` has a verified intervention window. Lateral faces are
context-only here, and floor/oblique critical placement stays fail-closed.

## Physics and causal metrics

| experiment | nominal easy | adapted easy | nominal hard | adapted hard | decision |
|---|---|---|---|---|---|
| E7c, one obstacle | accepted | accepted | rejected | accepted | verified |
| E10, 5 obstacles, seed 33101 | accepted | rejected | rejected | rejected | seed-confounded refusal |
| E10b, 5 obstacles, seed 32301 | accepted | accepted | rejected | accepted | verified |

E10b's CPU keep-out minimum is 288.78 mm versus the 50 mm
requirement. Nominal-hard torso contact is uniquely attributed to
`/World/ConstraintFrame/BindingHangingPanel` before reference drift at frame
135. The E10/E10b contrast also warns that a single successful seed
does not establish population-level context invariance.

## Existing crossed support

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e7c-four-cell-poster.png">
  <source src="../_static/lfh_progress/e7c-four-cell-replay.mp4" type="video/mp4">
</video>

The pose replay above remains useful for exact contact timing across all four source-086 cells.

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e7-cross-source-hard-poster.png">
  <source src="../_static/lfh_progress/e7-cross-source-hard-replay.mp4" type="video/mp4">
</video>

The cross-source view holds the I-beam constraint class fixed. The top row contains nominal hard
strikes; the bottom row contains adapted clears for sources 086, 089, and 090.

## Designed proposal distribution

![Trajectory-conditioned proposal support](../_static/lfh_progress/proposal-distribution.png)

`q_LFH_v1` assigns 0.25 mass to each independent source, then balances verified archetypes within
that source and easy/hard geometry equally. Window size or scene multiplication cannot give one
source more training weight.

`q_LFH_conditional_v1` adds an interpretable trajectory kernel over route phase, finite exposure,
engineering-window width, across-route extent, and operator. **A 2026-08-26 audit refuted the
conditioning at this scale**: with each source held out in turn, the kernel scores 1.51373 nats
against 1.60944 for a uniform prior, but **1.46416 for the same sampler with the kernel replaced by
a constant**. Feature-blind archetype counting therefore beats the fitted model, and the reported
gain over uniform belongs to marginal frequency. Top-3 recall 11/12 and 400/400 in-support sampling
are true by construction. Archetype identity is the part of the scene the executed pair does *not*
constrain; the part it does constrain, the placement window, is closed-form and unlearned.

| source | route progress | exposure (m) | engineering window (mm) | source mass | verified archetypes |
|---|---:|---:|---:|---:|---|
| `cf_005_056` | 0.651 | 0.50 | 142.04 | 0.25 | door_lintel, ibeam, shelf_plank |
| `lfh_086_crouch` | 0.544 | 0.10 | 33.41 | 0.25 | hanging_panel, ibeam, shelf_plank |
| `lfh_089_crouch` | 0.473 | 0.10 | 33.54 | 0.25 | door_lintel, ibeam, shelf_plank |
| `lfh_090_crouch` | 0.476 | 0.10 | 26.01 | 0.25 | door_lintel, ibeam, shelf_plank |

The count-only gate is green: every source has three verified archetypes. The stricter crossed
learning gate remains **closed** because only `shelf_plank` and `ibeam` are common to all four
sources. The next experiment should transfer `hanging_panel` to `cf_005_056`, 089, and 090; only
then can E5 compare critical, uniform-feasible, DCS-ranked, and visual-only samplers without
source–archetype confounding.

## LFH contract and next research step

1. Generate a motion, then execute both the original and proposed adaptation in empty-scene physics.
2. Measure trajectory-relative feasible support `(u*, normal, finite extent, keypoint, xi)` only
   from accepted executed pairs.
3. Sample one binding obstacle plus keep-out-certified context from the designed distribution
   `q_LFH`; use DCS only as a novelty/reporting term.
4. Execute all cells in physics. Complete a family only on
   accepted/accepted/rejected/accepted with unique binding attribution.
5. Learn proposal density, never natural obstacle frequency or a physics-verdict surrogate.

Next, do not add more overhead clutter. First make fresh motion→adaptation calibration reliable;
then obtain executed separation for a lateral or oblique operator. In parallel, finish the common
`hanging_panel` cross needed for the equal-budget learner comparison.

Machine-readable evidence: `docs/hallucination/q_lfh_v1.json`,
`e10b_mujoco_cross_render.json`, `e7_transfer.json`, and `e10_context_rich.json`. The conditional
evaluation and report are `autoresearch/iterate-260822-0031/conditional_eval.json` and
`docs/hallucination/REPORT_Q_LFH_CONDITIONAL_MUJOCO.md`. Total E7/E7c spend is 0.398 contended
GPU-h; E10/E10b and all generated families remain excluded from claim 5.
