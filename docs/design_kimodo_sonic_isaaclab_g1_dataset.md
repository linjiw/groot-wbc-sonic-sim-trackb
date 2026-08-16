# Design: Kimodo + SONIC + Isaac Lab G1 synthetic datasets

Status: M0 end-to-end spike proven 2026-08-14, replicated and extended 2026-08-15. Deployment-latent
parity MEASURED encoder-equivalent on 26 rollouts (8.1.1, 18.1). Acceptance gates corrected to compare
against the commanded motion rather than absolute thresholds (20), after three false rejections of
tracked crouches, kneels and bows. Scene-aware placement implemented (19). Fifteen fully-bound synthetic_g1
episodes exported as dataset_v2 (21.3, 22), after visual inspection found and fixed debug markers
leaking the tracking goal into the ego camera (21). Cluttered scenes are now generated around known
motion paths (22); photorealistic assets remain future work (22.5).

Target: Unitree G1 in household and factory USD scenes

Primary output: physics-validated LeRobot/GR00T episodes with SONIC latent actions

Secondary output: accepted and rejected motion-reference corpora for SONIC training and analysis

## 1. Decision

Yes, this repository can support a pipeline similar to the AnimoFlow example, with one important change: the final dataset should come from a physics-executed G1 in Isaac Lab, not directly from Kimodo's kinematic animation.

The responsibilities are:

- **Kimodo** generates diverse G1 kinematic references from text plus root-path and, later, end-effector constraints.
- **SONIC** turns those references into balanced, dynamically feasible G1 behavior and exposes the 64-dimensional latent action used by the existing VLA stack.
- **Isaac Sim/Isaac Lab** owns USD scene loading, collisions, dynamics, sensors, randomization, success predicates, and parallel rollout.
- **The dataset pipeline** preserves provenance, rejects unsafe or unsuccessful executions, renders accepted rollouts, and exports the existing LeRobot/GR00T action contract.

This is a better fit than reproducing the blog pipeline literally. Kimodo is intentionally kinematic; a plausible animation can still fall, clip furniture, violate joint or torque limits, miss a path, or fail an object interaction. The SONIC-in-Isaac stage converts a motion proposal into evidence that the behavior is executable.

The first deliverable should be **scene-conditioned locomotion**. Full household/factory manipulation is a later milestone because Kimodo does not replace grasp planning, task segmentation, or contact-aware object control.

## 2. Evidence and existing repository seams

### 2.1 Kimodo

Kimodo supports text prompts combined with sparse 2D root waypoints, dense 2D paths, full-body keyframes, and hand/foot constraints. Its G1 model produces a 34-joint motion representation and can export a 36-value MuJoCo qpos CSV: root XYZ, root quaternion in WXYZ order, and 29 G1 joint angles. The CSV is already Z-up and +X-forward, which is the preferred interchange format for this project.

Relevant upstream documentation:

- [Kimodo project and G1/SONIC application](https://research.nvidia.com/labs/sil/projects/kimodo/)
- [constraint types and coordinate conventions](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html)
- [G1 CSV and NPZ output formats](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/output_formats.html)
- [Python generation API](https://research.nvidia.com/labs/sil/projects/kimodo/docs/api_reference/model.html)

Kimodo generation should run locally in its own environment or container. The design does not require the AnimoFlow REST service. An API-backed generator can be added behind the same artifact contract later.

The local source audit at Kimodo commit `901a98b` confirmed that the G1 exporter writes a
headerless 36-column CSV and performs the Y-up/+Z-forward to Z-up/+X-forward transform before
writing it. The local generation path currently disables post-processing for G1, so the worker
must record that fact and validate the output; it must not claim that post-processing ran.

### 2.2 SONIC and G1 support already here

The repository already contains the difficult controller and representation pieces:

- The released SONIC Isaac Lab training/evaluation environment and G1 configuration live under [`gear_sonic/envs`](../gear_sonic/envs/).
- The motion library accepts G1 root transforms and 29-DoF joint trajectories. The existing conversion code documents MuJoCo/Isaac Lab joint ordering in [`convert_soma_csv_to_motion_lib.py`](../gear_sonic/data_process/convert_soma_csv_to_motion_lib.py).
- The wrapper caches the actual decoder-input latent in `env._full_latent`; this is the value to export as `action.motion_token`, not the policy's residual meta-action. See [`manager_env_wrapper.py`](../gear_sonic/envs/wrapper/manager_env_wrapper.py).
- Tiled RGB cameras, head-mounted ego cameras, object/table assets, contact sensors, and camera randomization already exist in [`modular_tracking_env_cfg.py`](../gear_sonic/envs/manager_env/modular_tracking_env_cfg.py).
- The current exporter already writes LeRobot/GR00T episodes with the 64D SONIC token and two 7D hand commands. See [`features_sonic_vla.py`](../gear_sonic/data/features_sonic_vla.py) and [`dataset_contract_sonic_vla.md`](dataset_contract_sonic_vla.md).
- A trajectory recorder exists in [`recorders.py`](../gear_sonic/envs/manager_env/mdp/recorders.py), although it is not yet an asynchronous episodic recorder suitable for large generation runs.

### 2.3 Isaac Lab functionality to reuse carefully

Isaac Lab provides vectorized environments, tiled cameras, imitation-learning recorders, and Isaac Lab Mimic. Current upstream documentation also includes a G1 locomanipulation data-generation workflow that combines manipulation demonstrations with point-to-point navigation.

- [Isaac Lab cameras and tiled rendering](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/core-concepts/sensors/camera.html)
- [Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)
- [Mimic subtask data generation API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab_mimic/isaaclab_mimic.datagen.html)

This repository currently declares Isaac Lab 2.3.2. Upstream `main` examples must therefore be treated as reference implementations until their APIs and extensions are confirmed against the pinned environment. The MVP must not silently upgrade Isaac Lab.

## 3. Goals and non-goals

### Goals

1. Generate open-ended G1 locomotion episodes in household and factory scenes from route, prompt, style, speed, and seed variation.
2. Execute every candidate with SONIC under Isaac physics before accepting it.
3. Record robot state, SONIC latent action, cameras, contacts, scene/object state, reference motion, task labels, and complete provenance.
4. Export a deterministic, validated LeRobot/GR00T view without losing a richer raw simulation record.
5. Partition by scene/layout/object identity so test scenes cannot leak into training through alternate routes or renders.
6. Extend the same pipeline to approach, reach, pick/place, carry, inspect, and workstation tasks without changing the core artifact contracts.

### Non-goals for the MVP

- Arbitrary articulated-object interaction such as opening every cabinet or operating every machine.
- Treating a visually plausible Kimodo clip as a successful robot demonstration.
- Training a new SONIC foundation checkpoint before the generation and validation loop is proven.
- Importing the AnimoFlow service or making dataset generation depend on an external job queue.
- Photorealistic rendering during the high-throughput physics pass.

## 4. End-to-end architecture

```text
USD scene registry
      |
      v
scene validation + collision-derived walkable map -----> cached scene package
      |                                                       |
      v                                                       |
task/route sampler                                            |
  A* -> string pull -> curvature/speed schedule               |
      |                                                       |
      v                                                       |
Kimodo job manifest -> G1 generation -> NPZ + qpos CSV        |
      |                                                       |
      v                                                       v
coordinate/joint adapter -> SONIC motion-lib reference -> Isaac Lab scene clones
                                                              |
                                                   SONIC physics execution
                                                              |
                                      +-----------------------+-------------------+
                                      |                                           |
                               rollout validator                           raw recorder
                                      |                                           |
                          +-----------+-----------+                               |
                          |                       |                               |
                      accepted                rejected/quarantine <---------------+
                          |
                 render + sensor replay
                          |
          canonical episode store + LeRobot/GR00T export
```

The stages communicate through versioned files rather than importing Kimodo inside Isaac Sim. That separation prevents CUDA and Python dependency conflicts, makes every stage restartable, and allows Kimodo generation and Isaac simulation to use different GPUs or machines.

## 5. Canonical contracts

### 5.1 Scene manifest

Each scene has a checked manifest rather than relying on filename conventions:

```yaml
schema_version: 1
scene_id: household_apartment_0001
domain: household                 # household | factory
usd_path: assets/scenes/apartment_0001/scene.usd
asset_hash: sha256:...
meters_per_unit: 1.0
up_axis: Z
static_collision_roots: [/World/Scene]
floor_hints: []                   # optional; geometry extraction must work without them
excluded_regions: []
spawn_regions:
  - name: living_room
    polygon_xy: [[...], [...]]
goal_regions:
  - name: kitchen_station
    polygon_xy: [[...], [...]]
task_stations: []
camera_rigs: [g1_head]
split_group: apartment_0001       # all variants stay in one dataset split
license_id: ...
```

The asset-ingestion gate checks units, Z-up convention, valid collision geometry, rigid-body APIs for dynamic objects, material/asset resolution, and a license record. It produces a scene hash, walkable-map cache, and validation report.

The repo-owned M0 fixture package is checked in at
[`gear_sonic/data/assets/scenes/g1_dataset`](../gear_sonic/data/assets/scenes/g1_dataset/):

- [`household_room.usda`](../gear_sonic/data/assets/scenes/g1_dataset/household_room.usda)
  is a furnished two-zone household layout with a two-metre doorway and a clear canonical route.
- [`factory_aisle.usda`](../gear_sonic/data/assets/scenes/g1_dataset/factory_aisle.usda)
  is a rack-lined factory aisle with a clear route detouring around a center pallet.

Both files are self-contained, repo-authored USDA made from a PhysX `Plane` support surface and
axis-aligned `Cube` obstacles. They are Z-up at one metre per unit, use `/World` as `defaultPrim`,
put the collision-backed support-floor surface exactly at `z=0`, and enable
`PhysicsCollisionAPI` on every solid. They do
not reference third-party meshes, materials, or textures. The package
[`manifest.json`](../gear_sonic/data/assets/scenes/g1_dataset/manifest.json) records each scene's
SHA-256 and byte size, Apache-2.0 redistribution metadata, repo-authored provenance, scene family,
`split_group`, walkable bounds, support floor, canonical route, and declared 0.45 m route-clearance
radius. The checked routes have static minimum obstacle clearances of 0.80 m in the household
scene and 0.70 m in the factory scene.

The standard-library-only preflight in
[`scene_asset_preflight.py`](../gear_sonic/dataset_generation/scene_asset_preflight.py) verifies
manifest integrity and path confinement, content hashes, self-containment, stage metadata,
collision coverage, floor support, and route-to-obstacle clearance without importing USD or Isaac
Sim. Run the exact M0 gate from the repository root with:

```bash
python scripts/research/check_g1_dataset_scenes.py
```

These scenes are deterministic functional fixtures for controller, loading, and dataset-pipeline
bring-up. They are not photorealistic, do not model articulated or dynamic household/factory
objects, and do not represent the mesh, material, semantic, or collision complexity required for
M1 production data. Their static preflight deliberately supports this package's `Plane` plus
axis-aligned `Cube` subset; USD/PhysX-native inspection remains an additional runtime gate for
general scenes.

### 5.2 Episode request

An episode request is immutable and fully seeded:

```yaml
schema_version: 1
episode_request_id: ...
scene_id: household_apartment_0001
task_family: navigation
task_prompt: walk carefully from the living room to the kitchen
style_prompt: relaxed natural walk
route_xy: [[0.0, 0.0], [1.2, 0.1], [2.8, -0.4]]
route_frame: scene_local
nominal_speed_mps: 0.8
duration_s: 4.5
kimodo_model: Kimodo-G1
kimodo_seed: 123
simulation_seed: 456
render_seed: 789
candidate_index: 0
scene_hash: sha256:...
controller_hash: sha256:...
physics_hash: sha256:...
```

Every derived artifact stores the request ID and parent hashes. Changing a route, prompt, model, controller, scene, or physics setting produces a new episode identity.

### 5.3 Coordinate and joint-order contract

Coordinate conversion is a hard correctness boundary and must be tested independently.

- Kimodo NPZ: right-handed, Y-up, +Z-forward, meters.
- Kimodo G1 qpos CSV: right-handed, Z-up, +X-forward, meters, root quaternion WXYZ, 29 joint values in MuJoCo order.
- Isaac Lab scene local frame: right-handed, Z-up; G1 forward is +X.
- SONIC motion library: root transform plus 29 G1 values, resampled to the configured 50 Hz control rate.

The adapter should prefer Kimodo's G1 qpos CSV so the upstream converter owns the Y-up to Z-up rotation. It must not apply that transform twice.

For each route, the adapter computes `T_scene_from_kimodo`, consisting of the scene-local route start and initial heading. Kimodo receives a canonical path whose first point is at the origin. After generation, `T_scene_from_kimodo` is applied to root translation and orientation before producing the SONIC motion entry. Isaac Lab then adds each clone's `env_origin`, as the current `TrackingCommand` already expects.

The conversion output contains:

```text
root_trans_offset [T, 3]
root_rot          [T, 4]  # existing motion-lib convention
dof               [T, 29]
pose_aa           [T, 30, 3]
fps               scalar, source 30 Hz
```

Kimodo normally generates at 30 Hz. Store that true source rate and let the existing motion library interpolate to the SONIC 50 Hz target; do not relabel or duplicate frames.

Required adapter tests:

1. Identity pose and one-axis rotations map to the expected Isaac Lab joint.
2. A +1 m Kimodo-forward displacement becomes +1 m G1-forward in Isaac Lab.
3. CSV -> motion-lib -> FK agrees with the Kimodo NPZ joint positions within a documented tolerance.
4. Quaternion norms, joint limits, and frame counts remain valid after resampling.
5. A route translated and rotated in the scene lands on the original route in scene coordinates.

### 5.4 Canonical raw episode

The raw simulation record is authoritative. LeRobot is a derived training view.

```text
episodes/<episode_id>/
  request.yaml
  provenance.json
  reference/
    constraints.json
    kimodo_motion.npz
    kimodo_qpos.csv
    sonic_motion.pkl
  sim/
    trajectory.npz
    events.jsonl
    metrics.json
    outcome.json
  sensors/
    ego_rgb.mp4
    left_wrist_rgb.mp4             # optional
    right_wrist_rgb.mp4            # optional
    ego_depth.mkv                  # optional sidecar
    instance_segmentation.mkv      # optional sidecar
  export/
    lerobot_episode.json
```

The trajectory contains, at minimum:

- timestamps and episode-relative frame indices;
- joint position/velocity, root pose/velocity, projected gravity, and end-effector pose;
- applied joint targets and the SONIC 64D decoder-input latent;
- left/right hand actions;
- reference qpos/root/body pose and tracking errors;
- object and articulated-object state when present;
- foot, self, scene, and task-object contact summaries;
- termination, task-success, safety, and rejection reason codes.

Never discard rejected rollouts immediately. Quarantine them with reason labels. They are useful for controller diagnosis and possible future recovery/failure training, but they must not enter the direct successful-demonstration training split.

### 5.5 LeRobot/GR00T view

Reuse the current `UNITREE_G1_SONIC` operational action interface:

```text
action.motion_token:          64
teleop.left_hand_joints:       7
teleop.right_hand_joints:      7
```

The reduced synthetic exporter retains only the registered G1 SONIC observations:

```text
observation.images.ego_view
observation.state
observation.projected_gravity
reference.g1_qpos
```

`observation.eef_state` and `observation.root_orientation` remain part of the live-VR profile,
not the proven `synthetic_g1` training view.

Add provenance and scene/task fields either as metadata or non-training features:

```text
scene_id, layout_id, object_ids, episode_request_id
generator_model, generator_seed, controller_hash, physics_hash
task_family, task_success, rejection_code
```

Do not write fake SMPL or teleoperation arrays for a Kimodo-G1 episode. The current live-collection profile requires `teleop.smpl_pose` because it is validating VR collection. Synthetic G1 data needs a separate `synthetic_g1` feature/modality profile that omits unavailable teleop fields and adds `reference.g1_qpos`. Milestone 0 must prove that the pinned GR00T loader accepts the reduced modality. If the loader truly requires SMPL, the solution is an explicit G1-to-SMPL/SOMA conversion stage or a training-config change, not all-zero placeholder data.

The raw Isaac recorder is post-physics-step: row `t` contains the state and image after action
`t` was applied. Behavioral-cloning export therefore keeps observations `0..N-2`, shifts policy
action fields to `1..N-1`, and drops one boundary row. It also converts Isaac Lab's articulation
joint order to the registered 29-joint MuJoCo/G1 order before labeling either robot state or
`reference.g1_qpos`. Both transformations are recorded in provenance and covered by sentinel
tests; an unshifted or un-reordered export is invalid for training.

The authoritative split unit is `split_group`, normally a complete scene or layout family:

- `train`: scene/layout/object groups used for training;
- `val`: held-out groups used for model selection;
- `test_seen`: unseen routes/seeds in training scene groups;
- `test_unseen_objects`: new object asset groups;
- `test_unseen_layouts`: entirely held-out scene/layout groups.

Camera variants and Kimodo seed variants of one physical episode stay in the same split.

## 6. Scene ingestion and route generation

### 6.1 Add general USD scenes to the SONIC environment

`MySceneCfg` currently builds a plane or generated rough terrain and optional object/table assets. Extend it with a `scene_usd` mode:

```yaml
terrain_type: scene_usd
scene_usd_path: /absolute/or/resolved/scene.usd
num_envs: 1
```

Isaac Lab 2.3.2's `TerrainImporterCfg(terrain_type="usd")` imports one USD at a global terrain
prim; it does not clone the complete scene under every environment namespace. The first
`scene_usd` implementation therefore requires `num_envs: 1`. Run different scene assets in
separate batches. A later vectorized implementation must explicitly spawn the scene under
`{ENV_REGEX_NS}`, verify collision filtering and environment origins, and measure memory before
claiming full-scene parallelism.

Use `replicate_physics: false` only when layouts or dynamic-object properties truly differ per environment. Static household/factory structure should remain collision-enabled but not be authored as thousands of dynamic rigid bodies.

### 6.2 Geometry-derived walkable map

Preprocess each USD scene once, using Isaac/PhysX geometry rather than Blender:

1. Compute scene bounds or use manifest bounds.
2. Sample a configurable XY grid, initially 5 cm.
3. Raycast downward to find a support surface with an upward normal.
4. Reject cells with excessive step/slope or no support.
5. Test a configurable G1 swept envelope above each support point against static collision geometry.
6. Remove task work surfaces, machinery interiors, inaccessible islands, and dynamic-object keep-out regions.
7. Erode by the measured G1 collision footprint plus safety margin.
8. Keep connected components that intersect declared spawn/goal regions.

Semantics such as `floor` and `workstation` improve quality but are not required. The cached result includes support height, normal, clearance, connected-component ID, and scene hash, not only a Boolean mask.

### 6.3 Route sampling

For each request:

1. Sample start/goal cells from compatible named regions with minimum geodesic distance.
2. Plan A* on an 8-connected grid with diagonal corner-cutting disabled.
3. String-pull against the clearance map.
4. Smooth only if every interpolated point retains the required clearance; never smooth through walls.
5. Reject curvature or doorway-width profiles outside the configured G1 envelope.
6. Assign speed and expected frame indices from arc length.
7. Resample into Kimodo sparse waypoints or a dense root path.

Long routes should be split at low-curvature, high-clearance points into bounded-duration clips with overlap. Multi-prompt generation can join semantic phases, but every stitched transition still passes kinematic and physics validation.

### 6.4 Initial task families

Household MVP:

- walk between rooms;
- approach a table, counter, couch, or shelf;
- stop and face a named station;
- walk while carrying no object;
- crouch or side-step only in explicitly cleared regions.

Factory MVP:

- traverse aisles;
- approach and face a workcell;
- perform an inspection route with stops;
- move between pickup/drop-off stations without carrying an object;
- use styles such as careful, brisk, side-step, or crouched only where compatible with clearance.

Moving forklifts, conveyors, doors, grasping, and tool use are outside the first acceptance gate.

## 7. Kimodo generation service

Implement a local worker with a filesystem queue:

```text
requests/<id>/request.yaml
requests/<id>/constraints.json
results/<id>/motion.npz
results/<id>/motion.csv
results/<id>/generation.json
```

The worker:

1. Selects the G1 model explicitly.
2. Converts the scene-local route to canonical Kimodo root constraints.
3. Computes duration from route length and nominal speed.
4. Generates `K` candidates per route with distinct seeds, initially 4.
5. Records whether Kimodo post-processing was available and enabled; the currently audited local
   G1 path disables it, so kinematic gates remain mandatory.
6. Exports both NPZ for diagnostics and G1 qpos CSV for the SONIC adapter.

Before simulation, candidates pass inexpensive kinematic gates:

- finite values and valid shapes;
- joint-limit and angular-velocity checks;
- root-path start/end and lateral-deviation checks;
- foot contact consistency and foot-skate estimate;
- robot swept-volume clearance against the cached scene map;
- maximum speed, acceleration, and turn-rate checks;
- no sustained root floating or floor penetration.

Kinematic gates save simulation time but never confer final acceptance.

## 8. SONIC physics execution

### 8.1 Execution mode

Use the released SONIC checkpoint in evaluation mode, loaded through the existing manager environment and action-transform module. Each environment receives one motion ID and starts at frame zero. Disable training reset noise for validation rollouts, but support a separate robustness pass with registered perturbations.

The controller runs at the repository's configured 50 Hz. Physics may use a smaller integration step through the existing Isaac Lab decimation settings. All physics parameters are serialized into `physics_hash`.

The recorder must capture the actual decoder input:

- In teacher/residual mode, `env._full_latent` is the encoded/quantized SONIC latent after any configured residual path.
- The policy's `meta_actions[:, :64]` can be a residual and is not automatically equivalent to deployment `token_state`.

#### 8.1.1 Latent representation parity with the C++ deployment path

This is now resolved for the released checkpoint; see
[`latent_parity.py`](../gear_sonic/dataset_generation/latent_parity.py) and
[`test_latent_parity.py`](../tests/dataset_generation/test_latent_parity.py).

**Same slot, same representation, same layout.** Both code paths write the decoder's flattened
token input:

| | Python training/eval | C++ deployment |
|---|---|---|
| Producer | `UniversalTokenModule._last_full_latent_flat` -> `env._full_latent` | encoder ONNX output `encoded_tokens` -> `token_state` observation |
| Value | post-quantization FSQ codes | post-quantization FSQ codes |
| Flatten | `all_tokens.view(*shape[:-2], -1)` | `selected_tokens.flatten(start_dim=-2)` |

Both flattens are row-major over `(max_num_tokens, token_dim)`, so token 0 occupies `[0:32]` and
token 1 occupies `[32:64]`. The released `actor_critic/universal_token/all_mlp_v1` declares
`max_num_tokens: 2` and `num_fsq_levels: 32`, giving the 64 values that
`gear_sonic_deploy/policy/release/observation_config.yaml` declares as `encoder.dimension`. A test
binds those two numbers so a change on either side fails CI rather than silently corrupting data.

**Measured result: the released eval configuration records pure encoder tokens.** A rollout on
2026-08-15 (`factory_aisle`, released checkpoint, one CUDA PhysX environment) recorded a 64D latent
that is **100% on the FSQ lattice**: max lattice deviation exactly `0`, residual RMS exactly `0`,
25 distinct values, all `k/16` with `k` in `[-13, 11]`. The recorded `action.motion_token` is
therefore byte-comparable with a C++ local-encoder `token_state`.

The mechanism is that `manager_env.config.action_transform_module_cfg` is null in every checked-in
config, so the wrapper's own ATM is `None`. `step()` then takes its `full_latent` pass-through
branch and stores the value the actor already computed, and the actor's internal ATM forward runs
with `latent_residual=None`. The residual code path in `step()` is unreachable in this
configuration.

**The residual path is still a live hazard for other configurations, which is why the gate stays.**
The C++ runtime contains no residual path at all: `token_state` is either the raw encoder ONNX
output or an external vector supplied over ZMQ/ROS2. But when a wrapper-level ATM *is* configured,
`step()` falls back to `action_mode="residual"` *even when `use_latent_residual` is false*, with
`latent_residual_mode` defaulting to `post_quantization` at scale 1.0. That configuration would
record `FSQ_codes + policy_residual` under the same feature name. This was a code-reading
prediction, and the measurement above falsified it *for the released eval config only* — not in
general.

**The two are numerically distinguishable, which makes this gateable.** `vector_quantize_pytorch.FSQ`
with `levels=[32]*32` emits values on an exact lattice — multiples of `1/16` in `[-1.0, +0.9375]`,
32 distinct values, exact in float32. This was verified against the installed implementation, not
assumed from the paper. So:

- every value on-lattice -> representation-identical to a local-encoder `token_state`;
- any value off-lattice -> a residual-carrying decoder input.

Out-of-range values are reported as a statistic, not as a separate representation class: an
unbounded post-quantization residual routinely pushes a token past the reachable code range, and
treating that as "unidentified" would mislabel legitimate residual data.

**Consequence for the dataset.** The live-VR collection path (`run_data_exporter.py`) writes
`action.motion_token` from deployment `token_state`, which is encoder-sourced and on-lattice. The
synthetic path writes `env._full_latent`, measured above as also on-lattice. **In the released
configuration the two conventions coincide**, so synthetic and real episodes can be mixed under one
feature name without a semantic conflict. That is the good outcome, and it is now a measured fact
rather than an assumption.

It is not, however, a property to take for granted: attaching a wrapper-level ATM, enabling
`use_latent_residual`, or switching to a student `direct_latent` policy would each change the
recorded convention while every shape, dtype, and schema check continued to pass. The mitigation is
therefore recording plus an opt-in gate. Every export embeds a `provenance.latent_representation`
block naming the contract and the measured verdict, and `--require-encoder-equivalent` fails any
dataset whose tokens leave the lattice.

### 8.2 New episodic recorder

The current `TrajectoryRecorderTerm` stores all environments until shutdown and writes one file per environment. Large-scale generation needs `SyntheticEpisodeRecorderTerm` with independent per-environment lifecycle:

- start buffer on reset;
- append at every control step;
- finalize immediately on success, failure, timeout, or motion end;
- write bounded chunks asynchronously so GPU memory does not grow with the run;
- attach request, route, motion, scene, controller, and seed IDs;
- store reason-coded outcomes for accepted and rejected episodes;
- support physics-only and render modes with identical timestamps.

The recorder should write the canonical raw episode first. A separate exporter produces LeRobot files after validation.

### 8.3 Two-pass simulation

Use two passes:

1. **Physics pass:** no cameras, many cloned environments, all state/contact metrics recorded.
2. **Render pass:** replay only accepted trajectories with ego and optional wrist/external cameras, fewer environments, RGB plus optional depth/segmentation.

This prevents camera rendering from becoming the acceptance-loop bottleneck and ensures every expensive render already corresponds to a physically accepted behavior. The render pass replays the accepted physical trajectory rather than re-running an uncontrolled episode.

## 9. Validation and acceptance

Validation has four layers.

### 9.1 Artifact validation

- Schema version and all parent hashes present.
- No missing frames, NaN, infinity, or non-monotonic timestamps.
- Joint, latent, hand-action, image, and object-state shapes match the declared embodiment.
- Physics and render frame counts align after documented resampling.

### 9.2 Robot safety and dynamic feasibility

- No fall termination or base-height collapse.
- Roll/pitch, joint position/velocity, action, torque estimate, and contact impulse remain within configured bounds.
- No disallowed self-collision or robot-scene contact.
- Allowed foot-ground contacts remain stable; sustained foot skate fails the episode.
- The SONIC tracking error stays within calibrated bounds.

### 9.3 Route and scene validity

- Executed root begins and ends near the requested route endpoints.
- Mean and 95th-percentile lateral path deviation stay under configured limits.
- Executed body clearance remains inside the validated traversable corridor.
- No wall, furniture, or machinery penetration occurs.
- The robot stops in the requested goal region and faces the requested final heading when applicable.

### 9.4 Task success

Each task family supplies a deterministic predicate. Navigation success is goal-region occupancy plus stability for a hold window. Later manipulation predicates use object pose, containment/contact state, grasp continuity, and robot stability rather than video-only judgment.

Initial numerical thresholds are provisional defaults, not claims of validated performance. Calibrate them from a manually reviewed 100-episode pilot, then freeze them in the dataset version. The pilot should start near:

- endpoint error: at most 0.35 m;
- 95th-percentile lateral deviation: at most 0.25 m;
- stable goal hold: at least 0.5 s;
- no disallowed collision and no fall;
- finite, monotonic, 50 Hz control record.

Record every individual gate result. A single `success: false` flag is insufficient for improving the generator or controller.

## 10. Manipulation extension

Kimodo alone is not a manipulation-data engine. Its end-effector constraints can create useful whole-body approach, reach, and gesture references, but object-relative contact must be generated and verified by a task layer.

The recommended extension is a hierarchical episode:

```text
Kimodo + SONIC locomotion
        -> stop/stance transition
        -> object-relative reach trajectory
        -> grasp/contact controller
        -> carry or manipulate with SONIC balance
        -> release and retreat
```

Use Isaac Lab Mimic/SkillGen or scripted task-space planning for the contact phases:

- Start with a small set of human or scripted successful manipulation seeds.
- Annotate object-relative subtasks.
- Generate spatial variants with Mimic.
- Convert the resulting end-effector goals into the SONIC teleop/three-point command path, then record the actual 64D SONIC latent and hand commands.
- Retain Isaac physics task predicates as the acceptance authority.

The upstream G1 locomanipulation workflow is a useful reference, but integration must be tested against Isaac Lab 2.3.2 and this repository's Unitree G1/SONIC action space. Its task-space or controller actions must not be mislabeled as SONIC latent actions.

Kimodo's G1 qpos export contains 29 body joints, not the deployed dexterous hand commands. The task layer must produce the two 7D hand actions explicitly. For locomotion-only episodes they should use a declared neutral/open-hand policy, not arbitrary random values.

## 11. Domain randomization

Randomization is applied through a versioned profile and recorded per episode.

Safe MVP randomizations:

- light intensity, color temperature, dome texture, and exposure;
- camera intrinsics/extrinsics within calibrated ranges;
- materials, colors, and texture variants;
- distractor placement outside the traversable corridor;
- static object placement inside declared layout constraints;
- floor friction and robot mass within conservative bounds;
- Kimodo prompt, style, speed, route, duration, and seed.

Do not randomize all factors at once during bring-up. Add one family at a time and compare acceptance rate, path error, contact statistics, and visual coverage against a fixed nominal suite.

Dynamic hazards require a time-aware planner and are a separate task family; a static occupancy grid is not sufficient.

## 12. Proposed code organization

```text
configs/dataset_generation/
  scenes.yaml
  g1_household_locomotion_mvp.yaml
  g1_factory_locomotion_mvp.yaml
  validation_v1.yaml

gear_sonic/dataset_generation/
  schemas.py                    # typed/versioned manifests
  scene_registry.py             # asset resolution, hashes, licenses
  scene_preprocess.py           # PhysX walkability/clearance cache
  route_sampler.py              # A*, smoothing, duration assignment
  kimodo_worker.py              # isolated CLI/filesystem adapter
  kimodo_motion_adapter.py      # qpos -> SONIC motion-lib
  latent_parity.py              # 64D token contract vs. C++ deployment
  rollout_runner.py             # vectorized SONIC physics pass
  validators.py                 # kinematic, physics, route, task gates
  render_replay.py              # accepted-trajectory sensor pass
  episode_store.py              # canonical raw layout
  lerobot_export.py             # synthetic_g1 profile

gear_sonic/envs/manager_env/mdp/
  recorders.py                  # add SyntheticEpisodeRecorderTerm

gear_sonic/scripts/
  generate_synthetic_g1_dataset.py

tests/dataset_generation/
  test_coordinate_contract.py
  test_kimodo_qpos_adapter.py
  test_latent_parity.py
  test_route_clearance.py
  test_episode_schema.py
  test_split_leakage.py
  test_lerobot_export.py
```

Extend `modular_tracking_env_cfg.py` minimally for general scene USD spawning, scene cameras, and collision categories. Keep route planning, Kimodo calls, validation, and export out of the environment configuration module.

The planned top-level command is:

```bash
python gear_sonic/scripts/generate_synthetic_g1_dataset.py \
  --config configs/dataset_generation/g1_household_locomotion_mvp.yaml \
  --stage all
```

Stages must also run independently: `preprocess`, `sample`, `generate`, `convert`, `simulate`, `validate`, `render`, and `export`.

## 13. Milestones and gates

### M0: representation and two-scene spike

Deliver:

- Kimodo G1 CSV -> SONIC motion-lib adapter;
- coordinate/joint-order golden tests;
- one simple room USD and one factory aisle USD;
- straight, curved, and stop/turn routes;
- SONIC physics rollouts with latent capture;
- one household and one factory synthetic LeRobot export readable by the pinned GR00T loader.

Gate:

- all coordinate and schema tests pass;
- manual FK comparison finds no axis or joint-order error;
- at least 20 accepted, visually reviewed physics episodes across both scenes;
- recorded `action.motion_token` is demonstrated to match deployment semantics, and its
  representation verdict is recorded in export provenance (section 8.1.1);
- no fake SMPL data is required by the proven training path.

### M1: locomotion dataset MVP

Deliver:

- cached geometry-derived walkability for at least 5 household and 5 factory layouts;
- route/prompt/seed candidate generation;
- asynchronous episodic recorder;
- reason-coded physics validation;
- accepted-only render replay with ego RGB;
- split-safe LeRobot export and dataset report.

Gate:

- 10,000 accepted episodes or an explicitly smaller pilot version;
- 100% artifact/schema validation;
- acceptance, rejection, and throughput statistics by scene and prompt family;
- manual review of a stratified sample with documented false-accept/false-reject findings;
- held-out layouts are absent from all training hashes.

### M2: approach and pose-constrained tasks

Deliver:

- task stations and final-heading/stance constraints;
- Kimodo hand/end-effector or full-body keyframe constraints;
- reach-without-contact, inspect, crouch, and workstation-approach tasks;
- wrist cameras and depth/segmentation sidecars as needed.

Gate:

- deterministic task predicates;
- no increase in unsafe-contact false accepts;
- task success and controller tracking reported separately.

### M3: contact-rich locomanipulation

Deliver:

- one household pick/place task and one factory transfer task;
- Mimic/SkillGen or scripted task-space seed pipeline;
- explicit 7D hand actions;
- object-relative subtask annotations and success predicates;
- failed-contact quarantine set.

Gate:

- object task success under physics, not kinematic replay;
- no action-space mismatch between Mimic, SONIC, and LeRobot export;
- held-out object/layout evaluation.

### M4: learning validation and scale-out

Deliver:

- GR00T fine-tuning comparison: real-only, synthetic-only, and real+synthetic;
- SONIC tracking analysis on accepted versus rejected Kimodo motion families;
- multi-GPU/multi-machine job queue and resumable manifests;
- dataset card with licenses, generator/controller versions, limitations, and bias analysis.

Gate:

- improvement on held-out physics tasks without regression on the real-data benchmark;
- reproducible dataset regeneration from manifests and pinned artifacts.

## 14. Validation commands to add

Unit and file-format tests should run outside Isaac Sim where possible:

```bash
pytest tests/dataset_generation -q
python scripts/research/check_g1_dataset_scenes.py
python scripts/research/check_sonic_vla_dataset.py /path/to/exported_dataset
```

Add a synthetic-profile flag or a new checker rather than weakening the live VR dataset checks globally:

```bash
python scripts/research/check_synthetic_g1_dataset.py \
  /path/to/exported_dataset \
  --episode-store /path/to/raw_episodes
```

Classify the recorded 64D latent against the deployment token contract (section 8.1.1). This runs
offline, needs neither Isaac Sim nor a TensorRT build, and also cross-checks the token width against
the C++ deploy observation config:

```bash
python scripts/research/check_latent_parity.py --trajectory /path/to/trajectory.pkl
python scripts/research/check_latent_parity.py --dataset /path/to/exported_dataset

# For a dataset that will be mixed with real teleop episodes, require the
# encoder-sourced convention instead of a residual-carrying decoder input:
python scripts/research/check_latent_parity.py \
  --dataset /path/to/exported_dataset --require-encoder-equivalent
```

The Isaac integration smoke test uses one environment, one short route, cameras disabled, and a fixed seed. A second smoke test replays its accepted trajectory with the ego camera and checks exact frame alignment.

## 15. Main risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Kimodo motion is plausible but dynamically hard | High rejection rate or falls | Generate multiple candidates; prefilter; track with SONIC; use rejection telemetry to narrow prompts/speeds. |
| Coordinate or joint-order error | Quietly corrupts all data | Golden FK and axis tests before any scale run; hash adapter version into every episode. |
| SONIC latent semantics differ between Python and C++ | Invalid VLA targets | Resolved for the released checkpoint (section 8.1.1): same slot, same post-quantization FSQ representation, same flatten order, width bound to the deploy observation config by test. The residual asymmetry is detected from the FSQ lattice and recorded in export provenance. |
| Real and synthetic episodes carry different `action.motion_token` conventions | VLA learns two incompatible action semantics from one feature name | `check_latent_parity.py` classifies every artifact; export provenance records the verdict; `--require-encoder-equivalent` gates datasets that will be mixed with real teleop data. |
| Scene USD has poor collisions or scale | Clipping and invalid navigation | Scene-ingestion gate, collision visualization, unit/up-axis checks, cached clearance audit. |
| Heavy scenes make vectorization/rendering expensive | Low throughput or OOM | Batch one scene at a time; physics/render two-pass; tune environment count from measured VRAM. |
| Current recorder buffers until shutdown | Memory growth and lost episodes | Add per-env episodic flush and atomic completion markers. |
| Synthetic profile conflicts with current SMPL-rich contract | Loader failure or fake data temptation | Prove a reduced `synthetic_g1` modality in M0; never substitute zero SMPL as real supervision. |
| Kimodo has no dexterous hand trajectory | Invalid manipulation labels | Separate body reference from task-layer 7D hand commands; defer contact tasks to M3. |
| Upstream Mimic examples differ from Isaac Lab 2.3.2 | Integration churn | Pin versions; port only required APIs; add an adapter boundary rather than upgrading the whole repo. |
| Scene/object leakage across splits | Inflated evaluation | Split by scene/layout/object hash before route sampling; enforce with an automated leakage test. |
| Asset/model licensing is unclear | Dataset cannot be redistributed | License manifest and redistribution flag for every scene, object, texture, model, and generated artifact. |

## 16. Recommended first implementation slice

Build M0 only, in this order:

1. Add typed request/result schemas and hash-based episode IDs.
2. Implement and test Kimodo G1 qpos conversion using three hand-authored golden motions.
3. Add `scene_usd` spawning for one simple room and one aisle.
4. Use manually authored clear routes first; do not block controller integration on automatic walkability extraction.
5. Run SONIC tracking in one Isaac Lab environment and record root/joints/contacts/`env._full_latent`.
6. Add physics and route gates, then replay accepted trajectories with the existing ego camera.
7. Export one `synthetic_g1` LeRobot dataset and load a batch through the pinned GR00T training code.
8. Only after those gates pass, implement geometry-derived route generation and parallel rollout.

This slice resolves the three highest-risk unknowns early: coordinate correctness, SONIC latent correctness, and GR00T schema compatibility. Once those are proven, scene count and motion count become scaling work rather than architecture work.

## 17. M0 implementation status (2026-08-14)

Completed in the first slice:

- Added immutable, versioned `EpisodeRequest`, `GenerationResult`, and `ArtifactRef` contracts
  with canonical JSON SHA-256 identities and tamper checks.
- Added a Kimodo 36-column qpos CSV adapter that validates WXYZ quaternions, preserves the
  verified 29-joint MuJoCo order, applies the scene-local start/yaw transform, and emits the
  existing SONIC motion-library schema.
- Added a conversion CLI that writes the joblib motion and a manifest containing input/output
  artifact hashes.
- Added two redistributable, repo-authored M0 scene fixtures: a primitive household room and a
  primitive factory aisle. Both are self-contained, metre-scale, Z-up USDA with `/World` as the
  default prim, collision-backed support at `z=0`, and documented G1-clear canonical routes.
- Added a scene-package manifest with content hashes, byte sizes, provenance, Apache-2.0
  redistribution metadata, split groups, support/walkable bounds, and route-clearance contracts,
  plus a dependency-free preflight/checker for integrity, stage, collision, floor, and static-route
  validation.
- Added the initial `scene_usd` terrain mode and its deliberate one-environment guard.
- Added offline golden tests for schema identity, qpos shape/quaternion validation, root-frame
  transforms, exact joint order, DOF axes, joblib output, conversion manifests, and USD path
  validation.
- Added a reduced `synthetic_g1` LeRobot feature/modality profile matching GR00T's registered
  `unitree_g1_sonic` inputs, plus validator support that forbids fabricated SMPL/VR fields.
- Expanded the trajectory recorder into schema v2 with real physics state, applied joint action,
  64D SONIC decoder-input latent, 36D G1 reference qpos, reference time/ID, and tracking metrics.
- Added a production head-mounted `ego_camera` recorder at 640x480 and 50 Hz. Camera and
  trajectory terms now both retain the first post-step frame, so an export cannot silently hide a
  one-frame offset.
- Added raw per-body contact vectors plus filtered left/right ankle-to-support-floor sensors.
  Reason-coded locomotion acceptance now covers episode length, reference displacement, signed
  executed progress, endpoint and p95 path error, root height/tilt, non-foot contact,
  foot-to-non-ground contact, and upward floor support. Derived caller summaries are not trusted;
  rejection reports identify the peak frame and robot body from raw evidence.
- Made the trajectory exporter enforce the physics-acceptance report instead of relying on a
  caller-provided success label, and preserved that report in the exported dataset provenance.
- Added provenance-preserving half-open subclip export. It validates and fully decodes the original
  trajectory/video, recursively slices every aligned raw array, re-runs schema and physics gates on
  the selected interval, performs the declared one-frame causal action shift, converts Isaac Lab
  joint arrays to MuJoCo order, and records every source/action/observation range.
- Runtime success manifests now content-bind the exact checkpoint, task, scene, motion input,
  calibrated ego-camera configuration, trajectory, and video. Export additionally requires a
  verified typed `EpisodeRequest -> GenerationResult -> ConversionResult` bundle and cross-checks
  it against the runtime hashes and task/scene claims.
- Added strict raw-trajectory and GR00T file gates. The latter requires aggregate statistics,
  float32 training features, a decodable 640x480 video with exact declared frame count/FPS, and at
  least the registered 40-frame SONIC action horizon.
- Added an optional official-loader smoke that instantiates GR00T's
  `ShardedSingleStepDataset(UNITREE_G1_SONIC)`, extracts a real step without padding, and verifies
  the registered state, 40-step action, RGB, and language shapes.
- Pinned the Python 3.10 inference extra to Isaac-GR00T revision
  `ab88b50c718f6528e1df9dcbaf75865d1b604760`. Following upstream `main` was no longer safe
  because upstream migrated to Python 3.12 after that SONIC-capable revision.
- Made video-writer shutdown explicit and joined encoder workers so completed batch exports do
  not leave PyAV threads and containers alive during interpreter teardown.
- Resolved the 64D latent representation question against the C++ deployment path (section 8.1.1)
  and added the offline gate that enforces it: a derived FSQ codebook lattice, a
  representation classifier, a cross-repository binding to the deploy observation config, a
  `provenance.latent_representation` block on every export, and a
  `check_latent_parity.py` CLI with a strict `--require-encoder-equivalent` mode.

Verified locally:

- The Kimodo and SONIC G1 MJCFs have the same 29 actuated joint names and order, with matching
  axes, limits, parent links, and link transforms.
- On Kimodo's 241-frame `06_root_waypoints` G1 demo, the audited Kimodo exporter followed by this
  adapter and SONIC forward kinematics produced 1.81 mm mean, 5.68 mm p95, and 21.45 mm maximum
  joint-position disagreement, with zero root-position error. These values provide the first
  measured adapter tolerance; more motion families should be sampled before freezing the gate.
- Isaac Sim 5.1 / Isaac Lab 2.3.2 ran the released SONIC checkpoint in one CUDA PhysX
  environment at a 5 ms physics step and 20 ms control step. Both the plane mode and the guarded
  one-environment `scene_usd` mode reached an explicit rollout-success marker.
- Fresh one-environment factory and household rollouts used the official bundled
  Kimodo-G1-RP single-text example, the released SONIC checkpoint, CUDA PhysX, each repo-owned
  scene, and a true link-mounted Isaac ego camera. Each raw trajectory and H.264 video contains
  exactly 175 frames at 640x480/50 Hz. The household input rotates the same motion onto that
  scene's canonical route. Filtered ankle sensors resolve the composed support prim
  `/World/ground/terrain/Structure/Floor` and provide nonzero upward floor forces in both scenes.
- Each complete 175-frame recorded rollout remains quarantined because frame 109 contains a
  non-foot hip/wrist contact: 320.15 N in the factory and 317.02 N in the household. These runtime
  captures span timestamps 0.00 through 3.48 seconds of the nominal five-second, 150-frame/30 Hz
  converted reference; they are not the complete Kimodo motion. The half-open raw interval
  `[43, 109)` independently passes all ten gates in both scenes. Factory metrics are 66 frames, 1.941 m
  reference displacement, 0.926 signed executed-progress ratio, 0.217 m endpoint error, 0.216 m
  p95 path error, 0.645 m minimum root height, 0.284 rad maximum tilt, zero non-foot force, zero
  foot/non-ground residual, and 95.5% supported frames. Household metrics are 1.941 m, 0.925,
  0.220 m, 0.219 m, 0.645 m, 0.283 rad, zero, zero, and 95.5%, respectively.
- Causal alignment produces 65 training rows: observation/reference frames `43..107` pair with
  policy actions `44..108`. Independent first/last-row checks found zero numeric error for state,
  projected gravity, motion token, and reference qpos after the audited IsaacLab-to-MuJoCo joint
  permutation. Earlier pre-audit LeRobot exports are explicitly invalid for training.
- Each final LeRobot v2.1 export has 65 float32 rows and a 65-frame, 640x480, 50 Hz H.264 ego
  video. The strict `synthetic_g1 --groot-loader-ready` file gate reports zero errors or warnings
  for both. Exact-source smokes at pinned Isaac-GR00T revision
  `ab88b50c718f6528e1df9dcbaf75865d1b604760` each construct one shard with 26 valid 40-step
  samples; registered state/action/video/language shapes all match. The isolated smoke stubs only
  the imported-but-unused `transformers.ProcessorMixin`; official dataset and decoder code are
  unmodified because a durable full GR00T install is not yet present.
- Export provenance verifies and embeds the explicit final/post-hoc `EpisodeRequest`, its
  `GenerationResult`, and the Kimodo-to-SONIC `ConversionResult`, then cross-binds those identities
  to the runtime task, scene hash, controller/config hashes, byte-identical motion input,
  trajectory/video pair, and ego-camera settings. This post-hoc request must not be represented as
  the original prompt record for the bundled Kimodo inference.
- The complete offline dataset-generation suite passes without importing Isaac Sim (156 tests).
  Lightweight scene/trajectory validators no longer eagerly import the optional joblib-based
  Kimodo adapter.
- `vector_quantize_pytorch.FSQ` with the released `levels=[32]*32` emits values on an exact
  lattice: multiples of `1/16` in `[-1.0, +0.9375]`, 32 distinct values, exact in float32. This was
  measured against the installed implementation over 4096 saturating samples, not taken from the
  paper. It is what makes an encoder-sourced `token_state` and a residual-carrying decoder input
  distinguishable from a recorded array alone.
- The released actor config (`max_num_tokens: 2` x `num_fsq_levels: 32`) and the deploy runtime
  config (`encoder.dimension: 64`) agree, and a test now fails if either side drifts.
- The C++ deploy runtime contains no residual path: `token_state` is either the encoder ONNX
  `encoded_tokens` output or an external vector supplied over ZMQ/ROS2.

## 18. Rollout replication and latent measurement (2026-08-15)

Two fresh one-environment rollouts were run from a checked-in reproducible runner,
[`run_kimodo_sonic_rollout.sh`](../scripts/research/run_kimodo_sonic_rollout.sh), using the bundled
Kimodo-G1-RP `01_single_text_prompt` demo ("A person walking forward quickly stumbles but maintains
their balance", 150 frames at 30 Hz, seed 43), the released SONIC checkpoint, and both repo-owned
scenes. Kimodo's own `MujocoQposConverter` produced the 36-column qpos CSV; the SONIC adapter placed
it on each scene's canonical route (factory at `(-7, 0)` yaw 0; household at `(0, -3.2)` yaw pi/2).

Each rollout recorded 259 frames at 50 Hz with the `SONIC_EVAL_SUCCESS` marker, a runtime success
manifest, a trajectory, and an ego video. The motion library resampled the 150-frame 30 Hz source to
249 frames at 50 Hz, and frame 248 is a reset boundary, so `[0, 249)` is exactly one motion pass.

### 18.1 Latent parity, measured

Both scenes independently recorded a 64D latent that is **100% on the FSQ codebook lattice**: max
lattice deviation exactly `0`, residual RMS exactly `0`, 25 distinct values, all `k/16` with `k` in
`[-13, 11]`, over 259 frames x 64 dims each. The verdict is `deployment_encoder_equivalent`.

This **falsifies the code-reading prediction** recorded earlier in section 8.1.1 that the recorded
latent would carry the policy residual. The residual branch in `step()` is unreachable while
`action_transform_module_cfg` is null, which it is in every checked-in config. The gate caught the
error, which is what it is for.

### 18.2 Acceptance, and an uncalibrated threshold

On the full 249-frame pass, **9 of 10 gates pass in both scenes** with comfortable margins:

| Gate | factory_aisle | household_room | Threshold |
|---|---|---|---|
| reference displacement | 3.702 m | 3.702 m | >= 0.25 m |
| executed progress ratio | 0.947 | 0.944 | >= 0.5 |
| endpoint error | 0.200 m | 0.219 m | <= 0.35 m |
| p95 path error | 0.225 m | 0.230 m | <= 0.25 m |
| min root height | 0.646 m | 0.643 m | >= 0.5 m |
| max root tilt | 0.283 rad | 0.283 rad | <= 0.6 rad |
| foot support fraction | 0.988 | 0.984 | >= 0.9 |
| **non-foot contact** | **33.8 N** | **268.8 N** | **<= 1.0 N** |

The only failing gate is `disallowed_robot_contact`. The contacting bodies are
`left/right_hip_roll_link` and `left/right_wrist_yaw_link` — **arm-brushing-hip self-contact
intrinsic to the walking gait, not scene collision** — and they exceed 1 N on 17.3% (factory) and
18.1% (household) of frames. A 1 N non-foot threshold therefore rejects normal SONIC locomotion.
This is the calibration gap section 9.4 anticipated: the threshold is a provisional default, and the
pilot must either raise it, restrict it to scene-object contacts, or exempt a declared
self-contact body set. Until then, whole-episode acceptance is dominated by this one gate.

The longest contact-free windows pass all ten gates:

- `factory_aisle` `[111, 183)`, 72 frames: 1.031 m displacement, 1.037 progress ratio, 0.197 m
  endpoint error, 0.227 m p95 path error, 0.683 m root height, 0.153 rad tilt, 1.000 foot support.
- `household_room` `[43, 109)`, 66 frames: 1.941 m displacement, 0.923 progress ratio, 0.224 m
  endpoint error, 0.222 m p95 path error, 0.643 m root height, 0.283 rad tilt, 0.939 foot support.

The household window and its metrics **reproduce the 2026-08-14 M0 result to three decimal places**
from a freshly exported Kimodo CSV and a fresh rollout, which is independent evidence that the
Kimodo-to-SONIC path is deterministic.

### 18.3 The contact gate was measuring the wrong thing

The `disallowed_robot_contact` gate rejected both episodes at a 1.0 N threshold. Investigating that
rejection produced the most consequential correction so far.

**The sensor cannot answer the question the gate asks.** Isaac's contact sensor records
`net_forces_w` per body: the *sum* of every contact acting on that body. One number cannot separate
"the wrist swung into its own hip" from "the hip struck a wall", yet those must be treated
completely differently — the first is a gait artifact, the second is a navigation failure.

**A control run settled it.** The same motion was run on bare `plane` terrain with no obstacles.
Peak non-foot contact there was **309.2 N — higher than either scene** (factory 33.8 N, household
268.8 N), at the same frames. Whatever was triggering the gate was not scene geometry.

**Newton's third law identifies the mechanism exactly.** `left_hip_roll_link` and
`left_wrist_yaw_link` recorded *bit-identical* force magnitudes and exactly negated vectors, as did
the right-side pair. Two tracked bodies of the same articulation booking equal and opposite net
force is an action-reaction pair: **the robot's wrists are striking its own hips during arm swing.**
Across all three rollouts every non-foot contact matched a partner to within 1e-4 N, and the
**unpaired (robot-to-environment) non-foot force was exactly 0.0000 N**. The robot never touched a
wall, rack, or pallet in any run.

**The fix uses that structure.**
[`contact_decomposition.py`](../gear_sonic/dataset_generation/contact_decomposition.py) greedily
matches per-frame forces into `f_a ~= -f_b` pairs. Matched force is self-contact; the unmatched
residual is environment contact, because scene geometry is not a sensor body and so has no
counterpart column. The single conflated gate is now two:

- `nonfoot_scene_contact` — unpaired non-foot force, **stays strict at 1.0 N**. This is the real
  navigation safety gate and it is what section 9.2 always intended.
- `self_contact` — paired force, provisional threshold anchored to nominal G1 body weight
  (~35 kg, ~343 N) as a physical scale, explicitly **not** calibrated. A self-impact exceeding body
  weight is unambiguously abnormal; below that it is a quality signal, not a safety failure.

The undecomposed maximum is retained in diagnostics as `raw_max_nonfoot_contact_force_n` so the
change stays auditable against pre-decomposition records.

**Effect.** Both full 249-frame passes now clear all eleven gates, where previously only 66- and
72-frame contact-free subclips survived — roughly 3.5x more usable data per rollout, obtained by
measuring the right quantity rather than by relaxing a threshold. The decomposition's limits are
documented in the module: a body touching scene and self in the same frame leaves an unpaired
residual that is attributed to external contact, which over-reports rather than hides a collision.

Still open: the `self_contact` threshold is a physical anchor, not a calibration. A 269-309 N
wrist-to-hip impact is a real tracking-quality concern even though it destabilises nothing
(root height 0.64 m, tilt 0.28 rad, 98% foot support). The reviewed pilot should decide whether to
tighten it, and whether repeated self-impact should downgrade an episode rather than reject it.

### 18.4 Sixteen-rollout batch: acceptance statistics

All eight bundled Kimodo-G1-RP demos were exported to qpos CSV, converted onto both scene routes,
and rolled out — sixteen rollouts, including demos expected to fail, because honest reject
statistics are worth more than a curated pass list. Analysed with
[`analyze_kimodo_rollout_batch.py`](../scripts/research/analyze_kimodo_rollout_batch.py).

**Latent parity: 16/16 `deployment_encoder_equivalent`, residual RMS 0.** With the three earlier
rollouts that is 19 independent confirmations across eight motions, two scenes, and a bare plane.
The section 18.1 result is not a fluke of one clip.

**Acceptance: 2/16.** Rejection reasons, most frequent first: `reference_path_tracking_error` (8),
`disallowed_robot_contact` (7), `excessive_self_contact` (4), `reference_endpoint_tracking_error`
(4), `not_locomotion` (4), `insufficient_executed_motion` (3), `fall_root_height` (2),
`fall_root_tilt` (2), `disallowed_foot_non_ground_contact` (2), `episode_too_short` (2).

These rejections are almost all correct:

- `07_text_terrain` climbs to z = 1.96 m — a jump. On flat ground the robot fell after 13 frames and
  was rejected as `episode_too_short` + `not_locomotion`.
- `08_text_object` translates 0.02 m. Correctly `not_locomotion`.
- `05_root_path` and `06_root_waypoints` carry 1.2-2.5 m of lateral deviation and walked into racks
  and walls.

**The decomposed contact gate discriminates in both directions**, which is the evidence that
section 18.3 loosened nothing:

| rollout | external (scene) N | self N | colliding bodies |
|---|---|---|---|
| `01_single_text_prompt` (both scenes) | **0.00** | 33.8 / 268.8 | none |
| `03_full_body_keyframes` (both scenes) | **0.00** | 282.5 / 282.8 | none |
| `08_text_object` (both scenes) | **0.00** | 911.0 / 893.0 | none |
| `04_ee_constraint` household | **0.00** | 61.1 | none |
| `04_ee_constraint` factory | **1060.2** | 61.3 | elbow, hip_roll, wrist_yaw |
| `05_root_path` factory | **1849.9** | 91.6 | knees, torso, hips, elbows, wrists |
| `06_root_waypoints` factory | **1576.1** | 27.1 | pelvis, torso, knees, hips, elbows |

Two results carry the argument. `08_text_object` records **911 N of self-contact with 0.00 N of
scene contact** — violent flailing in place, which the old conflated gate could not have told apart
from a wall strike. And `04_ee_constraint` records **0.00 N in the household room but 1060 N in the
factory aisle** from the *same motion*: a scene-dependent collision signal that only exists once the
two quantities are separated.

**Yield is limited by placement, not by the gates.** Every motion is dropped onto a scene's route
start with a fixed yaw, so a reference that curves 2.5 m laterally will hit something no matter how
good the controller is. Reaching twenty accepted episodes needs the geometry-derived route sampling
of section 6.3 — matching references to routes that can actually contain them — not more demo clips
or looser thresholds.

### 18.5 Runtime configuration defects found

Three defects made the first attempt silently wrong, and the runner now guards all three:

- **`render_frame_skip` defaults to 2**, so the recorder writes at 25 Hz. `trajectory_export.py`
  requires exactly 50 Hz, so the trajectory was un-exportable while every other check passed. The
  runner forces `render_frame_skip=1` and then re-reads the recorded `fps` rather than trusting the
  override.
- **The `gear_sonic` editable install in `env_isaaclab` resolves to a different checkout**
  (`groot-wbc-sonic-sim-trackb`), and its static editable finder does not know about subpackages
  added after install. `eval_agent_trl.py`'s own `sys.path.append(os.getcwd())` appends too late to
  win. Without `PYTHONPATH` set to this repository the rollout runs another checkout's code; the
  runner asserts the resolved `gear_sonic.__file__` before launching.
- **The `SONIC_EVAL_SUCCESS` marker and runtime manifest require `+success_manifest=<path>`.**
  Without it the run exits cleanly, writes artifacts, and produces no evidence of validity.
- **`+use_encoder=g1` must be pinned.** The released checkpoint ships three encoders (g1, teleop,
  smpl) and samples between them. A Kimodo G1-qpos reference has to be tracked by the g1 encoder;
  left unpinned, encoder choice is left to sampling. The export binding gate catches this, but only
  after the rollout has already been spent.
- **The exporter validates the complete capture before slicing**, so a run that continues past the
  end of its reference motion and loops is rejected wholesale for a non-monotonic motion clock.
  Capture exactly one pass instead: recorded frames are `max_render_steps - 1`.
- **Do not edit the runner while it is executing.** Bash reads a script incrementally, so an in-place
  edit shifts byte offsets under the running interpreter and produces a syntax error partway
  through. One rollout was lost this way.

### 18.6 First fully-bound synthetic_g1 dataset export

Both accepted episodes were exported end to end. Each carries **248 training rows** (249 captured
frames minus the one causal boundary frame), a 248-frame 640x480 50 Hz H.264 ego video, and a
complete `EpisodeRequest -> GenerationResult -> ConversionResult` bundle cross-bound to the runtime
capture. Both pass `check_sonic_vla_dataset.py --profile synthetic_g1 --groot-loader-ready` with
**zero errors and zero warnings**, and both pass `check_latent_parity.py --require-encoder-equivalent`.

For scale: the 2026-08-14 export produced 65 rows from the same source motion. Measuring contact
correctly rather than relaxing a threshold raised that to 248, a 3.8x increase per episode.

The embedded `provenance.latent_representation` block records the token contract and the measured
`deployment_encoder_equivalent` verdict at residual RMS 0, and `locomotion_acceptance.diagnostics`
records the contact decomposition (factory: 33.8 N self, 0.0 N external).

Getting there required binding several things at capture time rather than export time, because the
export gates cross-check them against the runtime manifest: the natural-language task string, the
scene id and hash, the pinned g1 encoder, and the byte-identical motion input. The conversion is
deterministic -- re-running it produced byte-identical motion PKLs -- which is what lets a
post-hoc typed bundle bind to an already-completed rollout at all.

**Not yet done:** the official Isaac-GR00T loader smoke. The pinned revision is not installed on
this machine, so the strongest available check is the offline `--groot-loader-ready` file gate,
which validates schema, dtypes, video decodability, frame/fps agreement and the registered 40-step
action horizon, but does not instantiate `ShardedSingleStepDataset` itself.

## 19. Scene-aware placement (2026-08-15)

Section 18.4 concluded that acceptance was placement-limited, not controller-limited: every
reference was dropped at a scene's route start with a fixed yaw, so a motion whose root path curves
2.5 m laterally walked into a rack whatever the controller did. This section closes that.

### 19.1 Method

[`route_placement.py`](../gear_sonic/dataset_generation/route_placement.py) searches yaw and
translation for placements whose whole reference path keeps a required clearance from every obstacle
footprint and stays inside the walkable rectangle.

Two properties make it trustworthy rather than merely convenient:

- **The searched transform is the adapter's transform.** Canonicalise the path so its first XY is at
  the origin, rotate about +Z by `scene_yaw`, add `scene_start_xyz` -- exactly what
  `kimodo_motion_adapter.transform_qpos_to_scene` does. A test asserts the two agree to 1e-9, so a
  planned placement *is* the placement the converter produces.
- **The obstacle model is imported from the gate, not reimplemented.**
  `scene_asset_preflight.load_scene_obstacle_map` exposes the same footprints the route-clearance
  gate uses: collision-enabled, non-floor solids whose vertical extent overlaps
  `[support_z, support_z + robot_height)`. Loading the checked-in scenes reproduces the documented
  0.70 m (factory) and 0.80 m (household) canonical-route clearances exactly.

**The tracking margin is measured, not guessed.** Planning against the *reference* path
underestimates risk because the executed root deviates from it. On accepted M0 episodes p95 path
error was 0.22-0.23 m and endpoint error 0.20-0.22 m, so the default margin is 0.30 m on top of the
scene's declared 0.45 m body radius, giving a 0.75 m requirement.

### 19.2 Planning result

For the six locomotion demos across both scenes, 34 configs were planned, every one at
clearance >= 0.75 m:

| motion | factory_aisle | household_room |
|---|---|---|
| `01_single_text_prompt` | 3 (0.90-1.61 m) | 3 (0.77-1.40 m) |
| `02_multi_text_ee_constraint` | 3 (0.77-1.65 m) | 3 (0.80-1.50 m) |
| `03_full_body_keyframes` | 3 (0.82-1.61 m) | 3 (0.75-1.46 m) |
| `04_ee_constraint` | 3 (0.78-1.56 m) | 3 (0.88-1.41 m) |
| `05_root_path` | 3 (0.75-0.82 m) | **1** (0.85 m) |
| `06_root_waypoints` | 3 (0.85-1.13 m) | 3 (0.78-0.79 m) |

`05_root_path` and `06_root_waypoints` are the two that previously drove the robot into racks at
1576-1850 N of scene contact; both now have placements the scenes can contain. `05_root_path` finds
only one placement in the household room, which is the honest answer for a 6.93 m path with 2.48 m
of lateral excursion inside a 9.4 x 7.4 m room.

`07_text_terrain` (a jump, peak z = 1.96 m) and `08_text_object` (0.02 m net translation) are
excluded from planning: neither is locomotion on flat ground, and the gates already reject them for
the right reasons. Their rejections stay in the quarantine record rather than being re-run.

### 19.3 Reproducibility

Three tools are now checked in, so none of this depends on ad-hoc commands:

- [`plan_kimodo_placements.py`](../scripts/research/plan_kimodo_placements.py) writes a versioned
  `placements.json` recording the margin, search resolution, per-scene clearance requirement, and
  every accepted placement.
- [`run_kimodo_placement_batch.sh`](../scripts/research/run_kimodo_placement_batch.sh) converts and
  rolls out each config, resuming rather than repeating completed work.
- [`build_kimodo_provenance_bundle.py`](../scripts/research/build_kimodo_provenance_bundle.py)
  builds the typed `EpisodeRequest -> GenerationResult` pair the exporter demands, binding
  `task_prompt`, `scene_hash` and `physics_hash` to the runtime manifest.

The planner emits `max_render_steps = resampled_frames + 1`, because `eval_agent_trl` records
`max_render_steps - 1` frames and the exporter validates the whole capture before slicing; this
captures exactly one motion pass with no clock reset.

## 20. Acceptance gates must compare against what was commanded

Three separate false rejections this session turned out to be the same design error, and the
correction is worth stating as a principle rather than three patches.

**The pattern.** A gate applied an *absolute* threshold to a quantity the reference motion
deliberately drives, so tracking the reference well was scored as failing.

| Gate | Rejected | Reference commanded | Executed vs reference | Verdict |
|---|---|---|---|---|
| `fall_root_height` @ 0.50 m | `02_multi_text_ee_constraint` at 0.408 m | pelvis to **0.323 m** (a crouch) | within **0.039 m** | tracked crouch, not a fall |
| `disallowed_robot_contact` @ 1.0 N | knee at 105-549 N | crouch puts the knee on the floor | force purely **+Z, vertical fraction 1.000** on all 56/58 frames | kneeling, not a rack strike |
| `fall_root_tilt` @ 0.60 rad | `03_full_body_keyframes` at 0.708 rad | **0.732 rad** bow (41.9 deg) | within **0.124 rad** | tracked bow, not a fall |

In every case the robot was doing exactly what it was told, and the gate could not tell the
difference between "the robot collapsed" and "the reference asked for a low, tilted, or
ground-supported pose".

**The principle.** *Gate on deviation from the command; reserve absolute limits for states no
command can legitimately request.* Each gate is now a pair:

| Quantity | Reference-relative gate | Absolute gate |
|---|---|---|
| root height | `root_height_tracking`: sank <= 0.15 m below command | `root_height`: >= 0.25 m (pelvis at knee height) |
| root tilt | `root_tilt_tracking`: <= 0.35 rad beyond command | `root_tilt`: <= 1.40 rad (torso near-horizontal) |
| non-foot contact | `nonfoot_scene_contact`: lateral push <= 1.0 N | `nonfoot_support_contact`: upward load <= 686 N (2x body weight) |

Every threshold is marked PROVISIONAL in the code with the measurement that motivated it. They are
physical anchors, not calibrations, and section 9.4's reviewed pilot still owns the final values.

**Effect on yield.** Of the completed placement-batch rollouts, acceptance went from 4/7 to **5/7**
on the tilt fix alone, and `02`'s rejection narrowed from four reason codes to one. Combined with
the earlier contact decomposition, a single rollout now yields a 248-row episode where the
2026-08-14 pipeline yielded 65 rows.

**What was deliberately not relaxed.** `02_multi_text_ee_constraint` still fails
`excessive_self_contact` at 712 N and 659 N, well past the 343 N body-weight anchor. That is a real
calibration question for the pilot, not a false reject to engineer away, and it stays rejected.

### 20.1 First multi-motion dataset

`dataset_v1` contains **five** exported episodes (248 rows each, 1240 training rows total) spanning
two Kimodo motions, two scenes, and three distinct placements of one motion. All five pass
`check_sonic_vla_dataset.py --profile synthetic_g1 --groot-loader-ready` with zero errors and zero
warnings, and all five pass `check_latent_parity.py --require-encoder-equivalent`. Rejected rollouts
are recorded with reason codes in `quarantine.txt` rather than discarded.

The export batch learned the same lesson the rollout runner did: the exporter writes a complete,
valid dataset and *then* aborts during interpreter teardown, so success is the explicit
`SONIC_LEROBOT_EXPORT_SUCCESS` marker plus the dataset on disk, never the process exit status.

## 21. Visual inspection found what no numeric gate could

Every schema and physics gate passed on twelve exported episodes. Rendering one episode and
*looking* at it found a defect that invalidates the visual modality outright.

### 21.1 Debug markers were rendered into the observation

The ego frames contained **large bright-yellow spheres**. They are Isaac Lab
`VisualizationMarkers` created by the motion command's `_set_debug_vis_impl` --
`goal_pos_visualizer`, `diffuse_color=(1.0, 1.0, 0.0)`, i.e. **the reference body positions the
policy is tracking**. `manager_env/commands/terms/motion.yaml` sets `debug_vis: true` by default,
the markers are spawned as real prims under `/Visuals`, and the ego `TiledCamera` renders them.

Measured on `01_single_text_prompt__factory_aisle__p0`: **243 of 249 frames contained marker
pixels, 37 frames exceeded 1% of the image, peaking at 15.1%.**

This is worse than a cosmetic artifact. It is **goal leakage**: the observation contains the
tracking target itself. A VLA trained on this can learn "follow the yellow blobs" instead of
grounding the language instruction and the scene, and then fail completely at deployment, where no
such markers exist. Every numeric check -- shapes, dtypes, fps, frame counts, decodability,
acceptance gates, latent parity -- passed on this data.

**Contamination is pose-dependent, which makes it worse.** The markers sit at reference body
positions, so whether they enter the ego frustum depends on the motion. `03_full_body_keyframes`
measured 0.01% peak yellow -- genuinely clean -- while `01_single_text_prompt` measured 15.1%. A
partially contaminated dataset is harder to notice than a uniformly broken one.

**Fix:** the runner now passes `++manager_env.commands.motion.debug_vis=False`. Re-measured on the
same configuration: peak yellow **15.11% -> 0.02%**, frames over 1%: **37 -> 0**. The residual is
the scene's own amber rack trim (`displayColor = (0.92, 0.65, 0.08)`), which is legitimate geometry.

### 21.2 A semantic evaluation script, and what it cannot do

[`evaluate_synthetic_g1_dataset.py`](../scripts/research/evaluate_synthetic_g1_dataset.py) asks what
schema validation cannot: is this data *worth training on*?

- **Overlay contamination** -- saturated marker-colour blobs, the defect above.
- **Camera attachment** -- ego image change must correlate with body motion, or the visual modality
  is decoration.
- **Action informativeness** -- distinct values, dead dimensions, and participation ratio of the
  64-dim latent. Measured on a real episode: 24 distinct values, 0 dead dims, effective rank 3.28.
- **Action-dynamics coupling** -- ridge regression from action to next-state change against a
  shuffled-action control. Measured: **R^2 = 0.717 vs shuffled -0.037**. The action genuinely
  explains the motion; behaviour cloning has something to learn.
- **Temporal continuity, episode duplication, language coverage.**

**The validator was itself validated by negative controls**, which is how two of its own defects
were found:

1. A first-half/second-half regression split reported R^2 = -1.11 on good data, because a walking
   trajectory is non-stationary and a linear map fitted on the first half extrapolates badly. Fixed
   by interleaving the folds.
2. **The causal-direction check does not work, and now says so.** Reversing the one-frame shift
   moved R^2 by 0.015 -- in the *wrong* direction. Even a deliberate 25-frame (0.5 s) offset only
   moved it 0.717 -> 0.556, still far above the shuffled baseline. The cause is measured: the latent
   has autocorrelation **r = 0.981 at lag 1** and 0.874 at lag 5, so neighbouring actions are
   interchangeable to any regressor. The one-frame causal contract is guaranteed structurally by the
   exporter's shift and its sentinel tests, **not** by this script. The metrics are still reported
   as diagnostics, but they raise no error, because a check that cannot fail on broken data would
   only manufacture false confidence.

Controls that do work: shuffled actions and constant actions are both correctly rejected.

### 21.3 `dataset_v2`

Twelve episodes, 248 rows each (2976 training rows), spanning two Kimodo motions, both scenes, and
three planned placements per motion-scene pair. Every episode passes all three gates:
`check_sonic_vla_dataset.py --groot-loader-ready` (0 errors, 0 warnings),
`check_latent_parity.py --require-encoder-equivalent`, and
`evaluate_synthetic_g1_dataset.py` (0 errors, 0 warnings).

`dataset_v1` is superseded and should not be trained on: six of its twelve episodes carry marker
contamination.

Per-episode summary figures are produced by
[`visualize_synthetic_g1_episode.py`](../scripts/research/visualize_synthetic_g1_episode.py):
ego contact sheet, top-down executed-vs-reference path over real obstacle footprints, height and
tilt against command, action heatmap, and the contact decomposition. Reviewing them confirmed the
behaviours the gates only described numerically -- `03_full_body_keyframes` is visibly a bow, with
the ego view pitching down to the floor and the robot's own hands entering frame.

## 22. Clutter generated around the motion, not the other way round (2026-08-16)

### 22.1 The inversion

Section 19 fitted motions into fixed scenes. That is placement-limited, and the episodes that survive
walk mostly empty floor -- not the cluttered-navigation data this dataset is for.

The motion is known first, so its swept corridor is known first.
[`clutter_scene_builder.py`](../gear_sonic/dataset_generation/clutter_scene_builder.py) generates the
room and its furniture *around* that corridor: every solid clears the path by the body radius plus
the tracking margin, and pieces are sampled just outside the corridor rather than uniformly, so the
robot threads between obstacles instead of crossing a sparsely furnished hall. Output is the same
`Plane` + axis-aligned `Cube` subset the existing preflight validates, so generated packages pass the
identical gate as hand-authored ones.

Generated: 6 scenes for the two straight/bow motions and 4 for the curved ones, 19-25 pieces each,
**21-30% floor occupancy**, nearest obstacle 0.75-0.80 m from the path. All 10 pass
`check_g1_dataset_scenes.py`.

### 22.2 The motions that used to fail are now the best ones

`05_root_path` and `06_root_waypoints` were the two worst cases in section 18.4: dropped into fixed
scenes they walked into racks at **1576-1850 N** of scene contact. With the scene generated around
them, their corridors become a horseshoe and an S-curve winding through furniture -- far better
navigation data than a straight walk.

**Measured lateral (scene) contact across all seven clutter rollouts: 0.00 N.** The generator's
guarantee holds in physics, not just in the planner.

Their remaining failure is now a clean, different thing: `reference_path_tracking_error` at p95
0.271-0.272 m against a provisional 0.25 m threshold -- **8% over, with zero collisions**, and
`05_root_path` lands its endpoint to 0.017 m. That is a controller-tracking limit and a threshold
calibration question, no longer a scene-collision problem. Separating those two was the point.

### 22.3 Third-person video

The dataset's own camera is ego-view, which is right for training and poor for review. Rendering the
same rollout with `+manager_env/recorders=render` (a chase camera at `eval_camera_offset`, 1280x720)
produces the view a human needs to judge whether the walk looks right. One trap: `render.yaml` does
not set `dataset_export_mode: 0`, so Isaac Lab's HDF5 exporter tries to write
`/tmp/isaaclab/logs/dataset.hdf5`, which on a shared machine belongs to another user and fails with
a permission error. Pass `++manager_env.recorders.dataset_export_mode=0`.

### 22.4 Duplicate behaviour vs visual variant

Seeding several clutter layouts for one motion produces **bit-identical state trajectories** -- the
furniture never touches the robot, so the physics is unchanged -- while the imagery differs (mean
per-pixel difference 21.2/255). That is not waste: it is the same behaviour in a different visual
context, which is exactly visual-robustness augmentation. It is also not behavioural diversity.
`evaluate_synthetic_g1_dataset.py` now separates the two, warns rather than errors, and reports the
distinct-behaviour count: **15 episodes cover 14 distinct behaviours.**

### 22.5 Photorealism: not done, and what it needs

The NVIDIA Omniverse asset server is reachable from this machine (verified HTTP 200 on
`Isaac/Environments/...`), so photorealistic residential assets are available. They are **not** used
yet, and the current scenes are untextured primitive proxies with plausible furniture dimensions.

Adopting real assets is not a drop-in change: the scene contract and its dependency-free preflight
deliberately validate only the `Plane` + axis-aligned `Cube` subset, and arbitrary meshes need
collision approximation, a licence/redistribution record per asset, and a USD-native geometry gate to
replace the static parser. The geometry and layout are the part that determines whether the
navigation data is any good; textures change what the ego camera sees, which matters for sim-to-real
but not for whether the corridor is traversable. Sequencing it after the layout work is deliberate.

Operational findings:

- Isaac/Hydra can print a fatal traceback yet return process status zero because the evaluation
  script exits through `os._exit(0)`. Automation must require the explicit rollout-success marker
  and expected artifacts; process status alone is insufficient.
- The working local runtime currently combines the Isaac Lab 2.3.2 / Isaac Sim 5.1 environment
  with SONIC dependencies from the older provisioned environment. A durable installation should
  install the pinned dependencies into one environment after disk space is recovered.

Still open before the M0 gate:

- Turn the real Kimodo NPZ/qpos/FK comparison into a redistributable checked-in fixture or a
  documented optional integration test; the current demo resides in the separate Kimodo checkout.
- Implement geometry-derived route sampling (section 6.3). This is now the single highest-value
  blocker for episode yield: acceptance is 2/16 mainly because every reference is dropped on a fixed
  route start with a fixed yaw, so laterally-curving motions collide regardless of controller
  quality (section 18.4).
- Calibrate the `self_contact` threshold. The 343 N default is a body-weight anchor, not a
  calibration; measured peaks span 8-911 N, and it is now the *only* thing rejecting
  `02_multi_text_ee_constraint` (712 N / 659 N). Decide whether repeated self-impact should
  downgrade an episode rather than reject it.
- Calibrate the other provisional pairs introduced in section 20: the 0.15 m / 0.35 rad
  command-deviation limits and the 0.25 m / 1.40 rad absolute limits.
- Finish the 34-config placement batch and export. It resumes without repeating work:
  `scripts/research/run_kimodo_placement_batch.sh --placements <W>/placements_ordered.json
  --work-dir <W>/placed`, then `scripts/research/export_kimodo_dataset_batch.sh`.
- Produce and visually review at least 20 accepted episodes. Twenty-one rollouts so far yield two
  fully-exported episodes; the videos have not been visually reviewed by a human.
- Run the official Isaac-GR00T loader smoke at the pinned revision; the package is not currently
  installed on this machine (section 18.6).
- Restore Kimodo *generation*. The 2026-08-15 rollouts still consume the official bundled demo
  motion re-exported through Kimodo's own converter; no newly sampled seed has been generated,
  because the local Kimodo environment still lacks a working `sm_120` PyTorch build.
- Restore a Kimodo generation environment with model weights on the RTX 5090. The local checkout's
  old CUDA/PyTorch environment lacks NumPy and does not support compute capability `sm_120`, while
  low free disk makes an unplanned reinstall unsafe. Current Kimodo proofs therefore use official
  bundled generated motions, not a newly sampled seed.
- Promote ephemeral smoke artifacts into the canonical raw episode store with immutable Kimodo,
  adapter, scene, controller-checkpoint, Hydra/physics, trajectory, video, and acceptance hashes.
- Implement the asynchronous per-environment recorder and accepted-trajectory render replay before
  scaling beyond one environment; the M0 camera proofs intentionally execute and render together.
