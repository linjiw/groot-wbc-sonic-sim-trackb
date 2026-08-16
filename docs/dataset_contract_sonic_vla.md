# SONIC VLA Dataset Contract

## Purpose

Datasets collected with `gear_sonic/scripts/launch_data_collection.py` must be
valid GR00T-flavored LeRobot v2.1 datasets for the `UNITREE_G1_SONIC`
embodiment. The policy action is the SONIC latent-control action:

```text
action.motion_token: 64
teleop.left_hand_joints: 7
teleop.right_hand_joints: 7
```

The combined GR00T action is therefore 78 dimensions per frame.

The `synthetic_g1` profile uses the same `UNITREE_G1_SONIC` training modalities but omits
human-only SMPL, VR, and planner telemetry. It adds `reference.g1_qpos` (36 values in Kimodo's
MuJoCo order) for provenance and validation; that reference is not part of the GR00T action.

## Synthetic State/Action Semantics

Isaac Lab articulation arrays are recorded in Isaac Lab joint order. Before export, all 29D body
state and reference-joint arrays are permuted to the registered MuJoCo/G1 order. The source order,
joint names, permutation, and output order are embedded in episode provenance; relabeling an
unpermuted array is forbidden.

The Isaac trajectory recorder runs after physics. A raw row therefore contains the image/state
resulting from the action selected immediately before that step. For behavioral cloning, an
accepted raw interval of `N` frames exports `N-1` rows: observations `0..N-2` pair with policy
actions `1..N-1`. Export provenance records the absolute source ranges and the dropped boundary.

Production synthetic export also requires:

- a schema-v2 runtime success manifest that hash-binds the checkpoint, scene, task, input motion,
  calibrated link-mounted `ego_camera`, trajectory, and video;
- a verified sibling bundle containing one `EpisodeRequest`, its `GenerationResult`, and the
  derived `ConversionResult`; and
- exact cross-checks between those typed identities and the runtime scene, task, controller,
  physics configuration, and SONIC motion hash.

## Required Layout

```text
<dataset>/
  data/chunk-000/
    episode_000000.parquet
  videos/chunk-000/
    observation.images.ego_view/
      episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    episodes_stats.jsonl
    stats.json
    tasks.jsonl
```

The validator also accepts the earlier flat `data/*.parquet` and
`videos/observation.images.ego_view/*.mp4` layout used by small research fixtures.

`meta/stats.json` is required by the official Isaac-GR00T episode loader. Use
`scripts/research/check_sonic_vla_dataset.py --profile synthetic_g1
--groot-loader-ready` for the strict file gate; the ordinary validator mode is
only a schema/plumbing check.

Wrist camera videos are optional for the MVP. If enabled, both `meta/info.json`
and `meta/modality.json` must include the corresponding `left_wrist` and
`right_wrist` video keys.

## Required Features

The validator checks the common columns below when parquet inspection dependencies are
available:

| Feature | Shape | Notes |
|---|---:|---|
| `observation.state` | robot joints | Whole-body G1 state from `RobotModel`. |
| `observation.projected_gravity` | 3 | Gravity vector in body frame. |
| `action.motion_token` | 64 | SONIC latent token. |
| `teleop.left_hand_joints` | 7 | Left hand command. |
| `teleop.right_hand_joints` | 7 | Right hand command. |
| `task_index` | 1 | LeRobot task index; mapped to `annotation.human.task_description` in modality config. |

The `live_vr` profile additionally requires `observation.eef_state` (14),
`observation.root_orientation` (4), and `teleop.smpl_pose` (63). The `synthetic_g1`
profile omits all three rather than fabricating unavailable human/VR signals, and instead requires
the diagnostic `reference.g1_qpos` (36). All numeric state/action/reference vectors in the
synthetic profile are stored as `float32`.

The registered `UNITREE_G1_SONIC` action horizon is 40 frames. Every accepted
episode must therefore contain at least 40 frames or the official single-step
dataset has no usable training sample.

## Required Modality Entries

`meta/modality.json` must include top-level keys:

```text
state
action
video
annotation
```

Important action mappings:

```json
{
  "action": {
    "motion_token": {
      "start": 0,
      "end": 64,
      "original_key": "action.motion_token"
    },
    "left_hand_joints": {
      "start": 0,
      "end": 7,
      "original_key": "teleop.left_hand_joints"
    },
    "right_hand_joints": {
      "start": 0,
      "end": 7,
      "original_key": "teleop.right_hand_joints"
    }
  }
}
```

## Split Policy

- `train`: successful demonstrations used for fine-tuning.
- `val`: 10-20% successful held-out demonstrations from seen objects/layouts.
- `test_seen`: same object and layout distribution as training.
- `test_unseen_objects`: new object instance/color/shape.
- `test_unseen_layouts`: new table/tray positions and distractors.

Do not mix discarded or failed episodes into `train` for the direct baseline.
Failure/recovery data may be added later only when it is intentionally labeled
and evaluated as a separate curriculum condition.

## Rejection Criteria

Reject or quarantine an episode if any condition is true:

- Missing `meta/modality.json`, `meta/info.json`, parquet data, or required
  ego-view video.
- Missing aggregate `meta/stats.json`, empty/undecodable ego video, or fewer
  than 40 accepted frames when applying the strict GR00T loader-ready gate.
- `action.motion_token` is not length 64.
- Either hand action is not length 7.
- Any required numeric feature contains NaN or infinity.
- SMPL pose is all zeros for recorded live-VR frames. Synthetic G1 episodes omit the field.
- Consecutive timestamps drift beyond the configured tolerance.
- The trajectory lacks exact 30-body contact-vector coverage or the two filtered
  ankle-to-support-floor force vectors.
- Executed route progress is insufficient, a non-foot contact occurs, a foot contacts something
  other than the registered floor, or floor-support coverage is below threshold.
- Joint-order or causal-alignment provenance is absent or inconsistent.
- Runtime artifact hashes, camera/scene/task claims, or the typed upstream lineage do not match.
- Episode is marked discarded in `meta/info.json`.
- The trial failed the task but is not explicitly labeled as a failure set.
