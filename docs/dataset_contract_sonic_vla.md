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

## Required Layout

```text
<dataset>/
  data/
    train-00000.parquet
  videos/
    observation.images.ego_view/
      episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

Wrist camera videos are optional for the MVP. If enabled, both `meta/info.json`
and `meta/modality.json` must include the corresponding `left_wrist` and
`right_wrist` video keys.

## Required Features

The validator checks these columns when parquet inspection dependencies are
available:

| Feature | Shape | Notes |
|---|---:|---|
| `observation.state` | robot joints | Whole-body G1 state from `RobotModel`. |
| `observation.eef_state` | 14 | Left/right wrist position and quaternion. |
| `observation.root_orientation` | 4 | Base quaternion. |
| `observation.projected_gravity` | 3 | Gravity vector in body frame. |
| `action.motion_token` | 64 | SONIC latent token. |
| `teleop.left_hand_joints` | 7 | Left hand command. |
| `teleop.right_hand_joints` | 7 | Right hand command. |
| `teleop.smpl_pose` | 63 | Used to detect stale teleop frames. |
| `task_index` | 1 | LeRobot task index; mapped to `annotation.human.task_description` in modality config. |

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
- `action.motion_token` is not length 64.
- Either hand action is not length 7.
- Any required numeric feature contains NaN or infinity.
- SMPL pose is all zeros for recorded frames.
- Consecutive timestamps drift beyond the configured tolerance.
- Episode is marked discarded in `meta/info.json`.
- The trial failed the task but is not explicitly labeled as a failure set.
