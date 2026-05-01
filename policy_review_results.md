# Policy Review Results

Date: 2026-05-01

## Prepared Assets

Available deployment policies:

- `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx`
- `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx`
- `gear_sonic_deploy/policy/release/model_encoder.onnx`
- `gear_sonic_deploy/policy/release/model_decoder.onnx`
- `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`

Downloaded training checkpoint:

- `sonic_release/last.pt` - 448 MB
- `sonic_release/config.yaml` - 28 KB

Prepared sample data:

- `sample_data/robot_filtered/210531/`
- `sample_data/smpl_filtered/`
- `sample_data/soma_filtered/210531/`

The full 30 GB SMPL archive was not downloaded. The checkpoint download used
`python download_from_hf.py --training --no-smpl`.

## Generated Review Videos

Closed-loop Decoupled WBC MuJoCo rollouts:

| Video | Result |
|---|---|
| `rollouts/review_wbc_balance.mp4` | 8.0 s, 400 frames, 640x480. Final base z = 0.747 m, root xy = (-0.066, -0.002). |
| `rollouts/review_wbc_walk.mp4` | 8.0 s, 400 frames, 640x480. Final base z = 0.749 m, root xy = (3.460, -0.471). |

SONIC target/reference review videos:

| Video | Meaning |
|---|---|
| `rollouts/review_ref_walking_quip_360_R_002__A428.mp4` | Kinematic replay of a shipped SONIC reference motion target. 9.1 s, 455 frames. |
| `rollouts/review_planner/planner_mode0_target_goal3m.mp4` | Short planner waypoint preview. 64 frames, root displacement about 1.47 m. |
| `rollouts/review_planner/planner_mode1_teleop_fwd_v10.mp4` | Short planner mode1 forward preview. 60 frames, root displacement about 1.06 m. |
| `rollouts/review_planner/planner_mode2_smpl_fwd_v10.mp4` | Short planner mode2 forward preview. 60 frames, root displacement about 1.06 m. |

Additional short planner videos were generated in `rollouts/review_planner/`.

## Model Signature Checks

ONNX Runtime loaded all key models successfully:

| Model | Inputs | Outputs |
|---|---|---|
| Decoupled Balance | `input [batch, 516]` | `output [batch, 15]` |
| Decoupled Walk | `input [batch, 516]` | `output [batch, 15]` |
| SONIC encoder | `obs_dict [1, 1762]` | `encoded_tokens [1, 64]` |
| SONIC decoder | `obs_dict [1, 994]` | `action [1, 29]` |
| SONIC planner | 11 planner tensors | `mujoco_qpos [1, 64, 36]`, `num_pred_frames [1]` |

Waypoint streamer dry run:

```text
python scripts_local/stream_planner_waypoint_zmq.py \
    --waypoint 3.0 0.0 0.0 --target-vel 1.0 --mode 0 --chunks 1 --dry-run
```

Result:

```text
64 frames, d_root=(1.47, -0.00) m
```

## Script Checks

Passed:

- `bash -n gear_sonic_deploy/deploy.sh`
- `bash gear_sonic_deploy/deploy.sh --help`
- Python compile for:
  - `decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py`
  - `scripts_local/replay_planner.py`
  - `scripts_local/replay_motion_clip.py`
  - `scripts_local/stream_planner_waypoint_zmq.py`

Blocked:

- Focused `decoupled_wbc` unit tests were not run to completion in `.venv_sim`.
  That environment lacks `gymnasium`, which belongs to the heavier
  `decoupled_wbc[full]` dependency set.

## Real-Robot Readiness

Current machine is not ready for SONIC real-robot deployment yet.

`python3 check_environment.py` reports:

- Python is 3.10.12; Isaac Lab training expects Python 3.11.
- `torch.cuda.is_available()` is false.
- Isaac Lab is not installed.
- Training packages such as Hydra, TRL, Transformers, Accelerate, and W&B are missing.
- `TensorRT_ROOT` is not set.

Separate shell checks also found:

- `just` is not on PATH.
- `ninja` is not on PATH.
- `clang` is not on PATH.
- CUDA toolkit `nvcc` is available at `/usr/local/cuda/bin/nvcc`.
- The ONNX Runtime Python package is installed in `.venv_sim`, but the ONNX
  Runtime C/C++ package is not visible to CMake (`onnxruntime_ROOT` unset).

## Recommendation

For review now, inspect:

1. `rollouts/review_wbc_balance.mp4`
2. `rollouts/review_wbc_walk.mp4`
3. `rollouts/review_ref_walking_quip_360_R_002__A428.mp4`
4. `rollouts/review_planner/planner_mode0_target_goal3m.mp4`
5. `rollouts/review_planner/planner_mode1_teleop_fwd_v10.mp4`
6. `rollouts/review_planner/planner_mode2_smpl_fwd_v10.mp4`

Before real hardware, install the C++ deploy toolchain and TensorRT, then run
SONIC sim2sim:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type keyboard sim
```

Only move to:

```bash
bash deploy.sh --input-type keyboard real
```

after sim2sim is stable and the G1 Ethernet/DDS network is confirmed.
