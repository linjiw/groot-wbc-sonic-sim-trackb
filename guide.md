# Policy Test and Deployment Guide

This guide answers three practical questions for this checkout:

1. What trained or released policies are available here?
2. Which policy should be tested first, and which one is suitable for real G1 deployment?
3. How do you run the policies in MuJoCo and on the real robot?

The repository contains two controller families. Keep them separate: they use
different observation schemas and different runtimes.

## Current Policy Inventory

### 1. Decoupled WBC release policies

Location:

```text
decoupled_wbc/sim2mujoco/resources/robots/g1/policy/
```

Files present in this checkout:

| File | Runtime | Input/output | Purpose |
|---|---|---|---|
| `GR00T-WholeBodyControl-Balance.onnx` | Python, ONNX Runtime | obs `[1,516]` -> action `[1,15]` | Standing and balancing when locomotion command is near zero |
| `GR00T-WholeBodyControl-Walk.onnx` | Python, ONNX Runtime | obs `[1,516]` -> action `[1,15]` | Lower-body walking command tracking |

These are the easiest policies to sanity-check because they only need MuJoCo
and ONNX Runtime. They are good for verifying that the local Python simulation
environment and G1 MJCF assets are working.

Important limitation: these are 15-action lower-body/waist policies used by the
decoupled WBC stack. They are not interchangeable with the SONIC C++ deployment
policy.

### 2. GEAR-SONIC release deployment policy

Location:

```text
gear_sonic_deploy/policy/release/
gear_sonic_deploy/planner/target_vel/V2/
gear_sonic_deploy/reference/example/
```

Files present in this checkout:

| File | Runtime | Input/output | Purpose |
|---|---|---|---|
| `policy/release/model_encoder.onnx` | C++ deploy stack | obs `[1,1762]` -> token `[1,64]` | Encodes motion/history into policy tokens |
| `policy/release/model_decoder.onnx` | C++ deploy stack | obs `[1,994]` -> action `[1,29]` | Produces 29-DoF G1 joint targets |
| `policy/release/observation_config.yaml` | C++ deploy stack | config | Defines observation layout for the policy |
| `planner/target_vel/V2/planner_sonic.onnx` | C++ deploy stack | 11 planner inputs -> `mujoco_qpos [1,64,36]` | Kinematic planner for locomotion modes |
| `reference/example/*` | C++ deploy stack | CSV motion clips | Reference motions such as walking, macarena, squat, kick, lunge |

This is the main real-robot deployment path. It uses `gear_sonic_deploy`, the
Unitree SDK2 DDS interface, TensorRT, and the compiled
`g1_deploy_onnx_ref` binary.

### 3. SONIC training checkpoint

The training code supports the `sonic_release` checkpoint and finetuning, but
the PyTorch checkpoint is not necessarily present in every checkout. Download it
when you need evaluation, finetuning, or ONNX export:

```bash
pip install huggingface_hub
python download_from_hf.py --training --no-smpl
```

That creates:

```text
sonic_release/last.pt
sonic_release/config.yaml
```

For full training or finetuning you also need motion data. The sample data is
already present in this checkout under `sample_data/`, and the full Bones-SEED
and SMPL datasets can be downloaded separately.

## Which Policy Should You Try First?

Use this order.

| Stage | Policy | Why |
|---|---|---|
| 1 | Decoupled WBC Balance/Walk in MuJoCo | Fastest pure-Python smoke test; no TensorRT or C++ build required |
| 2 | SONIC in MuJoCo sim2sim | Tests the actual deployment stack before touching hardware |
| 3 | SONIC release policy on real G1 | Recommended real-robot path |
| 4 | Your own trained SONIC checkpoint | Only after evaluation, ONNX export, and sim2sim validation |

Do not deploy a freshly trained checkpoint directly to hardware. First evaluate
it in Isaac Lab, export ONNX, run MuJoCo sim2sim, and only then run guarded
real-robot tests.

## One-Time Setup

From the repository root:

```bash
git lfs pull
python check_environment.py
```

Install the MuJoCo simulator environment:

```bash
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
```

Install or fetch the SONIC deployment assets if they are missing:

```bash
pip install huggingface_hub
python download_from_hf.py
```

For the C++ deployment stack, TensorRT must be installed. The docs require:

| Platform | TensorRT |
|---|---|
| x86_64 desktop | TensorRT 10.13 |
| Jetson / onboard Orin | TensorRT 10.7 with JetPack 6 |

Typical deploy setup:

```bash
export TensorRT_ROOT=$HOME/TensorRT
export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH
cd gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
```

## Test Decoupled WBC in MuJoCo

Use this to quickly confirm that the released Balance and Walk ONNX files run.

### Headless video test

From the repo root:

```bash
source .venv_sim/bin/activate

python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd balance \
    --out rollouts/wbc_balance.mp4

python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd walk \
    --out rollouts/wbc_walk.mp4
```

Expected result:

- `balance`: robot holds a stable standing pose.
- `walk`: robot walks forward from a 0.5 m/s command.

### Interactive Decoupled WBC MuJoCo viewer

The interactive script is:

```bash
source .venv_sim/bin/activate
cd decoupled_wbc/sim2mujoco
python scripts/run_mujoco_gear_wbc.py
```

Check the config before using it:

```text
decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml
```

In this checkout, the released policies are named:

```text
policy/GR00T-WholeBodyControl-Balance.onnx
policy/GR00T-WholeBodyControl-Walk.onnx
```

If `g1_gear_wbc.yaml` points to old names such as `policy/ft92.onnx` or
`policy/ft109.onnx`, update the YAML before running the interactive viewer.

Interactive keys:

| Key | Action |
|---|---|
| `w` / `s` | Forward / backward command |
| `a` / `d` | Left / right lateral command |
| `q` / `e` | Yaw command |
| `z` | Reset command, height, and rpy |
| `1` / `2` | Raise / lower target height |
| `3` / `4` | Roll command up/down |
| `5` / `6` | Pitch command up/down |
| `7` / `8` | Yaw pose command up/down |
| `m` / `n` | Increase / decrease gait frequency |

## Test SONIC in MuJoCo Sim2Sim

This is the real deployment stack running against a MuJoCo robot instead of
physical motors. Run it before every real-robot deploy.

### Terminal 1: MuJoCo sim bridge

From the repo root:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

This starts the MuJoCo G1 simulation and publishes/subscribes to Unitree DDS on
loopback.

### Terminal 2: C++ deploy policy

From another terminal:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type keyboard sim
```

The wrapper builds if needed and then runs:

```text
g1_deploy_onnx_ref <loopback_iface>
    policy/release/model_decoder.onnx
    reference/example/
    --obs-config policy/release/observation_config.yaml
    --encoder-file policy/release/model_encoder.onnx
    --planner-file planner/target_vel/V2/planner_sonic.onnx
    --input-type keyboard
    --output-type all
    --disable-crc-check
```

### Starting a SONIC sim session

In Terminal 2:

1. Press `]` to start the control system.
2. Click the MuJoCo viewer and press `9` to drop the simulated robot to the ground.
3. Return to Terminal 2.
4. Press `T` to play the current reference motion.
5. Press `N` or `P` to switch reference clips.
6. Press `R` to restart the current clip at frame 0.
7. Press `O` to emergency-stop and exit.

### Planner mode in MuJoCo

After starting control:

1. Press `ENTER` to enter Planner mode.
2. Press `1` for Slow Walk or `2` for Walk.
3. Press `W` to move forward.
4. Use `A` / `D` to steer, `,` / `.` to strafe, and `S` to move backward.
5. Use `9` / `0` to decrease/increase speed.
6. Press `ENTER` again to return to reference-motion mode.
7. Press `O` to stop.

Useful planner modes:

| Motion set | Keys | Notes |
|---|---|---|
| Locomotion | `1` slow walk, `2` walk, `3` run | Start with slow walk or walk |
| Squat/ground | cycle set with `N`, then `1`-`5` | Height is controlled with `-` / `=` |
| Boxing | cycle set with `N`, then `1`-`7` | Test in sim before hardware |
| Styled walks | cycle set with `N`, then `1`-`7` | Happy, stealth, injured, careful, etc. |

## Play SONIC Reference Motions

Reference clips are under:

```text
gear_sonic_deploy/reference/example/
```

Available clips in this checkout:

```text
dance_in_da_party_001__A464
dance_in_da_party_001__A464_M
forward_lunge_R_001__A359_M
macarena_001__A545
macarena_001__A545_M
neutral_kick_R_001__A543
neutral_kick_R_001__A543_M
squat_001__A359
tired_forward_lunge_R_001__A359_M
tired_one_leg_jumping_R_001__A359
tired_one_leg_jumping_R_001__A359_M
walking_quip_360_R_002__A428
walking_quip_360_R_002__A428_M
```

In keyboard mode:

| Key | Action |
|---|---|
| `T` | Play current clip |
| `R` | Restart current clip at frame 0 |
| `N` | Next clip |
| `P` | Previous clip |
| `Q` / `E` | Heading nudge |
| `I` | Reinitialize base heading |

For first hardware tests, prefer simple standing/walking references before
trying kicks, jumps, lunges, crawling, or dance clips.

## Real Robot Deployment: Recommended Path

Use the SONIC C++ stack for real G1 deployment.

### Real-robot safety checklist

Before running any real-robot command:

- Use a clear safety area.
- Keep a trained operator at the emergency stop.
- Start with the robot in a safe standing or damping posture.
- Use a gantry or spotter for first tests.
- Keep hands, cables, and tools away from the legs.
- Test the exact policy and motion in MuJoCo first.
- Know the emergency-stop key: `O` in the C++ terminal.
- If using gamepad, know the emergency-stop button: `Select`.

### Network setup

The G1 DDS network normally uses `192.168.123.x/24`.

Find your wired interface:

```bash
ip a
```

Assign an address on the same subnet, replacing `eno1` with your interface:

```bash
sudo ip addr add 192.168.123.99/24 dev eno1
sudo ip link set eno1 up
ping -c 2 192.168.123.161
```

If the ping works, `deploy.sh real` should auto-detect the interface.

Do not put Wi-Fi on the same `192.168.123.x` subnet while using DDS.

### Real robot with keyboard

From `gear_sonic_deploy/`:

```bash
source scripts/setup_env.sh
bash deploy.sh --input-type keyboard real
```

Sequence:

1. Confirm the real-robot warning prompt.
2. Press `]` to start control.
3. Press `T` to play a reference motion, or press `ENTER` for Planner mode.
4. In Planner mode, start with `1` Slow Walk or `2` Walk.
5. Use small commands first.
6. Press `O` immediately if motion is wrong.

### Real robot with manager mode

Manager mode is the recommended default once you are comfortable because it can
switch between keyboard, gamepad, and ZMQ interfaces at runtime.

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type manager real
```

Manager hotkeys:

| Hotkey | Interface |
|---|---|
| `Shift+1` (`!`) | Keyboard |
| `Shift+2` (`@`) | Gamepad |
| `Shift+3` (`#`) | ZMQ |
| `O` | Emergency stop from any interface |

### Real robot with gamepad

Use this only with a connected Unitree wireless gamepad.

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type gamepad real
```

Core gamepad controls:

| Button | Action |
|---|---|
| Start | Start control |
| Select | Emergency stop and exit |
| A | Play / resume |
| B | Pause / reset to idle |
| L1 / R1 | Previous / next motion or mode |
| F1 | Toggle Normal / Planner mode |
| Left stick | Movement direction |
| Right stick | Facing direction |
| L2 / R2 | Decrease / increase speed or height |

## ZMQ and VR Teleoperation

Use ZMQ when an external process streams motion targets, for example PICO VR
teleop or a custom motion publisher.

MuJoCo test:

```bash
# Terminal 1
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type zmq sim
```

Real robot:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --input-type zmq real
```

Then run the pose publisher in another terminal. For PICO:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_smpl --vis_vr3pt
```

In the C++ terminal:

1. Press `]` to start control.
2. Press `ENTER` to enable ZMQ streaming.
3. Press `ENTER` again to pause streaming.
4. Press `O` to stop.

Note: the current `deploy.sh` wrapper exposes `--zmq-host`, but not every
custom ZMQ flag in the lower-level binary. If you need custom port/topic values,
check `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
and run the lower-level `just run g1_deploy_onnx_ref ...` command directly.

## Training, Evaluation, and Export

Run this section in the Isaac Lab training environment with
`pip install -e "gear_sonic/[training]"`. Do not run these commands from the
lightweight `.venv_sim`; that environment is only for MuJoCo sim/deploy tests.
This section trains and exports the SONIC policy, not the kinematic planner.

### Train or finetune SONIC

Quick sample-data run:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

Finetune from the release checkpoint:

```bash
python download_from_hf.py --training --no-smpl

python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path_to_robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path_to_smpl_filtered>
```

Training checkpoints are saved under:

```text
logs_rl/TRL_G1_Track/<experiment_name>-<timestamp>/
```

### Evaluate a checkpoint

Metrics:

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=128
```

Render videos:

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=8 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/tmp/renders" \
    "~manager_env/recorders=empty" "+manager_env/recorders=render"
```

Good deployment candidates should have high success rate, low tracking error,
and stable rendered rollouts. Do not use reward alone as the hardware gate.

### Export a trained checkpoint to ONNX

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++num_envs=1 \
    +export_onnx_only=true
```

Expected export files are written to an `exported/` directory next to the
checkpoint:

| File pattern | Purpose |
|---|---|
| `*_encoder.onnx` | Combined encoder |
| `*_decoder.onnx` | Decoder used by deploy stack |
| `*_g1.onnx` | G1 encoder + decoder |
| `*_teleop.onnx` | Teleop encoder + decoder |
| `*_smpl.onnx` | SMPL encoder + decoder |

For C++ deployment, use the encoder/decoder pair. The `deploy.sh` wrapper
expects a checkpoint prefix and appends `_encoder.onnx` and `_decoder.onnx`.

Example layout:

```text
gear_sonic_deploy/policy/my_policy/model_encoder.onnx
gear_sonic_deploy/policy/my_policy/model_decoder.onnx
gear_sonic_deploy/policy/my_policy/observation_config.yaml
```

Run it in sim:

```bash
cd gear_sonic_deploy
bash deploy.sh \
    --cp policy/my_policy/model \
    --obs-config policy/my_policy/observation_config.yaml \
    --planner planner/target_vel/V2/planner_sonic.onnx \
    --motion-data reference/example/ \
    --input-type keyboard \
    sim
```

Only after that sim2sim test is stable should you try:

```bash
bash deploy.sh \
    --cp policy/my_policy/model \
    --obs-config policy/my_policy/observation_config.yaml \
    --planner planner/target_vel/V2/planner_sonic.onnx \
    --motion-data reference/example/ \
    --input-type keyboard \
    real
```

## Decoupled WBC Real-Robot Stack

The decoupled WBC side also has a real/sim deployment launcher:

```bash
python decoupled_wbc/scripts/deploy_g1.py --help
```

It creates a tmux session named `g1_deployment` and starts panes for:

- G1 control loop
- Teleoperation policy
- Data exporter
- Optional camera viewer
- Optional MuJoCo sim loop

The default model path in the dataclass is:

```text
policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx
```

Use this stack when you are specifically working on the decoupled WBC teleop or
data-collection workflow. For first real G1 policy bring-up, use the SONIC
`gear_sonic_deploy` stack instead.

## Common Problems

| Symptom | Likely cause | Fix |
|---|---|---|
| ONNX file is tiny or unreadable | Git LFS assets not pulled | Run `git lfs pull` |
| `run_mujoco_gear_wbc.py` cannot find `ft92.onnx` | Config points to old local policy names | Update `g1_gear_wbc.yaml` to released ONNX paths |
| `deploy.sh` says `model_decoder.onnx` is missing | SONIC assets not downloaded | Run `python download_from_hf.py` |
| `find_package(TensorRT)` fails | `TensorRT_ROOT` unset or wrong version | Export `TensorRT_ROOT` and use required TensorRT version |
| MuJoCo sim starts but policy does nothing | C++ deploy was not started, or `]` was not pressed | Start Terminal 2 and press `]` |
| Robot moves in sim but not real | DDS interface not on `192.168.123.x` | Configure wired NIC and verify `ping 192.168.123.161` |
| Real robot motion is aggressive at start | Bad initial posture or wrong reference frame | Stop with `O`, reset posture, reinitialize heading with `I` |
| ZMQ stream ignored | Port/topic/protocol mismatch | Use default port/topic first, then debug custom settings |

## Recommended First Full Test Sequence

1. `git lfs pull`.
2. Run the decoupled WBC headless balance video.
3. Run the decoupled WBC headless walk video.
4. Build `gear_sonic_deploy`.
5. Run SONIC MuJoCo sim2sim with `--input-type keyboard sim`.
6. Play one walking reference clip.
7. Enter Planner mode and test Slow Walk.
8. Configure real G1 wired network.
9. Run `bash deploy.sh --input-type keyboard real`.
10. Start control with `]`.
11. Test idle/slow walk only.
12. Stop with `O`.
13. Review logs and motor temperatures before trying harder motions.

Deeper references:

- `deploy.md`: broad deployment walkthrough.
- `DEPLOY_READINESS.md`: current machine readiness notes.
- `docs/source/getting_started/quickstart.md`: official sim2sim quick start.
- `docs/source/tutorials/keyboard.md`: keyboard controls and planner modes.
- `docs/source/tutorials/gamepad.md`: gamepad controls.
- `docs/source/tutorials/manager.md`: runtime input switching.
- `docs/source/tutorials/zmq.md`: ZMQ motion streaming.
- `docs/source/user_guide/training.md`: SONIC training, evaluation, and ONNX export.
