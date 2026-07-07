# Deployment Guide — GR00T-WholeBodyControl

Everything needed to run the released policies in **MuJoCo** and on the **real Unitree G1**, organized so you can pick a scenario and copy the commands.

> **Safety.** Robots hurt. Before any real-robot run, keep an operator at the keyboard ready to hit the emergency stop (**`O`** in the C++ terminal, **Select** on the gamepad, or **A+B+X+Y** on PICO). Clear a safety zone. Use at your own risk.

---

## 1. TL;DR — pick a scenario

| I want to… | Track | Command |
|---|---|---|
| Record a quick MuJoCo video of the released Balance/Walk policies, no interactivity | A (Decoupled WBC) | `python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py --cmd walk --out rollouts/walk.mp4` |
| Drive the released Balance/Walk policies with keyboard in an on-screen MuJoCo viewer | A | `python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py` |
| Run the SONIC policy + planner in MuJoCo with keyboard control | B (SONIC) | T1 `python gear_sonic/scripts/run_sim_loop.py` • T2 `cd gear_sonic_deploy && ./deploy.sh --input-type keyboard sim` |
| Play reference-motion clips (macarena, squat, kick, …) in MuJoCo | B | Same as above, press `]` → `T` |
| Real robot: SONIC planner + keyboard / gamepad / manager | B | `cd gear_sonic_deploy && ./deploy.sh --input-type manager real` |
| Real robot: replay the planner waypoint video (e.g. `planner_mode0_target_goal3m`) over Ethernet | B | T1 `./deploy.sh --input-type zmq real` • T2 `python scripts_local/stream_planner_waypoint_zmq.py --waypoint 3 0 0 --target-vel 1.0` (see §6.4) |
| Real robot: VR whole-body teleop (PICO) | B | T1 run_sim_loop or real robot • T2 `deploy.sh --input-type zmq_manager` • T3 `pico_manager_thread_server.py` |
| Full decoupled_wbc real-robot stack (camera + teleop + data collection, tmux) | A | `python decoupled_wbc/scripts/deploy_g1.py <flags>` |

---

## 2. Two deployment tracks, one repo

There are **two independent controller families**, shipped side-by-side. They do not share a runtime.

### Track A — Decoupled WBC (the N1.5 / N1.6 controller)

- **What it is.** Two small ONNX policies that together cover standing/balancing and locomotion. "Decoupled" because the lower body is learned and the upper body is IK.
- **Models.** Already LFS-pulled in this repo:
  - `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx` — obs [1,516] → action [1,15]
  - `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx` — obs [1,516] → action [1,15]
- **Runtime.** Pure Python: `onnxruntime` + `mujoco`. No TensorRT, no C++ build, no Isaac Lab.
- **Inputs.** Keyboard only (for the included scripts). No gamepad/VR/ROS integration in the sim2mujoco script.
- **Logic.** Each control step (every 4 sim steps @ 200 Hz = 50 Hz control):
  1. Assemble a 86-dim observation (command, ang-vel, gravity, joint pos/vel, last action).
  2. Append to a 6-step ring buffer → flatten to 516-dim.
  3. If `‖loco_cmd‖ ≤ 0.05` → **Balance** policy, else → **Walk** policy.
  4. Action is rescaled to a joint-position target, applied via PD control at 200 Hz.

### Track B — SONIC (the GEAR-SONIC production stack)

- **What it is.** A single whole-body motion-tracking foundation model plus a kinematic planner. One policy drives everything — walking, crawling, teleop, dance clips — by tracking target poses.
- **Models** (pulled via `python download_from_hf.py`):
  - `gear_sonic_deploy/policy/release/model_encoder.onnx` — obs [1, 1762] → tokens [1, 64]
  - `gear_sonic_deploy/policy/release/model_decoder.onnx` — obs [1, 994] → action [1, 29] (29-DoF G1)
  - `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` — 11 inputs → `mujoco_qpos [1,64,36]` + `num_pred_frames`
  - `gear_sonic_deploy/reference/example/` — 13 reference clips (walking, macarena, squat, kick, dance, lunge, one-leg jump, …)
- **Runtime.** C++ / TensorRT binary (`g1_deploy_onnx_ref`). Uses Unitree SDK2 DDS. 500 Hz LowCmd to the robot.
- **Inputs.** keyboard / gamepad / gamepad_manager / manager / zmq / zmq_manager / ros2.
- **Logic.**
  1. **Motion target selection** — one of: replay a reference clip, receive a streamed motion over ZMQ, or let the kinematic planner generate motion from a target velocity + mode.
  2. **Optional encoder** compresses a history window into a 64-dim token (mode 0 / 1, toggleable with `Z`).
  3. **Decoder** consumes current obs + target + encoder tokens → 29 joint targets.
  4. Commands published over DDS as Unitree `LowCmd_` at 500 Hz. Planner runs at 10 Hz.

**If you don't have TensorRT + a C++ build environment, you cannot run Track B.** Track A is the fallback for pure-Python experimentation.

---

## 3. What policies / motion clips are shipped

### ONNX policies

| File | Track | Size | Purpose |
|---|---|---|---|
| `GR00T-WholeBodyControl-Balance.onnx` | A | 1.8 MB | Standing / small corrections, used when locomotion command ≈ 0 |
| `GR00T-WholeBodyControl-Walk.onnx` | A | 1.8 MB | Walking in commanded direction |
| `model_encoder.onnx` | B | 50 MB | History → 64-dim token (mode-switchable) |
| `model_decoder.onnx` | B | 40 MB | Obs + target + token → 29 joint actions |
| `planner_sonic.onnx` | B | 774 MB | Kinematic planner: generates 64-frame `mujoco_qpos` trajectories from high-level commands |

### SONIC reference motion clips (`gear_sonic_deploy/reference/example/`)

Cycle with `N` / `P` in Track B keyboard mode:

- `walking_quip_360_R_002__A428` — 360° walking turn
- `macarena_001__A545` — Macarena dance
- `neutral_kick_R_001__A543` — Right-leg kick
- `squat_001__A359` — Squat
- `forward_lunge_R_001__A359_M` — Forward lunge
- `tired_forward_lunge_R_001__A359_M`, `tired_one_leg_jumping_R_001__A359`, `tired_one_leg_jumping_R_001__A359_M`
- `dance_in_da_party_001__A464`, `dance_in_da_party_001__A464_M`

### SONIC planner "motion sets" (in Planner mode, cycle with `N`/`P`)

- **Set 0 — Locomotion (Standing):** Slow Walk (0.2–0.8 m/s), Walk, Run (1.5–3.0 m/s), Happy, Stealth, Injured
- **Set 1 — Squat variants**
- **Set 2 — Boxing**
- **Set 3 — Styled walking**
- Additional sets: kneeling, hand/elbow crawling (selectable with `1`–`8` inside each set)

Full index: `docs/source/references/planner_onnx.md`.

---

## 4. Prerequisites

### Common (both tracks)

```bash
# From repo root, once:
sudo apt install -y git-lfs tmux ffmpeg
git lfs install
git lfs pull
python check_environment.py                 # sanity check
```

### Track A — Decoupled WBC MuJoCo sim only

```bash
bash install_scripts/install_mujoco_sim.sh  # creates .venv_sim with mujoco, onnxruntime, pinocchio, unitree_sdk2py
source .venv_sim/bin/activate
# The sim2mujoco script also wants pynput + torch + yaml; install if missing:
uv pip install pynput torch pyyaml onnxruntime imageio imageio-ffmpeg
```

### Track A — Decoupled WBC real robot

```bash
pip install -e "decoupled_wbc[full]"        # heavy: ROS deps, lerobot, realsense, cv_bridge, etc.
```

### Track B — SONIC models

```bash
python download_from_hf.py                  # pulls encoder/decoder/planner → gear_sonic_deploy/
# or, without the 774 MB planner:
python download_from_hf.py --no-planner
```

### Track B — SONIC C++ deploy build

```bash
# TensorRT 10.x from https://developer.nvidia.com/tensorrt/download/10x
export TensorRT_ROOT=$HOME/TensorRT
export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH
cd gear_sonic_deploy
bash scripts/install_deps.sh                # installs just, cmake, clang, onnxruntime C package
source scripts/setup_env.sh
just build                                  # or: ./deploy.sh sim (builds, then runs)
./target/release/run_tests                  # optional: C++ unit tests
```

### Real-robot networking (wired Ethernet)

Unitree G1 runs its DDS bus on a wired `192.168.123.x/24` subnet; power up the
robot, plug its Ethernet cable into your host NIC, and bring up a static IP on
that NIC in the same subnet:

```bash
# one-time per session (replace eno1 with your NIC from `ip a`)
sudo ip addr add 192.168.123.99/24 dev eno1
sudo ip link set eno1 up

# persistent via NetworkManager (optional):
nmcli con add type ethernet ifname eno1 con-name g1-wired \
    ip4 192.168.123.99/24 ipv4.method manual

ping -c 2 192.168.123.161   # default robot IP; should respond in <1 ms
```

`deploy.sh real` auto-detects the first NIC with a 192.168.123.x address, so
you do not need `--iface` after the ping succeeds. Do not run Wi-Fi in the
same subnet (DDS multicast collisions). Disabling Unitree's high-level
locomotion service is **not** necessary — SONIC takes the lock itself via
`MotionSwitcherClient`.

---

## 5. Track A — Decoupled WBC, step-by-step

### 5.1 MuJoCo — headless recording (what we ran in this session)

```bash
source .venv_sim/bin/activate

# Walk command 0.5 m/s forward, 8 s, saves MP4:
python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd walk \
    --out rollouts/wbc_walk.mp4

# Balance — idle stance, zero command:
python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd balance \
    --out rollouts/wbc_balance.mp4
```

Options: `--fps` (default 50), `--width` / `--height` (default 640×480), `--config` (swap YAML to use a different XML / policy pairing).

### 5.2 MuJoCo — interactive viewer

```bash
source .venv_sim/bin/activate
cd decoupled_wbc/sim2mujoco
python scripts/run_mujoco_gear_wbc.py
```

Key bindings (from the pynput listener):

| Key | Action |
|---|---|
| `w` / `s` | Forward / backward command (±0.1) |
| `a` / `d` | Left / right sideways command (±0.1) |
| `q` / `e` | Yaw rate command (±0.1) |
| `z` | Reset all commands to init |
| `1` / `2` | Raise / lower target height (±0.05 m) |
| `3`–`8` | Roll / pitch / yaw target (±0.2 rad) |
| `m` / `n` | Gait frequency (±0.1 Hz) |

> `run_mujoco_gear_wbc_gait.py` is the same script with gait-phase clock inputs enabled — useful if you're training a gait-aware variant.

### 5.3 Real robot — full decoupled_wbc stack

```bash
# After pip install -e "decoupled_wbc[full]" and the robot is powered on, sitting:
python decoupled_wbc/scripts/deploy_g1.py --help
```

`deploy_g1.py` spawns a **tmux session `g1_deployment`** with panes for control loop, data collection, teleop, and camera viewer. It wraps `decoupled_wbc.control.main.*` entry points. Attach with `tmux attach -t g1_deployment`.

---

## 6. Track B — SONIC, step-by-step

### 6.1 MuJoCo — two-terminal sim2sim

**Terminal 1** — MuJoCo simulator (Python, on the host):

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
# (optional flags — see `python gear_sonic/scripts/run_sim_loop.py --help`)
# --env_name default
# --enable-offscreen / --enable-image-publish   (for camera publishing)
```

This process:
- spins a MuJoCo simulator at `SIMULATE_DT = 5 ms` (200 Hz),
- publishes `LowState_` / `IMUState_` on **loopback DDS** (`INTERFACE: "lo"`),
- subscribes to `LowCmd_` from the deploy binary,
- renders the viewer at `VIEWER_DT = 20 ms`.

It is the **sim bridge** — without it, the C++ deploy binary has nothing to command.

**Terminal 2** — C++ deployment (from `gear_sonic_deploy/`):

```bash
cd gear_sonic_deploy
./deploy.sh --input-type keyboard sim      # or: manager / gamepad / zmq / zmq_manager / ros2
```

`deploy.sh sim` automatically:
1. Resolves the loopback interface (`lo` on Linux, `lo0` on macOS).
2. Sets `--disable-crc-check` (MuJoCo bridge doesn't sign LowCmd).
3. Checks that all required ONNX + YAML files exist under `policy/release/`, `planner/target_vel/V2/`, `reference/example/`.
4. Sources `scripts/setup_env.sh` (adds TensorRT + CUDA to `LD_LIBRARY_PATH`).
5. Runs `just build` (incremental).
6. Execs the compiled binary:

```bash
just run g1_deploy_onnx_ref $IFACE \
    policy/release/model_decoder.onnx \
    reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type <mode> --output-type all --zmq-host localhost \
    --disable-crc-check
```

**Start the session** (same on every `--input-type`):

1. In Terminal 2, press `]` → control system starts, obs are populated.
2. In the MuJoCo window, press `9` → drops the simulated robot to the ground.
3. Back in Terminal 2, press `T` → plays the currently selected reference motion.
4. `N` / `P` → next / previous reference clip. `T` again to play it.
5. `R` → restart current motion at frame 0 without killing the policy.
6. `ENTER` → switch to Planner mode (real-time locomotion). `W A S D` to move, `,` / `.` to strafe, `N`/`P` to change motion set, `1`–`8` to change mode, `9`/`0` speed, `-`/`=` height.
7. `O` → emergency stop and exit. Always.

### 6.2 Real robot

Same commands as sim, drop `sim` → `real`:

```bash
cd gear_sonic_deploy
./deploy.sh --input-type manager real
```

`deploy.sh real`:
1. Scans NICs for an IP in `192.168.123.x`; falls back to the first non-loopback NIC with a warning.
2. **Does not** pass `--disable-crc-check` — the G1 requires signed LowCmds.
3. Prompts for `y/n` confirmation before starting (watch the banner).
4. Launches the same binary, same keys apply.

**Pre-flight for real:**

- G1 is in damping mode (not locked), suspended in a gantry or with a spotter.
- The robot is in a **safe initial posture** (sitting, slightly hunched).
- You can reach the E-stop key / controller select button.
- Keep motion sets first-run in **Walk / Slow Walk** until you're confident in the workspace.

### 6.3 Real robot — replaying the planner waypoint rollout over Ethernet

This is how to deploy the exact behaviour shown in
`rollouts/planner_mode0_target_goal3m.mp4` (SONIC planner with a 3 m forward
waypoint target) on the physical G1. The robot and host must be on the same
wired `192.168.123.x` subnet.

**Why it needs streaming, not a hotkey.** The `planner_sonic.onnx` input
`has_specific_target` is exposed on the C++ side in `localmotion_kplanner_*.hpp`,
but no shipped input interface (`keyboard`, `gamepad`, `manager`, `zmq`, `ros2`)
ever writes to it — the field stays zero, so pressing `W` / clicking on the
gamepad cannot send the robot to a waypoint. Two options:

- **(A — recommended) Stream a precomputed rollout.** Run the planner ONNX on
  the host with `has_specific_target=1`, serialize the 64-frame×36-col qpos as
  a 29-DoF joint-pos trajectory, and publish to the C++ binary's
  `--input-type zmq` pose topic. Helper script shipped:
  `scripts_local/stream_planner_waypoint_zmq.py`.
- **(B) Patch the C++ planner** to pull `target_position`/`heading` from one of
  the ZMQ topics (a handful of lines in `zmq_manager.hpp` + `localmotion_kplanner_*.hpp`
  around the existing `has_specific_target_` / `specific_target_positions_`
  members). Cleaner long-term, but a source change, so out of scope for a
  first bring-up.

**Ethernet setup.**

```bash
# On the host laptop, wired NIC in the robot's subnet:
sudo ip addr add 192.168.123.99/24 dev eno1              # adjust eno1 to your NIC
sudo ip link set eno1 up
ping -c 2 192.168.123.161                                # G1 default — should respond
```

`deploy.sh real` autodetects the first NIC with a `192.168.123.x` address, so
as long as the ping succeeds you do not have to pass `--iface`.

**Pre-flight on the robot.** G1 in damping mode, suspended or with a spotter,
arms free of the gantry, E-stop in reach. Confirm the robot pose is close to
the Python planner's seed (standing, z≈0.75 m) — otherwise the streamed target
will diverge from the actual state and the first few frames will jerk.

**Terminal 1** — C++ deploy binary on the host, with ZMQ input:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh                  # loads TensorRT / CUDA paths
./deploy.sh \
    --input-type zmq \
    --zmq-host 127.0.0.1 --zmq-port 5556 \
    --zmq-topic pose \
    --output-type zmq \
    real
# Answer y at the confirmation prompt. Wait for "Init done".
```

`deploy.sh real` will, in order: resolve the 123.x interface, skip
`--disable-crc-check`, cmake+just-build the binary (~20 s incremental), source
TensorRT from `$TensorRT_ROOT`, and exec `g1_deploy_onnx_ref`.

**Terminal 2** — Python streamer (in the `.venv_sim` env, since we only need
`onnxruntime`, `numpy`, `pyzmq`):

```bash
source .venv_sim/bin/activate && unset PYTHONPATH
python scripts_local/stream_planner_waypoint_zmq.py \
    --waypoint 3.0 0.0 0.0 \
    --target-vel 1.0 \
    --mode 0 \
    --chunks 8 \
    --host '*' --port 5556
```

Flags:
- `--waypoint X Y Z` — goal in the planner's local frame (metres). The published
  video used `3 0 0`; try `2 1 0` to curve, `0 0 0` to stand.
- `--target-vel` — m/s forwarded to the planner; 1.0 m/s for a steady walk.
- `--mode` — `0=g1`, `1=teleop`, `2=smpl`. Mode 0 only *moves* when a waypoint
  is supplied; modes 1/2 will walk in `movement_direction` even without a
  waypoint.
- `--chunks` — how many 64-frame planner calls to chain (496 frames ≈ 10 s at 50 Hz).
- `--dry-run` — compute the trajectory and print stats without publishing.
  Useful to sanity-check the waypoint before the binary is running.

**Back in Terminal 1**, once the streamer prints `[zmq] bound …`:

1. Press `]` → control system starts, motors engage.
2. Press `9` if you're in a MuJoCo bridge (for real robot, skip — the robot is
   already on the ground).
3. Press `T` to begin tracking the streamed motion. You should see the robot
   start walking toward the waypoint.
4. `O` any time to emergency-stop and exit.

**What's being streamed.** One `pose` message per 64-frame chunk on topic
`pose`, Protocol v1 layout (`joint_pos[N,29]` f32, `joint_vel[N,29]` f32,
`body_quat_w[N,4]` f32 wxyz, `frame_index[N]` i64, `catch_up` u8). Header is
JSON padded to 1280 bytes — same schema as
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_zmq_manager.py`. The
streamer sends frame 0 with `catch_up=1` to reset the merger, and chains
subsequent chunks at 90 % of real-time cadence so the motion merger keeps
its buffer full.

**Things that break this flow.**
- Wrong Python env — `pyzmq` must be installed in the env you're running the
  streamer from (`.venv_sim` already has it via `install_mujoco_sim.sh`).
- Port mismatch — C++ binary listens on `--zmq-port` (default 5556); streamer
  publishes there. If you bind on `127.0.0.1` in T1 you must use `--host
  127.0.0.1` in T2, not `*`.
- Header-size mismatch — this stack is on the 2026-03-24 spec (`HEADER_SIZE =
  1280`). Older streamers using 512 bytes will silently be ignored.
- Robot not starting from standing — streamed pose frame 0 assumes z≈0.75 m
  identity quaternion. If the real G1 is squatted/sitting, the first
  policy frame will try to yank it upright. Prefer to press `]` with the
  robot already standing stably, or adapt the streamer to seed the planner
  context with the robot's actual LowState qpos (read via ZMQ debug stream on
  port 5557 or via DDS directly).

### 6.4 Output types

Add `--output-type` to `deploy.sh`:

| Value | Publishes |
|---|---|
| `zmq` (default) | Debug / telemetry on `tcp://*:5557` |
| `ros2` | ROS2 topics (requires build with `HAS_ROS2`, needs ROS 2 Humble) |
| `all` | Both |

---

## 7. Input modes explained (Track B)

The `--input-type` flag picks what the C++ binary listens to. All modes can emergency-stop with `O` (keyboard) or `Select` (gamepad).

| Mode | Source | Use when | Command |
|---|---|---|---|
| `keyboard` | Your terminal | Sim / desk tests | `./deploy.sh --input-type keyboard sim` |
| `gamepad` | Unitree wireless gamepad (via `LowState_.wireless_remote`) | Real robot outdoors | `./deploy.sh --input-type gamepad real` |
| `gamepad_manager` | Gamepad + on-board manager | Real robot with planner | `./deploy.sh --input-type gamepad_manager --planner-file ... real` |
| `manager` | **Any interface, hot-swappable** at runtime | Default recommendation for real | `./deploy.sh --input-type manager real` |
| `zmq` | External ZMQ publisher of SMPL or G1 qpos | Custom motion streamers, mocap | `./deploy.sh --input-type zmq --zmq-host <ip> --zmq-port 5556 --zmq-topic pose sim` |
| `zmq_manager` | ZMQ + manager (planner + VR streaming) | **PICO VR teleop** | `./deploy.sh --input-type zmq_manager sim` |
| `ros2` | ROS 2 subscriber | ROS integrations | `./deploy.sh --input-type ros2 real` |

### Manager mode — runtime switching

Once launched in `manager`:

| Hotkey | Switches to |
|---|---|
| `Shift+1` (`!`) | Keyboard |
| `Shift+2` (`@`) | Gamepad |
| `Shift+3` (`#`) | ZMQ |

Each switch triggers a safety reset: motion target snaps back to frame 0, planner disables.

### ZMQ / VR teleop (3 terminals)

```bash
# T1 — MuJoCo sim (or skip for real robot):
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# T2 — deploy with zmq_manager:
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim        # or real

# T3 — PICO pose publisher:
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_vr3pt --vis_smpl --waist_tracking
```

ZMQ protocol details: `docs/source/tutorials/zmq.md` documents the message schema so you can write your own streamer without PICO hardware.

---

## 8. Key binding cheat sheet (Track B, keyboard mode)

### System

| Key | Action |
|---|---|
| `]` | Start control system |
| `O` | **Emergency stop + exit** |
| `ENTER` | Toggle Normal / Planner mode |
| `I` | Re-initialize base quaternion (reset heading to zero) |
| `Z` | Toggle encoder mode 0 ↔ 1 (if encoder loaded) |
| `F` | TTS motor temperatures |

### Normal mode (reference motion tracking)

| Key | Action |
|---|---|
| `T` | Play current reference motion |
| `R` | Restart motion at frame 0 (without killing policy) |
| `N` / `P` | Next / previous motion clip |
| `Q` / `E` | Nudge heading ±π/12 rad |

### Planner mode (real-time locomotion)

| Key | Action |
|---|---|
| `W` / `S` | Forward / backward |
| `A` / `D` | Walk + steer heading left / right |
| `,` / `.` | Strafe left / right |
| `Q` / `E` | Facing direction (planner level), ±π/6 |
| `J` / `L` | Heading (policy level), ±π/12 |
| `N` / `P` | Next / previous motion set |
| `1`–`8` | Pick mode within set |
| `9` / `0` | Decrease / increase speed |
| `-` / `=` | Decrease / increase body height (0.2–0.8 m, non-standing sets) |
| `T` | Play |
| `R` / `` ` `` / `~` | **Instant momentum reset** |

### Gamepad mode (real robot only)

| Button | Action |
|---|---|
| **Start** | Start control |
| **Select** | **Emergency stop + exit** |
| **A** | Play reference motion |
| **B** | Pause at frame 0 |
| **R1** / **L1** | Next / previous motion |
| **X** or **Y** | Reinitialize base quaternion |
| **D-pad L/R** | Heading ±0.1 rad |
| **F1** | Toggle Normal / Planner |
| Left stick | Movement direction |
| Right stick | Facing direction (Planner mode) |
| **R2** / **L2** | Speed or height ± |

---

## 9. Common workflows

### "I just want to see the released Balance/Walk policies move"

```bash
source .venv_sim/bin/activate
python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py --cmd walk --out rollouts/walk.mp4
mpv rollouts/walk.mp4
```

### "Play the Macarena reference motion in MuJoCo"

```bash
# T1:
python gear_sonic/scripts/run_sim_loop.py
# T2:
cd gear_sonic_deploy && ./deploy.sh --input-type keyboard sim
# In T2: press `]`, then in MuJoCo press `9`, back to T2:
#   press `N` repeatedly until the terminal shows "macarena_001__A545"
#   press `T` to play
```

### "Drive the robot around with WASD in MuJoCo"

```bash
# T1:
python gear_sonic/scripts/run_sim_loop.py
# T2:
cd gear_sonic_deploy && ./deploy.sh --input-type keyboard sim
# In T2: `]`, then in MuJoCo press `9`, back to T2:
#   ENTER → Planner mode, then `2` (Walk), then `W` to move forward
```

### "Deploy the `planner_mode0_target_goal3m` video on the real G1"

Full detail in §6.3. Two-terminal short form:

```bash
# T1 — C++ deploy, real G1 on 192.168.123.x
cd gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh --input-type zmq --zmq-host 127.0.0.1 --zmq-port 5556 real
# answer y, press `]` after Init done, then `T` once streamer is running

# T2 — Python streamer (on the same host)
source .venv_sim/bin/activate && unset PYTHONPATH
python scripts_local/stream_planner_waypoint_zmq.py \
    --waypoint 3.0 0.0 0.0 --target-vel 1.0
```

Prereqs: §4 TensorRT + C++ build complete (see `DEPLOY_READINESS.md` for the
outstanding items on this machine), Ethernet to `192.168.123.x` up, robot
standing in damping mode.

### "Deploy SONIC on the real G1"

```bash
# Robot powered, suspended/spotted, damping on, 192.168.123.x network up.
cd gear_sonic_deploy
source scripts/setup_env.sh          # sources TensorRT, CUDA, LD_LIBRARY_PATH
./deploy.sh --input-type manager real
# Deploy.sh prompts y/n. Then: `]`, wait for Init done, then gamepad or keyboard.
```

### "VR teleop to stream my body pose to the robot"

See `docs/source/tutorials/vr_wholebody_teleop.md`. Summary: Terminal 1 `run_sim_loop` (or skip for real), Terminal 2 `deploy.sh --input-type zmq_manager`, Terminal 3 `pico_manager_thread_server.py --manager --vis_vr3pt ...`.

---

## 10. Logging and telemetry

`deploy.sh` accepts:

- `--target-motion-logfile <path.csv>` — written motion targets per frame
- `--planner-motion-logfile <path.csv>` — planner outputs
- `--policy-input-logfile <path.csv>` — observations fed to the decoder
- `--enable-csv-logs` — turn on default CSV logs
- `--enable-motion-recording` — record ZMQ/planner sessions for replay
- `--logs-dir <dir>` — override log base (default `logs/<timestamp>/`)

ZMQ debug stream is on `tcp://*:5557` topic `g1_debug` by default — point your own subscriber for live telemetry.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `git lfs pull` does nothing, ONNX files are a few bytes | LFS not initialized | `git lfs install && git lfs pull` |
| `find_package(TensorRT REQUIRED)` fails in `just build` | `TensorRT_ROOT` unset | `export TensorRT_ROOT=$HOME/TensorRT` |
| `deploy.sh` says "Missing file: policy/release/model_decoder.onnx" | Models not downloaded | `python download_from_hf.py` |
| MuJoCo window blank | No DISPLAY / running headless | Use the record script or `xvfb-run` |
| `ModuleNotFoundError: lark / launch_testing` in pytest or sim | ROS leaked into venv via PYTHONPATH | `unset PYTHONPATH` before activating venv |
| Robot twitches / falls immediately on `T` | Robot was not dropped, or not in safe initial posture | In MuJoCo press `9` first, or on real robot ensure the damping posture is correct |
| ZMQ streaming seems to lag | Conflate disabled | Add `--zmq-conflate` |
| "ZMQ header size" mismatch error | Old streamer on pre-2026-03-24 protocol | Header size changed to 1280 bytes; update publisher |
| `--input-type gamepad` in sim does nothing | Gamepad is real-robot-only (reads `LowState_.wireless_remote`) | Use `keyboard` or `manager` in sim |
| Waypoint key (e.g. "go to 3 m ahead") does nothing in Planner mode | C++ runtime leaves `has_specific_target=0`; no input interface writes it | Use §6.3 streamer to replay a precomputed planner rollout over ZMQ, or patch the input interface |
| MuJoCo sim is fast/slow vs wall time | `SIMULATE_DT` and `VIEWER_DT` in `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml` | Tune, or enable `sim_sync_mode` |

---

## 12. File map (where everything lives)

```
GR00T-WholeBodyControl/
├── decoupled_wbc/
│   ├── sim2mujoco/                       # Track A sim (self-contained Python)
│   │   ├── scripts/
│   │   │   ├── run_mujoco_gear_wbc.py             # interactive viewer
│   │   │   ├── run_mujoco_gear_wbc_record.py      # headless MP4 recorder (added)
│   │   │   └── run_mujoco_gear_wbc_gait.py        # gait-clock variant
│   │   └── resources/robots/g1/
│   │       ├── g1_gear_wbc.xml / g1.xml           # MJCF scenes
│   │       ├── g1_gear_wbc.yaml / g1.yaml         # original configs
│   │       ├── g1_gear_wbc_headless.yaml          # headless variant (added)
│   │       └── policy/
│   │           ├── GR00T-WholeBodyControl-Balance.onnx
│   │           └── GR00T-WholeBodyControl-Walk.onnx
│   └── scripts/deploy_g1.py              # Track A real-robot launcher (tmux)
│
├── gear_sonic/                           # Track B training + sim bridge
│   ├── scripts/
│   │   ├── run_sim_loop.py                        # MuJoCo ↔ DDS bridge (T1)
│   │   └── pico_manager_thread_server.py          # VR pose streamer (T3)
│   └── utils/mujoco_sim/wbc_configs/
│       └── g1_29dof_sonic_model12.yaml            # sim interface + scene cfg
│
├── gear_sonic_deploy/                    # Track B C++ runtime
│   ├── deploy.sh                                  # main launcher
│   ├── .justfile                                  # build / run / test
│   ├── src/g1/g1_deploy_onnx_ref/                 # C++ source
│   │   ├── include/input_interface/               # keyboard, gamepad, zmq, ros2, manager
│   │   ├── include/output_interface/              # zmq, ros2
│   │   ├── include/control_policy.hpp             # decoder runner
│   │   ├── include/encoder.hpp                    # encoder runner
│   │   └── include/localmotion_kplanner*.hpp      # planner runner
│   ├── policy/release/                            # decoder+encoder+obs YAML (HF)
│   ├── planner/target_vel/V2/                     # planner ONNX (HF)
│   ├── reference/example/                         # 13 motion clips (in repo)
│   └── g1/                                        # URDF / MJCF variants
│
└── rollouts/                             # output videos (you create)
```

---

## 13. Where to read more

- `docs/source/getting_started/quickstart.md` — narrative version of §6.1.
- `docs/source/tutorials/keyboard.md`, `gamepad.md`, `zmq.md`, `manager.md`, `vr_wholebody_teleop.md`, `data_collection.md` — per-input deep dives.
- `deploy-design.md` (repo root) — architecture rationale: why we don't call Unitree's high-level locomotion RPC, how motion targets flow through the runtime.
- `docs/source/references/planner_onnx.md` — full planner input/output spec, motion-set/mode index map.
- `CLAUDE.md` — quick orientation for future AI assistants working in this repo.
- `DEPLOY_READINESS.md` — current asset-readiness snapshot for this machine (generated 2026-04-16).
