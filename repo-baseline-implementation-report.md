# GR00T-WholeBodyControl Baseline and Implementation Report

Date: 2026-06-27

This report summarizes what this repository currently provides, what should be treated as baseline, what implementation surfaces already exist, and what is still missing before building a design beyond this repo. It is written from the current local checkout, including local untracked notes, helper scripts, and validation artifacts. It does not assume that every dependency or hardware path is ready on this machine; those readiness limits are called out explicitly.

## 0. How to read this report

Use this document as a handoff map before designing anything new on top of the repo.

- "Baseline" means a released asset, architecture, workflow, or interface we should preserve and reuse unless we have a clear reason to replace it.
- "Provided implementation" means code that already exists in the repo and can be extended, wrapped, or called directly.
- "Local additions" means untracked or locally created helper docs/scripts/videos in this checkout. They are useful, but they should be reviewed before treating them as upstream-owned.
- "Missing" means work our design must provide. The repo gives a strong control substrate, but it does not provide our task-specific planner, policy, dataset, validation plan, or real-robot proof.

## 1. Executive summary

`GR00T-WholeBodyControl` is a humanoid whole-body-control monorepo centered on Unitree G1. It contains three major controller or motion systems:

1. `decoupled_wbc/`: the older GR00T N1.5/N1.6 style controller, with learned lower-body locomotion and upper-body IK.
2. `gear_sonic/` plus `gear_sonic_deploy/`: the GEAR-SONIC research and deployment stack. This is the main modern baseline for whole-body motion tracking, teleoperation, planner-driven locomotion, and VLA integration.
3. `motionbricks/`: a preview real-time latent generative motion system, separate from the SONIC controller but aligned with future motion-generation research.

The repo is not a single application. It is a collection of research, simulation, deployment, teleoperation, data collection, and documentation paths. The most important architectural split is:

- Training and evaluation mostly live in Python under `gear_sonic/`, using Isaac Lab, Hydra configs, and a forked TRL PPO trainer.
- Real G1 deployment lives in C++ under `gear_sonic_deploy/`, using TensorRT, ONNX Runtime assets, Unitree SDK2 DDS, and a 500 Hz low-level command writer.
- Local visualization and dry-run tooling can use MuJoCo without hardware.

For our own design beyond the repo, the strongest baseline is not one file or model. It is the full SONIC abstraction:

```text
input source -> target motion or latent token -> SONIC policy -> low-level G1 joint command
```

The safest extension strategy is to preserve that abstraction and choose one integration point:

- produce better target motions,
- produce external latent tokens,
- add a new planner/source of high-level commands,
- fine-tune/retrain SONIC,
- or run a VLA/policy above SONIC.

### 1.1 Baseline versus implementation at a glance

| Area | Baseline provided | Implementation already present | What our design must still provide |
|---|---|---|---|
| Decoupled WBC | Balance/Walk ONNX policies, G1 MuJoCo assets, Python lower-body control path. | ONNX Runtime runners, teleop/IK stack, G1 env helpers, tests, local headless recorder. | Usually nothing unless we need a simple comparison controller or a new decoupled policy. |
| SONIC policy | Released encoder/decoder ONNX, PyTorch checkpoint, observation contract, universal-token architecture. | Isaac Lab training/eval code, TRL PPO fork, data conversion, ONNX export, runtime wrappers. | Task-specific data, reward/config changes, fine-tuned checkpoint, metrics, sim-to-real validation if we alter the policy. |
| SONIC deploy | C++20 TensorRT runtime, Unitree SDK2 low-level DDS path, planner/reference/ZMQ target unification. | `deploy.sh`, CMake/Just build, input interfaces, output/debug interfaces, safety checks, 500 Hz command writer. | Local TensorRT/toolchain setup, sim2sim validation, real G1 validation, any custom input/protocol adapter. |
| Kinematic planner | `planner_sonic.onnx` and target-velocity planner contract. | C++ planner integration, local Python replay, waypoint ZMQ streamer prototype. | New planner model or waypoint interface if our design needs behavior beyond released planner modes. |
| Teleop and data collection | PICO/VR, ZMQ manager, camera/data exporter workflows. | Python teleop servers, input readers, camera clients, dataset scripts, docs. | Our task episodes, calibration, recording policy, data QA, and possibly task-specific retargeting. |
| VLA over SONIC | 64-token plus hand-joint action interface and launcher wiring. | `run_vla_inference.py`, `launch_inference.py`, inference utilities, camera integration. | Fine-tuned VLA model, prompts/tasks, latency policy, hand/action formatting, safety validation. |
| MotionBricks | Generative motion preview and pretrained checkpoint structure. | MuJoCo demo, VQ-VAE/pose/root training code, G1 motion representation docs. | Adapter from generated motion to SONIC target streams, or a separate research integration. |
| Documentation and evidence | Sphinx docs, root README, local architecture/deploy/readiness notes. | Practical guides, rollout videos, model-signature checks. | Our design spec, validation logs, PR-ready notes, and final operating procedure. |

## 2. Current repository scope

Top-level structure:

| Path | Role |
|---|---|
| `decoupled_wbc/` | Python decoupled whole-body control stack: lower-body ONNX RL policy, upper-body IK/interpolation, teleop/data/sim support. |
| `gear_sonic/` | SONIC training, evaluation, data processing, teleoperation, camera, VLA inference, and MuJoCo utilities. |
| `gear_sonic_deploy/` | C++20 real-time deploy stack for G1, TensorRT inference, Unitree SDK2 DDS I/O, kinematic planner, input/output interfaces. |
| `motionbricks/` | Real-time generative motion preview stack with VQ-VAE, pose/root backbones, MuJoCo demo, G1 skeleton assets. |
| `docs/` | Sphinx documentation for install, quickstart, training, deploy, teleop, VLA, ZMQ, observation configs. |
| `install_scripts/` | Environment bootstrappers for MuJoCo sim, PICO/VR teleop, data collection, camera server, inference, ROS. |
| `external_dependencies/` | Unitree Python SDK and XRoboToolkit support dependencies. |
| `systemd/` | Camera service unit support. |
| `sample_data/` | Small sample robot, SMPL, and SOMA motion data for smoke tests. |
| `sonic_release/` | Local downloaded PyTorch SONIC checkpoint and config in this checkout. |
| `rollouts/` | Local generated MP4 rollout/replay evidence. |
| `scripts_local/` | Local helper scripts for replaying/streaming planner and reference outputs. |

Important local documentation already present:

| File | Purpose |
|---|---|
| `architecture-overview.md` | Detailed architectural map of the monorepo and research ideas. |
| `deploy-design.md` | Deep dive into the C++ G1 deploy runtime. |
| `deploy.md` | Practical deployment guide across decoupled WBC and SONIC. |
| `guide.md` | Policy inventory, test order, MuJoCo and real-robot operating guide. |
| `DEPLOY_READINESS.md` | Local readiness audit: assets present, toolchain missing, next deploy steps. |
| `policy_review_results.md` | Local model signature checks and generated review-video summary. |
| `rollouts/README.md` | Explanation of locally generated MP4 videos and what they validate. |

### 2.1 Runtime and dependency boundaries

The repo intentionally does not use one Python environment for everything. This is a baseline design constraint, not an accident.

| Workflow | Main path | Environment/dependencies | Notes |
|---|---|---|---|
| Python formatting/tests | repo root, `decoupled_wbc/tests/` | Python 3.10+, Black, isort, Ruff, pytest. | Root `pyproject.toml` is tooling-focused; package installs are under subprojects. |
| Decoupled WBC sim | `decoupled_wbc/sim2mujoco/` | `.venv_sim`, MuJoCo, ONNX Runtime, Pinocchio, PyZMQ. | Fastest local policy smoke test. |
| Decoupled WBC full stack | `decoupled_wbc/control/` | `pip install -e "decoupled_wbc[full]"`. | Heavy teleop/data/robot dependencies including ROS-related packages. |
| SONIC training/eval | `gear_sonic/` | Isaac Lab Python 3.11 env, Hydra, Accelerate, TRL, Transformers, W&B. | Training is GPU-heavy and separate from `.venv_sim`. |
| SONIC sim bridge | `gear_sonic/scripts/run_sim_loop.py` | `.venv_sim` plus Unitree Python SDK/MuJoCo dependencies. | Provides Unitree-compatible DDS state/command bridge for C++ deploy. |
| SONIC real deploy | `gear_sonic_deploy/` | C++20, CMake, Just, Ninja, Clang, TensorRT, CUDA toolkit, ONNX Runtime C/C++. | Real G1 path; same binary is used for sim and hardware. |
| VR teleop | `gear_sonic/scripts/pico_manager_thread_server.py` | `.venv_teleop`, PyZMQ, Pinocchio, optional visualization packages. | Streams VR/teleop targets into SONIC. |
| Data collection | `gear_sonic/scripts/launch_data_collection.py` | `.venv_data_collection`, camera/ZMQ/LeRobot dependencies. | Produces task episodes for later policy/VLA work. |
| VLA inference | `gear_sonic/scripts/run_vla_inference.py` | `.venv_inference`, Isaac-GR00T, PyZMQ, camera clients. | High-level VLA emits SONIC token and hand actions. |
| MotionBricks | `motionbricks/` | Its own PyTorch/MuJoCo training/demo dependencies. | Preview research stack, not wired into SONIC deploy by default. |

### 2.2 Asset and source-of-truth boundaries

Some files are code interfaces; others are model/data artifacts. For extension work, these boundaries matter:

| Boundary | Source of truth | Why it matters |
|---|---|---|
| Decoupled WBC policy shape | Balance/Walk ONNX inputs/outputs and sim2mujoco observation builder. | These policies are 15-action lower-body policies; they cannot replace SONIC's 29-action decoder. |
| SONIC policy observation layout | `gear_sonic_deploy/policy/release/observation_config.yaml`. | Any custom ONNX must match this config, or ship a matching replacement config. |
| SONIC training composition | Hydra configs under `gear_sonic/config/`. | Most research changes should start as config overrides before code forks. |
| Deploy model format | `model_encoder.onnx`, `model_decoder.onnx`, `planner_sonic.onnx`. | C++ deploy converts ONNX to TensorRT engines and validates names/shapes. |
| Robot command interface | Unitree SDK2 DDS topics and C++ command writer. | Real robot control is low-level joint command at 500 Hz. |
| External integration | ZMQ protocols and manager input modes. | This is the cleanest place for a new planner, retargeter, or VLA action source. |
| Local evidence | `rollouts/`, `policy_review_results.md`, `DEPLOY_READINESS.md`. | These validate local assets and dry-run paths, not full hardware readiness. |

## 3. Baseline 1: Decoupled WBC

### 3.1 What it is

`decoupled_wbc/` is the older GR00T N1.5/N1.6 controller family. It is a decoupled design:

- lower body: learned ONNX policies,
- upper body: interpolation and IK,
- runtime: Python, ONNX Runtime, MuJoCo or Unitree SDK based paths.

The lower-body policy observation is a compact history-based vector. In the local MuJoCo recorder, each frame builds an 86-dimensional observation and stacks 6 frames into a 516-dimensional ONNX input.

### 3.2 Provided baseline policies

Released ONNX files:

```text
decoupled_wbc/sim2mujoco/resources/robots/g1/policy/
  GR00T-WholeBodyControl-Balance.onnx
  GR00T-WholeBodyControl-Walk.onnx
```

Baseline signatures from local review:

| Model | Input | Output | Purpose |
|---|---|---|---|
| Balance | `[batch, 516]` | `[batch, 15]` | Standing and balance when locomotion command is near zero. |
| Walk | `[batch, 516]` | `[batch, 15]` | Lower-body walking command tracking. |

These models are good smoke-test baselines because they do not require TensorRT, C++ deploy, Isaac Lab, or real hardware.

### 3.3 Provided implementation pieces

Key implementation areas:

| Path | Role |
|---|---|
| `decoupled_wbc/control/policy/` | Python policy wrappers and interpolation policies. |
| `decoupled_wbc/control/robot_model/` | Robot model and joint grouping logic. |
| `decoupled_wbc/control/teleop/` | GUI/CLI teleop, IK solvers, input device handlers. |
| `decoupled_wbc/control/envs/g1/` | G1 sim and SDK environment support. |
| `decoupled_wbc/scripts/deploy_g1.py` | Tmux orchestrator for the decoupled WBC real/sim stack. |
| `decoupled_wbc/sim2mujoco/` | Lightweight MuJoCo ONNX evaluation path. |
| `decoupled_wbc/tests/` | Pytest suite around policy, IK, data, and sim paths. |

### 3.4 Local additions around this baseline

Local untracked files add a headless verification path:

```text
decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc_headless.yaml
decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py
```

This recorder:

- avoids interactive viewer dependencies,
- runs Balance or Walk policy in MuJoCo,
- records MP4 videos with offscreen rendering,
- confirms the released policies can execute locally.

Generated local evidence:

```text
rollouts/wbc_balance.mp4
rollouts/wbc_walk.mp4
rollouts/review_wbc_balance.mp4
rollouts/review_wbc_walk.mp4
```

### 3.5 How to use it as a baseline

Use Decoupled WBC for:

- fast environment sanity checks,
- verifying MuJoCo rendering and G1 assets,
- simple walking/balance comparison,
- regression evidence before touching SONIC or real hardware.

Do not treat Decoupled WBC as interchangeable with SONIC. It has different action dimensions, observation schema, runtime assumptions, and controller architecture.

## 4. Baseline 2: GEAR-SONIC

### 4.1 What it is

GEAR-SONIC is the main modern whole-body control baseline in this repo. It is a motion-tracking foundation model for humanoid behavior. It learns to track and generalize from large-scale human motion data, then deploys as a unified 29-DoF G1 controller.

SONIC has two repo halves:

```text
gear_sonic/         # research, training, eval, data, inference utilities
gear_sonic_deploy/  # real-time C++ deployment runtime
```

The key design idea is a shared latent/token space. Multiple control modalities are converted into a common representation, then decoded into full-body joint targets.

### 4.2 Universal Token research baseline

SONIC uses modality-specific encoders feeding a finite scalar quantized latent bottleneck:

```text
G1 proprio/history encoder
SMPL motion encoder
VR/teleop encoder
        |
        v
shared token bottleneck
        |
        v
decoder -> 29-DoF G1 action
```

Implementation landmark:

```text
gear_sonic/trl/modules/universal_token_modules.py
```

Training configuration landmark:

```text
gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml
```

The training setup uses auxiliary alignment losses so G1, SMPL, and teleop encoders learn compatible latent representations. The repo contains configs and loss implementations for:

- G1 reconstruction,
- G1 to SMPL latent alignment,
- G1 to teleop latent alignment,
- teleop to SMPL consistency,
- cycle consistency through re-encoded decoded outputs.

Important paths:

```text
gear_sonic/config/aux_losses/
gear_sonic/trl/losses/token_losses.py
gear_sonic/trl/trainer/ppo_trainer_aux_loss.py
```

### 4.3 Training baseline

Training lives under `gear_sonic/` and requires Isaac Lab. It is not intended to run inside the lightweight MuJoCo environment.

Provided training pieces:

| Path | Role |
|---|---|
| `gear_sonic/train_agent_trl.py` | Main training entrypoint. |
| `gear_sonic/eval_agent_trl.py` | Evaluation and ONNX export entrypoint. |
| `gear_sonic/eval_exp.py` | Continuous evaluator over experiment directories. |
| `gear_sonic/config/` | Hydra config tree for algorithms, envs, rewards, observations, actors, trainers. |
| `gear_sonic/envs/manager_env/` | Isaac Lab manager-based environment implementation. |
| `gear_sonic/envs/wrapper/` | Runtime wrapper exposing motion libs, latent residuals, markers. |
| `gear_sonic/trl/` | Vendored/forked TRL trainer, modules, losses, callbacks. |
| `gear_sonic/data_process/` | SOMA/Bones-SEED conversion and filtering scripts. |

Training baseline command shape:

```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

Important constraints:

- Isaac Lab Python environment is required.
- Python 3.11.x is required for training.
- Full-scale recipes expect substantial GPU resources.
- The repo uses a local `gear_sonic/trl/` fork; entry scripts intentionally manage `sys.path`.

### 4.4 Isaac Lab environment baseline

SONIC training uses Isaac Lab manager terms:

| Component | Path | Purpose |
|---|---|---|
| Commands | `gear_sonic/envs/manager_env/mdp/commands.py` | Motion library sampling, tracking windows, encoder modes. |
| Observations | `gear_sonic/envs/manager_env/mdp/observations.py` | Policy, critic, tokenizer observation terms. |
| Rewards | `gear_sonic/envs/manager_env/mdp/rewards.py` | Tracking, anchor, action-rate, joint-limit, anti-shake terms. |
| Terminations | `gear_sonic/envs/manager_env/mdp/terminations.py` | Failure thresholds and adaptive reset behavior. |
| Events | `gear_sonic/envs/manager_env/mdp/events.py` | Randomization and perturbation curriculum. |
| Robots | `gear_sonic/envs/manager_env/robots/` | G1 and H2 robot definitions. |

This matters for extension work because the environment is not a monolithic class. Most training behavior is composed through Hydra groups. A new experiment should usually be a config override, not a code fork.

### 4.5 Provided SONIC deploy baseline assets

Local release assets present in this checkout:

```text
gear_sonic_deploy/policy/release/model_encoder.onnx
gear_sonic_deploy/policy/release/model_decoder.onnx
gear_sonic_deploy/policy/release/observation_config.yaml
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
gear_sonic_deploy/reference/example/
```

Local signature checks from `policy_review_results.md`:

| Asset | Signature / role |
|---|---|
| `model_encoder.onnx` | `obs_dict [1, 1762] -> encoded_tokens [1, 64]` |
| `model_decoder.onnx` | `obs_dict [1, 994] -> action [1, 29]` |
| `planner_sonic.onnx` | 11 planner inputs -> `mujoco_qpos [1,64,36]`, `num_pred_frames [1]` |
| `observation_config.yaml` | Runtime contract for observation names, order, dimensions, and encoder subsets. |

Reference motion clips are provided under:

```text
gear_sonic_deploy/reference/example/
```

Examples include walking, macarena, squat, kick, lunge, one-leg jump, and dance clips, with mirrored variants.

### 4.6 Low-latency SONIC baseline

Latest upstream adds a low-latency SONIC variant. The repo includes support in:

```text
download_from_hf.py
docs/source/getting_started/download_models.md
docs/source/tutorials/vla_inference.md
gear_sonic_deploy/policy/release/observation_config_low_latency.yaml
```

The actual low-latency ONNX model files are not shown in the current local `gear_sonic_deploy/policy/low_latency/` path. They are downloaded with:

```bash
python download_from_hf.py --low-latency
```

Expected layout after download:

```text
gear_sonic_deploy/policy/low_latency/
  model_encoder.onnx
  model_decoder.onnx
  observation_config.yaml
```

This is relevant for our design if latency is a first-order requirement. Any new controller variant must keep its ONNX files and observation config matched.

## 5. Baseline 3: SONIC C++ deployment runtime

### 5.1 What it is

`gear_sonic_deploy/` is the production-like deploy stack. It is the path intended for real G1 hardware.

Main executable:

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
```

Wrapped launcher:

```text
gear_sonic_deploy/deploy.sh
```

Build system:

```text
gear_sonic_deploy/.justfile
gear_sonic_deploy/CMakeLists.txt
gear_sonic_deploy/cmake/
```

### 5.2 Critical architectural baseline

The deploy stack does not use Unitree's high-level locomotion service as the walking controller. It takes low-level ownership:

```text
target generator -> SONIC policy -> joint targets -> PD-like low-level commands -> Unitree DDS rt/lowcmd
```

The repo itself owns the locomotion/motion-tracking behavior. Unitree SDK2 is used for transport, robot state, and command publication.

### 5.3 Runtime layers

| Layer | Role | Main paths |
|---|---|---|
| Input | Keyboard, gamepad, manager, ZMQ, ZMQ manager, ROS2. | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/` |
| Motion target | Reference clip, streamed motion, or planner-generated motion. | `motion_data_reader.hpp`, planner headers |
| Inference | TensorRT encoder and decoder inference from ONNX. | `encoder.hpp`, `control_policy.hpp`, `src/TRTInference/` |
| Robot I/O | Unitree DDS low-state subscription, low-command publication, Dex3 hands. | main C++ runtime, Unitree SDK2 thirdparty |
| Output/debug | ZMQ and ROS2 state/debug outputs. | `include/output_interface/` |

Threading baseline:

| Thread | Rate | Responsibility |
|---|---:|---|
| Input | 100 Hz | Poll active input source and update control/planner commands. |
| Control | 50 Hz | Build observations, run encoder/policy, update command state. |
| Planner | 10 Hz | Generate or refresh planner motion. |
| Command writer | 500 Hz | Publish latest `LowCmd` to robot. |

State machine:

```text
INIT -> WAIT_FOR_CONTROL -> CONTROL
```

### 5.4 Motion target unification

SONIC deploy unifies three target-generation routes:

| Route | Target source | Use case |
|---|---|---|
| Reference motion | CSV clips in `reference/example/` | Demos, validation, fixed clips. |
| Streamed motion | ZMQ/ROS2 incoming motion | Teleop, external retargeting, custom target publisher. |
| Planner motion | `planner_sonic.onnx` | Runtime locomotion from velocity, mode, height, waypoint commands. |

All three end up as motion sequences that feed the same observation builder and policy decoder. This is the main reusable architecture for our design.

### 5.5 Observation config contract

The observation config is a hard contract between training/export and deployment:

```text
gear_sonic_deploy/policy/release/observation_config.yaml
```

It specifies:

- enabled observation terms,
- runtime order,
- dimensions,
- encoder token dimension,
- encoder-specific observation subsets.

The C++ runtime validates observation dimensions against ONNX input shapes. This is one of the most important constraints for any custom model or policy variant. If we alter observations, we must export matching ONNX and ship a matching config.

### 5.6 ZMQ protocols

ZMQ is a major integration point. Current stack supports multiple streaming protocols:

- joint-based streaming,
- SMPL-based streaming,
- joint plus SMPL streaming,
- token-only streaming,
- manager command/planner topics.

Important compatibility detail:

```text
ZMQ header size = 1280 bytes
```

Any external client must match the repo's current wire format. Local script `scripts_local/stream_planner_waypoint_zmq.py` explicitly uses this size.

### 5.7 Real robot interfaces

Production DDS topics:

| Topic | Direction | Purpose |
|---|---|---|
| `rt/lowstate` | subscribe | Low-level robot state. |
| `rt/secondary_imu` | subscribe | Torso IMU. |
| `rt/lowcmd` | publish | Low-level motor commands. |
| `rt/dex3/left/cmd` | publish | Left hand commands. |
| `rt/dex3/right/cmd` | publish | Right hand commands. |
| `rt/dex3/left/state` | subscribe | Left hand state. |
| `rt/dex3/right/state` | subscribe | Right hand state. |

Before control, the runtime uses `MotionSwitcherClient` to release conflicting Unitree services.

## 6. Baseline 4: Kinematic planner

### 6.1 What it is

The SONIC planner is a separate ONNX model:

```text
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

It predicts future full-body MuJoCo `qpos` trajectories from:

- recent motion context,
- target velocity,
- planner mode,
- movement direction,
- facing direction,
- random seed,
- optional specific target positions/headings,
- allowed token counts,
- height.

The planner is not the low-level controller. It generates target motion, and the SONIC policy tracks that target.

### 6.2 Local planner findings

Local scripts rendered planner outputs without C++/TensorRT:

```text
scripts_local/replay_planner.py
rollouts/planner_*.mp4
```

Observed from `rollouts/README.md`:

- mode 0 with velocity only can idle from neutral context,
- mode 0 moves when supplied a waypoint target,
- mode 1 and mode 2 generate forward locomotion from neutral seed,
- waypoint `target_pos=[3,0,0]` produced near-3m root displacement over the longer rollout.

This is useful for our design because it shows waypoint-conditioned planner behavior can be used as an external high-level command route.

### 6.3 Local waypoint streamer

Local file:

```text
scripts_local/stream_planner_waypoint_zmq.py
```

This script:

- runs planner ONNX in Python using ONNX Runtime CPU provider,
- chains planner chunks,
- converts qpos to joint positions, joint velocities, body quaternions, and frame indices,
- streams a `pose` topic over ZMQ to C++ deploy.

It is a prototype for a design-beyond-repo path: external planning process -> ZMQ pose stream -> unmodified SONIC deploy runtime.

## 7. Baseline 5: VLA integration

### 7.1 What it provides

The repo now includes an end-to-end VLA inference path where Isaac-GR00T supplies higher-level policy actions and SONIC executes whole-body control.

Important files:

| Path | Role |
|---|---|
| `gear_sonic/scripts/run_vla_inference.py` | Python inference client: camera/state input, VLA server call, ZMQ action output. |
| `gear_sonic/scripts/launch_inference.py` | Tmux launcher for C++ deploy, VLA inference, keyboard publisher, data exporter. |
| `gear_sonic/utils/inference/` | VLA observation splitting, timing, initial pose token, helper logic. |
| `install_scripts/install_inference.sh` | Creates inference environment. |
| `docs/source/tutorials/vla_inference.md` | Full VLA inference guide. |
| `docs/source/tutorials/vla_workflow.md` | Full collect/fine-tune/deploy workflow. |

### 7.2 VLA action space

The SONIC embodiment uses a 78-dimensional action space:

```text
64-dim motion token + 7 left hand joints + 7 right hand joints
```

This matters for our design. If our system operates above SONIC, it may produce latent tokens and hands rather than raw joint-space references.

### 7.3 Runtime topology

The VLA path typically consists of:

```text
Isaac-GR00T PolicyServer
        |
        v
run_vla_inference.py
        |
        v
C++ deploy over ZMQ manager
        |
        v
G1 low-level control
```

Camera server and optional data exporter run alongside this.

### 7.4 What is not provided

The repo provides the integration path, but not our task-specific VLA design. To use this route for a new design, we need:

- a fine-tuned Isaac-GR00T model for the target task,
- correct embodiment tag and action formatting,
- camera setup,
- prompt/task specification,
- data collection and processing pipeline,
- sim and hardware validation.

## 8. Baseline 6: Teleoperation and data collection

### 8.1 VR and Isaac Teleop

The repo supports VR whole-body teleoperation through PICO and includes Isaac Teleop / CloudXR support in the current checkout.

Important paths:

```text
gear_sonic/scripts/pico_manager_thread_server.py
gear_sonic/utils/teleop/
gear_sonic/utils/teleop/isaac_teleop_client.py
gear_sonic/utils/teleop/input_readers.py
install_scripts/install_pico.sh
docs/source/tutorials/vr_wholebody_teleop.md
docs/source/tutorials/isaac_teleop_publisher_setup.md
```

The relevant support files include:

- in-process CloudXR / DeviceIO Isaac Teleop client,
- input reader utilities,
- tests for input readers,
- Docker/ROS updates for Thor/JetPack setup,
- low-latency observation support in C++.

### 8.2 Data collection

Important paths:

```text
gear_sonic/scripts/launch_data_collection.py
gear_sonic/scripts/run_data_exporter.py
gear_sonic/scripts/process_dataset.py
gear_sonic/utils/data_collection/
docs/source/tutorials/data_collection.md
```

Data sources are primarily ZMQ based:

- robot state from C++ deploy output,
- pose/planner messages from teleop manager,
- camera data from camera server,
- keyboard recording controls.

This path provides a baseline for collecting VLA/teleop episodes, but it does not provide our new dataset or task policy.

## 9. Baseline 7: MotionBricks

### 9.1 What it is

`motionbricks/` is a preview generative motion system. It complements SONIC but is not currently the deployed controller.

Main idea:

- VQ-VAE tokenizes motion,
- pose transformer generates pose tokens,
- root transformer predicts continuous root trajectories,
- MuJoCo demo streams generated G1 motion interactively.

Important paths:

| Path | Role |
|---|---|
| `motionbricks/README.md` | Setup, demo, and training instructions. |
| `motionbricks/scripts/interactive_demo_g1.py` | Interactive MuJoCo demo. |
| `motionbricks/scripts/train_vqvae.py` | VQ-VAE training. |
| `motionbricks/scripts/train_pose.py` | Pose model training. |
| `motionbricks/scripts/train_root.py` | Root model training. |
| `motionbricks/docs/motion_representation.md` | Motion feature representation. |
| `motionbricks/motionbricks/` | Model, motionlib, geometry, training code. |

### 9.2 What it provides as baseline

MotionBricks gives us a different design direction:

```text
interactive/generative motion authoring -> generated motion trajectory -> possible SONIC target stream
```

It is currently useful as:

- a generative-motion research reference,
- a possible target-motion generator,
- a separate MuJoCo interactive demo,
- a codebase for motion representations and G1 skeleton handling.

### 9.3 What it does not provide yet

MotionBricks is not yet integrated into the SONIC deployment loop. It does not directly replace:

- SONIC policy decoder,
- C++ deploy control loop,
- Unitree low-level runtime,
- Isaac Lab training stack.

If our design uses MotionBricks, we likely need an adapter from generated motion to SONIC-compatible target streams or a new training route.

## 10. Local implementation and validation artifacts

The current local workspace contains additional implementation/reporting artifacts beyond upstream. These are important because they show the current project direction.

### 10.1 Local documents

| File | Meaning |
|---|---|
| `DEPLOY_READINESS.md` | Local audit of what is ready and what blocks C++ deploy. |
| `policy_review_results.md` | Evidence from model loading, ONNX signatures, and generated videos. |
| `architecture-overview.md` | Research/architecture map. |
| `deploy-design.md` | Deploy runtime deep dive. |
| `deploy.md` | Practical deployment scenario guide. |
| `guide.md` | Policy inventory and recommended test flow. |
| `rollouts/README.md` | Index and interpretation of local MP4 outputs. |

### 10.2 Local scripts

| File | Purpose |
|---|---|
| `decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py` | Headless Decoupled WBC MuJoCo recorder. |
| `scripts_local/replay_motion_clip.py` | Kinematic rendering of SONIC reference clips. |
| `scripts_local/replay_planner.py` | Planner ONNX rollout and MP4 render. |
| `scripts_local/stream_planner_waypoint_zmq.py` | Planner rollout streamed over ZMQ to deploy runtime. |
| `scripts_local/start_training_tmux.sh` | Training session tmux helper and Isaac Lab reminders. |

### 10.3 Local rollout evidence

Representative local videos:

```text
rollouts/wbc_balance.mp4
rollouts/wbc_walk.mp4
rollouts/ref_*.mp4
rollouts/planner_*.mp4
rollouts/review_*.mp4
```

These validate:

- Decoupled WBC policies can run locally in MuJoCo.
- SONIC reference clips render correctly as kinematic targets.
- SONIC planner ONNX can be invoked from Python.
- Waypoint and planner modes produce inspectable target trajectories.

They do not yet validate:

- full SONIC closed-loop control in C++ sim2sim,
- TensorRT conversion,
- real G1 hardware behavior,
- a custom task policy.

### 10.4 Local workspace status

As of this report, several useful files are local/untracked rather than known committed upstream files:

```text
AGENTS.md
CLAUDE.md
architecture-overview.md
deploy-design.md
deploy.md
guide.md
repo-baseline-implementation-report.md
decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc_headless.yaml
decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py
rollouts/
scripts_local/
```

Treat these as project working artifacts. They are valuable for implementation planning, but before a PR they should be sorted into one of three buckets:

- keep and commit because they are part of the design baseline,
- move into formal docs/tests/scripts with review,
- or leave local because they are only operator notes or generated evidence.

## 11. Current local asset status

Observed present in this checkout:

| Asset | Status |
|---|---|
| Decoupled WBC Balance/Walk ONNX | Present. |
| SONIC release encoder/decoder ONNX | Present. |
| SONIC planner ONNX | Present. |
| SONIC reference clips | Present. |
| SONIC PyTorch checkpoint `sonic_release/last.pt` | Present locally. |
| Sample robot/SMPL/SOMA data | Present locally. |
| Local rollout videos | Present. |
| Low-latency SONIC download support | Present in code/docs. |
| Low-latency ONNX model directory | Not observed locally; download with `python download_from_hf.py --low-latency`. |
| MotionBricks code/assets | Present. |
| MotionBricks pretrained checkpoints | LFS-managed and opt-in; verify with file size before relying on them. |

## 12. Current local readiness status

Based on `DEPLOY_READINESS.md` and `policy_review_results.md`, the current state is:

Ready:

- local repo structure and documentation,
- Git LFS release assets,
- Decoupled WBC MuJoCo smoke-test path,
- SONIC ONNX signature inspection,
- planner/reference target rendering,
- sample data and release checkpoint for training smoke tests.

Blocked or not yet validated:

- C++ SONIC sim2sim deploy,
- real G1 deploy,
- TensorRT environment,
- full C++ toolchain setup,
- ONNX Runtime C/C++ CMake package visibility,
- Isaac Lab training environment in the default shell,
- full closed-loop SONIC policy rollout without the C++ deploy runtime.

Specific previously identified deploy blockers:

- `TensorRT_ROOT` unset,
- `just` missing,
- `ninja` missing,
- `clang` missing,
- ONNX Runtime C/C++ package not visible via `onnxruntime_ROOT`,
- CUDA toolkit visible but needs compatibility with selected TensorRT.

This means the current repo is strong for baseline inspection and dry-run evidence, but not yet complete for deploying SONIC on real hardware from this machine.

## 13. What should be considered baseline

For our design work, baseline means "do not reimplement this unless we have a clear reason."

### 13.1 Baseline models

- Decoupled WBC Balance/Walk ONNX for simple MuJoCo policy sanity checks.
- SONIC release encoder/decoder ONNX for 29-DoF G1 whole-body control.
- SONIC planner ONNX for target-motion generation.
- SONIC PyTorch checkpoint for evaluation, fine-tuning, and ONNX export.
- Low-latency SONIC variant as an optional latency-focused deployment baseline.

### 13.2 Baseline runtime abstractions

- `MotionSequence` abstraction for reference, streamed, and planner targets.
- Observation config as deployment contract.
- ZMQ as external integration mechanism.
- TensorRT C++ runtime for real G1 control.
- Unitree SDK2 DDS low-level command path.
- 50 Hz policy loop and 500 Hz command publication.
- Manager input mode for switching control sources.

### 13.3 Baseline workflows

- `check_environment.py` before sim/training/deploy.
- Decoupled WBC MuJoCo smoke test.
- SONIC reference/planner target visualization.
- SONIC C++ deploy in MuJoCo sim before hardware.
- Isaac Lab evaluation before ONNX export.
- ONNX export through `eval_agent_trl.py`.
- Real hardware only after sim2sim stability.

### 13.4 Baseline documentation

Use these as canonical references:

```text
README.md
docs/source/getting_started/
docs/source/tutorials/
docs/source/user_guide/
docs/source/references/
architecture-overview.md
deploy-design.md
DEPLOY_READINESS.md
policy_review_results.md
```

## 14. What is not provided as baseline

The repo does not currently provide:

- our new design objective or architecture,
- a custom planner beyond released `planner_sonic.onnx`,
- a trained custom SONIC checkpoint for our target behavior,
- a trained task-specific Isaac-GR00T VLA model,
- full real-robot validation logs from this local checkout,
- hardware safety certification,
- a Python-only full closed-loop SONIC replacement for the C++ deploy stack,
- a complete local TensorRT/C++ setup,
- an integrated MotionBricks-to-SONIC deploy pipeline,
- a new dataset for our target task.

These are the likely areas where our implementation begins.

## 15. Main extension points for our design

### 15.1 External target generator

We can build a process that emits SONIC-compatible motion targets over ZMQ. This is the least invasive path.

Reuse:

- C++ deploy runtime,
- SONIC policy,
- observation config,
- low-level robot I/O,
- ZMQ protocol,
- `MotionSequence` tracking path.

Build:

- our planner/generator,
- conversion to joint/SMPL/token protocol,
- timing and buffering,
- safety gates.

Local prototype:

```text
scripts_local/stream_planner_waypoint_zmq.py
```

### 15.2 External latent-token policy

We can produce token-only protocol outputs. This bypasses motion decoding and feeds the decoder token state directly.

Reuse:

- SONIC decoder,
- C++ control loop,
- low-level G1 deployment.

Build:

- model that predicts 64-dim token state,
- hand joint outputs if needed,
- ZMQ protocol v4 compatibility,
- checkpoint-specific initial token handling.

This is aligned with the VLA path.

### 15.3 New high-level planner

We can replace or augment `planner_sonic.onnx` while keeping SONIC as tracker.

Reuse:

- target-motion path,
- planner integration interface,
- deployed policy.

Build:

- new planner model,
- ONNX export,
- planner input/output adapter,
- config and runtime integration.

Risk:

- generated motions must stay inside what SONIC can track.

### 15.4 SONIC fine-tuning

We can fine-tune the SONIC policy itself.

Reuse:

- Isaac Lab manager env,
- `sonic_release` config,
- PPO trainer,
- data pipeline,
- evaluation/export path.

Build:

- task/motion dataset,
- reward/termination changes if needed,
- config override,
- evaluation metrics,
- ONNX export and deploy package.

Risk:

- training is expensive,
- sim-to-real validation is required,
- observation config must match export.

### 15.5 VLA-over-SONIC

We can build a higher-level VLA policy that emits SONIC token actions and hand commands.

Reuse:

- VLA inference client,
- C++ deploy ZMQ manager,
- camera server,
- data exporter,
- SONIC low-level controller.

Build:

- task data collection,
- Isaac-GR00T fine-tune,
- prompt/task interface,
- camera calibration/latency validation,
- initial pose and safety workflow.

This is the most task-general path but has the most moving pieces.

### 15.6 MotionBricks as generator

We can use MotionBricks to generate target motions and stream them to SONIC.

Reuse:

- MotionBricks generative model,
- G1 motion representation,
- SONIC target stream path.

Build:

- adapter from MotionBricks output to SONIC deploy motion fields,
- timing/resampling,
- sim validation,
- possibly training or conditioning changes.

This is a research-heavy route, not currently wired by upstream.

## 16. Recommended implementation roadmap beyond this repo

### Phase 0: Freeze and verify baseline

Goal: ensure the local checkout can reproduce known behavior.

Actions:

1. Run `python check_environment.py --sim`.
2. Regenerate Decoupled WBC videos.
3. Regenerate SONIC reference and planner videos.
4. Confirm ONNX signatures.
5. Keep `rollouts/README.md` updated.

Exit criteria:

- Balance and Walk MP4s look correct.
- SONIC planner/reference replays render.
- No LFS pointer issues.

### Phase 1: Complete C++ deploy environment

Goal: make SONIC C++ sim mode work.

Actions:

1. Install/configure TensorRT.
2. Install `just`, `ninja`, `clang`.
3. Install ONNX Runtime C/C++ package and set `onnxruntime_ROOT` if needed.
4. Source `gear_sonic_deploy/local_env.sh` and `scripts/setup_env.sh`.
5. Build with `just build`.
6. Run C++ unit tests if available.
7. Run `deploy.sh --input-type keyboard sim` with MuJoCo bridge.

Exit criteria:

- C++ binary builds.
- TensorRT engines compile/cache.
- MuJoCo sim2sim can track reference/planner motions.

### Phase 2: Choose design integration point

Decision:

- Are we building a target generator?
- A token policy?
- A planner replacement?
- A SONIC fine-tune?
- A VLA layer?
- A MotionBricks-to-SONIC adapter?

Recommended initial path:

```text
external target generator over ZMQ -> existing SONIC deploy
```

Reason:

- least invasive,
- preserves upstream deploy safety/runtime,
- easy to validate in sim,
- compatible with real robot once deploy is stable.

### Phase 3: Implement minimal design prototype

Build only the part not already provided by the repo:

- generator/model,
- adapter to repo protocol,
- logging,
- validation scripts,
- safety bounds,
- repeatable launch command.

Avoid modifying:

- low-level Unitree command writer,
- observation ordering,
- action scaling,
- policy internals,
- deploy threading,
- unless the design explicitly requires it.

### Phase 4: Validation gates

Gate sequence:

1. Offline output sanity checks.
2. Kinematic MuJoCo rendering.
3. C++ sim2sim with existing SONIC policy.
4. Stress test timing and stale-data behavior.
5. Hardware dry run with robot supported/guarded.
6. Real task rollout.
7. Data/log review.

Do not skip sim2sim.

### Phase 5: Training or fine-tuning if needed

Only train once the target-generation baseline is insufficient.

Training work should start from:

```text
gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml
sonic_release/last.pt
sample_data/
```

Then progress to full data.

## 17. Key technical risks

### 17.1 Observation mismatch

Any custom ONNX policy must match its `observation_config.yaml`. If dimensions/order drift, deployment will fail or behave incorrectly.

### 17.2 Token mismatch

The 64-dim latent token is checkpoint-specific. Initial pose tokens and token-only VLA actions should not be reused blindly across checkpoints.

### 17.3 Planner target out of distribution

SONIC tracks target motion. If a generator produces physically implausible or out-of-distribution targets, the policy may fail even if the target looks acceptable kinematically.

### 17.4 ZMQ version drift

Header size and field layouts matter. Current stack expects a 1280-byte header. External clients must match the C++ parser.

### 17.5 Build/runtime environment split

The repo intentionally uses separate environments:

- `.venv_sim` for MuJoCo,
- Isaac Lab Python for training,
- `.venv_teleop` for VR teleop,
- `.venv_data_collection` for data collection,
- `.venv_inference` for VLA inference,
- C++ environment for deploy.

Mixing these will create false failures.

### 17.6 Hardware safety

Real robot deployment uses low-level joint commands at 500 Hz. Any new design must be validated in sim and constrained before hardware.

## 18. File landmarks for implementation

If extending the repo, start from these files:

| Need | Start here |
|---|---|
| Understand repo architecture | `architecture-overview.md` |
| Understand C++ deploy | `deploy-design.md` |
| Deploy command flow | `gear_sonic_deploy/deploy.sh` |
| Main real-time runtime | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` |
| SONIC release experiment config | `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml` |
| Universal token implementation | `gear_sonic/trl/modules/universal_token_modules.py` |
| PPO with aux losses | `gear_sonic/trl/trainer/ppo_trainer_aux_loss.py` |
| Motion command term | `gear_sonic/envs/manager_env/mdp/commands.py` |
| Observation terms | `gear_sonic/envs/manager_env/mdp/observations.py` |
| ONNX export/eval | `gear_sonic/eval_agent_trl.py` |
| VLA inference | `gear_sonic/scripts/run_vla_inference.py` |
| VLA tmux launcher | `gear_sonic/scripts/launch_inference.py` |
| Teleop manager | `gear_sonic/scripts/pico_manager_thread_server.py` |
| ZMQ message builders | `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py` |
| Decoupled WBC recorder | `decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py` |
| Local planner replay | `scripts_local/replay_planner.py` |
| Local waypoint streamer | `scripts_local/stream_planner_waypoint_zmq.py` |
| Environment audit | `check_environment.py` |

## 19. Practical baseline commands

### 19.1 Check assets/environment

```bash
python check_environment.py --sim
python check_environment.py --deploy
python check_environment.py --training
```

Use each mode in the correct environment.

### 19.2 Decoupled WBC MuJoCo smoke test

```bash
source .venv_sim/bin/activate
unset PYTHONPATH
python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd balance \
    --out rollouts/wbc_balance.mp4
python decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_record.py \
    --config g1_gear_wbc_headless.yaml \
    --cmd walk \
    --out rollouts/wbc_walk.mp4
```

### 19.3 SONIC reference replay

```bash
source .venv_sim/bin/activate
unset PYTHONPATH
python scripts_local/replay_motion_clip.py \
    --clip gear_sonic_deploy/reference/example/squat_001__A359 \
    --out rollouts/ref_squat_001__A359.mp4
```

### 19.4 SONIC planner replay

```bash
source .venv_sim/bin/activate
unset PYTHONPATH
python scripts_local/replay_planner.py --chunks 8
```

### 19.5 Planner waypoint ZMQ dry run

```bash
source .venv_sim/bin/activate
unset PYTHONPATH
python scripts_local/stream_planner_waypoint_zmq.py \
    --waypoint 3.0 0.0 0.0 \
    --target-vel 1.0 \
    --mode 0 \
    --chunks 1 \
    --dry-run
```

### 19.6 C++ deploy target once toolchain is ready

```bash
cd gear_sonic_deploy
source local_env.sh
source scripts/setup_env.sh
just build
./deploy.sh --input-type keyboard sim
```

Only after sim:

```bash
./deploy.sh --input-type keyboard real
```

## 20. Bottom line

This repository already gives us a strong baseline for G1 whole-body control:

- released Decoupled WBC policies for quick MuJoCo validation,
- released SONIC encoder/decoder/planner assets,
- a full Isaac Lab training/evaluation stack,
- a real-time C++ TensorRT deployment stack,
- ZMQ/teleop/VLA integration surfaces,
- sample data and local rollout evidence,
- MotionBricks as a future generative-motion direction.

What it does not yet give us is our own design. The repo should be treated as the control and deployment substrate. Our implementation should first decide where to attach:

```text
target motion, latent tokens, planner model, SONIC fine-tune, VLA policy, or MotionBricks adapter
```

The recommended first design path is an external target-motion or waypoint generator that streams to the existing SONIC deploy runtime over ZMQ. That lets us build beyond the repo while preserving the strongest baseline: the released SONIC policy, observation contract, real-time deploy loop, and low-level G1 safety structure.
