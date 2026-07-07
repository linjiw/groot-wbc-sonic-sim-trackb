# GR00T-WholeBodyControl — Architecture & Research Overview

A detailed walkthrough of what the **GR00T-WholeBodyControl** monorepo contains, what the **GEAR-SONIC** subproject contributes, and how the training stack plugs into NVIDIA **Isaac Lab**.

> Scope: this doc is meant to be read by an engineer/researcher joining the project. It is grounded in actual file paths, classes, and config keys (linked inline) so you can jump from concept → code. It is current as of the merge that landed on `main` 2026-05-23 (post-SONIC release + MotionBricks + VLA pipeline).

---

## 1. Bird's-eye view of the monorepo

`GR00T-WholeBodyControl` hosts **three controller families** for humanoid robots (primarily the Unitree G1, with some H2 support) plus the shared infrastructure that lets them be trained, simulated, deployed, and teleoperated.

```
GR00T-WholeBodyControl/
├── decoupled_wbc/         # RL-lower + IK-upper controller used in GR00T N1.5 / N1.6
├── gear_sonic/            # SONIC training stack (PPO on Isaac Lab) ─┐
├── gear_sonic_deploy/     # SONIC C++ runtime (TensorRT on robot) ─┘ same model
├── motionbricks/          # NEW (2026-04-27): real-time latent motion generative model
├── docs/                  # Sphinx docs (deployment_code, training_code, conventions, …)
├── install_scripts/       # uv-driven venv installers per use case
├── external_dependencies/ # Unitree SDK2 python bindings, XRoboToolkit C++ service
├── systemd/               # composed_camera_server.service (boot-time camera ZMQ)
├── download_from_hf.py    # HF Hub fetcher for checkpoints + SMPL motion data
├── check_environment.py   # Pre-flight: Python version, LFS, CUDA, IsaacLab, TRT
└── deploy-design.md       # The single best deep-dive on the C++ runtime
```

### The three controller families at a glance

| Controller | Where it lives | Trained on | Deployed via | Used for |
|---|---|---|---|---|
| **Decoupled WBC** | `decoupled_wbc/` | (offline, ONNX shipped) | Python + ONNX Runtime + MuJoCo / Unitree SDK2 | GR00T N1.5/N1.6 — locomotion (RL) + arms (IK) |
| **GEAR-SONIC** | `gear_sonic/` + `gear_sonic_deploy/` | **PPO on Isaac Lab** (4096 envs) | **C++ + TensorRT** on Jetson, 500 Hz | Generalist motion-tracking foundation model |
| **MotionBricks** | `motionbricks/` | PyTorch Lightning, **MuJoCo only** | MuJoCo interactive demo | Latent generative motion (VQ-VAE + transformer) |

Only **GEAR-SONIC** trains inside Isaac Lab today. Decoupled WBC consumes pre-trained ONNX; MotionBricks is MuJoCo-only and is on a roadmap to be folded into the SONIC pipeline (see `motionbricks/README.md`).

---

## 2. GEAR-SONIC — the research story

SONIC is the headline release. It is a **multi-modal motion-tracking foundation model** that unifies three control paradigms behind a shared discrete-token bottleneck so one policy can be driven by robot state, human motion capture, or VR teleop interchangeably.

### 2.1 Core research idea — the "Universal Token"

Implemented at [`gear_sonic/trl/modules/universal_token_modules.py:33-99`](gear_sonic/trl/modules/universal_token_modules.py).

```
                ┌── encoder_g1 ────┐
proprioception ─┤                  │
                ├── encoder_smpl ──┼──► FSQ quantizer ──► decoder ──► action_mean (29 DOF)
SMPL pose ──────┤                  │   (2 tokens,
                └── encoder_teleop ┘    32 levels each)
VR 3-point ─────┘                                  ▲
                                                   │
                                       proprioception (always-on)
```

* **Three modality-specific encoders** (G1, SMPL, VR Teleop) each produce a latent.
* A **Finite Scalar Quantizer** (FSQ, ~2 tokens × 32 levels — `max_num_tokens: 2`, `num_fsq_levels: 32`) compresses the latent into a discrete code shared across modalities.
* A **shared decoder** emits 29-DOF joint targets (G1 full-dex 12-finger model).
* At inference the runtime picks an encoder; at training all three are sampled (`encoder_sample_probs: {g1: 1.0, teleop: 1.0, smpl: 1.0}`, [`config/exp/manager/universal_token/all_modes/sonic_release.yaml`](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml)).

### 2.2 Auxiliary losses — the alignment scaffolding

Configured at [`gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml`](gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml), implemented in [`gear_sonic/trl/losses/token_losses.py`](gear_sonic/trl/losses/token_losses.py). These ride alongside PPO inside a custom trainer (§2.4):

| Loss term | Coef | What it does |
|---|---|---|
| `g1_recon` | 0.01 | FK-based reconstruction of G1 decoder outputs (6D rot + 3D pos, geodesic) |
| `g1_smpl_latent` | 1.0 | Pull SMPL encoder latent toward G1 encoder latent |
| `g1_teleop_latent` | 1.0 | Pull teleop encoder latent toward G1 encoder latent |
| `teleop_smpl_latent` | 1.0 | Cross-modal SMPL ↔ teleop consistency |
| `reencoded_smpl_g1_latent` | 1.0 | **Cycle-consistency**: SMPL latent → G1 decode → re-encode → match SMPL latent |

This is effectively a multi-modal contrastive framework on top of PPO. Cycle-consistency validates that the FSQ bottleneck preserves enough information to be modality-invariant.

### 2.3 Multi-mode observation surface

[`config/manager_env/observations/tokenizer/unitoken_all_noz.yaml`](gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz.yaml) defines what each encoder consumes. The same critic sees a privileged superset ([`observations/critic/privileged_mf_hist.yaml`](gear_sonic/config/manager_env/observations/critic/privileged_mf_hist.yaml)) — classic asymmetric actor-critic.

| Mode | Encoder consumes | Use case |
|---|---|---|
| **g1** | 29-DOF joint pos+vel over 10 future frames + 6D anchor orientation | online reactive tracking from robot state |
| **smpl** | 22 SMPL body joints + root orientation + wrist positions | imitating mocap clips |
| **teleop** | Lower-body cmd + VR 3-point pos/orn (head + 2 hands) | live VR control |

Policy always also reads proprioceptive history; output is always a 29-D diagonal Gaussian with clamped std (0.001–0.5).

### 2.4 PPO with auxiliary losses, on a forked TRL

Hugging Face's [TRL](https://github.com/huggingface/trl) 0.28 is **vendored and patched** under [`gear_sonic/trl/`](gear_sonic/trl/) — that's why `train_agent_trl.py` has the `sys.path` dance (CLAUDE.md warns: do not "simplify" it).

Key files:

* [`trl/trainer/ppo_trainer.py`](gear_sonic/trl/trainer/ppo_trainer.py) — wraps policy + value + (optional) discriminator into one `PolicyAndValueWrapper` to dodge DDP gradient-sync issues; dispatches `mode="policy" | "policy_distill" | "vae_policy_deterministic" | "value" | "train_disc" | ...`.
* [`trl/trainer/ppo_trainer_aux_loss.py`](gear_sonic/trl/trainer/ppo_trainer_aux_loss.py) — `TRLAuxLossPPOTrainer` extends PPO to pull `aux_losses` + `aux_loss_coef` out of the forward dict and add a weighted sum to the PPO objective.
* [`trl/modules/actor_critic_modules.py`](gear_sonic/trl/modules/actor_critic_modules.py) — actor with running-mean-std normalization, optional temporal history buffer, optional batchnorm.
* [`trl/modules/universal_token_modules.py`](gear_sonic/trl/modules/universal_token_modules.py) — the encoder→FSQ→decoder block.

PPO hyperparameters (`config/algo/ppo_im_phc.yaml`): `clip_param=0.2`, `gamma=0.99`, `lam=0.95`, `entropy_coef=0.01`, **adaptive LR with KL target** (`desired_kl=0.01`, range `1e-5 … 2e-4`). Standard scaffolding, but everything around it (multi-encoder dispatch, aux losses, multi-modal sampling) is custom.

### 2.5 Distributed training

[`train_agent_trl.py`](gear_sonic/train_agent_trl.py) drives training via **Hugging Face Accelerate** with `DistributedDataParallelKwargs` and a 6000 s `InitProcessGroupKwargs` timeout (sim steps can be slow). Per-rank seed offset, main-process-only checkpoint/WandB. Recipe targets **64+ GPUs** for from-scratch, single-node 8-GPU for finetune. Sync of advantage normalization across ranks (`sync_advantage_normalization: True`) keeps PPO stable at this scale.

### 2.6 Data pipeline

[`gear_sonic/data_process/`](gear_sonic/data_process/) handles motion-data conversion:

* `convert_soma_csv_to_motion_lib.py` — Bones-SEED G1 CSV → internal motion library (joint trajectories, target 30 Hz from 120 Hz source).
* `filter_and_copy_bones_data.py` — filter/clean the corpus.
* `extract_soma_joints_from_bvh.py` — BVH → SOMA skeleton.
* `split_pkl_files.py` — train/val split.

Released data: `bones_seed_smpl/` (SMPL motion) + `motion_lib_bones_seed/robot_filtered` (G1 joint motion). Bones-SEED itself comes from <https://huggingface.co/datasets/bones-studio/seed>.

### 2.7 Evaluation & export

* [`eval_agent_trl.py`](gear_sonic/eval_agent_trl.py) — rollout + metrics + optional video. **Exports per-encoder ONNXes** (`_g1.onnx`, `_smpl.onnx`, `_teleop.onnx`) plus standalone encoder/decoder for downstream deployment.
* [`eval_exp.py`](gear_sonic/eval_exp.py) — daemonized continuous evaluator: watches the experiment dir, picks up new checkpoints, runs eval, logs to WandB.

The eval/ONNX-export path is the bridge from research code → C++ runtime: nothing else of the SONIC model crosses into the deploy stack.

---

## 3. How GEAR-SONIC integrates with Isaac Lab

This is the part most worth understanding deeply: Isaac Lab is not used as a black box — SONIC writes its own manager terms.

### 3.1 Environment wrapper

Wrapped at [`envs/wrapper/manager_env_wrapper.py`](gear_sonic/envs/wrapper/manager_env_wrapper.py). The wrapper takes an Isaac Lab `ManagerBasedRLEnv`, grabs the `"motion"` command term to expose the motion library, optionally wires a `"force"` perturbation term, and adds runtime knobs the policy uses but the env doesn't know about:

* **Latent residual injection** (post- or pre-quantization) — lets a higher-level policy nudge the FSQ latent at inference time.
* **Student-direct-latent mode** — for distilled vision students that emit the latent directly.
* **Visualization markers** for contact / anchor points when not headless.

### 3.2 Custom manager terms — the table that matters

Isaac Lab's MDP factoring is `commands / observations / rewards / terminations / events`. SONIC fills each:

| Manager component | File | What's custom |
|---|---|---|
| **Commands** | [`envs/manager_env/mdp/commands.py`](gear_sonic/envs/manager_env/mdp/commands.py) | `TrackingCommand` — the motion library: sampling, multi-future framing (10 frames @ 0.1 s, plus separate SMPL window at 0.02 s), adaptive-failure sampling (`adp_samp_failure_rate_max_over_mean: 200`), upper-body augmentation prefixes, encoder-index one-hot |
| **Observations** | [`envs/manager_env/mdp/observations.py`](gear_sonic/envs/manager_env/mdp/observations.py) | ~50 terms across policy/critic/tokenizer groups; multi-future framing in egocentric body frame; noise injection for robustness |
| **Rewards** | [`envs/manager_env/mdp/rewards.py`](gear_sonic/envs/manager_env/mdp/rewards.py) | 5-point body tracking (torso + 2 wrists + feet implicit), anchor-pos/orn tracking, action-rate L2, joint-limit penalties, feet-acc, anti-shake angular vel |
| **Terminations** | [`envs/manager_env/mdp/terminations.py`](gear_sonic/envs/manager_env/mdp/terminations.py) | Adaptive per-env thresholds on anchor deviation; feed back into the command term's failure stats |
| **Events** | [`envs/manager_env/mdp/events.py`](gear_sonic/envs/manager_env/mdp/events.py) | Mass / friction / damping randomization, COM offset, external push perturbations, curriculum (`events/tracking/level0_4`) |

Robots live under [`envs/manager_env/robots/`](gear_sonic/envs/manager_env/robots/) — `g1.py` (G1 29-DOF, 12-finger dex model) and `h2.py` (alternative humanoid). The scene/terrain config is in [`envs/manager_env/modular_tracking_env_cfg.py`](gear_sonic/envs/manager_env/modular_tracking_env_cfg.py), which also handles per-env camera spawning with OpenCV-style intrinsics (`fx, fy, cx, cy, k1..k6, p1, p2, s1..s4`).

### 3.3 Sim params

* PhysX backend (Isaac Sim 4.5 / Isaac Lab).
* Control frequency: **50 Hz** (0.02 s per step) — matches the deploy runtime exactly.
* Default scale: **4096 parallel envs** per GPU (see `num_envs: 4096`).
* Terrain: `trimesh` for SONIC release (per `sonic_release.yaml`).

### 3.4 Why "Manager" envs and not direct env classes

The Hydra config tree under [`gear_sonic/config/manager_env/`](gear_sonic/config/manager_env/) is the actual interface researchers use — they compose experiments by overriding individual manager groups:

```yaml
defaults:
  - /algo: ppo_im_phc
  - /manager_env: base_env
  - /aux_losses: universal_token/g1_recon_and_all_latent
  - override /actor_critic: universal_token/all_mlp_v1
  - override /manager_env/observations/tokenizer: unitoken_all_noz
  - override /manager_env/observations/policy: local_dir_hist
  - override /manager_env/observations/critic: privileged_mf_hist
  - override /manager_env/events: tracking/level0_4
  - override /manager_env/terminations: tracking/base_adaptive_strict_ori_foot_xyz
  - override /manager_env/rewards: tracking/base_5point_local_feet_acc
```

This is how Isaac Lab's manager-based design pays off: an ablation = a one-line override, not a code fork.

---

## 4. SONIC deploy runtime (`gear_sonic_deploy/`)

The research model is only useful if it runs on the robot at 500 Hz with hard real-time guarantees. The deploy stack is a separate, **C++ / TensorRT** project. The single best resource is [`deploy-design.md`](deploy-design.md); summary follows.

### 4.1 Critical design choice

The runtime does **not** call Unitree's high-level locomotion service (`g1_loco_client`). Instead it generates full-body motion targets itself (via reference clip / ZMQ stream / kinematic planner) and tracks them with the SONIC policy, publishing low-level `LowCmd` joints directly to `rt/lowcmd` at 500 Hz. The point of SONIC is that one policy replaces Unitree's stack.

### 4.2 Four-layer architecture

```
Input  → Motion Target → Inference (TensorRT) → Robot I/O (DDS)
```

| Layer | Implemented by | Rate |
|---|---|---|
| **Input** — keyboard, Unitree wireless gamepad (bytes inside `LowState_.wireless_remote`), ZMQ, ZMQ-manager, ROS 2 | [`include/input_interface/`](gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/) | 100 Hz |
| **Motion target** — preloaded CSV clip, streamed motion, or kinematic planner | `motion_data_reader.hpp`, `localmotion_kplanner*.hpp` | 10 Hz (planner) |
| **Inference** — optional encoder (1751→64-dim token) + control policy (obs→29-D action), both TRT-engine from ONNX | `encoder.hpp`, `control_policy.hpp` | 50 Hz |
| **Robot I/O** — `ChannelPublisher<LowCmd_>`, `ChannelSubscriber<LowState_>`, `MotionSwitcherClient`, Dex3 hand DDS | Unitree SDK2 + `dex3_hands.hpp` | 500 Hz |

Four threads via `CreateRecurrentThreadEx()` carry these rates. The state machine is `INIT → WAIT_FOR_CONTROL → CONTROL`.

### 4.3 Models the runtime expects

```
gear_sonic_deploy/
├── policy/release/
│   ├── model_encoder.onnx      # 48 MB, obs → 64-D token
│   ├── model_decoder.onnx      # 40 MB, token+prop → 29-D action
│   └── observation_config_sonic_release.yaml
└── planner/target_vel/V2/
    └── planner_sonic.onnx      # 739 MB, locomotion planner
```

The **observation config YAML** is the contract: it lists which observation terms are enabled and in what order, per-mode encoder subsets, and history depths. Cross-version drift between training & deploy is caught here.

### 4.4 Kinematic planner

[`localmotion_kplanner_tensorrt.hpp`](gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp) runs a separate TRT engine at 10 Hz that takes a `MovementState` (mode ∈ {IDLE, SLOW_WALK, RUN, SQUAT, CRAWL, BOXING}, movement+facing direction, speed, height) and emits a future `MotionSequence` (30 Hz qpos, resampled to 50 Hz, blended into the existing trajectory). v0/v1/v2 model variants differ in inputs (v2 adds height + specific target + token-count control). This is what lets the gamepad command "walk forward at 0.6 m/s" while the policy actually executes it.

### 4.5 Build & launch

```bash
cd gear_sonic_deploy
just build                                      # cmake → release
./deploy.sh real                                # auto-detects 192.168.123.x
./deploy.sh sim --input-type zmq_manager        # loopback + scripted input
./deploy.sh --policy-fp16 --planner-fp16 real   # FP16 engines
```

`deploy.sh` resolves the network interface, sources `setup_env.sh`, runs `just build`, and execs `g1_deploy_onnx_ref` with all the path flags. CMake requires `TensorRT_ROOT`; CUDA is auto-discovered (with a Jetson fallback path).

---

## 5. VLA inference pipeline (the newest layer)

Added 2026-05-07 (commit `d028795`). Lets Isaac-GR00T **N1.7 VLA** policies drive the G1 *through* SONIC — vision-language-action up top, SONIC WBC underneath.

### 5.1 Wiring

* [`gear_sonic/scripts/run_vla_inference.py`](gear_sonic/scripts/run_vla_inference.py) — standalone runner, **no ROS 2**. Pure ZMQ:
  * Robot state ← ZMQ SUB on `g1_debug` topic (from C++ `zmq_output_handler`)
  * Actions → ZMQ PUB (protocol v4: latent tokens + hand joints — bypasses motion decoding)
  * Camera ← ComposedCameraClientSensor (TCP from camera server)
  * Keyboard ← ZMQKeyboardSubscriber
* [`gear_sonic/scripts/launch_inference.py`](gear_sonic/scripts/launch_inference.py) — tmux multiplexer that launches deploy + VLA + keyboard pub + data exporter side by side.
* [`gear_sonic/utils/inference/vla_utils.py`](gear_sonic/utils/inference/vla_utils.py) — observation splitting (whole-body `q` → arm/leg/hand groups), latency compensation, "should I trigger new inference?" gating.
* [`gear_sonic/utils/inference/initial_poses.py`](gear_sonic/utils/inference/initial_poses.py) — **`LATENT_INITIAL_MOTION_TOKEN`**, a 64-D motion token for stable standing. Checkpoint-specific — must be regenerated whenever the SONIC checkpoint changes.
* [`install_scripts/install_inference.sh`](install_scripts/install_inference.sh) — creates `.venv_inference/` with Isaac-GR00T, PyZMQ, Pinocchio, msgpack.

Inference cadence: 2.5 Hz with a 40-step action horizon; control loop consumes from a sliding cache. This is how a slow VLA can drive a 50 Hz controller.

### 5.2 Why it matters

Before this, deploying a VLA on G1 required cobbling together two stacks. Now: Isaac-GR00T emits latent SONIC tokens → ZMQ → C++ deploy executes them. Both ends speak the same FSQ token language, so the integration is essentially a memcpy.

---

## 6. Decoupled WBC (`decoupled_wbc/`)

The original GR00T N1.5 / N1.6 controller. **Not trained or run in Isaac Lab.** Lives next to SONIC for continuity with shipped models.

### 6.1 Architecture

[`G1DecoupledWholeBodyPolicy`](decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py) merges two independent controllers:

* **Lower body — RL (ONNX)**: 86-D obs (vel cmds, gravity, joint pos/vel, last action) → 15-D action. Two pretrained policies ship: `GR00T-WholeBodyControl-Walk.onnx` and `GR00T-WholeBodyControl-Balance.onnx` (under `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/`). Loaded by [`G1GearWbcPolicy`](decoupled_wbc/control/policy/g1_gear_wbc_policy.py) via ONNX Runtime.
* **Upper body — interpolation + IK**: [`InterpolationPolicy`](decoupled_wbc/control/policy/interpolation_policy.py) drives toward a target pose at bounded joint speed. Hand & arm IK in [`control/teleop/solver/`](decoupled_wbc/control/teleop/solver/) using Pinocchio + qpsolvers (Pink-style QP-IK).

The decoupling is literal — joints are partitioned via `robot_model.get_joint_group_indices("lower_body" | "upper_body")` and each half runs its own controller; there is **no learned coupling** between them.

### 6.2 Engineering bits worth knowing

* **Multi-sim support** in [`control/envs/g1/sim/`](decoupled_wbc/control/envs/g1/sim/): MuJoCo, RoboCasa, and Unitree SDK2 bridge — same control loop runs against all three.
* **Teleop stack** in [`control/teleop/`](decoupled_wbc/control/teleop/): Qt6 GUI (`gui/gui3d.py`), VR/iPhone/Manus/Vive/LeapMotion device handlers, hand+body IK solvers. Console entry point `decoupled_wbc = decoupled_wbc.control.teleop.gui.cli:cli` in `decoupled_wbc/pyproject.toml`.
* **Deployment orchestrator** at [`scripts/deploy_g1.py`](decoupled_wbc/scripts/deploy_g1.py): `G1Deployment` launches camera server + viewer + sim + control loop + data exporter as separate tmux panes.
* **ONNX eval harness** in [`sim2mujoco/`](decoupled_wbc/sim2mujoco/): self-contained MuJoCo evaluator that runs the same ONNXes outside the deploy stack. Useful for regression testing policies without robot hardware.
* **Test suite** under `decoupled_wbc/tests/` covers policy interpolation, robot model kinematics, IK retargeting, meshcat visualization, and sim data collection (pytest rootdir is set here in `pyproject.toml`).

### 6.3 Why it's still here

It's the controller currently shipping inside the GR00T N1.5 / N1.6 foundation models. SONIC is the successor research direction, but Decoupled WBC remains the production path for those releases.

---

## 7. MotionBricks (`motionbricks/`)

A 2026-04-27 preview release: a real-time **latent generative model** for motion, separate from SONIC's control policy. Pure PyTorch Lightning + MuJoCo, **no Isaac Lab today**.

### 7.1 Three networks

1. **VQ-VAE** ([`vqvae/models/motion_vqvae.py`](motionbricks/motionbricks/vqvae/models/motion_vqvae.py)): tokenizes local pose features into a 512-code, 4-head EMA codebook. Conv1D down-sampling (factor 2 or 4), DoubleCond decoder conditioned on a boundary pose + root.
2. **Pose Backbone** ([`motion_backbone/neural_modules/pose_backbone.py`](motionbricks/motionbricks/motion_backbone/neural_modules/pose_backbone.py)): transformer over discrete pose tokens, generates future pose tokens via cosine-annealed masking.
3. **Root Backbone** ([`motion_backbone/neural_modules/root_backbone.py`](motionbricks/motionbricks/motion_backbone/neural_modules/root_backbone.py)): transformer that predicts continuous root trajectories (XZ position + heading) from start/end constraints; outputs are decoded by a DoubleCondDecoder.

### 7.2 Motion representation

[`docs/motion_representation.md`](motionbricks/docs/motion_representation.md) — a dual root representation (lossless global↔local conversion) over a 34-joint G1 skeleton, totalling **418 dims/frame**: 5-D global or 4-D local root + 409-D body features (`ric_data` 99 + `global_rot_data` 204 (6D rotations) + `local_vel` 102 + `foot_contacts` 4). Z-score normalized with stats stored in `out/.../stats/motion/`.

### 7.3 What ships

Pretrained checkpoints (VQVAE, pose, root) live under `motionbricks/out/...` and are **opt-in via Git LFS** — the new `.lfsconfig` excludes `motionbricks/out/**` from the default `git lfs pull` to avoid a ~2.2 GiB download. Fetch with:

```bash
git lfs pull --include="motionbricks/out/**" --exclude=""
```

### 7.4 Interactive demo

[`scripts/interactive_demo_g1.py`](motionbricks/scripts/interactive_demo_g1.py) — MuJoCo passive viewer, WASD keyboard control, ~15 k FPS generative sampling. Calls `demo_agent.full_agent.generate_new_frames(control_signals, dt)` to stream new motion. Linux-only X11 passive key-grab guards against MuJoCo viewer keybinds.

### 7.5 Roadmap

`motionbricks/README.md` mentions a **future merge into the SONIC pipeline** (~1 month). That would be the natural place for Isaac Lab integration — currently motionbricks is purely a generative model trained offline.

---

## 8. Shared infrastructure

### 8.1 Environments — there is no single venv

Each use case has its own uv-managed venv (CLAUDE.md has the canonical table):

| Use case | Venv | Installer |
|---|---|---|
| SONIC training | (Isaac Lab's own Python 3.11) | `pip install -e "gear_sonic/[training]"` |
| MuJoCo simulator | `.venv_sim` | `install_scripts/install_mujoco_sim.sh` |
| VR teleop | `.venv_teleop` (`.venv_pico`) | `install_scripts/install_pico.sh` |
| Data collection (LeRobot) | `.venv_data_collection` | `install_scripts/install_data_collection.sh` |
| Camera server (OAK-D) | `.venv_camera` | `install_scripts/install_camera_server.sh` |
| **VLA inference (NEW)** | `.venv_inference` | `install_scripts/install_inference.sh` |
| Deploy runtime | (none — C++ build) | `cd gear_sonic_deploy && just build` |

Training **requires Python 3.11.x exactly** (Isaac Lab constraint). Everything else wants ≥3.10. `python check_environment.py [--training | --deploy]` verifies versions, LFS pull status (samples an STL + an ONNX), CUDA/PyTorch, Isaac Lab, and `TensorRT_ROOT`.

### 8.2 Asset & data flow

```
HF Hub (nvidia/GEAR-SONIC) ──┬── download_from_hf.py ──► gear_sonic_deploy/policy/release/*.onnx
                             │                          gear_sonic_deploy/planner/.../planner_sonic.onnx
                             └── download_from_hf.py --training ──► sonic_release/last.pt
                                                                    bones_seed_smpl/

Bones-SEED (HF datasets/bones-studio/seed) ──► data_process/convert_soma_csv_to_motion_lib.py
                                            ──► data/motion_lib_bones_seed/robot_filtered

Git LFS: meshes, ONNX, planner — opt out of motionbricks/out/** via .lfsconfig
```

### 8.3 Lint / test / docs

* `make run-checks` — isort --check, black --check, ruff check (`./lint.sh --fix` for the same rules with autofix).
* Ruff: `E, F, I`, `line-length=115`, target `py310`. Black: `line-length=100`. `external_dependencies/` and `gear_sonic/dexmg` are excluded.
* `pytest decoupled_wbc/tests` — pytest rootdir is set there in `pyproject.toml`.
* C++ tests: `./target/release/run_tests` after `just build` in `gear_sonic_deploy/`.
* Sphinx: `sphinx-build -b html docs/source docs/build/html`.

---

## 9. Glossary of file landmarks

If you only memorize five paths, make them these:

| Path | Why |
|---|---|
| [`deploy-design.md`](deploy-design.md) | The single best deep-dive on the C++ runtime |
| [`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`](gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml) | The "what is SONIC release" config — Hydra defaults pull in everything else |
| [`gear_sonic/trl/modules/universal_token_modules.py`](gear_sonic/trl/modules/universal_token_modules.py) | The Universal Token implementation |
| [`gear_sonic/envs/manager_env/mdp/commands.py`](gear_sonic/envs/manager_env/mdp/commands.py) | How the motion library plugs into Isaac Lab as a Command term |
| [`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`](gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp) | The four-thread C++ main loop the policy actually lives in on the robot |

---

## 10. Things that will bite you (collected from CLAUDE.md + observed)

* Running `gear_sonic/train_agent_trl.py` without the Isaac Lab env fails fast. A mis-versioned Python (must be 3.11.x) is only caught by `check_environment.py --training`.
* `git clone` without `git lfs pull` silently produces pointer files. `check_environment.py` samples an STL mesh and an ONNX to detect this.
* **ZMQ header size changed to 1280 bytes** in the 2026-03-24 stack update — cross-version ZMQ clients/servers desync silently. Match versions.
* `gear_sonic/dexmg/` is Ruff-excluded; don't expect lint feedback there. Same for `external_dependencies/`.
* `LATENT_INITIAL_MOTION_TOKEN` in `gear_sonic/utils/inference/initial_poses.py` is **tied to a specific SONIC checkpoint** — if you swap checkpoints you must regenerate it.
* Do **not** simplify the `sys.path` block at the top of `gear_sonic/train_agent_trl.py`; the file is run as a script and would otherwise import the wrong `trl`.
* MotionBricks checkpoints (~2.2 GiB) are now opt-in. Default `git lfs pull` won't fetch them.

---

*Authored on the merge that landed on `main` 2026-05-23, after the 20-commit pull bringing SONIC release + VLA pipeline + MotionBricks preview.*
