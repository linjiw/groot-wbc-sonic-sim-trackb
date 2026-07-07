# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

`GR00T-WholeBodyControl` hosts two controller families for NVIDIA humanoid robots (primarily Unitree G1):

- **`decoupled_wbc/`** — the decoupled controller (RL lower body + IK upper body) used in GR00T N1.5 / N1.6.
- **`gear_sonic/`** (training) + **`gear_sonic_deploy/`** (C++ runtime) — the GEAR-SONIC motion-tracking foundation model, its PPO training stack on Isaac Lab, and the on-robot TensorRT inference stack.

Additional support packages: `docs/` (Sphinx), `install_scripts/` (uv-based venv setup for sim/teleop/data/camera), `systemd/` (service units), `external_dependencies/`.

## Environments

There is no single all-in-one env — each use case has a dedicated one (see README's "Which environment do I need?" table):

| Use case | Environment | Entry |
|---|---|---|
| Train / finetune SONIC | Isaac Lab's Python 3.11 env | `pip install -e "gear_sonic/[training]"` |
| MuJoCo sim | `.venv_sim` | `bash install_scripts/install_mujoco_sim.sh` |
| VR teleop | `.venv_teleop` | `bash install_scripts/install_pico.sh` |
| Data collection | `.venv_data_collection` | `bash install_scripts/install_data_collection.sh` |
| Camera server | `.venv_camera` | `bash install_scripts/install_camera_server.sh` |
| Deploy on real robot | C++ build in `gear_sonic_deploy/` | see below |

Install scripts use `uv` and create isolated venvs automatically. Training **requires Python 3.11.x exactly** (Isaac Lab constraint) — everything else wants ≥3.10.

Before running anything: `python check_environment.py [--training|--deploy]` verifies Python version, Git LFS pull status, CUDA/PyTorch, Isaac Lab, and (for deploy) `TensorRT_ROOT`.

**Git LFS is mandatory.** Meshes, ONNX policies, and planner models are LFS-tracked; without `git lfs pull` you get tiny pointer files and silent runtime failures.

## Common commands

### Python lint / format / test (run from repo root)

```bash
make run-checks          # isort --check, black --check, ruff check
make format              # isort + black (writes files)
./lint.sh --fix          # Ruff + Black auto-fix (same rules as CI)
pytest decoupled_wbc/tests                           # full Python test suite (pytest rootdir is this)
pytest decoupled_wbc/tests/path/to/test_foo.py::TestBar::test_baz   # single test
```

Ruff config selects `E`, `F`, `I` with `line-length = 115`, target `py310`; Black uses `line-length = 100`. `external_dependencies/` and `gear_sonic/dexmg` are excluded. `pytest` rootdir = `decoupled_wbc/tests/` (set in `pyproject.toml`), so pass paths relative to repo root or use absolute paths.

### C++ deployment build (`gear_sonic_deploy/`)

Uses [`just`](https://just.systems) — the `.justfile` lives in `gear_sonic_deploy/`.

```bash
cd gear_sonic_deploy
just build                          # cmake configure + build (Release)
just build Debug                    # debug build
just run g1_deploy_onnx_ref <iface> <decoder.onnx> <motion_dir> ...   # direct launch
./target/release/run_tests          # C++ unit tests (after build)
just clean                          # wipe build/ and target/
```

Use `./deploy.sh [sim|real|<iface>|<ip>]` as the wrapped launcher — it resolves the network interface (loopback for sim, `192.168.123.x` for real G1), checks model files, sources `scripts/setup_env.sh`, runs `just build`, and execs `g1_deploy_onnx_ref` with decoder/encoder/planner/motion paths. Takes flags like `--cp`, `--obs-config`, `--planner`, `--input-type`, `--output-type`.

`TensorRT_ROOT` must be exported (e.g. `export TensorRT_ROOT=$HOME/TensorRT`) before building; `CUDAToolkit_ROOT` / `CUDA_HOME` are auto-picked up.

### Training (requires Isaac Lab env activated)

```bash
# Download checkpoint + SMPL data from HF
python download_from_hf.py --training

# Convert + filter Bones-SEED G1 CSVs into motion lib
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --input ... --output ... --fps 30 --fps_source 120 --individual --num_workers 16
python gear_sonic/data_process/filter_and_copy_bones_data.py --source ... --dest ...

# Finetune from released checkpoint (64+ GPUs recommended)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

`gear_sonic/train_agent_trl.py` manipulates `sys.path` at import time because running it as a script would otherwise shadow HuggingFace's `trl` with the local `gear_sonic/trl/` package — do not "simplify" the sys.path block. Eval: `gear_sonic/eval_agent_trl.py` / `eval_exp.py`.

### Docs

```bash
sphinx-build -b html docs/source docs/build/html
```

## Architecture at a glance

### Deploy runtime (`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`)

Read `deploy-design.md` for the full explanation. The critical design point: **the runtime does not call Unitree's high-level locomotion RPC** (`g1_loco_client`). Instead it generates full-body motion targets itself and tracks them with a learned policy, publishing low-level `LowCmd` joints via Unitree SDK2 DDS at 500 Hz.

Four layers:

1. **Input** — keyboard / Unitree wireless gamepad (carried in `LowState_.wireless_remote`) / ZMQ / ZMQ manager / ROS2. See `include/input_interface/`.
2. **Motion target** — preloaded reference clip (`motion_data_reader.hpp`), streamed motion, or kinematic planner output (`localmotion_kplanner*.hpp`, 10 Hz).
3. **Inference** — optional encoder (`encoder.hpp`) → control policy (`control_policy.hpp`), both TensorRT-accelerated from ONNX.
4. **Robot I/O** — `ChannelPublisher<LowCmd_>`, `ChannelSubscriber<LowState_>`/`<IMUState_>`, `MotionSwitcherClient`, Dex3 DDS topics, `CreateRecurrentThreadEx`.

Hardware constants and joint index mappings live in `robot_parameters.hpp` / `policy_parameters.hpp`. Scene/URDF variants under `gear_sonic_deploy/g1/` (23dof, 29dof, 29dof_with_hand).

A separate lighter path exists at `decoupled_wbc/sim2mujoco/` for ONNX-in-MuJoCo evaluation — **not** the real-robot runtime.

### Training stack (`gear_sonic/`)

- Hydra configs in `gear_sonic/config/` (`base.yaml`, `exp/`, `manager_env/`, `algo/`, `actor_critic/`, `trainer/`, `callbacks/`, `aux_losses/`, `opt/`). Exp selection is via `+exp=...` CLI override.
- Isaac Lab manager envs under `gear_sonic/envs/manager_env/` + wrappers in `gear_sonic/envs/wrapper/`.
- RL trainer is a fork of HuggingFace TRL under `gear_sonic/trl/` — has its own `trainer/`, `losses/`, `callbacks/`, `modules/`. This is why `sys.path` manipulation matters in entry scripts.
- Data pipeline in `gear_sonic/data_process/` converts SOMA/Bones-SEED CSV → internal motion library format.

### Decoupled WBC (`decoupled_wbc/`)

- `control/` — policy runtime, robot model, sensors, teleop GUI/CLI (`control/teleop/gui/cli:cli` is the `decoupled_wbc` console entry point), visualization, main loops.
- `scripts/deploy_g1.py` — Python-side G1 deploy path (separate from the C++ stack).
- `sim2mujoco/` — MuJoCo ONNX evaluation harness (self-contained `requirements.txt`).
- `tests/` — pytest suite organized under `control/`, `data/`, `sim/`, `replay_data/`; configured via `conftest.py`.
- Optional extras are heavy and split into `full` and `dev` extras (see `decoupled_wbc/pyproject.toml`) — `full` pulls in ROS-style deps (pin, pin-pink, cv-bridge, realsense, lerobot at a pinned git SHA, etc.).

## Conventions (from AGENTS.md)

- Python: 4-space, `snake_case` funcs/modules, `CamelCase` classes.
- C++ (`gear_sonic_deploy/`): C++20, 2-space indent, 120-col (`.clang-format`), headers under `include/`, sources under `src/`.
- Keep a PR scoped to one of deploy / teleop / training / docs unless changes are tightly coupled.
- Add C++ unit tests under `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/`; add Python tests as `test_*.py` under `decoupled_wbc/tests/...`.
- Commits should be imperative; sign off with `git commit -s` for DCO.

## Things that will bite you

- Running `gear_sonic/train_agent_trl.py` without activating the Isaac Lab env fails fast with a clear message — but a mis-versioned Python (must be 3.11.x) is caught only by `check_environment.py --training`.
- After `git clone`, running anything without `git lfs pull` silently loads pointer files. `check_environment.py` samples an STL mesh and an ONNX to detect this.
- ZMQ header size changed to **1280 bytes** in the 2026-03-24 stack update — cross-version ZMQ clients/servers will desync silently.
- `gear_sonic/dexmg/` is excluded from Ruff; don't expect lint feedback there. `external_dependencies/` is excluded from Black/isort/Ruff/mypy too.
