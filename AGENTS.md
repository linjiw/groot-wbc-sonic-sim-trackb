# Repository Guidelines

## Project Structure & Module Organization
This repository combines Python control stacks, C++ deployment code, and docs:

- `decoupled_wbc/`: Python package for control/data workflows; tests live in `decoupled_wbc/tests/`.
- `gear_sonic/`: Python utilities for teleoperation, MuJoCo simulation, networking, and robot data.
- `gear_sonic_deploy/`: C++20 deployment stack (TensorRT/ONNX Runtime), scripts, and `just`-based builds.
- `docs/`: Sphinx documentation sources under `docs/source/`.
- `install_scripts/`: setup helpers (for example MuJoCo sim env bootstrap).

Keep changes scoped to the relevant module; avoid mixing deploy, teleop, and docs edits in one PR unless tightly coupled.

## Build, Test, and Development Commands
- `git lfs pull`: fetch tracked model/assets after clone.
- `make run-checks`: run `isort`, `black --check`, and `ruff check` from repo root.
- `./lint.sh --fix`: auto-fix Python lint/style issues with Ruff + Black.
- `make format`: apply `isort` + `black` formatting.
- `pytest decoupled_wbc/tests`: run Python test suite.
- `cd gear_sonic_deploy && just build`: configure and build C++ deploy binaries.
- `cd gear_sonic_deploy && ./target/release/run_tests`: run C++ unit tests after build.
- `sphinx-build -b html docs/source docs/build/html`: build docs locally.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` modules/functions, `CamelCase` classes.
- Format Python with Black (configured in `pyproject.toml`), organize imports with isort/Ruff (`I` rules).
- Lint with Ruff (`E`, `F`, `I`) before opening a PR.
- C++ (`gear_sonic_deploy/`): C++20, 2-space indentation, 120-column limit (`.clang-format`), headers/sources split under `include/` and `src/`.

## Testing Guidelines
- Prefer focused tests near modified code:
  - Python: add `test_*.py` under `decoupled_wbc/tests/...`.
  - C++: add unit tests under `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/`.
- Some tests require optional runtime deps (for example ROS/ffmpeg); document skips and local setup in PR notes.
- Run at least impacted module tests before submitting.

## Commit & Pull Request Guidelines
- Follow short, imperative commit summaries (history includes styles like `update docs`, `[fix] ...`).
- Recommended commit format: `<scope>: <imperative summary>` (example: `deploy: fix ZMQ reconnect handling`).
- Sign off commits for DCO compliance: `git commit -s`.
- PRs should include:
  - What changed and why.
  - Exact validation commands run.
  - Linked issue(s).
  - Screenshots/logs for UI, teleop, or deployment behavior changes.
