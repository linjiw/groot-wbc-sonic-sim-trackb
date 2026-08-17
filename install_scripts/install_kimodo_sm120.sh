#!/usr/bin/env bash
# Build Kimodo a generation environment that runs on Blackwell / sm_120 (RTX 5090).
#
# The pre-existing Kimodo envs carry torch 2.5.1+cu124, whose fatbin stops at sm_90, so
# every CUDA launch on a 5090 dies with "no kernel image is available for execution on
# the device". Kimodo pins no torch version, so the fix is a fresh env on a cu128 wheel.
#
# This env is deliberately SEPARATE from the Isaac Lab 2.3.2 / Isaac Sim 5.1 env. The two
# stages exchange qpos CSV files on disk and must never share a dependency graph.
#
# Usage:  bash install_scripts/install_kimodo_sm120.sh [venv_dir] [kimodo_src]

set -euo pipefail

VENV="${1:-/data/Humanoid/kimodo_sm120}"
SRC="${2:-$HOME/kimodo}"
PY_VERSION="3.11"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

command -v uv >/dev/null || { echo "uv not found on PATH"; exit 1; }
[ -f "$SRC/pyproject.toml" ] || { echo "no Kimodo checkout at $SRC"; exit 1; }

echo "==> creating venv at $VENV (python $PY_VERSION)"
uv venv --python "$PY_VERSION" "$VENV"
PY="$VENV/bin/python"

echo "==> installing torch from $TORCH_INDEX"
VIRTUAL_ENV="$VENV" uv pip install --index-url "$TORCH_INDEX" torch torchvision

echo "==> verifying sm_120 kernels are present before installing anything else"
"$PY" - <<'PROBE'
import sys

import torch

arch = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"arch list: {arch}")
if not torch.cuda.is_available():
    sys.exit("CUDA not available in the new env")
capability = torch.cuda.get_device_capability()
target = f"sm_{capability[0]}{capability[1]}"
print(f"device {torch.cuda.get_device_name(0)} is {target}")
if target not in arch:
    sys.exit(f"{target} missing from the wheel's arch list -- wrong index URL")
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
print(f"matmul on device ok: {float((x @ x).sum()):.3f}")
PROBE

echo "==> installing Kimodo (editable, C++ MotionCorrection extension skipped)"
# MotionCorrection is a CMake extension used for post-hoc motion cleanup; generation does
# not need it, and building it here would drag CMake/toolchain risk onto the critical path.
SKIP_MOTION_CORRECTION_IN_SETUP=1 VIRTUAL_ENV="$VENV" \
  uv pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -e "$SRC"

echo "==> import check"
"$PY" -c "import kimodo, torch; print('kimodo', kimodo.__file__); print('torch', torch.__version__)"

echo
echo "Kimodo sm_120 env ready: $VENV"
echo "Generate with: $PY -m kimodo.scripts.generate ..."
