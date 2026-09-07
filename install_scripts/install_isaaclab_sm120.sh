#!/usr/bin/env bash
# Build the Isaac Lab training/physics environment on Blackwell (sm_120).
#
# This env runs the half of the work .venv_research cannot: physics rollouts through a frozen
# SONIC policy (run_approved_manifest.py and friends) and SONIC training/finetuning. It also
# carries the whole dataset_generation library, because the physics entry points import both.
#
# THE TRAP, and the reason this script exists: `isaacsim==5.1.0` resolves torch 2.7.0+cu126,
# whose fatbin stops at sm_90. On a 5080/5090 that imports fine and reports cuda available, then
# dies on the first kernel launch with "no kernel image is available for execution on the device".
# The fix is to re-pull the SAME torch version from the cu128 index afterwards -- note that this
# needs --reinstall-package, since to a resolver 2.7.0+cu126 already satisfies "torch==2.7.0".
#
# Usage:  bash install_scripts/install_isaaclab_sm120.sh [venv_dir] [isaaclab_src]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO_ROOT/.venv_isaaclab}"
SRC="${2:-$HOME/IsaacLab}"
PY_VERSION="3.11"          # Isaac Lab constraint, exactly 3.11.x
ISAACSIM_VERSION="5.1.0"
ISAACLAB_TAG="v2.3.2"      # the version the root README pins
TORCH_VERSION="2.7.0"      # whatever isaacsim resolved; only the CUDA build changes below
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

command -v uv >/dev/null || { echo "uv not found on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

if [ ! -d "$SRC" ]; then
  echo "==> cloning Isaac Lab $ISAACLAB_TAG to $SRC"
  git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$SRC"
fi

echo "==> creating venv at $VENV (python $PY_VERSION)"
uv venv --python "$PY_VERSION" "$VENV"
PY="$VENV/bin/python"

echo "==> installing Isaac Sim $ISAACSIM_VERSION (~16 GB, slow)"
uv pip install --python "$PY" "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com

echo "==> re-pulling torch $TORCH_VERSION from $TORCH_INDEX so sm_120 kernels exist"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    "torch==$TORCH_VERSION" "torchvision==0.22.0" "torchaudio==$TORCH_VERSION"

"$PY" - <<'PROBE'
import sys

import torch

print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("CUDA not available")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
if arch not in torch.cuda.get_arch_list():
    sys.exit(f"{arch} missing from {torch.cuda.get_arch_list()} -- the cu128 swap did not take")
x = torch.randn(1024, 1024, device="cuda")
assert torch.isfinite((x @ x).sum())
print(f"{arch} kernels present and a real matmul launched")
PROBE

echo "==> installing the Isaac Lab packages from $SRC"
for pkg in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
    uv pip install --python "$PY" -e "$SRC/source/$pkg"
done

echo "==> installing gear_sonic[training] plus the dataset_generation runtime deps"
uv pip install --python "$PY" -e "$REPO_ROOT/gear_sonic/[training]"
uv pip install --python "$PY" \
    mujoco pin pyarrow imageio imageio-ffmpeg opencv-python-headless matplotlib pandas pytest rich

echo "==> verifying"
env -u PYTHONPATH "$PY" "$REPO_ROOT/check_environment.py" --training

cat <<EOM

==> done.

Next, two things this script deliberately does NOT do for you:

  1. Accept the NVIDIA Omniverse EULA. The first Isaac Sim launch prompts for it interactively
     and will hang a non-interactive run. Either launch once by hand and answer Yes, or set
     OMNI_KIT_ACCEPT_EULA=YES once you have read and accepted the agreement at
     https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html

  2. Download the SONIC checkpoint and SMPL data (~30 GB, and the HF repo is gated):
       env -u PYTHONPATH $PY $REPO_ROOT/download_from_hf.py --training
EOM
