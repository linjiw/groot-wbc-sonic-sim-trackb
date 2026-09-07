#!/usr/bin/env bash
# Build the CPU/torch research environment for the hallucination (LFH / LfLH) work.
#
# This is the environment REPLICATION.md calls "everything CPU-side": candidate sets, LfLH
# training, geometry, MuJoCo rendering, and the tests/dataset_generation suite. It does NOT
# contain Isaac Lab, so it cannot run physics rollouts -- see check_environment.py --training.
#
# Deliberately separate from the Kimodo generation env (install_kimodo_sm120.sh): the two
# stages exchange qpos CSV files on disk and must never share a dependency graph.
#
# torch comes from the cu128 index because Blackwell (RTX 5080/5090, sm_120) has no kernels
# in the default cu124 wheels.
#
# Usage:  bash install_scripts/install_research.sh [venv_dir]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO_ROOT/.venv_research}"
PY_VERSION="3.11"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

command -v uv >/dev/null || { echo "uv not found on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

echo "==> creating venv at $VENV (python $PY_VERSION)"
uv venv --python "$PY_VERSION" "$VENV"
PY="$VENV/bin/python"

echo "==> installing torch from $TORCH_INDEX"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" torch

echo "==> verifying the GPU's arch is in the torch fatbin before installing anything else"
"$PY" - <<'PROBE'
import sys

import torch

print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    print("WARNING: no CUDA device visible; the CPU paths still work, rendering will not")
    sys.exit(0)
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
print(f"device {torch.cuda.get_device_name(0)}  {arch}")
if arch not in torch.cuda.get_arch_list():
    sys.exit(f"{arch} missing from {torch.cuda.get_arch_list()} -- wrong torch wheel")
PROBE

echo "==> installing the research dependencies"
uv pip install --python "$PY" \
    numpy scipy matplotlib pandas pyarrow pyyaml joblib tqdm loguru \
    mujoco pin imageio imageio-ffmpeg pillow opencv-python-headless \
    vector-quantize-pytorch \
    pytest isort black ruff

cat <<EOM

==> done. The two flags REPLICATION.md documents are still required:

  env -u PYTHONPATH $PY <script>

  env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    $PY -m pytest tests/dataset_generation -q -p no:cacheprovider

Rendering needs MUJOCO_GL=egl (the scripts set it themselves). An EGLError traceback printed
*after* a render succeeds is interpreter teardown, not a failure -- read the result line.
EOM
