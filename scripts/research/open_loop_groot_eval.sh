#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a GR00T open-loop evaluation command for the UNITREE_G1_SONIC MVP.

This wrapper intentionally does not guess the Isaac-GR00T eval entrypoint if the
external repo changes. It validates the required paths and prints the exact
command shape to run from Isaac-GR00T.

Usage:
  scripts/research/open_loop_groot_eval.sh \
    --groot-repo /path/to/Isaac-GR00T \
    --model-path /path/to/checkpoint \
    --dataset-path /path/to/cleaned_lerobot_dataset \
    [--embodiment-tag UNITREE_G1_SONIC]
EOF
}

GROOT_REPO=""
MODEL_PATH=""
DATASET_PATH=""
EMBODIMENT_TAG="UNITREE_G1_SONIC"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --groot-repo)
      GROOT_REPO="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --embodiment-tag)
      EMBODIMENT_TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GROOT_REPO" || -z "$MODEL_PATH" || -z "$DATASET_PATH" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$GROOT_REPO" ]]; then
  echo "Isaac-GR00T repo not found: $GROOT_REPO" >&2
  exit 1
fi
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset path not found: $DATASET_PATH" >&2
  exit 1
fi

cat <<EOF
Validated inputs.

Run the current Isaac-GR00T open-loop/eval entrypoint from:
  $GROOT_REPO

Required arguments:
  --model-path $MODEL_PATH
  --dataset-path $DATASET_PATH
  --embodiment-tag $EMBODIMENT_TAG
  --modality-config-path gr00t/configs/data/embodiment_configs.py

Before fine-tuning or deployment, also run:
  scripts/research/check_sonic_vla_dataset.py $DATASET_PATH
EOF
