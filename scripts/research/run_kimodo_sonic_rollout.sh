#!/usr/bin/env bash
# Run one SONIC physics rollout of a converted Kimodo G1 reference in a repo-owned scene.
#
# The M0 spike proved this path with ad-hoc commands whose artifacts were ephemeral. This
# script exists so a rollout is reproducible and its outputs land in a named directory.
#
# Isaac/Hydra can print a fatal traceback and still exit zero (eval_agent_trl.py ends in
# os._exit(0)), so success here is defined by the explicit SONIC_EVAL_SUCCESS marker AND the
# expected artifacts, never by process status.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/research/run_kimodo_sonic_rollout.sh \
    --scene household_room|factory_aisle \
    --motion /abs/path/motion.pkl \
    --out /abs/path/output_dir \
    [--checkpoint /abs/path/last.pt] \
    [--python /abs/path/python] \
    [--scene-package /abs/path/scene_package] \
    [--max-steps auto|<int>] \
    [--task "natural language instruction"]

Writes <out>/trajectories, <out>/renders, <out>/rollout.log and requires the
SONIC_EVAL_SUCCESS marker before reporting success.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE=""
MOTION=""
OUT=""
CHECKPOINT="${REPO_ROOT}/sonic_release/last.pt"
PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/env_isaaclab/bin/python}"
MAX_STEPS=auto
# Bound at capture time, not at export time: the export binding gate cross-checks the
# exported task string against the runtime manifest, so the language annotation a VLA
# will train on has to be decided here.
TASK="kimodo_locomotion"
EXTRA_OVERRIDES=()
SCENE_PACKAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --scene-package) SCENE_PACKAGE="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    # Extra Hydra overrides, whitespace-separated, appended last so they win. Added for the
    # fixed observer camera: the ego view rides the head and the chase view follows the robot,
    # so neither shows gait against a stationary background, and a pinned camera needs
    # render_results, a camera offset and track_root=False that this script otherwise fixes.
    --extra) EXTRA_OVERRIDES+=($2); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for required in SCENE MOTION OUT; do
  if [[ -z "${!required}" ]]; then
    echo "ERROR: --${required,,} is required" >&2
    usage
    exit 2
  fi
done

# `--scene plane` is the obstacle-free control condition. The contact sensor records
# net force per body and cannot separate self-contact from scene contact, so a bare
# plane run is the reference: any non-foot contact there is necessarily self-contact.
if [[ "$SCENE" == "plane" ]]; then
  TERRAIN_OVERRIDES=(++manager_env.config.terrain_type=plane)
  SCENE_USD=""
else
  # Scenes live in one of several packages under assets/scenes: the hand-authored
  # g1_dataset package and any generated clutter package. Search them all rather
  # than hard-coding, so a new package does not silently fail to resolve.
  SCENE_USD=""
  SCENE_PACKAGES=("${REPO_ROOT}"/gear_sonic/data/assets/scenes/*/)
  if [[ -n "$SCENE_PACKAGE" ]]; then
    SCENE_PACKAGES=("${SCENE_PACKAGE%/}/" "${SCENE_PACKAGES[@]}")
  fi
  for pkg in "${SCENE_PACKAGES[@]}"; do
    if [[ -e "${pkg}${SCENE}.usda" ]]; then SCENE_USD="${pkg}${SCENE}.usda"; break; fi
  done
  if [[ -z "$SCENE_USD" ]]; then
    echo "ERROR: scene '${SCENE}' not found in any package under assets/scenes" >&2
    exit 2
  fi
  TERRAIN_OVERRIDES=(
    ++manager_env.config.terrain_type=scene_usd
    ++manager_env.config.scene_usd_path="$SCENE_USD"
  )
  if [[ ! -e "$SCENE_USD" ]]; then
    echo "ERROR: missing required path: $SCENE_USD" >&2
    exit 2
  fi
fi

for path in "$MOTION" "$CHECKPOINT" "$PYTHON_BIN"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT/trajectories" "$OUT/renders"
LOG="$OUT/rollout.log"

# Capture one full pass of *this* motion, not a fixed number of steps. A constant was
# inherited from a corpus whose two motions were the same length; with a varied library it
# silently truncates. Measured on a 40-episode batch, a hard-coded 149 captured 2.94 s of
# every 5.00 s motion -- 59% -- cutting off exactly the distinctive second half of the
# composite behaviours ("walks, then squats to pick up", "walks, pauses and looks around,
# then continues"). One extra step past the pass gives the exporter the single complete
# pass it requires without starting a second one.
if [[ "$MAX_STEPS" == "auto" ]]; then
  MAX_STEPS=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$MOTION" <<'PY'
import sys

import joblib

library = joblib.load(sys.argv[1])
entry = next(iter(library.values()))
frames = len(entry["root_trans_offset"])
source_fps = float(entry["fps"])
# The environment steps at 50 Hz regardless of the source motion's frame rate, and the
# recorder writes max_render_steps - 1 frames. Asking for exactly the pass length therefore
# stops one frame short of the end rather than one frame past it -- which matters, because
# one frame past is the first frame of the *next* pass, and every capture then contains a
# reset that has to be split back out.
print(int(round(frames / source_fps * 50)))
PY
  ) || { echo "ERROR: could not read motion length from $MOTION" >&2; exit 2; }
  echo "max_steps=auto -> $MAX_STEPS (one full pass)"
fi

echo "scene=$SCENE"
echo "scene_usd=$SCENE_USD"
echo "motion=$MOTION"
echo "checkpoint=$CHECKPOINT"
echo "out=$OUT"

# PYTHONPATH is REPLACED (not extended) with this repository, for two reasons:
#   1. The shell profile puts ROS's Python 3.10 site-packages on PYTHONPATH, which
#      shadows pinocchio inside this 3.11 environment.
#   2. The `gear_sonic` editable install in env_isaaclab resolves to a DIFFERENT
#      checkout (groot-wbc-sonic-sim-trackb), and its static editable finder does not
#      know about subpackages added after install (e.g. dataset_generation). The
#      script's own `sys.path.append(os.getcwd())` appends too late to win. Without
#      this line the rollout silently runs another checkout's code.
# motion.debug_vis defaults to TRUE, which draws the reference-motion goal markers
# (yellow spheres, diffuse_color=(1,1,0)) as real scene prims. The ego TiledCamera
# renders them, so the recorded observation contains the tracking target itself --
# a train/deploy visual mismatch AND goal leakage that lets a VLA follow the markers
# instead of grounding language and scene. Recording a dataset requires it off.
#
# The release checkpoint ships three encoders (g1/teleop/smpl) and samples between them.
# A Kimodo G1-qpos reference must be tracked by the g1 encoder, so +use_encoder=g1 below
# pins it. Leaving it unpinned silently leaves encoder choice to sampling, and the export
# binding gate rejects the result.
cd "$REPO_ROOT" || exit 2
RESOLVED=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c 'import gear_sonic; print(gear_sonic.__file__)' 2>/dev/null)
if [[ "$RESOLVED" != "$REPO_ROOT/gear_sonic/__init__.py" ]]; then
  echo "ERROR: gear_sonic resolves to '$RESOLVED', expected '$REPO_ROOT/gear_sonic/__init__.py'" >&2
  exit 2
fi
echo "gear_sonic resolves to: $RESOLVED"

# Hydra reads an unquoted comma in an override value as a list separator, so a natural
# task string ("crouch down low, then stand up and walk forward") aborts the run with
# "Ambiguous value for argument". Shell quoting does not help -- the quotes must survive
# into the argument Hydra itself parses. Single quotes inside the value are escaped for
# the same reason. Almost every task-language string has a comma in it, so this is the
# normal case rather than an edge one.
TASK_OVERRIDE="'${TASK//\'/\\\'}'"

env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" gear_sonic/eval_agent_trl.py \
  +checkpoint="$CHECKPOINT" \
  +headless=True \
  ++num_envs=1 \
  "${TERRAIN_OVERRIDES[@]}" \
  ++manager_env.config.render_ego=True \
  ++manager_env.config.save_rendering_dir="$OUT/renders" \
  ++manager_env.config.save_trajectory_dir="$OUT/trajectories" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION" \
  ++manager_env.config.render_frame_skip=1 \
  ++manager_env.commands.motion.debug_vis=False \
  ++dataset_scene_id="$SCENE" \
  ++dataset_task="$TASK_OVERRIDE" \
  +use_encoder=g1 \
  "${EXTRA_OVERRIDES[@]}" \
  ++max_render_steps="$MAX_STEPS" \
  +success_manifest="$OUT/success_manifest.json" \
  "~manager_env/recorders=empty" "+manager_env/recorders=dataset" \
  > "$LOG" 2>&1
STATUS=$?

echo "process exit status: $STATUS (not authoritative)"

if ! grep -q "SONIC_EVAL_SUCCESS" "$LOG"; then
  echo "FAIL: no SONIC_EVAL_SUCCESS marker in $LOG"
  echo "--- last 40 log lines ---"
  tail -40 "$LOG"
  exit 1
fi

TRAJ_COUNT=$(find "$OUT/trajectories" -name "*.trajectory.pkl" | wc -l)
RENDER_COUNT=$(find "$OUT/renders" -name "*.mp4" | wc -l)
if [[ "$TRAJ_COUNT" -eq 0 ]]; then
  echo "FAIL: SONIC_EVAL_SUCCESS present but no trajectory artifact was written"
  exit 1
fi
if [[ ! -f "$OUT/success_manifest.json" ]]; then
  echo "FAIL: SONIC_EVAL_SUCCESS present but no runtime success manifest was written"
  exit 1
fi

# The dataset exporter requires exactly 50 Hz. render_frame_skip defaults to 2,
# which silently records at 25 Hz and produces un-exportable trajectories, so
# check the recorded rate rather than trusting the override took effect.
env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$OUT" <<'PYCHECK'
import pickle, sys
from pathlib import Path

out = Path(sys.argv[1])
paths = sorted(out.glob("trajectories/*.trajectory.pkl"))
if not paths:
    raise SystemExit("no trajectory pickle found")
for path in paths:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    fps = float(payload.get("fps", 0.0))
    frames = payload.get("total_frames")
    print(f"{path.name}: fps={fps:g} frames={frames}")
    if abs(fps - 50.0) > 1e-6:
        raise SystemExit(f"FAIL: recorded at {fps:g} Hz; the exporter requires 50 Hz")
PYCHECK
if [[ $? -ne 0 ]]; then
  echo "FAIL: recorded trajectory did not meet the 50 Hz dataset contract"
  exit 1
fi

grep "SONIC_EVAL_SUCCESS" "$LOG" | tail -1
echo "trajectory artifacts: $TRAJ_COUNT"
echo "render artifacts: $RENDER_COUNT"
echo "PASS"
