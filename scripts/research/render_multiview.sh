#!/usr/bin/env bash
# Render one rollout from every available camera, one view per pass.
#
# Isaac's recorder writes a single camera per run, so each view is its own pass over
# the same deterministic motion; the frames line up because the physics is identical.
# Views: ego (head), wrist (left/right), chase (third person), overhead (top-down).
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/research/render_multiview.sh --scene SCENE --motion /abs/motion.pkl \
      --out /abs/out_dir [--max-steps N] [--views ego,chase,overhead,wrist]
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE=""; MOTION=""; OUT=""; MAX_STEPS=250; VIEWS="ego,chase,overhead,wrist"
PY="${PYTHON:-${HOME}/miniconda3/envs/env_isaaclab/bin/python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --views) VIEWS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$SCENE" && -n "$MOTION" && -n "$OUT" ]] || { usage; exit 2; }
cd "$REPO_ROOT" || exit 2

SCENE_USD=""
for pkg in "$REPO_ROOT"/gear_sonic/data/assets/scenes/*/; do
  [[ -e "${pkg}${SCENE}.usda" ]] && { SCENE_USD="${pkg}${SCENE}.usda"; break; }
done
[[ -n "$SCENE_USD" ]] || { echo "ERROR: scene $SCENE not found" >&2; exit 2; }

run_view() {
  local view="$1"; shift
  local dir="$OUT/$view"; mkdir -p "$dir"
  env PYTHONPATH="$REPO_ROOT" "$PY" gear_sonic/eval_agent_trl.py \
    +checkpoint="$REPO_ROOT/sonic_release/last.pt" +headless=True ++num_envs=1 \
    ++manager_env.config.terrain_type=scene_usd \
    ++manager_env.config.scene_usd_path="$SCENE_USD" \
    ++manager_env.config.save_rendering_dir="$dir" \
    ++manager_env.config.save_trajectory_dir="$dir/trajectories" \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION" \
    ++manager_env.config.render_frame_skip=1 \
    ++manager_env.commands.motion.debug_vis=False \
    ++manager_env.recorders.dataset_export_mode=0 \
    +use_encoder=g1 ++max_render_steps="$MAX_STEPS" \
    "$@" > "$dir/render.log" 2>&1
  local n; n=$(find "$dir" -name "*.mp4" | wc -l)
  if [[ "$n" -gt 0 ]]; then echo "  OK   $view -> $(find "$dir" -name '*.mp4' | head -1)"
  else echo "  FAIL $view (see $dir/render.log)"; fi
}

IFS=',' read -ra WANTED <<< "$VIEWS"
for view in "${WANTED[@]}"; do
  case "$view" in
    ego)
      run_view ego ++manager_env.config.render_ego=True \
        "~manager_env/recorders=empty" "+manager_env/recorders=dataset" ;;
    chase)
      # Raised and pulled back: the default [2,2,1] offset clips inside furniture in
      # dense rooms and blanks part of the video.
      run_view chase ++manager_env.config.render_results=True \
        ++manager_env.config.render_width=1280 ++manager_env.config.render_height=720 \
        "++manager_env.config.eval_camera_offset=[3.0,3.0,2.5]" \
        "~manager_env/recorders=empty" "+manager_env/recorders=render" ;;
    overhead)
      # KNOWN BROKEN: renders 249 blank frames. Kept so the failure is reproducible;
      # use the matplotlib top-down plots for overhead review in the meantime.
      # The recorder defaults to eval_camera; enabling overview_camera is not enough,
      # it must also be selected by name or the pass silently re-renders the chase view.
      run_view overhead ++manager_env.config.render_results=True \
        ++manager_env.config.overview_camera=True ++manager_env.config.group_camera=True \
        ++manager_env.config.render_width=1280 ++manager_env.config.render_height=1280 \
        "~manager_env/recorders=empty" "+manager_env/recorders=render" \
        ++manager_env.recorders.render_envs.camera_name=overview_camera \
        ++manager_env.recorders.render_envs.track_root=False ;;
    wrist)
      run_view wrist ++manager_env.config.render_ego=True \
        ++manager_env.config.cameras.wrist_cameras=True \
        "~manager_env/recorders=empty" "+manager_env/recorders=render" \
        ++manager_env.recorders.render_envs.camera_name=left_wrist_camera \
        ++manager_env.recorders.render_envs.track_root=False ;;
    *) echo "  skip unknown view: $view" ;;
  esac
done
echo MULTIVIEW_DONE
