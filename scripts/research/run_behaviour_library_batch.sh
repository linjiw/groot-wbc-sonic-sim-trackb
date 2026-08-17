#!/usr/bin/env bash
# Take a directory of generated motions all the way to analysed rollouts, resumably.
#
# Phase B runs unattended overnight on a shared GPU, so this is built for a job that will
# be interrupted: every stage skips work whose success marker AND artifact already exist,
# and success is never inferred from an exit code -- Isaac can print a fatal traceback and
# still exit zero.
#
# Stages, in order:
#   1. screen    joint-limit saturation, before any GPU time is spent
#   2. scene     generate a room around each surviving motion
#   3. convert   motion CSV -> SONIC motion library
#   4. rollout   physics, ego render, trajectory recording
#   5. analyse   acceptance, contact decomposition, latent parity
#
# Usage:
#   scripts/research/run_behaviour_library_batch.sh \
#       --motions /data/.../taxonomy/motions --work /data/.../taxonomy/batch [--limit 20]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOTIONS=""
WORK=""
LIMIT=0
SEEDS=0
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
MJCF="${MJCF:-$HOME/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml}"
MAX_STEPS=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motions) MOTIONS="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --scene-package) SCENES_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for required in MOTIONS WORK; do
  [[ -n "${!required}" ]] || { echo "ERROR: --${required,,} is required" >&2; exit 2; }
done
[[ -d "$MOTIONS" ]] || { echo "ERROR: no such directory: $MOTIONS" >&2; exit 2; }

mkdir -p "$WORK"/{csv,motions,rollouts,logs}
SCENES_DIR="${SCENES_DIR:-$REPO_ROOT/gear_sonic/data/assets/scenes/g1_library}"

echo "=== stage 1: screen ==="
"$PYTHON_BIN" "$REPO_ROOT/scripts/research/screen_kimodo_motions.py" \
  --csv-dir "$MOTIONS" --mjcf "$MJCF" --json "$WORK/screen.json" \
  > "$WORK/logs/screen.log" 2>&1
tail -3 "$WORK/logs/screen.log"

# Copy only the motions that passed, under the kimodo_<name>.csv naming the scene builder
# expects. Copy rather than move: the generated library stays intact.
mapfile -t KEEP < <("$PYTHON_BIN" - "$WORK/screen.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
for record in report["records"]:
    if record["passed"]:
        print(record["csv"])
PY
)
echo "${#KEEP[@]} motion(s) passed screening"

names=()
count=0
for csv in "${KEEP[@]}"; do
  [[ "$LIMIT" -gt 0 && "$count" -ge "$LIMIT" ]] && break
  stem="${csv%.csv}"
  cp -n "$MOTIONS/$csv" "$WORK/csv/kimodo_${stem}.csv"
  names+=("$stem")
  count=$((count+1))
done
echo "selected ${#names[@]} motion(s) for rollout"
[[ ${#names[@]} -gt 0 ]] || { echo "nothing to do"; echo LIBRARY_BATCH_DONE; exit 0; }

echo
echo "=== stage 2: scenes ==="
if [[ ! -f "$SCENES_DIR/manifest.json" ]] || [[ ! -s "$SCENES_DIR/manifest.json" ]]; then
  env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    "$REPO_ROOT/scripts/research/build_clutter_scenes.py" \
    --csv-dir "$WORK/csv" --out "$SCENES_DIR" --motions "${names[@]}" --seeds "$SEEDS" \
    > "$WORK/logs/scenes.log" 2>&1
  tail -2 "$WORK/logs/scenes.log"
else
  echo "SKIP (manifest exists; delete $SCENES_DIR to rebuild)"
fi

echo
echo "=== stages 3-4: convert and roll out ==="
mapfile -t SCENE_ROWS < <(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$SCENES_DIR" <<'PY'
import json, re, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
scenes = manifest["scenes"]
scenes = list(scenes.values()) if isinstance(scenes, dict) else scenes
for scene in scenes:
    sid = scene["scene_id"]
    x, y = scene["generation"]["scene_start_xy"]
    # scene_id is clutter_<motion>_s<seed>; recover the motion to find its CSV.
    motion = re.sub(r"^clutter_|_s\d+$", "", sid)
    print(f"{sid}\t{motion}\t{x}\t{y}")
PY
)

ok=0; skipped=0; failed=0
for row in "${SCENE_ROWS[@]}"; do
  IFS=$'\t' read -r sid motion sx sy <<< "$row"
  out="$WORK/rollouts/$sid"
  if [[ -f "$out/success_manifest.json" ]]; then
    skipped=$((skipped+1)); continue
  fi
  motion_pkl="$WORK/motions/$sid.pkl"
  if [[ ! -f "$motion_pkl" ]]; then
    if ! env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
          "$REPO_ROOT/gear_sonic/data_process/convert_kimodo_to_motion_lib.py" \
          --input "$WORK/csv/kimodo_${motion}.csv" --output "$motion_pkl" \
          --motion-key "$sid" --source-fps 30 \
          --scene-start "$sx" "$sy" 0.0 --scene-yaw 0.0 \
          > "$WORK/logs/$sid.convert.log" 2>&1; then
      echo "CONVFAIL $sid"; failed=$((failed+1)); continue
    fi
  fi

  # Task language carries commas; run_kimodo_sonic_rollout.sh quotes it for Hydra.
  "$REPO_ROOT/scripts/research/run_kimodo_sonic_rollout.sh" \
    --scene "$sid" --motion "$motion_pkl" --out "$out" --max-steps "$MAX_STEPS" \
    --task "walk through the room, avoiding the furniture" \
    > "$WORK/logs/$sid.runner.log" 2>&1
  if grep -q "^PASS" "$WORK/logs/$sid.runner.log"; then
    echo "RUN-OK   $sid"; ok=$((ok+1))
  else
    echo "RUN-FAIL $sid"; failed=$((failed+1))
  fi
done
echo "rollouts: ran_ok=$ok skipped=$skipped failed=$failed"

echo
echo "=== stage 5: analyse ==="
env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  "$REPO_ROOT/scripts/research/report_behaviour_diversity.py" \
  --root "$WORK/rollouts" --json "$WORK/diversity.json" 2>&1 | tail -12 \
  || echo "(no complete rollouts to analyse yet)"

echo LIBRARY_BATCH_DONE
