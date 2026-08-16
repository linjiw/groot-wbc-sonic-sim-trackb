#!/usr/bin/env bash
# Convert and roll out every config in a placements.json produced by
# scripts/research/plan_kimodo_placements.py.
#
# Each config carries a planned (start_xy, yaw) whose reference path clears the
# scene's obstacles, and a max_render_steps that captures exactly one motion pass.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/research/run_kimodo_placement_batch.sh \
    --placements /path/placements.json \
    --work-dir /path/workdir \
    [--limit N] [--python /abs/path/python]

Writes <work-dir>/motions/<config_id>.pkl and <work-dir>/rollouts/<config_id>/.
Already-completed configs (those with a success_manifest.json) are skipped.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLACEMENTS=""
WORK=""
LIMIT=0
PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/env_isaaclab/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --placements) PLACEMENTS="$2"; shift 2 ;;
    --work-dir) WORK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ -z "$PLACEMENTS" || -z "$WORK" ]]; then usage; exit 2; fi
cd "$REPO_ROOT" || exit 2
mkdir -p "$WORK/motions" "$WORK/rollouts"

# One TSV line per config keeps the shell loop free of JSON parsing.
CONFIGS=$("$PYTHON_BIN" - "$PLACEMENTS" <<'PYEOF'
import json, sys
payload = json.load(open(sys.argv[1]))
for entry in payload["configs"]:
    placement = entry["placement"]
    print("\t".join([
        entry["config_id"], entry["scene_id"], entry["csv"],
        f"{placement['start_xy'][0]:.6f}", f"{placement['start_xy'][1]:.6f}",
        f"{placement['yaw_rad']:.9f}", str(entry["max_render_steps"]),
        f"{placement['clearance_m']:.3f}",
    ]))
PYEOF
)

declare -A TASKS=(
  [factory_aisle]="walk through the factory aisle"
  [household_room]="walk across the room"
)

total=0; ok=0; skipped=0; failed=0
while IFS=$'\t' read -r config_id scene csv sx sy yaw steps clearance; do
  [[ -z "$config_id" ]] && continue
  total=$((total+1))
  if [[ "$LIMIT" -gt 0 && "$total" -gt "$LIMIT" ]]; then break; fi
  out="$WORK/rollouts/$config_id"
  if [[ -f "$out/success_manifest.json" ]]; then
    echo "SKIP     $config_id (already complete)"; skipped=$((skipped+1)); continue
  fi
  motion="$WORK/motions/${config_id}.pkl"
  if ! env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        gear_sonic/data_process/convert_kimodo_to_motion_lib.py \
        --input "$csv" --output "$motion" --motion-key "$config_id" --source-fps 30 \
        --scene-start "$sx" "$sy" 0.0 --scene-yaw "$yaw" \
        > "$WORK/rollouts/${config_id}.convert.log" 2>&1; then
    echo "CONVFAIL $config_id"; failed=$((failed+1)); continue
  fi
  scripts/research/run_kimodo_sonic_rollout.sh \
    --scene "$scene" --motion "$motion" --out "$out" \
    --max-steps "$steps" --task "${TASKS[$scene]}" \
    > "$WORK/rollouts/${config_id}.runner.log" 2>&1
  if grep -q "^PASS" "$WORK/rollouts/${config_id}.runner.log"; then
    echo "RUN-OK   $config_id  (clearance ${clearance} m)"; ok=$((ok+1))
  else
    echo "RUN-FAIL $config_id"; failed=$((failed+1))
  fi
done <<< "$CONFIGS"

echo
echo "configs=$total ran_ok=$ok skipped=$skipped failed=$failed"
echo PLACEMENT_BATCH_DONE
