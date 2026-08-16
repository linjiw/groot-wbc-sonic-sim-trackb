#!/usr/bin/env bash
# Export every accepted rollout in a placement batch to a LeRobot synthetic_g1 dataset.
#
# Only rollouts that pass the locomotion acceptance gate are exported; rejected ones
# are listed so the quarantine set stays visible rather than silently dropped.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/research/export_kimodo_dataset_batch.sh \
    --work-dir /path/workdir \
    --placements /path/placements.json \
    --kimodo-demos /path/kimodo-g1-rp \
    --csv-dir /path/with/kimodo_*.csv \
    --out-dir /path/datasets \
    [--python /abs/isaac/python] [--export-python /abs/lerobot/python]

`lerobot` is not installed in the Isaac environment, so conversion/provenance use
--python and the LeRobot write uses --export-python.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=""; PLACEMENTS=""; DEMOS=""; CSV_DIR=""; OUT_DIR=""
PYTHON_BIN="${PYTHON:-${HOME}/miniconda3/envs/env_isaaclab/bin/python}"
EXPORT_PYTHON="${EXPORT_PYTHON:-${REPO_ROOT}/.venv_data_collection/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK="$2"; shift 2 ;;
    --placements) PLACEMENTS="$2"; shift 2 ;;
    --kimodo-demos) DEMOS="$2"; shift 2 ;;
    --csv-dir) CSV_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --export-python) EXPORT_PYTHON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
for v in WORK PLACEMENTS DEMOS CSV_DIR OUT_DIR; do
  if [[ -z "${!v}" ]]; then echo "ERROR: --${v,,} is required" >&2; usage; exit 2; fi
done
cd "$REPO_ROOT" || exit 2
mkdir -p "$OUT_DIR" "$WORK/provenance"

SCENE_MANIFEST="gear_sonic/data/assets/scenes/g1_dataset/manifest.json"
accepted=0; rejected=0; exported=0; failed=0
REJECTED_LIST=""

for manifest in "$WORK"/rollouts/*/success_manifest.json; do
  [[ -e "$manifest" ]] || continue
  rollout_dir="$(dirname "$manifest")"
  config_id="$(basename "$rollout_dir")"
  traj="$rollout_dir/trajectories/000000.trajectory.pkl"
  video="$rollout_dir/renders/000000.mp4"
  [[ -f "$traj" && -f "$video" ]] || { echo "SKIP     $config_id (missing artifacts)"; continue; }

  # Gate first: only accepted physics is worth the export cost.
  verdict=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$traj" <<'PYEOF'
import pickle, sys
from gear_sonic.dataset_generation.trajectory_acceptance import evaluate_locomotion_trajectory
from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory
payload = pickle.load(open(sys.argv[1], "rb"))
raw = validate_sonic_trajectory(payload)
if not raw.ok:
    print("REJECT raw:" + ";".join(raw.errors)); raise SystemExit(0)
report = evaluate_locomotion_trajectory(payload)
print("ACCEPT" if report.accepted else "REJECT " + ",".join(report.rejection_reasons))
PYEOF
)
  if [[ "$verdict" != ACCEPT* ]]; then
    echo "REJECT   $config_id  ${verdict#REJECT }"
    rejected=$((rejected+1)); REJECTED_LIST+="$config_id ${verdict#REJECT }"$'\n'; continue
  fi
  accepted=$((accepted+1))

  # config_id is <motion>__<scene>__<placement> for placement batches, but generated
  # clutter scenes use their own ids, so prefer the motion recorded in the manifest.
  motion_id=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c "
import json,sys
cfg=[e for e in json.load(open('$PLACEMENTS'))['configs'] if e['config_id']=='$config_id']
print(cfg[0].get('motion_id','') if cfg else '')" 2>/dev/null)
  [[ -n "$motion_id" ]] || motion_id="${config_id%%__*}"
  scene_id=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c "
import json,sys; print(json.load(open('$manifest'))['capture_context']['scene_id'])")
  # Scenes may come from any package under assets/scenes, so search them all for the
  # content hash rather than assuming the hand-authored manifest.
  scene_hash=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c "
import glob, json, sys
for path in glob.glob('gear_sonic/data/assets/scenes/*/manifest.json'):
    for entry in json.load(open(path))['scenes']:
        if entry['scene_id'] == '$scene_id':
            print(entry['sha256']); sys.exit(0)
sys.exit('scene $scene_id not found in any manifest')")
  task=$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c "
import json; print(json.load(open('$manifest'))['capture_context']['task'])")

  prov="$WORK/provenance/$config_id"
  # The rollout's motion was converted from values the batch runner formatted to
  # fixed precision (%.6f start, %.9f yaw). Re-deriving with full float repr would
  # produce a different transform and a different hash, so format identically here.
  read -r start_x start_y yaw end_x end_y <<< "$(env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - \
      "$PLACEMENTS" "$config_id" <<'PYEOF'
import json, sys
config = [
    entry for entry in json.load(open(sys.argv[1]))["configs"]
    if entry["config_id"] == sys.argv[2]
][0]
placement = config["placement"]
print(
    f"{placement['start_xy'][0]:.6f}",
    f"{placement['start_xy'][1]:.6f}",
    f"{placement['yaw_rad']:.9f}",
    f"{placement['end_xy'][0]:.6f}",
    f"{placement['end_xy'][1]:.6f}",
)
PYEOF
)"

  # route_xy[0] must equal the ConversionResult scene start exactly; the exporter
  # binds them. Passing the same formatted literal to both keeps them identical.
  if ! env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      scripts/research/build_kimodo_provenance_bundle.py \
      --rollout-dir "$rollout_dir" --demo-meta "$DEMOS/$motion_id/meta.json" \
      --motion-npz "$DEMOS/$motion_id/motion.npz" --qpos-csv "$CSV_DIR/kimodo_${motion_id}.csv" \
      --route-xy "$start_x" "$start_y" "$end_x" "$end_y" \
      --out-dir "$prov" > "$prov.bundle.log" 2>&1; then
    echo "BUNDLFAIL $config_id"; failed=$((failed+1)); continue
  fi

  if ! env PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      gear_sonic/data_process/convert_kimodo_to_motion_lib.py \
      --input "$CSV_DIR/kimodo_${motion_id}.csv" --output "$prov/motion.pkl" \
      --motion-key "$config_id" --source-fps 30 \
      --scene-start "$start_x" "$start_y" 0.0 --scene-yaw "$yaw" \
      --generation-result "$prov/generation_result.json" \
      --source-artifact-name kimodo_qpos_csv \
      --manifest "$prov/conversion_result.json" >> "$prov.bundle.log" 2>&1; then
    echo "CONVFAIL  $config_id"; failed=$((failed+1)); continue
  fi

  # Process status is NOT authoritative: the exporter reliably writes a complete,
  # valid dataset and then aborts during interpreter teardown (a PyAV/torch/OpenCV
  # shutdown crash in these environments). Success is the explicit marker plus the
  # dataset actually being on disk -- the same rule the rollout runner uses for
  # eval_agent_trl's os._exit(0).
  export_log="$WORK/provenance/$config_id.export.log"
  env PYTHONPATH="$REPO_ROOT" "$EXPORT_PYTHON" scripts/research/export_sonic_trajectory.py \
      "$traj" "$video" "$OUT_DIR/$config_id" \
      --task "$task" --camera-provenance isaac_sim \
      --runtime-success-manifest "$manifest" \
      --scene-id "$scene_id" --scene-hash "$scene_hash" \
      --upstream-provenance-json "$prov/conversion_result.json" \
      --overwrite-existing > "$export_log" 2>&1
  if grep -q "SONIC_LEROBOT_EXPORT_SUCCESS" "$export_log" \
     && [[ -f "$OUT_DIR/$config_id/meta/info.json" ]]; then
    frames=$(grep -o "frames=[0-9]*" "$export_log" | tail -1)
    echo "EXPORTED $config_id  ($frames)"; exported=$((exported+1))
  else
    echo "EXPFAIL  $config_id -> $export_log"
    grep -E "Error|error|Exception" "$export_log" | tail -2 | sed 's/^/           /'
    failed=$((failed+1))
  fi
done

echo
echo "accepted=$accepted rejected=$rejected exported=$exported failed=$failed"
if [[ -n "$REJECTED_LIST" ]]; then
  printf '%s' "$REJECTED_LIST" > "$WORK/quarantine.txt"
  echo "quarantine reasons written to $WORK/quarantine.txt"
fi
echo EXPORT_BATCH_DONE
