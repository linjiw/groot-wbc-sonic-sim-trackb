#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the G0/G0B CG-WBC golden-path proof for one tiny SONIC/VLA dataset.

Usage:
  scripts/research/run_g0_golden_path.sh \
    --dataset /data/g1_fetch_place_tiny \
    --observation-config gear_sonic_deploy/policy/release/observation_config.yaml \
    --curriculum-config configs/research/curriculum_graph_fetch_place.yaml \
    --mvp-config configs/research/fetch_place.yaml \
    --report-dir outputs/research/g0_g1_fetch_place_tiny \
    [--groot-repo /path/to/Isaac-GR00T] \
    [--model-path /path/to/checkpoint]

The runner is intentionally non-destructive. It validates data, builds the
curriculum manifest, writes stage-wise reports, records open-loop pass/skip/fail
honestly, runs the MVP eval wrapper, and updates curriculum state. It does not
modify SONIC runtime, ZMQ, observation ordering, or the action interface.
EOF
}

DATASET=""
OBS_CONFIG=""
CURRICULUM_CONFIG=""
MVP_CONFIG=""
REPORT_DIR=""
GROOT_REPO=""
MODEL_PATH=""
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --observation-config|--obs-config)
      OBS_CONFIG="$2"
      shift 2
      ;;
    --curriculum-config)
      CURRICULUM_CONFIG="$2"
      shift 2
      ;;
    --mvp-config)
      MVP_CONFIG="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    --groot-repo)
      GROOT_REPO="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
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

if [[ -z "$DATASET" || -z "$OBS_CONFIG" || -z "$CURRICULUM_CONFIG" || -z "$MVP_CONFIG" || -z "$REPORT_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"

status_json() {
  local path="$1"
  local status="$2"
  local step="$3"
  local detail="${4:-}"
  "$PYTHON_BIN" - "$path" "$status" "$step" "$detail" <<'PY'
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
record = {
    "status": sys.argv[2],
    "step": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "[G0] checking G1 SONIC schema"
"$PYTHON_BIN" scripts/research/check_g1_sonic_schema.py \
  --observation-config "$OBS_CONFIG" \
  | tee "$REPORT_DIR/schema_check.log"

echo "[G0] validating dataset"
"$PYTHON_BIN" scripts/research/check_sonic_vla_dataset.py \
  --dataset-path "$DATASET" \
  --json "$REPORT_DIR/dataset_check.json" \
  | tee "$REPORT_DIR/dataset_check.log"

echo "[G0] building curriculum manifest"
"$PYTHON_BIN" scripts/research/build_curriculum_manifest.py \
  --dataset-path "$DATASET" \
  --config "$CURRICULUM_CONFIG" \
  --output "$REPORT_DIR/manifest.jsonl"

echo "[G0] summarizing manifest"
"$PYTHON_BIN" scripts/research/summarize_curriculum_eval.py \
  --manifest "$REPORT_DIR/manifest.jsonl" \
  --config "$CURRICULUM_CONFIG" \
  --output-json "$REPORT_DIR/manifest_summary.json" \
  --output-md "$REPORT_DIR/manifest_summary.md"

mkdir -p "$REPORT_DIR/open_loop"
if [[ -n "$GROOT_REPO" && -n "$MODEL_PATH" ]]; then
  echo "[G0] running open-loop wrapper"
  set +e
  bash scripts/research/open_loop_groot_eval.sh \
    --groot-repo "$GROOT_REPO" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET" \
    > "$REPORT_DIR/open_loop/log.txt" 2>&1
  OPEN_LOOP_RC=$?
  set -e
  if [[ "$OPEN_LOOP_RC" -eq 0 ]]; then
    status_json "$REPORT_DIR/open_loop/status.json" "passed" "open_loop" "wrapper validated inputs"
  else
    status_json "$REPORT_DIR/open_loop/status.json" "failed" "open_loop" "see open_loop/log.txt"
  fi
else
  echo "[G0] skipping open-loop wrapper: --groot-repo and --model-path were not provided"
  status_json "$REPORT_DIR/open_loop/status.json" "skipped_missing_groot_inputs" "open_loop" "provide --groot-repo and --model-path to validate the GR00T eval boundary"
fi
cp "$REPORT_DIR/open_loop/status.json" "$REPORT_DIR/open_loop_status.json"

echo "[G0] running MVP eval matrix"
bash scripts/research/run_mvp_eval_matrix.sh \
  --dataset "$DATASET" \
  --obs-config "$OBS_CONFIG" \
  --manifest "$REPORT_DIR/manifest.jsonl" \
  --config "$MVP_CONFIG" \
  --curriculum-config "$CURRICULUM_CONFIG" \
  --report-dir "$REPORT_DIR/mvp_eval"

echo "[G0] writing initial curriculum state"
"$PYTHON_BIN" - "$CURRICULUM_CONFIG" "$REPORT_DIR/curriculum_state.initial.json" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
import yaml
from scripts.research.update_curriculum_state import initial_state
config = Path(sys.argv[1])
out = Path(sys.argv[2])
graph = yaml.safe_load(config.read_text(encoding="utf-8"))["curriculum"]
out.write_text(json.dumps(initial_state(graph), indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[G0] updating curriculum state"
"$PYTHON_BIN" scripts/research/update_curriculum_state.py \
  --config "$CURRICULUM_CONFIG" \
  --metrics "$REPORT_DIR/mvp_eval/report.json" \
  --state-in "$REPORT_DIR/curriculum_state.initial.json" \
  --state-out "$REPORT_DIR/curriculum_state.updated.json"

echo "[G0] writing final G0 report"
"$PYTHON_BIN" - "$REPORT_DIR" "$DATASET" "$OBS_CONFIG" "$CURRICULUM_CONFIG" "$MVP_CONFIG" <<'PY'
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
report_dir = Path(sys.argv[1])
dataset, obs_config, curriculum_config, mvp_config = sys.argv[2:6]

def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default

summary = load(report_dir / "manifest_summary.json", {"stage_metrics": {}})
dataset_check = load(report_dir / "dataset_check.json", {})
open_loop = load(report_dir / "open_loop_status.json", {})
state = load(report_dir / "curriculum_state.updated.json", {})
record = {
    "status": "complete",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "dataset": dataset,
    "observation_config": obs_config,
    "curriculum_config": curriculum_config,
    "mvp_config": mvp_config,
    "dataset_check": dataset_check,
    "open_loop_status": open_loop,
    "stage_metrics": summary.get("stage_metrics", {}),
    "artifacts": {
        "schema_check_log": "schema_check.log",
        "dataset_check_json": "dataset_check.json",
        "manifest": "manifest.jsonl",
        "manifest_summary_json": "manifest_summary.json",
        "manifest_summary_md": "manifest_summary.md",
        "mvp_report_json": "mvp_eval/report.json",
        "mvp_report_md": "mvp_eval/report.md",
        "curriculum_state_updated": "curriculum_state.updated.json",
    },
    "runtime_boundary": {
        "sonic_runtime_modified": False,
        "zmq_protocol_modified": False,
        "observation_ordering_modified": False,
        "action_interface_modified": False,
    },
    "unlocked_stages": [sid for sid, values in state.get("stages", {}).items() if values.get("unlocked")],
}
(report_dir / "g0_report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (report_dir / "g0_report.md").open("w", encoding="utf-8") as f:
    f.write("# G0 Real Dataset Golden-Path Report\n\n")
    f.write(f"- Status: `{record['status']}`\n")
    f.write(f"- Dataset: `{dataset}`\n")
    f.write(f"- Dataset validation ok: `{dataset_check.get('ok')}`\n")
    f.write(f"- Open-loop status: `{open_loop.get('status', 'unknown')}`\n")
    f.write("\n## Runtime boundary\n\n")
    for key, value in record["runtime_boundary"].items():
        f.write(f"- {key}: `{str(value).lower()}`\n")
    f.write("\n## Stage metrics\n\n")
    f.write("| Stage | Episodes | Success | Fall | Drop | Schema | Competence | Gates |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for stage, metrics in sorted(record["stage_metrics"].items()):
        comp = metrics.get("competence")
        comp_text = "" if comp is None else f"{comp:.3f}"
        f.write(
            f"| {stage} | {metrics.get('num_episodes', 0)} | "
            f"{metrics.get('success_rate', 0):.3f} | {metrics.get('fall_rate', 0):.3f} | "
            f"{metrics.get('drop_rate', 0):.3f} | {metrics.get('schema_valid_rate', 0):.3f} | "
            f"{comp_text} | {metrics.get('hard_gates_pass')} |\n"
        )
PY

echo "[G0] done: $REPORT_DIR"
