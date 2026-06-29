#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the non-destructive MVP validation matrix for a GR00T-SONIC dataset.

Usage:
  scripts/research/run_mvp_eval_matrix.sh \
    --dataset outputs/g1_red_cup_to_tray_cleaned \
    [--obs-config gear_sonic_deploy/policy/release/observation_config.yaml] \
    [--manifest outputs/research/g0/manifest.jsonl] \
    [--config configs/research/fetch_place.yaml] \
    [--curriculum-config configs/research/curriculum_graph_fetch_place.yaml] \
    [--report-dir reports/g1_red_cup_to_tray]

When --manifest and --curriculum-config are supplied, report.json contains
stage_metrics suitable for scripts/research/update_curriculum_state.py.
EOF
}

DATASET=""
OBS_CONFIG="gear_sonic_deploy/policy/release/observation_config.yaml"
MANIFEST=""
MVP_CONFIG=""
CURRICULUM_CONFIG=""
REPORT_DIR="reports/g1_red_cup_to_tray"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --obs-config|--observation-config)
      OBS_CONFIG="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --config|--mvp-config)
      MVP_CONFIG="$2"
      shift 2
      ;;
    --curriculum-config)
      CURRICULUM_CONFIG="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
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

if [[ -z "$DATASET" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/report.md"
LEGACY_REPORT="$REPORT_DIR/mvp_eval_matrix.md"
STATUS=0

run_section() {
  local title="$1"
  shift

  {
    echo
    echo "## $title"
    echo
    echo '```text'
  } >> "$REPORT"

  set +e
  "$@" >> "$REPORT" 2>&1
  local rc=$?
  set -e

  {
    echo '```'
    echo
    echo "Exit status: $rc"
  } >> "$REPORT"

  if [[ "$rc" -ne 0 ]]; then
    STATUS="$rc"
  fi
}

{
  echo "# MVP Eval Matrix"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- Git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "- Dataset: $DATASET"
  echo "- Observation config: $OBS_CONFIG"
  if [[ -n "$MVP_CONFIG" ]]; then
    echo "- MVP config: $MVP_CONFIG"
  fi
  if [[ -n "$CURRICULUM_CONFIG" ]]; then
    echo "- Curriculum config: $CURRICULUM_CONFIG"
  fi
  if [[ -n "$MANIFEST" ]]; then
    echo "- Manifest: $MANIFEST"
  fi
  echo
  echo "## Git Status"
  echo
  echo '```text'
  git status --short 2>/dev/null || true
  echo '```'
} > "$REPORT"

run_section "Schema Check" "$PYTHON_BIN" scripts/research/check_g1_sonic_schema.py --obs-config "$OBS_CONFIG"
run_section "Dataset Check" "$PYTHON_BIN" scripts/research/check_sonic_vla_dataset.py "$DATASET"

if [[ -n "$MANIFEST" && -n "$CURRICULUM_CONFIG" ]]; then
  run_section "Manifest Stage Summary" "$PYTHON_BIN" scripts/research/summarize_curriculum_eval.py \
    --manifest "$MANIFEST" \
    --config "$CURRICULUM_CONFIG" \
    --output-json "$REPORT_DIR/report.json" \
    --output-md "$REPORT_DIR/manifest_summary.md"
else
  "$PYTHON_BIN" - "$REPORT_DIR/report.json" "$DATASET" "$STATUS" <<'PY'
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
record = {
    "stage_metrics": {},
    "metadata": {
        "dataset": sys.argv[2],
        "status": int(sys.argv[3]),
        "note": "No manifest/curriculum config supplied; stage metrics were not generated.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    },
}
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

cat "$REPORT"
cp "$REPORT" "$LEGACY_REPORT"
echo "Wrote $REPORT"
echo "Wrote $REPORT_DIR/report.json"
exit "$STATUS"
