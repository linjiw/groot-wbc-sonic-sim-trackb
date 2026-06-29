#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the non-destructive MVP validation matrix for a GR00T-SONIC dataset.

Usage:
  scripts/research/run_mvp_eval_matrix.sh \
    --dataset outputs/g1_red_cup_to_tray_cleaned \
    [--obs-config gear_sonic_deploy/policy/release/observation_config.yaml] \
    [--report-dir reports/g1_red_cup_to_tray]
EOF
}

DATASET=""
OBS_CONFIG="gear_sonic_deploy/policy/release/observation_config.yaml"
REPORT_DIR="reports/g1_red_cup_to_tray"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --obs-config)
      OBS_CONFIG="$2"
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
REPORT="$REPORT_DIR/mvp_eval_matrix.md"
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
  echo
  echo "## Git Status"
  echo
  echo '```text'
  git status --short 2>/dev/null || true
  echo '```'
} > "$REPORT"

run_section "Schema Check" "$PYTHON_BIN" scripts/research/check_g1_sonic_schema.py --obs-config "$OBS_CONFIG"
run_section "Dataset Check" "$PYTHON_BIN" scripts/research/check_sonic_vla_dataset.py "$DATASET"

cat "$REPORT"

echo "Wrote $REPORT"
exit "$STATUS"
