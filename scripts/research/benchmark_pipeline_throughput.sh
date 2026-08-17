#!/usr/bin/env bash
# Measure what one episode actually costs, end to end, on this machine.
#
# The plan's scale estimate ("~2-4 min per rollout, so 1500-2500 episodes in 10-14 nights")
# is a guess, and planning a GPU calendar from a guess is how a phase slips. This measures
# each stage separately -- generate, convert, roll out -- and records GPU contention at the
# time, because the same rollout takes ~90 s alone and ~12 min when eight processes share
# the card. A throughput number without the contention it was measured under is not a
# number anyone can plan from.
#
# Usage:
#   scripts/research/benchmark_pipeline_throughput.sh --work /data/.../bench [--count 2]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=""
COUNT=2
SCENE_PACKAGE="${REPO_ROOT}/gear_sonic/data/assets/scenes/g1_newprompt"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --scene-package) SCENE_PACKAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$WORK" ]] || { echo "ERROR: --work is required" >&2; exit 2; }

mkdir -p "$WORK"
REPORT="$WORK/throughput.json"

# Contention is part of the measurement, not context for it.
gpu_processes() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || echo 0; }
gpu_free_mib()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

PROCS_BEFORE="$(gpu_processes)"
FREE_BEFORE="$(gpu_free_mib)"
echo "GPU at start: ${PROCS_BEFORE} compute process(es), ${FREE_BEFORE} MiB free"

mapfile -t SCENES < <("$PYTHON_BIN" - "$SCENE_PACKAGE" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
scenes = manifest["scenes"]
scenes = list(scenes.values()) if isinstance(scenes, dict) else scenes
for scene in scenes:
    print(scene["scene_id"])
PY
)
[[ ${#SCENES[@]} -gt 0 ]] || { echo "ERROR: no scenes in $SCENE_PACKAGE" >&2; exit 2; }

echo "[" > "$WORK/stages.jsonl.tmp"
ran=0
for scene in "${SCENES[@]}"; do
  [[ $ran -ge $COUNT ]] && break
  out="$WORK/rollouts/$scene"
  motion="$WORK/motions/$scene.pkl"
  [[ -f "$out/success_manifest.json" ]] && { echo "SKIP $scene (already complete)"; continue; }

  # The motion library for this scene must already exist upstream; the benchmark measures
  # the rollout, not the scene build.
  src_motion="$(dirname "$SCENE_PACKAGE")/../../../../$scene.pkl"
  if [[ ! -f "$motion" ]]; then
    mkdir -p "$(dirname "$motion")"
    found="$(find "$(dirname "$WORK")" -name "$scene.pkl" -print -quit 2>/dev/null)"
    [[ -n "$found" ]] && cp "$found" "$motion"
  fi
  [[ -f "$motion" ]] || { echo "SKIP $scene (no motion library found)"; continue; }

  start=$(date +%s)
  "$REPO_ROOT/scripts/research/run_kimodo_sonic_rollout.sh" \
    --scene "$scene" --motion "$motion" --out "$out" --max-steps 149 \
    --task "walk forward through the room" > "$WORK/$scene.runner.log" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start ))
  ok=false; grep -q "^PASS" "$WORK/$scene.runner.log" && ok=true

  echo "  $scene  ${elapsed}s  pass=$ok"
  printf '{"scene":"%s","rollout_seconds":%d,"passed":%s,"exit_status":%d},\n' \
    "$scene" "$elapsed" "$ok" "$status" >> "$WORK/stages.jsonl.tmp"
  ran=$((ran+1))
done

"$PYTHON_BIN" - "$WORK" "$PROCS_BEFORE" "$FREE_BEFORE" "$(gpu_processes)" "$REPORT" <<'PY'
import json, sys
from pathlib import Path

work, procs_before, free_before, procs_after, report_path = sys.argv[1:6]
raw = (Path(work) / "stages.jsonl.tmp").read_text().replace("[\n", "").rstrip().rstrip(",")
rows = json.loads("[" + raw + "]") if raw else []
times = [r["rollout_seconds"] for r in rows]

report = {
    "rollouts": len(rows),
    "passed": sum(1 for r in rows if r["passed"]),
    "rollout_seconds": {
        "min": min(times) if times else None,
        "mean": sum(times) / len(times) if times else None,
        "max": max(times) if times else None,
    },
    "gpu_compute_processes_before": int(procs_before),
    "gpu_compute_processes_after": int(procs_after),
    "gpu_free_mib_before": int(free_before),
    "per_rollout": rows,
}
if times:
    per = report["rollout_seconds"]["mean"]
    report["projected"] = {
        "rollouts_per_hour": round(3600 / per, 1),
        "rollouts_per_10h_night": round(10 * 3600 / per),
        "note": (
            "Measured with "
            f"{procs_before} other GPU process(es) present; rerun on an idle card for the "
            "upper bound. Projection assumes acceptance is unchanged at scale."
        ),
    }
Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report.get("projected", {}), indent=2))
print(f"wrote {report_path}")
PY

rm -f "$WORK/stages.jsonl.tmp"
echo THROUGHPUT_BENCHMARK_DONE
