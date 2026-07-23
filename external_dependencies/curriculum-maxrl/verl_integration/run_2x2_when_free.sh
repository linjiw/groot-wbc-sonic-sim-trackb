#!/bin/bash
# Wait for the GPU (cosmos SFT holds it), then run the 2x2 grid sequentially.
# Cell order puts the two most informative arms first.
cd "$(dirname "$0")/.."
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free_mb" -gt 18000 ] && break
  sleep 300
done
echo "GPU free (${free_mb}MB) at $(date) — starting 2x2"
for cell in "maxrl true" "maxrl false" "grpo true" "grpo false"; do
  set -- $cell
  echo "=== CELL: estimator=$1 curriculum=$2 ($(date)) ==="
  ESTIMATOR=$1 CURRICULUM=$2 bash smollm/smollm_a10g.sh 2>&1 \
    | grep -E "step:|val|Curriculum|curriculum/|Error|error|Traceback" | tail -200
done
echo "2x2 DONE $(date)"
