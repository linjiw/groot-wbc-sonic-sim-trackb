#!/bin/bash
# Stall detector for the GPU experiment queue.
# Exits (triggering a notification) if anything looks wrong:
#  - a train.py is running but its output jsonl hasn't grown in 25 min (hang)
#  - no train.py is running AND the queue scripts are gone (queue died early)
# Exits 0 quietly when the entire queue (efficiency + final sweep) completes.
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while true; do
  sleep 300
  queue_alive=$(pgrep -f "sweep_efficiency.sh|sweep_final.sh" | wc -l)
  train_alive=$(pgrep -f "train.py --teacher" | wc -l)
  if [ "$queue_alive" -eq 0 ] && [ "$train_alive" -eq 0 ]; then
    echo "QUEUE COMPLETE at $(date): all sweeps finished."
    ls -la efficiency.json long_falp_hsdense_s0.jsonl matched_falp_p4_hsdense_s0.jsonl 2>/dev/null
    exit 0
  fi
  if [ "$train_alive" -gt 0 ]; then
    newest=$(ls -t *.jsonl 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
      if [ "$age" -gt 1500 ]; then
        echo "STALL DETECTED at $(date): train.py running but $newest untouched for ${age}s"
        ps aux | grep train.py | grep -v grep | tr -s ' ' | cut -d' ' -f13-20
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
        exit 1
      fi
    fi
  elif [ "$queue_alive" -gt 0 ]; then
    : # between configs (queue script sleeping/waiting) — fine
  fi
done
