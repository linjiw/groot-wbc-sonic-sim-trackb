#!/bin/bash
# matched wall-clock protocol: 2400s RL time per config
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
for cfg in "uniform maxrl" "frontier maxrl" "learnability maxrl" "frontier_alp maxrl" "uniform grpo" "frontier grpo"; do
  set -- $cfg
  echo "=== $1 + $2 (2400s) ==="
  python3 train.py --teacher $1 --estimator $2 --steps 999999 --max-seconds 2400 \
    --eval-every 25 --seed 0 --lr 1e-4 --out matched_$1_$2_s0.jsonl 2>&1 | grep -vi warn | grep -E "^step|^FINAL|post"
done
echo "ALL DONE"
