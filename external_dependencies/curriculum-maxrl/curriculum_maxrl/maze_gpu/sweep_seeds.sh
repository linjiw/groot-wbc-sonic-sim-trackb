#!/bin/bash
# Multi-seed confirmation: 4 key configs x seeds 1,2 at matched 2400s.
# (SFT warmstart is built per-seed automatically by train.py on first run.)
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while pgrep -f "train.py --teacher" > /dev/null; do sleep 60; done
for seed in 1 2; do
  for cfg in "uniform maxrl" "frontier_alp maxrl" "uniform grpo"; do
    set -- $cfg
    echo "=== seed $seed: $1 + $2 (2400s) ==="
    python3 train.py --teacher $1 --estimator $2 --steps 999999 --max-seconds 2400 \
      --eval-every 25 --seed $seed --lr 1e-4 --out matched_$1_$2_s$seed.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
  done
  echo "=== seed $seed: frontier + maxrl --hindsight (2400s) ==="
  python3 train.py --teacher frontier --estimator maxrl --hindsight --steps 999999 --max-seconds 2400 \
    --eval-every 25 --seed $seed --lr 1e-4 --out matched_frontier_maxrl_hs_s$seed.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
done
echo "SEEDS DONE"
