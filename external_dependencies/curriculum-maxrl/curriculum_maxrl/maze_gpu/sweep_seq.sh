#!/bin/bash
# wait for any existing train.py to finish (sweep A runs first)
while pgrep -f "train.py --teacher" > /dev/null; do sleep 30; done
for cfg in "uniform grpo" "frontier grpo" "frontier rloo" "frontier_alp maxrl"; do
  set -- $cfg
  echo "=== $1 + $2 ==="
  python3 train.py --teacher $1 --estimator $2 --steps 300 --eval-every 25 \
    --seed 0 --lr 1e-4 --out sweep_$1_$2_s0.jsonl 2>&1 | grep -vi warn | grep -E "^step|post"
done
