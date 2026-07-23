#!/bin/bash
# Sweep 2: hindsight x teacher, matched wall-clock (2400s each).
# Waits for any running train.py (sweep 1) to finish first.
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while pgrep -f "train.py --teacher" > /dev/null; do sleep 60; done
for cfg in "uniform maxrl --hindsight" "frontier maxrl --hindsight" "advmass_gpu skip skip"; do
  set -- $cfg
  if [ "$1" = "advmass_gpu" ]; then continue; fi
  echo "=== $1 + $2 $3 (2400s) ==="
  python3 train.py --teacher $1 --estimator $2 $3 --steps 999999 --max-seconds 2400 \
    --eval-every 25 --seed 0 --lr 1e-4 --out matched_$1_$2_hs_s0.jsonl 2>&1 | grep -vi warn | grep -E "^step|^FINAL|post"
done
echo "SWEEP2 DONE"
