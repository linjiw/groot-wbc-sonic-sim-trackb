#!/bin/bash
# Final push: (1) long-horizon champion run (4x budget) — tests whether
# level 6+ is a duration question; (2) gamma=4 concentration on GPU;
# (3) dense-hindsight multi-seed due diligence.
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while pgrep -f "train.py --teacher" > /dev/null; do sleep 60; done
echo "=== LONG: frontier_alp + maxrl + dense hindsight (9600s) ==="
python3 train.py --teacher frontier_alp --estimator maxrl --hindsight-dense \
  --steps 999999 --max-seconds 9600 --eval-every 50 --seed 0 --lr 1e-4 \
  --out long_falp_hsdense_s0.jsonl --save-ckpt long_falp_hsdense_s0.pt 2>&1 | grep -vi warn | grep -E "^step|^FINAL|post|saved"
echo "=== GAMMA4: frontier_alp(power=4) + maxrl + dense hindsight (2400s) ==="
python3 train.py --teacher frontier_alp --teacher-power 4 --estimator maxrl --hindsight-dense \
  --steps 999999 --max-seconds 2400 --eval-every 25 --seed 0 --lr 1e-4 \
  --out matched_falp_p4_hsdense_s0.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
for seed in 1 2; do
  echo "=== SEED $seed: frontier_alp + maxrl + dense hindsight (2400s) ==="
  python3 train.py --teacher frontier_alp --estimator maxrl --hindsight-dense \
    --steps 999999 --max-seconds 2400 --eval-every 25 --seed $seed --lr 1e-4 \
    --out matched_falp_maxrl_hsdense_s$seed.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
done
echo "FINAL SWEEP DONE"
