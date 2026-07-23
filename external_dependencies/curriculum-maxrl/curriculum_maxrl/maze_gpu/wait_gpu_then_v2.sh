#!/bin/bash
# Wait until the A10G has >18GB free, then run the FrontierMax v2 pilot + A/B/C.
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free_mb" -gt 18000 ]; then break; fi
  sleep 120
done
echo "GPU free (${free_mb}MB) — starting v2 pilot"
python3 train.py --teacher frontier_alp --estimator maxrl --hindsight-dense --hindsight-to-teacher \
  --steps 30 --eval-every 10 --seed 0 --lr 1e-4 --out pilot_v2.jsonl 2>&1 | grep -vi warn | grep -E "^step|^FINAL|post"
echo "=== pilot done; matched A/B/C (2400s each) ==="
# A: current best (frontier_alp, sparse hindsight)  B: dense  C: dense + teacher feedback
python3 train.py --teacher frontier_alp --estimator maxrl --hindsight --steps 999999 --max-seconds 2400 \
  --eval-every 25 --seed 0 --lr 1e-4 --out matched_falp_maxrl_hs_s0.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
python3 train.py --teacher frontier_alp --estimator maxrl --hindsight-dense --steps 999999 --max-seconds 2400 \
  --eval-every 25 --seed 0 --lr 1e-4 --out matched_falp_maxrl_hsdense_s0.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
python3 train.py --teacher frontier_alp --estimator maxrl --hindsight-dense --hindsight-to-teacher --steps 999999 --max-seconds 2400 \
  --eval-every 25 --seed 0 --lr 1e-4 --out matched_falp_maxrl_hsdense_tt_s0.jsonl 2>&1 | grep -vi warn | grep -E "^FINAL|post"
echo "V2 ABC DONE"
