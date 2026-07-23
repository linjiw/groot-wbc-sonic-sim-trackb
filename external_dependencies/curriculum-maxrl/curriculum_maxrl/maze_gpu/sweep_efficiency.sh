#!/bin/bash
# Inference-efficiency study: retrain 3 headline configs with checkpoint
# saving (matched 2400s), then measure samples-to-90%-coverage per level.
cd /home/ec2-user/work/curriculumrl/maxrl/curriculum_maxrl/maze_gpu
while pgrep -f "train.py --teacher" > /dev/null; do sleep 60; done
for cfg in "uniform maxrl ''" "uniform grpo ''" "frontier_alp maxrl --hindsight-dense"; do
  set -- $cfg
  extra=$(echo $3 | tr -d "'")
  name="ck_$1_$2$( [ -n "$extra" ] && echo _hsd )"
  echo "=== $name (2400s) ==="
  python3 train.py --teacher $1 --estimator $2 $extra --steps 999999 --max-seconds 2400 \
    --eval-every 50 --seed 0 --lr 1e-4 --out ${name}.jsonl --save-ckpt ${name}.pt 2>&1 \
    | grep -vi warn | grep -E "^FINAL|post|saved"
done
echo "=== efficiency eval ==="
python3 eval_efficiency.py ck_uniform_maxrl.pt ck_uniform_grpo.pt ck_frontier_alp_maxrl_hsd.pt 2>&1 | grep -vi warn
echo "EFFICIENCY DONE"
