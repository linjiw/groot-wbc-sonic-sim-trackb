# Registered test: explicit target and interference margins

This protocol is frozen before the main run. The preceding [uncertainty experiment](UNCERTAINTY_LEARNING_V1_RESULT.md)
found that 271 of 295 rejected new-model proposals failed only the upright-interference
margin. Its frozen preference loss can be below 0.001 with only 5 mm upright overlap,
although evaluation requires 10 mm. This experiment changes that objective mismatch.

## Hypothesis and loss

Adding an explicit interference penalty will increase the raw fraction of proposals
meeting both geometric margins across all 113 evaluation placements. The registered
prediction is improvement over the corresponding uncertainty-only preference baseline
in at least two of three optimizer seeds, separately for KL 0.02 and KL 0. This is a
development criterion; report all regressions and target-clearance changes. A majority
of improvements alone does not admit a model for execution or establish generalization.

For each proposal, aggregate clearances over the same nominal-plus-four-corner training
placements as the previous experiment. Let `d_target` be the minimum over target
recordings and placements, and `d_upright` the maximum over upright recordings and
placements; each recording clearance is already the minimum over its body capsules
and recorded frames. With `m = 0.01 m`, use

```text
B_target  = [max(0, m - d_target) / m]^2
B_upright = [max(0, m + d_upright) / m]^2
L = a * preference + 5 * B_target + 5 * B_upright + beta * KL
```

The preference decoder, costs, mixture density and prior remain frozen. The new penalty
is zero exactly when the supplied upright clearances meet their signed threshold. This
soft penalty does not make all generated samples valid or cover unsampled poses/time.

## Arms, controls and budget

| Arm | Preference weight a | KL beta | New interference weight |
|---|---:|---:|---:|
| margin_full | 1 | 0.02 | 5 |
| margin_no_kl | 1 | 0 | 5 |
| constraints_full | 0 | 0.02 | 5 |
| constraints_no_kl | 0 | 0 | 5 |

The two constraint-only arms test whether preference adds value once the binary margins
are explicit. Their comparison is descriptive; do not reinterpret a tie as evidence of
preference learning. For this fixed binary bank, satisfying both geometric conditions
already strongly favors the target under the frozen cost model.

Run each new arm at seeds 8121/8122/8123 for 300 Adam updates, learning rate 0.002,
float64 CPU, two threads, gradient norm clip 10. The four-component motion-conditioned
mixture and all initialization remain unchanged. Each update draws two proposals per
component and checks nominal placement plus four distinct uniformly drawn corners from
the 16-corner translation/yaw box. Proposal RNG is seed + 10000; corner RNG is seed +
30000. All new arms use the same draws and five placement queries per proposal/update.
Final checkpoint only, with no early stopping or best-step selection.

The comparators are the six saved `robust_full` / `robust_no_kl` cells of the prior
experiment. Verify their checkpoints, proposal files and complete clearance arrays by
hash; recompute their metrics from those arrays. Their evaluation is reused, not
rerun or counted as new evidence. Updates, proposal draws, corner schedules and query
counts match the new preference arms. The constraint-only arms share that budget.

Budget: 12 new training cells; at most 3600 seconds for training and separately 3600
seconds for analysis; no new physics/GPU training. Two-update engineering smoke cells
use a distinct directory and are excluded. Nonfinite loss/gradient, source hash mismatch,
geometry disagreement >1e-8 m, incomplete cells or budget exhaustion stops the stage.
Existing output directories are never overwritten. Record failures.

## Evaluation, split and evidence

Use the original V1 source carrier and input split unchanged: references and execution
seeds 7901/7902 in gradients, previously observed development seed 7903 in evaluation
only. No independent-carrier generalization claim is available. No scene labels or
analytic feasible intervals enter training.

Draw a 256-proposal batch with seed + 20000 and audit its first 64 unfiltered proposals
per cell, preserving the earlier sampling schedule. The same 81 grid + 32 fixed interior
placements are inherited byte-for-byte from the prior manifest: x/y ±2 cm, z ±1 cm,
yaw ±0.02 rad. The interior set is excluded from optimization but has been inspected in
prior development; it is not a fresh confirmatory test set.

Evaluate all eight recordings using the independent NumPy full-axis capsule/box query.
A proposal passes only if every placement clears every target recording by 10 mm and
interferes with every upright recording by the 10 mm capsule-overlap criterion. Negative
clearance is an overlap proxy, not physical penetration depth. Cross-check the first
eight proposals per new cell at all 113 placements against the differentiable query.

Report nominal, 81-grid, 32-interior, combined-113, gradient-source-only, target-only and
upright-only success counts, all with denominator 64. Classify all failures as target,
upright, or both. Record the per-seed change in joint validity and target clearance.
Also count occupied bins among accepted proposals in the fixed 30-by-70 station/height
partition over [0.35,0.65] × [1.10,1.45]. This is an accepted-sample dispersion count,
not the fraction of true feasible support covered. Do not infer coverage improvement
from increased yield alone. Save raw proposals/clearances, hashes, checkpoints, losses,
training/evaluation times, and all eligibility flags as false.

## Decision

If margin alignment reliably improves both geometric conditions, freeze the candidate
and move to continuous-placement checking and independent-carrier tests. If the
constraint-only ablation matches or beats preference, keep the simpler objective as a
competitive baseline and narrow the claim to conditional geometric constraint synthesis.
Do not claim recovered real-world scene frequencies, preference beyond the candidate
bank, motion-input dependence, certified imported-body collision, or physical success.
A failed or mixed result gets its own diagnosed follow-up rather than extra tuning
inside this registration.

## Reproduce

```bash
.venv_research/bin/python scripts/research/motion2scene_margin_learning.py prepare \
  --previous /home/linjiw/research-data/groot-wbc/m2s-uncertainty-learning-v1/result.json \
  --protocol docs/motion2scene/MARGIN_LEARNING_V1.md \
  --out /home/linjiw/research-data/groot-wbc/m2s-margin-learning-v1
.venv_research/bin/python scripts/research/motion2scene_margin_learning.py run \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-margin-learning-v1/manifest.json
.venv_research/bin/python scripts/research/motion2scene_margin_learning.py analyze \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-margin-learning-v1/manifest.json
```
