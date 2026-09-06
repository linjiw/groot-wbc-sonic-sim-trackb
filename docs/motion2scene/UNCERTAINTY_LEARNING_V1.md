# Placement uncertainty: registered inverse-learning follow-up

This protocol is frozen before the main run. It follows the [first inverse-learning
result](INVERSE_LEARNING_V1_RESULT.md): selected nominal proposals failed 9 of 81
placement checks in two learned cells. Representative checks do not estimate the
robust yield of a distribution. The present experiment tests that missing quantity.

## Hypothesis and controlled change

Hypothesis: explicitly training against placement perturbations improves the fraction
of generated scenes that preserve the desired crouch-clear / upright-interference
relationship across the registered evaluation placements. Prediction: both KL families
improve robust valid count over their nominal counterpart in at least two of three
optimizer seeds. This prediction is a development decision criterion, not a significance test.

The frozen V1 bank, target motion encoder, four-component correlated Gaussian mixture,
logit station/height parameterization, fixed candidate-motion costs, and geometric
queries remain unchanged. There are no scene labels, analytic feasible intervals, new
motion operators, controller updates, or new physics runs. Only aggregation over
placement uncertainty changes. Pose offsets are global x/y translation ±2 cm, vertical
translation ±1 cm, and yaw ±0.02 rad. These are a designed stress domain, not an
estimated distribution of execution or sensing error.

Each optimizer update samples two latent proposals per mixture component. All eight
proposals receive the nominal pose plus four distinct corners drawn without replacement
from the 16 corners of the four-dimensional offset box. Corner draws use a separate
random generator seeded with optimizer seed + 30000, so latent noise is unchanged.
Across these five placements, each target recording contributes its minimum clearance;
each upright recording contributes its maximum clearance. The frozen preference loss
and target-feasibility penalty then operate on those worst values. Separate extrema
provide a conservative selection surrogate relative to a shared placement offset.
This is sampled worst-case training, not exhaustive or certified robust optimization.

Vertical broad-phase culling expands the entire beam domain by the maximum vertical
offset before excluding capsules. Reusing the nominal cull would silently omit possible
contacts at the domain boundaries. All retained frames and complete capsule axes are queried.

## Registered budget and comparison

- Two new arms: `robust_full` (KL 0.02), `robust_no_kl` (KL 0).
- Three optimizer seeds: 8121, 8122, 8123; 300 updates; Adam learning rate 0.002;
  target-feasibility weight 5; gradient clipping norm 10; CPU float64, two threads.
- Six new training cells; at most 3600 seconds for training and separately 3600 for
  analysis. Engineering smoke runs use two updates and are never pooled.
- Nominal comparators are the six saved V1 `full` and `no_kl` checkpoints/proposal
  files, with their source hashes checked. They used the same architecture, optimizer,
  update count, and latent sample schedule. New arms make five times as many placement
  queries per update: update budgets are matched, geometric compute budgets are not.
- Final checkpoint only; no validation-based checkpoint or seed selection.
- Reference poses and execution seeds 7901/7902 enter gradients. Execution seed 7903
  enters evaluation only; it was previously observed in development and is not a fresh
  confirmatory holdout. Every model still sees just carrier 41002.

## Evaluation and failure accounting

Evaluate the fixed first 64 proposals from each cell's 256-proposal draw. New arms also
make the full 256-sized draw before taking the prefix, preserving the V1 random-number
schedule; they do not redraw until accepted. For each of all 768 proposals (12 cells ×
64), check all eight reference/execution recordings at:

1. All 81 placements in the Cartesian product {-limit, 0, +limit} for four offsets.
2. An additional 32 interior placements drawn once uniformly using NumPy seed 92301,
   fixed in the manifest and never used by the optimizer.

Nominal is offset zero at index 0; indices 0–80 are the grid; 81–112 are the interior
set. A proposal passes a set only if, at every placement, all crouch recordings have
clearance ≥10 mm and every upright recording has capsule overlap proxy ≥10 mm.
Positive clearance is primitive separation; negative clearance is not physical
penetration depth. Report nominal, grid, interior, and combined pass counts separately,
plus target-only clearance and gradient-source-only validity. Denominators always
include rejected proposals. Do not select only favorable representatives.

An independent NumPy full-axis geometry implementation evaluates every proposal and
placement, with proposal-specific broad phase. Cross-check the first eight proposals
of every cell at all 113 offsets against the differentiable implementation; abort if
maximum absolute disagreement exceeds 1e-8 m. Save all clearance arrays, placements,
proposals, checkpoints, source hashes, histories, timings, and failures. The primary
comparison is the per-seed change in combined 113-placement valid count. Three seeds
and one selected carrier do not support population-level inference. Distribution
coverage is not re-estimated here; improved yield may reflect concentration.

## Carrier gate and next research decision

Audit the existing timing and repeated-pair studies by source hash. Different execution
seeds, perturbations, or derivatives of one motion are not independent motion carriers.
The existing timing study tests three source seeds; repeatability currently concerns
only 41002. Do not create an alleged cross-carrier train/test split from repetitions.

If robust yield improves consistently, carry this objective into a newly registered
multi-carrier study. Otherwise retain the frozen nominal baseline and inspect whether
failure comes from target feasibility, insufficient upright interference, or distribution
collapse before changing the objective. A further tuning experiment gets a new protocol.

Before a model can be called a general motion-conditioned generator, build independent
source-carrier train/validation/test groups, compare a shuffled or absent motion input
and a converged per-motion optimizer, and test different obstacle families. Before
execution admission, establish imported collision geometry coverage, between-frame
clearance, obstacle support geometry, and obstacle-present tracking. Finite placement
samples certify neither all continuous offsets nor inter-frame motion. No Q4-admitted
ladder, training-eligible scene, physical success, or safety guarantee follows here.

## Reproduction

```bash
.venv_research/bin/python scripts/research/motion2scene_uncertainty_learning.py prepare \
  --source /home/linjiw/research-data/groot-wbc/m2s-inverse-learning-v1/manifest.json \
  --protocol docs/motion2scene/UNCERTAINTY_LEARNING_V1.md \
  --out /home/linjiw/research-data/groot-wbc/m2s-uncertainty-learning-v1
.venv_research/bin/python scripts/research/motion2scene_uncertainty_learning.py run \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-uncertainty-learning-v1/manifest.json
.venv_research/bin/python scripts/research/motion2scene_uncertainty_learning.py analyze \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-uncertainty-learning-v1/manifest.json
```

Output directories must be new. A rerun requires a distinct output path; existing
registered manifests and results are never overwritten.
