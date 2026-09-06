# Registered event-representation and scale experiment

Status: freeze this protocol and implementation before the first registered fit.
This is a reference-only development study on the existing eight-parent bank. Previously
observed validation/test parents stay excluded from fitting, but are not fresh confirmatory
samples. Original Q4 and scene-admission flags remain false. No GPU or physics is used.

## Contribution and claim–evidence contract

The [previous grouped experiment](CARRIER_LEARNING_V1_RESULT.md) yields 0/576 valid
conditioned test proposals despite feasible per-motion search solutions. Its output
barely responds to event location. The new contribution is a local, motion-only proposal
head that retains route position, plus a controlled experiment separating representation,
training compute, parameter count and training-source count. It does not claim novelty
against the literature or a physical collision guarantee.

| Hypothesis | Changed component / comparator | Evidence and unit | Limitation |
|---|---|---|---|
| Local features help obstacle placement | Local versus pooled, constant, shuffled inputs | All-113 validity / 144 on each of two test source parents | Same robot, beam family, three edited events |
| More optimization helps | 2400 versus 600 updates on the same run | Paired budget snapshots per source, with seed/event detail | Two budgets do not define a scaling law |
| More capacity helps | Width 128 versus 32, all else fixed | Equal geometry-query budgets; parameter/time counts | One width change, not tuned optima |
| More motion sources help | Four versus two training parents | Equal updates/queries, nested parent sets | Two selected parents are not a randomized data-size sweep |

## Retained data, geometry and information

Reuse the hash-pinned `m2s-carrier-learning-v1/manifest.json`, its 24 target reference cases,
source registry, all previous source/operator hashes, and existing train/validation/test
split: 41001–41004 / 41005–41006 / 41007–41008. All local-crouch derivatives stay with
the parent. Training case order is balanced by epochs; validation is report only.

The target encoder sees 29 capsule midpoint/axis/radius sequences and the target root's
horizontal route. Subtract per-frame root XY from capsule midpoint XY; retain absolute
vertical coordinates. Append normalized route progress and elapsed-time fraction.
Interpolate these 205 features to 64 equally spaced route-progress samples. Repeated
progress samples retain their first frame for encoding only. Stationary routes are
rejected; they require a separate time-anchor representation. No event ID, construction
station, alternative motion, scene label or analytic feasible interval enters the model.

Fit per-channel mean and standard deviation using only the selected training cases,
with a fixed 0.02 standard-deviation floor. Save these values and the exact training IDs
in registration; recompute and verify them on load. The constant control receives zeros
*after* normalization, including zero progress/time input channels.

This is a deliberately minimal implementation of the
[event design](EVENT_CONDITIONED_GENERATOR_DESIGN.md): one shared full-body linear layer
rather than body-region pooling, and no velocity or masked-motion pretraining. Those
extensions remain separate hypotheses. Encoding is resampled; the collision evaluator
still checks every original reference frame and every retained full capsule axis, using
the existing exact culling contract.

Beam dimensions stay depth/width/thickness 0.10/1.20/0.10 m. Height stays [1.10,1.45] m,
station [0.10,0.90]. Route-based world decoding, shared transform, margin definitions,
fixed target/alternative costs and placement perturbation domain are unchanged.

## Model and ablations

All six arms use eight anchor distributions. Anchor `a = 0,...,7` maps local station
latent `z0` to `0.10 + 0.10 * (a + sigmoid(z0))`; height is `1.10 + 0.35 * sigmoid(z1)`.
Thus the hand-designed station intervals cover the whole domain for every arm.

The feature encoder is Linear(205,width) → LayerNorm → SiLU, followed by a width-preserving
kernel-5 temporal convolution with SiLU, residual connection and LayerNorm. Local features
are linearly interpolated at anchor centers 0.15,...,0.85. A shared linear head predicts
six residual parameters per anchor, added to eight trainable base distributions: logit,
two latent means, two scales and correlation. Means/logits/correlation initialize at zero,
scales at 0.75 with a 0.03 softplus floor and correlation bound 0.95. The head initializes
with normal standard deviation 0.001 and zero bias. There is no feasible-height initialization.

| Arm | Width | Input/aggregation | Training parents |
|---|---:|---|---:|
| local | 32 | Correct local features | 4 |
| pooled | 32 | Correct encoder, mean across time then shared across anchors | 4 |
| constant | 32 | Zero features; same local head and anchor bases | 4 |
| shuffled | 32 | Fixed cyclic next-case training input; geometry target unchanged | 4 |
| wide | 128 | Correct local features | 4 |
| half_data | 32 | Correct local features | 2: 41001,41002 |

The shuffled permutation is over the ordered 12 training cases only, as in the previous
study; it changes event identity and sometimes parent identity. Evaluate that checkpoint
with the correct target input. The half-data arm uses six cases and computes normalization
only on those six. It receives twice as many visits per case at an equal update budget.
Do not interpret the comparison as a randomized estimate over all possible two-parent sets.

At test time also rotate early→middle→late→early inputs within each excluded parent for
the `local` arm at both budgets, with evaluated geometry unchanged. No new training is
required. Width and parent comparisons are one-factor changes from `local`, not a full
capacity × data factorial. Larger input sensitivity by itself is not an acceptance metric.

## Fitting, scale and cost

Run optimizer seeds 8321/8322/8323 for each arm: **18 independent fits**. Adam learning
rate 0.001, float64 CPU, two threads, deterministic PyTorch operations, global gradient
norm clipping 10. Save snapshots at **600 and 2400 updates** from each uninterrupted
training run. The short snapshot is a prefix of the long run, not an independent replicate.
No checkpoint selection, early stopping, restarts or outcome-dependent hyperparameter changes.

One reparameterized sample from each of eight anchors supplies eight proposals per update.
The expected objective weights each anchor loss by its softmax probability:
`preference + 5 * target-margin barrier + 5 * upright-interference barrier`, no KL or entropy
term. Check nominal pose and four distinct random corners from the existing 16-corner
uncertainty domain. Both margins are 10 mm. RNG offsets from the optimizer seed remain
+10000 proposals, +30000 placements and +40000 shuffled case epochs.

Every arm has exactly the same per-update geometry budget. A 600-step snapshot has used
48,000 reference minima; a 2400-step fit has used 192,000. Total registered training uses
3,456,000 reference minima. These are whole-reference queries, not primitive-pair operation
counts. Report parameter count, cumulative fitting time at each snapshot and sampling time;
do not sum prefix fitting time twice. Width changes CPU overhead even at equal geometry cost.
The scale claim concerns compute, parameter count and the named source subset separately.

## Independent evaluation and predictions

Draw a full 256 samples per checkpoint/case with seed +20000, then check its fixed first
16, without rejection/resampling before reporting raw yield. Use the same draw schedule
for swapped inputs. Six arms plus the local input swap, two budgets, three seeds and 12
excluded cases yield **504 new audit rows / 8,064 proposals**. Each parent/mode/budget
aggregate has 144 proposals (3 events × 3 seeds × 16); independent test n remains two.

Use the existing 81 grid plus 32 fixed interior poses: x/y ±2 cm, z ±1 cm, yaw ±0.02 rad.
Independent NumPy evaluates all 16 proposals × 113 placements × two references. Cross-check
the first four proposals of every row at every placement against PyTorch; abort above
1e-8 m disagreement. This adds 1,822,464 independent reference queries and 455,616 cross-checks.
Save every scene, per-proposal verdict, raw clearance array, source/checkpoint hash, nominal
validity, target clearance and upright interference. The finite placement sample is not
continuous placement certification, and sampled capsule overlap is not physical penetration depth.

Reuse the previously observed direct-search and random-prior comparators by taking the
first 16 of their 32 frozen draws, with all existing raw hashes verified. These add 72
context rows; no new baseline fitting, draw or independent query is implied. Their optimizer
seeds are 8221–8223, unlike the new arms. They supply useful search context, not a newly
matched-seed comparator or an optimized analytic-search ceiling. No amortization claim
should rely on fitting time alone or ignore the missing analytic comparator.

Predeclared descriptive predictions (strict inequalities, on **both** test parents):

1. At 2400 updates, local beats pooled, constant and shuffled inputs.
2. At 2400 updates, swapping the input event lowers local validity.
3. Local improves from 600 to 2400 updates.
4. Width 128 improves on width 32 at 2400 updates.
5. Four training parents improve on two at 2400 updates.

Report all five independently, including regressions, ties, validation and per-seed/event
variation. These are descriptive decision criteria, not significance tests. One successful
axis does not support “scale always helps.” If scale fails, preserve that outcome and
separate optimization, representation and diversity limitations before another study.

## Stop conditions and reproduction

Pin this protocol, new module/driver, parent manifest and previous result before fitting.
Full parent provenance is transitively checked through the frozen carrier loader. Stop on
hash mismatch, nonfinite loss/gradient, incomplete fit set, normalization leakage, or query
disagreement. Training stage cap: 7200 wall seconds; analysis cap: 3600. Failures preserve
artifacts and are not silently retried. An engineering smoke runs 12 updates per arm at
seed 8321, evaluates only a training case, and lives under `smoke/`; exclude it from results.

```bash
.venv_research/bin/python scripts/research/motion2scene_event_scaling.py register \
  --carriers /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/manifest.json \
  --previous /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/result.json \
  --protocol docs/motion2scene/EVENT_SCALING_V1.md \
  --out /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1
.venv_research/bin/python scripts/research/motion2scene_event_scaling.py smoke \
  --registration /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/registration.json
.venv_research/bin/python scripts/research/motion2scene_event_scaling.py train \
  --registration /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/registration.json
.venv_research/bin/python scripts/research/motion2scene_event_scaling.py analyze \
  --registration /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/registration.json
```

Every output directory must be new. No old study or source is overwritten. This study
adds no reference motions, learned downstream policy, admitted scenes or physical trials.
