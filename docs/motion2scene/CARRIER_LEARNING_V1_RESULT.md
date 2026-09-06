# Grouped motion-conditioning experiment

The current generator does **not** demonstrate useful motion conditioning. Across the
two excluded test carriers, zero of 576 fixed conditioned proposals satisfies both
geometric margins at all 113 placements. Swapping the input event gives the same
acceptance counts. Both descriptive predictions in the
[registered protocol](CARRIER_LEARNING_V1.md) fail.

The next development direction is the
[event-conditioned proposal design](EVENT_CONDITIONED_GENERATOR_DESIGN.md): retain
location-dependent motion features, compare against constant-feature anchors and
analytic search, and keep the same full-sequence geometry checks. That follow-up
architecture is saved as a design; it has not been trained.

## What was built

Eight original walking references supply eight source groups. The existing local-crouch
operator constructs a 55 mm target drop at each of three route stations: 0.25, 0.50 and
0.75. All 24 targets and eight neutral parents pass the existing joint-limit and
self-intersection gates. Start/end poses, horizontal route and root orientation are
preserved. Construction took 42.875 seconds. No new controller trial was run.

Four parents supply 12 training cases; two parents supply six report-only validation
cases; two parents supply six excluded test cases. The split precedes construction and
fitting. All derivatives stay with their parent. Previously inspected parent references
make this a development experiment, rather than a fresh confirmatory test.

The bank, registry, implementation, source CSVs and protocol are hash-pinned. Training
receives capsule motion sequences and geometric preference/clearance losses, with no
obstacle-placement labels. Event station is construction metadata, not a network input.
The original collection remains ineligible for hallucinator/policy admission; this new
reference-only diagnostic does not change its qualification flags.

## Comparison

Nine generator runs compare correct, constant and shuffled training inputs at three
optimizer seeds. Each uses 1200 updates over the 12 training cases. A test-time input
swap rotates the three events within each excluded parent. Thirty-six separate
per-motion searches use 300 updates each on excluded target geometry. This comparator
pays adaptation cost and is not claimed to have converged. A random-prior baseline
uses the same scene domain.

Each of the six evaluated modes supplies 32 fixed draws per event/optimizer-seed cell.
All draws, including failures, are independently checked at 113 placements. A proposal
passes only when every placement clears its target by 10 mm and interferes with the
upright alternative by the 10 mm capsule overlap criterion. The overlap criterion is
not physical penetration depth. Aggregation within a carrier has denominator 288;
the independent test sample size is two source carriers.

## Complete results

Each entry is the number valid at all 113 placements out of **288 fixed proposals**.
Validation remains report-only; no checkpoint was selected using these results.

| Source carrier | Split | Correct input | Constant | Shuffled | Swapped event | Direct search | Random |
|---|---|---:|---:|---:|---:|---:|---:|
| 41005 | validation | 6 | 0 | 8 | 6 | 180 | 4 |
| 41006 | validation | 0 | 0 | 5 | 0 | 249 | 10 |
| 41007 | test | 0 | 9 | 45 | 0 | 158 | 8 |
| 41008 | test | 0 | 29 | 43 | 0 | 185 | 11 |

Correct-input yield is 0% on each test carrier. Direct search reaches 54.9% and 64.2%,
and every excluded event/optimizer cell has at least one passing draw. Thus the sampled
geometry admits solutions for these cases; the conditioned model has not learned to
produce them under this budget. Constant and shuffled training also outperform correct
inputs here. This is not evidence that shuffling is a better learning method: it is a
failed conditioning experiment with substantial optimization dependence.

The swapped-input control preserves every cell's acceptance count, including the six
validation passes. Both predeclared descriptive predictions fail. No significance test
or claim of a general population effect follows from two test parents.

The pooled test counts below retain nominal and uncertainty-stressed checks separately.
Pooling describes the draws; it does not increase the independent source count.

| Mode | Nominal valid / 576 | All-113 valid / 576 | Target margin at all placements | Upright margin at all placements |
|---|---:|---:|---:|---:|
| Correct input | 0 | 0 | 300 | 0 |
| Constant input | 64 | 38 | 296 | 150 |
| Shuffled training | 112 | 88 | 192 | 264 |
| Swapped test event | 0 | 0 | 300 | 0 |
| Per-motion search | 492 | 343 | 482 | 429 |
| Random prior | 34 | 19 | 281 | 237 |

The conditioned model already fails at nominal placement. None of its 576 test draws
satisfies the all-placement upright interference margin; 276 also miss target clearance.
These failure conditions overlap. The result cannot be explained solely by a stricter
jitter audit. All 216 event/seed/mode rows, proposal values and raw clearance hashes are
retained in the [portable evidence](evidence/carrier-learning.json).

## Post-hoc input diagnosis

A separate [checkpoint diagnosis](evidence/carrier-input-sensitivity.json) compares the
middle/late event inputs with the early event on the same parent, using matched latent
draws. It does not select checkpoints or resample proposals. On the two test carriers,
the input's maximum coordinate change is 0.120–0.122 m, but the largest paired output
station change is only 1.426e-8 in normalized route progress, and the largest output
height change is 3.556e-9 m. Flattened mixture parameters change by at most 3.496e-7.
The network is effectively insensitive to this intervention.

Between 37.5% and 48.4% of pooled features have absolute value above 0.999. This is
consistent with saturation and attenuation of event information. It does not prove
that global pooling is the cause: coordinates are present in the input, and optimization,
feature scaling and representation are confounded. The proposed normalized pooled and
anchor-based controls are designed to distinguish these explanations.

## Compute accounting

All 45 registered training cells completed without retries. Training-stage wall time,
including checkpoint/sample output, was 957.195 seconds on two CPU threads. The separate
engineering smoke is not included in these totals.

| Mode | Fitted models | Measured fitting seconds |
|---|---:|---:|
| Correct input | 3 | 170.257 |
| Constant input | 3 | 169.255 |
| Shuffled training | 3 | 166.882 |
| Direct per-motion search | 36 | 450.618 |

Each generator model costs about 56–58 seconds to fit across 12 training cases; each
direct search costs about 12–13 seconds on one excluded case. Their different update
counts and parameter learning rates are fixed in the protocol. Neither raw fitting
time nor optimizer-step count alone establishes an amortization advantage.

The nine generator runs and 36 direct searches each use 864,000 training reference
clearance evaluations: updates × eight proposals × five placements × two references.
Together they use 1,728,000 reference evaluations. These counts describe complete
reference minima, not individual capsule/box primitive operations, whose number depends
on the geometry. The independent final audit took 329.979 seconds and adds 1,562,112 NumPy reference evaluations
and 195,264 NumPy/PyTorch comparisons. Their maximum absolute disagreement is
5.5512e-17 m, below the registered 1e-8 m abort threshold. Sampling/checkpoint overhead and checking cost
must remain visible when comparing this system with direct search.

## Interpretation limits

Passing the reference gates does not establish successful tracking, semantic behavior
retention or obstacle-present execution. These are 29-capsule proxies at sampled
frames. Neither the imported collider contract nor between-frame clearance has been
certified. The 113 placement checks also do not certify the continuous uncertainty set;
the earlier selected-scene placement certificates do not transfer automatically.

Three edited crouch locations on straight walking motions cannot establish novel event
interpolation, other obstacle families, high-DOF behavior generalization or real-world
scene frequencies. The constant-input control retains route-dependent scene decoding.
Optimization seeds, edited events and proposal draws do not increase the carrier count.

## Reproduction and validation

The registration, bank, checkpoints, raw clearance arrays and completed results belong
under `/home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/`. The portable page
exports shorten local paths; reproducing the geometry requires the trusted source bundle
and G1 assets. A fresh clone alone does not contain those motion CSVs.

Use the registration/build/smoke/train/analyze commands in the
[protocol](CARRIER_LEARNING_V1.md). Every stage requires a fresh output path; never
overwrite the registered experiment. Export the completed audit with:

```bash
.venv_research/bin/python scripts/research/render_motion2scene_carrier_report.py \
  --result /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/result.json
```

The engineering smoke completes three short generator runs and one short direct search
in a separate directory. It is excluded from the experiment. All 65 impacted geometry,
inverse-model, margin, uncertainty, placement, lineage and earlier diagnostic tests pass.


The full bank manifest hash is
`sha256:59aed5b0dc96fe652162c2faa4d7b33842a805f090306050f6881d9d38efd0b3`.
The [export manifest](assets/carrier-manifest.json) pins the result snapshots, registry,
input diagnosis, renderer and figure. Previous study sources and exports remain frozen.

Exact impacted test command:

```bash
.venv_research/bin/python -m pytest \
  tests/dataset_generation/test_capsule_box_exact.py \
  tests/dataset_generation/test_capsule_box_torch.py \
  tests/dataset_generation/test_motion2scene_inverse.py \
  tests/dataset_generation/test_motion2scene_uncertainty.py \
  tests/dataset_generation/test_motion2scene_margins.py \
  tests/dataset_generation/test_placement_certificate.py \
  tests/dataset_generation/test_motion2scene_carriers.py \
  tests/dataset_generation/test_motion2scene_beam_teacher.py \
  tests/dataset_generation/test_motion2scene_repeatability.py \
  tests/dataset_generation/test_motion2scene_timing_diagnostic.py -q
```

Black and Ruff (`E,F,I`) pass for the new carrier module, experiment driver, renderer
and tests; `git diff --check` passes. The renderer independently recomputes acceptance
from all saved raw clearance arrays before exporting. Local Chrome checks at 1440×1000
and 390×844 decode the new figure with no page JavaScript exceptions or horizontal
document overflow. All 87 local HTML links/anchors and 40 output hashes across the six
follow-up export manifests pass. The figure and mobile page were visually inspected.
These page changes are local; no publication or push was performed.
