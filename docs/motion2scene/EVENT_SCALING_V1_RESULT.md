# Event representation and scale: development results

The local event representation passes the two registered conditioning criteria. At 2,400
updates it accepts **110/144 and 133/144** proposals on the two excluded test sources,
versus 2/144 and 4/144 for the otherwise matched pooled encoder. Swapping the input event
reduces both local counts to zero. These are reference-geometry development results.

**Data helped more than parameter or update scale in this comparison.** Doubling training
parents from two to four improves pooled test yield from 188/288 (65.3%) to 243/288 (84.4%)
at equal update budgets: +19.1 percentage points. Quadrupling updates for the small,
full-data local model leaves pooled test yield unchanged. The wider model is worse overall.
The [protocol](EVENT_SCALING_V1.md) remains frozen and all five predictions are reported below.

## What changed

The prior model compressed a motion into one pooled vector and failed its conditioning
test. The new model retains local features at eight route anchors. All controls receive
the same anchor-based scene parameterization, including the constant-input model.
A successful local-versus-constant comparison would therefore concern learned features
beyond the built-in route/station prior.

The encoder uses training-only feature normalization, horizontal root-relative body
geometry, absolute vertical geometry, route progress and time fraction. It resamples
encoding to 64 points, while the verifier retains all original frames and capsule axes.
The minimal implementation omits the proposed velocity features, body-region pooling
and masked-motion pretraining; those remain untested extensions.

The six arms compare local, pooled, constant and shuffled input, plus a wider local
network and a local network trained on two rather than four source parents. Each fit
is measured at 600 and 2,400 updates. The small network has 12,118 parameters; the wider
network has 109,750 (9.06× as many parameters from 4× hidden width). All arms evaluate
eight geometry proposals per update, so the budget comparison does not hide a larger
proposal batch inside the wider model.

## Scope of “scale”

These are three separate measurements: optimizer updates, parameter count and training
source count. There are only two values per axis, and the data-size comparison uses one
nested subset. This cannot establish a general scaling law or guarantee that further
increases help. Additional optimizer seeds and early/middle/late edits remain dependent
on their source parent. Four training sources are still a small development dataset.

No new reference or physics data are collected. The two excluded test parents have
been inspected in earlier studies; this is a representation diagnosis using familiar
development data, not fresh confirmatory generalization. The three event positions are
shared between splits. Novel event-phase interpolation, other motion families, multiple
objects and executed humanoid behavior remain untested.

## All source-level outcomes

Entries are all-placement-valid proposals **out of 144 per source**: three events × three
optimizer seeds × 16 fixed draws. Independent test n is two source parents; the two budget
snapshots are dependent measurements from each uninterrupted fit. Validation is report-only.

| Arm | Updates | 41005 validation | 41006 validation | 41007 test | 41008 test |
|---|---:|---:|---:|---:|---:|
| Local | 600 | 133 | 128 | 112 | 131 |
| Local | 2400 | 143 | 136 | 110 | 133 |
| Pooled | 600 | 0 | 0 | 1 | 4 |
| Pooled | 2400 | 0 | 0 | 2 | 4 |
| Constant | 600 | 0 | 0 | 6 | 5 |
| Constant | 2400 | 0 | 0 | 4 | 3 |
| Shuffled | 600 | 1 | 0 | 0 | 4 |
| Shuffled | 2400 | 0 | 0 | 0 | 2 |
| Wide | 600 | 107 | 126 | 85 | 133 |
| Wide | 2400 | 124 | 127 | 95 | 133 |
| Two parents | 600 | 71 | 42 | 63 | 87 |
| Two parents | 2400 | 88 | 67 | 77 | 111 |
| Swapped event | 600 | 0 | 1 | 0 | 0 |
| Swapped event | 2400 | 0 | 0 | 0 | 0 |
| Earlier direct search | reused | 95 | 126 | 79 | 93 |
| Earlier random prior | reused | 3 | 4 | 3 | 5 |

The earlier direct/random rows reuse the first 16 frozen draws per cell from the prior
study, at optimizer seeds 8221–8223. They are disclosed context, not new matched-seed
runs or an analytic-search ceiling. New arms use 8321–8323. The local model's higher
raw yield than this search budget does not establish an end-to-end amortization advantage.

## Registered predictions and scale interpretation

| Prediction, on both test sources | Outcome | Evidence |
|---|---|---|
| Local beats pooled, constant and shuffled at 2,400 | Met | 110/133 versus pooled 2/4, constant 4/3 and shuffled 0/2 |
| Swapping the input event lowers local validity | Met | 110/133 → 0/0 |
| 4× updates improve the full-data local model | Not met | 112/131 → 110/133; pooled 243/288 at both budgets |
| Width 128 improves over width 32 at 2,400 | Not met | 110/133 → 95/133; one regression and one tie |
| Four parents improve over two at 2,400 | Met | 77/111 → 110/133 at equal geometry budget |

The wider model has 9.06× as many parameters and loses 15 accepted proposals overall
(5.21 percentage points). This concerns the fixed optimization recipe; it does not prove
that more capacity is intrinsically harmful. Extra updates help the *two-parent* arm
(150→188/288) and the wide arm (218→228/288), even though the full-data small model's
pooled test count stays at 243. Scale effects depend on the arm and source.

Validation yield for the small local model rises from 261 to 279/288 while pooled test
yield is unchanged. That distinction prevents selecting a claim from validation alone.
The data-size result uses one nested subset, so source composition remains confounded
with size. A larger study needs repeated subset orders and fresh excluded source identities.

## Seed variation and remaining failures

Each entry below pools three events, **out of 48 per optimizer seed**. Preserve the source
unit; these seed columns are not additional independent motion carriers.

| Arm | Updates | 41007: 8321 / 8322 / 8323 | 41008: 8321 / 8322 / 8323 |
|---|---:|---|---|
| Local | 600 | 35 / 42 / 35 | 41 / 45 / 45 |
| Local | 2400 | 34 / 40 / 36 | 43 / 42 / 48 |
| Wide | 600 | 33 / 18 / 34 | 48 / 47 / 38 |
| Wide | 2400 | 44 / 32 / 19 | 48 / 48 / 37 |
| Two parents | 600 | 20 / 21 / 22 | 25 / 28 / 34 |
| Two parents | 2400 | 25 / 17 / 35 | 32 / 35 / 44 |

At 2,400 updates, 45/288 local test proposals still fail: 43 miss the upright-interference
margin and two miss target clearance, with no overlap in this set. At 600 updates the
same 45 failures split into 32 interference and 13 clearance failures. More optimization
changes the failure mix without improving total yield. Nominal local test validity is
285/288 at 2,400, compared with 243/288 under all 113 placements.

The early event on carrier 41007 is the weakest local case at 2,400: only 18/48 draws pass,
versus 45/48 for its middle event and 47/48 for its late event. The separate fixed fresh
sample below also fails on that early event. Averaged acceptance must not conceal this gap.

A [post-hoc support diagnostic](evidence/event-support-diagnostic.json) counts accepted
samples in fixed 0.02-station × 0.01-m-height bins. Across the six local test cases at
2,400 updates, accepted draws occupy only 3–7 bins per case. This is bin occupancy at a
fixed sample budget, not an estimate of feasible-support coverage or scene realism.
No diversity or coverage prediction was retroactively added to the protocol.

## Research decision

Keep local event features and the small model as the development baseline. Prioritize
more distinct training sources and withheld event phases over a wider network or a
blanket increase in optimizer updates. The 600-update snapshot is a useful low-cost
comparison in the next study because it matches the longer run's pooled test yield,
while differing on individual sources; this does not replace the current protocol's
2,400-update primary endpoint.

The next [data-scaling design](DATA_SCALING_DESIGN.md) proposes repeated nested subsets
and separate equal-compute/equal-visits comparisons. Include early events explicitly,
and measure the alternative-interference gap on fresh cases. Compare analytic search
and bounded geometric refinement as prospective alternatives before claiming that the
learned sampler reduces total work per valid scene. Maintain independent rejection and
the unresolved imported-body, temporal and continuous-placement contracts.

The [source recheck](SCALING_SOURCE_NOTES.md) distinguishes the LfLH distribution learner,
its downstream policy, and critical-point factorization. Those sources motivate the
research program; they are not evidence for the performance numbers here.

## Completed fitting cost

All 18 fits completed, saving 36 budget snapshots. Training-stage wall time was
1,902.722 seconds (31.71 minutes), including checkpoint/sample output. No registered
fit was restarted. The separate six-run, 12-update engineering smoke is excluded.

| Arm | Parameters | Total fitting seconds at 600 updates, 3 seeds | Total at 2,400 updates, 3 seeds |
|---|---:|---:|---:|
| Local | 12,118 | 77.715 | 311.654 |
| Pooled | 12,118 | 79.477 | 320.772 |
| Constant | 12,118 | 77.903 | 314.467 |
| Shuffled | 12,118 | 78.956 | 316.488 |
| Wide | 109,750 | 80.338 | 321.847 |
| Two training parents | 12,118 | 79.387 | 317.146 |

The 600-step column is a prefix of the 2,400-step column; do not add both to estimate
total fitting cost. Final training uses 3,456,000 whole-reference geometry minima across
the 18 fits. Primitive capsule/box operation counts depend on the geometry and are not
represented by that number. Similar fitting times for the two widths concern this
geometry-heavy CPU workload; they are not general model-compute scaling evidence.

## Fixed fresh-sampling check

The preselected local/8321/2,400 checkpoint, case `41007_event0`, sampling seed 9321
produces **0/8 accepted** proposals without retries. All eight clear the target by
35.25–43.92 mm, but their worst-placement upright clearances range from −6.43 to +2.13 mm;
the required upper bound is −10 mm. Thus every rejection is due to insufficient upright
interference, despite station proposals near the early event (0.2530–0.2604).

This result distinguishes locating an event from learning its full geometric relationship.
It is a separate engineering sample, not part of the registered 8,064 proposal audit or
a representative estimate selected after seeing outcomes. The plan is saved under
`sampler-plan.json`, and every draw and refusal remains in `fresh-sampler/result.json`.
The sampler returns beam poses and accepted indices; here the accepted set is empty.

Measured model/draw time was 0.000971 seconds and independent checking took 0.357608
seconds. These component timings exclude loading and normalization verification and
were measured while the main audit ran; they are not end-to-end deployment latency or
an isolated runtime comparison. No physical scene is certified.

The completed independent audit took **439.802 seconds**. It checks all 8,064 new draws,
adding 1,822,464 NumPy reference queries and 455,616 NumPy/PyTorch comparisons. Maximum
absolute disagreement is **9.715e-17 m**, below the registered 1e-8 m abort threshold.
All 504 new rows and 72 reused baseline rows are preserved in the
[portable evidence](evidence/event-scaling.json), with raw clearance hashes.

## Collision and distribution limits

All reported acceptance requires both a 10 mm target-clearance margin and a 10 mm
upright-interference criterion at every one of the fixed 113 placements. This is a
reference-capsule proxy at sampled frames. It does not certify actual imported colliders,
continuous time, continuous placement uncertainty, controller response or robot safety.
The original motion-bank admission flags remain false.

No obstacle-location labels supervise the model. However, the scene family, route
parameterization, alternative motion, preference costs, margins and placement prior
supply modeling assumptions. Motion alone does not identify the frequency of real-world
scene layouts. Any realism or frequency prior will need a separately documented source.

## Reproduction

The [protocol](EVENT_SCALING_V1.md) contains register/smoke/train/analyze commands. Local
artifacts live under `/home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/`. The trusted
source motion bundle and G1 assets are needed to reconstruct features and geometry; a
fresh clone alone cannot reproduce this experiment. Prefix budget snapshots are saved
from each uninterrupted run and must not be counted as 36 independent fits.

After analysis and the fixed fresh-sampling check below, export the verified report with:

```bash
.venv_research/bin/python scripts/research/render_motion2scene_event_report.py \
  --result /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/result.json
```

A separate sampler draws a fixed batch from a saved checkpoint and independently rejects
failed placements without retries. It exports beam poses in the original parent reference
world and records the shared translation used by the checker. The fixed demonstration
uses the protocol's primary `local` arm, seed 8321, 2,400-step snapshot, case
`41007_event0`, and a fresh sampling seed 9321; this choice precedes the completed audit.

```bash
.venv_research/bin/python scripts/research/motion2scene_sample_event_beams.py \
  --registration /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/registration.json \
  --cell /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/runs/local_8321/2400/result.json \
  --case 41007_event0 --seed 9321 --count 8 \
  --out /home/linjiw/research-data/groot-wbc/m2s-event-scaling-v1/fresh-sampler
```

Output directories must be new. Saved successes and failures are preserved; no earlier
source, protocol, checkpoint or exported evidence is replaced.

## Validation and provenance

The frozen registration hash is
`sha256:aceab9a9b4340500c563899b2b7a44c5003906d0a7cfeaf0b4ea562ccae8fbd9`.
The [export manifest](assets/event-manifest.json) pins the renderer and all seven exported
result/registration/sampler/support/figure artifacts. The renderer rechecks all checkpoint
and raw-clearance hashes and recomputes each acceptance count before export. Older study
sources and exports are unchanged.

All **69 impacted tests pass**, including new tests for horizontal translation invariance,
event preservation, training-only normalization, anchor bounds, finite gradients, and
local versus pooled event sensitivity:

```bash
.venv_research/bin/python -m pytest \
  tests/dataset_generation/test_capsule_box_exact.py \
  tests/dataset_generation/test_capsule_box_torch.py \
  tests/dataset_generation/test_motion2scene_inverse.py \
  tests/dataset_generation/test_motion2scene_uncertainty.py \
  tests/dataset_generation/test_motion2scene_margins.py \
  tests/dataset_generation/test_placement_certificate.py \
  tests/dataset_generation/test_motion2scene_carriers.py \
  tests/dataset_generation/test_motion2scene_events.py \
  tests/dataset_generation/test_motion2scene_beam_teacher.py \
  tests/dataset_generation/test_motion2scene_repeatability.py \
  tests/dataset_generation/test_motion2scene_timing_diagnostic.py -q
```

Black and Ruff (`E,F,I`) pass for the new event module, training driver, sampler, renderer
and tests. `git diff --check` passes. Chrome checks at 1440×1000 and 390×844 decode both
new figures with no page JavaScript exceptions or horizontal document overflow. Both
figures and the mobile layout were visually inspected. All 101 local page links/anchors
and 47 output hashes across the seven follow-up export manifests pass.

Post-run environment inventory: Python 3.11.16, NumPy 2.4.6, PyTorch 2.11.0+cu128,
Matplotlib 3.11.1. Fitting and checking used CPU; the installed CUDA build does not imply
GPU use. Package binaries were not hash-pinned, so cross-environment bitwise reproduction
is not asserted. Page updates are local; no push or publication was performed.
