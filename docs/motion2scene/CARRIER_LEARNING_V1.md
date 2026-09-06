# Registered reference-only test of motion conditioning

## Why this experiment

The previous [margin experiment](MARGIN_LEARNING_V1_RESULT.md) improved raw validity on
one motion carrier. It cannot show that the learned motion input matters. The existing
eight-carrier reference ladder collection also places every crouch at station 0.55,
which permits an especially weak constant-placement solution. This experiment introduces
three separated event locations using the existing operator, then excludes complete
source carriers from generator fitting.

This is a new **reference-geometry development diagnostic**, not admission of the old
ladder bank. The original collection's `hallucinator_training_allowed: false`, zero
Q4 admissions and failed execution results remain intact. We do not train a downstream
policy or relabel reference-side checks as controller qualification. Prior carrier
references/measurements have been inspected; the excluded groups are not fresh
confirmatory test data. There are only two test source carriers.

## Data construction and groups

Freeze the eight straight-walking base CSVs from the original controlled-duck registration,
seeds 41001–41008, before construction. For every base, call the current existing
`local_crouch` operator at route stations 0.25, 0.50 and 0.75, target drop 0.055 m,
window 0.18. Pin the operator, FK, robot MJCF, reference gate and geometry sources.
There are 24 target cases and eight unique neutral parents; no new edit operator or
physics execution is introduced. Event windows stay within the episode. Require unchanged
start/final poses, root horizontal route/orientation, frame count, and finite paired data.

Run the existing Q0/Q1 reference gates on every constructed target and neutral. Retain
all results in a carrier registry. Any Q0/Q1 failure stops training on this bank without
replacing the case or weakening a gate. The model bank is built from the persisted CSVs,
with all recorded reference frames and complete capsule axes. Canonicalize both members
using the neutral parent's initial XY translation; never align them independently.

Split by root source seed **before generating or fitting**:

| Split | Source seeds | Event cases |
|---|---|---:|
| train | 41001–41004 | 12 |
| validation, report only | 41005–41006 | 6 |
| excluded test | 41007–41008 | 6 |

All derivatives, neutral counterparts and future repeats inherit their parent's group.
No validation-based checkpoint selection, tuning or early stopping. Carrier 41002, used
throughout earlier development, is in training. Every candidate retains false execution
qualification and scene-admission flags. No obstacle-present result follows.

## Generator and controls

Retain the four-component correlated Gaussian mixture, temporal whole-body encoder,
fixed motion costs and explicit margin loss from the previous no-KL candidate. Expand
station output bounds to [0.10,0.90] to cover the new events; height stays [1.10,1.45] m.
Beam depth/width/thickness stay 0.10/1.20/0.10 m. Event station, carrier ID and split are
metadata only; the network sees target capsule midpoint/axis/radius sequences.

Train three modes at optimizer seeds 8221/8222/8223:

- `conditioned`: correct target-motion features.
- `constant`: the same network and parameters, with all-zero motion features.
- `shuffled`: a fixed one-position cyclic derangement of the ordered 12 training-case
  feature sequences, while geometry/costs remain attached to the original target case.
  The derangement uses no excluded-group data. At evaluation use the real target input.

All modes retain the same deterministic route-based scene builder. Consequently the
constant control tests learned motion features beyond that builder; it is not entirely
motion-independent in world coordinates. The 1200-update budget gives 100 target-case
visits each, in independently shuffled 12-case epochs. Adam learning rate 0.002, CPU
float64, two threads, norm clipping 10. Each update draws two proposals from every
mixture component and checks nominal pose plus four distinct corners of the prior
16-corner pose domain. Loss = preference + 5 × target barrier + 5 × interference barrier;
KL weight zero. Both margins remain 10 mm. Use separate deterministic RNGs: seed +
10000 proposals, +30000 corners, +40000 case order.

Save final checkpoints from all nine amortized cells. No one-carrier checkpoint is
fine-tuned on excluded carriers.

For every excluded validation/test case and every optimizer seed, fit a separate
unconditioned mixture for 300 updates with Adam learning rate 0.03. This is a disclosed
fixed-budget per-motion search comparator, not an established converged optimum. The
larger direct parameter learning rate addresses the much smaller parameterization;
it is fixed before this study and is not claimed optimal. There are 36 direct cells.
Each direct cell uses the same geometric loss and per-step query count, but gains target
geometry during adaptation. Report its cost rather than implying zero-shot equivalence.

Also evaluate a standard-normal latent prior, and a `swapped_input` control using the
conditioned checkpoint: rotate early→middle→late→early input motions within each excluded
carrier, keeping evaluated target geometry unchanged. The latter requires no new training
and isolates event identity from carrier identity.

## Evaluation and registered predictions

For each checkpoint/case make the full 256-proposal mixture draw with RNG seed + 20000,
then audit the first 32 without resampling. Use the same draw schedule for swapped
inputs; draw 256 standard-normal latents then take 32 for the random-prior comparator.
No event-location label or analytic feasible interval supplies scene targets.

Check all proposals at the previous 81 grid plus 32 fixed interior placements: x/y ±2 cm,
z ±1 cm and yaw ±0.02 rad. This is a reused development stress domain. A scene passes
only if every placement clears its reference target and interferes with its reference
upright by 10 mm. The independent NumPy query evaluates all frames/axes; cross-check
the first four proposals of every cell/case at every placement against PyTorch and
abort if maximum disagreement exceeds 1e-8 m.

Six evaluated modes × four excluded carriers × three events × three optimizer seeds
= **216 rows**, 32 proposals each. Save every proposal and raw 32×113×2 clearance array.
Report per-event/seed counts, target and upright conditions, and nominal versus all-113
validity. Aggregate each mode within each source carrier across three events and three
optimizer seeds: denominator 288 proposals, while the independent unit remains the
source carrier. Preserve event and seed variation; do not turn them into extra carriers.

Primary development prediction: conditioned raw all-placement validity strictly exceeds
both constant and shuffled-training controls on **each** test carrier, 41007 and 41008.
Secondary prediction: swapping input event lowers validity on each test carrier.
These are predeclared descriptive criteria, not statistical significance tests. Report
validation separately and report direct search/random baselines regardless of outcome.
If conditioning fails, do not claim learned input dependence or amortization benefit;
use the result to distinguish representation, data or optimization bottlenecks.

A positive result would still concern three edited event locations, reference capsule
geometry, one robot and a beam family. It would not show novel event-location interpolation,
new obstacle types, real-world scene-frequency learning or successful executed motion.
Accepted scenes still require the separate placement/time/body geometry contracts.

## Budget and reproduction

Registration precedes construction. Bank and registry hashes are frozen before fitting.
Allow 3600 seconds each for construction, training (9 amortized + 36 direct cells), and
analysis. No GPU training or new physics. An engineering smoke shares the constructed
bank but uses only seed 8221, 12 updates per amortized mode, and two direct updates on
one case; it writes a separate `smoke/` directory and is excluded from results.
Hash mismatch, reference gate failure, incomplete cell set, nonfinite loss/gradient or
query disagreement stops the relevant stage and preserves existing artifacts.

```bash
.venv_research/bin/python scripts/research/motion2scene_carrier_learning.py register \
  --carriers /home/linjiw/motion2scene/experiments/registrations/E1_CONTROLLED_DUCK_LADDERS_V1.json \
  --protocol docs/motion2scene/CARRIER_LEARNING_V1.md \
  --out /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1
.venv_research/bin/python scripts/research/motion2scene_carrier_learning.py build \
  --registration /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/registration.json
.venv_research/bin/python scripts/research/motion2scene_carrier_learning.py smoke \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/manifest.json
.venv_research/bin/python scripts/research/motion2scene_carrier_learning.py train \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/manifest.json
.venv_research/bin/python scripts/research/motion2scene_carrier_learning.py analyze \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-carrier-learning-v1/manifest.json
```

Output paths must be new; no frozen study is overwritten.
