# Source diversity and withheld event phases, v1

Prospective development protocol, 2026-09-05. Freeze this document and driver in a
registration before constructing targets or fitting. No new physics, original-bank
admission, controller change or edit operator is involved.

## Question and independent units

Does adding distinct neutral motion sources help a fixed local event model, including
on event locations absent from training? Source-generation identity is the independent
unit. Draws, optimizer seeds and five derivatives of one source are repeated measures.
All sources were previously observed in other development studies: this is not a fresh
confirmatory test and does not estimate a population scaling law.

Preserve existing roles: 41001–41004 train, 41005–41006 validation, 41007–41008 test.
For this separate reference-only study assign 42001–42004 train, 42005–42006 validation,
42007–42008 test, before constructing targets. The 420xx sources were historically
held out for an already completed route-retention study; these new development roles
do not retroactively alter that study or its qualification flags.

Construct five local_crouch targets per parent, 16 parents / 80 targets total, with
unchanged drop 0.055 m, window 0.18, registered MJCF and 30 Hz. Training uses stations
0.25, 0.50, 0.75 only. Stations 0.35 and 0.65 are withheld from all fitting and
normalization. Save every target at ten decimals, reload before FK and screening;
retain neutral/target gates and complete source hashes. Any Q0/Q1 failure, endpoint
change, nonfinite value or lineage mismatch stops training without replacement.
A failed gate is an acquisition result, not evidence of model performance.

## Fixed comparisons

Use the previous EventMixture width 32 (12,118 parameters), its target-only features,
training-only normalization, loss_for (preference + 5× each explicit margin), eight
stratified proposals, nominal plus four random uncertainty corners, Adam 0.001,
float64 CPU, two threads and gradient clipping 10. No scene labels, teacher samples,
checkpoint selection or hyperparameter tuning on excluded sources/phases.

| Arm | Training parents | Saved updates |
|---|---|---|
| base4 | 41001–41004 | 600, 1200 |
| add_a6 | base4 + 42001–42002 | 600, 900 |
| add_b6 | base4 + 42003–42004 | 600, 900 |
| all8 | base4 + 42001–42004 | 600, 1200 |

These are two nested paths 4→6→8, sharing their endpoints. Six-parent subsets are
predeclared halves, not independent replications or random draws from a population.
Use optimizer seeds 8421, 8422, 8423 for each arm (12 fits / 24 snapshots).
Compare all arms at 600 for equal query budgets; compare 4/600, 6/900, 8/1200
for exactly 50 visits per training case. The base4/1200 endpoint is an extra-update
control for all8/1200. Each update makes 8×5×2 reference-clearance minima queries.
Keep these compute contracts separate; source diversity and update count can interact.

## Independent audit and predeclared predictions

Evaluate all eight validation/test parents, each at all five phases. Each cell draws
256 proposals with seed optimizer_seed+20000 and audits the first eight, with no retries.
Use the same 113 placement offsets, independent NumPy full-sequence capsule–box
clearances, target >= +0.010 m and neutral <= -0.010 m at every offset. Cross-check
the first two proposals at all offsets with Torch; disagreement >1e-8 m stops analysis.
Store all 960 raw rows (7,680 proposals), including failures, and per-parent counts.
Report seen (72 draws per parent/cell) and unseen phases (48) separately, plus each
optimizer seed. Report both clearance-failure types, which can overlap. Do not infer
continuous-time, imported-robot or continuous-placement guarantees from this audit.

Directional hypotheses: all8/600 improves over base4/600, and all8/1200 improves over
base4/600, on **each of four test parents at unseen phases**. Report each predicate
and both six-parent paths whether or not they agree. Additional control: compare
all8/1200 against base4/1200 on that same per-parent endpoint. Ties do not count as
improvement. Validation is descriptive; no model or budget is selected from it.

CPU caps: construction 600 s, training 3600 s, analysis 1800 s, no GPU. Record failures,
wall time, normalization and all checkpoint/sample/raw hashes. Do not rerun failed
scientific outcomes with a different seed. Future execution qualification requires its
own registered obstacle-present experiment and full body/temporal collision contract.
