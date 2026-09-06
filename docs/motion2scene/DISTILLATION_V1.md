# Training-source search distillation v1: registered development pilot

2026-09-05. Freeze this protocol, driver and new loss/search helper before teacher
construction. CPU float64, two threads, no GPU or physical execution. Keep every
prior protocol, model, source bank and export unchanged.

Question: can geometric search on training motions improve amortized raw proposals
and reduce online correction? This is a development experiment on old 410xx/420xx
sources. Do not load or evaluate the 430xx fresh-audit pool. No new motion generation.

## Teacher and roles

Use the 24 original training derivatives: sources 41001–41004 and 42001–42004,
stations 0.25, 0.50, 0.75. Use the frozen all8/1200 checkpoint 8421 as a common teacher
initializer for all students. Every student retains its own original checkpoint
(8421, 8422, 8423) and normalization. The common teacher is an explicit asymmetry.

For each training case in stored order, create exactly eight starts: four learned
(the first four of a 256-draw batch), two uniform, and two stratified, one in each
station half [0.10,0.50] and [0.50,0.90]. Heights uniform [1.10,1.45]. RNG seeds are
118421 + case index for learned starts and 138421 + case index for the uniform and
stratified draws, sequentially. Correct with the frozen 17-evaluation pattern search,
original trust bounds and 17 placement offsets. Independently audit all eight outputs
at all 113 offsets, crosschecking the first two against Torch (finite, error ≤1e-8 m).

Retain one earliest accepted output per 0.02-station / 0.01-m height bin. Preserve
all rejected outputs and empty teacher sets; no replacement or success quota. Empty
sets contribute zero set loss and remain in geometric training. Stop if all sets are
empty. This is synthetic supervision from motion geometry, not natural-scene labels.
24 jobs: 110,976 search + 43,392 audit + 10,848 crosscheck = 165,216 teacher queries.
Cap teacher stage at 600 s. Training eligibility is confined to this reference-only
pilot, with no original-bank admission or execution qualification.

## Learning

Warm-start each of three existing checkpoints; optimizer reset to Adam, lr 0.0003,
gradient clipping 10. Never refit normalization or alter architecture. Four arms:

- geometry600: continue the existing geometric/preference objective for 600 updates.
- geometry2666: same objective for 2,666 updates, a stronger query-budget control.
- set600: 600 updates using only the teacher-set loss (skip updates for empty sets).
- hybrid600: 600 updates, existing objective plus 10 times teacher-set loss.

The set loss is a weighted energy distance in normalized station/height coordinates,
using one reparameterized draw per existing anchor and softmax mixture weights:
2 E[d(X,Y)] − E[d(X,X')] − E[d(Y,Y')]. Y is uniform over retained teacher points.
Use d(x,y)=sqrt(||x−y||²+1e-12) for finite derivatives at coincidence. This is sample
matching, not likelihood fitting; it avoids pretending the bounded piecewise decoder
has an implemented density. Four RNG streams per original seed: proposal +210000,
pose +220000, training order +230000, set draws +240000. Common streams across arms;
set draws do not advance geometry streams. Use all 24 training cases each shuffled
cycle. Record losses, empty-set skips, queries, fitting times and checkpoint hashes.

Each geometric update spends 8 proposals × 5 offsets × 2 alternatives = 80 queries.
Charge the entire common teacher cost to EACH student when comparing against the
query-budget control, although the actual experiment computes it once. Hybrid cost
per student: 165,216 + 48,000 = 213,216 queries; geometry2666 uses 213,280, 64 more.
Set600 uses teacher queries but no online training geometry. Geometry600 isolates
adding set supervision at the same update count, not the same total query count.
Cap all fitting at 1,800 s. Save only the registered final checkpoints; no selection.

## Development evaluation

Use the 16 old excluded cases (eight 410xx/420xx validation/test sources, stations
0.35 and 0.65), labeled observed development throughout. Three fitting seeds, eight
outputs each: 48 jobs, 384 outputs per arm, 48 per source. Generate 256 scene draws
per model/case with draw seed original seed +250000, retain first eight.

Nine arms: original raw, geometry600 raw, geometry2666 raw, set600 raw, hybrid600 raw,
original pattern17, hybrid pattern17, hybrid pattern5, uniform pattern5. Pattern17
uses the frozen routine. Pattern5 evaluates the original start and one full round of
four coordinate probes (station ±0.05, height ±0.03), retains the earliest maximum
margin-slack incumbent, and keeps the original trust/domain bounds. It computes
exactly five evaluations, not a prefix extracted after a 17-evaluation run. Uniform
starts use original seed +260000 and the same full station/height domain.

Pattern17 costs 4,624 and pattern5 costs 1,360 search queries per job (70.6% reduction).
Raw uses zero. All 432 arm/job rows and 3,456 outputs receive the full independent
113-placement audit, with the same two-output Torch crosscheck. Search and audit
queries, setup, sampling and wall time remain separate. Evaluation cap 1,800 s.

## Prospective descriptive predictions

P0: every training case yields at least one retained teacher placement (report failure;
only an all-empty teacher stops fitting).
P1: hybrid raw has no source-count loss versus original raw and strict pooled gain.
P2: hybrid raw has no source-count loss versus geometry2666 raw and strict pooled gain.
P3: hybrid pattern5 has no source-count loss versus original pattern17 (ties pass),
with the fixed 70.6% reduction in search queries. No total-cost claim follows alone.
P4: hybrid raw's mean accepted-bin count per job is at least the original raw count.

Report every source, both phases, all fitting seeds, target and neutral failures,
all five predicates, and actual resource cost. Fitting seeds/derivatives/draws are
repeated measures, not independent source samples. Do not tune or rerun failures.
A successful pilot would motivate a separately registered fresh-source confirmation;
it would not establish fresh transfer, continuous collision, natural scene frequency,
complex multi-object generation or physical execution.
