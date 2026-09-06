# Station-balanced analytic-teacher distillation

Registered 2026-09-06 before constructing this teacher or fitting its students.
The previous 58 retained teacher endpoints cover 33.6% of independently mapped
training station bins on average. The mechanism tested here is broader verified
station support, with the width-32 model, normalization, geometry loss and original
training/development source roles retained. This is a reference-only development
experiment, independent of the concurrently registered physics pilot.

Training uses the original 24 cases from sources 41001–41004 and 42001–42004.
Evaluation uses the original 16 development cases from 41005–41008 and 42005–42008.
All 430xx sources remain excluded. No development reference map, achieved trajectory,
contact record or execution qualification informs teacher construction or fitting.

For each training case, reuse the frozen analytic global search proposal mechanism:
20 station centres, six height offsets around their analytic nominal heights and 16
ranked nominal probes; 136 candidates at 17 search offsets. Independently audit the
highest-search-slack candidate at each of all 20 stations, including negative-slack
winners, at 113 placements. Retain one passing witness per station with equal weight;
retain search failures, independent failures and unresolved stations in raw artifacts.
This is geometry-informed bounded height exploration, not complete feasible-set mapping.
An empty teacher case stops fitting and remains a recorded failure.

Teacher cost is exactly 24 × (4,624 search + 4,520 audit + 452 crosscheck) = **230,304
whole-motion queries**. Audit two winners with Torch as well as all 20 with the
independent evaluator. No second teacher search follows an audit rejection. Do not use
the existing grid map as teacher input; compare support only after freezing the bank.

Fit three seeds 8421–8423 from their original all8/1200 checkpoints:

- Coverage hybrid: 600 updates, existing geometry loss plus 10 × existing energy-distance
  set loss on the equal-station teacher. Cycle cases through the same seeded random
  permutations as the old hybrid; 25 updates per case. Cost 48,000 fitting queries.
- Matched geometry continuation: ceil((230,304 + 48,000)/80) = **3,479 updates**,
  same learning rate, optimizer, initialization and geometry sampling streams. Cost
  278,320 queries, 16 more than the teacher-plus-coverage recipe's 278,304.

Retain original and existing hybrid checkpoints as frozen comparators; no architecture,
loss coefficient or optimization sweep. The deterministic analytic distinct solver's
127/128 result remains a serious comparator, including its preserved audit rejection.
Shared teacher construction is paid once in actual aggregate compute; also show its
full cost charged to each student when making the conservative matched-control comparison.
Inherited original fitting and old-hybrid teacher/fitting costs remain additional.

Evaluate original, old hybrid, new matched continuation and coverage hybrid on identical
eight-request draws per case/seed (seed + 250000, existing 256-draw/first-eight convention),
at budgets 0/5/9/17 and adaptive 5→9→17. Fixed searches use the exact prefix of the
existing pattern procedure. Adaptive search stops each proposal only on search-side
margin success at 5 or 9 evaluations; final independent audit never supplies feedback.
Measure actual per-output evaluations and latency. Each job pays 1,808 independent audit
and 452 crosscheck queries. Raw has zero optimization queries but still pays verification.

Before reading development maps, freeze every generated output and audit outcome.
Then report verified yield, ordinary accepted station/height-bin occupancy and the
fixed reference-relative station coverage at eight requested outputs, by source and
seed. Bootstrap paired source means (eight groups, 10,000 draws, seed 6262); interpret
intervals descriptively. Keep all audit failures, not only accepted-output diversity.

Predictions: P1 teacher mean reference station support exceeds the old teacher;
P2 coverage raw has at least the matched continuation's pooled yield, higher mean
reference station coverage and no source loses more than 2/48 accepted proposals;
P3 at some fixed budget ≤9, coverage hybrid matches original17 yield and coverage with
the same source guardrail. Report each predicate independently. A coverage/yield tradeoff
does not satisfy P2/P3; do not choose a favorable metric after results.

Limits: two CPU threads, no GPU use, teacher ≤600 s, fitting ≤1,800 s, evaluation ≤3,600 s.
Record preprocessing, teacher, fitting, sampling, search, audit and crosscheck costs and
latency distributions. If predictions fail, retain the existing learned-pattern method
and analytic comparator; do not enlarge the model or retune the teacher in this lineage.
