# Fresh source transfer audit v1 — prospective protocol

2026-09-05. Register this file and the implementation before generation. This is a
reference geometry experiment, with no controller execution or new training.

## Question and proposed contribution

Does learned motion conditioning provide useful initialization for bounded obstacle
search on previously unobserved source motions? The proposed contribution is a
motion-conditioned proposal plus a fixed geometric correction and independent
checker. The comparison isolates initialization from search using uniform pattern
search. The independent unit is a source motion, not a derivative or sampled beam.

## Acquisition and stop rules

Attempt exactly eight neutral source identities, generation seeds 43001–43008, in
that order. Before launch inventory JSON metadata under the repository, sibling
motion2scene repository and research-data/groot-wbc; record hashes and reject any
integer occurrence of these seeds, including inferred generation-report ranges.
This establishes freshness within those inventoried local records only.

Use the existing neutral prompt, cached encoding, Kimodo-G1-RP-v1 snapshot
3020ad8c419c244e0429d360163730c63c4ed011 and existing CSV generator, all hash-pinned.
Generate 4.0 seconds, 30 Hz, 100 denoising steps, independent_per_prompt seed design.
Use CPU with four BLAS/OpenMP threads, CUDA hidden, local checkpoint directory and
Hugging Face offline. The GPU was occupied at preflight; no GPU use is authorized by
this registration. Cap the whole generation subprocess at 1800 seconds. Do not
change device, prompt, seed, steps or model after observing results. Retain partial
outputs, errors and logs. No replacement, retry, or continuation after timeout.

Require exactly eight complete CSV/sidecar pairs with matching seed/model/prompt,
120 rows and 36 finite columns. A generator exit status alone is insufficient.
Construct both local_crouch derivatives for every generated source, at stations
0.35 and 0.65, drop 0.055 m and window 0.18; no new operator. Record all attempted
constructions and failures, including Q0 embodiment, Q1 self-intersection, frozen
route_semantics_v2_preregistered valid_straight and endpoint/route preservation.
Gate the persisted targets, and compute FK from the persisted references. Cap
construction at 600 seconds. Generation incompleteness or any reference gate failure
stops inference for the entire acquisition; retain the full funnel. No qualifying
subset analysis. An infrastructure failure is recorded separately from a measured
motion gate failure. All eight source candidates and all derivatives are permanently
excluded from fitting, normalization, distillation and checkpoint/method selection.

## Frozen inference and audit

Reuse all three all8/1200 checkpoints (8421, 8422, 8423), including original feature
normalization, from source-phase-v1. No refitting. For each of 16 cases and three
checkpoints, sample 256 learned draws with draw seed checkpoint seed + 80000 and
retain the same 136-candidate prefix as the previous station study. Common random
numbers across cases are intentional. Raw outputs use the first eight. Rank selects
eight by stable descending margin slack. Gradient, probe_refine, multistart and
pattern start from those same eight and use the frozen implementations. The seventh
arm, uniform_pattern, uses eight float64 Torch uniform draws over station [0.10,0.90]
and beam height [1.10,1.45], with seed checkpoint seed + 90000, followed by exactly
the same frozen pattern search. Uniform draws are also shared across cases for a
checkpoint. There is no learned/uniform output-level correspondence.

Each assisted arm spends 4,624 capsule/placement/alternative queries per job: 136
candidate evaluations, 17 search perturbations, two motions. Raw spends zero search
queries. Local searches retain original trust bounds ±0.05 station and ±0.03 m
height, 17 evaluations, fixed domains and earliest-tie incumbent selection. Frozen
beam dimensions, route convention, proxy capsules and 1 cm target/neutral margins
are unchanged. Search CPU float64, two threads, deterministic algorithms, 1800 s cap.

Independently audit all eight outputs per arm/job using 113 registered perturbations
and NumPy full capsule geometry; crosscheck the first two outputs against Torch at
all 113 perturbations, finite values and max difference ≤1e-8 m. Audit cap 1200 s.
This gives 48 jobs, 336 arm/job rows, 2,688 outputs; each arm has 384 outputs, each
source 48 per arm. Query totals: search 1,331,712; NumPy audit 607,488; crosscheck
151,872. These finite tests are not continuous collision or tracking guarantees.

## Predictions, metrics and interpretation fixed before acquisition

Primary prediction: learned pattern has strictly more accepted outputs than uniform
pattern on each of the eight source motions. This demanding directional predicate
is descriptive; it is not a significance test. Report all eight paired source-count
differences and their minimum, even when the predicate fails.

Secondary predictions: learned pattern has no source-count loss versus gradient and
strictly greater total accepted count; it also loses zero corresponding gradient
successes. Ties fail the strict-gain clause. Report accepted counts for all seven arms,
all sources, both event phases and all checkpoint seeds; target and neutral failures;
paired rescues/losses only for methods sharing learned initial draws; accepted-bin
counts as output diversity, not feasible-support coverage. Do not select a method
or tune against these observations. Report the eight-candidate acquisition funnel
before any inference denominators. Failure to acquire all sources leaves predictions
unassessed, not false or successful.

Account for generation, bank construction, inherited training, feature/model setup,
learned and uniform sampling, search and independent audit separately. Per-row shared
sampling times must not be multiplied across methods. Include full-stage elapsed
costs and state that timing is a local measurement on a shared machine.

## Claim boundary and next work

A successful audit supports fresh-source transfer only within generated straight
walking, the existing crouch operator and this two-parameter beam family. CPU
acquisition is a backend change from the old GPU bank and must be disclosed. The
29-capsule/13-link proxy and discrete time samples do not certify the full imported
robot or continuous motion. No physical execution, natural-scene likelihood,
multi-object feasibility or humanoid-wide collision guarantee follows.

If transfer holds, register continuous placement/time verification next, followed by
obstacle-present execution with the frozen controller. Independently, use geometric
search on training sources to supervise distillation aimed at fewer inference queries;
keep this entire audit pool excluded. Motion alone underdetermines obstacle scenes:
self-supervised geometric constraints define admissibility under stated assumptions,
not the unique natural scene distribution.
