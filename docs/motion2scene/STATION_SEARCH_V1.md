# Fixed-budget station search, v1

Prospective development protocol, 2026-09-05. Freeze protocol, implementations, source
bank and selected checkpoints before search. No model fitting, new motions or edit
operators, controller changes, physics, scene admissions or larger trust region.

## Question and contribution

The previous fixed fresh batch accepted 5/8; its three refusals never changed station
and reached the downward height bound. Test whether local station exploration helps
at the same query budget. This adds a reusable search component after the small event
model. The plateau explanation is a hypothesis, not an established gradient diagnosis.
All assisted methods have access to target AND neutral reference geometry at inference.
The neural forward pass still consumes target motion only.

Use the same 16 excluded cases: all four validation and four test parents at phases
0.35 and 0.65, with three frozen all8/1200 checkpoints 8421–8423. Main panel: fresh
256-draw batches with seed checkpoint_seed+60000, use first 136 for ranking and first
eight for local methods. This gives 48 main jobs; sources are already observed, so
fresh draws are NOT fresh sources. Keep source roles and phase exclusion unchanged.
A separate replay job uses checkpoint 8421, case 42007_event3, draw seed 9521 and the
previous sampler's exact eight inputs. Do not pool replay with the main panel.

## Fixed methods and costs

Six output arms per job: raw; learned ranking of 136 candidates; original bounded
16-step Adam refinement; and three new local methods below. Each returns exactly eight
outputs, including failures. All five assisted methods use 17×8×17×2 = 4,624 reference
minima queries per job (17 search placements: nominal and 16 corners). Raw has zero
search queries. Count duplicates at clipped boundaries and repeated starting evaluations.
No retries, budget-dependent replacement or audit-based output selection.

All local trajectories remain inside the ORIGINAL proposal's station ±0.05, height
±0.03 m box, intersected with the global [0.1,0.9]×[1.1,1.45] domain. Rank/select by
min(target_min−0.010, −neutral_max−0.010), stable ties retaining earliest candidate.
Adam uses lr 0.02 in normalized global-domain coordinates, summed squared margin
hinges, gradient clipping 10 and the same callback as the frozen baseline.

- **probe_refine**: query shifts 0, −0.025, +0.025, −0.05, +0.05 at initial height.
  Select the best per input, reset Adam and take 11 steps with 12 evaluations.
  Total 5+12=17; the chosen probe is deliberately queried again and counted.
- **multistart**: query −0.025,+0.025 stations at initial height, then reset Adam
  independently at shifts 0,−0.05,+0.05, taking four updates/five evaluations per branch.
  Total 2+3×5=17. Select across every candidate, not just branch endpoints.
- **pattern**: query initial, then four rounds of four coordinate probes around the
  current best candidate. Round scales are 1,1/2,1/4,1/8; offsets are ±0.05×scale
  station and ±0.03×scale height. All four probes share the pre-round incumbent.
  Total 1+4×4=17, no gradients. Select across every candidate.

CPU float64/two threads, deterministic algorithms. Report query and wall-time costs
separately; no matched-FLOP or real-time claim. Inherited model training is not free.

## Audit, predicates and evidence limits

Audit all six outputs per job with independent full-sampled-motion NumPy capsule–box
queries at the prior 113 placements, including two-proposal Torch cross-checks at every
placement. Stop for nonfinite values, disagreement >1e-8 m, input/hash mismatch, missing
jobs, out-of-bounds search or cap breach. Save all trajectories and all outputs.
294 audit rows / 2,352 outputs total: main 288 rows / 2,304 outputs, replay six rows /
48 outputs. Main validation and test each have 192 outputs per arm, 48 per parent.
The source parent is the independent unit; seeds, phases and draws are repeated measures.

For each new method, separately predeclare: (1) no test parent loses valid count versus
original gradient refinement AND pooled test count strictly increases; (2) no same-index
original-gradient successes are lost on the independent audit; (3) replay valid count
exceeds its original 5/8. Publish all nine predicates without selecting a winning method
from validation/test outcomes. Count condition (1) is descriptive, not statistical
noninferiority. Replay is a targeted known-failure diagnostic, not generalization evidence.
Also report ranking/raw controls, target and neutral failure types, per-seed counts,
paired rescues/losses, search time and descriptive accepted-bin occupancy (.02 station,
.01 m height). Bins are not feasible-support coverage.

Caps: 1,800 s search and 1,200 s audit, zero GPU. No continuation with changed seeds or
settings after a scientific failure. Positive witnesses establish finite reference
feasibility only; unsuccessful search cannot prove emptiness. Continuous placement/time,
imported colliders, obstacle-present execution and downstream policy utility remain open.
