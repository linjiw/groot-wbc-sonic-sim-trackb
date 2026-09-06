# Matched-query geometric search and bounded refinement, v1

Prospective reference-only development protocol, 2026-09-05. Freeze protocol,
implementations and sources before running. No fitting, new motion, new edit operator,
controller change, physics run or Q4 admission.

## Contribution and question

Add a reusable bounded geometric refinement layer after the existing learned proposer.
Test whether its initialization supplies an advantage over uniform geometric search,
and whether extra learned draws suffice without gradients. The mechanism changes only
inference; it uses the target AND neutral reference geometries. This additional inference
information/cost must be distinguished from the target-only neural forward pass.
No novelty, real-world distribution, execution or collision-certificate claim is made.

Use all 16 previously excluded cases at stations 0.35 and 0.65: four validation parents
41005,41006,42005,42006 and four test parents 41007,41008,42007,42008. Three existing
all8/1200 checkpoints from source_phase_v1, seeds 8421–8423. This is the previous fixed
equal-visits endpoint, not selection of the better six-source subset. Sources and their
outcomes have been observed; this remains development evidence. Derivatives and draws
are repeated measures, not independent sources. No tuning on these cases.

## Arms and resource contract

Each job is a checkpoint × motion, 48 jobs. Output exactly eight proposals per arm,
including failures. No retries, replacement outputs, stopping on success or hidden
selection using the independent audit.

- **raw**: reuse the prior eight neural proposals and their saved 113-placement audit.
- **learned_rank**: draw 256 with seed checkpoint_seed+20000 and use the first 136;
  score nominal plus all 16 placement corners and retain the best eight.
- **uniform_rank**: 136 uniform draws over station [0.1,0.9], height [1.1,1.45],
  NumPy default_rng(checkpoint_seed+50000), ranked in the same way.
- **learned_refine**: refine the same eight raw learned proposals for 16 Adam steps.
- **uniform_refine**: refine the first eight of the uniform candidate pool identically.

Rank by min(target_min−0.010, −neutral_max−0.010), over the 17 search placements;
stable ties keep earlier indices. Refinement uses the sum of the two normalized squared
hinge violations, Adam 0.02 in global-domain normalized station/height coordinates,
gradient clipping 10, float64 CPU/two threads. Clamp each trajectory to its initial
station ±0.05 and height ±0.03 m, intersected with the global domain. Evaluate before
updates and after step 16: 17 batches × eight candidates. Keep the best incumbent for
each initial candidate by the same ranking slack, ties retaining the earliest iterate.
Save all 17 iterates and clearances, not only winners. No claim that this preserves
acceptance on search placements absent from the 17-point set.

All four assisted arms make 136×17×2 = **4,624** reference-clearance minima queries
per job. These match query counts, not FLOPs or latency: gradients are more expensive.
Raw uses no search queries. Report actual preprocessing/sampling/search/audit wall time,
raw checkpoint training cost as inherited (not rerun), and queries per accepted output.
Report accepted occupied station/height bins (0.02 / 0.01 m) descriptively; duplicates
or near-duplicates do not count as independent scenes or feasible-support coverage.
Uniform rank/refinement are geometric baselines, not exhaustive analytic search.

## Independent audit and predictions

Audit every assisted output with full-sequence independent NumPy capsule–box geometry
at the original 113 placements. Cross-check the first two proposals at all 113 with
Torch; stop if error exceeds 1e-8 m or any value is nonfinite. Retain 192 new rows /
1,536 new outputs and the 48 raw rows / 384 reused outputs. Source-level denominators
are 48 draws per arm; validation/test each total 192. Both 10 mm margins must pass at
every placement. Report target failures and neutral-interference failures separately;
they can overlap. Compare each refined learned proposal with its same-index raw input,
reporting rescued and newly failed outputs on the independent audit.

Directional predicates (strict improvements, ties fail): learned_refine improves
valid count over raw on each of four test parents; learned_refine beats uniform_refine
on each test parent; learned_refine beats learned_rank on each test parent. Separately
report whether 42007/0.35 gains any valid refined learned output (prior raw 0/24).
Publish all predicates regardless of outcome. Failed search is not an emptiness proof;
positive valid witnesses establish only feasibility under the finite reference audit.

CPU cap: 1,800 s search and 1,200 s audit, zero GPU. Fail on missing/changed inputs,
wrong draw prefix, nonfinite gradients, incomplete rows or cap breach. Preserve failed
runs. Hash-pin the prior bank, result, selected checkpoints, driver and refinement layer.
Prior raw arrays and source implementation pins are verified before use. Continuous
placement/time, imported body geometry, obstacle-present execution, natural scene
frequencies and downstream training benefits remain unvalidated.
