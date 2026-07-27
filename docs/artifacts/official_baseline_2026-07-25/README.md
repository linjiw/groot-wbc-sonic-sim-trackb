# Official SONIC Baseline vs. ZPD Pilot

## Baseline reconstructed

The audit compares this fork at `f7dc739` with official upstream
`NVlabs/GR00T-WholeBodyControl` at `4141c342` (observed 2026-07-25).
The released SONIC run is a three-encoder G1/teleop/SMPL PPO training stage on
filtered BONES-SEED, initialized from an earlier checkpoint. The checkpoint's
embedded config records 4,096 environments, a 250-iteration motion reload
period, and the official adaptive motion sampler. Public BONES-SEED contains
142,220 motions; applying the repository's filename filter leaves 129,785
eligible robot motions (8.74% removed). `sample_data` contains only two walking
motions and is not treated as baseline training data.

The official sampler partitions every sequence into 50-frame bins. It estimates
failure rate from exposure-normalized episode counts and non-timeout
terminations, mixes 90% adaptive probability with a 10% uniform floor, applies
sequence-length weights, and can start up to 200 frames before a sampled failure.

## Single-axis comparison

Both arms use the same release weights, fresh optimizer/sampler state
(`+resume=false`), data, seed, 128 environments, PPO settings, 200 iterations,
and `im_eval` metrics.

- Baseline: `adaptive_sampling.signal=failure_rate` with the complete release
  sampler configuration.
- Treatment: `adaptive_sampling.signal=learnability`, where survival has a
  Beta posterior and bin utility is the exact posterior expectation
  `E[p(1-p)]`. Optimism, evidence decay, and family sharing are disabled.

Thus the treatment deprioritizes both mastered bins (`p≈1`) and persistently
impossible bins (`p≈0`), while emphasizing the moving competence frontier near
`p≈0.5`. This creates a curriculum through sampling redistribution rather than
through a manually ordered motion syllabus.

The independent comparator contract in
`scripts/research/run_sonic_paired_experiment.py` hash-binds the initialization,
paired data inventory, commands, and metric coverage. It intentionally does not
require the earlier SIM-D1 headroom gate.

## Frozen pilot data

`configs/research/bones_seed_official_zpd_pilot128.json` was selected before any
policy rollout using only official metadata, the release filename filter, and
category-by-duration strata. It contains all 20 eligible categories and all 8
packages. Selection digest:
`d0f8bc4841cdb9769e17bf2da7014fdc23d15b109d9398ad63fc21f271c5a18b`.

Upstream revisions, sizes, and SHA-256 values are pinned in
`configs/research/bones_seed_official_source_lock.json`. Final flat robot/SMPL
PKLs were validated against their original 30 Hz paired timeline and then
content-hashed by `build_bones_seed_paired_manifest.py`. The materialized cohort
contains 128 pairs, 28,444 robot frames at 30 Hz, and 47,239 runtime SMPL frames
at 50 Hz. Its paired-data digest is
`d82ac458c3a6a3134b47cba2fbc2ef3ae63c099a5d050b190b2d1ad31ea5ad91`;
the manifest-file digest is
`2da4a3cef01be348b7c74735aed3a9b8c9d3285d5605f739dfbdd902deb6dc2a`.

This is a deliberately broad pilot rather than a proportional miniature of the
full corpus. It covers all 20 categories and all 78 non-empty
category-by-duration strata, but represents only 0.0986% of eligible motions and
over-samples rare strata. With 128 environments, all 128 motions are resident;
the 200-iteration run therefore tests curriculum allocation among 1,006 temporal
bins, not the official 250-iteration cross-motion reload behavior.

## Evaluation and current status

The primary evaluator remains the official `ImEvalCallback`: success rate,
MPJPE-L, and MPJPE-G. MPJPE-L plus success/progress is the primary local tracking
readout; MPJPE-G is also reported but interpreted with horizontal root drift.
The callback's duplicate initial temporal-policy inference was removed, and
first-transition/page-symmetry regressions now pass.

All official G1 and seven SMPL archive parts passed their frozen size and
SHA-256 records. Selective extraction, conversion, paired validation, and the
seed-0 executable comparator preflight have passed. The training protocol saves
matched checkpoints at iterations 50/100/150/200 and measures an official
`im_eval` learning curve from the common release initialization through 614,400
policy-environment transitions. Protocol v2 includes progress rate, rejects
bounded/filtered evaluation commands, and evaluates the identical iteration-zero
checkpoint once before reusing its content-identical metrics for both arms. This
removes a zero-budget simulation-noise delta and reduces the curve to nine GPU
evaluations without dropping a paired row.

The common release initialization has now been evaluated over all 128 frozen
motions: success is 94.531%, progress is 96.235%, MPJPE-L is 30.654 mm, and
MPJPE-G is 146.250 mm. Seven motions fail. This strong starting point means
learning-curve AUC and transitions-to-matched-performance are more diagnostic
than final success alone.

The seed-0 paired run subsequently completed both 200-iteration arms and exact
128-motion final evaluations. ZPD versus the official sampler changed success
by -1.563 percentage points, progress by +0.961 points, MPJPE-L by +0.623 mm,
and MPJPE-G by -10.429 mm. Its sampling distribution was materially flatter
(608.10 versus 282.79 effective bins), but the primary MPJPE-L screen did not
pass and both arms regressed from release initialization. This is a mixed
single-seed pilot, not an efficacy claim. See `seed0_pilot_summary.json` and the
repository-level research handoff for exact hashes, interpretation, and review
requests. The saved-checkpoint curve has now passed all 10 rows with exact
coverage. ZPD trails on all four normalized AUC metrics; see
`learning_curve_summary.json`. Protocol stabilization, not scale-up, is next.
