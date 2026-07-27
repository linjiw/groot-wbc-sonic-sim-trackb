# Official BONES-SEED ZPD Research Handoff

Status: 2026-07-27. This document separates completed evidence from proposed work so
another team can review or continue the experiment without reconstructing its history.

## Research question

The causal question is narrow: does replacing SONIC's official adaptive-sampling utility
with a zone-of-proximal-development (ZPD) utility improve motion-imitation learning under
otherwise identical training?

- Control: official `adaptive_sampling.signal=failure_rate`.
- Treatment: `signal=learnability`, with a Beta posterior per 50-frame bin and utility
  `E[p(1-p)] = ab / ((a+b)(a+b+1))`.
- Fixed across arms: release initialization, fresh optimizer and sampler state, motion
  cohort, seed, PPO budget, evaluator, and all non-sampler configuration.
- Disabled in the primary treatment: optimism, evidence decay, and family sharing.

The ZPD utility should emphasize bins near the current competence frontier while reducing
mass on mastered and persistently impossible bins. It creates a curriculum by reallocating
sampling probability, not by manually ordering motions or changing rewards.

## Frozen data and protocol

The pilot contains 128 paired robot/SMPL motions selected outcome-blind from eligible
BONES-SEED metadata using category-by-duration strata. It covers 20 categories, 8 archive
packages, and all 78 non-empty strata, but only 0.0986% of eligible motions. The paired
dataset digest is `d82ac458...ad91`; its manifest digest is `2da4a3ce...dc2a`.

Each arm used 128 environments, 200 PPO iterations, and 24 steps per environment:
614,400 policy transitions per arm. All 128 motions remain resident, so this pilot tests
allocation among 1,006 temporal bins; it does not test the official 250-iteration
cross-motion reload behavior. Checkpoints were saved at iterations 50, 100, 150, and 200.
Evaluation uses the repository's official `ImEvalCallback` over the exact frozen cohort.
Success, progress, and MPJPE-L are primary; world-frame MPJPE-G is secondary.

## Completed evidence

An independent 32-environment paged evaluation of the release checkpoint found success
`0.9453125`, progress `0.9623540`, MPJPE-L `30.654 mm`, and MPJPE-G `146.250 mm`.
The matched 128-environment learning curve re-evaluated iteration 0 once and reused the
content-identical result for both arms: success `0.9609375`, progress `0.9718712`,
MPJPE-L `30.999 mm`, and MPJPE-G `152.407 mm`. The latter is the AUC anchor.

Both seed-0 arms completed training and exact-coverage final evaluation. The comparator
reports no control mismatch, metric warning, checkpoint warning, or validation error.

| Metric | Official failure rate | ZPD learnability | ZPD − control |
|---|---:|---:|---:|
| Success | 0.578125 | 0.562500 | -0.015625 |
| Progress | 0.716056 | 0.725671 | +0.009615 |
| MPJPE-L | 42.377 mm | 43.000 mm | +0.623 mm |
| MPJPE-G | 213.241 mm | 202.812 mm | -10.429 mm |
| MPJPE-PA | 31.308 mm | 31.043 mm | -0.265 mm |

The sampler mechanism clearly changed. ZPD ended with 608.10 effective bins versus
282.79 for the control, maximum probability/uniform of 6.74 versus 17.68, and zero versus
four concentrated bins. ZPD's final training reward was 13.95 versus 11.94. These are
activation and behavior observations, not proof of better policy performance.

Checkpoint-state analysis also exposes a design limitation: 827/1,006 ZPD bins (82.21%)
remain prior-dominated at the registered threshold of at most two observed failures;
the control is similar at 831/1,006 (82.60%). ZPD recorded 3,747 selected target-bin
draws and 1,493 observed failures. The teacher therefore activated, but most posterior
utilities remained evidence-poor during this short resident-cohort run.

The preregistered single-seed MPJPE-L screen does not pass: treatment is `+0.623 mm`
worse. Other metrics are mixed, and both trained policies regress substantially from the
release initialization. One seed cannot support a superiority, null, or regression claim.

The saved-checkpoint curve completed all 10 paired rows using nine GPU evaluations and
one content-identical iteration-0 reuse. Every row has exact 128-motion coverage:

| Iteration | Control success | ZPD success | Control MPJPE-L | ZPD MPJPE-L |
|---:|---:|---:|---:|---:|
| 0 | 0.960938 | 0.960938 | 30.999 mm | 30.999 mm |
| 50 | 0.773438 | 0.789062 | 38.163 mm | 39.952 mm |
| 100 | 0.773438 | 0.742188 | 38.664 mm | 41.072 mm |
| 150 | 0.625000 | 0.632812 | 41.300 mm | 41.313 mm |
| 200 | 0.578125 | 0.562500 | 42.377 mm | 43.000 mm |

Normalized AUC also favors the control on every registered metric: ZPD minus control is
`-0.00391` success, `-0.00117` progress, `+1.130 mm` MPJPE-L, and `+29.144 mm`
MPJPE-G. The main finding is therefore protocol-level: both arms degrade sharply by
iteration 50, while this ZPD formulation flattens sampling without improving sample
efficiency in seed 0.

## Reproduction and implementation map

- Sampler implementation: `gear_sonic/utils/motion_lib/motion_lib_base.py`.
- Paired causal contract: `scripts/research/run_sonic_paired_experiment.py`.
- Multi-seed/result gate: `scripts/research/run_sonic_multiseed.py`.
- Learning curve: `scripts/research/plan_sonic_paired_learning_curve.py`.
- Pilot and scale-up specs: `configs/research/bones_seed_official_zpd_pilot128.json` and
  `configs/research/bones_seed_official_zpd_scale512.json`.

Focused CPU validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/research/test_aggregate_sonic_comparisons.py \
  tests/research/test_run_sonic_multiseed.py \
  tests/research/test_plan_sonic_paired_learning_curve.py \
  tests/research/test_run_sonic_paired_experiment.py \
  tests/research/test_zpd_teacher_sampler.py
```

The experiment template contains machine-local Python and checkpoint paths. A new runner
must replace those paths, materialize the ignored motion cohort from the pinned source
locks, and run preflight before using GPU time. Raw motions, logs, and checkpoints are
intentionally excluded from Git.

## Review and help requested

1. Review why 200 iterations degrade both arms from a strong release initialization:
   learning rate/optimizer reset, fine-tuning horizon, resident-cohort overfitting, and
   checkpoint-selection policy are the leading hypotheses.
2. Review metric priority and the final/AUC decision rule. In particular, confirm that
   MPJPE-L plus progress should remain primary and MPJPE-G secondary.
3. Provision shared storage for the pinned 512-motion scale-up. The archives require
   about 55.8 GB before checkpoints. An administrator can create, for example,
   `/data/Humanoid/sonic_bones_seed_official` owned by the training user:

   ```bash
   sudo install -d -o robotixx -g robotixx /data/Humanoid/sonic_bones_seed_official
   ```
4. Review the proposed credible scale-up before launch: 512 motions, 128 environments,
   625 iterations, seeds 0/1/2, evaluations at 0/125/250/375/500/625, with resident-set
   coverage/turnover and tripwire telemetry reported alongside policy metrics.
5. Help choose and independently reproduce a safe fine-tuning control before additional
   ZPD seeds: lower learning rate, preserved optimizer state, or a shorter/best-checkpoint
   horizon. The current result is a pilot diagnosis, not a scale-up authorization.

The source lock under `configs/research/` records the exact historical launch state at
`f7dc739`. It should be used as artifact provenance; later orchestration fixes necessarily
make a live byte-for-byte verification fail unless a new lock is captured.
