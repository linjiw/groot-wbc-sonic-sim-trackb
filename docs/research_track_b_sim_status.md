# Track B SONIC SIM status and next gate

Date: 2026-07-07

This document is the repo-level handoff for the current Track B / SONIC simulation research stack. It intentionally keeps the interpretation narrow: the current result is a trained-checkpoint evaluation validity gate, **not** an adaptive-sampling performance result.

## Protected references

| Reference | Commit / artifact | Purpose |
|---|---:|---|
| `sim-micro-adaptive-sampling-v0` | `d6bb536` | Original adaptive-sampling micro experiment reference. |
| `sim-m1-bounded-eval-metric-complete` | `98c5afc` | Metric-complete bounded eval gate. |
| SIM-M1 bundle | `/home/robotixx/sonic-sim-m1-bounded-eval-metric-complete.bundle` | Local protected handoff bundle, 378 MB. |
| SIM-M2-pre commit | `6327804` | Variant-specific post-training checkpoint eval gate. |

Recent research stack:

```text
6327804 Add SIM-M2-pre posttrain checkpoint eval
98c5afc Add SIM-M1 bounded eval metric smoke
e493d48 Add bounded SONIC eval micro metrics
d6bb536 Add SONIC adaptive sampling micro experiment
672c94b Add SONIC paired experiment launcher
c23f2ba Add SONIC manifest comparison harness
e668198 Add SONIC experiment manifest harness
f5d6554 Add SONIC log metric summarizer
```

## SIM-M2-pre result

Goal: verify that the paired harness evaluates each variant's own 10-iteration trained checkpoint instead of silently reusing `sonic_release/last.pt`.

Config:

```text
configs/research/sonic_paired_sample_micro_posttrain_eval.json
```

Artifacts generated locally:

```text
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/comparison.json
```

Checkpoint provenance:

| Variant | Checkpoint | SHA256 | Release checkpoint? |
|---|---|---|---|
| `uniform_sampling_micro` | `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_uniform_sampling_micro_posttrain_seed0-20260701_022413/last.pt` | `f1803557a1b8735f2eb20bcab4ccb8fe3df93d78e06d21f89947d959ee4ec8eb` | no |
| `adaptive_sampling_micro` | `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_adaptive_sampling_micro_posttrain_seed0-20260701_022443/last.pt` | `9c82a8131faeb9954f7917e2cf19953a0a71301ea965487b8447a1bb8d94dc9d` | no |

Comparison result:

| Field | Value |
|---|---:|
| `manifest_count` | 2 |
| `control_mismatches` | 0 |
| `validation_errors` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `ok_for_causal_comparison` | true |

Rows:

| Variant | `train.mean_rewards` | `eval.ok` | `eval.all.mpjpe_g` |
|---|---:|---:|---:|
| `uniform_sampling_micro` | 0.98515 | true | 31.901 |
| `adaptive_sampling_micro` | 1.02088 | true | 31.925 |

Interpretation: SIM-M2-pre passes as a trained-checkpoint evaluation validity gate. It proves the harness can train the two micro variants, save distinct post-training checkpoints, record checkpoint provenance, evaluate each checkpoint under bounded MPJPE-complete eval, and compare the resulting manifests without control, metric, validation, or checkpoint warnings.

Non-claim: this does **not** show adaptive-sampling benefit. It is one seed, 10 iterations, tiny `sample_data`, and bounded smoke eval. The MPJPE values are effectively tied.

## Verified test command

Run from repo root with the IsaacLab conda environment activated:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -m pytest -q \
  tests/research/test_sonic_eval_metric_smoke.py \
  tests/research/test_run_sonic_paired_experiment.py \
  tests/research/test_compare_sonic_manifests.py \
  tests/research/test_im_eval_callback_config.py \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py
```

Latest local verification: `39 passed in 3.55s` on 2026-07-07.

## Recommended next gate: SIM-M2

Next step: **3-seed paired micro causal sanity check**.

Design constraints:

| Dimension | Value |
|---|---|
| Seeds | 0, 1, 2 |
| `num_envs` | 8 |
| Learning iterations | 10 |
| Data | `sample_data/robot_filtered` + `sample_data/smpl_filtered` only |
| Only changed condition | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable` |
| Checkpoint saving | `++callbacks.model_save.save_last_frequency=10` |
| Eval | bounded MPJPE-complete eval per trained checkpoint |
| Aggregation | `aggregate_comparison.json` + `aggregate_table.md` |

SIM-M2 exit criteria:

- [ ] All seed-level comparisons emit `ok_for_causal_comparison=true`.
- [ ] All eval metrics are finite.
- [ ] No control mismatches.
- [ ] No metric warnings.
- [ ] No checkpoint warnings.
- [ ] `aggregate_comparison.json` exists.
- [ ] `aggregate_table.md` exists.
- [ ] Interpretation stays limited to micro causal-sanity validity unless the 3-seed result supports a stronger claim.

## Suggested implementation order

1. Add seed-specific configs or a multi-seed wrapper for seeds `0,1,2`.
2. Ensure every seed writes variant-specific checkpoints and provenance.
3. Run train+eval for each seed/variant pair.
4. Materialize per-seed `comparison.json` files.
5. Add an aggregate script/table over all seed-level comparisons.
6. Gate on warnings/errors before interpreting any metric deltas.

Do not scale to BONES-SEED, longer training, or performance claims until SIM-M2 passes.

## SIM-M2 result: 3-seed paired micro causal sanity

Date: 2026-07-07

Implementation added an aggregate comparison utility:

```text
scripts/research/aggregate_sonic_comparisons.py
tests/research/test_aggregate_sonic_comparisons.py
```

Runtime orchestrator used for this local gate:

```text
outputs/research/paired_sample_micro_sim_m2/run_sim_m2.py
```

Command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python outputs/research/paired_sample_micro_sim_m2/run_sim_m2.py --seeds 0 1 2 --execute
```

Artifacts:

```text
outputs/research/paired_sample_micro_sim_m2/seed0/comparison.json
outputs/research/paired_sample_micro_sim_m2/seed1/comparison.json
outputs/research/paired_sample_micro_sim_m2/seed2/comparison.json
outputs/research/paired_sample_micro_sim_m2/aggregate_comparison.json
outputs/research/paired_sample_micro_sim_m2/aggregate_table.md
```

Aggregate gate status:

| Field | Value |
|---|---:|
| `comparison_count` | 3 |
| `seeds` | `[0, 1, 2]` |
| `ok_for_causal_comparison` | true |
| `control_mismatches` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `validation_errors` | 0 |

Seed rows:

| Seed | Uniform train mean reward | Adaptive train mean reward | Uniform MPJPE-G | Adaptive MPJPE-G | Adaptive - uniform MPJPE-G |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.98515 | 1.02088 | 31.901 | 31.925 | +0.024 |
| 1 | 0.88738 | 0.89901 | 31.637 | 31.596 | -0.041 |
| 2 | 0.95787 | 0.81643 | 34.257 | 34.312 | +0.055 |

Aggregate descriptive stats:

| Metric | Value |
|---|---:|
| mean uniform MPJPE-G | 32.5983 |
| mean adaptive MPJPE-G | 32.6110 |
| mean delta, adaptive - uniform | +0.0127 |
| sample std of delta | 0.0490 |

Interpretation: SIM-M2 passes as a 3-seed micro causal-sanity **validity gate**. The harness now survives seed expansion, preserves variant-specific checkpoint provenance, emits finite bounded eval metrics, and aggregates seed-level comparisons without control, metric, validation, or checkpoint warnings.

Non-claim: there is still no adaptive-sampling performance benefit here. Deltas are tiny and mixed-sign over a deliberately tiny 10-iteration sample-data smoke. The right conclusion is that the comparison/eval machinery is ready for a more meaningful next gate, not that adaptive sampling improves SONIC.

Recommended next gate after SIM-M2:

1. Preserve the SIM-M2 aggregate as the protected micro validity reference.
2. Add a slightly more meaningful but still bounded `SIM-M3` gate before BONES-SEED scaling, for example:
   - same 3 seeds,
   - modestly longer training budget,
   - still `sample_data` or a tiny fixed curated motion subset,
   - same checkpoint-provenance and bounded-eval requirements,
   - pre-register an effect-size threshold before looking at results.
3. Only if SIM-M3 shows stable, non-trivial directionality should we consider BONES-SEED or larger training.

## SIM-M3 pre-registered bounded effect-size gate

Date: 2026-07-07

Goal: test whether the SIM-M2-valid harness shows any stable adaptive-sampling direction under a still-small but less trivial training budget. This is **not** a BONES-SEED or paper-performance gate.

Controlled variables:

| Dimension | SIM-M3 value |
|---|---|
| Seeds | `0, 1, 2` |
| `num_envs` | 8 |
| Learning iterations | 50 |
| Dataset | `sample_data/robot_filtered` + `sample_data/smpl_filtered` only |
| Only changed condition | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable` |
| Adaptive extra knob | `adaptive_sampling.uniform_sampling_rate=0.1` |
| Checkpoint save cadence | `++callbacks.model_save.save_last_frequency=50` |
| Eval | same bounded MPJPE-complete eval as SIM-M2 |
| Aggregation | `scripts/research/aggregate_sonic_comparisons.py` with effect gate |

Pre-registered effect gate for a **candidate signal** only:

```text
metric = eval.all.mpjpe_g
lower_is_better = true
mean_delta_adaptive_minus_uniform <= -0.5 MPJPE-G
improved_seed_count >= 2 of 3
all seed-level comparisons ok_for_causal_comparison=true
control_mismatches = metric_warnings = checkpoint_warnings = validation_errors = 0
```

Decision rule:

- If the validity gate fails, fix the harness; do not interpret metrics.
- If validity passes but the effect gate fails, conclude no adaptive-sampling signal at this bounded budget.
- If validity and effect gate both pass, label it only as a SIM-M3 candidate signal and design the next fixed-data gate before any BONES-SEED scaling.
- No adaptive-sampling performance claim is allowed from SIM-M3 alone.

## SIM-M3 result: bounded effect-size probe

Date: 2026-07-07

Command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python outputs/research/paired_sample_micro_sim_m3/run_sim_m3.py --seeds 0 1 2 --execute
```

Artifacts:

```text
outputs/research/paired_sample_micro_sim_m3/seed0/comparison.json
outputs/research/paired_sample_micro_sim_m3/seed1/comparison.json
outputs/research/paired_sample_micro_sim_m3/seed2/comparison.json
outputs/research/paired_sample_micro_sim_m3/aggregate_comparison.json
outputs/research/paired_sample_micro_sim_m3/aggregate_table.md
```

Aggregate validity gate:

| Field | Value |
|---|---:|
| `comparison_count` | 3 |
| `seeds` | `[0, 1, 2]` |
| `ok_for_causal_comparison` | true |
| `control_mismatches` | 0 |
| `metric_warnings` | 0 |
| `checkpoint_warnings` | 0 |
| `validation_errors` | 0 |

Pre-registered effect gate:

| Field | Value |
|---|---:|
| metric | `eval.all.mpjpe_g` |
| threshold | adaptive - uniform <= -0.5 |
| minimum improved seeds | 2 of 3 |
| mean adaptive - uniform | +0.110667 |
| improved seeds | 1 of 3 |
| passes effect gate | false |

Seed rows:

| Seed | Uniform train mean reward | Adaptive train mean reward | Uniform MPJPE-G | Adaptive MPJPE-G | Adaptive - uniform MPJPE-G |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.98755 | 0.93542 | 31.835 | 32.037 | +0.202 |
| 1 | 0.93287 | 0.89537 | 31.554 | 31.696 | +0.142 |
| 2 | 0.96641 | 0.97600 | 34.244 | 34.232 | -0.012 |

Interpretation: SIM-M3 passes the harness validity gate but fails the pre-registered candidate-effect gate. This is a useful negative result: a modestly longer 50-iteration sample-data gate still does not show a stable adaptive-sampling benefit. Two seeds are worse for adaptive under MPJPE-G and the lone improved seed is effectively tied.

Decision: do not scale this adaptive-sampling mechanism to BONES-SEED as-is. The next step should shift from scaling to diagnosis. Candidate next diagnostics:

1. Audit whether adaptive sampling actually changes the sampled motion/bin distribution over 50 iterations (`Env/adp_samp/*` logs, per-seed concentration/effective-bin metrics).
2. Add an aggregate diagnostic table for adaptive-sampler telemetry, not just reward/MPJPE.
3. If the sampler is active but not helpful, test a different bounded mechanism or sampling schedule before any larger data/training expansion.
4. Preserve SIM-M3 as a negative bounded-effect reference.
