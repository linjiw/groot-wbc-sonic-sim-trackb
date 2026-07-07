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
