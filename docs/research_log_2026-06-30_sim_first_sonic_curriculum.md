# 2026-06-30 — Simulation-first SONIC curriculum research log

## Paper goal framing

We are reframing the project around a simulation-first humanoid-motion pipeline rather than a real-robot data-collection gate.

**Paper-level hypothesis:** competence-gated curriculum structure over humanoid motion / SONIC control evaluation can improve reliability and safety diagnostics for G1 loco-manipulation compared with unstructured or uniform progression, while preserving the fixed SONIC/VLA action contract.

**Non-negotiable interface boundary:**

```text
Isaac-GR00T / high-level policy -> 64D SONIC motion token + 7D left hand + 7D right hand -> fixed SONIC WBC -> G1
```

Do not change SONIC deployment observation ordering, ZMQ protocol, low-level command generation, or action dimensions for the paper MVP. Research work should add measurement, curriculum state, dataset QA, and eval harnesses around the stack.

## Corrected system interpretation

Earlier work incorrectly prioritized `/data/g1_fetch_place_tiny_raw` and real-camera/robot collection markers. That is a later VLA/hardware path, not the current simulation-first research gate.

Correct current priority:

1. Use `env_isaaclab` with IsaacLab / Isaac Sim.
2. Validate SONIC training/eval on humanoid motion datasets.
3. Establish reproducible sample-data health run.
4. Probe released checkpoint evaluation/render/export path.
5. Acquire/prepare full BONES-SEED / GEAR-SONIC motion data for scale-up.
6. Define controlled curriculum-vs-baseline experiments only after the sim/eval harness is stable.

## Fixed blocker already resolved

A real IsaacLab blocker was found and fixed:

```text
PermissionError: /tmp/IsaacLab/usd_*
```

Cause: `/tmp/IsaacLab` was owned by another user and not writable by `robotixx`.

Fix committed:

```text
9948c29 Use writable IsaacLab USD cache for G1 asset
```

The G1 URDF converter now uses a writable cache under `~/.cache/isaaclab/usd/g1_model_12_dex`.

## Local evidence so far

### Available local data/checkpoints

```text
sample_data/robot_filtered: present, 2 files, ~696K
sample_data/smpl_filtered: present, 2 files, ~2.6M
sonic_release/last.pt: present, ~448M
```

Missing local full-scale training data:

```text
data/motion_lib_bones_seed/robot_filtered
data/smpl_filtered
data/bones_seed_smpl
```

### HF access findings

`nvidia/GEAR-SONIC` model repo is listable anonymously and contains:

```text
sonic_release/last.pt
sample_data/*
bones_seed_smpl/bones_seed_smpl.tar.part_aa ... part_ag
```

`bones-studio/seed` dataset metadata is listable but downloads are gated without access approval/token:

```text
403 GatedRepoError: Access to dataset bones-studio/seed is restricted
```

This means full G1 CSV source acquisition requires either accepted HF access or an already downloaded local copy. The preprocessed SMPL parts from `nvidia/GEAR-SONIC` are available but are not sufficient alone for full SONIC training because the G1 `robot_filtered` motion-lib path is also required.

## Current running experiment

Started background health run:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=16 headless=True use_wandb=false \
  exp_var=sample_health_100it_bg \
  ++algo.config.num_learning_iterations=100 \
  ++algo.config.save_interval=999999 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

Log path is written to:

```text
outputs/research/sonic_health/latest.log.path
```

Final result: the 100-iteration sample-data health run completed successfully.

```text
Log: outputs/research/sonic_health/sample_health_100it_20260630_171007.log
Run dir: logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_sample_health_100it_bg-20260630_171009
Learning iteration: 100
Total episodes: 1600
Total timesteps: 38400
Total time: 581.13s
Final mean rewards: 0.85156
Final mean length: 10.52
Final error_anchor_pos: 0.0912
Final error_body_pos: 0.0898
Tracebacks: none observed
Artifacts: config.yaml, meta.yaml, .hydra logs, last.pt
```

This is a health/profiling run only; it is not a convergence claim.

## Released-checkpoint eval smoke

First eval attempt loaded IsaacLab and ran the rollout but failed at metrics post-processing:

```text
ModuleNotFoundError: No module named 'smpl_sim'
```

Fix applied in `env_isaaclab`:

```bash
python -m pip install 'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git'
```

Import verified:

```text
smpl_sim_import_ok
```

Retry command completed successfully:

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++eval_callbacks=im_eval \
  ++run_eval_loop=False \
  ++num_envs=2 \
  ++algo.config.eval.num_eval_episodes=4 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2 \
  +manager_env/terminations=tracking/eval
```

Output evidence:

```text
Log: outputs/research/eval_release/released_checkpoint_sample_eval_retry_smplsim_20260630_172748.log
Hydra eval dir: logs_eval/20260630_172749-TEST
All:  mpjpe_g: 130.802, mpjpe_l: 18.728, mpjpe_pa: 11.824
Succ: mpjpe_g: 199.754, mpjpe_l: 20.975, mpjpe_pa: 12.139
Terminated: 1 during the sampled sequence
Exit code: 0
```

Interpretation: released-checkpoint eval harness now runs end-to-end on sample data and emits MPJPE metrics. The sampled sequence is not a strong success claim because termination occurred and the sample set has only two walk motions; use this as an eval-harness gate, not a paper metric.

## Reproducible metric extraction harness

Added a paper-facing summarizer:

```text
scripts/research/summarize_sonic_logs.py
tests/research/test_sonic_log_summary.py
```

Purpose: convert noisy IsaacLab/SONIC terminal logs into stable JSON/Markdown artifacts for later paired comparisons.

Validation:

```text
python -m pytest -q tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py

15 passed in 0.68s
```

Current sample summary generated by the harness:

```bash
python scripts/research/summarize_sonic_logs.py \
  --train-log outputs/research/sonic_health/sample_health_100it_20260630_171007.log \
  --eval-log outputs/research/eval_release/released_checkpoint_sample_eval_retry_smplsim_20260630_172748.log \
  --output-json outputs/research/sonic_sample_summary.json \
  --output-md outputs/research/sonic_sample_summary.md
```

Key parsed fields:

```text
train.ok=true
train.learning_iteration=100
train.mean_rewards=0.85156
train.total_timesteps=38400
eval.ok=true
eval.all.mpjpe_g=130.802
eval.all.mpjpe_l=18.728
eval.terminated_final=1
```

This is now the minimum reporting contract for the next simulation experiments.

## Paired experiment manifest harness

Added a manifest builder/validator so every future baseline/curriculum run can be compared from an explicit, reproducible record rather than implicit shell history:

```text
scripts/research/sonic_experiment_manifest.py
tests/research/test_sonic_experiment_manifest.py
```

The manifest schema records:

```text
experiment_id
hypothesis
variant
seed
git_commit
controlled_variables
datasets
checkpoint
train/eval commands
summary artifact path
parsed metrics
interpretation/status
```

Validation:

```text
python -m pytest -q \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py

19 passed in 0.69s
```

Current sample manifest generated from real artifacts:

```text
outputs/research/sonic_sample_experiment_manifest.json
outputs/research/sonic_sample_experiment_manifest.md
```

Key fields:

```text
experiment_id=sample_release_eval_seed0
variant=released_checkpoint_sample_eval
seed=0
checkpoint=sonic_release/last.pt
datasets.robot_motion=sample_data/robot_filtered
datasets.smpl_motion=sample_data/smpl_filtered
git_commit=f5d6554
metrics.train.total_timesteps=38400
metrics.eval.all.mpjpe_g=130.802
status=needs_review
interpretation=harness_ok_not_convergence__eval_sequence_terminated__sample_data_only
```

This establishes the comparison unit for the paper: future runs should add one manifest per seed/variant, then compare manifests under fixed datasets, checkpoints, commands, and evaluation settings.

## Manifest comparison harness

Added a comparison script that consumes one or more manifest JSON files and emits a paper-facing JSON/Markdown comparison table:

```text
scripts/research/compare_sonic_manifests.py
tests/research/test_compare_sonic_manifests.py
```

Purpose:

- validate every supplied manifest using the manifest schema,
- extract stable train/eval fields into a table,
- check whether controlled variables, dataset paths, and checkpoint match across variants,
- fail loudly when mismatches would break causal comparison.

Validation:

```text
source /home/robotixx/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
python -m pytest -q \
  tests/research/test_compare_sonic_manifests.py \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py

23 passed in 0.72s
```

Current one-manifest smoke comparison generated from the real sample manifest:

```bash
python scripts/research/compare_sonic_manifests.py \
  --manifest outputs/research/sonic_sample_experiment_manifest.json \
  --output-json outputs/research/sonic_sample_manifest_comparison.json \
  --output-md outputs/research/sonic_sample_manifest_comparison.md
```

Key fields:

```text
manifest_count=1
control_mismatches=[]
validation_errors=[]
ok_for_causal_comparison=false
row.experiment_id=sample_release_eval_seed0
row.metrics.train.mean_rewards=0.85156
row.metrics.eval.all.mpjpe_g=130.802
row.metrics.eval.terminated_final=1
```

The `ok_for_causal_comparison=false` value is intentional for the current artifact because only one manifest exists. Once baseline and curriculum manifests are supplied for the same seed/dataset/checkpoint, the comparison harness will mark the pair causal-comparison-ready only if controls match.

## Paired experiment launcher/spec harness

Added a dry-run-friendly paired experiment launcher:

```text
scripts/research/run_sonic_paired_experiment.py
tests/research/test_run_sonic_paired_experiment.py
configs/research/sonic_paired_sample_validation_fixture.json
```

Purpose:

- encode a paired baseline/curriculum experiment as an explicit JSON spec,
- keep the fixed dataset/checkpoint/seed controls in one place,
- optionally execute train/eval commands with captured logs,
- summarize logs or consume existing summaries,
- build per-variant manifests,
- run the manifest comparison harness,
- emit one final run-plan directory for paper bookkeeping.

Validation:

```text
source /home/robotixx/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
python -m pytest -q \
  tests/research/test_run_sonic_paired_experiment.py \
  tests/research/test_compare_sonic_manifests.py \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py

26 passed in 0.70s
```

Dry-run fixture generated without launching heavy IsaacLab training:

```bash
python scripts/research/run_sonic_paired_experiment.py \
  --spec configs/research/sonic_paired_sample_validation_fixture.json \
  --output-dir outputs/research/paired_sample_validation_fixture \
  --dry-run \
  --repo-root /home/robotixx/GR00T-WholeBodyControl
```

Generated artifacts:

```text
outputs/research/paired_sample_validation_fixture/run_plan.json
outputs/research/paired_sample_validation_fixture/run_plan.md
outputs/research/paired_sample_validation_fixture/baseline_fixture/manifest.json
outputs/research/paired_sample_validation_fixture/curriculum_fixture/manifest.json
outputs/research/paired_sample_validation_fixture/comparison.json
outputs/research/paired_sample_validation_fixture/comparison.md
```

Verified fields:

```text
dry_run=true
variant_count=2
manifest_count=2
control_mismatches=0
validation_errors=0
ok_for_causal_comparison=true
```

Important caveat: this fixture intentionally reuses the existing sample summary for both variants. It validates harness plumbing only; it is not a baseline/curriculum scientific result. The next real experiment should replace the fixture summaries with fresh logs from two actually distinct commands.

## Paired micro-experiment execution gate

Added an executable paired micro-experiment spec:

```text
configs/research/sonic_paired_sample_micro_execute.json
```

Controlled contrast:

| Variant | Changed condition |
|---|---|
| `uniform_sampling_micro` | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false` |
| `adaptive_sampling_micro` | `manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=true`, `uniform_sampling_rate=0.1` |

Fixed controls:

```text
seed=0
num_envs=8
num_learning_iterations=10
dataset_robot=sample_data/robot_filtered
dataset_smpl=sample_data/smpl_filtered
checkpoint=sonic_release/last.pt for eval smoke
```

Execution command attempted:

```bash
python scripts/research/run_sonic_paired_experiment.py \
  --spec configs/research/sonic_paired_sample_micro_execute.json \
  --output-dir outputs/research/paired_sample_micro_execute \
  --execute \
  --repo-root /home/robotixx/GR00T-WholeBodyControl
```

The wrapper command timed out at 600s after both 10-iteration train runs and eval logs had been produced, but before the launcher wrote the final run plan for the second variant. I then re-ran the launcher in `--dry-run` mode against the same output directory to summarize the already-produced logs and build manifests/comparison without re-launching IsaacLab.

Evidence from generated config files confirms the intended single experimental knob:

```text
logs_rl/.../sonic_release_uniform_sampling_micro_seed0-20260630_184900/config.yaml:
  adaptive_sampling.enable: false

logs_rl/.../sonic_release_adaptive_sampling_micro_seed0-20260630_185404/config.yaml:
  adaptive_sampling.enable: true
```

Training summaries:

| Variant | Iterations | Timesteps | Mean reward | Anchor pos error | Body pos error | Train OK |
|---|---:|---:|---:|---:|---:|---|
| `uniform_sampling_micro` | 10 | 1920 | 0.98515 | 0.1207 | 0.1041 | true |
| `adaptive_sampling_micro` | 10 | 1920 | 1.02088 | 0.1078 | 0.0904 | true |

Adaptive-sampling diagnostics appeared only in the adaptive variant log, including:

```text
Env/adp_samp/prob_max_over_uniform: 3.4619
Env/adp_samp/effective_num_bins: 69.4364
Env/adp_samp/num_concentrated_bins: 0.0000
```

Comparison artifacts:

```text
outputs/research/paired_sample_micro_execute/run_plan.json
outputs/research/paired_sample_micro_execute/uniform_sampling_micro/summary.json
outputs/research/paired_sample_micro_execute/adaptive_sampling_micro/summary.json
outputs/research/paired_sample_micro_execute/uniform_sampling_micro/manifest.json
outputs/research/paired_sample_micro_execute/adaptive_sampling_micro/manifest.json
outputs/research/paired_sample_micro_execute/comparison.json
```

Final comparison status after tightening the comparison harness:

```text
manifest_count=2
control_mismatches=0
validation_errors=0
metric_warnings=4
ok_for_causal_comparison=false
warning_fields=['eval.all.mpjpe_g', 'eval.ok']
```

Interpretation: the paired micro-experiment validates that the training contrast is executable under fixed simulation controls and that adaptive-sampling telemetry is visible. It is not paper-grade causal evidence yet because the eval smoke did not emit `All:` MPJPE metrics for these runs (`eval.ok=false`, although there were no tracebacks). The next gate is to make eval bounded and metric-complete for micro runs before scaling.

The comparison harness was also tightened so `ok_for_causal_comparison` now requires primary train/eval metrics to be present and healthy, not just matching controls.

Validation after this change:

```text
python -m pytest -q \
  tests/research/test_run_sonic_paired_experiment.py \
  tests/research/test_compare_sonic_manifests.py \
  tests/research/test_sonic_experiment_manifest.py \
  tests/research/test_sonic_log_summary.py \
  tests/research/test_curriculum_sampler.py \
  tests/research/test_curriculum_gates.py \
  tests/research/test_manifest_builder_fixture.py \
  tests/research/test_data_collection_launcher.py

27 passed in 0.71s
```

## Bounded metric-complete micro eval gate

Root cause of the previous missing-metric micro eval was twofold:

1. full-sequence sample eval takes roughly 3.8 minutes per variant because sample SMPL motions are 2002 frames long;
2. `ImEvalCallback` exits with `os._exit(0)` for eval-only runs, so redirected/stdout-captured eval summaries could be lost unless metric prints are flushed before exit.

Implemented fixes:

```text
gear_sonic/trl/callbacks/im_eval_callback.py
gear_sonic/config/callbacks/im_eval.yaml
tests/research/test_im_eval_callback_config.py
scripts/research/run_sonic_paired_experiment.py
tests/research/test_run_sonic_paired_experiment.py
configs/research/sonic_paired_sample_micro_execute.json
```

Changes:

- added optional `callbacks.im_eval.max_eval_steps`, default `null`, to cap smoke-eval length without changing full eval behavior;
- flushed `Success Rate`, `Progress Rate`, `All:`, and `Succ:` prints before eval-only `os._exit(0)`;
- updated the paired launcher to rebuild stale summaries when logs are newer than summary JSON;
- updated the sample micro spec to use `++callbacks.im_eval.max_eval_steps=200` for bounded eval smoke.

Bounded eval probe:

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++eval_callbacks=im_eval \
  ++run_eval_loop=False \
  ++num_envs=2 \
  ++callbacks.im_eval.max_eval_steps=200 \
  ++algo.config.eval.num_eval_episodes=4 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2 \
  +manager_env/terminations=tracking/eval
```

Evidence:

```text
log=outputs/research/eval_release/micro_metric_bounded_max200_flush_20260630_193028.log
rc=0
Success Rate: 1.0000000000
Progress Rate: 1.0000000000
All: mpjpe_g: 16.493, mpjpe_l: 13.255, mpjpe_pa: 9.168, ...
Succ: mpjpe_g: 16.493, mpjpe_l: 13.255, mpjpe_pa: 9.168, ...
```

Regenerated paired micro summaries/manifests/comparison from the existing train logs plus new bounded eval logs:

```text
outputs/research/paired_sample_micro_execute/comparison.json
manifest_count=2
control_mismatches=0
validation_errors=0
metric_warnings=0
ok_for_causal_comparison=true
```

Rows:

```text
uniform_sampling_micro:  train.mean_rewards=0.98515, eval.ok=true, eval.all.mpjpe_g=16.493, terminated_final=0
adaptive_sampling_micro: train.mean_rewards=1.02088, eval.ok=true, eval.all.mpjpe_g=16.493, terminated_final=0
```

Interpretation: this is now a metric-complete sample-data **execution validation** artifact. It still is not a scientific performance claim because both variants share the same released-checkpoint eval smoke; the value is proving that the paper harness can enforce controls, detect stale summaries, run bounded metric eval, and produce a comparison accepted by the stricter metric gate.

## SIM-M1 formal eval-only and paired metric-complete gate

Advisor directive: freeze the first executed paired training contrast before additional eval work, then prove bounded eval metric completeness before scaling.

Protected reference:

```text
git tag sim-micro-adaptive-sampling-v0 d6bb536
bundle=/home/robotixx/sonic-sim-micro-adaptive-sampling-v0.bundle
```

The tag `d6bb536` is explicitly labeled as an **executed training contrast with incomplete eval metrics**, not as a positive result.

Added diagnostic and eval-only tooling:

```text
scripts/research/diagnose_sonic_eval_logs.py
scripts/research/run_sonic_eval_metric_smoke.py
configs/research/sonic_eval_micro_metric_complete.json
configs/research/sonic_paired_sample_micro_metric_complete.json
tests/research/test_sonic_eval_metric_smoke.py
```

Diagnostic artifact for the current paired eval logs:

```text
outputs/research/paired_sample_micro_execute/eval_diagnostics.json
uniform_sampling_micro: traceback=false, timeout=false, contains_all_mpjpe=true, candidate_metric_lines=4
adaptive_sampling_micro: traceback=false, timeout=false, contains_all_mpjpe=true, candidate_metric_lines=4
```

Eval-only SIM-M1 command:

```bash
python scripts/research/run_sonic_eval_metric_smoke.py \
  --spec configs/research/sonic_eval_micro_metric_complete.json \
  --output-dir outputs/research/sonic_eval_micro_metric_complete \
  --repo-root /home/robotixx/GR00T-WholeBodyControl
```

Eval-only SIM-M1 result:

```text
outputs/research/sonic_eval_micro_metric_complete/result.json
returncode=0
timed_out=false
eval_ok=true
primary_mpjpe_metric=mpjpe_g
primary_mpjpe_value=17.807
ok=true
```

Summary artifact:

```text
outputs/research/sonic_eval_micro_metric_complete/summary.json
eval.ok=true
eval.all.mpjpe_g=17.807
eval.traceback_count=0
eval.success_rate_final=1.0
eval.progress_rate_final=1.0
```

Parser correction: bounded eval logs emit final `Success Rate:` and `Progress Rate:` lines after tqdm-style interim `Succ rate:` status lines. The summarizer now prefers the final aggregate `Success Rate:` value and records `progress_rate_final`, preventing interim tqdm status from contaminating final eval summaries.

Metric-complete paired comparison artifact:

```text
outputs/research/paired_sample_micro_metric_complete/comparison.json
manifest_count=2
control_mismatches=0
validation_errors=0
metric_warnings=0
ok_for_causal_comparison=true
```

Rows:

```text
uniform_sampling_micro:  eval.ok=true, eval.all.mpjpe_g=16.493, finite=true
adaptive_sampling_micro: eval.ok=true, eval.all.mpjpe_g=16.493, finite=true
```

SIM-M1 exit criteria are satisfied for sample-data bounded eval reliability. This is still not an adaptive-sampling performance claim because the eval command uses the released checkpoint; the correct next gate is variant-specific post-training checkpoint eval, then SIM-M2 multi-seed causal sanity.

## SIM-M2-pre variant-specific post-training checkpoint eval

Goal: prove that the comparison harness can evaluate each variant's own 10-iteration trained checkpoint rather than silently reusing `sonic_release/last.pt`.

Protected SIM-M1 reference:

```text
git tag sim-m1-bounded-eval-metric-complete 98c5afc
bundle=/home/robotixx/sonic-sim-m1-bounded-eval-metric-complete.bundle
```

Initial checkpoint audit of the first executed micro contrast (`outputs/research/paired_sample_micro_execute`) found no checkpoint artifacts because the micro run used a save cadence larger than its 10-iteration budget:

```text
outputs/research/paired_sample_micro_execute/checkpoint_provenance.json
uniform_sampling_micro.exists=false
adaptive_sampling_micro.exists=false
```

Narrow correction: reran the same 10-iteration micro commands with checkpoint saving enabled via:

```text
++callbacks.model_save.save_last_frequency=10
```

No training scale was increased.

Checkpoint provenance for the post-train eval gate:

```text
outputs/research/paired_sample_micro_posttrain_eval/checkpoint_provenance.json
uniform_sampling_micro:
  path=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_uniform_sampling_micro_posttrain_seed0-20260701_022413/last.pt
  size_bytes=448600614
  sha256=f1803557a1b8735f2eb20bcab4ccb8fe3df93d78e06d21f89947d959ee4ec8eb
  is_release_checkpoint=false
adaptive_sampling_micro:
  path=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_adaptive_sampling_micro_posttrain_seed0-20260701_022443/last.pt
  size_bytes=448601904
  sha256=9c82a8131faeb9954f7917e2cf19953a0a71301ea965487b8447a1bb8d94dc9d
  is_release_checkpoint=false
checks:
  uniform_vs_adaptive_distinct_paths=true
  uniform_vs_adaptive_distinct_sha256=true
```

Added manifest/comparison support for this gate:

```text
checkpoint_source=trained_variant_checkpoint
checkpoint_provenance={sha256,size_bytes,mtime,is_release_checkpoint}
```

The comparison layer now allows distinct checkpoint paths only when every manifest declares `checkpoint_source=trained_variant_checkpoint`, and it raises `checkpoint_warnings` if a trained-variant comparison uses `sonic_release/last.pt`, duplicates checkpoint paths, or provenance marks a release checkpoint.

SIM-M2-pre config:

```text
configs/research/sonic_paired_sample_micro_posttrain_eval.json
```

Generated artifacts:

```text
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/uniform_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/summary.json
outputs/research/paired_sample_micro_posttrain_eval/adaptive_sampling_micro/manifest.json
outputs/research/paired_sample_micro_posttrain_eval/comparison.json
```

SIM-M2-pre comparison result:

```text
manifest_count=2
control_mismatches=0
validation_errors=0
metric_warnings=0
checkpoint_warnings=0
ok_for_causal_comparison=true
```

Rows:

```text
uniform_sampling_micro:  train.mean_rewards=0.98515, eval.ok=true, eval.all.mpjpe_g=31.901
adaptive_sampling_micro: train.mean_rewards=1.02088, eval.ok=true, eval.all.mpjpe_g=31.925
```

Interpretation: SIM-M2-pre passes as a trained-checkpoint evaluation validity gate. It proves the harness can train the two micro variants, save distinct post-training checkpoints, record checkpoint provenance, evaluate each checkpoint under bounded MPJPE-complete eval, and compare the resulting manifests without control/metric/checkpoint warnings. It is still not an adaptive-sampling performance result because this is one seed, 10 iterations, tiny `sample_data`, and bounded smoke eval.

Validation after this change:

```text
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

39 passed in 1.97s
```

## Controlled variables for paper-grade experiments

Keep fixed unless explicitly ablated:

| Variable | Fixed value / policy |
|---|---|
| simulator | IsaacLab / Isaac Sim headless |
| env | `env_isaaclab` |
| robot | Unitree G1 29-DoF dex model |
| SONIC config | `manager/universal_token/all_modes/sonic_release` |
| action interface | 64D latent token + hands, unchanged |
| sample-data smoke | `sample_data/robot_filtered`, `sample_data/smpl_filtered` |
| seed policy | report seed and use paired comparisons for ablations |
| hardware | excluded until sim gates pass |

Candidate ablations after harness stability:

1. uniform motion sampling vs adaptive/curriculum stage sampling,
2. released checkpoint eval vs short finetune from checkpoint,
3. sample-data smoke vs full BONES-SEED filtered corpus,
4. stage-wise failure taxonomies rather than aggregate reward only.

## Next gates

### Gate A — sample health run

Exit evidence:

- 100 iterations complete with no traceback,
- log dir exists,
- config/meta written,
- final reward/termination metrics summarized,
- no claim beyond stack health.

### Gate B — released checkpoint eval/render smoke

Exit evidence:

- `gear_sonic/eval_agent_trl.py +checkpoint=sonic_release/last.pt` runs on sample data,
- metrics and/or render output produced,
- failures categorized as config/data/checkpoint/harness rather than hidden.

### Gate C — full motion-data acquisition path

Exit evidence:

- either HF gated access is available and `bones-studio/seed` G1 source can be downloaded, or an existing local source is located,
- `data/motion_lib_bones_seed/robot_filtered` and `data/smpl_filtered` are present,
- file counts and conversion/filter steps are logged.

### Gate D — paper experiment plan

Exit evidence:

- frozen baseline command,
- frozen curriculum command,
- paired seeds,
- metrics schema,
- failure taxonomy,
- stop/go criteria before larger compute.

## Interpretation so far

The project is viable in the intended simulation-first direction: IsaacLab launches, G1 asset conversion works after cache fix, and SONIC sample motion training runs. The main open blocker for paper-scale experiments is not hardware; it is full motion-data acquisition/preparation and a robust eval harness around released checkpoints.
