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
