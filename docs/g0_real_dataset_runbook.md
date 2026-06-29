# G0B Real Dataset Golden-Path Runbook

G0B proves that one real, tiny SONIC/VLA static pick/place dataset can traverse
the CG-WBC research loop without changing SONIC/VLA runtime interfaces.

This is an infrastructure milestone, not a model-performance claim.

## Scope

Use a deliberately boring dataset:

- Robot: Unitree G1 with the SONIC/VLA data collection stack.
- Task: `pick up the red cup and place it on the tray`.
- Environment: fixed table, fixed tray, fixed red cup/can, fixed lighting, fixed cameras.
- Motion: static standing pick/place only.
- Excluded: walking, perturbations, clutter, synthetic data, SONIC retraining, ZMQ/runtime changes.
- Size: 10-20 honest episodes.

Suggested labels:

| Stage | Meaning | Suggested episodes |
|---|---|---:|
| S1 | static reach | 3-5 |
| S2 | grasp/lift | 3-5 |
| S3 | place nearby | 4-8 |
| S4 | stand-stabilized pick/place | optional 3-5 |

S5-S7 should normally be empty for G0B.

## Dataset contract

The cleaned dataset should look like a SONIC/VLA LeRobot dataset:

```text
/data/g1_fetch_place_tiny/
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
  data/
    train-00000-of-00001.parquet
  videos/
    observation.images.ego_view/
      *.mp4
```

Required action dimensions:

- `action.motion_token`: 64
- `teleop.left_hand_joints`: 7
- `teleop.right_hand_joints`: 7

Each episode should be truthfully labeled with:

```json
{
  "stage": "S3",
  "success": true,
  "fall": false,
  "drop": false,
  "difficulty": 0.0,
  "task_family": "fetch_place",
  "prompt": "pick up the red cup and place it on the tray",
  "object": "red_cup",
  "target": "tray"
}
```

If the raw collection path does not store those fields directly, pass a JSONL
label file to the manifest builder in a follow-up run. Do not modify the SONIC
runtime interface just to carry research metadata.

## Data quality rules

Accept:

- slow motions
- short pauses
- minor hand adjustments
- minor camera variation
- operator-like natural reaching

Reject or label as failure:

- falls
- object drops
- wrong object or target
- severe camera blackout
- missing action rows
- stale/static episodes
- mismatched frame/action counts
- aborted recordings without a failure label

A tiny truthful dataset is more valuable than a larger ambiguous one.

## Environment

Use the research-layer environment that already passes tests:

```bash
source /home/robotixx/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

## Command

Once `/data/g1_fetch_place_tiny` exists:

```bash
bash scripts/research/run_g0_golden_path.sh \
  --dataset /data/g1_fetch_place_tiny \
  --observation-config gear_sonic_deploy/policy/release/observation_config.yaml \
  --curriculum-config configs/research/curriculum_graph_fetch_place.yaml \
  --mvp-config configs/research/fetch_place.yaml \
  --report-dir outputs/research/g0_g1_fetch_place_tiny
```

If a GR00T checkpoint/repo boundary is available, add:

```bash
  --groot-repo /path/to/Isaac-GR00T \
  --model-path /path/to/checkpoint
```

If those are omitted, the runner records the open-loop step as an honest skip.

## Expected artifacts

```text
outputs/research/g0_g1_fetch_place_tiny/
  schema_check.log
  dataset_check.log
  dataset_check.json
  manifest.jsonl
  manifest_summary.json
  manifest_summary.md
  open_loop/
    status.json
  open_loop_status.json
  mvp_eval/
    report.md
    report.json
    manifest_summary.md
    mvp_eval_matrix.md
  curriculum_state.initial.json
  curriculum_state.updated.json
  g0_report.md
  g0_report.json
```

## G0B success criteria

A run satisfies G0B when:

1. G1 SONIC schema validation passes.
2. The real tiny SONIC/VLA dataset passes validation.
3. A curriculum manifest is generated.
4. A stage-wise dataset summary is generated.
5. The open-loop wrapper runs or records a clear skip/fail status.
6. The MVP eval matrix produces `report.md` and `report.json`.
7. The curriculum state updater consumes `report.json` and appends history.
8. No SONIC runtime, ZMQ protocol, observation ordering, or action interface is modified.

Expected state behavior for a 10-20 episode tiny dataset:

- S0 should be unlocked by construction.
- S1-S4 may have metrics, but should usually remain insufficient for full advancement.
- S5-S7 should remain locked/underfilled.

If a tiny static dataset unlocks the whole graph, treat that as a gating bug.

## Fixture dry-run

Before real robot data exists, the runner can be smoke-tested on the schema-only
fixture. This proves orchestration, not learning performance:

```bash
python scripts/research/create_tiny_sonic_vla_fixture.py \
  --output /tmp/sonic_vla_tiny \
  --episodes 2 \
  --frames 8

bash scripts/research/run_g0_golden_path.sh \
  --dataset /tmp/sonic_vla_tiny \
  --observation-config gear_sonic_deploy/policy/release/observation_config.yaml \
  --curriculum-config configs/research/curriculum_graph_fetch_place.yaml \
  --mvp-config configs/research/fetch_place.yaml \
  --report-dir outputs/research/g0_fixture_dryrun
```

The fixture is schema-only. Do not use it for training, model evaluation, or
robotics performance claims.
