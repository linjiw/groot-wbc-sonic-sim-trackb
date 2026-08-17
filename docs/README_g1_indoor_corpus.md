# G1 indoor corpus — track index

Everything for the Kimodo + SONIC + Isaac Lab synthetic G1 dataset. This file is the map;
each entry says what the thing is *for*, because the filenames do not.

Work lives inside `GR00T-WholeBodyControl` alongside an unrelated LACE track. Files here are
the dataset track only; anything matching `*lace*` belongs to the other one.

## Start here

| Document | What it answers |
|---|---|
| [`plan_g1_indoor_corpus_v1.md`](plan_g1_indoor_corpus_v1.md) | The plan to a benchmark-grade, real-anchored dataset, with progress against it |
| [`datasheet_g1_indoor_corpus.md`](datasheet_g1_indoor_corpus.md) | What the corpus contains, its distribution, and its design criteria |
| [`guidance_2026-08-17_gate_and_labels.md`](guidance_2026-08-17_gate_and_labels.md) | Latest external guidance and what was done about it |

## Findings worth reading before changing anything

| Document | The finding |
|---|---|
| [`round2_diversity_findings.md`](round2_diversity_findings.md) | Five things the first diverse batch broke, and one hypothesis that was wrong |
| [`motion_prefilter_validation.md`](motion_prefilter_validation.md) | Joint saturation predicts self-contact at r=0.933 — and the two scales that are not interchangeable |
| [`kimodo_output_licensing.md`](kimodo_output_licensing.md) | We can release the motions; do not vendor the weights |

## Pipeline, in the order it runs

| Stage | Module | Script |
|---|---|---|
| Prompt design | `prompt_taxonomy` | `build_prompt_taxonomy.py` |
| Prompt encoding | `kimodo_prompt_cache` | `encode_kimodo_prompts.py` |
| Motion generation | — | `generate_kimodo_motions.py` |
| Reference screening | `motion_prefilter`, `self_intersection` | `screen_kimodo_motions.py` |
| Scene building | `clutter_scene_builder`, `eval_scene_builder` | `build_clutter_scenes.py`, `build_eval_scenes.py` |
| Route planning | `route_placement`, `scene_route_sampler` | `plan_kimodo_placements.py`, `sample_scene_routes.py` |
| Rollout | — | `run_kimodo_sonic_rollout.sh`, `run_behaviour_library_batch.sh` |
| Grading | `trajectory_acceptance`, `gate_policy`, `episode_outcome`, `trajectory_segments` | — |
| Contact analysis | `contact_decomposition`, `swept_volume` | — |
| Export | `trajectory_export`, `latent_parity` | `smoke_groot_sonic_loader.py` |
| Analysis | `behaviour_diversity`, `episode_semantics` | `report_behaviour_diversity.py`, `analyze_gate_margins.py`, `analyze_horizon_acceptance.py`, `analyze_reference_motions.py` |
| Review | `scene_build_sheet` | `build_corpus_review.py`, `render_corpus_review_page.py`, `build_scene_build_sheet.py` |

## Environments

Three, deliberately separate, exchanging files rather than sharing a dependency graph:

- **Isaac Lab** (`~/miniconda3/envs/env_isaaclab`) — rollouts, grading, analysis, tests
- **Kimodo sm_120** (`/data/robotixx/envs/kimodo_sm120`) — motion generation only
- **GR00T loader** (`/data/robotixx/envs/groot_loader`) — the pinned `ab88b50c` export smoke

Tests run as `env -u PYTHONPATH PYTHONPATH=<repo> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
<isaac python> -m pytest tests/dataset_generation -p no:cacheprovider`. The `-u PYTHONPATH`
matters: the shell profile puts ROS's site-packages ahead of this environment's.

## Working artifacts

`/data/robotixx/groot-wbc-kimodo-m0/` — on the second disk, because `/` runs at 91%.

- `taxonomy/` — prompts, encoded cache, generated motion libraries
- `review/` — the visual review package: clips by behaviour, traces, figures, `REVIEW.md`
- `*/rollouts/` — recorded trajectories, ego renders, success markers
- `dataset_v2/` — the exported LeRobot episodes

Scene packages live under `gear_sonic/data/assets/scenes/`, which is gitignored: scenes are
regenerable from their builder plus a seed, and the manifest records the hash.
