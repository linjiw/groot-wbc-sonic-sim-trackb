# Gate audit, 2026-08-19: where the code actually is against the navigation plan

Produced by six independent readers, one per gate/dimension, each restricted to reading the
repository and the artifact tree and required to anchor every claim to a file path; then
synthesised and spot-checked. It is a *measurement of the codebase*, not a plan, and it is dated
because it will go stale.

Read it beside [the plan](guidance/2026-08-19_feasibility_first_navigation.md), which it corrects
in several places — the plan was written from documents, this was written from code.

---

# SweepCF — Research Lead Synthesis, 2026-08-19

Six readers, six dimensions, one plan. Below is what the code says, where the plan is wrong, and what to build. I spot-checked the load-bearing claims myself; corrections to the readers are marked.

---

## 1. Where the project actually is

**G0 — pre-registered selector comparison: IN PROGRESS, and it will not finish. Blocker: the frozen test set has no geometry.**
The honest verified-family count is **2 independent units of counterfactual evidence**, not 3. `mf_005_c08` is the one operator-built matched family (one 120-frame straight walk plus its own local_crouch 0.8 edit, overhead shelf, 97.7 mm window, executed root reaching x=3.45–4.09 m against a station at x=2.229 m — the route genuinely meets the obstacle). `duck_002` and `duck_003` are the *same* mined pair `cf_005_056` at two shelf heights, which the prereg's own rule ("five shelf heights on one nominal are never five families") makes one unit. Both units are the overhead regime on a straight-line walk: zero lateral, zero floor, one behaviour. Three counts on the record disagree and **all three err permissively**: the progress board says 3 (counting x002/x003, which the same page's withdrawal notice declares void); `report_dataset_counts.py` says 2 for `matched/` by re-admitting `mf_x003_c08`, whose cells stop 1.2 m short of the shelf, because `verdict()` never calls `scene_route_check`; and `sweepcf_release/index/families.csv` lists only the two duck families and omits `mf_005_c08` entirely. Meanwhile the frozen 30-scene test set is a spreadsheet — I confirmed `gear_sonic/data/assets/scenes/g1_counterfactual/` holds 48 `.usda` files and **zero** are `sf_*`; `materialize_test_scenes.py` has never been run with `--write`. Dataset A and dataset B do not exist. There is no loader from a verified family to a `Family`/`Candidate` object, so the selector has only ever seen `numpy.random`. The 0.963 headline is a synthetic harness smoke test whose ground truth is the rule the model is handed as a feature — reproducible, honest in its docstrings, and not evidence. Amendment 2 freezes the family set on 2026-08-24; the honest outcome is that it freezes at 2 and the comparison is **reported as not-run**.

**G1 — real data pipeline: IN PROGRESS, further along than the plan believes, but on the wrong corpus. Blocker: no exporter can take a counterfactual family cell to LeRobot.**
There are 29 real `synthetic_g1` LeRobot exports on disk with genuine Isaac-rendered 640×480 h264 ego video (248 frames @ 50 fps), the strict validator returns `ok: true, errors: 0`, and `smoke_groot_sonic_loader.py` returns a real 40-step `UNITREE_G1_SONIC` sample. The plan's "we only have a tiny fixture" is wrong. The real gap is sharper: **all 29 are placement/clutter-phase episodes** on straight two-point routes with generic task strings; `find … -name '*.parquet'` returns nothing under `matched/` or `counterfactual/`. The project's entire asset — the counterfactual supervision — exists only as 31 CSV rows and 25 trajectory pickles, in no trainable format. `export_kimodo_dataset_batch.sh` structurally cannot be pointed at a family (it demands an `EpisodeRequest→GenerationResult→ConversionResult` bundle no family directory has), and the corpus is positives-only by design, which is the exact inverse of what a pairwise-ranking feasibility loss needs. `distance_to_constraint` does not exist; it is computed and thrown away in three places (`scene_route_check.py:76`, `:111`, `local_adaptation.py:236`), each time inside a return statement that only wants a scalar. Depth: zero pixels exist anywhere; I confirmed `camera_data_types` appears in exactly three lines of one file and is set by no config.

**G2 — frozen-SONIC closed loop: NOT STARTED. Blocker: every motion assignment in the codebase is an episode reset.**
There is no closed loop and no trainable environment. All 89 observation functions and 26 term YAMLs are motion-reference or proprioception; there is not one exteroceptive, goal, route, or phase term. The only exteroception hook, `height_map`, is dead three ways (uninstalled GitHub-only package; rays hard-coded downward so an overhead beam is invisible by construction; targets `/World/envs/env_*/Object` with `/World/ground` commented out, so it cannot see a SweepCF scene at all). The action space is 29-D joint position; nothing selects among motions. `TrackingCommand._resample_command` is reachable only from reset or clip exhaustion and teleports the robot; all four wrapper APIs end in `reset_all()`. And I confirmed the hard cap: `scene_usd.py:20-25` raises unless `num_envs == 1`, measured at ~6.5 control-steps/s and 373–584 s per rollout under contention.

**G3 — motion bank and SONIC fine-tuning: IN PROGRESS on the bank, NOT STARTED on fine-tuning.**
Two adaptation operators are real and heavily rolled out (`local_crouch`, `local_arm_tuck`), with the causal invariants asserted in code. **Correction to the Gate-3 reader:** `deployable_retiming.py` is no longer untested and uncalled — the working tree now carries `tests/dataset_generation/test_deployable_retiming.py` (319 lines, 27 tests including the quaternion-sign-flip case the reader flagged as highest-risk) and `scripts/research/build_deployable_skill.py`, both uncommitted, written today at 16:08–16:17, after that audit. What remains genuinely missing is transitions (there is **no gait-phase representation anywhere** in the repo — grep for `gait_phase|stance_phase|swing_phase` in `dataset_generation/` and `scripts/research/` returns nothing) and a stop/abstain clip. Fine-tuning has never happened: all 26 `config.yaml` files under `logs_rl` carry `checkpoint: null` and `motion_file: sample_data/robot_filtered`. ONNX export's call site is commented out and both `exported/` directories are empty.

**G4 — sequential traversal: NOT STARTED.** It is downstream of two things that do not exist: phase-aligned transition clips and mid-episode `switch_motion`. Nothing else about it has been attempted.

**G5 — route counterfactuals Y[S,r,m]: NOT STARTED, and further from starting than the plan implies.** `sample_routes` draws a fresh random start/goal per attempt, de-dupes on the endpoint 4-tuple, and runs a deterministic A* that returns one optimal path — so **two routes sharing endpoints are not expressible**. The "independent_routes: 3" certification on all 12 eval scenes means three unrelated journeys, not a fork. `SampledRoute` carries no height envelope, so "passable if you crouch" is inexpressible. `ClutterSceneSpec` holds one `path_xy` and samples furniture randomly with no topology objective. No schema anywhere carries a `route_id`. The route is not a parameter of a family today; it *is* the family's substrate.

**G6 — sim2sim then real G1: NOT STARTED, zero execution.** Not one line of the SONIC deploy stack has ever run on this machine: no `build/`, no `target/`, no logs, no `state_logger` CSVs, no deploy artifacts. `git log` on `gear_sonic_deploy/` shows only upstream NVlabs merges. `ninja` and `clang` are missing, `TensorRT_ROOT` and `onnxruntime_ROOT` are unset, `local_env.sh` was never created. There is no working SONIC sim2sim path: `run_sim_loop.py` is a DDS-bridged MuJoCo robot waiting for `LowCmd` from a binary that has never been compiled.

---

## 2. Contradictions and corrections

The plan was written from documents. Here is every place the code disagrees.

1. **"The VLA interface has a skeleton but no real training data."** False. 29 real validated exports with real Isaac ego video, and the GR00T loader smoke test the plan lists as a Gate-1 item is already green. The true statement is: *no counterfactual cell has ever been exported*.
2. **"Depth is a configuration change plus a re-render, not new physics."** Wrong. `recorders.py:133` hardcodes `cam.data.output["rgb"]` and appends to an imageio h264 writer, a container that cannot carry metric depth. It needs a new recorder branch and a lossless sidecar (npz/exr), plus a contract and validator entry.
3. **"The paired learning-curve infrastructure can be reused directly."** It cannot. `plan_sonic_paired_learning_curve.py` hard-fails unless `sampler_pair_contract == 'official_failure_rate_vs_zpd_learnability_v1'` (l.140-142), unless checkpoint iterations are exactly (0,50,100,150,200) (l.113-116), and it calls a BONES-SEED-specific preflight (l.454). Its two arms are two *samplers*, not two controllers. It has no collision, edit-survival, or forgetting metric.
4. **"SONIC's pipeline supports fine-tuning from the release checkpoint and ONNX export."** True as a capability, false as a status. Never run from `sonic_release/last.pt`, never on a Kimodo/SweepCF PKL, and the ONNX export call site is commented out with both `exported/` dirs empty.
5. **"Wrap the 2×2 rollout into a trainable Isaac Lab environment."** Wrapping is not the work. The observation and action spaces the plan specifies do not exist in any form, the only exteroception hook is dead three ways, and `num_envs==1` is enforced by a raise.
6. **"Deep crouches are rejected because they cannot reach the goal."** Retracted the same day, in the project's own prediction register: `score_family_batch.py` bypassed `gate_policy.classify_episode`, so **endpoint error never gated those episodes**. The lag curve (0.257 → 0.509 m with crouch depth) survives as a quality column, not as a rejection. The live blocker in the overhead band is now `unstable_reference_drift`, which nobody has investigated. `deployable_retiming.py`'s own docstring still argues from the retracted framing.
7. **"The causal reference holds the root path fixed."** True for `local_arm_tuck` (full root xyz + quat bit-identical). For `local_crouch`, root XY and yaw are identical but root Z is deliberately lowered every frame (`local_adaptation.py:355`) so the soles stay planted — the pelvis descends ~0.14 m.
8. **"Multiple routes per scene already exist."** They exist and mean something else: three different start/goal pairs, used as a proxy for "this room is not a tunnel fitted to one trajectory". Reading `independent_routes: 3` as half-built route counterfactuals is wrong.
9. **"Preserve the 30-scene frozen test set as Stage 0."** There is nothing to preserve but a parameter table. Zero `sf_*.usda` exist (verified). Worse, `materialize_test_scenes.py:57-68` has a defect: the lateral regime's comment promises "one solid to each side, so the robot must pass between" and `piece_for` returns a single piece on the +y side only. Ten of the thirty frozen scenes do not implement the constraint they declare.
10. **"Pre-window frames are identical across labels by construction."** True of reference joint angles, unverified at the pixel level. Measured: nominal-vs-crouch frames within the same scene at f045–f060 differ by 0.74–2.08 mean-abs per channel with 4.5–7.6% of pixels off by >2, against 10.8–12.7 for the easy-vs-hard scene contrast. Probably H.264 artefacts; nothing tests or reports it.
11. **"Fitted counterfactual families are training material only" is an enforced rule.** It is dead code. `eval_scene_gate` has no non-test caller. Separately, `g1_counterfactual/` has no `manifest.json` and the rollout runner resolves scenes by globbing `<scene_id>.usda`, so family scenes bypass `scene_asset_preflight` entirely. Both mechanisms that would constrain a forked-route scene are inert.
12. **"Latent parity has been verified on real data."** It has, and it means less than it sounds. The check proves the recorded 64-D tokens are exact post-quantisation FSQ codes (residual RMS 0.0 over 259×64) and that the deploy YAML still declares `encoder.dimension: 64`. It never loads `model_encoder.onnx`, never runs the C++ encoder, and never compares two runtimes numerically. **Nothing establishes that `sonic_release/last.pt` — which produced every SweepCF label — and the deploy ONNX are the same model.** The one test cross-checking the closed-form lattice against the real FSQ is skipped for a missing dependency, so CI validates the lattice against itself.
13. **The plan's §9.3 deployment checklist is over-budgeted.** Watchdog (500 ms), token timeout (200 ms), planner timeout (1 s), E-stop, and per-stage latency timers are all already implemented upstream. Budget time for the *build*, not for these. What is genuinely missing is same-model sim/real parity and a navigation-manager arbitration between the `planner` and `pose` topics — the C++ `ZMQManager` forces a binary mode choice, so a body-aware planner cannot steer and emit motion tokens at the same time.
14. **Latent bug:** `deploy.sh`'s argument parser has a catch-all `*) INTERFACE_MODE="$1"` that silently swallows `--zmq-port`, `--zmq-topic`, `--zmq-conflate` — flags the binary supports and which both `docs/source/tutorials/zmq.md` and the streaming script's docstring instruct users to pass to `deploy.sh`. The first person on a non-default port silently gets 5556. Also `just test` is a stub that echoes and runs nothing; the only C++ test is FK.
15. **Internal integrity, independent of the plan:** the release's two indexes describe disjoint corpora (`families/index.jsonl` has `mf_005_c08` + `mf_x00*` + `duck_*`; `index/families.csv` has `duck_*` + `n_013_*`), and the headline verified family appears in zero rows of `episodes.csv`. Three columns are structurally empty from regex-over-directory-name bugs: `operator` 0/31, `behaviour_class` 'other' 31/31, `video_ego` 0/31 despite four real ego mp4s sitting in the release. And 28 of 78 GR00T action dims plus 14 of 43 state dims are hard-coded zeros (`neutral_hand`), which no validator catches.
16. **Correction to my own readers:** the Gate-3 report's "deployable_retiming has zero tests, zero callers, zero artifacts" is stale as of the working tree. It now has 27 tests and a driver script, uncommitted, written after that audit. The claim that it has never produced an artifact still holds.

---

## 3. The critical path to a running Gate-2 closed loop

**Host decision, made:** Gate 2 runs in **Isaac Lab at `num_envs=1`**. Gate 2 as the plan defines it is an *evaluation* loop over a single obstacle — depth in, one choice, frozen SONIC executes, goal reached — and does not need thousands of environments; the selector is trained offline on the counterfactual corpus. Isaac Lab is where frozen SONIC actually runs and where every accept/reject label in the corpus was produced. The mjlab port is the right bet for G4/T2+ *training*, and the SONIC-in-MuJoCo parity spike should run in parallel as a separate track — but Gate 2 must not wait on it.

| # | Build | File | Unblocks | GPU |
|---|---|---|---|---|
| 1 | Widen `RouteCheck` with a per-frame `distance_m`; add `capsule_box_clearance_series` returning per-frame min + binding capsule | `gear_sonic/dataset_generation/scene_route_check.py` | every downstream per-frame signal | no |
| 2 | `constraint_distance.py` + exporter over the 294 existing pickles | `gear_sonic/dataset_generation/constraint_distance.py`, `scripts/research/export_constraint_distance.py` | the "when to crouch" supervision; G1 record, G2 observation, G3 onset, G5 progress | no |
| 3 | Depth sink in the recorder + `camera_data_types: [rgb, depth]` in both config layers | `gear_sonic/envs/manager_env/mdp/recorders.py`, `gear_sonic/config/base_eval.yaml` | the Gate-2 observation; RQ2's depth tier | write no / exercise yes |
| 4 | `export_family_batch.py`: family cell → LeRobot, including rejected cells as a separate split, carrying physics labels + family/route/operator ids | `scripts/research/export_family_batch.py` (new) | any offline training of F(o,m) at all | no |
| 5 | Offline feasibility model: frozen backbone features over decision frames + depth → per-candidate survival | `gear_sonic/dataset_generation/feasibility_model.py` (new) | the "choose" in depth→choose→execute | yes (small) |
| 6 | `TrackingCommand.switch_motion(env_ids, motion_id, start_frame)` — re-run only the re-anchor block at `commands.py:3404-3422`, skip `write_joint_state_to_sim`/`write_root_state_to_sim`; drop `next(iter(library.values()))` in the runner | `gear_sonic/envs/manager_env/mdp/commands.py`, `scripts/research/run_kimodo_sonic_rollout.sh` | mid-episode behaviour change — the single highest-leverage env change | write no / verify yes |
| 7 | Multi-entry motion-library packer (nominal + crouch + tuck in one PKL) | `gear_sonic/dataset_generation/kimodo_motion_adapter.py` | a candidate bank inside one rollout | no |
| 8 | Goal-reached / collision / abstain terminations and an in-env success signal | `gear_sonic/config/manager_env/terminations/`, new termination fns | "reach goal" as an outcome rather than an offline grade | write no / run yes |
| 9 | Two-timescale driver: frozen SONIC at 50 Hz inner, selector at decision points outer, one obstacle, one scene | `scripts/research/run_closed_loop_episode.py` (new) | **Gate 2** | yes |

Items 1, 2, 4, 6, 7 are CPU. Items 3, 5, 8, 9 need the GPU only to *run*, not to write. That is the whole point of section 4.

---

## 4. What can be built RIGHT NOW without a GPU

Ranked by value. Every one is self-contained, unit-testable on CPU today, and most are prerequisites for two or more gates.

**1. Per-frame constraint distance — widen two return types, then add the module.**
Files: `gear_sonic/dataset_generation/scene_route_check.py` (edit), `gear_sonic/dataset_generation/constraint_distance.py` (new), `scripts/research/export_constraint_distance.py` (new).
What it does: stops the plan's most-wanted signal being destroyed inside three return statements, then joins executed root path + capsule clearance + obstacle station into a per-frame record written as an `.npz` beside each cell's `trajectories/`. Data already exists: every pickle carries `root_pos_w`, `body_pos_w`, `body_quat_w`, `body_names`, and every family scene's box is recoverable via `build_counterfactual_family.rendered_shelf_box`.
Tested by: synthetic straight-line paths with analytically known remaining arclength; a regression asserting the new per-frame series' min equals the old scalar `capsule_box_clearance`; a real-data smoke test on `mf_005_c08/w_nominal_hard` asserting the first negative-clearance frame precedes the recorded contact frame.
Unblocks: G1 (episode record), G2 (observation term), G3 (adaptation onset), G5 (route progress).

**2. Family-cell → LeRobot exporter, including rejected cells.**
File: `scripts/research/export_family_batch.py` (new, ~250 lines).
What it does: iterates `<family>/<cell>/{trajectories,renders}` — which already exist for 25 cells — and emits `synthetic_g1` episodes without the placement-batch `EpisodeRequest` bundle, writing `family_id`, `cell_role`, `operator`, `accepted`, `rejection_reasons`, `contact_body`, `max_force`, `drift_rate`, `min_clearance_mm`, `route_reaches_obstacle` into a `meta/labels.jsonl` sidecar, and routing rejected cells into a separate split.
Tested by: run the existing strict validator (`check_sonic_vla_dataset.py --groot-loader-ready`) on the output; assert the rejected split contains exactly the cells `gate_policy.classify_episode` rejects; assert every exported episode's label row round-trips to the same verdict as `episodes.csv`.
Unblocks: all offline feasibility training. Today **zero** counterfactual supervision is in a trainable format.

**3. Honesty fixes before the 2026-08-24 freeze.**
Files: `scripts/research/report_dataset_counts.py` (gate `verdict()`/`classify_family()` on `scene_route_check`), `scripts/research/build_dataset_index.py` (read `operator`/`behaviour_class` from the family manifest instead of regexing the cell directory name; fix the `video_ego` lookup), then regenerate `index/episodes.csv` over the same families-root that produced `families/index.jsonl`.
Tested by: a fixture family whose executed path stops short of the shelf must score `void`, not `verified` — that is exactly `mf_x003_c08`, so use it as the regression case; assert `operator` and `behaviour_class` are non-empty for every row of a regenerated index.
Unblocks: the integrity of the number Amendment 2 freezes by timestamp in five days.

**4. Materialise the frozen test scenes — after fixing the one-sided lateral gap.**
File: `scripts/research/materialize_test_scenes.py` (fix `piece_for` at l.57-68, then run with `--write`).
What it does: turns 30 rows of parameter table into 30 `.usda` files, minutes of CPU. Record the geometry sha256 and an explicit decision note against the freeze rules in `scene_first_v1.json`, because fixing the gap changes what those scenes mean.
Tested by: reuse `tests/dataset_generation/test_rendered_shelf_roundtrip.py`'s pattern — read each emitted scene back and assert the lateral regime yields two solids with the declared gap between them, and that the synthesized route enters each obstacle's footprint under `check_route_meets_obstacle`.
Unblocks: G0's primary and co-primary metrics, and every observation tier. This is the single cheapest thing standing between the project and a test set that exists.

**5. Gait-phase estimator + `motion_transitions.py`.**
File: `gear_sonic/dataset_generation/motion_transitions.py` (new).
What it does: estimates stance/swing phase from foot ground-contact forces (recorded per frame) or ankle height on a reference clip — a representation that exists nowhere in the repo today — then stitches walk → adapted → walk at matched phase with a C1-continuous blend. Put the estimator beside `local_adaptation._sole_height` so both read the same collision capsules.
Tested by: phase is monotone within a stride and wraps exactly once per contact cycle; a stitched clip has no velocity discontinuity above a stated threshold at either seam; the stitched root path deviates from the concatenated inputs by less than `ROUTE_TOLERANCE_M`.
Unblocks: G4, Isaac T2, and the plan's receding-horizon loop — none of which can run on frame-0-only entry into a skill.

**6. `switch_motion`, with the re-anchor math extracted as a pure function.**
File: `gear_sonic/envs/manager_env/mdp/commands.py`.
What it does: splits the re-anchoring block at `commands.py:3404-3422` out of `_resample_command` so the reference can be re-targeted mid-episode without a teleport. Extract `reanchor(delta_pos_w, delta_ori_w, body_pos_w, body_quat_w) -> ...` as a pure tensor function so the math is CPU-unit-testable even though the term itself needs Isaac to run.
Tested by: CPU tests on the extracted function — a switch at frame k with identical clips is the identity; a switch preserves the anchor's world pose; yaw composition round-trips. GPU only to confirm SONIC does not fall.
Unblocks: G2, G4.

**7. Real-family loader + geometry/motion feature extractors.**
File: `gear_sonic/dataset_generation/family_dataset.py` (new).
What it does: family manifest + scene USDA → `Family(scene={overhead_clearance_m, left_gap_m, right_gap_m, floor_height_m}, candidates=[Candidate(profile, cost, succeeds)])`. Reuse `swept_volume.body_capsules_world` and `motion_envelope` for the motion side; the geometry side is the shelf box read back from the rendered scene. Design the `Family` dataclass **with a route axis from the start** so G5 is a population, not a refactor.
Tested by: `mf_005_c08` loads to four candidates whose `succeeds` flags match `family.json`'s `cells` verdicts exactly; the derived `overhead_clearance_m` equals `easy/hard_shelf_underside_m` from the manifest to 1e-6.
Unblocks: G0's real-data selector, G5's route axis.

**8. Numeric encoder/decoder parity harness.**
File: `scripts/research/check_onnx_runtime_parity.py` (new).
What it does: runs `model_encoder.onnx` (1762→64) and `model_decoder.onnx` (994→29) on CPUExecutionProvider against the same observations the torch checkpoint sees, and asserts agreement. This is the parity claim everyone already believes exists.
Tested by: it *is* the test — max abs token difference and max abs action difference against stated tolerances, on frames drawn from a real rollout.
Unblocks: any claim that SweepCF labels are deployment-relevant. Also un-skip the FSQ cross-check by pinning `vector_quantize_pytorch` into the test extra.

**9. Same-endpoint multi-route planner + per-envelope route feasibility.**
File: `gear_sonic/dataset_generation/scene_route_sampler.py`.
What it does: `sample_routes_between(obstacles, room, start_xy, goal_xy, *, envelopes)` — fix endpoints, plan with the existing A*, mask the first corridor, replan for a second homotopy class; return a `RouteSet` carrying `route_id`, corridor label, `path_length_m`, `min_clearance_m`, and the envelope each route was planned for. Then score each route at several body envelopes so "passable only if crouched" becomes a field.
Tested by: a synthetic two-corridor room yields exactly two routes with the same endpoints and disjoint corridor cells; lowering `swept_top_m` opens the low corridor and raising it closes it (the mechanism is already proven by an existing test).
Unblocks: G5 entirely.

**10. MJCF renderer for SweepCF scenes.**
File: `gear_sonic/dataset_generation/clutter_scene_builder.py` (add `render_scene_mjcf` beside `render_scene_usda`).
Tested by: assert the USD and MJCF box sets are identical (centre + half-size) for every scene in `gear_sonic/data/assets/scenes/`.
Unblocks: the mjlab track, without committing to it.

**11. Small, cheap, correct:** fix `deploy.sh`'s flag swallowing; copy the full `cameras_cfg` block (intrinsics, extrinsics, clipping) into `runtime_manifest.build_runtime_capture_context` and delete the duplicated constants in `perception_timing.py`; write a driver that runs `perception_timing` against `mf_005_c08` for the first time; carry `dof_vel`/`root_lin_vel_w`/`root_ang_vel_w`/`root_quat_w` through `build_synthetic_g1_frame` (they are recorded and simply not copied); state in the contract that the hand channel is inert.

---

## 5. Recommended single next deliverable

**Build `gear_sonic/dataset_generation/constraint_distance.py`, plus the two return-type widenings it needs.** It is CPU-only, its inputs are already on disk for 294 rollouts, it is the plan's single most-emphasised missing field, and it is a prerequisite for G1, G2, G3 and G5 simultaneously. Specification follows; no further questions should be needed.

### 5.1 Edits to `gear_sonic/dataset_generation/scene_route_check.py`

Add a per-frame field to `RouteCheck`, defaulted and excluded from equality so existing construction sites and comparisons keep working:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RouteCheck:
    frames_inside: int
    closest_approach_m: float
    suggested_shift_x_m: float
    #: (T,) planar metres from each path point to the footprint; 0.0 while inside.
    distance_m: np.ndarray | None = field(default=None, repr=False, compare=False)
```

`check_route_meets_obstacle` already computes `distance` at line 76; pass it through unchanged (`distance_m=distance`). Do not alter any existing field, name, or `explain()` string.

Add a series form of the clearance, and reimplement the scalar in terms of it:

```python
def capsule_box_clearance_series(
    starts: np.ndarray,          # (T, C, 3)
    ends: np.ndarray,            # (T, C, 3)
    radii: np.ndarray,           # (C,)
    box: tuple[float, float, float, float, float, float],  # (x0,y0,z0,x1,y1,z1)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame smallest capsule-to-box gap and the index of the capsule that binds it.

    Returns ``(clearance_m, capsule_index)`` with shapes ``(T,)`` float64 and ``(T,)`` int64.
    Negative clearance means overlap by that much. Sampling density (9 points per capsule
    axis) is unchanged from ``capsule_box_clearance``.
    """
```

`capsule_box_clearance` keeps its exact current signature and `(float, int, int)` return, implemented as `series → global argmin`. Its numerical output must be bit-identical to today's; that is a test, not an aspiration.

### 5.2 New module `gear_sonic/dataset_generation/constraint_distance.py`

```python
"""Per-frame metres to the binding constraint, on the executed trajectory.

Three functions in this package already compute this quantity and destroy it inside their
own return statement. A model trained without it learns THAT a crouch is needed and not WHEN.
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ConstraintDistance:
    fps: float
    #: (T,) cumulative executed arclength normalised to [0, 1]; nondecreasing.
    route_progress: np.ndarray
    #: Total executed planar path length in metres.
    route_length_m: float
    #: (T,) signed arclength to the obstacle station: positive before, negative after.
    remaining_to_station_m: np.ndarray
    #: (T,) planar metres from the root to the obstacle footprint; 0.0 while inside it.
    footprint_distance_m: np.ndarray
    #: (T,) smallest gap from any collision capsule to the obstacle box; negative = penetration.
    body_clearance_m: np.ndarray
    #: (T,) index into the capsule array of the capsule that binds each frame.
    binding_capsule: np.ndarray
    #: (T,) name of the link owning the binding capsule.
    binding_body: tuple[str, ...]
    #: Frame at which |remaining_to_station_m| is smallest.
    station_frame: int
    #: Frame at which body_clearance_m is smallest.
    bottleneck_frame: int
    #: First frame with body_clearance_m <= 0.0, or -1 if the body never touches the box.
    first_penetration_frame: int
    #: True when the executed path has effectively zero length (robot never moved).
    route_degenerate: bool

    def as_dict(self) -> dict[str, object]:
        """Flat, npz-writable mapping. Arrays stay arrays; scalars stay scalars;
        ``binding_body`` becomes a ``(T,)`` array of dtype ``<U``."""
```

Public functions:

```python
def route_progress_and_arclength(root_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(progress in [0,1], cumulative arclength in metres, total length).

    ``root_xy`` is (T, 2). Matches ``local_adaptation.route_progress`` exactly for the
    normalised output, including its degenerate-path linspace fallback.
    """

def remaining_arclength_to_station(
    root_xy: np.ndarray,
    station_xy: tuple[float, float],
) -> tuple[np.ndarray, int, bool]:
    """Signed metres of executed path remaining until the station.

    The station is projected onto the path by nearest point (not nearest frame — a station
    beside the route must still be measured along the route). Returns
    ``(remaining_m, station_frame, degenerate)``; ``remaining_m[t] = arclength[station_frame]
    - arclength[t]``, so it is positive before the station and negative after.
    """

def constraint_distance_from_payload(
    payload: dict,
    box: tuple[float, float, float, float, float, float],
    *,
    station_xy: tuple[float, float] | None = None,
    fps: float | None = None,
) -> ConstraintDistance:
    """Assemble the record for one executed cell.

    ``payload`` is a trajectory-pickle payload — already passed through
    ``trajectory_segments.best_evaluable_payload`` by the caller — and must carry
    ``root_pos_w``, ``body_pos_w``, ``body_quat_w``, ``body_names``; ``fps`` falls back to
    ``payload["fps"]``. ``box`` is the world AABB physics loaded, obtained from
    ``build_counterfactual_family.rendered_shelf_box``. ``station_xy`` defaults to the box
    footprint centre.
    """

def write_npz(record: ConstraintDistance, path) -> "pathlib.Path":
    """Write ``record.as_dict()`` with ``np.savez_compressed``. Returns the path written."""
```

### 5.3 CLI `scripts/research/export_constraint_distance.py`

`--family-root <dir>` (a family directory, or a root containing many), `--scene-root` (defaults to `gear_sonic/data/assets/scenes/g1_counterfactual`), `--dry-run`. For each cell it resolves the scene `.usda` the cell was rolled in, reads the box back out of the file physics loaded (never recomputed from builder variables), loads and segments the pickle, and writes `<cell>/constraint_distance.npz`. It prints one line per cell with `frames`, `route_length_m`, `min_clearance_mm`, `station_frame`, `first_penetration_frame`, and a `SKIP <reason>` for any cell missing a scene or a pickle. It writes nothing when the record cannot be built; it never guesses a box.

### 5.4 Edge cases, all of which must be handled explicitly

- `T == 0` → raise `ValueError` naming the cell. `T == 1` → all series length 1, `route_length_m = 0.0`, `route_degenerate = True`.
- Zero-length path (robot never moved, total < 1e-9 m) → `route_degenerate = True`, `route_progress` falls back to `linspace(0, 1, T)` matching `local_adaptation.route_progress`, and `remaining_to_station_m` is the straight-line planar distance to the station rather than an arclength.
- Station never reached → `remaining_to_station_m` stays positive throughout; `station_frame` is the closest-approach frame. This is the `mf_x003_c08` case and must not raise.
- Penetration → `body_clearance_m` goes negative and must **not** be clipped at zero.
- Root path inside the footprint → `footprint_distance_m` is exactly `0.0` on those frames (not a small positive number).
- Missing collision links in `body_names` → catch `body_capsules_world`'s `ValueError` and re-raise with the cell path in the message.
- NaN or Inf anywhere in `root_pos_w`/`body_pos_w`/`body_quat_w` → raise; do not silently propagate.
- A payload spanning a reset → the module does not segment; it documents that the caller must pass `best_evaluable_payload(...)[0]`, and the CLI does so.
- `station_xy` supplied but far off-route → still valid; nearest-point projection handles it, and this is the case test 12 below covers.

### 5.5 Test file `tests/dataset_generation/test_constraint_distance.py`

Repo house style is sentence-shaped test names. Required cases:

1. `test_a_straight_walk_toward_a_box_counts_down_the_metres_that_remain` — a 100-frame path from x=0 to x=5 at constant speed with a station at x=3: `remaining_to_station_m` decreases monotonically, equals `3.0 - x_t` to 1e-9, and crosses zero at exactly one frame.
2. `test_a_path_that_stops_short_never_reaches_the_station` — the remaining series stays strictly positive, `station_frame` is the last frame, and nothing raises.
3. `test_a_station_beside_the_route_is_measured_along_the_route_not_across_it` — a station 2 m lateral of a straight path still projects to the nearest path point, and `remaining_to_station_m` is arclength, not Euclidean.
4. `test_the_per_frame_clearance_agrees_with_the_scalar_it_replaces` — for a random but seeded capsule set, `capsule_box_clearance_series(...)[0].min()` equals `capsule_box_clearance(...)[0]` exactly, and its `argmin` equals the returned frame.
5. `test_a_capsule_driven_into_the_box_reports_a_negative_gap` — clearance is negative on the overlapping frames and `first_penetration_frame` is the first such frame.
6. `test_a_body_that_never_touches_reports_no_penetration_frame` — `first_penetration_frame == -1`.
7. `test_route_progress_matches_the_construction_time_function` — `route_progress_and_arclength(path)[0]` equals `local_adaptation.route_progress(path)` elementwise, so the executed-side and reference-side definitions can never drift.
8. `test_a_robot_that_never_moved_is_flagged_rather_than_dividing_by_zero` — `route_degenerate` is True, no NaN appears in any series.
9. `test_the_root_inside_the_footprint_measures_exactly_zero` — `footprint_distance_m` is `0.0`, not `1e-17`, on interior frames.
10. `test_a_payload_with_a_nan_is_refused` — raises `ValueError`.
11. `test_the_record_round_trips_through_npz` — `write_npz` then `np.load` recovers every array bitwise and `binding_body` as strings.
12. `test_the_verified_family_penetrates_before_its_recorded_contact_frame` — marked `@pytest.mark.skipif` on the data path, loads `/data/robotixx/groot-wbc-kimodo-m0/matched/mf_005_c08/w_nominal_hard/trajectories/000000.trajectory.pkl` and the box from `mf_005_c08_hard.usda`, and asserts: `route_progress` is nondecreasing and bounded in [0, 1]; `route_reaches_obstacle` is true (`footprint_distance_m` hits zero); and `first_penetration_frame` is at or before the contact frame recorded in that cell's `success_manifest.json`. This is the test that proves the module measures the physical world and not just its own arithmetic.

**Done means:** `pytest tests/dataset_generation/test_constraint_distance.py tests/dataset_generation/test_scene_route_check.py -q` is green, `export_constraint_distance.py --family-root /data/robotixx/groot-wbc-kimodo-m0/matched` writes an `.npz` for every cell that has a scene and a pickle, and no existing caller of `capsule_box_clearance` or `RouteCheck` changed behaviour.