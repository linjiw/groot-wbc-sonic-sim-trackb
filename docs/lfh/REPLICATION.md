# Replicating the LFH / LfLH work on another machine

Everything needed to continue this line of work, written so a fresh checkout on a different PC can
get to the same state. Read `docs/lfh/lflh-relationship-and-3d.md` first for what the project is
trying to do, then this for how to run it.

**Status of the results, in one line:** the geometric LFH pipeline (closed-form window, physics
verification) is sound and its overhead results reproduce bit-exact; the learned LfLH results were
retracted twice and the current honest numbers are in
`docs/hallucination/lflh_sdf.json`. Do not quote anything from `REPORT_LFLH_MULTI_OBSTACLE.md`
addenda 2–3 — they carry a retraction banner.

---

## 1. What is in the repository, and what is not

| | where | in git? |
|---|---|---|
| Library | `gear_sonic/dataset_generation/hallucination/` | yes |
| Entry points | `scripts/research/hallucination/` | yes |
| Tests (872) | `tests/dataset_generation/` | yes |
| Reports, registers, JSON results | `docs/hallucination/`, `docs/lfh/` | yes |
| Constraint specs | `specs/hallucination/` | yes |
| Authored USD scenes | `gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e{3,6,7,10,17}/`, `hallucinated_variants_v1_e1/` | yes (LFS) |
| Rendered videos and figures | `docs/source/_static/{lfh_cases,lflh_scenes,lfh_e17}/` | yes |
| **Motion clips, rollouts, trajectories** | **`/data/robotixx/groot-wbc-kimodo-m0/` (466 MB)** | **NO** |

All 90 scene files carry the geometry the physics actually ran against: every one that a run
record or approved manifest pins by SHA-256 (36 + 16) was hash-verified before being committed. A
sweep over every committed JSON reports **no scene referenced but untracked**.

**The artifact tree is not in git and is the one hard dependency.** Every script defaults to
`DATA_ROOT = /data/robotixx/groot-wbc-kimodo-m0`. On a new machine either copy that tree to the
same path, or pass `--source-dir` / `--candidates` explicitly. The parts that matter:

```
sweepcf_release/motions/clips/        150 generated qpos CSVs, the motion pool
hallucination/run_records/*.json      immutable physics run records (the verdicts)
hallucination/<experiment>/           per-cell rollouts: trajectories/, success_manifest.json
lfh_crouch_calibration/, lfh_crouch_ladder/   prepared motion pairs
```

If you only want the **LfLH** work, you need `sweepcf_release/motions/clips/` and nothing else —
the candidate sets are built from reference clips by forward kinematics, no physics required.

## 2. Environment

Three environments, none interchangeable. See the root `README` table; the ones this work uses:

```bash
# Everything CPU-side: candidate sets, LfLH training, geometry, rendering, tests
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python <script>

# Tests -- the two flags are required, see docs below
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ~/miniconda3/envs/env_isaaclab/bin/python -m pytest tests/dataset_generation -q -p no:cacheprovider
```

* `env -u PYTHONPATH` — the shell profile puts ROS's Python 3.10 site-packages on `PYTHONPATH`,
  which shadows `pinocchio` with a 3.10 build and breaks collection under 3.11.
* `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — ROS's `launch_testing` plugin autoloads and fails on a
  missing `lark`.
* A `malloc_consolidate` / `corrupted size vs. prev_size` abort printed **after** the pass line is
  isaacsim/torch interpreter teardown in this conda env, not a test failure. Read the `N passed`
  line, never the exit status.
* Rendering needs `MUJOCO_GL=egl` (the scripts set it themselves) and a GPU with a free context.

For **physics** you additionally need Isaac Lab, `sonic_release/last.pt`, and `git lfs pull`.
`python check_environment.py --training` verifies this.

## 3. The LfLH pipeline, end to end

```bash
R=~/GR00T-WholeBodyControl && cd $R
PY="env -u PYTHONPATH $HOME/miniconda3/envs/env_isaaclab/bin/python"

# 1. Candidate sets: per clip, a nominal plus crouches and one-sided arm tucks,
#    with per-station up/left/right body extents in the executed route frame.
#    ~1 min per clip. 64 sets are already committed as lflh_candidates_large.json.
$PY scripts/research/hallucination/build_candidate_sets.py --clips 24

# 2. Train against TRUE signed clearance, with a held-out split. This is the
#    current pipeline; it supersedes train_lflh.py.
$PY scripts/research/hallucination/train_lflh_sdf.py \
    --clips 16 --holdout 5 --steps 1200 --samples 2 --stride 8 \
    --target crouch_070 --model-out /tmp/sdf.pt

# 2b. The bound the model is measured against: the best two-sided margin any single box can
#     reach per clip, by exhaustive search. Report efficiency against this, never a raw margin.
#     --stride MUST match the trainer's, or a box is searched at one resolution and scored at
#     another; that alone flipped a scene from passing to failing.
$PY scripts/research/hallucination/measure_scene_ceiling.py \
    --clips 16 --targets crouch_070 crouch_040 --epsilon 0.25 --stride 8

# 2c. What one forward pass is worth in units of search, plus the motion-blind ablation.
$PY scripts/research/hallucination/compare_amortisation.py --model-in /tmp/sdf.pt

# 3. Render generated scenes beside the exhaustive optimum for the same clip, ranked on
#    measured clearance and discrimination. --with-oracle is what makes the figure honest.
$PY scripts/research/hallucination/render_lflh_sdf_scenes.py \
    --target crouch_070 --epsilon 0.25 --clips 16 --holdout 5 --stride 8 --draws 24 \
    --render-top 3 --with-oracle --model-in /tmp/sdf.pt \
    --ceiling docs/hallucination/scene_ceiling_16.json \
    --out-dir docs/source/_static/lflh_ceiling

# render_lflh_scenes.py is the OLDER envelope-decoder renderer, retained only so the
# retraction stays reproducible.
```

`train_lflh.py` is the **older** envelope-decoder trainer. It is retained because the retraction
must stay reproducible, not because it should be used.

## 4. The geometric (non-learned) pipeline

This is the part that works and is physics-verified.

```bash
# Screen the clip pool for usable crouch amplitudes
$PY scripts/research/hallucination/screen_crouch_ladder.py

# Propose a scene for one motion, from D_phi, refusing empty windows
$PY scripts/research/hallucination/propose_scene_from_motion.py \
    --reference-csv <clip.csv>

# Author a scene from an accepted executed pair, then verify in physics
$PY scripts/research/hallucination/synthesize_from_ladder.py
$PY scripts/research/hallucination/build_ladder_family_manifest.py --seed 34007
$PY scripts/research/hallucination/approve_phase2_manifests.py \
    --timestamp "$(date -Iseconds)" --only E17_LADDER_FAMILY_PROPOSED.json
$PY scripts/research/hallucination/run_approved_manifest.py \
    --manifest docs/hallucination/manifests/E17_LADDER_FAMILY_APPROVED.json \
    --run-record <record.json>
```

**Governance applies** (`docs/hallucination/GOVERNANCE.md`): register predictions before spending
GPU, write a hash-pinned manifest before launch, 8 contended GPU-h/day. New manifests must have
their SHA-256 added to `SOURCES` in `approve_phase2_manifests.py` before they can be approved.

## 5. Reproducing the key results

```bash
# The golden check: CAL3 support re-derives bit-exact from stored trajectories.
# If this breaks, a geometry change broke something published.
$PY - <<'EOF'
import sys, pickle; sys.path.insert(0, '.')
from pathlib import Path
from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload
b = Path('/data/robotixx/groot-wbc-kimodo-m0/hallucination/crouch_calibration/lfh_089_crouch')
r = {}
for role in ('nominal', 'adapted'):
    res = best_evaluable_payload(pickle.load((b/role/'trajectories/000000.trajectory.pkl').open('rb')))
    t = extract_keypoints(res[0] if isinstance(res, tuple) else res)
    r[role] = overhead_face_reach(t, (1.8716927, 0.1385739), 'x', 0.10, 3.0).reach_m
print('EXACT:', r['adapted']+0.018044 == 1.244572004265374
              and r['nominal']-0.018044 == 1.2781074882246337)
EOF

# D_phi, the one validated learned component
$PY scripts/research/hallucination/build_delivery_corpus.py
$PY scripts/research/hallucination/build_reach_response_corpus.py
$PY scripts/research/hallucination/fit_reach_delivery_model.py   # 39.2% RMSE cut vs identity

# 94-clip case study and its figure
$PY scripts/research/hallucination/run_case_study.py
```

## 6. What is true, what is retracted

**Stands.**

* Overhead critical-support geometry; CAL3 golden re-derives bit-exact after 11 code fixes.
* `D_phi`: executed reach from a reference clip, RMSE 9.71 mm vs identity's 15.95, leave-one-motion-out
  over 46 clips (`REPORT_DELIVERY_MODEL.md`).
* Verified families E17 (4 cells) and E18 (12 cells, four archetypes at one critical point).
* LFH-E12 ladder, LFH-E16b seed repeatability (window range ≤ 7.94 mm at a stable amplitude).
* `regret = xi * |W|` and `regret + necessity = delta_strike + |W|`, both test-verified.
* The LfLH decoder *decides*: empty scene → nominal, overhead bar → crouch, left obstacle → left
  tuck. Dual objective works: relaxing a scene by 10 cm restores the nominal in 23/24.

**Retracted — do not cite.**

* The archetype-conditioning result (`REPORT_LFLH_COMPARISON.md`, fully retracted): feature-blind
  counting beat it.
* `REPORT_LFLH_MULTI_OBSTACLE.md` addenda 2–3: match 0.990 / plausibility 1.00 were measured on an
  inert obstacle; the explained motion physically intersected an obstacle in 21/24 scenes; the
  ablation control was confounded; there was no train/test split.

**Current LfLH numbers** — `docs/hallucination/lflh_sdf_c070_eps.json`, held out, `crouch_070` at
`eps = 0.25`:

| | model | random from the prior |
|---|---:|---:|
| selection | 81.2% | 58.8% |
| robot clear | 50.0% | 0.0% |
| **counterfactual** (clears *and* is chosen) | **32.5%** | **0.0%** |
| best-of-24 margin vs the exhaustive optimum | **72% median** | — |
| beats a 24-draw random search | **5 / 5 clips** | — |
| beats a **1024**-draw random search | **5 / 5 clips** | — |

Report the counterfactual rate, never selection alone: the random control scores 58.8% on selection
while burying the robot 320 mm deep, because the deepest crouch wins by being swallowed least.

`docs/hallucination/lflh_sdf.json` is the earlier run and is **not** a result — its objective was
infeasible and its clearance metric was the training surrogate. Retained as the record of a
diagnosed failure.

## 7. Known open issues

1. ~~**Robot-clear rate is under 20%.**~~ **Explained, 2026-08-28.** The barrier was correct and
   the model was not under-trained: the *target* was outside the feasible set. Demanding 40 mm of
   clearance plus 30 mm of strike at a 40 mm crouch exceeds what the geometry can hold — an
   exhaustive box search puts the ceiling at a median of 33.2 mm. The run's final barrier of 0.011
   per clip-sample decodes to a 55 mm residual, matching its own reported median clearance to
   1 mm. See `docs/hallucination/REPORT_SCENE_CEILING.md`. Margins are now derived from the
   amplitude, and `measure_scene_ceiling.py` reports the bound any hallucinator is measured against.
   A second defect fell out of the same reading: every "robot clear" figure before that date used
   the *soft* minimum the loss is trained through, which is never below the true distance, so a
   scene could be reported clear with the robot inside a box. `exact_clearance` is now the reported
   quantity.
2. **Missing from the design spec**: trackability and progress terms in the candidate cost, a
   secondary-contact penalty over links outside the binding group, and a structured constraint
   `kappa` with an explicit regime and surface normal (obstacles are currently axis-aligned boxes
   with direction only implicit).
3. ~~**No checkpoints are saved**~~ — `train_lflh_sdf.py --model-out` is now used by every run, and
   `render_lflh_sdf_scenes.py --model-in` / `compare_amortisation.py --model-in` re-evaluate a
   saved model without retraining.
4. **`tuck_right` does not amortise** although it works per clip — likely a sign convention in the
   lateral gate, since `tuck_left` does.
5. **Oriented faces are half-done**: `overhead_face_reach` accepts `route_yaw_rad`, but scene
   authoring and keep-out are still axis-aligned, so no curved-route family can be built yet. The
   94-clip case study measures the cost: misalignment reaches 52.6°.
6. **Scaling untested.** `lflh_candidates_large.json` has 64 sets ready for it.
7. **`crouch_040` does not train**, though its ceiling is the *largest* (38.2 mm against
   `crouch_070`'s 20.8 at the same `eps`). A shallow target leaves the deeper crouches as escape
   routes for the decoder. Geometric feasibility and learnability point in opposite directions;
   see `REPORT_SCENE_CEILING.md` section 5.0.
8. **Conditioning is unproven.** The model is genuinely conditional -- blinding the motion moves its
   output by 84% of its across-clip spread -- but conditioning beats the motion-blind control on
   only 2 of 5 held-out clips. Needs `lflh_candidates_large.json` (64 sets) to settle.
9. **The rival set is the open question.** "The scene excludes the alternative" has two readings —
   some cheaper candidate is struck, or every cheaper candidate is. They coincide only when the
   edit ladder has a single rung. `measure_scene_ceiling.py` now reports both; LFH-E19 in the
   prediction register registers what the difference is expected to be, before the run. Regret
   must be applied as a *margin*, not a gate: gating puts the optimum on a rival's zero crossing,
   where a change of frame stride from 4 to 8 flipped the verdict on clip 030.
