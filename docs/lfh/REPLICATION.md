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
| Authored USD scenes | `gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e17/` | yes (LFS) |
| Rendered videos and figures | `docs/source/_static/{lfh_cases,lflh_scenes,lfh_e17}/` | yes |
| **Motion clips, rollouts, trajectories** | **`/data/robotixx/groot-wbc-kimodo-m0/` (466 MB)** | **NO** |

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
    --clips 16 --holdout 5 --steps 1200 --samples 2 --stride 8

# 3. Render generated scenes against the motions that produced them.
$PY scripts/research/hallucination/render_lflh_scenes.py \
    --target crouch_040 --clips 12 --steps 1800 --anneal-prior \
    --out-dir docs/source/_static/lflh_scenes
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

**Current honest LfLH numbers** — `docs/hallucination/lflh_sdf.json`, out of sample:
selection 28.7% vs 15.0% random, robot-clear 18.8% vs 0.0% random.

## 7. Known open issues

1. **Robot-clear rate is under 20%.** The signed-clearance barrier is correct but the model is
   under-trained against it. Longer runs, and a `m_hit`/`m_clear` two-sided margin loss.
2. **Missing from the design spec**: trackability and progress terms in the candidate cost, a
   secondary-contact penalty over links outside the binding group, and a structured constraint
   `kappa` with an explicit regime and surface normal (obstacles are currently axis-aligned boxes
   with direction only implicit).
3. **No checkpoints are saved** by the trainers, so no reported model can be re-evaluated. Add
   `--model-out` usage (the flag exists in `train_lflh_sdf.py`).
4. **`tuck_right` does not amortise** although it works per clip — likely a sign convention in the
   lateral gate, since `tuck_left` does.
5. **Oriented faces are half-done**: `overhead_face_reach` accepts `route_yaw_rad`, but scene
   authoring and keep-out are still axis-aligned, so no curved-route family can be built yet. The
   94-clip case study measures the cost: misalignment reaches 52.6°.
6. **Scaling untested.** `lflh_candidates_large.json` has 64 sets ready for it.
