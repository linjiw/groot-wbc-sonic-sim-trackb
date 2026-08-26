# LFH Phase-1 Coverage

**Status:** CPU-only sweepcf_dcs_v2 baseline, generated from the additive release
index.
Physics verdicts are unchanged; source-family scenes were read only to rebuild their measured index
fields. No frozen scene, rollout, or prediction-register artifact was used for target selection or
modified.

## Evidence Accounting

The repaired contract yields **2 independent causal
families** across **3 verified scene variants**. This passes the
Phase-0 review's fixed acceptance test of exactly two independent families.

| causal source | verified variants | artifact ids |
|---|---:|---|
| `cf_005_056` | 2 | `duck_002`, `duck_003` |
| `mf_005_c08` | 1 | `mf_005_c08` |

Variants are retained as scene evidence but never advance the independent-family count.

Route intersection is necessary but not sufficient for source verification. The repaired contract
refuses **2 artifact variants** whose loaded constraint is
inconsistent with its manifest binding station; the largest measured offset is
**2000.0 mm**. These variants remain visible in the index as
`not_verified` instead of being discarded or counted.

## Coverage Baseline

- Episodes indexed: **68** (60 physics-graded).
- Coupling-valid keypoint-DCS bins occupied by canonical cells from exact verified variants:
  **2 / 120**
  (1.7%).
- Eligible canonical occupancy rows: **12**; all observed
  episodes would touch **3** target bins, but probe
  and refused-variant rows do not suppress causal-family targets.
- Empty generation targets emitted: **118**.
- Motion rows: **150**; reference semantic gate known for
  **75**; executed empty-room trackability known for
  **0**.
- **45** motions have acceptance evidence only in nonempty scenes;
  this is not promoted to LFH trackability. Fully gated LFH candidates today: **0**.

The v2 denominator contains only coupling-valid crouch/overhead and arm-tuck/lateral templates in
`configs/research/sweepcf_dcs_v2.json`. It removes the 48 v1 crouch/shoulder cells that
`local_crouch` cannot target. Ground-support and unavailable edit operators are not silently
counted as missed opportunities. Absolute DCS occupancy remains a reporting/novelty statistic;
trajectory-conditioned support, route phase, and finite face extent determine proposal feasibility.

## Unknown-Bin Audit

| field | unknown episodes | share |
|---|---:|---:|
| `source_family_id` | 24 | 35.3% |
| `constraint_axis` | 24 | 35.3% |
| `constraint_coordinate_bucket` | 24 | 35.3% |
| `binding_keypoint` | 32 | 47.1% |
| `margin_bucket` | 32 | 47.1% |
| `edit_behaviour_class` | 14 | 20.6% |

Unknowns are retained rather than reconstructed from names. In particular, the present index does
not carry a measured lateral gap coordinate for the wall attempts, so those coordinates remain
unknown and lateral target bins remain honest targets. `duck_000` is also intentionally source
`unknown`: it is a two-probe work directory with no `family.json`, unlike `duck_001` and later
variants whose manifests explicitly identify `cf_005_056`.

## Reproducible Validation

The reported **21 focused tests** are 10 LFH contract tests, 5 route-intersection tests, and 6
sealed-split immutability tests. They are run as an explicit list because collecting the entire
`tests/dataset_generation/` directory on this machine hits a pre-existing pandas/numpy version
incompatibility (numpy 1.21.5 while the installed pandas requires at least 1.22.4):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv_sim/bin/python -m pytest   tests/dataset_generation/test_sweepcf_coverage.py   tests/dataset_generation/test_scene_route_check.py   tests/dataset_generation/test_scene_first_testset.py -q
```

## Generated Artifacts

- [`marginal_coverage.csv`](coverage/marginal_coverage.csv): every configured marginal plus
  observed extras and explicit unknowns.
- [`joint_binding_axis_margin.csv`](coverage/joint_binding_axis_margin.csv): semantic anatomy ×
  constraint axis × signed margin.
- [`joint_behaviour_operator_role.csv`](coverage/joint_behaviour_operator_role.csv): edit behaviour
  × operator × four-cell role.
- [`coverage_curve.csv`](coverage/coverage_curve.csv) and
  [`coverage_curve.png`](coverage/coverage_curve.png): coverage growth by independent causal source,
  not by scene variant.
- [`targets.json`](coverage/targets.json): ranked empty admissible bins for later CPU proposal work.
- [`summary.json`](coverage/summary.json): machine-readable values used in this report.

## Interpretation and Gate

The current evidence is overhead-only at the independent-family level and concentrated in the
`head_torso` semantic capsule group. The high empty-bin count is therefore a measured description
of corpus thinness, not evidence that generated scenes should be rolled out immediately. Phase 2
must first provide tested placement and keepout machinery, and every later physics batch remains
manifest- and user-approval-gated.

Reproduce this report with:

```bash
python scripts/research/hallucination/render_coverage.py \
  --episodes <release>/index/episodes.csv \
  --families <release>/index/families.csv \
  --motions <release>/index/motions.csv \
  --out docs/hallucination/coverage --report docs/hallucination/COVERAGE.md \
  --expect-independent-families 2
```
