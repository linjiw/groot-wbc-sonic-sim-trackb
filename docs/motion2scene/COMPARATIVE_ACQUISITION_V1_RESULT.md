# Comparative acquisition v1: current result

0/20 first-slice physics runs complete; recorded status `yielded_gpu_contention`. No new paired learning labels or selector fits are claimed. The latest GPU gate yielded at 5224 MiB free against the fixed 7500 MiB floor.

The frozen generator and no-contrast baseline now receive the actual deployed d040 CSV geometry. Reconstructing both SONIC entries gives maximum root-array difference 1.4901161e-8 m and joint/axis-angle difference 3.3651304e-11, within the declared 1e-7 absolute tolerance; FPS and field sets match exactly. No d055 reference is loaded. Legacy helper slot names are explicitly mapped to d040 arrays.

| Arm | Assigned generated slots | Geometry-admitted | Contrast geometry passes | Station range |
| --- | --- | --- | --- | --- |
| uniform | 9 | 9 | 0 | 0.347–0.885 |
| analytic | 9 | 8 | 9 | 0.400–0.680 |
| no_contrast | 9 | 9 | 0 | 0.152–0.827 |
| motion2scene | 9 | 9 | 9 | 0.529–0.557 |

These are reference-geometry outcomes, not action labels. Motion2Scene concentrates its assigned scenes near station 0.53–0.56; the analytic arm covers 0.40–0.68. The analytic slot overlapping a reserved test neighborhood is retained as a charged rejection. This concentration could matter for training coverage, but no downstream comparison has tested that hypothesis.

All 64 generator outputs and costs are retained. Thirty-six generated slots are assigned, with three common background scenes reused across four arms for 48 arm-assigned groups (39 unique scenes before rejection). The first slice contains only the first two fixed slots per arm when eligible plus absent and blocked controls. Remaining slots and raised controls require a subsequent spend manifest; no outcome-dependent refill is allowed.

Direct capture records the 214 features, delivered rays, joint order, robot state, reference phase and clocks before the actual command. An independent registered audit will compare that capture with the recorder, recompute features and check the loaded reference banks. Old 0.20 s labels remain separate.

Final transfer ancestors remain unreserved; the third bounded inventory exceeded its 180 s ceiling. [Retained failure](evidence/final-transfer-v3-failure.json).

Validation: 190 impacted tests passed, one optional-dependency test skipped. All 229 registered artifact hashes matched. An AST comparison confirms the frozen command logic is unchanged except for the direct pre-command capture block. This is software validation, not an execution result. [Validation record](evidence/comparative-acquisition-prelaunch_validation.json).

The main 24/48/96-data-budget comparison, robot-data fitting, independent-layout evaluation and final source transfer remain unfinished. Geometry admission is not proof that these examples teach a useful selector.

[Protocol](COMPARATIVE_ACQUISITION_V1.md) · [Frozen proposals](evidence/comparative-acquisition-proposals.json) · [Run record](evidence/comparative-acquisition-run_record.json) · [Source snapshot](evidence/comparative-acquisition-source-20260906/research-source.tar.gz)
