# Independent-layout evaluation: frozen learners

**120/600 assigned physics executions completed; 120 admitted after block audits.** The first wave assigns 120 executions at station 0.35. Remaining assignments are preserved, and this is not a completed-panel success rate.

Measured cost in this evaluation: **1.123825 contended GPU h**. Training remains fixed at eleven complete encounters per arm, twenty fitted policies and source 41002. Five optimizer seeds do not create five datasets or independent sources.

Among admitted executions, 0 request d040; 82 log refusal. Walking succeeds in 44 refusal cases, so those neither-feasible predictions are contradicted by the measured fallback outcome. Refusal is a classification output, not a qualified stopping action.

The headline comparison is Motion2Scene versus the strong analytic data arm. All contact failures remain in the reported denominators. Pending or unadmitted cells remain distinct from measured failures.

The frozen scorer flags force >1 N and therefore accepts <=1 N, with body-origin crossing plus 0.3 s upright stability in the first episode. Peak forces, resets, denied requests and return logs remain in the complete rows. This is not a native-geometry continuous-time safety certificate or a guarantee of later neutral recovery.

| Layout / seed | Height | Uniform | Analytic | No contrast | Motion2Scene |
| --- | --- | --- | --- | --- | --- |
| layout_00 / 8511 | 1.180 m | 0/5 | 0/5 | 0/5 | 0/5 |
| layout_00 / 8512 | 1.180 m | 0/5 | 0/5 | 0/5 | 0/5 |
| layout_01 / 8511 | 1.270 m | 5/5 | 5/5 | 5/5 | 5/5 |
| layout_01 / 8512 | 1.270 m | 0/5 | 0/5 | 0/5 | 0/5 |
| layout_02 / 8511 | 1.360 m | 5/5 | 5/5 | 5/5 | 5/5 |
| layout_02 / 8512 | 1.360 m | 5/5 | 5/5 | 5/5 | 5/5 |
| layout_03 / 8511 | 1.180 m | pending | pending | pending | pending |
| layout_03 / 8512 | 1.180 m | pending | pending | pending | pending |
| layout_04 / 8511 | 1.270 m | pending | pending | pending | pending |
| layout_04 / 8512 | 1.270 m | pending | pending | pending | pending |
| layout_05 / 8511 | 1.360 m | pending | pending | pending | pending |
| layout_05 / 8512 | 1.360 m | pending | pending | pending | pending |
| layout_06 / 8511 | 1.180 m | pending | pending | pending | pending |
| layout_06 / 8512 | 1.180 m | pending | pending | pending | pending |
| layout_07 / 8511 | 1.270 m | pending | pending | pending | pending |
| layout_07 / 8512 | 1.270 m | pending | pending | pending | pending |
| layout_08 / 8511 | 1.360 m | pending | pending | pending | pending |
| layout_08 / 8512 | 1.360 m | pending | pending | pending | pending |
| layout_09 / 8511 | 1.180 m | pending | pending | pending | pending |
| layout_09 / 8512 | 1.180 m | pending | pending | pending | pending |
| layout_10 / 8511 | 1.270 m | pending | pending | pending | pending |
| layout_10 / 8512 | 1.270 m | pending | pending | pending | pending |
| layout_11 / 8511 | 1.360 m | pending | pending | pending | pending |
| layout_11 / 8512 | 1.360 m | pending | pending | pending | pending |
| control_absent / 8511 | no beam | pending | pending | pending | pending |
| control_absent / 8512 | no beam | pending | pending | pending | pending |
| control_raised / 8511 | 2.000 m | pending | pending | pending | pending |
| control_raised / 8512 | 2.000 m | pending | pending | pending | pending |
| control_blocked / 8511 | 0.000 m | pending | pending | pending | pending |
| control_blocked / 8512 | 0.000 m | pending | pending | pending | pending |

## Paired fit differences on completed traversal blocks only

| Optimizer seed | Matched layout/physics encounters per policy | Motion2Scene minus analytic passes | Difference on completed blocks |
| --- | --- | --- | --- |
| 8501 | 6 | 0 | +0.0 pp |
| 8502 | 6 | 0 | +0.0 pp |
| 8503 | 6 | 0 | +0.0 pp |
| 8504 | 6 | 0 | +0.0 pp |
| 8505 | 6 | 0 | +0.0 pp |

## Commands and measured contact on admitted blocks

Counts below use five fits per arm and block. A refusal executes the declared walking fallback; it does not stop the robot. When no learner requests d040, these executions leave its counterfactual physical outcome unmeasured.

| Block | Arm | d040 requests | Refusals | Resets | Maximum beam force through passage |
| --- | --- | --- | --- | --- | --- |
| 00 | uniform | 0/5 | 5/5 | 5 | 1243.498 N |
| 00 | analytic | 0/5 | 5/5 | 5 | 1243.498 N |
| 00 | no_contrast | 0/5 | 1/5 | 5 | 1243.498 N |
| 00 | motion2scene | 0/5 | 2/5 | 5 | 1243.498 N |
| 01 | uniform | 0/5 | 5/5 | 5 | 618.310 N |
| 01 | analytic | 0/5 | 5/5 | 5 | 618.310 N |
| 01 | no_contrast | 0/5 | 1/5 | 5 | 618.310 N |
| 01 | motion2scene | 0/5 | 2/5 | 5 | 618.310 N |
| 02 | uniform | 0/5 | 5/5 | 0 | 0.000 N |
| 02 | analytic | 0/5 | 5/5 | 0 | 0.000 N |
| 02 | no_contrast | 0/5 | 2/5 | 0 | 0.000 N |
| 02 | motion2scene | 0/5 | 2/5 | 0 | 0.000 N |
| 03 | uniform | 0/5 | 5/5 | 0 | 57.947 N |
| 03 | analytic | 0/5 | 5/5 | 0 | 57.947 N |
| 03 | no_contrast | 0/5 | 0/5 | 0 | 57.947 N |
| 03 | motion2scene | 0/5 | 2/5 | 0 | 57.947 N |
| 04 | uniform | 0/5 | 5/5 | 0 | 0.000 N |
| 04 | analytic | 0/5 | 5/5 | 0 | 0.000 N |
| 04 | no_contrast | 0/5 | 3/5 | 0 | 0.000 N |
| 04 | motion2scene | 0/5 | 4/5 | 0 | 0.000 N |
| 05 | uniform | 0/5 | 4/5 | 0 | 0.000 N |
| 05 | analytic | 0/5 | 5/5 | 0 | 0.000 N |
| 05 | no_contrast | 0/5 | 2/5 | 0 | 0.000 N |
| 05 | motion2scene | 0/5 | 2/5 | 0 | 0.000 N |

Exactly-1 N maxima: 0. Complete per-run probabilities, contact traces and command legality results are referenced by each published block result.

No significance or noninferiority claim follows from this partial, single-source development comparison. Absent/raised adaptation rates and blocked refusal rates belong to separate suites; refusal does not physically stop the robot.

The remaining work is to finish this fixed panel, execute the scripted comparators and perceptual shortcut diagnostics, evaluate genuinely new source ancestors under the same transition contract, and acquire the larger separately fitted data budgets with full cost accounting.

[Before-run protocol](INDEPENDENT_LAYOUT_EXECUTION_V1.md) · [Complete summary](evidence/independent-layout-summary.json) · [Master assignment](evidence/independent-layout-master.json) · [Learning comparison methods draft](LEARNING_COMPARISON_METHODS_DRAFT.md)

## Reproduce the fixed learners

The small [selector bundle](evidence/selector-bundle-20260906/selectors.tar.gz), [archive manifest](evidence/selector-bundle-20260906/selectors-manifest.json) and [instructions](evidence/selector-bundle-20260906/README.md) contain all twenty original checkpoints and the exact training inputs. The [CPU reproduction check](evidence/selector-bundle-20260906/selectors-reproduction.json) matches every weight, normalization value and final training loss. These verification fits do not add primary models or physics trials. SONIC and source motion assets remain external.

The [recorded evaluation inputs](evidence/independent-layout-decisions.json) also support portable CPU verification of each evaluated request, refusal and probability. Run `scripts/research/verify_motion2scene_layout_readouts.py` from the matching source snapshot. This checks the readout, not the physics or an unexecuted alternative's outcome.

With the source snapshot extracted at the working directory and the two selector archive/manifest files in `bundle/`, the CPU check needs NumPy and PyTorch (versions are recorded in the selector manifest):

```bash
PYTHONPATH=. python scripts/research/verify_motion2scene_layout_readouts.py \
  --bundle bundle --decisions independent-layout-decisions.json
```

[Measured readout reproduction receipt](evidence/independent-layout-readout-reproduction.json).

[Focused related-work positioning](RELATED_WORK_POSITIONING_20260906.md). The contribution under test is the training-data construction method; motion-based environment learning and perceptive humanoid skill selection already have relevant prior work.

## Post hoc readout diagnostics

These CPU observation interventions were specified after the first completed block. They measure decision sensitivity; no baseline physics success is inferred.

| Arm | Measured readouts | Request changes with absent rays | Refusal changes with absent rays | Constant-phase request agreement |
| --- | --- | --- | --- | --- |
| analytic | 30 | 0 | 0 | 0 |
| motion2scene | 30 | 0 | 4 | 30 |
| no_contrast | 30 | 0 | 6 | 30 |
| uniform | 30 | 0 | 2 | 30 |

Across both output heads, the largest absolute probability change is 1; the mean is 0.0833602. Exact cross-layout non-ray state matches: 6/6 block comparisons (including each seed's self-reference).

[Diagnostic protocol](LAYOUT_READOUT_DIAGNOSTIC_V1.md) · [Complete readout evidence](evidence/layout-readout_diagnostic_result.json)

## Execution replay

Fixed subset: all four data arms, optimizer seed 8501, both physics seeds 8511/8512, all three first-station heights (24 of 120 executions). Recorded Isaac Lab states and measured forces; MuJoCo renders the replay without integrating dynamics. Full capture intervals retain resets. One observed motion source; visual meshes are not the collision audit.

[Video](assets/independent-layout-wave1.mp4) · [Bound recording provenance](evidence/independent-layout-demo.json)

[Download the research source](evidence/independent-layout-source-20260906/research-source.tar.gz) · [Source manifest](evidence/independent-layout-source-20260906/research-source-manifest.json)

## Validation

The focused CPU regression suite passes 175 tests; one native-geometry test module skips because this CPU environment lacks the optional USD `pxr` package. Isaac Lab supplies USD for the actual geometry/physics audits. Exact commands:

```bash
PYTHONPATH=. .venv_research/bin/pytest -q tests/dataset_generation/test_motion2scene*.py tests/dataset_generation/test_capsule_box*.py tests/dataset_generation/test_placement_certificate.py
```
