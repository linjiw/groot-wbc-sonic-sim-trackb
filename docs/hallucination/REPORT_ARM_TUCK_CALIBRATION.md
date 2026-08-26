# LFH Arm-Tuck Delivery Calibration

Calibration result: **12/12 new empty-room captures accepted**, all with zero external collision. The four registered anchor-level acceptance predictions were confirmed.

| motion / side | alpha | commanded body window | executed body window | ratio |
|---|---:|---:|---:|---:|
| `lfh_089_arm_tuck_left` | 0.333 | 20.29 mm | -3.65 mm | -0.180 |
| `lfh_089_arm_tuck_left` | 0.667 | 42.22 mm | 1.78 mm | 0.042 |
| `lfh_089_arm_tuck_left` | 1.000 | 56.58 mm | 4.32 mm | 0.076 |
| `lfh_086_arm_tuck_left` | 0.333 | 20.60 mm | 7.47 mm | 0.363 |
| `lfh_086_arm_tuck_left` | 0.667 | 40.02 mm | 12.21 mm | 0.305 |
| `lfh_086_arm_tuck_left` | 1.000 | 57.28 mm | 8.33 mm | 0.145 |
| `lfh_092_arm_tuck_right` | 0.333 | 21.59 mm | 1.99 mm | 0.092 |
| `lfh_092_arm_tuck_right` | 0.667 | 43.40 mm | 2.70 mm | 0.062 |
| `lfh_092_arm_tuck_right` | 1.000 | 58.00 mm | 17.41 mm | 0.300 |
| `lfh_084_arm_tuck_right` | 0.333 | 21.90 mm | 1.44 mm | 0.066 |
| `lfh_084_arm_tuck_right` | 0.667 | 42.26 mm | 4.45 mm | 0.105 |
| `lfh_084_arm_tuck_right` | 1.000 | 56.82 mm | 5.25 mm | 0.092 |

## Conservative scene-spend gate

| side / binding model | estimate at 60 mm | lower bound | 20 mm gate |
|---|---:|---:|---|
| `wrist_left` / `lateral_left` | 6.66 mm | 1.10 mm | **refused** |
| `wrist_right` / `lateral_right` | 10.84 mm | 4.76 mm | **refused** |

The preregistered delivery prediction is **confirmed**: neither conservative wrist-response bound reaches the 20 mm scene-spend floor. Trackability is therefore not the blocker; executed operator delivery is. E3 lateral scene instantiation remains refused until a stronger existing-operator cohort produces an evidence-backed bound above the floor.

D_phi now fits **4 per-keypoint models** from 120 response rows; unsupported nonmoving keypoints stay explicitly refused. Models propose geometry only and never replace physics verdicts.

Actual serial spend: **0.116 contended GPU-h**.
