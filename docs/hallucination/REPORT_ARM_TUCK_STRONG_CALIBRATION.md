# LFH Strong Arm-Tuck Calibration

CAL2 completed its registered funnel: **5 proposed -> 4 rolled out -> 3 accepted + 1 rejected -> 1 dependency skip**. Every accepted strong edit had zero external contact.

| motion / side | role | alpha | outcome | commanded body window | executed body window |
|---|---|---:|---|---:|---:|
| `lfh_086_arm_tuck_left` | strong | 2.000 | accepted | 116.50 mm | 128.46 mm |
| `lfh_094_arm_tuck_left` | nominal |  | rejected |  |  |
| `lfh_094_arm_tuck_left` | strong | 2.000 | skipped_dependency | 117.98 mm |  |
| `lfh_092_arm_tuck_right` | strong | 2.500 | accepted | 95.37 mm | 46.11 mm |
| `lfh_084_arm_tuck_right` | strong | 2.500 | accepted | 100.15 mm | 23.84 mm |

## Scene-spend decision

Strong-left is **refused as unidentifiable**: `094/left` nominal rejected for reference endpoint tracking and external robot contact, so alpha=2 has only one motion. The accepted `086/left` observation is retained, but it cannot license a scene or invalidate the supported CAL1 range.

Strong-right is fully replicated at alpha=2.5. At a mean wrist command of 261.33 mm, the corrected monotone model estimates 52.80 mm delivery with a conservative lower bound of **23.84 mm**. The 20 mm gate is therefore **cleared** for a finite-face CPU search; it does not license scene instantiation by itself.

The preregistered below-20 prediction is **partially falsified**: the right side is adjudicated and the left side is refused for missing matched evidence. No lateral scene is instantiated from an unreplicated side or from this scalar response result alone.

The response sidecar now contains **150 unique rows**; 4 supported per-keypoint models fit. Actual serial spend: **0.030 contended GPU-h**.
