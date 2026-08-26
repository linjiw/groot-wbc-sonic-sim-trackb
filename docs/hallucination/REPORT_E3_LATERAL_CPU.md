# LFH Lateral-Gap CPU Synthesis

CAL2's strong-right delivery result licensed this CPU search, not scene physics. The search uses the full separation of two world-fixed faces, every collision capsule, and the operator's time-mapped active interval.

The fixed grid evaluated **24 trials**. **1** survived anatomy, temporal, 20 mm window, and 2x-noise-clearance gates; one best station per motion was eligible for selection.

| motion | status | station progress | face depth | raw window | certified window | binder |
|---|---|---:|---:|---:|---:|---|
| `lfh_084_arm_tuck_right` | CPU-certified | 0.3925 | 0.30 m | 165.80 mm | 56.41 mm | `wrist_right` |
| `lfh_092_arm_tuck_right` | refused |  |  |  |  | no temporally active, same-side, noise-certified full-gap window |

The selected `084/right` pinch-panel pair fills a configured DCS target and reproduces the required CPU sign pattern (easy: clear/clear; hard: strike/clear). Its panels extend from each inner face by the declared across-route extent, closing the earlier skirt path. `092/right` is refused because its apparent early-route windows bind before the strong edit becomes active; the central face has no positive all-body window.

No generated-scene verdict is claimed here. Physics remains the next gate.
