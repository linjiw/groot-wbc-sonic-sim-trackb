# LFH Empty-Room Probe Report

The V5 batch completed **6/6 gradeable captures**. The one registered prediction was confirmed: `mf_005_c08` nominal is accepted.

| pair | nominal | adapted | interpretation |
|---|---|---|---|
| `mf_005_c08` | accepted | rejected | nominal confirmed; crouch refused by endpoint transport cost |
| `lfh_089_arm_tuck_left` | accepted | accepted | eligible arm-tuck response pair |
| `lfh_092_arm_tuck_right` | accepted | accepted | eligible arm-tuck response pair |

## Delivery observations

- `lfh_089_arm_tuck_left`: commanded body window 56.58 mm; executed 4.32 mm; delivery ratio 0.076; prior predicted 42.00 mm; scene proposal **refused (<20 mm)**.
- `lfh_092_arm_tuck_right`: commanded body window 58.00 mm; executed 17.41 mm; delivery ratio 0.300; prior predicted 41.62 mm; scene proposal **refused (<20 mm)**.

This batch contributes **20** rows to `docs/hallucination/lfh_cpu/keypoint_response.csv`. At this gate D_phi remains honestly refused: **0 models fit**, because the first batch supplies only one amplitude per side/keypoint, below the three-level contract.

All six cells recorded zero external collision force. The `mf` adapted rejection is therefore transport-limited: endpoint error 0.396 m exceeds the 0.35 m trackability gate, with no scene-contact confound.

Actual V5 serial spend: **0.059 contended GPU-h** (plus 0.010 GPU-h for the preserved, verdict-void V3 plane capture).
