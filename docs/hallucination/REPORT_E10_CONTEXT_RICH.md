# E10b Multi-Obstacle Context Report

The seed-matched five-obstacle scene is **verified**.

| role | outcome | external-contact attribution |
|---|---|---|
| `nominal_easy` | accepted | none |
| `adapted_easy` | accepted | none |
| `nominal_hard` | rejected | binding_constraint |
| `adapted_hard` | accepted | none |

The nominal-hard torso contact occurs at frame 120 and is uniquely attributed to `/World/ConstraintFrame/BindingHangingPanel` before drift at frame 135. Four context obstacles remain non-causal; the CPU minimum context clearance is 288.78 mm.

The lateral, floor-level, and high-overhead faces demonstrate route-relative scene composition only. They are not promoted into verified critical-axis support.
