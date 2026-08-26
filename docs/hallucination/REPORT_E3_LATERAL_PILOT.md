# E3 Lateral-Gap Pilot Report

The registered four-cell prediction is **falsified**: 3/4 outcomes matched. Nominal-easy, nominal-hard, and adapted-easy reproduced the requested pattern, but adapted-hard contacted a binding panel and rejected.

| cell | outcome | expected | binding clearance | first contact | attribution | drift onset |
|---|---|---|---:|---:|---|---:|
| `nominal_easy` | accepted | accepted | 37.20 mm |  | none |  |
| `nominal_hard` | rejected | rejected | -18.24 mm | 118 | binding_constraint |  |
| `adapted_easy` | accepted | accepted | 72.34 mm |  | none |  |
| `adapted_hard` | rejected | accepted | 3.03 mm | 115 | binding_constraint | 142 |

The adapted-hard collision is causal, not secondary clutter: first contact at frame 115 is uniquely nearest `/World/ConstraintFrame/BindingPinchPanelLeft` and precedes 0.15 m reference drift at frame 142. Contact includes `right_hip_roll_link, right_wrist_yaw_link`. Its executed binding clearance fell to 3.03 mm from the +26.42 mm empty-room CPU prediction.

Therefore the candidate is **refused**, and the empty DCS bin is not marked covered. The result says a static empty-room capsule window, even after an E1a-derived margin, does not certify adapted clearance once the lateral constraint changes the controller trajectory. A wider retry would be a new registered intervention, not a repair of this result.

The accepted easy-scene trajectories provide the conservative retry gate at the same station and face extent. Their required full gaps are 0.976216 m nominal and 0.906489 m adapted. The adapted envelope already exceeds the target-bin upper edge (0.900 m) by 6.49 gap-mm before uncertainty; adding the registered 36.088 mm gap uncertainty requires at least 0.942577 m. Therefore no retry in the registered 0.8–0.9 m bin has a context-conditioned CPU clearance certificate. The bin remains refused; widening beyond 0.9 m would target a different DCS bucket.

Funnel: **24 CPU trials -> 1 instantiated candidate -> 4/4 rollouts -> 0 verified families**. Actual serial spend: **0.040 contended GPU-h**.
