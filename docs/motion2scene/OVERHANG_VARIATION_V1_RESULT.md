# Overhang variation v1: complete Isaac result

All 42 registered Isaac Lab executions completed. 26/42 pass the original contact-free passage criterion; the denominator includes blind and blocked controls.

Registered predictions: `{"p1_nominal": true, "p2_pose_separation": false, "p3_negative_specificity": true, "p4_delay_boundary": true, "p5_measurement_contract": true}`.

| Condition | Reactive | Blind | Oracle |
| --- | --- | --- | --- |
| nominal | 2/2 | 0/2 | 2/2 |
| height_minus10mm | 2/2 | 0/2 | 2/2 |
| height_plus10mm | 2/2 | 0/2 | 2/2 |
| station_minus15cm | 1/2 | 0/2 | 1/2 |
| station_plus15cm | 2/2 | 0/2 | 2/2 |
| absent | 2/2 | not run | not run |
| raised | 2/2 | not run | not run |
| blocked | 0/2 | not run | not run |
| delay_100ms | 2/2 | nominal reused | nominal reused |
| delay_250ms | 2/2 | nominal reused | nominal reused |
| delay_500ms | 0/2 | nominal reused | nominal reused |

**The pose-separation prediction fails.** Reactive and oracle each pass 9/10 pose trials; all ten blind trials contact the beam. At station −15 cm in seed 8042, reactive and oracle both record 429.159 N in one 5 ms sample. They cross without a recorded fall, but fail contact-free passage. This is a limit of the executed reference/scene pairing, not a sensor-only error.

All six negative controls report zero overhang detections and no switch. Absent/raised pass 4/4; the two blocked controls collide, fail to cross and reset. Both 100 ms and realized 260 ms delay trials pass with zero measured force. Both 500 ms trials refuse switching and still contact (579.134 / 494.174 N). Refusal does not solve obstacle avoidance.

Actual new cost: 0.400278 contended GPU h. Capacity yields resume the same manifest and skip completed cells; no scientific replacement. See the complete 42-row JSON for each force peak/duration/impulse, reset/fall, packet, switch and bank audit.

A denied command is not successful traversal. These are five poses and two physics seeds on one previously observed source; no downstream policy was trained. The 250 ms request is delivered at 260 ms because sensing runs at 50 Hz. The obstacle in the blocked-underpass control is a floor-to-beam-top cuboid.

[Protocol](OVERHANG_VARIATION_V1.md) · [Results](evidence/overhang-variation.json) · [Manifest](evidence/overhang-variation-manifest.json) · [Next research gates](PROJECT_COMPLETION_STATUS.md)

The whole panel consumed 0.400278 contended GPU h; completing the remaining 31 cells after the previous 11-cell publication consumed 0.293148 h. [Completion lifecycle audit](evidence/variation-completion-lifecycle.json) verifies 1,264 artifact references and no remaining batch simulator processes.
