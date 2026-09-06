# Action labels and timing: complete result

All twelve label/timing executions completed; 16/16 scene–seed groups have complete, matched command-outcome labels. These remain development examples, not a training corpus.

Registered predictions: `{"p1_complete_paired_labels": true, "p2_control_outcomes": true, "p3_explicit_command_reproduction": true, "p4_legal_time_rescue_8042": true, "p5_measurement_contract": true}`.

| Seed | Condition | Walk outcome | d040 command outcome | Pair valid |
| --- | --- | --- | --- | --- |
| 8041 | nominal | False | True | True |
| 8041 | height_minus10mm | False | True | True |
| 8041 | height_plus10mm | False | True | True |
| 8041 | station_minus15cm | False | True | True |
| 8041 | station_plus15cm | False | True | True |
| 8041 | absent | True | True | True |
| 8041 | raised | True | True | True |
| 8041 | blocked | False | False | True |
| 8042 | nominal | False | True | True |
| 8042 | height_minus10mm | False | True | True |
| 8042 | height_plus10mm | False | True | True |
| 8042 | station_minus15cm | False | False | True |
| 8042 | station_plus15cm | False | True | True |
| 8042 | absent | True | True | True |
| 8042 | raised | True | True | True |
| 8042 | blocked | False | False | True |

The label compares committing to neutral with a request to enter d040 at 0.20 s plus shared return. Exact recorded pre-decision root/joint state, velocities, applied actions, motion tokens and references match; this is not a hidden-state snapshot. All four outcome combinations are allowed, even if this small panel does not exhibit every combination.

| Seed | Requested time (s) | Command executed | Passage | Peak beam force (N) | Contact duration (s) |
| --- | --- | --- | --- | --- | --- |
| 8041 | 0.20 | True | True | 0.000 | 0.000 |
| 8041 | 0.30 | True | True | 0.000 | 0.000 |
| 8041 | 0.40 | True | True | 0.000 | 0.000 |
| 8042 | 0.20 | True | False | 429.159 | 0.005 |
| 8042 | 0.30 | True | True | 0.000 | 0.000 |
| 8042 | 0.40 | True | False | 564.862 | 0.005 |

Seed 8042 passes at tested times [0.3].

New actual cost: 0.142543 contended GPU h. The six missing controls and six timing runs are distinct cells; repeated source/seed/interface checks do not add independent ancestors.

The reset-aware analysis-loader revision is explicit and hash-bound; physics manifest, thresholds, commands and predictions remain unchanged. Refusal is not stopping. No new policy-learning benefit, third skill or full collision certificate is claimed.

Validation: 187 impacted tests passed and one optional-dependency test skipped. All 292 unique registered artifact references matched their recorded hashes. The frozen failed v3 input-audit script retains one E501 prose-line exemption; other checked sources pass Black and Ruff. [Completion audit](evidence/action-completion-audit.json).

[Protocol](ACTION_LABEL_COMPLETION_V1.md) · [All rows](evidence/action-label-completion.json) · [Analysis revision](evidence/action-label-analysis-revision.json) · [Learning comparison plan](LEARNING_UTILITY_PLAN_V1.md)
