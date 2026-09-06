# Policy-data readiness v1: complete labels are still missing

The audited 42-cell panel produces **10/16 complete scene–seed skill labels**. Nine pairs have neutral failure and d040 success; the earlier-beam pair in seed 8042 has both skills fail. The absent, raised and blocked cases in both seeds lack oracle d040 executions. Missing comparators are unknown, never inferred successes or failures.

The feature matrix is **16 × 144**, from the neutral/no-prior-switch sensor packet at 0.2 s: twelve upper rays plus three conditional lower rays each, encoded as range, hit height and query/hit masks. Object names, scene identity, generator arm, outcome labels and active skill are excluded from the feature array. Source/condition metadata remains separate for audit. The feature contract assumes the current world-height convention and does not establish coordinate-invariant transfer.

All rows are **interface development only**, from ancestor 41002. Training is disabled; zero learning runs occurred. A complete two-skill label refers to tested skills, not all physically possible actions.

| Seed | Condition | Neutral | d040 | Label |
| --- | --- | --- | --- | --- |
| 8041 | nominal | False | True | measured |
| 8041 | height_minus10mm | False | True | measured |
| 8041 | height_plus10mm | False | True | measured |
| 8041 | station_minus15cm | False | True | measured |
| 8041 | station_plus15cm | False | True | measured |
| 8041 | absent | see measured neutral trace | unknown | missing_comparator |
| 8041 | raised | see measured neutral trace | unknown | missing_comparator |
| 8041 | blocked | see measured neutral trace | unknown | missing_comparator |
| 8042 | nominal | False | True | measured |
| 8042 | height_minus10mm | False | True | measured |
| 8042 | height_plus10mm | False | True | measured |
| 8042 | station_minus15cm | False | False | neither_tested_skill_passes |
| 8042 | station_plus15cm | False | True | measured |
| 8042 | absent | see measured neutral trace | unknown | missing_comparator |
| 8042 | raised | see measured neutral trace | unknown | missing_comparator |
| 8042 | blocked | see measured neutral trace | unknown | missing_comparator |

Next register six oracle controls (absent/raised/blocked × 8041/8042), retain the current failed pose, and independently audit the new panel’s outer-envelope crossing horizon. Then freeze a source-disjoint acquisition/split plan and matched policy architecture, label budget and optimizer steps for Motion2Scene, uniform, analytic/grid and explicitly specified unconstrained same-family generation. This prototype dataset cannot test learning benefit.

Validation: feature identity/label invariance, missing-vs-infeasible labels and malformed packet rejection are covered by focused tests. The full impacted set passes 72 tests.

[Variation result](OVERHANG_VARIATION_V1_RESULT.md) · [Readiness evidence](evidence/policy-readiness.json) · [Feature array](evidence/policy-decision-features.npz) · [Registration](evidence/policy-readiness-registration.json) · [Receipt](evidence/policy-readiness-receipt.json)
