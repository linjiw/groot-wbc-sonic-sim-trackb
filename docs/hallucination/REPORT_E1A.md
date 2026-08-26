# E1a Repeatability Report

Outcome stability: **16/16 matches; 0 flips**. The LFH stop line did not trigger.

The executed-capsule/USDA-AABB instrument gives a source-inclusive clearance noise floor of **18.044 mm**. The spread between the two fresh repeats alone is **11.348 mm**. The source-inclusive value is the conservative tolerance used for later drift reports.

| family / cell | shipped source | repeat 1 | repeat 2 | fresh spread | inclusive spread |
|---|---:|---:|---:|---:|---:|
| `duck_003/nominal_easy` | 89.675 | 85.904 | 88.820 | 2.915 | 3.770 |
| `duck_003/nominal_hard` | -0.049 | -0.112 | -0.121 | 0.009 | 0.072 |
| `duck_003/adapted_easy` | 193.427 | 177.634 | 178.387 | 0.753 | 15.794 |
| `duck_003/adapted_hard` | 55.617 | 42.959 | 37.574 | 5.385 | 18.044 |
| `mf_005_c08/nominal_easy` | 56.181 | 53.189 | 53.544 | 0.355 | 2.993 |
| `mf_005_c08/nominal_hard` | -0.035 | -0.160 | -0.056 | 0.104 | 0.125 |
| `mf_005_c08/adapted_easy` | 101.726 | 105.519 | 116.867 | 11.348 | 15.141 |
| `mf_005_c08/adapted_hard` | 10.827 | 15.262 | 24.006 | 8.744 | 13.179 |

Instrument: minimum executed G1 collision-capsule clearance to the exact `LowShelf_00` AABB read from the USDA loaded by physics. Negative values overlap. This is distinct from KCS reference-side capsule prediction and historical boundary bisection; millimetre values are diagnostics, while physics outcomes remain binding.

Actual serial spend: **0.158 contended GPU-h**.
