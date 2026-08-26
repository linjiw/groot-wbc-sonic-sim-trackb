# LFH-E12 — Crouch Amplitude Ladder

Cohort of 12 motions across 10 body modes, drawn from the whole gated clip pool rather than one body mode. Registered before spend in `docs/prediction_register.md`.

| motion | mode | straight | nominal | d040 | d055 | d070 | delivered | raw window | eng. window |
|---|---|---:|---|---|---|---|---:|---:|---:|
| `ladder_148` | squat_pick | 0.998 | accepted | accepted | accepted | rejected | 55 mm | 54.5 mm | 18.4 mm |
| `ladder_065` | side_step | 0.995 | rejected | skipped | skipped | skipped | - | - | - |
| `ladder_079` | backward | 0.998 | rejected | skipped | skipped | skipped | - | - | - |
| `ladder_034` | walk_look | 0.956 | accepted | accepted | accepted | accepted | 70 mm | 51.3 mm | 15.2 mm |
| `ladder_008` | walk | 0.998 | rejected | skipped | skipped | skipped | - | - | - |
| `ladder_120` | turn_in_place | 0.963 | accepted | rejected | rejected | rejected | - | - | - |
| `ladder_138` | reach_walk | 0.997 | accepted | accepted | accepted | accepted | 70 mm | 70.2 mm | 34.1 mm |
| `ladder_126` | carry_walk | 0.995 | accepted | accepted | rejected | rejected | 40 mm | 43.7 mm | 7.6 mm |
| `ladder_091` | stand_to_walk | 0.997 | accepted | rejected | rejected | rejected | - | - | - |
| `ladder_058` | duck_under | 0.998 | rejected | skipped | skipped | - | - | - | - |
| `ladder_064` | side_step | 0.992 | accepted | rejected | rejected | rejected | - | - | - |
| `ladder_076` | backward | 0.998 | accepted | accepted | rejected | rejected | 40 mm | 31.5 mm | -4.6 mm |

## Registered predictions

- `P1_at_least_8_new_sources` — **falsified**. {"predicted": ">= 8 clips accept the nominal and at least one rung", "observed": 5}
- `P2_median_delivered_between_40_and_70` — **confirmed**. {"predicted": "median largest-accepted amplitude in [40, 70] mm", "observed_median_mm": 55.0}
- `P3_monotone_within_motion` — **confirmed**. {"predicted": "no motion accepts a deeper rung after rejecting a shallower one", "violations": []}
- `P4_at_least_6_clear_20mm_window` — **falsified**. {"predicted": ">= 6 accepted pairs clear a 20 mm engineering window", "observed": 1}
- `P5_lateral_coupling_predicts_rejection` — not adjudicated. {"predicted": "lateral coupling above 20 mm predicts rejection", "motions_with_coupling_above_20mm": [], "of_those_rejected_at_that_rung": []}
