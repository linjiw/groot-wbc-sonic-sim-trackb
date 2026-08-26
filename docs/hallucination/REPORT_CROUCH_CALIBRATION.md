# Crouch Executed-Window Calibration Report

The CAL3 cohort prediction is **confirmed**. 3/4 matched pairs produced accepted nominal and adapted executions, and the largest temporally active exact head/torso window was 72.26 mm.

| pair | nominal | adapted | max eligible raw window | finite-face trials | target matches |
|---|---|---|---:|---:|---:|
| `lfh_090_crouch` | accepted | accepted | 62.10 mm | 12 | 8 |
| `lfh_086_crouch` | accepted | accepted | 72.26 mm | 12 | 5 |
| `lfh_089_crouch` | accepted | accepted | 69.62 mm | 12 | 7 |
| `lfh_095_crouch` | accepted | rejected |  | 0 | 0 |

Funnel: **15 motions -> 60 CPU settings -> 4 calibrated pairs -> 8/8 rollouts -> 3 accepted pairs -> 36 finite-face trials -> 20 target-matched scene candidates**. Actual serial spend: **0.063 contended GPU-h**.

The SweepCF-DCS v2 audit reopens the `head_torso` × `overhead` × `1.2_1.3` × `clear_25_50` target. V1 had incorrectly marked it occupied from calibration `probe` episodes rather than canonical cells of an exact verified variant. The same immutable CAL3 executions therefore yield CPU-valid proposal support; they do not yet establish a verified family.

The 18.044 mm value used here is a conservative empirical repeatability envelope from E1a, not a calibrated confidence bound. E6 applies it symmetrically to the nominal-strike and adapted-clear sides before selecting any physics cell.
