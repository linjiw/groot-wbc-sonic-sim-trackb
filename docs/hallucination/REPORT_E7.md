# E7 Source-Conditioned Archetype Transfer Report

The registered primary is **confirmed**: 5/6 variants are verified across 3/3 sources. The 6/6 stretch prediction is **falsified**.

| source | archetype | easy N/A | hard N/A | contact frame / drift | decision |
|---|---|---|---|---|---|
| `lfh_086_crouch` | `door_lintel` | accepted/accepted | rejected/accepted | 122 / 95 | **refused** |
| `lfh_086_crouch` | `ibeam` | accepted/accepted | rejected/accepted | 123 / 137 | **verified** |
| `lfh_089_crouch` | `door_lintel` | accepted/accepted | rejected/accepted | 103 / 121 | **verified** |
| `lfh_089_crouch` | `ibeam` | accepted/accepted | rejected/accepted | 109 / 117 | **verified** |
| `lfh_090_crouch` | `door_lintel` | accepted/accepted | rejected/accepted | 99 / 114 | **verified** |
| `lfh_090_crouch` | `ibeam` | accepted/accepted | rejected/accepted | 99 / 115 | **verified** |

All six nominal-hard contacts are uniquely attributed to `torso_link` on the authored binding primitive, and every intended-clear cell has zero external contact. Five contacts precede 0.15 m reference drift. Source 086's door-lintel contact occurs after drift (frame 122 versus 95), so that variant is refused even though its four outcome labels match. No door jamb, I-beam web, or top flange becomes a secondary cause.

Scene-conditioned engineering intervals span 26.07–39.07 mm. Source 086 preserves the E6c-verified 0.10 m exposure, so this transfer does not reintroduce the long-face instability.

Spend: **0.340 contended GPU-h** across 24 rollouts. The evidence reaches the registered transfer primary, but source 086 still has only two verified archetypes (shelf and I-beam). E5 remains blocked pending one causally clean replacement archetype for that source; a matching outcome pattern alone is insufficient.
