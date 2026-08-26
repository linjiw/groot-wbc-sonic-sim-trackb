# E7c Source-086 Replacement Report

The hanging-panel replacement is **verified**. It preserves source 086's motion pair, station, 0.10 m exposure, coordinates, and margin rule while replacing only the E7 door-lintel archetype.

| role | outcome | external contact |
|---|---|---|
| `nominal_easy` | accepted | none |
| `adapted_easy` | accepted | none |
| `nominal_hard` | rejected | binding_constraint |
| `adapted_hard` | accepted | none |

Nominal-hard contact is uniquely attributed to `torso_link` on `/World/ConstraintFrame/BindingHangingPanel` at frame 120, before drift at frame 135. The refused door-lintel contact occurred after drift (122 versus 95), so the replacement changes causal cleanliness rather than merely recovering outcome labels.

Spend: **0.057 contended GPU-h** across four rollouts. With the verified shelf and I-beam, this gives source 086 three clean archetypes. This satisfies the count-only gate, but not the stricter crossed E5 gate: a third archetype must still be verified across all four sources.
