# E6c Finite-Exposure Ablation Report

The adaptive source-086 family is **verified**. Reducing the route-aligned shelf face from 0.30 m to 0.10 m changed the adapted-easy outcome from rejected to accepted while holding the motion pair, operator, station, archetype, DCS target, and margin rule fixed.

| role | outcome | external contact |
|---|---|---|
| `nominal_easy` | accepted | none |
| `adapted_easy` | accepted | none |
| `nominal_hard` | rejected | binding_constraint |
| `adapted_hard` | accepted | none |

Nominal-hard contact occurs at frame 124 on `torso_link` and is uniquely attributed to `/World/ConstraintFrame/BindingShelfPlank`; reference drift begins at frame 133. The runner-up primitive is 3902.1 mm farther away.

Adapted-easy drift rate fell from 0.15097 m/s to 0.12820 m/s. This is one paired source/seed intervention: it establishes finite exposure as a required proposal variable here, not a population-level effect estimate.

Spend: **0.056 contended GPU-h** across four rollouts. E6 remains 2/3 by preregistration; E6c independently makes source 086 extractable and brings the available source count to four when combined with `cf_005_056`, 089, and 090.
