# E1b Golden Physics Report

Golden physics verdict: **PASS**. The generated shelf reproduced **4/4** outcomes, and the hard/nominal rejection was uniquely attributed to the binding primitive.

| cell | outcome | binding clearance | source drift | within E1a floor | contact |
|---|---|---:|---:|:---:|---|
| `nominal_easy` | accepted | 89.253 mm | 0.421 mm | yes | none |
| `nominal_hard` | rejected | -0.107 mm | 0.058 mm | yes | binding_constraint at f107 |
| `adapted_easy` | accepted | 216.788 mm | 23.361 mm | **no** | none |
| `adapted_hard` | accepted | 80.286 mm | 24.668 mm | **no** | none |

The conservative E1a floor is **18.044 mm**. Two adapted cells exceed it by at most **6.625 mm**. Per `docs/lfh/design-plan.md`, this drift is reported rather than used to replace the binding physics gates; it is an empirical warning for E2's context-invariance premise.

KCS reference-side predictions versus executed clearances (easy orig/edit, hard orig/edit) are `[89.577, 202.302, -68.0, 66.663]` versus `[89.253, 216.788, -0.107, 80.286]` mm. The three non-penetrating cells differ by at most 14.486 mm, inside the E1a floor. Hard/orig is the preregistered instrument-mismatch case: the KCS reference proxy predicts -68 mm while the executed capsule instrument saturates near first contact.

Contact attribution uses the first >1 N lateral-or-downward external force frame. The nearest authored cube is accepted only when it is unique by more than 0.01 mm; room-shell or other context geometry would be `secondary_contact`. The hard/nominal contact preceded 0.15 m reference drift (107 < 123) and was carried by torso geometry.

Actual serial spend: **0.039 contended GPU-h**.
