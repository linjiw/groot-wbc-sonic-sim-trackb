# E6 Independent-Source Critical-Window Report

The registered primary prediction is **confirmed**: 2/3 CAL3 sources reproduced a complete canonical 2x2 pattern. The 3/3 stretch prediction is falsified.

| source | easy nominal/edit | hard nominal/edit | hard contact | decision |
|---|---|---|---|---|
| `lfh_086_crouch` | accepted/rejected | not run/not run |  | refused: adapted_easy_outcome_mismatch; hard_cells_not_authorized_after_easy_failure |
| `lfh_089_crouch` | accepted/accepted | rejected/accepted | binding_constraint | verified |
| `lfh_090_crouch` | accepted/accepted | rejected/accepted | binding_constraint | verified |

Sources 089 and 090 both pass outcome, unique binding attribution, contact-before-drift, and no-secondary-contact gates. Source 086 is refused before hard physics: its adapted-easy trajectory crossed the reference-drift threshold without external contact. This is direct evidence that route phase and finite exposure belong in the proposal distribution; empty-room clearance alone is insufficient.

Spend: **0.148 contended GPU-h** across 10 rollouts. The two verified families remain isolated from claim 5. They add independent source evidence in one DCS cell; they do not yet establish a learned distribution or the four-source/three-archetype E5 readiness gate.
