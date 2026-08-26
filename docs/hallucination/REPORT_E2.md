# E2 Variant-Transfer Report

Primary result: **2/3 variants verified** against the preregistered ≥2 threshold. All 12/12 physics outcomes matched.

| archetype | pattern | hard binding contact | secondary contact | max source drift | face offset | verdict |
|---|:---:|---|:---:|---:|---:|---|
| `door_lintel` | 4/4 | unique | no | 7.188 mm | 0.000000 mm | **verified** |
| `hvac_duct` | 4/4 | unique, after drift (f110 >= f30) | no | 7.786 mm | 0.000000 mm | **refused** |
| `ibeam` | 4/4 | unique | no | 15.507 mm | 0.000000 mm | **verified** |

All three hard/nominal collisions were carried by `torso_link` and uniquely nearest the authored binding primitive by far more than the 0.01 mm attribution resolution. Door-lintel and I-beam contact preceded 0.15 m reference drift. HVAC is honestly refused because drift began at frame 30 before binding contact at frame 110. No rollout produced secondary contact; in particular, door-lintel jambs remained clear.

Funnel: **12 proposed -> 12 CPU/preflight-passing -> 12 rolled out -> 12 scored -> 2/3 variants verified**. Authored binding-face offsets span 0.000000-0.000000 mm.

Clearance drift is diagnostic: **12/12 cells** lie within the 18.044 mm E1a floor. The outcome pattern and contact identity are invariant inside that broad empirical envelope; E2 therefore supports the causal pattern but does not validate a universal millimetre-level context-invariance claim.

Actual serial spend: **0.112 contended GPU-h**.

The 12 episode rows and three variant rows are staged under `e2_index/` using the repository's additive LFH provenance contract. They remain separate from the Phase-1 baseline until the planned E4 before/after re-render.

![E2 ego-view contact sheet](e2_contact_sheet.png)
