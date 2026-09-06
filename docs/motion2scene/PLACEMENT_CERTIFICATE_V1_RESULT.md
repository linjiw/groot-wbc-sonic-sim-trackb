# Conditional certificates over continuous beam placement

Both fixed representatives pass the bounded placement checker over the **entire**
translation/yaw domain, conditional on the static capsule model and the stated numerical
error allowance. This extends the earlier finite-placement checks for these two selected
scenes. It does not cover the imported robot geometry or motion between recorded frames.

## Measured outcome

The [registered diagnostic](PLACEMENT_CERTIFICATE_V1.md) fixes optimizer seed 8121 and
two arms before checking. Each arm contributes the proposal with greatest observed
113-placement margin slack among its 64 draws. Selection uses all eight recordings,
including the previously observed 7903 execution. These deliberately favorable examples
cannot estimate distribution yield or independent-carrier generalization.

| Arm | Proposal index | Station | Beam underside (m) | Finite-sample slack (mm) | Queries | Discharged cells | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| margin_no_kl | 15 | 0.60201155 | 1.27041490 | 3.1491 | 1419 | 710 | conditional_pass |
| constraints_no_kl | 59 | 0.60173526 | 1.27089306 | 3.1758 | 1623 | 812 | conditional_pass |

Neither run exhausts its 4095-query or 120-second budget. Runtime is 3.16 and 3.64
seconds respectively. Both leave zero unresolved cells. The smallest conservative
margin-slack lower bounds across discharged cells are 1.196 and 2.941 micrometres.
These are slack beyond the required 10 mm after subtracting cell-motion and numerical
allowances, not estimates of the actual worst clearance.

The domain is world x/y ±2 cm, z ±1 cm and yaw ±0.02 rad. At every pose in it, the
conditional result covers target clearance ≥10 mm and upright overlap proxy ≥10 mm
for the eight recorded reference/execution sequences. Complete capsule axes and every
recorded frame are considered. Different upright recordings or pose cells may use
different interference witnesses.

## Why this covers more than a grid

A fixed segment's distance to a box changes by at most the occupied-set displacement
of that box. For each placement cell the algorithm bounds that displacement using its
translation half-diagonal and the rotation chord bound. It subtracts this bound and
an assumed numerical allowance from observed margin slack. A successful leaf therefore
covers its entire cell, not just its center. Recursively discharged leaves partition
the original uncertainty domain.

The [separate trace audit](evidence/placement-certificate-audit.json) verifies every
split has two complementary children, each non-root domain has exactly one parent,
all terminal domains satisfy both inequalities, and leaf volumes cover the full root
domain to numerical tolerance. It also cross-checks every queried offset and all eight
recordings against PyTorch: **3042 poses × 8 recordings = 24,336 comparisons**, maximum
absolute disagreement **1.388 × 10⁻¹⁷ m**. This is a post-run numerical/topology audit,
not an additional registered yield experiment or a proof of floating-point error bounds.

## Conditions that remain unresolved

The 1e-8 m absolute numerical allowance is an assumption covering query and bound
arithmetic, supported by these comparisons but not established using interval arithmetic.
Accordingly, the result is named `conditional_pass`, not an unconditional numerical proof.
The capsule model is still not a certified outer/inner representation of every imported
collider. The certificate also lacks between-frame motion, tracking beyond the recorded
runs, supporting beam geometry, and obstacle-present physics. Capsule overlap is not
physical penetration depth. All scene-admission and execution-eligibility flags remain false.

This makes a useful next research direction concrete: learn efficient proposals, then
certify or reject their geometric relationships under an explicit uncertainty contract.
For execution claims, establish numerical and imported-geometry bounds and the temporal
contract before admitting scenes. For learned-model claims, test independent carriers
and include rejection/certification cost in throughput and coverage comparisons.

## Reproduce and inspect

The [candidate manifest](evidence/placement-certificate-manifest.json) fixes source hashes,
selection, bounds and budgets. [Portable results](evidence/placement-certificate.json)
include each certificate summary and its full-trace hash. Full subdivision traces and
all queried per-recording clearances are retained at
`/home/linjiw/research-data/groot-wbc/m2s-placement-certificate-v1/`.

Prepare/run commands are in the [protocol](PLACEMENT_CERTIFICATE_V1.md). Audit with:

```bash
.venv_research/bin/python scripts/research/motion2scene_audit_certificate.py \
  --result /home/linjiw/research-data/groot-wbc/m2s-placement-certificate-v1/result.json \
  --out /path/to/new-certificate-audit.json
```

Seven focused tests cover box-corner displacement, complete partition coverage, an
interior violation missed by endpoints/center, budget exhaustion, point-level numerical
uncertainty, an upright-interference failure, and invalid/nonfinite inputs. The complete
impacted geometry/research suite passes 61 tests.
