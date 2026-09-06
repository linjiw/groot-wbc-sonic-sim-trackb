# All 384 frozen outputs have a conditional local full-pose region

2026-09-06. Every previously accepted learned-pattern request in the fresh audit has
a positive six-dimensional local pose certificate under the **static 29-capsule
reference model** and assumed 1e-8 m numerical allowance. The audit covers all 384
requests, also 384 distinct within-case placements, across eight source groups,
two events and three fitting seeds. No nominal scene was moved, selected away or
regenerated. The 430xx pool remains excluded from fitting and method selection.

Each certificate covers a cube with translation components **±1.443374 mm** and
world rotation-vector components **±0.002389090 rad** about the nominal beam pose.
Orientation is `Exp(delta_rotvec) R_nominal`; this includes full three-dimensional
rotation, not only yaw. The cube has positive chart volume
**2.624305e-15 m³ rad³**. The exponential chart is nonsingular on this small domain,
so it establishes nonzero local volume in R³ × SO(3), conditional on the model.
This number is coordinate volume, not a numerical Haar-volume integral.

| Source group | Local conditional passes | Independent diagnostic probes passing |
| --- | ---: | ---: |
| 43001 | 48/48 | 3,456/3,456 |
| 43002 | 48/48 | 3,456/3,456 |
| 43003 | 48/48 | 3,456/3,456 |
| 43004 | 48/48 | 3,456/3,456 |
| 43005 | 48/48 | 3,456/3,456 |
| 43006 | 48/48 | 3,456/3,456 |
| 43007 | 48/48 | 3,456/3,456 |
| 43008 | 48/48 | 3,456/3,456 |

All domains share these conservative sizes because the new query caps positive
clearance at 20 mm and this cap binds the nominal slack for every output. Each has
at least **4.999997 mm** certified slack beyond both 10 mm requirements after the
box-displacement and numerical allowances. No maximal-volume claim follows.

The analytic displacement inequality covers every pose in each cube. The 64 corners
and eight deterministic interior probes per output—**27,648 checked poses** in
total—are implementation diagnostics, not the logical basis of continuity. Motion
frames remain discrete. The independently implemented rotation-vector matrix matches
SciPy in focused tests, and corner-displacement tests check all box vertices.

## Preserved failed reproduction predicate and its diagnosis

Registered P2 (positive local domains) and P3 (all probes pass) pass. **P1, direct
reproduction of the saved nominal scores, fails on all 384 requests**, with maximum
disagreement 39.805 mm. The new protocol incorrectly called 20 mm the existing
fresh-audit clearance cap; that original audit uses 100 mm. Comparing differently
capped positive distances is not a valid raw-score reproduction test.

A separate [cap audit](evidence/local-pose-cap-audit.json), preserving the original
protocol/result and failed P1, reruns the original oracle at every nominal pose:

- Original 100 mm oracle versus saved fresh-audit scores: maximum error **0 m**.
- New 20 mm scores versus `min(original_score, 0.020)`: maximum error **0 m**.
- Recomputed certificate parameters and saved probe counts: **384/384 agree**.

Capping above the 10 mm requirement is conservative and preserves the needed
1-Lipschitz clearance bound. Thus this implementation mismatch does not invalidate
the smaller local domains. It does invalidate an assertion that the original P1
passed; that assertion is not made. Neither nominal scenes nor domains were changed
in response to probes, and no original acceptance labels were rewritten.

## Cost, reproduction and remaining gap

The primary batch uses **56,064 whole-motion queries**, **11.956 s** wall time and
zero GPU time. The separate cap audit adds **768 queries**, for **56,832** total;
its bookkeeping/runtime is additional to the primary timing. The all-output table
and raw per-pose score/witness hashes are in the [portable certificate result](evidence/fresh-local-pose-certificate.json).
The [export receipt](assets/local_pose-progress-receipt.json) records original and
public hashes. Original artifacts live under
`/home/linjiw/research-data/groot-wbc/m2s-fresh-local-pose-certificate-v1`.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv_research/bin/python \
  scripts/research/motion2scene_fresh_local_pose_certificate.py --out /path/to/new-output
PYTHONPATH=. .venv_research/bin/python scripts/research/audit_motion2scene_local_pose_certificate.py \
  --result /path/to/new-output/result.json --out /path/to/new-output/cap_audit.json
PYTHONPATH=. .venv_research/bin/pytest -q \
  tests/dataset_generation/test_motion2scene_local_pose_certificate.py
```

Three focused tests pass; Black and Ruff checks pass. This is a local six-dimensional
certificate for every frozen output. **The entire original ±2 cm/yaw uncertainty
domain, native imported-body geometry on this 384-output batch, and motion between
frames are not certified.** The separate native temporal result concerns carrier
41002 and cannot be combined with this source batch to claim a joint guarantee.
