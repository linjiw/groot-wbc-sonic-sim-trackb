# LFH Critical-Set CPU Continuation

**Status:** CPU gate green after the Phase-2 review; physics remains unauthorized.

## Review response and corrected finding

`REVIEW_PHASE2.md` accepted the original CPU machinery and explicitly withheld
physics authorization. Implementing the LFH finite-face reach check then exposed
a missing preflight: enlarging the historical plank from 0.5 m to 1.0811 m along
the route extended the constraint beyond the local crouch. The generated hard
cell changed adapted capsule clearance from **+66.663 mm** to **−38.914 mm**.
That would spend GPU on a geometry pair already known not to implement the target.

D2-005 therefore supersedes D2-003 only along the route. The corrected golden
keeps the source's 0.5 m temporal footprint, retains 3.0 m across-route coverage,
requires both routes to cross the face, and checks all four capsule signs:

| Cell | Original | Edited | Target matched |
|---|---:|---:|---:|
| easy | +89.577 mm | +202.302 mm | yes |
| hard | −68.000 mm | +66.663 mm | yes |

## LFH method increment

- `keypoints.py` losslessly partitions all 29 capsules on 14 owners into the 10
  accepted semantic groups and caches immutable rollout/hash extractions.
- `reach.py` computes analytic capsule support over finite overhead faces and
  one-sided, height-banded lateral reach at 2 cm route stations.
- `window.py` emits the certified interval, binding group, critical frames, and
  all-group margins. Duck reach is 1.301048 / 1.124332 m, a 176.716 mm window;
  it agrees with the stored 178.125 mm bisection result inside its 5 mm tolerance.
- `propose.py` intersects DCS bins with `W_cert`, solves minimum alpha, enforces
  operator/anatomy coupling, and screens every semantic group. Output is tier 1
  only; capsule keep-out and physics remain mandatory.
- `delivery.py` implements the exact response-table schema and deterministic
  monotone fit with uncertainty. D-phi v0 is honestly refused: R1 contains zero
  eligible commanded/executed operator pairs. The V2 probe proposal starts the
  table only after accepted paired executions.

## Gate disposition

The original E1b proposal is invalidated by stale scene hashes and D2-005.
`E1B_DUCK003_PHYSICS_PROPOSED_V2.json` and
`MINIMAL_PLANE_PROBES_PROPOSED_V2.json` are new, hash-pinned, explicitly
`not_authorized` proposals. E1a is unchanged. No register, preregistration,
paper, frozen split, or GPU artifact was touched. Phase 3 remains blocked until
E1a and corrected E1b physics pass and the user reviews the V2 manifests.

## Reproduction

```bash
./.venv_sim/bin/python scripts/research/hallucination/run_e1_cpu.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv_sim/bin/python -m pytest \
  tests/dataset_generation/test_hallucination_{keypoint_window,delivery,propose}.py \
  tests/dataset_generation/test_hallucination_{spec_and_stage,archetypes,keepout}.py -q
```

Result: **36 passed** across the six files; the five-file KCS-focused subset is
26 passed.
