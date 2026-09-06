# Bounded continuous-placement diagnostic

This is a separately registered follow-up to the margin-learning experiment, not part
of its raw-yield metric. The algorithm/protocol are written before that experiment's
results are available; the candidate manifest will pin the completed result before
certificate queries run. No trained source, objective or checkpoint is changed.

## Mathematical contract

For a box rotating about its own center, let R be its horizontal half-diagonal. A pose
cell with translation half widths hx/hy/hz and yaw half width hθ ≤ π has occupied-set
Hausdorff displacement from its center bounded by

```text
B = sqrt(hx² + hy² + hz²) + 2 R sin(hθ / 2).
```

This follows by mapping corresponding box points under translation and rotation,
bounding their displacement by the triangle inequality and the rotation chord length.
The distance from a fixed segment to the box is 1-Lipschitz in that occupied-set
Hausdorff distance. Subtracting a fixed capsule radius, capping distance, and taking
finite minima/maxima preserve this bound. The callback returns the minimum target
clearance and maximum upright clearance, after minima over all recorded capsules/frames
in each recording. Each target must clear; every upright recording needs an interference
witness, possibly at a different frame or body part.

At each cell center compute target/upright clearance. With numerical-error allowance e,
discharge the entire cell only when target − B − e ≥ 0.01 m and upright + B + e ≤ −0.01 m.
A point whose violation exceeds e is a counterexample. Otherwise split the cell along
the largest translation/rotation displacement contribution. Success requires every
cell to be discharged. Exhausting the query/precision/time budget returns unresolved.
Never promote a passing grid or partial partition to a continuous-placement certificate.

The **assumed** absolute numerical allowance is 1e-8 m, covering query/bound arithmetic;
it has not been formally established with interval arithmetic. Therefore success is
named `conditional_pass`, conditional on that allowance, the static capsule model,
and the Lipschitz contract. It is not an unconditional numerical proof, a certificate
for motion between recorded frames, an imported-body enclosure proof, or physical safety.

## Frozen demonstration scope

Take optimizer seed 8121 from `margin_no_kl` and `constraints_no_kl`, regardless of their
relative performance. In each arm choose the first of its 64 proposals maximizing the
minimum target/upright margin slack over all eight recordings and all 113 finite
placements. This deliberately favors the best observed geometric candidate, including
previously observed execution seed 7903; it cannot estimate distribution success or
unseen-carrier generalization. Do not replace an arm/seed if its candidate fails.

Attempt both candidates over the full box x/y ±2 cm, z ±1 cm, yaw ±0.02 rad. Limit each
to 4095 oracle calls and 120 seconds. Use the frozen independent NumPy primitive query
with every recorded frame and capsule, plus its conservative proposal-specific broad
phase. Save the complete subdivision trace and queried per-recording values. Report
conditional pass, counterexample and unresolved separately. No success prediction is
registered: this is a feasibility/cost diagnostic for the sufficient-bound algorithm.

Unit tests must include a 1-Lipschitz interior violation missed by endpoints/center,
budget exhaustion, numerical uncertainty at a point, partition coverage, interference
failure, and box-corner displacement under rotation/translation.

## Reproduce

```bash
.venv_research/bin/python scripts/research/motion2scene_certify_placements.py prepare \
  --result /home/linjiw/research-data/groot-wbc/m2s-margin-learning-v1/result.json \
  --protocol docs/motion2scene/PLACEMENT_CERTIFICATE_V1.md \
  --out /home/linjiw/research-data/groot-wbc/m2s-placement-certificate-v1
.venv_research/bin/python scripts/research/motion2scene_certify_placements.py run \
  --manifest /home/linjiw/research-data/groot-wbc/m2s-placement-certificate-v1/manifest.json
```
