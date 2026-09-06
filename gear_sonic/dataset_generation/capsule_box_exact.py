"""Analytic static capsule–AABB clearance for scene-proposal diagnostics.

For a segment p(t), squared distance to an AABB is piecewise quadratic. The only
breakpoints are crossings of the six box planes. Minimize each quadratic on its
interval rather than sampling the segment. Results are exact up to floating-point
roundoff for this primitive model, not a continuous-time or mesh collision verdict.
"""

from __future__ import annotations

import numpy as np


def segment_box_distance(starts, ends, lower, upper):
    """Return nonnegative segment-to-box distance with arbitrary leading batch axes."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if starts.ndim < 1 or starts.shape[-1] != 3 or ends.shape != starts.shape:
        raise ValueError("segment endpoints must have equal (..., 3) shapes")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
        raise ValueError("box bounds must be ordered 3-vectors")
    if not all(np.isfinite(value).all() for value in (starts, ends, lower, upper)):
        raise ValueError("geometry must be finite")
    delta = ends - starts
    crossings = []
    for bound in (lower, upper):
        # Parallel coordinates introduce no breakpoints; duplicate t=0 is harmless.
        time = np.divide(bound - starts, delta, out=np.zeros_like(delta), where=delta != 0)
        crossings.append(np.clip(time, 0, 1))
    zero = np.zeros(starts.shape[:-1] + (1,))
    knots = np.sort(np.concatenate([zero, zero + 1, *crossings], axis=-1), axis=-1)
    left, right = knots[..., :-1], knots[..., 1:]
    middle = (left + right) / 2
    points = starts[..., None, :] + middle[..., None] * delta[..., None, :]
    active = (points < lower) | (points > upper)
    closest = np.clip(points, lower, upper)
    # On each interval, only coordinates outside the box contribute to the quadratic.
    numerator = np.sum(
        np.where(active, delta[..., None, :] * (closest - starts[..., None, :]), 0), axis=-1
    )
    denominator = np.sum(np.where(active, delta[..., None, :] ** 2, 0), axis=-1)
    optimum = np.divide(numerator, denominator, out=middle.copy(), where=denominator > 0)
    optimum = np.clip(optimum, left, right)
    nearest = starts[..., None, :] + optimum[..., None] * delta[..., None, :]
    residual = nearest - np.clip(nearest, lower, upper)
    return np.sqrt(np.min(np.sum(residual**2, axis=-1), axis=-1))


def capsule_box_clearance(starts, ends, radii, lower, upper):
    """Axis distance minus radius; negative certifies overlap, not penetration depth.

    Radii broadcast against the leading endpoint axes. Positive values are exact
    primitive separation. Negative values saturate at -radius for axes inside the box.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("radii must be finite and nonnegative")
    distance = segment_box_distance(starts, ends, lower, upper)
    if np.broadcast_shapes(distance.shape, radii.shape) != distance.shape:
        raise ValueError("radii must broadcast to the segment batch shape")
    return distance - radii
