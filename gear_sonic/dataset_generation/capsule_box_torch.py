"""Differentiable static capsule/box separation for inverse scene learning.

Minimize the piecewise-quadratic axis-to-box distance, without axis samples.
The result is exact for the primitives up to floating point. Gradients are
piecewise derivatives, not a continuous-time or imported-robot certificate.
Keep capsule_box_exact.py as an independent NumPy evaluation implementation.
"""

from __future__ import annotations

import torch


def segment_box_distance(starts, ends, lower, upper):
    """Batched segment/AABB distance; (..., 3) tensors broadcast over leading axes.

    Zero-distance and parallel cases have finite subgradients. At geometric
    feature switches the derivative is not generally unique.
    """
    tensors = (starts, ends, lower, upper)
    if any(x.ndim < 1 or x.shape[-1] != 3 or not x.is_floating_point() for x in tensors):
        raise ValueError("geometry must be floating (..., 3) tensors")
    if any(x.dtype != starts.dtype or x.device != starts.device for x in tensors):
        raise ValueError("geometry must share dtype and device")
    if not all(torch.isfinite(x).all() for x in tensors) or torch.any(lower > upper):
        raise ValueError("geometry must be finite with ordered box bounds")
    starts, ends, lower, upper = torch.broadcast_tensors(*tensors)
    delta = ends - starts
    parallel = delta == 0
    denominator = torch.where(parallel, torch.ones_like(delta), delta)
    crossings = [
        torch.where(parallel, torch.zeros_like(delta), (bound - starts) / denominator).clamp(0, 1)
        for bound in (lower, upper)
    ]
    zero = torch.zeros_like(starts[..., :1])
    knots = torch.cat((zero, zero + 1, *crossings), dim=-1).sort(dim=-1).values
    left, right = knots[..., :-1], knots[..., 1:]
    middle = (left + right) / 2
    points = starts[..., None, :] + middle[..., None] * delta[..., None, :]
    low, high = lower[..., None, :], upper[..., None, :]
    active = (points < low) | (points > high)
    closest = torch.maximum(low, torch.minimum(points, high))
    numerator = torch.where(
        active, delta[..., None, :] * (closest - starts[..., None, :]), 0.0
    ).sum(dim=-1)
    quadratic = torch.where(active, delta[..., None, :].square(), 0.0).sum(dim=-1)
    optimum = torch.where(
        quadratic > 0,
        numerator / torch.where(quadratic > 0, quadratic, torch.ones_like(quadratic)),
        middle,
    )
    optimum = torch.maximum(left, torch.minimum(optimum, right))
    nearest = starts[..., None, :] + optimum[..., None] * delta[..., None, :]
    residual = nearest - torch.maximum(low, torch.minimum(nearest, high))
    # vector_norm defines a finite zero subgradient; sqrt(sum(x**2)) does not.
    return torch.linalg.vector_norm(residual, dim=-1).amin(dim=-1)


def capsule_box_clearance(starts, ends, radii, lower, upper):
    """Positive separation, negative primitive overlap (not penetration depth)."""
    if not torch.isfinite(radii).all() or torch.any(radii < 0):
        raise ValueError("radii must be finite and nonnegative")
    distance = segment_box_distance(starts, ends, lower, upper)
    if torch.broadcast_shapes(distance.shape, radii.shape) != distance.shape:
        raise ValueError("radii must broadcast to the segment batch")
    return distance - radii


def capsule_yaw_box_clearance(starts, ends, radii, center, half_size, yaw):
    """Static capsules against a yaw-rotated box, preserving tensor broadcast axes."""
    if not torch.isfinite(half_size).all() or torch.any(half_size < 0):
        raise ValueError("half sizes must be finite and nonnegative")
    cosine, sine = torch.cos(yaw), torch.sin(yaw)

    def local(points):
        offset = points - center
        return torch.stack(
            (
                cosine * offset[..., 0] + sine * offset[..., 1],
                -sine * offset[..., 0] + cosine * offset[..., 1],
                offset[..., 2],
            ),
            dim=-1,
        )

    return capsule_box_clearance(local(starts), local(ends), radii, -half_size, half_size)
