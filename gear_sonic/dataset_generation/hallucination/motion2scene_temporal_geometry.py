"""Conditional clearance bounds for linear/shortest-SLERP recorded rigid-body poses.

These bounds describe the declared interpolant, not unrecorded simulator dynamics.
Floating-point error is an explicit assumed allowance, not a validated interval proof.
"""

import numpy as np


def normalized_poses(positions, quaternions):
    p = np.asarray(positions, dtype=np.float64)
    q = np.asarray(quaternions, dtype=np.float64)
    if (
        p.ndim != 3
        or p.shape[-1] != 3
        or len(p) < 2
        or p.shape[1] == 0
        or q.shape != (*p.shape[:2], 4)
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("requires finite synchronized (T,B,3)/(T,B,4) poses, T>=2")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
        raise ValueError("body quaternions must be approximately unit wxyz")
    return p, q / norms


def interpolate_poses(positions, quaternions, subdivisions):
    """Keep every endpoint and subdivide each interval uniformly; quaternion order wxyz."""
    p, q = normalized_poses(positions, quaternions)
    if not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    a, b = q[:-1], q[1:]
    dot = (a * b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    t = (np.arange(subdivisions) / subdivisions)[None, :, None, None]
    angle = angle[:, None]
    # sinc avoids the zero-angle singularity without a near-angle approximation.
    denominator = np.sinc(angle / np.pi)
    left = (1 - t) * np.sinc((1 - t) * angle / np.pi) / denominator
    right = t * np.sinc(t * angle / np.pi) / denominator
    qq = left * a[:, None] + right * b[:, None]
    pp = (1 - t) * p[:-1, None] + t * p[1:, None]
    return (
        np.concatenate([pp.reshape(-1, p.shape[1], 3), p[-1:]]),
        np.concatenate([qq.reshape(-1, q.shape[1], 4), q[-1:]]),
    )


def interval_displacement(positions, quaternions, body_names, capsules):
    """Bound capsule-axis displacement to the temporally nearest interval endpoint.

    Translation moves by at most ||p1-p0||/2. A body-local segment with endpoint norms
    at most R rotates by at most half the geodesic angle theta, giving 2R sin(theta/4).
    Its constant-radius Minkowski sphere does not enlarge this displacement bound.
    Shape order matches body_capsules_world. Each returned entry covers a whole interval.
    """
    p, q = normalized_poses(positions, quaternions)
    names = list(body_names)
    if len(names) != p.shape[1] or len(set(names)) != len(names):
        raise ValueError("requires unique names matching the body axis")
    translation = np.linalg.norm(np.diff(p, axis=0), axis=-1) / 2
    theta = 2 * np.arccos(np.clip(np.abs((q[:-1] * q[1:]).sum(-1)), 0, 1))
    bounds = []
    for owner, shapes in capsules.items():
        if owner not in names:
            raise ValueError(f"unrecorded collision owner: {owner}")
        index = names.index(owner)
        for shape in shapes:
            vertices = np.asarray([shape.start, shape.end])
            if not np.isfinite(vertices).all() or not np.isfinite(shape.radius) or shape.radius < 0:
                raise ValueError("invalid capsule")
            radius = np.linalg.norm(vertices, axis=-1).max()
            bounds.append(translation[:, index] + 2 * radius * np.sin(theta[:, index] / 4))
    if not bounds:
        raise ValueError("empty geometry")
    return np.stack(bounds, axis=1)


def clearance_lower_bounds(clearances, displacement, numerical_allowance_m=1e-8):
    """Bound clearance on every interval; negative bounds mean unresolved, not collision."""
    c, d = np.asarray(clearances), np.asarray(displacement)
    if (
        c.ndim != 2
        or len(c) < 2
        or d.shape != (len(c) - 1, c.shape[1])
        or not np.isfinite(c).all()
        or not np.isfinite(d).all()
        or (d < 0).any()
        or not np.isfinite(numerical_allowance_m)
        or numerical_allowance_m < 0
    ):
        raise ValueError("invalid clearance/bound arrays or numerical allowance")
    return np.minimum(c[:-1], c[1:]) - d - numerical_allowance_m
