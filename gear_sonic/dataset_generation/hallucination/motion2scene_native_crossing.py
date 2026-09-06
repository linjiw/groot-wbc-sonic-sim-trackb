"""Downstream support of authored native outer capsules at recorded poses."""

import numpy as np

from gear_sonic.dataset_generation.swept_volume import body_capsules_world


def downstream_support(positions, quaternions, names, shapes, beam, allowance_m=0.00200001):
    if not np.isfinite(allowance_m) or allowance_m < 0:
        raise ValueError("invalid outward allowance")
    if set(shapes) - set(names):
        raise ValueError("unrecorded collision owners")
    a, b, radii, _ = body_capsules_world(positions, quaternions, names, capsules=shapes)
    normal = np.array([np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"]), 0.0])
    center = np.array([*beam["center_xy_m"], 0.0])
    values = np.minimum((a - center) @ normal, (b - center) @ normal) - radii - allowance_m
    if values.ndim != 2 or not np.isfinite(values).all() or not values.shape[1]:
        raise ValueError("invalid or empty support")
    return values.min(1)


def stable_finish(downstream, upright, required_m, fps=50):
    downstream = np.asarray(downstream)
    upright = np.asarray(upright, dtype=bool)
    if (
        downstream.ndim != 1
        or downstream.shape != upright.shape
        or not np.isfinite(downstream).all()
    ):
        raise ValueError("invalid support/stability series")
    good = (downstream >= required_m) & upright
    window = int(np.ceil(0.3 * fps)) + 1
    return next(
        (i + window for i in range(len(good) - window + 1) if good[i : i + window].all()), None
    )
