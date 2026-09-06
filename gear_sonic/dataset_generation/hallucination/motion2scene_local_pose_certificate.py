"""Conditional nonzero local pose domains for a fixed finite capsule/box query."""

import numpy as np


def local_pose_certificate(target, alternative, half_extents, *, margin=0.01, error=1e-8):
    """Certify a translation × rotation-vector cube about a nominal box pose.

    Orientation is Exp(delta_rotvec) @ R_nominal, not additive Euler angles. A box
    point moves by at most ||delta_translation|| + 2 R sin(||delta_rotvec||/2).
    Here R bounds all box corners about its center. Clearance aggregates are
    1-Lipschitz in occupied-box displacement. This is conditional on the geometry
    and the supplied numerical allowance, and says nothing about unsampled time.
    """
    h = np.asarray(half_extents, dtype=np.float64)
    if (
        h.shape != (3,)
        or not np.isfinite(h).all()
        or (h <= 0).any()
        or not np.isfinite([target, alternative, margin, error]).all()
        or margin <= 0
        or error < 0
    ):
        raise ValueError("requires finite scores, positive box/margin and nonnegative error")
    slack = float(min(target - margin, -margin - alternative) - error)
    base = {"nominal_margin_slack_after_error_m": slack}
    if slack <= 0:
        return {**base, "status": "unresolved", "chart_volume_m3_rad3": 0.0}
    radius = float(np.linalg.norm(h))
    translation = min(0.02, slack / (4 * np.sqrt(3)))
    rotation = min(0.02, slack / (4 * np.sqrt(3) * radius))
    angle = np.sqrt(3) * rotation
    displacement = np.sqrt(3) * translation + 2 * radius * np.sin(angle / 2)
    lower = float(slack - displacement)
    volume = float((2 * translation) ** 3 * (2 * rotation) ** 3)
    if not 0 < angle < np.pi or lower <= 0 or volume <= 0:
        return {**base, "status": "unresolved", "chart_volume_m3_rad3": 0.0}
    return {
        **base,
        "status": "conditional_pass",
        "translation_halfwidth_m": float(translation),
        "rotvec_component_halfwidth_rad": float(rotation),
        "maximum_rotation_angle_rad": float(angle),
        "box_corner_radius_m": radius,
        "box_displacement_bound_m": float(displacement),
        "margin_slack_lower_bound_m": lower,
        "chart_volume_m3_rad3": volume,
        "dimension": 6,
    }


def rotvec_matrix(vector):
    """SO(3) exponential for a world-frame rotation vector using stable Rodrigues."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("requires a finite three-dimensional rotation vector")
    x, y, z = v
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.linalg.norm(v)
    return (
        np.eye(3)
        + np.sinc(angle / np.pi) * skew
        + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (skew @ skew)
    )
