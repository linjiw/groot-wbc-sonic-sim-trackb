"""Finite placement-uncertainty objectives for the frozen inverse beam model.

Offsets are world dx, dy, dz in metres and additive yaw in radians. These queries
cover recorded capsule poses, not the imported robot or continuous motion/poses.
"""

from dataclasses import replace

import torch

from ..capsule_box_torch import capsule_yaw_box_clearance
from .motion2scene_inverse import domain_capsules, inverse_terms, route_centers


def uncertainty_capsules(state, domain, max_vertical_offset):
    """Cull only outside the entire perturbed height domain and clearance cap."""
    if max_vertical_offset < 0:
        raise ValueError("vertical uncertainty must be nonnegative")
    expanded = replace(
        domain,
        height_low=domain.height_low - max_vertical_offset,
        height_high=domain.height_high + max_vertical_offset,
    )
    return domain_capsules(state, expanded)


def perturbed_clearances(scenes, offsets, clouds, route, progress, yaw, domain):
    """Return (proposals, offsets, recordings) capped geometric clearances.

    Caller must supply clouds culled for the full offset domain (or full clouds).
    A common offset set is applied to every proposal and recording.
    """
    if offsets.ndim != 2 or offsets.shape[-1] != 4 or len(offsets) == 0:
        raise ValueError("offsets must be nonempty (J,4)")
    if not torch.isfinite(offsets).all():
        raise ValueError("offsets must be finite")
    xy = route_centers(scenes[:, 0], route, progress)
    nominal = torch.cat((xy, (scenes[:, 1] + domain.thickness / 2)[:, None]), -1)
    centers = nominal[:, None, :] + offsets[None, :, :3]
    half = scenes.new_tensor([domain.depth, domain.width, domain.thickness]) / 2
    angles = (yaw + offsets[:, 3])[None, :, None]
    values = []
    for starts, ends, radii in clouds:
        if len(starts) == 0:
            values.append(centers[..., 0] * 0 + domain.clearance_cap)
        else:
            clearance = capsule_yaw_box_clearance(
                starts[None, None],
                ends[None, None],
                radii[None, None],
                centers[:, :, None],
                half,
                angles,
            )
            values.append(clearance.amin(-1).clamp(max=domain.clearance_cap))
    return torch.stack(values, -1)


def uncertainty_terms(clearances, costs, target_mask):
    """Worst target clearance and easiest alternative across supplied offsets.

    Independent extrema upper-bound the per-offset selection loss. Both classes
    must satisfy their geometric conditions at every supplied offset; averaging
    across offsets would allow failures to be hidden by successful placements.
    """
    worst = torch.where(target_mask[None], clearances.amin(1), clearances.amax(1))
    return inverse_terms(worst, costs, target_mask)
