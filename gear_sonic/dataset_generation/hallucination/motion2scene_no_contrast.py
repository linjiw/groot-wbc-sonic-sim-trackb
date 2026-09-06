"""Target-only ablation: interference is absent from loss, search and rejection."""

import torch

from .motion2scene_station_search import station_search


def target_penalty(target_clearances, margin=0.01):
    if target_clearances.ndim != 2 or target_clearances.shape[1] == 0:
        raise ValueError("target-only (proposal, placement) clearances required")
    if not torch.isfinite(target_clearances).all() or margin <= 0:
        raise ValueError("invalid target clearances or margin")
    return ((margin - target_clearances.amin(1)).clamp(min=0) / margin).square()


def target_only_pattern(initial, query_target):
    """Reuse the 17-query budget, with no alternative supplied to search."""

    def query(scenes):
        target = query_target(scenes)
        if target.ndim != 2 or not torch.isfinite(target).all():
            raise ValueError("finite target-only clearance array required")
        # The shared search expects two channels. This constant sentinel always has
        # greater slack than any target entry, so selection depends only on target.
        sentinel = -torch.ones_like(target) * (target.detach().abs().max() + 1)
        return torch.stack((sentinel, target), dim=-1)

    output, trace = station_search(initial, query, "pattern")
    trace["neutral_channel_is_sentinel"] = True
    return output, trace


def target_only_accept(target_clearances, margin=0.01):
    # Rejection also omits the neutral witness; audit it separately after freezing.
    target_penalty(target_clearances, margin)
    return target_clearances.amin(1) >= margin
