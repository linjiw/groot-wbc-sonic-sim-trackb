"""Explicit finite-set clearance and interference penalties for inverse scenes.

These differentiable penalties express the audit margins. They are not hard
constraints or certificates for continuous placement, time, or imported geometry.
"""

import math

from .motion2scene_uncertainty import uncertainty_terms


def margin_terms(clearances, costs, target_mask, *, margin=0.01):
    """Preference and two normalized squared hinge penalties per proposal.

    Clearances have axes (proposal, placement, recording). The hardest target
    and easiest upright alternative across the supplied placements determine
    the two margins; every recording must satisfy its respective condition.
    The preference decoder retains its frozen 10 mm cost scale independently.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    selection, _, target, upright = uncertainty_terms(clearances, costs, target_mask)
    target_penalty = ((margin - target).clamp(min=0) / margin).square()
    upright_penalty = ((margin + upright).clamp(min=0) / margin).square()
    return selection, target_penalty, upright_penalty, target, upright
