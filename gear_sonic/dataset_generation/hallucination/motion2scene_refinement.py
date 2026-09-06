"""Bounded proposal refinement with a separate geometric query callback.

Scores cover only callback-supplied placements and sampled body geometry. Independent
verification remains necessary; an incumbent can regress on unqueried placements.
"""

import torch


def margin_slack(values, margin=0.01):
    """Worst slack of target clearance and neutral interference; last axis is N,T."""
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[1] == 0:
        raise ValueError("expected nonempty (proposals, placements, neutral/target)")
    return torch.minimum(values[:, :, 1].amin(1) - margin, -values[:, :, 0].amax(1) - margin)


def bounded_refine(initial, query, *, steps=16, learning_rate=0.02):
    """Seventeen fixed evaluations, retaining each proposal's best observed incumbent.

    Adam operates in global-domain normalized coordinates. Motion from the initial
    proposal is bounded by 0.05 route fraction and 0.03 m height. Ties keep the earlier
    incumbent. Every candidate is retained in the trace, including failed proposals.
    """
    if initial.ndim != 2 or initial.shape[-1] != 2 or len(initial) == 0:
        raise ValueError("expected nonempty (proposals, 2)")
    low, span = initial.new_tensor([0.1, 1.1]), initial.new_tensor([0.8, 0.35])
    if (
        not torch.isfinite(initial).all()
        or torch.any(initial < low)
        or torch.any(initial > low + span)
    ):
        raise ValueError("initial proposal outside beam domain")
    if steps < 0 or learning_rate <= 0:
        raise ValueError("invalid optimization budget")
    radius = initial.new_tensor([0.05, 0.03])
    lower = torch.maximum(initial - radius, low)
    upper = torch.minimum(initial + radius, low + span)
    position = torch.nn.Parameter((initial.detach().clone() - low) / span)
    optimizer = torch.optim.Adam([position], lr=learning_rate)
    scenes_trace, values_trace = [], []
    best_score = initial.new_full((len(initial),), -torch.inf)
    best_index = torch.zeros(len(initial), dtype=torch.long)
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        scenes = low + span * position
        values = query(scenes)
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite refinement query")
        score = margin_slack(values)
        improved = score.detach() > best_score
        best_score = torch.maximum(best_score, score.detach())
        best_index[improved] = step
        scenes_trace.append(scenes.detach().clone())
        values_trace.append(values.detach().clone())
        if step == steps:
            break
        target = ((0.01 - values[:, :, 1].amin(1)).clamp(min=0) / 0.01).square()
        neutral = ((0.01 + values[:, :, 0].amax(1)).clamp(min=0) / 0.01).square()
        (target + neutral).sum().backward()
        if position.grad is None or not torch.isfinite(position.grad).all():
            raise FloatingPointError("nonfinite or absent refinement gradient")
        torch.nn.utils.clip_grad_norm_([position], 10)
        optimizer.step()
        with torch.no_grad():
            position.copy_(
                torch.maximum(torch.minimum(position, (upper - low) / span), (lower - low) / span)
            )
    scenes_trace = torch.stack(scenes_trace)
    return scenes_trace[best_index, torch.arange(len(initial))], {
        "scenes": scenes_trace,
        "clearances": torch.stack(values_trace),
        "selected": best_index,
    }
