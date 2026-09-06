"""Fixed-budget local searches for motion-conditioned station/height proposals."""

import torch

from .motion2scene_refinement import margin_slack

METHODS = ("probe_refine", "multistart", "pattern")


def station_search(initial, query, method):
    """Return the best of 17 queries per input within its ORIGINAL trust box.

    The callback returns (candidates, placements, neutral/target). Incumbent selection
    uses only these search placements; the output still needs independent checking.
    Every queried candidate, including duplicates at bounds, consumes the fixed budget.
    """
    if method not in METHODS or initial.ndim != 2 or initial.shape[-1] != 2 or not len(initial):
        raise ValueError("invalid method or initial shape")
    low, span = initial.new_tensor([0.1, 1.1]), initial.new_tensor([0.8, 0.35])
    if (
        not torch.isfinite(initial).all()
        or torch.any(initial < low)
        or torch.any(initial > low + span)
    ):
        raise ValueError("initial outside domain")
    radius = initial.new_tensor([0.05, 0.03])
    lower, upper = torch.maximum(initial - radius, low), torch.minimum(initial + radius, low + span)
    scenes_trace, values_trace = [], []

    def clamp(scenes):
        return torch.maximum(torch.minimum(scenes, upper), lower)

    def evaluate(scenes):
        values = query(scenes)
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite search query")
        scenes_trace.append(scenes.detach().clone())
        values_trace.append(values.detach().clone())
        return values

    def winner():
        indices = torch.stack([margin_slack(v) for v in values_trace]).argmax(0)
        return torch.stack(scenes_trace)[indices, torch.arange(len(initial))], indices

    def gradient(start, steps):
        position = torch.nn.Parameter((start.detach().clone() - low) / span)
        optimizer = torch.optim.Adam([position], lr=0.02)
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            values = evaluate(low + span * position)
            if step == steps:
                break
            target = ((0.01 - values[:, :, 1].amin(1)).clamp(min=0) / 0.01).square()
            neutral = ((0.01 + values[:, :, 0].amax(1)).clamp(min=0) / 0.01).square()
            (target + neutral).sum().backward()
            if position.grad is None or not torch.isfinite(position.grad).all():
                raise FloatingPointError("nonfinite or absent gradient")
            torch.nn.utils.clip_grad_norm_([position], 10)
            optimizer.step()
            with torch.no_grad():
                position.copy_(
                    torch.maximum(
                        torch.minimum(position, (upper - low) / span), (lower - low) / span
                    )
                )

    if method == "probe_refine":
        with torch.no_grad():
            for shift in [0.0, -0.025, 0.025, -0.05, 0.05]:
                evaluate(clamp(initial + initial.new_tensor([shift, 0.0])))
            start, _ = winner()
        gradient(start, 11)
    elif method == "multistart":
        with torch.no_grad():
            for shift in [-0.025, 0.025]:
                evaluate(clamp(initial + initial.new_tensor([shift, 0.0])))
        for shift in [0.0, -0.05, 0.05]:
            gradient(clamp(initial + initial.new_tensor([shift, 0.0])), 4)
    else:
        with torch.no_grad():
            evaluate(initial)
            for scale in [1.0, 0.5, 0.25, 0.125]:
                center, _ = winner()
                for delta in [
                    [-0.05 * scale, 0.0],
                    [0.05 * scale, 0.0],
                    [0.0, -0.03 * scale],
                    [0.0, 0.03 * scale],
                ]:
                    evaluate(clamp(center + initial.new_tensor(delta)))
    output, indices = winner()
    if len(scenes_trace) != 17:
        raise RuntimeError("query budget changed")
    return output, {
        "initial": initial.detach().clone(),
        "scenes": torch.stack(scenes_trace),
        "clearances": torch.stack(values_trace),
        "selected": indices,
    }
