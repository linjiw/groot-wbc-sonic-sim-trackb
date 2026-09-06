"""Sample-set supervision and a genuinely five-evaluation correction."""

import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import margin_slack


def set_distance(scenes, weights, teacher):
    """Weighted softened energy distance; all inputs use physical station/height."""
    if scenes.ndim != 2 or scenes.shape[1] != 2 or weights.shape != scenes.shape[:1]:
        raise ValueError("expected scenes (N,2) and weights (N,)")
    if teacher.ndim != 2 or teacher.shape[1] != 2:
        raise ValueError("expected teacher (M,2)")
    if not all(torch.isfinite(x).all() for x in [scenes, weights, teacher]):
        raise ValueError("nonfinite set inputs")
    if torch.any(weights < 0) or not torch.allclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("weights must form a probability distribution")
    if len(teacher) == 0:
        return scenes.sum() * 0 + weights.sum() * 0
    scale = scenes.new_tensor([0.8, 0.35])
    x, y = scenes / scale, teacher.detach() / scale

    def distance(a, b):
        return ((a[:, None] - b[None]).square().sum(-1) + 1e-12).sqrt()

    return (
        2 * (weights[:, None] * distance(x, y)).sum() / len(y)
        - (weights[:, None] * weights[None] * distance(x, x)).sum()
        - distance(y, y).mean()
    )


def pattern_five(initial, query):
    if initial.ndim != 2 or initial.shape[1] != 2 or not torch.isfinite(initial).all():
        raise ValueError("expected finite initial (N,2)")
    low, high = initial.new_tensor([0.1, 1.1]), initial.new_tensor([0.9, 1.45])
    if torch.any(initial < low) or torch.any(initial > high):
        raise ValueError("initial outside domain")
    delta = initial.new_tensor([0.05, 0.03])
    lower, upper = torch.maximum(low, initial - delta), torch.minimum(high, initial + delta)
    scenes, values = [], []
    with torch.no_grad():
        for offset in [[0, 0], [-0.05, 0], [0.05, 0], [0, -0.03], [0, 0.03]]:
            candidate = (initial + initial.new_tensor(offset)).clamp(lower, upper)
            measured = query(candidate)
            if measured.shape != (len(initial), 17, 2) or not torch.isfinite(measured).all():
                raise ValueError("invalid search clearances")
            scenes.append(candidate)
            values.append(measured)
        scenes, values = torch.stack(scenes), torch.stack(values)
        selected = torch.stack([margin_slack(v) for v in values]).argmax(0)
        return scenes[selected, torch.arange(len(initial))], {
            "initial": initial,
            "scenes": scenes,
            "clearances": values,
            "selected": selected,
        }
