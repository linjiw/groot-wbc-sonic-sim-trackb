"""Exact pattern prefixes and selective 5/9/17 search without audit feedback."""

import torch

from .motion2scene_refinement import margin_slack


def budgeted_pattern(initial, query, budget):
    if budget not in (5, 9, 17, "adaptive"):
        raise ValueError("registered budget must be 5, 9, 17 or adaptive")
    if initial.ndim != 2 or initial.shape[1] != 2 or not len(initial):
        raise ValueError("expected a nonempty station/height matrix")
    low, high = initial.new_tensor([0.1, 1.1]), initial.new_tensor([0.9, 1.45])
    if not torch.isfinite(initial).all() or torch.any(initial < low) or torch.any(initial > high):
        raise ValueError("initial outside finite registered domain")
    delta = initial.new_tensor([0.05, 0.03])
    lower, upper = torch.maximum(low, initial - delta), torch.minimum(high, initial + delta)
    output = initial.detach().clone()
    best = initial.new_full((len(initial),), -float("inf"))
    counts = torch.zeros(len(initial), dtype=torch.long)
    traces = []

    def evaluate(indices, scenes):
        values = query(scenes)
        if values.shape != (len(indices), 17, 2) or not torch.isfinite(values).all():
            raise ValueError("invalid search-side geometry")
        slack = margin_slack(values)
        improved = slack > best[indices]
        output[indices[improved]] = scenes[improved]
        best[indices[improved]] = slack[improved]
        counts[indices] += 1
        traces.append(
            {"indices": indices.clone(), "scenes": scenes.clone(), "clearances": values.clone()}
        )

    with torch.no_grad():
        active = torch.arange(len(initial))
        evaluate(active, initial)
        for stage, scale in enumerate((1.0, 0.5, 0.25, 0.125)):
            center = output[active].clone()
            for offset in (
                (-0.05 * scale, 0),
                (0.05 * scale, 0),
                (0, -0.03 * scale),
                (0, 0.03 * scale),
            ):
                scenes = torch.maximum(
                    torch.minimum(center + initial.new_tensor(offset), upper[active]), lower[active]
                )
                evaluate(active, scenes)
            used = 5 + stage * 4
            if budget != "adaptive" and used == budget:
                break
            if budget == "adaptive" and used in (5, 9):
                active = active[best[active] < 0]
                if not len(active):
                    break
    return output, {
        "initial": initial.detach().clone(),
        "output": output,
        "evaluations_per_output": counts,
        "final_search_slack": best,
        "queries": sum(len(t["indices"]) * 34 for t in traces),
        "evaluations": traces,
    }
