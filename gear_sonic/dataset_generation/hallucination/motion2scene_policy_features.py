"""Geometry-only packet features and explicit missing physics labels."""

import numpy as np


def packet_features(packet):
    rays = packet["rays"]
    if len(rays) != 12:
        raise ValueError("requires twelve upper rays")
    values = []
    for ray in rays:
        hit = ray["hit"]
        values.extend(
            [
                float(hit is not None),
                hit["distance"] / 3 if hit else 1,
                hit["position"][2] / 2.5 if hit else 0,
            ]
        )
        lower = ray["lower_rays"]
        if len(lower) not in (0, 3):
            raise ValueError("requires either zero or three conditional lower rays")
        for i in range(3):
            observed = i < len(lower)
            hit = lower[i]["hit"] if observed else None
            values.extend(
                [float(observed), float(hit is not None), hit["distance"] / 3 if hit else 1]
            )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (144,) or not np.isfinite(result).all():
        raise ValueError("nonfinite or malformed policy features")
    return result


def feasibility_label(neutral_pass, crouch_pass):
    """Missing execution is unknown, never an infeasible or positive label."""
    if neutral_pass is None or crouch_pass is None:
        return {"status": "missing_comparator", "feasible": None, "minimum_cost_skill": None}
    if not isinstance(neutral_pass, bool) or not isinstance(crouch_pass, bool):
        raise ValueError("requires boolean measured passage outcomes")
    feasible = [neutral_pass, crouch_pass]
    return {
        "status": "measured" if any(feasible) else "neither_tested_skill_passes",
        "feasible": feasible,
        "minimum_cost_skill": next((i for i, passed in enumerate(feasible) if passed), None),
    }
