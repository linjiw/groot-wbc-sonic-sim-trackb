"""Finite-data outcome ambiguity, action ceilings and masked Bernoulli loss infima."""

import numpy as np


def observation_groups(features):
    x = np.array(features, dtype=np.float32, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite observation matrix required")
    x[x == 0] = 0  # Signed zero carries no information for this predictor.
    groups = {}
    for i, row in enumerate(x):
        groups.setdefault(row.tobytes(), []).append(i)
    return list(groups.values())


def empirical_limits(features, labels, mask):
    y = np.asarray(labels, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    groups = observation_groups(features)
    if y.shape != (len(features), 2) or mask.shape != y.shape:
        raise ValueError("two action outcomes and a matching known-label mask required")
    if not np.isfinite(y[mask]).all() or not np.isin(y[mask], [0, 1]).all() or not mask.any():
        raise ValueError("observed labels must be binary, with at least one observed outcome")
    complete = mask.all(1)
    observed_best = paired_best = 0
    entropy_sum = 0.0
    rows = []
    for ids in groups:
        known = mask[ids]
        values = y[ids]
        frequencies = []
        for a in range(2):
            v = values[known[:, a], a]
            p = float(v.mean()) if len(v) else None
            frequencies.append(p)
            if p is not None and 0 < p < 1:
                entropy_sum += len(v) * (-p * np.log(p) - (1 - p) * np.log1p(-p))
        selected = [i for i in ids if complete[i]]
        counts = y[selected].sum(0) if selected else np.zeros(2)
        obs = int(counts.max())
        pair = int(y[selected].max(1).sum()) if selected else 0
        observed_best += obs
        paired_best += pair
        rows.append(
            {
                "indices": ids,
                "complete_pairs": len(selected),
                "success_counts": counts.tolist(),
                "probabilities": frequencies,
                "observation_best_count": obs,
                "pair_best_count": pair,
                "action_ambiguity_count": pair - obs,
            }
        )
    n = int(complete.sum())
    return {
        "examples": len(y),
        "known_outcomes": int(mask.sum()),
        "complete_pairs": n,
        "groups": rows,
        "masked_bce_infimum": entropy_sum / int(mask.sum()),
        "observation_action_ceiling": observed_best / n if n else None,
        "paired_action_ceiling": paired_best / n if n else None,
        "action_ambiguity_gap": (paired_best - observed_best) / n if n else None,
        "scope": (
            "empirical complete-pair action ceilings; BCE uses all known outcomes; "
            "not a population guarantee"
        ),
    }
