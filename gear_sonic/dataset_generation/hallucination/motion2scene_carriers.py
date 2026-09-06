"""Grouping and shared-frame contracts for reference-only motion conditioning tests."""

import numpy as np


def carrier_split(seed):
    if seed in (41001, 41002, 41003, 41004):
        return "train"
    if seed in (41005, 41006):
        return "validation"
    if seed in (41007, 41008):
        return "test"
    raise ValueError("unregistered source carrier")


def shared_origin_pair(neutral, target):
    """Canonicalize both references with the same translation; preserve relative geometry."""
    neutral, target = np.asarray(neutral), np.asarray(target)
    if neutral.shape != target.shape or neutral.ndim != 2 or neutral.shape[1] != 36:
        raise ValueError("paired G1 references must share (T,36) shape")
    if not np.isfinite([neutral, target]).all():
        raise ValueError("references must be finite")
    if not np.allclose(neutral[:, :2], target[:, :2], atol=1e-8, rtol=0):
        raise ValueError("operator changed the shared horizontal route")
    if not np.allclose(neutral[:, 3:7], target[:, 3:7], atol=1e-8, rtol=0):
        raise ValueError("operator changed the root orientation")
    first, second = neutral.copy(), target.copy()
    first[:, :2] -= neutral[0, :2]
    second[:, :2] -= neutral[0, :2]
    return first, second


def training_feature_case(mode, case_id, train_ids):
    """Fixed derangement uses training cases only; no validation/test feature leakage."""
    if case_id not in train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("requires unique training cases and a training target")
    if mode == "conditioned":
        return case_id
    if mode == "constant":
        return None
    if mode == "shuffled" and len(train_ids) > 1:
        return train_ids[(train_ids.index(case_id) + 1) % len(train_ids)]
    raise ValueError("invalid conditioning control")
