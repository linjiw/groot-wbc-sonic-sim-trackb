"""Prevent carrier leakage, event-label leakage, and independent pair alignment."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import (
    carrier_split,
    shared_origin_pair,
    training_feature_case,
)


def test_all_derivatives_stay_in_parent_split():
    expected = {
        "train": {41001, 41002, 41003, 41004},
        "validation": {41005, 41006},
        "test": {41007, 41008},
    }
    observed = {key: set() for key in expected}
    for seed in range(41001, 41009):
        for _ in range(3):
            observed[carrier_split(seed)].add(seed)
    assert expected == observed
    assert not (observed["train"] & observed["test"])
    with pytest.raises(ValueError):
        carrier_split(41009)


def test_pair_translation_preserves_geometry_and_rejects_route_changes():
    neutral = np.zeros((10, 36))
    neutral[:, 0] = np.linspace(4, 5, 10)
    neutral[:, 1] = 3
    target = neutral.copy()
    target[4:6, 2] -= 0.05
    first, second = shared_origin_pair(neutral, target)
    np.testing.assert_allclose(first - second, neutral - target)
    np.testing.assert_array_equal(first[0, :2], [0, 0])
    target[:, 0] += 0.01
    with pytest.raises(ValueError, match="horizontal route"):
        shared_origin_pair(neutral, target)


def test_shuffled_features_are_deranged_and_never_read_excluded_cases():
    train = [f"{s}_event{e}" for s in range(41001, 41005) for e in range(3)]
    shuffled = [training_feature_case("shuffled", x, train) for x in train]
    assert set(shuffled) == set(train)
    assert all(a != b for a, b in zip(train, shuffled))
    assert training_feature_case("constant", train[0], train) is None
    assert training_feature_case("conditioned", train[0], train) == train[0]
    with pytest.raises(ValueError):
        training_feature_case("shuffled", "41007_event0", train)


def test_orientation_and_shape_mismatch_rejected():
    first = np.zeros((10, 36))
    second = first.copy()
    second[2, 3] = 0.01
    with pytest.raises(ValueError, match="orientation"):
        shared_origin_pair(first, second)
    with pytest.raises(ValueError, match="shape"):
        shared_origin_pair(first, second[:-1])
