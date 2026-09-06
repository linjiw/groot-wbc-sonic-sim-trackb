import copy

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_policy_features import (
    feasibility_label,
    packet_features,
)


def packet():
    return {"rays": [{"hit": None, "lower_rays": []} for _ in range(12)]}


def test_features_ignore_names_and_labels_but_preserve_lower_observation_mask():
    p = packet()
    empty = packet_features(p)
    p["rays"][0]["hit"] = {"distance": 1.5, "position": [9, 8, 1.3], "path": "beam"}
    p["rays"][0]["lower_rays"] = [{"hit": None}] * 3
    first = packet_features(p)
    assert not np.array_equal(empty, first)
    p["scene_id"], p["label"], p["active"] = "secret", 1, 1
    p["rays"][0]["hit"]["path"] = "wall"
    np.testing.assert_array_equal(first, packet_features(p))
    q = copy.deepcopy(p)
    q["rays"][0]["lower_rays"][0]["hit"] = {"distance": 0.5}
    assert not np.array_equal(packet_features(q), first)


def test_missing_label_is_not_a_failure_and_both_feasible_is_retained():
    assert feasibility_label(False, None)["status"] == "missing_comparator"
    assert feasibility_label(False, False)["status"] == "neither_tested_skill_passes"
    assert feasibility_label(True, True) == {
        "status": "measured",
        "feasible": [True, True],
        "minimum_cost_skill": 0,
    }
    assert feasibility_label(False, True)["minimum_cost_skill"] == 1
    with pytest.raises(ValueError):
        feasibility_label(1, True)


def test_invalid_packet_fails_closed():
    p = packet()
    p["rays"][0]["hit"] = {"distance": float("nan"), "position": [0, 0, 1]}
    with pytest.raises(ValueError):
        packet_features(p)
