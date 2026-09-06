import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_distinct import (
    distinct_indices,
)


def test_balances_stations_before_taking_second_height_and_preserves_refusals():
    trace = {
        "candidates": np.array([[0.2, 1.2], [0.2, 1.21], [0.4, 1.2], [0.4, 1.21]]),
        "station_ids": np.array([0, 0, 1, 1]),
        "slack": np.array([0.03, 0.02, 0.01, -0.01]),
    }
    assert distinct_indices(trace, count=4).tolist() == [0, 2, 1, 3]


def test_clamped_duplicate_coordinates_are_never_double_counted():
    trace = {
        "candidates": np.array([[0.2, 1.1], [0.2, 1.1], [0.4, 1.2]]),
        "station_ids": np.array([0, 0, 1]),
        "slack": np.array([0.03, 0.02, 0.01]),
    }
    indices = distinct_indices(trace, count=2)
    assert len(np.unique(trace["candidates"][indices], axis=0)) == 2
