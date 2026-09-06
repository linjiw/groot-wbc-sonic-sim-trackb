import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import (
    analytic_proposals,
    reference_coverage,
    station_bins,
)


def test_empty_grid_is_unknown_not_zero_coverage():
    result = reference_coverage([[0.12, 1.2]], np.array([False]), [[0.5, 1.25]], np.array([True]))
    assert result["reference_relative_station_coverage"] is None
    assert result["accepted_bins_outside_reference_support"]


def test_only_accepted_scenes_cover_reference_stations():
    grid = [[0.12, 1.2], [0.16, 1.2]]
    result = reference_coverage(grid, np.array([True, True]), grid, np.array([True, False]))
    assert result["reference_relative_station_coverage"] == 0.5
    assert station_bins([[0.9, 1.45]]).tolist() == [19]
    with pytest.raises(ValueError):
        station_bins([[0.91, 1.2]])


def test_analytic_proposal_localizes_pair_separation_without_event_id():
    x = np.linspace(0, 4, 101)
    route = np.c_[x, np.zeros(len(x))]
    states = {}
    for label in ("neutral", "d055"):
        z = np.full(len(x), 1.3)
        if label == "d055":
            z[(x > 1.4) & (x < 2.6)] -= 0.1
        p = np.stack([x, np.zeros(len(x)), z], -1)[:, None]
        states[label] = {"label": label, "starts": p, "ends": p, "radii": np.array([0.02])}
    scenes, _ = analytic_proposals(states, route, 0.0, depth=0.1, width=1.2, count=4)
    assert np.all((scenes[:, 0] > 0.35) & (scenes[:, 0] < 0.65))
    assert np.allclose(scenes[:, 1], 1.27)
