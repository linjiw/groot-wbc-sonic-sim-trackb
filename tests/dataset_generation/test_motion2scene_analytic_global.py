import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search


def test_budget_and_station_cycling_use_search_checks_only():
    nominal = np.c_[0.12 + np.arange(20) * 0.04, np.full(20, 1.27)]
    observed = []

    def query(scenes):
        observed.extend(scenes.tolist())
        values = np.full((len(scenes), 17, 2), -0.02)
        values[:, :, 1] = np.where(scenes[:, 0] < 0.2, 0.02, -0.02)[:, None]
        return values

    outputs, trace = global_search(nominal, np.arange(20), query)
    assert len(observed) == 136
    assert int(trace["passing_station_count"]) == 2
    assert len(np.unique(outputs, axis=0)) == 2
    assert np.all(outputs[:, 0] < 0.2)


def test_no_passing_station_keeps_failures_in_denominator():
    nominal = np.c_[0.12 + np.arange(20) * 0.04, np.full(20, 1.27)]
    outputs, trace = global_search(nominal, np.arange(20), lambda x: np.zeros((len(x), 17, 2)))
    assert len(outputs) == 8
    assert int(trace["passing_station_count"]) == 0
    with pytest.raises(ValueError):
        global_search(nominal, np.arange(20), lambda x: np.full((len(x), 17, 2), np.nan))
