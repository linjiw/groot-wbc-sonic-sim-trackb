from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_ray_observer import (
    normalize_hit,
    observe_overhead,
)


def hit(path="/World/ground/terrain/CounterfactualBeam", distance=2.0, z=1.3):
    return SimpleNamespace(collision=path, distance=distance, position=[2.7, 0.2, z])


class Query:
    def __init__(self, hits):
        self.hits = hits

    def raycast_all(self, origin, direction, distance, callback):
        assert distance == 3.0
        assert np.isclose(np.linalg.norm(direction), 1)
        for h in self.hits:
            if not callback(h):
                break


def test_typed_and_mapping_hits_match():
    expected = normalize_hit(hit())
    assert expected == normalize_hit(
        {"collision": hit().collision, "distance": 2, "position": [2.7, 0.2, 1.3]}
    )


def test_robot_is_ignored_and_nearest_scene_hit_controls_occupancy():
    query = Query([hit("/World/envs/env_0/Robot/torso_link", 0.1), hit()])
    occupied, _, rays = observe_overhead(np.array([0, 0, 0.8]), [1, 0, 0, 0], query)
    assert occupied and len(rays) == 12
    assert all(r["hit"]["distance"] == 2.0 for r in rays)
    # An occluder is retained; the beam is not selected by identity.
    query = Query([hit("/World/occluder", 1.0, 2.0), hit()])
    assert not observe_overhead(np.zeros(3), [1, 0, 0, 0], query)[0]


def test_empty_query_and_raised_beam_are_negative():
    for query in (Query([]), Query([hit(z=2.0)])):
        assert not observe_overhead(np.zeros(3), [1, 0, 0, 0], query)[0]


def test_callback_failure_is_raised_outside_engine_callback():
    with pytest.raises(RuntimeError, match="observation is invalid"):
        observe_overhead(np.zeros(3), [1, 0, 0, 0], Query([object()]))
    with pytest.raises(ValueError, match="invalid ray hit"):
        normalize_hit(hit(distance=float("nan")))
