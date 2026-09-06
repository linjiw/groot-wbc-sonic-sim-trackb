import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_native_crossing import (
    downstream_support,
    stable_finish,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule


def test_extent_delays_crossing_and_rotation_changes_support():
    shapes = {"body": [CollisionCapsule((0, 0, 0), (0.4, 0, 0), 0.1)]}
    p = np.array([[[0.3, 0, 1]], [[0.3, 0, 1]]])
    q = np.array([[[1, 0, 0, 0]], [[0, 0, 0, 1]]])
    beam = {"yaw_rad": 0, "center_xy_m": [0, 0]}
    support = downstream_support(p, q, ["body"], shapes, beam, allowance_m=0)
    np.testing.assert_allclose(support, [0.2, -0.2])
    assert np.all(p[:, 0, 0] > 0.15)  # origins cross in both poses; the rear extent does not
    with pytest.raises(ValueError, match="owners"):
        downstream_support(p, q, ["other"], shapes, beam)


def test_stability_requires_full_window_and_retains_failure():
    assert stable_finish([0.2] * 15, [True] * 15, 0.15) is None
    assert stable_finish([0.2] * 16, [True] * 16, 0.15) == 16
    assert stable_finish([0.2] * 16, [False] + [True] * 15, 0.15) is None
