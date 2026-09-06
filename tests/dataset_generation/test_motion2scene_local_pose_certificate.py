from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.dataset_generation.hallucination.motion2scene_local_pose_certificate import (
    local_pose_certificate,
    rotvec_matrix,
)


def test_full_rotation_displacement_bounds_every_box_corner():
    h = np.array([0.05, 0.6, 0.05])
    certificate = local_pose_certificate(0.02, -0.03, h)
    assert certificate["status"] == "conditional_pass"
    assert certificate["dimension"] == 6
    corners = np.array(list(product((-1, 1), repeat=3))) * h
    for signs in product((-1, 1), repeat=6):
        t = np.array(signs[:3]) * certificate["translation_halfwidth_m"]
        v = np.array(signs[3:]) * certificate["rotvec_component_halfwidth_rad"]
        moved = corners @ rotvec_matrix(v).T + t
        assert (
            np.linalg.norm(moved - corners, axis=-1).max()
            <= certificate["box_displacement_bound_m"] + 1e-12
        )
    assert certificate["margin_slack_lower_bound_m"] > 0


def test_zero_margin_or_numerical_uncertainty_cannot_supply_positive_volume():
    assert local_pose_certificate(0.01, -0.02, [1, 1, 1])["status"] == "unresolved"
    assert local_pose_certificate(0.02, -0.010000001, [1, 1, 1])["chart_volume_m3_rad3"] == 0
    with pytest.raises(ValueError):
        local_pose_certificate(float("nan"), -0.02, [1, 1, 1])


def test_rotation_vector_implementation_matches_independent_scipy():
    rng = np.random.default_rng(6781)
    for v in [np.zeros(3), np.array([1e-12, 0, 0]), *rng.normal(size=(30, 3))]:
        np.testing.assert_allclose(
            rotvec_matrix(v), Rotation.from_rotvec(v).as_matrix(), atol=1e-14
        )
