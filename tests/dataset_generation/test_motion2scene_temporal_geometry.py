import numpy as np
import pytest

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_temporal_geometry import (
    clearance_lower_bounds,
    interpolate_poses,
    interval_displacement,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world


def clearance(p, q, shapes, center):
    a, b, r, _ = body_capsules_world(p, q, ["body"], capsules=shapes)
    return capsule_box_clearance(a - center, b - center, r, -np.ones(3) * 0.05, np.ones(3) * 0.05)


def test_translation_tunneling_is_detected_and_endpoints_preserved():
    p = np.array([[[0.0, 0, 0]], [[2.0, 0, 0]]])
    q = np.array([[[1.0, 0, 0, 0]], [[-1.0, 0, 0, 0]]])
    shapes = {"body": [CollisionCapsule((0, 0, 0), (0, 0, 0), 0.01)]}
    center = np.array([1, 0, 0])
    coarse = clearance(p, q, shapes, center)
    assert (coarse > 0).all()
    bound = interval_displacement(p, q, ["body"], shapes)
    assert clearance_lower_bounds(coarse, bound).min() < 0
    pp, qq = interpolate_poses(p, q, 4)
    np.testing.assert_allclose(pp[::4], p)
    assert np.allclose(np.linalg.norm(qq, axis=-1), 1)
    assert clearance(pp, qq, shapes, center).min() < 0


def test_rotating_offset_sphere_sweeps_through_box():
    p = np.zeros((2, 1, 3))
    angle = np.array([-0.7, 0.7])
    q = np.c_[np.cos(angle), np.zeros((2, 2)), np.sin(angle)][:, None]
    shapes = {"body": [CollisionCapsule((1, 0, 0), (1, 0, 0), 0.01)]}
    center = np.array([1, 0, 0])
    coarse = clearance(p, q, shapes, center)
    assert (coarse > 0).all()
    bounds = clearance_lower_bounds(coarse, interval_displacement(p, q, ["body"], shapes))
    pp, qq = interpolate_poses(p, q, 100)
    fine = clearance(pp, qq, shapes, center)
    assert bounds.min() <= fine.min() < 0


def test_interval_bounds_cover_dense_independent_subdivision():
    rng = np.random.default_rng(6931)
    shapes = {"body": [CollisionCapsule((0.2, -0.1, 0.3), (-0.1, 0.5, 0.1), 0.02)]}
    for _ in range(30):
        p = rng.normal(size=(2, 1, 3))
        q = rng.normal(size=(2, 1, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        pp, qq = interpolate_poses(p, q, 4)
        c = clearance(pp, qq, shapes, np.zeros(3))
        lower = clearance_lower_bounds(c, interval_displacement(pp, qq, ["body"], shapes))
        dense_p, dense_q = interpolate_poses(pp, qq, 100)
        dense = clearance(dense_p, dense_q, shapes, np.zeros(3))
        for i in range(4):
            assert lower[i, 0] <= dense[i * 100 : (i + 1) * 100 + 1].min() + 1e-10


def test_stationary_and_invalid_data():
    p = np.zeros((2, 1, 3))
    q = np.tile([1.0, 0, 0, 0], (2, 1, 1))
    pp, qq = interpolate_poses(p, q, 4)
    assert np.allclose(pp, 0) and np.allclose(qq, q[0])
    for invalid in (0, 1.5):
        with pytest.raises(ValueError):
            interpolate_poses(p, q, invalid)
    q[:] = 0
    with pytest.raises(ValueError):
        interpolate_poses(p, q, 4)
    with pytest.raises(ValueError):
        clearance_lower_bounds(np.ones((2, 1)), np.ones((1, 1)), -1)
