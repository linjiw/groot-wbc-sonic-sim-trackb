"""Independent analytic cases for capsule–box distance and missed sampled contacts."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.capsule_box_exact import (
    capsule_box_clearance,
    segment_box_distance,
)


def test_contact_between_five_axis_samples():
    start, end = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])
    lower, upper = np.array([0.30, 0.32, -0.01]), np.array([0.40, 0.34, 0.01])
    points = start + np.linspace(0, 1, 5)[:, None] * (end - start)
    sampled = np.linalg.norm(points - np.clip(points, lower, upper), axis=1).min() - 0.005
    assert sampled > 0
    assert capsule_box_clearance(start, end, 0.005, lower, upper) == pytest.approx(-0.005)


def test_interior_stationary_minimum_outside_box():
    # Segment x+y=4 is closest to unit-square corner (1,1) at (2,2).
    assert segment_box_distance([0, 4, 0], [4, 0, 0], [0, 0, -1], [1, 1, 1]) == pytest.approx(
        np.sqrt(2)
    )


def test_degenerate_parallel_and_interior_segments():
    starts = [[2, 2, 2], [-2, 2, 0.5], [0.2, 0.3, 0.4]]
    ends = [[2, 2, 2], [2, 2, 0.5], [0.8, 0.7, 0.6]]
    actual = segment_box_distance(starts, ends, [0, 0, 0], [1, 1, 1])
    assert actual == pytest.approx([np.sqrt(3), 1, 0])


def test_reversal_translation_and_batch_invariance():
    rng = np.random.default_rng(718)
    starts, ends = rng.normal(size=(2, 20, 3)), rng.normal(size=(2, 20, 3))
    lo, hi = np.array([-0.2, -0.3, -0.4]), np.array([0.6, 0.5, 0.7])
    expected = segment_box_distance(starts, ends, lo, hi)
    assert np.allclose(expected, segment_box_distance(ends, starts, lo, hi))
    shift = np.array([17, -31, 6])
    assert np.allclose(
        expected, segment_box_distance(starts + shift, ends + shift, lo + shift, hi + shift)
    )
    assert np.allclose(
        expected.ravel(), segment_box_distance(starts.reshape(-1, 3), ends.reshape(-1, 3), lo, hi)
    )


def test_against_independent_scalar_convex_minimizer():
    from scipy.optimize import minimize_scalar

    rng = np.random.default_rng(943)
    lo, hi = np.array([-0.2, -0.3, -0.4]), np.array([0.6, 0.5, 0.7])
    for start, end in zip(rng.normal(size=(50, 3)), rng.normal(size=(50, 3))):

        def objective(t):
            point = start + t * (end - start)
            return np.sum((point - np.clip(point, lo, hi)) ** 2)

        result = minimize_scalar(
            objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-13}
        )
        expected = np.sqrt(min(result.fun, objective(0), objective(1)))
        assert segment_box_distance(start, end, lo, hi) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1))
def test_reject_invalid_radii(bad):
    with pytest.raises(ValueError):
        capsule_box_clearance([0, 0, 0], [1, 1, 1], bad, [0, 0, 0], [1, 1, 1])


def test_reject_bad_bounds_and_shapes():
    with pytest.raises(ValueError):
        segment_box_distance([0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1])
    with pytest.raises(ValueError):
        capsule_box_clearance([0, 0, 0], [1, 1, 1], [1, 2], [0, 0, 0], [1, 1, 1])
