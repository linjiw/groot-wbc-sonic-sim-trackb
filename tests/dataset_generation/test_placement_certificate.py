"""Conservative placement bounds must not promote samples or budget exhaustion to proof."""

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.placement_certificate import (
    box_displacement_bound,
    certify_placement,
)


def test_rotation_translation_bound_encloses_corresponding_box_corners():
    rng = np.random.default_rng(481)
    radius = np.hypot(0.05, 0.6)
    limits = np.array([0.02, 0.02, 0.01, 0.02])
    bound = box_displacement_bound(limits, radius)
    for _ in range(100):
        offset = rng.uniform(-limits, limits)
        c, s = np.cos(offset[3]), np.sin(offset[3])
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = np.array(
            [[x, y, z] for x in [-0.05, 0.05] for y in [-0.6, 0.6] for z in [-0.05, 0.05]]
        )
        displacement = corners @ rotation.T + offset[:3] - corners
        assert np.linalg.norm(displacement, axis=-1).max() <= bound + 1e-15


def test_subdivision_covers_linear_query_domain_with_valid_bounds():
    result = certify_placement(
        lambda p: (0.04 + p[0], -0.04 + p[0]),
        [-0.02, 0, 0, 0],
        [0.02, 0, 0, 0],
        horizontal_radius=0.6,
        query_error=1e-8,
    )
    assert result["status"] == "conditional_pass"
    leaves = [r for r in result["trace"] if r["decision"] == "certified_cell"]
    assert sum(r["upper"][0] - r["lower"][0] for r in leaves) == pytest.approx(0.04)
    assert all(r["margin_slack_lower_bound_m"] >= 0 for r in leaves)


def test_grid_passing_endpoints_and_center_do_not_hide_interior_collision():
    # 1-Lipschitz V-shaped target clearance has a dip between those three points.
    def query(p):
        return 0.005 + abs(p[0] - 0.01), -0.1

    assert all(query(np.array([x, 0, 0, 0]))[0] >= 0.01 for x in (-0.02, 0, 0.02))
    result = certify_placement(query, [-0.02, 0, 0, 0], [0.02, 0, 0, 0], horizontal_radius=0.6)
    assert result["status"] == "counterexample"
    assert result["counterexample"]["target_clearance_m"] < 0.01


def test_budget_exhaustion_is_unresolved_even_when_sample_passes():
    result = certify_placement(
        lambda p: (0.011, -0.011), [-0.02] * 4, [0.02] * 4, horizontal_radius=0.6, max_queries=1
    )
    assert result["status"] == "unresolved"
    assert result["queries"] == 1
    assert result["unresolved_cells"] == 2


def test_query_error_prevents_false_point_certificate():
    result = certify_placement(
        lambda p: (0.010000001, -0.010000001),
        [0] * 4,
        [0] * 4,
        horizontal_radius=0.6,
        query_error=1e-8,
    )
    assert result["status"] == "unresolved"


def test_upright_point_violation_is_counterexample():
    result = certify_placement(lambda p: (0.1, -0.005), [0] * 4, [0] * 4, horizontal_radius=0.6)
    assert result["status"] == "counterexample"


def test_invalid_bound_and_nonfinite_query_rejected():
    with pytest.raises(ValueError):
        box_displacement_bound([0, 0, 0, 4], 0.6)
    with pytest.raises(ValueError, match="finite clearance"):
        certify_placement(lambda p: (float("nan"), -0.1), [0] * 4, [0] * 4, horizontal_radius=0.6)
