"""Gates for scene-aware placement of Kimodo reference motions.

The regression these guard: the 2026-08-15 batch accepted 2/16 rollouts because
every motion was dropped at a scene's route start with a fixed yaw, so laterally
curving references collided with racks at up to 1850 N regardless of controller
quality.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.kimodo_motion_adapter import transform_qpos_to_scene
from gear_sonic.dataset_generation.route_placement import (
    DEFAULT_TRACKING_MARGIN_M,
    canonical_path_xy,
    plan_placements,
    summarize_placements,
    transform_path,
)
from gear_sonic.dataset_generation.scene_asset_preflight import (
    SceneObstacleMap,
    load_scene_obstacle_map,
)

SCENES = ("factory_aisle", "household_room")


def _straight_path(length_m: float = 3.0, points: int = 40) -> np.ndarray:
    xs = np.linspace(0.0, length_m, points)
    return np.stack([xs, np.zeros_like(xs)], axis=1)


def _open_map(**overrides) -> SceneObstacleMap:
    defaults = dict(
        scene_id="unit_test",
        obstacles=(("/pillar", (-0.5, -0.5, 0.5, 0.5)),),
        walkable_min_xy=(-10.0, -10.0),
        walkable_max_xy=(10.0, 10.0),
        support_z=0.0,
        robot_clearance_height_m=1.9,
        route_clearance_radius_m=0.45,
        declared_route_xy=((-5.0, -5.0), (5.0, -5.0)),
    )
    defaults.update(overrides)
    return SceneObstacleMap(**defaults)


def _plan(path, obstacle_map, **kwargs):
    """Coarse search defaults: these tests assert behaviour, not search resolution."""
    kwargs.setdefault("translation_step_m", 1.0)
    kwargs.setdefault("yaw_steps", 8)
    return plan_placements(path, obstacle_map, **kwargs)


# --------------------------------------------------------------------------
# Transform agreement with the adapter -- the correctness anchor
# --------------------------------------------------------------------------


def test_planner_transform_matches_the_adapter_exactly():
    """A planned placement must be the placement the converter actually produces."""
    rng = np.random.default_rng(0)
    qpos = np.zeros((25, 36), dtype=np.float64)
    qpos[:, 0] = np.linspace(1.5, 4.0, 25)  # start away from origin on purpose
    qpos[:, 1] = rng.normal(scale=0.05, size=25)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0  # unit wxyz quaternion

    start = (-7.0, 0.6, 0.0)
    yaw = 0.7

    planned = transform_path(canonical_path_xy(qpos), start[:2], yaw)
    adapted = transform_qpos_to_scene(qpos, scene_start_xyz=start, scene_yaw=yaw)[:, :2]

    assert np.allclose(planned, adapted, atol=1e-9)


def test_canonical_path_puts_the_first_sample_at_the_origin():
    qpos = np.zeros((8, 36))
    qpos[:, 0] = np.arange(8) + 5.0
    qpos[:, 1] = 2.0
    path = canonical_path_xy(qpos)
    assert np.allclose(path[0], (0.0, 0.0))
    assert np.allclose(path[-1], (7.0, 0.0))


def test_canonical_path_rejects_malformed_input():
    with pytest.raises(ValueError):
        canonical_path_xy(np.zeros((5,)))
    with pytest.raises(ValueError):
        canonical_path_xy(np.zeros((5, 1)))


def test_transform_path_rotation_is_about_plus_z():
    path = np.array([[0.0, 0.0], [1.0, 0.0]])
    rotated = transform_path(path, (0.0, 0.0), math.pi / 2.0)
    assert np.allclose(rotated[-1], (0.0, 1.0), atol=1e-12)


# --------------------------------------------------------------------------
# Clearance behaviour
# --------------------------------------------------------------------------


def test_every_returned_placement_meets_the_required_clearance():
    obstacle_map = _open_map()
    required = obstacle_map.route_clearance_radius_m + DEFAULT_TRACKING_MARGIN_M

    placements = _plan(_straight_path(), obstacle_map, max_results=5)

    assert placements
    for placement in placements:
        assert placement.clearance_m >= required
        placed = transform_path(_straight_path(), placement.start_xy, placement.yaw_rad)
        points = [(float(x), float(y)) for x, y in placed]
        recomputed, _ = obstacle_map.clearance_to_obstacles(points)
        assert recomputed >= required


def test_placements_stay_inside_the_walkable_rectangle():
    obstacle_map = _open_map()
    placements = plan_placements(_straight_path(), obstacle_map, max_results=5)

    for placement in placements:
        placed = transform_path(_straight_path(), placement.start_xy, placement.yaw_rad)
        points = [(float(x), float(y)) for x, y in placed]
        assert obstacle_map.within_walkable_bounds(points)


def test_a_larger_tracking_margin_never_admits_more_placements():
    obstacle_map = _open_map()
    lenient = _plan(_straight_path(), obstacle_map, max_results=64, tracking_margin_m=0.0)
    strict = _plan(_straight_path(), obstacle_map, max_results=64, tracking_margin_m=1.5)
    assert min(p.clearance_m for p in strict) >= min(p.clearance_m for p in lenient)


def test_a_scene_with_no_room_yields_no_placement():
    """A path longer than the walkable box cannot be placed anywhere."""
    obstacle_map = _open_map(walkable_min_xy=(-1.0, -1.0), walkable_max_xy=(1.0, 1.0))
    assert _plan(_straight_path(length_m=8.0), obstacle_map, max_results=4) == []


def test_returned_placements_are_spread_not_clustered():
    obstacle_map = _open_map()
    placements = plan_placements(_straight_path(), obstacle_map, max_results=5)
    assert len(placements) >= 2
    separations = [
        math.dist(a.start_xy, b.start_xy) + abs(a.yaw_rad - b.yaw_rad)
        for i, a in enumerate(placements)
        for b in placements[i + 1 :]
    ]
    assert min(separations) > 0.0


def test_results_are_ordered_by_descending_clearance():
    placements = _plan(_straight_path(), _open_map(), max_results=6)
    clearances = [placement.clearance_m for placement in placements]
    assert clearances == sorted(clearances, reverse=True)


def test_short_paths_are_rejected():
    with pytest.raises(ValueError, match="at least two points"):
        _plan(np.zeros((1, 2)), _open_map())


def test_summary_reports_count_and_clearance_span():
    placements = _plan(_straight_path(), _open_map(), max_results=4)
    summary = summarize_placements(placements)
    assert summary["count"] == len(placements)
    assert summary["min_clearance_m"] <= summary["max_clearance_m"]
    assert summarize_placements([]) == {"count": 0}


def test_decimation_does_not_let_a_path_cut_a_corner():
    """A dense detour around a pillar must not be simplified into a collision."""
    obstacle_map = _open_map()
    theta = np.linspace(-math.pi / 2, math.pi / 2, 400)
    detour = np.stack([np.sin(theta) * 2.0, -np.cos(theta) * 2.0], axis=1)
    detour = detour - detour[0]

    placements = _plan(detour, obstacle_map, max_results=3, max_path_points=12)

    required = obstacle_map.route_clearance_radius_m + DEFAULT_TRACKING_MARGIN_M
    for placement in placements:
        placed = transform_path(detour, placement.start_xy, placement.yaw_rad)
        full_clearance, _ = obstacle_map.clearance_to_obstacles(
            [(float(x), float(y)) for x, y in placed]
        )
        assert full_clearance >= required


# --------------------------------------------------------------------------
# Real scene packages
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene_id", SCENES)
def test_repo_scene_obstacle_map_loads_and_matches_declared_clearance(scene_id):
    obstacle_map = load_scene_obstacle_map(scene_id)

    assert obstacle_map.obstacles
    assert obstacle_map.route_clearance_radius_m == pytest.approx(0.45)
    declared, _ = obstacle_map.clearance_to_obstacles(obstacle_map.declared_route_xy)
    # The checked-in canonical routes are documented as 0.80 m (household) and
    # 0.70 m (factory); both must clear the declared body radius.
    assert declared >= obstacle_map.route_clearance_radius_m


def test_unknown_scene_is_rejected():
    with pytest.raises(ValueError, match="is not in"):
        load_scene_obstacle_map("no_such_scene")


@pytest.mark.parametrize("scene_id", SCENES)
def test_a_short_straight_walk_can_be_placed_in_every_repo_scene(scene_id):
    placements = plan_placements(
        _straight_path(length_m=2.0, points=20),
        load_scene_obstacle_map(scene_id),
        max_results=2,
        translation_step_m=0.5,
        yaw_steps=8,
    )
    assert placements, f"no placement found in {scene_id}"
