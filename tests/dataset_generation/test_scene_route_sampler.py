"""Tests for planning routes through a fixed scene (the scene-first direction)."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.scene_route_sampler import (
    GENERATION_FPS,
    MAX_SEGMENT_FRAMES,
    Obstacle,
    RouteSamplingError,
    sample_routes,
)

ROOM = (8.0, 6.0)


def empty_room_routes(**kwargs):
    return sample_routes([], ROOM, count=2, seed=1, **kwargs)


def test_plans_routes_in_an_empty_room():
    routes = empty_room_routes()
    assert routes
    for route in routes:
        assert route.path_length_m >= 2.0
        assert route.trajectory_xy.shape[1] == 2


def test_routes_keep_the_clearance_they_claim():
    obstacles = [Obstacle(rect=(-0.5, -3.0, 0.5, 1.0)), Obstacle(rect=(-0.5, 2.0, 0.5, 3.0))]
    routes = sample_routes(obstacles, ROOM, count=3, seed=3)
    for route in routes:
        assert route.min_clearance_m >= route.required_clearance_m - 1e-9


def test_overhead_pieces_do_not_block_the_floor():
    """A shelf at 1.6 m is something the robot walks under, not around."""
    wall = [Obstacle(rect=(-0.4, -3.0, 0.4, 3.0), z_base=0.0)]
    overhead = [Obstacle(rect=(-0.4, -3.0, 0.4, 3.0), z_base=1.6)]
    with pytest.raises(RouteSamplingError):
        # A full-height wall across the room leaves the two halves disconnected, and no
        # start/goal pair that needs crossing can be planned.
        sample_routes(wall, (3.0, 6.0), count=1, seed=0, min_length_m=2.0, max_attempts=40)
    assert sample_routes(overhead, (3.0, 6.0), count=1, seed=0, min_length_m=2.0)


def test_swept_top_decides_whether_a_piece_blocks():
    piece = [Obstacle(rect=(-0.4, -3.0, 0.4, 3.0), z_base=1.0)]
    # With a 1.317 m swept top the piece is in the way; lower the robot and it is not.
    with pytest.raises(RouteSamplingError):
        sample_routes(piece, (3.0, 6.0), count=1, seed=0, swept_top_m=1.317, max_attempts=40)
    assert sample_routes(piece, (3.0, 6.0), count=1, seed=0, swept_top_m=0.9)


def test_impassable_room_raises_instead_of_returning_nothing():
    """An empty list would let a batch report success having produced no routes."""
    blocked = [Obstacle(rect=(-4.0, -3.0, 4.0, 3.0))]
    with pytest.raises(RouteSamplingError, match="clearance"):
        sample_routes(blocked, ROOM, count=1, seed=0)


def test_trajectory_is_constant_speed_at_the_generation_frame_rate():
    (route,) = sample_routes([], ROOM, count=1, seed=5, speed_mps=0.8)
    steps = np.linalg.norm(np.diff(route.trajectory_xy, axis=0), axis=1)
    # Samples are evenly spaced in arc length, so chords are equal along straights and
    # very slightly shorter across a corner, where the chord cuts the turn. That is
    # geometry, not drift: a fraction of a percent, not a speed profile.
    assert steps.std() / steps.mean() < 0.01
    assert steps.mean() == pytest.approx(0.8 / GENERATION_FPS, rel=0.02)


def test_duration_follows_from_length_and_speed():
    (route,) = sample_routes([], ROOM, count=1, seed=7, speed_mps=1.0)
    assert route.duration_s == pytest.approx(route.path_length_m / 1.0, rel=0.05)


def test_routes_never_exceed_the_single_segment_frame_ceiling():
    # A slow speed in a big room is the case that would overrun 300 frames.
    routes = sample_routes([], (12.0, 12.0), count=4, seed=11, speed_mps=0.3)
    for route in routes:
        assert route.frames <= MAX_SEGMENT_FRAMES


def test_kimodo_conversion_swaps_the_ground_axes():
    """mujoco (x, y, z) = kimodo (z, x, y), so scene (X, Y) becomes kimodo (Y, X)."""
    (route,) = sample_routes([], ROOM, count=1, seed=13)
    root2d = route.to_kimodo_root2d()
    assert root2d.shape == route.trajectory_xy.shape
    assert root2d[:, 0] == pytest.approx(route.trajectory_xy[:, 1])
    assert root2d[:, 1] == pytest.approx(route.trajectory_xy[:, 0])


def test_routes_stay_inside_the_room():
    routes = sample_routes([], ROOM, count=3, seed=17)
    for route in routes:
        assert np.all(np.abs(route.trajectory_xy[:, 0]) <= ROOM[0] / 2 + 1e-9)
        assert np.all(np.abs(route.trajectory_xy[:, 1]) <= ROOM[1] / 2 + 1e-9)


def test_sampling_is_deterministic_for_a_seed():
    first = sample_routes([], ROOM, count=2, seed=23)
    second = sample_routes([], ROOM, count=2, seed=23)
    for a, b in zip(first, second):
        assert a.trajectory_xy == pytest.approx(b.trajectory_xy)


def test_a_wider_body_needs_a_wider_gap():
    gap = [Obstacle(rect=(-3.0, -0.35, 3.0, 0.35))]
    narrow_room = (7.0, 3.0)
    # A 0.20 m half-width with no margin threads the gap beside the slab; 0.60 m does not.
    assert sample_routes(
        gap, narrow_room, count=1, seed=29, body_half_width_m=0.20, margin_m=0.0
    )
    with pytest.raises(RouteSamplingError):
        sample_routes(
            gap, narrow_room, count=1, seed=29, body_half_width_m=0.60, margin_m=0.0,
            max_attempts=40,
        )
