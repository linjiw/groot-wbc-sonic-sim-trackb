"""Tests for keeping the distance to the constraint instead of reducing it to a verdict.

The properties under test are that the series is the un-reduced form of numbers the gate already
trusts — so the two can never disagree — and that its sign carries the thing a scalar cannot: the
difference between not yet and too late.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.constraint_distance import (
    ConstraintDistance,
    constraint_distance,
    constraint_distance_from_payload,
    route_arclength,
)
from gear_sonic.dataset_generation.scene_route_check import (
    capsule_box_clearance,
    capsule_box_clearance_series,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule

#: A beam across the path, its underside at 1.30 m.
BOX = (2.0, -1.0, 1.30, 2.5, 1.0, 1.50)


def straight(frames: int = 100, *, x0: float = 0.0, x1: float = 5.0) -> np.ndarray:
    root = np.zeros((frames, 3))
    root[:, 0] = np.linspace(x0, x1, frames)
    root[:, 2] = 0.74
    return root


def one_capsule(root: np.ndarray, *, top: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A body of a single upright link whose capsule tops out at ``top`` metres."""
    frames = len(root)
    body_pos = root[:, None, :].copy()
    body_pos[:, 0, 2] = top
    body_quat = np.zeros((frames, 1, 4))
    body_quat[:, 0, 0] = 1.0
    return body_pos, body_quat, ["torso_link"]


#: One small spherical capsule at the torso's own origin, so the clearance the module reports is the
#: analytic distance from that point to the beam less the radius, and the expectations below can be
#: reasoned about rather than recorded.
#:
#: The radius is deliberately not zero. Penetration is expressed *through* it -- the point-to-box
#: distance of a point inside a box is zero, not negative -- so a zero-radius capsule can never
#: report an overlap, and a fixture built from one would quietly assert that nothing ever collides.
TORSO_RADIUS_M = 0.05
POINT_TORSO = {
    "torso_link": (
        CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=TORSO_RADIUS_M),
    )
}


# ---- the series is the un-reduced scalar -----------------------------------------------------


def test_the_series_minimum_is_the_scalar_the_gate_uses():
    """If these two ever disagree, an episode's record and its verdict are telling different
    stories about the same geometry."""
    rng = np.random.default_rng(0)
    starts = rng.uniform(-1.0, 3.0, size=(40, 5, 3))
    ends = starts + rng.uniform(-0.2, 0.2, size=(40, 5, 3))
    radii = rng.uniform(0.02, 0.10, size=5)
    scalar, frame, capsule = capsule_box_clearance(starts, ends, radii, BOX)
    series, binding = capsule_box_clearance_series(starts, ends, radii, BOX)
    assert series.min() == scalar
    assert int(np.argmin(series)) == frame
    assert int(binding[frame]) == capsule


def test_the_series_has_one_value_per_frame():
    rng = np.random.default_rng(1)
    starts = rng.uniform(-1.0, 3.0, size=(17, 4, 3))
    series, binding = capsule_box_clearance_series(starts, starts, rng.uniform(0.02, 0.1, 4), BOX)
    assert series.shape == (17,)
    assert binding.shape == (17,)
    assert binding.dtype == np.int64


def test_the_footprint_distance_survives_the_route_check():
    """The gate keeps the closest approach; the record needs the whole approach."""
    from gear_sonic.dataset_generation.scene_route_check import check_route_meets_obstacle

    check = check_route_meets_obstacle(straight()[:, :2], (2.0, -1.0, 2.5, 1.0))
    assert check.distance_m is not None
    assert len(check.distance_m) == 100
    assert check.distance_m.min() == pytest.approx(check.closest_approach_m)


def test_the_new_field_does_not_change_what_the_gate_compares():
    """`distance_m` is excluded from equality, so two checks that agreed before still agree."""
    from gear_sonic.dataset_generation.scene_route_check import check_route_meets_obstacle

    first = check_route_meets_obstacle(straight()[:, :2], (2.0, -1.0, 2.5, 1.0))
    second = check_route_meets_obstacle(straight()[:, :2], (2.0, -1.0, 2.5, 1.0))
    assert first == second


# ---- arclength and the station ----------------------------------------------------------------


def test_arclength_is_metres_not_frames():
    assert route_arclength(straight()[:, :2])[-1] == pytest.approx(5.0)


def test_the_remaining_distance_changes_sign_at_the_station():
    """Not yet and too late are different states. A model given only |distance| cannot tell a
    robot that has not reached the beam from one that has already passed under it."""
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert record.remaining_to_station_m[0] > 0
    assert record.remaining_to_station_m[-1] < 0
    assert record.remaining_to_station_m[record.station_frame] == pytest.approx(0.0, abs=1e-9)


def test_the_station_is_found_along_the_route_not_in_x():
    """On a route that turns, one x is reached twice and one arclength is reached once."""
    frames = 120
    root = np.zeros((frames, 3))
    root[:, 0] = 2.25 - 2.0 * np.cos(np.linspace(0.0, np.pi, frames))
    root[:, 1] = 2.0 * np.sin(np.linspace(0.0, np.pi, frames)) - 1.0
    root[:, 2] = 0.74
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert np.all(np.diff(record.remaining_to_station_m) <= 1e-12), "must fall monotonically"


def test_route_progress_runs_from_nought_to_one():
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert record.route_progress[0] == pytest.approx(0.0)
    assert record.route_progress[-1] == pytest.approx(1.0)
    assert np.all(np.diff(record.route_progress) >= -1e-12)


# ---- clearance ---------------------------------------------------------------------------------


def test_a_body_under_the_beam_never_overlaps_it():
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert record.first_overlap_frame is None
    assert record.body_clearance_m.min() > 0


def test_a_body_through_the_beam_reports_when_it_first_overlapped():
    """The frame is the point. A Boolean says the episode struck something; the frame says where
    along the approach the decision was already too late."""
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.35)  # inside the beam's z span
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert record.first_overlap_frame is not None
    assert record.body_clearance_m[record.first_overlap_frame] < 0
    assert record.body_clearance_m[record.first_overlap_frame - 1] >= 0


def test_the_bottleneck_is_where_the_obstacle_binds_hardest():
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert record.bottleneck_frame == int(np.argmin(record.body_clearance_m))


def test_the_binding_body_is_named_every_frame():
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert len(record.binding_body) == record.frames
    assert set(record.binding_body) == {"torso_link"}


# ---- time to the constraint ---------------------------------------------------------------------


def test_time_to_the_station_is_distance_over_speed():
    root = straight(frames=101, x0=0.0, x1=5.0)
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(
        root, body_pos, body_quat, names, BOX, fps=50.0, capsules=POINT_TORSO
    )
    speed = 5.0 / 100 * 50.0  # metres per second, constant on this path
    expected = record.remaining_to_station_m[0] / speed
    assert record.seconds_to_station[0] == pytest.approx(expected, rel=1e-6)


def test_a_stopped_robot_has_no_time_to_contact_rather_than_a_large_one():
    """A large finite number would be read as a safe one."""
    root = np.zeros((50, 3))
    root[:, 2] = 0.74
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    assert np.all(np.isnan(record.seconds_to_station))


# ---- refusing to guess ---------------------------------------------------------------------------


def test_an_episode_that_cannot_locate_its_body_is_an_error_not_a_zero():
    with pytest.raises(KeyError, match="cannot locate its own body"):
        constraint_distance_from_payload({"root_pos_w": straight()}, BOX)


def test_a_single_frame_has_no_constraint_distance():
    root = straight(frames=1)
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    with pytest.raises(ValueError, match="at least two frames"):
        constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)


def test_the_record_serialises_to_arrays_without_losing_the_absent_overlap():
    root = straight()
    body_pos, body_quat, names = one_capsule(root, top=1.20)
    record = constraint_distance(root, body_pos, body_quat, names, BOX, capsules=POINT_TORSO)
    arrays = record.as_arrays()
    assert isinstance(record, ConstraintDistance)
    assert int(arrays["first_overlap_frame"]) == -1, "None must round-trip as a sentinel, not 0"
    assert arrays["route_progress"].shape == (100,)
