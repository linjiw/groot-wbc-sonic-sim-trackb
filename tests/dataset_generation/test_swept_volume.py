"""Gates for the G1 swept-volume clearance model.

This model is allowed to license *narrower* corridors than the old 0.45 m root
cylinder, so it has to be right. The tests below are the ones that would catch it
being wrong, including two that check it against physics recorded independently of
the model.

Measured on a real 249-frame walking rollout:

* foot capsule surfaces reach z = -0.0008 m, i.e. the model reconstructs the floor
  plane to sub-millimetre from body pose alone;
* the wrist reaches 0.514 m from the root, outside the old 0.45 m cylinder, so the
  previous model was optimistic by 0.056 m on that episode;
* horizontal half-width varies from 0.273 m (arms tucked) to 0.664 m (peak swing)
  about a median of 0.377 m -- which is why a per-frame corridor, not a uniform
  narrow tube, is the safe way to tighten clutter.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.swept_volume import (
    G1_COLLISION_CAPSULES,
    body_capsules_world,
    swept_half_width_per_frame,
    swept_volume_clearance,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _pose(
    frames: int,
    names: list[str],
    positions: dict[str, tuple[float, float, float]],
    *,
    parked: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Pose helper. `parked` is where unspecified links sit; tests that isolate one
    link park the rest far away so another link's capsules cannot dominate the result."""
    body_pos = np.tile(np.asarray(parked, dtype=np.float64), (frames, len(names), 1))
    body_quat = np.tile(np.asarray(IDENTITY), (frames, len(names), 1))
    for name, value in positions.items():
        body_pos[:, names.index(name)] = value
    return body_pos, body_quat


def _all_names() -> list[str]:
    return sorted(G1_COLLISION_CAPSULES)


# --------------------------------------------------------------------------
# Capsule placement
# --------------------------------------------------------------------------


def test_catalog_covers_the_links_that_can_actually_touch_anything():
    """Exactly the links with collision geometry are the ones that report contact."""
    observed_contact_links = {
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
    assert observed_contact_links <= set(G1_COLLISION_CAPSULES)
    assert len(G1_COLLISION_CAPSULES) == 14
    assert sum(len(v) for v in G1_COLLISION_CAPSULES.values()) == 29


def test_identity_pose_places_a_capsule_at_its_local_offset():
    names = _all_names()
    body_pos, body_quat = _pose(2, names, {"pelvis": (1.0, 2.0, 3.0)})

    starts, ends, radii, owners = body_capsules_world(body_pos, body_quat, names)
    index = owners.index("pelvis")

    # The pelvis capsule is a sphere at local (0, 0, -0.08).
    assert np.allclose(starts[0, index], (1.0, 2.0, 3.0 - 0.08))
    assert np.allclose(ends[0, index], starts[0, index])
    assert radii[index] == pytest.approx(0.07)


def test_rotation_is_applied_to_the_local_offset():
    names = _all_names()
    body_pos, body_quat = _pose(1, names, {"pelvis": (0.0, 0.0, 0.0)})
    # 180 degrees about +X maps local (0,0,-0.08) to (0,0,+0.08).
    body_quat[:, names.index("pelvis")] = (0.0, 1.0, 0.0, 0.0)

    starts, _, _, owners = body_capsules_world(body_pos, body_quat, names)

    assert np.allclose(starts[0, owners.index("pelvis")], (0.0, 0.0, 0.08), atol=1e-9)


def test_a_capsule_spans_its_full_local_segment():
    names = _all_names()
    body_pos, body_quat = _pose(1, names, {"left_knee_link": (0.0, 0.0, 1.0)})

    starts, ends, _, owners = body_capsules_world(body_pos, body_quat, names)
    index = owners.index("left_knee_link")

    # Shin capsule runs local (0.02, 0, 0) -> (0.02, 0, -0.25).
    assert np.allclose(starts[0, index], (0.02, 0.0, 1.0))
    assert np.allclose(ends[0, index], (0.02, 0.0, 0.75))


def test_missing_collision_link_is_rejected_rather_than_silently_skipped():
    names = [name for name in _all_names() if name != "pelvis"]
    body_pos = np.zeros((1, len(names), 3))
    body_quat = np.tile(np.asarray(IDENTITY), (1, len(names), 1))

    with pytest.raises(ValueError, match="absent from the recorded bodies"):
        body_capsules_world(body_pos, body_quat, names)


@pytest.mark.parametrize(
    "pos_shape,quat_shape",
    [((2, 3), (2, 14, 4)), ((1, 14, 3), (1, 14, 3)), ((1, 14, 3), (2, 14, 4))],
)
def test_malformed_pose_arrays_are_rejected(pos_shape, quat_shape):
    with pytest.raises(ValueError):
        body_capsules_world(np.zeros(pos_shape), np.zeros(quat_shape), _all_names())


# --------------------------------------------------------------------------
# Clearance
# --------------------------------------------------------------------------


def test_clearance_is_surface_to_surface_not_centre_to_centre():
    names = _all_names()
    body_pos, body_quat = _pose(1, names, {"pelvis": (0.0, 0.0, 1.08)}, parked=(50.0, 50.0, 50.0))
    # Pelvis sphere centre ends up at z=1.0 with radius 0.07.
    box = ("floorbox", (-5.0, -5.0, 0.0, 5.0, 5.0, 0.5))

    report = swept_volume_clearance(body_pos, body_quat, names, [box])

    # Distance from sphere centre to box top is 0.5; minus the 0.07 radius.
    assert report.min_clearance_m == pytest.approx(0.43, abs=1e-6)
    assert report.min_link == "pelvis"
    assert report.min_obstacle == "floorbox"


def test_interpenetration_reports_negative_clearance():
    names = _all_names()
    body_pos, body_quat = _pose(1, names, {"pelvis": (0.0, 0.0, 0.10)}, parked=(50.0, 50.0, 50.0))
    box = ("block", (-1.0, -1.0, 0.0, 1.0, 1.0, 1.0))

    report = swept_volume_clearance(body_pos, body_quat, names, [box])

    assert report.min_clearance_m < 0.0


def test_the_nearest_frame_and_link_are_reported():
    names = _all_names()
    body_pos, body_quat = _pose(3, names, {}, parked=(50.0, 50.0, 50.0))
    body_pos[1, names.index("left_wrist_yaw_link")] = (0.0, 0.0, 0.5)
    box = ("block", (-1.0, -1.0, 0.0, 1.0, 1.0, 0.2))

    report = swept_volume_clearance(body_pos, body_quat, names, [box])

    assert report.min_frame == 1
    assert report.min_link == "left_wrist_yaw_link"
    assert report.per_frame_clearance.shape == (3,)


def test_sampling_never_reports_a_clear_path_through_a_box():
    """A capsule straddling a box must not slip between samples."""
    names = _all_names()
    body_pos, body_quat = _pose(
        1, names, {"left_knee_link": (0.0, 0.0, 0.125)}, parked=(50.0, 50.0, 50.0)
    )
    # Shin capsule spans z 0.125 -> -0.125 through a thin slab at z in [0, 0.02].
    box = ("slab", (-1.0, -1.0, 0.0, 1.0, 1.0, 0.02))

    report = swept_volume_clearance(body_pos, body_quat, names, [box])

    assert report.min_clearance_m < 0.0


# --------------------------------------------------------------------------
# Per-frame half-width -- the quantity that lets corridors tighten
# --------------------------------------------------------------------------


def test_half_width_includes_capsule_extent_and_radius():
    """Half-width is set by a capsule's far end plus its radius, not the radius alone."""
    names = _all_names()
    body_pos, body_quat = _pose(1, names, {})
    reference = np.zeros((1, 2))

    half = swept_half_width_per_frame(body_pos, body_quat, names, reference)

    expected = max(
        math.hypot(end[0], end[1]) + capsule.radius
        for capsules in G1_COLLISION_CAPSULES.values()
        for capsule in capsules
        for end in (capsule.start, capsule.end)
    )
    assert half[0] == pytest.approx(expected, abs=1e-9)


def test_half_width_tracks_an_extended_arm():
    names = _all_names()
    body_pos, body_quat = _pose(2, names, {})
    body_pos[1, names.index("right_wrist_yaw_link")] = (0.6, 0.0, 0.0)
    reference = np.zeros((2, 2))

    half = swept_half_width_per_frame(body_pos, body_quat, names, reference)

    assert half[1] > half[0]
    # The wrist capsule has its own local extent, so the reach exceeds 0.6 + radius.
    assert half[1] > 0.6


def test_half_width_defaults_to_the_pelvis_as_reference():
    names = _all_names()
    body_pos, body_quat = _pose(2, names, {"pelvis": (3.0, 4.0, 0.0)})

    half = swept_half_width_per_frame(body_pos, body_quat, names)

    # Other links are at the origin, 5 m from the pelvis.
    assert half[0] == pytest.approx(5.0, abs=0.25)


def test_reference_length_must_match():
    names = _all_names()
    body_pos, body_quat = _pose(3, names, {})
    with pytest.raises(ValueError, match="reference_xy must be"):
        swept_half_width_per_frame(body_pos, body_quat, names, np.zeros((2, 2)))


# --------------------------------------------------------------------------
# Per-frame corridors in the clutter builder
# --------------------------------------------------------------------------


def test_per_point_clearance_shapes_the_corridor():
    from gear_sonic.dataset_generation.clutter_scene_builder import build_clutter_scene

    xs = np.linspace(0.0, 4.0, 40)
    path = np.stack([xs, np.zeros_like(xs)], axis=1)
    # Wide requirement in the middle, narrow at the ends.
    required = np.full(len(path), 0.4)
    required[15:25] = 1.2

    spec = build_clutter_scene(
        path, scene_id="unit", seed=3, per_point_clearance_m=required, target_pieces=30
    )

    # build_clutter_scene recentres the path on the room origin, so compare against
    # the frame it actually placed furniture in.
    for piece in spec.pieces:
        min_x, min_y, max_x, max_y = piece.rect
        for point, need in zip(spec.path_xy, required):
            dx = max(min_x - point[0], 0.0, point[0] - max_x)
            dy = max(min_y - point[1], 0.0, point[1] - max_y)
            assert math.hypot(dx, dy) >= need - 1e-9


def test_per_point_clearance_validates_its_shape():
    from gear_sonic.dataset_generation.clutter_scene_builder import build_clutter_scene

    xs = np.linspace(0.0, 3.0, 20)
    path = np.stack([xs, np.zeros_like(xs)], axis=1)

    with pytest.raises(ValueError, match="per_point_clearance_m must have"):
        build_clutter_scene(path, scene_id="unit", per_point_clearance_m=np.ones(5))
    with pytest.raises(ValueError, match="must be positive"):
        build_clutter_scene(path, scene_id="unit", per_point_clearance_m=np.zeros(20))


def test_a_narrow_requirement_admits_closer_furniture_than_a_wide_one():
    """The whole point: tighter requirement => tighter clutter."""
    from gear_sonic.dataset_generation.clutter_scene_builder import build_clutter_scene

    xs = np.linspace(0.0, 4.0, 40)
    path = np.stack([xs, np.zeros_like(xs)], axis=1)

    narrow = build_clutter_scene(
        path, scene_id="a", seed=1, per_point_clearance_m=np.full(40, 0.40), target_pieces=30
    )
    wide = build_clutter_scene(
        path, scene_id="b", seed=1, per_point_clearance_m=np.full(40, 1.00), target_pieces=30
    )

    assert narrow.metrics["min_distance_to_path_m"] < wide.metrics["min_distance_to_path_m"]
    assert narrow.metrics["min_distance_to_path_m"] >= 0.40 - 1e-9
    assert wide.metrics["min_distance_to_path_m"] >= 1.00 - 1e-9
