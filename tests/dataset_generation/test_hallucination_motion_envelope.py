from __future__ import annotations

import numpy as np

from gear_sonic.dataset_generation.hallucination.keypoints import SemanticCapsuleTracks
from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope


def tracks_along_route(root_xy: np.ndarray) -> SemanticCapsuleTracks:
    frames = len(root_xy)
    starts = np.zeros((frames, 1, 3), dtype=np.float64)
    starts[:, 0, :2] = root_xy
    starts[:, 0, 2] = 1.0
    ends = starts.copy()
    return SemanticCapsuleTracks(
        starts=starts,
        ends=ends,
        radii=np.asarray([0.10]),
        owners=("torso_link",),
        groups=("head_torso",),
        root_pos_w=np.column_stack((root_xy, np.zeros(frames))),
        root_quat_w=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
    )


def test_curved_route_does_not_alias_distant_frames_into_station_width() -> None:
    # The two horizontal branches share an x projection but are three metres apart. A tangent-
    # slab filter over every frame aliases the far branch into the first and reports ~3 m width.
    lower = np.column_stack((np.linspace(0.0, 2.0, 21), np.zeros(21)))
    connector = np.column_stack((np.full(31, 2.0), np.linspace(0.0, 3.0, 31)))[1:]
    upper = np.column_stack((np.linspace(2.0, 0.0, 21), np.full(21, 3.0)))[1:]
    root = np.concatenate((lower, connector, upper), axis=0)

    envelope = extract_envelope(
        tracks_along_route(root),
        "u_route",
        fractions=np.asarray([0.10]),
        along_route_m=0.20,
    )

    assert envelope.left_m[0] < 0.25
    assert envelope.right_m[0] < 0.25
    assert envelope.up_m[0] == 1.10


def test_straight_route_width_includes_capsule_radius() -> None:
    root = np.column_stack((np.linspace(0.0, 2.0, 21), np.zeros(21)))
    envelope = extract_envelope(
        tracks_along_route(root),
        "straight",
        fractions=np.asarray([0.50]),
        along_route_m=0.20,
    )

    assert envelope.left_m[0] == 0.10
    assert envelope.right_m[0] == 0.10
