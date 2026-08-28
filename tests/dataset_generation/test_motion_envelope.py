"""Tests for reducing a motion to the numbers that pair it with another."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.motion_envelope import (
    MIN_USEFUL_SPREAD_M,
    REGIME_FIELD,
    EnvelopeSignature,
    best_overhead_station,
    mine_pairs,
    score_pair,
    silhouette_at_stations,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule

TEST_CAPSULES = {
    "pelvis": (CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=0.10),),
    "torso_link": (CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.2), radius=0.10),),
}


def traverse(
    frames: int = 100,
    duck_from: int | None = None,
    duck_to: int | None = None,
    torso_z: float = 1.0,
    duck_z: float = 0.7,
) -> dict:
    """A body walking from x=0 to x=4, optionally ducking over a stretch of its route."""
    pos = np.zeros((frames, 2, 3))
    x = np.linspace(0.0, 4.0, frames)
    pos[:, 0, 0] = x
    pos[:, 0, 2] = 0.75
    pos[:, 1, 0] = x
    heights = np.full(frames, torso_z)
    if duck_from is not None:
        heights[duck_from:duck_to] = duck_z
    pos[:, 1, 2] = heights
    quat = np.zeros((frames, 2, 4))
    quat[:, :, 0] = 1.0
    return {
        "body_pos_w": pos,
        "body_quat_w": quat,
        "body_names": ["pelvis", "torso_link"],
        "root_pos_w": pos[:, 0, :],
        "root_quat_w": quat[:, 0, :],
        "fps": 50.0,
    }


def signature(**overrides) -> EnvelopeSignature:
    base = dict(
        episode_id="e",
        behaviour="walk",
        frames=100,
        duration_s=2.0,
        start_xy=(0.0, 0.0),
        goal_xy=(4.0, 0.0),
        path_length_m=4.0,
        net_displacement_m=4.0,
        heading_change_rad=0.0,
        mean_speed_mps=2.0,
        min_silhouette_peak_m=1.30,
        min_half_width_m=0.30,
        min_foot_apex_m=0.13,
    )
    base.update(overrides)
    return EnvelopeSignature(**base)


# ---- stations ----------------------------------------------------------------------------


def test_a_station_the_motion_never_reaches_is_nan_not_clear():
    """Scoring an unreachable station as clearance would invent windows out of absence."""
    peaks = silhouette_at_stations(traverse(), np.array([12.0]), capsules=TEST_CAPSULES)
    assert np.isnan(peaks[0])


def test_the_station_profile_finds_where_the_duck_happens():
    payload = traverse(duck_from=50, duck_to=70)  # x roughly 2.0 to 2.8
    stations = np.array([0.5, 2.4, 3.8])
    peaks = silhouette_at_stations(payload, stations, span=0.2, capsules=TEST_CAPSULES)
    assert peaks[1] < peaks[0]
    assert peaks[1] < peaks[2]


def test_the_best_station_is_where_the_motions_differ_not_the_midpoint():
    """The whole point of scanning. On the real family the midpoint gave a 0.053 m window
    where the duck's own station offered 0.18 m."""
    nominal = traverse()
    adapted = traverse(duck_from=70, duck_to=90)  # ducks late, past the midpoint
    station, spread = best_overhead_station(
        nominal, adapted, span=0.2, resolution_m=0.05, capsules=TEST_CAPSULES
    )
    assert spread > 0.25
    assert station > 2.4, "the midpoint is x=2.0; the duck is later than that"


def test_two_identical_motions_have_no_station_that_separates_them():
    _, spread = best_overhead_station(traverse(), traverse(), span=0.2, capsules=TEST_CAPSULES)
    assert spread == pytest.approx(0.0, abs=1e-9)


def test_motions_whose_routes_do_not_overlap_yield_no_station():
    far = traverse()
    far["root_pos_w"] = far["root_pos_w"] + np.array([100.0, 0.0, 0.0])
    station, spread = best_overhead_station(traverse(), far)
    assert np.isnan(station)
    assert spread == 0.0


# ---- pairing -----------------------------------------------------------------------------


def test_a_pair_going_to_different_places_is_refused_however_wide_the_gap():
    """A duck walking 2 m and a walk turning through 5 m differ in head height, but a shelf
    separating them says the two motions went to different places, not that geometry chose
    the behaviour."""
    nominal = signature(episode_id="a")
    adapted = signature(
        episode_id="b",
        min_silhouette_peak_m=0.9,
        goal_xy=(1.0, 6.0),
        path_length_m=9.0,
        duration_s=5.0,
    )
    candidate = score_pair(nominal, adapted, "overhead")
    assert not candidate.compatible
    assert any("goals" in reason for reason in candidate.reasons)


def test_a_compatible_pair_with_a_real_gap_is_accepted():
    candidate = score_pair(
        signature(episode_id="a"),
        signature(episode_id="b", min_silhouette_peak_m=1.10),
        "overhead",
    )
    assert candidate.compatible
    assert candidate.spread_m == pytest.approx(0.20)


def test_a_gap_too_small_to_place_an_obstacle_in_is_refused():
    candidate = score_pair(
        signature(episode_id="a"),
        signature(episode_id="b", min_silhouette_peak_m=1.30 - MIN_USEFUL_SPREAD_M / 2),
        "overhead",
    )
    assert not candidate.compatible


def test_order_matters_because_only_one_direction_can_be_right():
    """Swapping nominal and adapted proposes that the *taller* motion is the adapted one,
    which is not a family."""
    low = signature(episode_id="low", min_silhouette_peak_m=1.05)
    high = signature(episode_id="high", min_silhouette_peak_m=1.30)
    assert score_pair(high, low, "overhead").compatible
    assert not score_pair(low, high, "overhead").compatible


def test_the_floor_regime_wants_the_adapted_motion_to_lift_higher():
    """Overhead and lateral want a smaller number; floor wants a larger one. Getting the
    sign wrong would propose exactly the pairs that cannot work."""
    nominal = signature(episode_id="a", min_foot_apex_m=0.13)
    adapted = signature(episode_id="b", min_foot_apex_m=0.30)
    assert score_pair(nominal, adapted, "floor").compatible
    assert not score_pair(adapted, nominal, "floor").compatible


def test_an_unmeasurable_field_is_refused_rather_than_scored():
    candidate = score_pair(
        signature(episode_id="a"),
        signature(episode_id="b", min_foot_apex_m=float("nan")),
        "floor",
    )
    assert not candidate.compatible
    assert any("not measurable" in reason for reason in candidate.reasons)


def test_an_unknown_regime_is_an_error():
    with pytest.raises(ValueError, match="unknown regime"):
        score_pair(signature(), signature(episode_id="b"), "ceiling")


def test_mining_ranks_the_widest_compatible_window_first():
    signatures = [
        signature(episode_id="tall", min_silhouette_peak_m=1.30),
        signature(episode_id="mid", min_silhouette_peak_m=1.20),
        signature(episode_id="low", min_silhouette_peak_m=1.00),
    ]
    ranked = mine_pairs(signatures, "overhead")
    assert ranked[0].nominal == "tall" and ranked[0].adapted == "low"
    assert all(candidate.compatible for candidate in ranked)


def test_mining_never_pairs_a_motion_with_itself():
    signatures = [
        signature(episode_id=f"e{i}", min_silhouette_peak_m=1.3 - 0.1 * i) for i in range(4)
    ]
    for candidate in mine_pairs(signatures, "overhead"):
        assert candidate.nominal != candidate.adapted


def test_the_regimes_carry_a_direction_each():
    for field, direction in REGIME_FIELD.values():
        assert field.startswith("min_")
        assert direction in (-1, +1)


# ---- signed, one-sided envelopes -----------------------------------------------------------

#: Capsules for the two-armed fixture below, keyed by the links it actually records.
LATERAL_CAPSULES = {
    "pelvis": (CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=0.10),),
    "left_wrist_yaw_link": (
        CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=0.04),
    ),
    "right_wrist_yaw_link": (
        CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=0.04),
    ),
}


def leaning(frames: int = 60, left: float = 0.30, right: float = 0.10) -> dict:
    """A body whose two sides are deliberately different widths."""
    payload = traverse(frames)
    payload["body_names"] = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    bodies = np.zeros((frames, 3, 3))
    bodies[:, :, 0] = payload["root_pos_w"][:, [0]]
    bodies[:, 1, 1] = left
    bodies[:, 2, 1] = -right
    payload["body_pos_w"] = bodies
    payload["body_quat_w"] = np.tile(np.array([1.0, 0, 0, 0]), (frames, 3, 1))
    return payload


def test_the_two_sides_are_reported_separately():
    """The whole point: a symmetric maximum reports the wide side for both."""
    from gear_sonic.dataset_generation.motion_envelope import signed_half_widths

    left, right = signed_half_widths(leaning(), capsules=LATERAL_CAPSULES)
    assert left.max() > right.max()


def test_both_sides_are_positive_distances():
    from gear_sonic.dataset_generation.motion_envelope import signed_half_widths

    left, right = signed_half_widths(leaning(), capsules=LATERAL_CAPSULES)
    assert (left > 0).all() and (right > 0).all()


def test_a_one_sided_search_returns_which_side_to_put_the_obstacle_on():
    from gear_sonic.dataset_generation.motion_envelope import best_one_sided_station

    nominal = leaning(left=0.35, right=0.35)
    adapted = leaning(left=0.35, right=0.15)  # narrowed on the right only
    station, side, window = best_one_sided_station(
        nominal, adapted, span=0.2, capsules=LATERAL_CAPSULES
    )
    assert side == "right"
    assert window > 0.1


def test_a_one_sided_reduction_is_invisible_to_the_symmetric_measure():
    """Why the signed version exists. Narrowing one side leaves the symmetric maximum
    untouched, so a real and usable reduction reports as zero."""
    from gear_sonic.dataset_generation.motion_envelope import (
        best_lateral_station,
        best_one_sided_station,
    )

    nominal = leaning(left=0.35, right=0.35)
    adapted = leaning(left=0.35, right=0.15)
    _, symmetric = best_lateral_station(nominal, adapted, span=0.2, capsules=LATERAL_CAPSULES)
    _, _, one_sided = best_one_sided_station(nominal, adapted, span=0.2, capsules=LATERAL_CAPSULES)
    assert symmetric == pytest.approx(0.0, abs=1e-6)
    assert one_sided > 0.15


def test_routes_that_do_not_overlap_yield_no_station():
    from gear_sonic.dataset_generation.motion_envelope import best_one_sided_station

    far = leaning()
    far["root_pos_w"] = far["root_pos_w"] + np.array([100.0, 0.0, 0.0])
    station, side, window = best_one_sided_station(leaning(), far, capsules=LATERAL_CAPSULES)
    assert np.isnan(station) and side == "" and window == 0.0


def test_width_splits_into_what_an_arm_tuck_can_move_and_what_it_cannot():
    """An arm tuck narrows the robot only where the arms are what makes it widest.

    Where a hip, knee or ankle is already as wide, tucking changes the silhouette not at all,
    and that is a property of the clip and the station rather than of operator strength. It
    should make the generator refuse rather than be discovered after four rollouts.
    """
    from gear_sonic.dataset_generation.motion_envelope import width_decomposition

    payload = leaning(left=0.35, right=0.10)
    payload["body_names"] = ["pelvis", "left_wrist_yaw_link", "left_ankle_roll_link"]
    capsules = {
        "pelvis": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.05),),
        "left_wrist_yaw_link": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.04),),
        "left_ankle_roll_link": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.04),),
    }
    station = float(np.median(payload["root_pos_w"][:, 0]))
    result = width_decomposition(payload, station, "left", span=1.0, capsules=capsules)
    assert set(result) >= {
        "arm_width_m",
        "nonarm_floor_m",
        "available_reduction_m",
        "critical_capsule",
        "arms_are_widest",
    }
    assert result["available_reduction_m"] >= 0.0


def test_a_side_step_is_a_genuine_refusal_case():
    """Measured on the corpus: a side-step's ankle reaches 0.451 m while its arms sit at
    0.340 m, so the available reduction is exactly zero. That is what a refusal control looks
    like, as opposed to a clip that merely scored low under a metric fixed in advance."""
    from gear_sonic.dataset_generation.motion_envelope import width_decomposition

    payload = leaning(left=0.20, right=0.05)
    payload["body_names"] = ["pelvis", "left_wrist_yaw_link", "left_ankle_roll_link"]
    payload["body_pos_w"][:, 2, 1] = 0.45  # the ankle, far outside the arm
    capsules = {
        "pelvis": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.05),),
        "left_wrist_yaw_link": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.04),),
        "left_ankle_roll_link": (CollisionCapsule(start=(0, 0, 0), end=(0, 0, 0), radius=0.04),),
    }
    station = float(np.median(payload["root_pos_w"][:, 0]))
    result = width_decomposition(payload, station, "left", span=1.0, capsules=capsules)
    assert not result["arms_are_widest"]
    assert result["available_reduction_m"] == 0.0
    assert "ankle" in result["critical_capsule"]


def test_an_unknown_side_is_refused():
    from gear_sonic.dataset_generation.motion_envelope import width_decomposition

    with pytest.raises(ValueError, match="side must be"):
        width_decomposition(leaning(), 1.0, "up", capsules=LATERAL_CAPSULES)
