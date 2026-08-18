"""Tests for asking whether the robot did the behaviour its label claims."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.behaviour_predicates import (
    PREDICATES,
    PredicateError,
    check_behaviour,
    check_duck_under,
    check_narrow_pass,
    check_pause,
    check_side_step,
    check_stand_to_walk,
    check_step_over,
    check_turn_in_place,
    check_walk_to_stop,
    coverage,
)

FPS = 50.0


def episode(
    xy: np.ndarray,
    heading: np.ndarray | None = None,
    torso_z: np.ndarray | None = None,
) -> dict:
    frames = len(xy)
    root = np.zeros((frames, 3))
    root[:, :2] = xy
    root[:, 2] = 0.75
    yaw = np.zeros(frames) if heading is None else heading
    quat = np.zeros((frames, 4))
    quat[:, 0] = np.cos(yaw / 2)
    quat[:, 3] = np.sin(yaw / 2)
    payload = {"root_pos_w": root, "root_quat_w": quat, "fps": FPS, "total_frames": frames}
    if torso_z is not None:
        payload["body_names"] = ["pelvis", "torso_link"]
        bodies = np.zeros((frames, 2, 3))
        bodies[:, 1, 2] = torso_z
        payload["body_pos_w"] = bodies
    return payload


def walk(frames: int, speed: float = 0.8) -> np.ndarray:
    xy = np.zeros((frames, 2))
    xy[:, 0] = np.arange(frames) * speed / FPS
    return xy


# ---- pause -------------------------------------------------------------------------------

def test_a_walk_with_a_real_stop_satisfies_pause():
    xy = np.concatenate([walk(60), np.tile(walk(1)[-1] + walk(60)[-1], (40, 1)), walk(60) + 1.0])
    assert check_pause(episode(xy)).satisfied


def test_a_continuous_walk_does_not_satisfy_pause():
    """The pause is the point; a smooth walk is a mislabel, not a near miss."""
    result = check_pause(episode(walk(200)))
    assert not result.satisfied
    assert "longest stop" in result.reason


def test_stopping_and_never_resuming_is_not_a_pause():
    xy = np.concatenate([walk(60), np.tile(walk(60)[-1], (140, 1))])
    result = check_pause(episode(xy))
    assert not result.satisfied
    assert "never resumed" in result.reason


# ---- turn in place -----------------------------------------------------------------------

def test_turning_without_translating_satisfies_turn_in_place():
    frames = 120
    heading = np.linspace(0.0, math.pi, frames)
    assert check_turn_in_place(episode(np.zeros((frames, 2)), heading)).satisfied


def test_turning_while_walking_away_is_not_turn_in_place():
    """Root sliding across the floor is the failure the review card names."""
    frames = 120
    heading = np.linspace(0.0, math.pi, frames)
    result = check_turn_in_place(episode(walk(frames, speed=1.5), heading))
    assert not result.satisfied
    assert "translated" in result.reason


def test_barely_turning_is_not_a_turn():
    frames = 120
    heading = np.linspace(0.0, 0.2, frames)
    result = check_turn_in_place(episode(np.zeros((frames, 2)), heading))
    assert not result.satisfied
    assert "heading change" in result.reason


# ---- side step ---------------------------------------------------------------------------

def test_moving_sideways_while_facing_forward_satisfies_side_step():
    frames = 120
    xy = np.zeros((frames, 2))
    xy[:, 1] = np.arange(frames) * 0.6 / FPS      # travel along +y
    assert check_side_step(episode(xy)).satisfied  # heading stays at 0, i.e. facing +x


def test_turning_and_walking_does_not_pass_as_a_side_step():
    """The disguise the card warns about: turn to face the target, then walk at it.

    The robot ends up beside where it started either way, so displacement alone cannot
    tell the two apart -- only the heading can.
    """
    frames = 160
    heading = np.concatenate([np.linspace(0.0, math.pi / 2, 40), np.full(120, math.pi / 2)])
    xy = np.zeros((frames, 2))
    xy[40:, 1] = np.arange(120) * 0.6 / FPS        # walks along +y after turning to face it
    result = check_side_step(episode(xy, heading))
    assert not result.satisfied
    assert "turn, not a side-step" in result.reason


def test_a_forward_walk_in_a_rotated_frame_is_still_a_forward_walk():
    """Starting already facing +y and walking +y is not a side-step either, and the
    predicate says so on the lateral-travel grounds rather than the heading ones."""
    frames = 120
    xy = np.zeros((frames, 2))
    xy[:, 1] = np.arange(frames) * 0.6 / FPS
    result = check_side_step(episode(xy, np.full(frames, math.pi / 2)))
    assert not result.satisfied
    assert "lateral" in result.reason


def test_walking_forward_is_not_a_side_step():
    result = check_side_step(episode(walk(120)))
    assert not result.satisfied
    assert "lateral" in result.reason


def test_a_stationary_episode_cannot_be_assessed_as_a_side_step():
    with pytest.raises(PredicateError, match="did not move"):
        check_side_step(episode(np.zeros((60, 2))))


# ---- duck under --------------------------------------------------------------------------

def test_a_torso_that_dips_and_recovers_satisfies_duck_under():
    frames = 200
    torso = np.full(frames, 1.10)
    torso[80:120] = 0.90
    assert check_duck_under(episode(walk(frames), torso_z=torso)).satisfied


def test_a_torso_held_high_is_not_a_duck():
    frames = 200
    result = check_duck_under(episode(walk(frames), torso_z=np.full(frames, 1.10)))
    assert not result.satisfied
    assert "dropped only" in result.reason


def test_ducking_without_recovering_fails():
    frames = 200
    torso = np.full(frames, 1.10)
    torso[100:] = 0.90
    result = check_duck_under(episode(walk(frames), torso_z=torso))
    assert not result.satisfied
    assert "never came back up" in result.reason


def test_a_known_shelf_height_turns_the_duck_into_a_clearance_check():
    """Without the shelf the predicate can only say a duck happened, and it says which."""
    frames = 200
    torso = np.full(frames, 1.10)
    torso[80:120] = 0.98
    shallow = check_duck_under(episode(walk(frames), torso_z=torso), shelf_z=0.95)
    assert not shallow.satisfied
    assert "never got below the shelf" in shallow.reason
    assert check_duck_under(episode(walk(frames), torso_z=torso), shelf_z=1.05).satisfied


# ---- start and stop ----------------------------------------------------------------------

def test_walking_then_holding_satisfies_walk_to_stop():
    xy = np.concatenate([walk(120), np.tile(walk(120)[-1], (80, 1))])
    result = check_walk_to_stop(episode(xy))
    assert result.satisfied
    assert result.provisional, "no reviewed sample has calibrated this threshold yet"


def test_still_moving_at_the_end_is_not_a_stop():
    result = check_walk_to_stop(episode(walk(200)))
    assert not result.satisfied
    assert "still moving" in result.reason


def test_starting_from_rest_satisfies_stand_to_walk():
    xy = np.concatenate([np.zeros((40, 2)), walk(160)])
    assert check_stand_to_walk(episode(xy)).satisfied


def test_already_moving_is_not_a_start_from_standing():
    result = check_stand_to_walk(episode(walk(200)))
    assert not result.satisfied
    assert "of its peak speed" in result.reason


def test_a_slow_start_is_judged_against_its_own_peak_not_an_absolute_speed():
    """Five accepted starts opened at 0.13-0.16 m/s and went on to 1.25-1.79 m/s.

    An absolute 0.12 m/s threshold false-rejected every one of them while the behaviour was
    plainly present, which is why the criterion is a fraction of the episode's own peak.
    """
    xy = np.concatenate([walk(20, speed=0.15), walk(180, speed=1.6) + 0.06])
    result = check_stand_to_walk(episode(xy))
    assert result.satisfied
    assert result.measurements["opening_fraction"] < 0.25


def test_walk_look_has_no_predicate_yet_rather_than_the_wrong_one():
    """It was wired to the pause check, which called all four accepted episodes
    mislabelled -- but "looks around" is a head yaw excursion while the root keeps going,
    and root speed cannot see it. A wrong predicate manufactures findings."""
    assert check_behaviour("walk_look", episode(walk(120))) is None


# ---- dispatch ----------------------------------------------------------------------------

def test_a_behaviour_without_a_predicate_returns_none_not_a_pass():
    """"Nobody wrote the check" and "checked and correct" are different facts."""
    assert check_behaviour("carry_walk", episode(walk(120))) is None


def test_dispatch_runs_the_right_predicate():
    result = check_behaviour("side_step", episode(walk(120)))
    assert result is not None and result.behaviour == "side_step"


def test_coverage_reports_which_behaviours_are_still_on_the_honour_system():
    reported = coverage(["side_step", "carry_walk", "duck_under"])
    assert reported == {"side_step": True, "carry_walk": False, "duck_under": True}


def test_every_registered_predicate_is_callable():
    for behaviour, predicate in PREDICATES.items():
        assert callable(predicate), behaviour


# ---- step over ---------------------------------------------------------------------------

def stepping(frames: int, left_apex: float, right_apex: float) -> dict:
    """Two feet, each swinging once to its own apex, while the body walks forward."""
    payload = episode(walk(frames))
    payload["body_names"] = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
    bodies = np.zeros((frames, 3, 3))
    bodies[:, :, 0] = payload["root_pos_w"][:, [0]]
    swing = np.sin(np.linspace(0, math.pi, frames)) ** 2
    bodies[:, 1, 2] = 0.035 + left_apex * swing
    bodies[:, 2, 2] = 0.035 + right_apex * swing
    payload["body_pos_w"] = bodies
    return payload


def test_both_feet_lifting_high_satisfies_step_over():
    assert check_step_over(stepping(120, 0.25, 0.24)).satisfied


def test_a_dragged_trailing_foot_fails_even_when_the_leading_one_clears():
    """The review card's exact warning: watch the trailing foot, not the leading one.

    Checking the maximum across both feet would pass this, because the leading foot's apex
    is the maximum.
    """
    result = check_step_over(stepping(120, 0.30, 0.06))
    assert not result.satisfied
    assert "only rose" in result.reason
    assert result.measurements["apex_left_ankle_roll_link_m"] > 0.25


def test_a_known_obstacle_turns_the_apex_into_a_clearance_test():
    payload = stepping(120, 0.30, 0.28)
    high = check_step_over(payload, obstacle_x=0.96, obstacle_top_z=0.20)
    assert high.satisfied
    assert "clearance_left_ankle_roll_link_m" in high.measurements
    low = check_step_over(payload, obstacle_x=0.96, obstacle_top_z=0.60)
    assert not low.satisfied
    assert "below the obstacle top" in low.reason


def test_a_recording_without_two_feet_cannot_be_assessed():
    payload = episode(walk(60))
    payload["body_names"] = ["pelvis"]
    payload["body_pos_w"] = np.zeros((60, 1, 3))
    with pytest.raises(PredicateError, match="ankle_roll"):
        check_step_over(payload)


# ---- narrow pass -------------------------------------------------------------------------

def widening(frames: int, widths: np.ndarray) -> dict:
    """A body walking forward with two side links at a controllable half-width."""
    payload = episode(walk(frames))
    payload["body_names"] = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    bodies = np.zeros((frames, 3, 3))
    bodies[:, 0, 0] = payload["root_pos_w"][:, 0]
    bodies[:, 1, 0] = payload["root_pos_w"][:, 0]
    bodies[:, 2, 0] = payload["root_pos_w"][:, 0]
    bodies[:, 1, 1] = widths
    bodies[:, 2, 1] = -widths
    payload["body_pos_w"] = bodies
    return payload


def tucking(frames: int = 120, walking: float = 0.55, tucked: float = 0.26) -> dict:
    widths = np.full(frames, walking)
    widths[frames // 3 : 2 * frames // 3] = tucked
    return widening(frames, widths)


def test_a_tuck_that_opens_out_again_satisfies_narrow_pass():
    assert check_narrow_pass(tucking()).satisfied


def test_a_constant_width_walk_does_not_narrow():
    result = check_narrow_pass(widening(120, np.full(120, 0.55)))
    assert not result.satisfied
    assert "fell only" in result.reason


def test_narrowing_without_opening_out_again_fails():
    widths = np.full(120, 0.55)
    widths[60:] = 0.26
    result = check_narrow_pass(widening(120, widths))
    assert not result.satisfied
    assert "never opened out again" in result.reason


def test_a_side_step_is_not_a_narrowing_which_is_the_whole_point():
    """Translating sideways leaves the width untouched.

    This is why side_step cannot serve the lateral geometry regime, and why arm_tuck and
    shoulder_turn had to be added: mining the corpus refused every side_step/walk pair, and
    60 of those refusals were that the half-width simply never differed.
    """
    frames = 120
    payload = widening(frames, np.full(frames, 0.55))
    payload["root_pos_w"][:, 1] = np.arange(frames) * 0.6 / FPS     # travel along +y
    payload["body_pos_w"][:, :, 1] += payload["root_pos_w"][:, None, 1]
    assert not check_narrow_pass(payload).satisfied


def test_a_known_gap_turns_the_tuck_into_a_clearance_check():
    payload = tucking(walking=0.55, tucked=0.26)
    assert check_narrow_pass(payload, gap_half_width_m=0.30).satisfied
    tight = check_narrow_pass(payload, gap_half_width_m=0.20)
    assert not tight.satisfied
    assert "never got inside" in tight.reason


def test_width_is_measured_across_the_heading_not_the_world_axis():
    """A motion walking along +y is not two metres wide."""
    frames = 120
    payload = tucking(frames)
    rotated = tucking(frames)
    yaw = math.pi / 2
    rotated["root_quat_w"][:, 0] = math.cos(yaw / 2)
    rotated["root_quat_w"][:, 3] = math.sin(yaw / 2)
    rotated["root_pos_w"][:, 0] = 0.0
    rotated["root_pos_w"][:, 1] = np.arange(frames) * 0.8 / FPS
    offsets = payload["body_pos_w"][:, :, :2] - payload["root_pos_w"][:, None, :2]
    rotated["body_pos_w"][:, :, 0] = rotated["root_pos_w"][:, None, 0] - offsets[:, :, 1]
    rotated["body_pos_w"][:, :, 1] = rotated["root_pos_w"][:, None, 1] + offsets[:, :, 0]
    assert check_narrow_pass(rotated).measurements["walking_half_width_m"] == pytest.approx(
        check_narrow_pass(payload).measurements["walking_half_width_m"], abs=1e-6
    )


def test_a_recording_without_body_positions_cannot_be_assessed():
    payload = episode(walk(60))
    with pytest.raises(PredicateError, match="width cannot be measured"):
        check_narrow_pass(payload)


def test_both_new_lateral_modes_dispatch_to_the_width_predicate():
    for behaviour in ("arm_tuck", "shoulder_turn"):
        result = check_behaviour(behaviour, tucking())
        assert result is not None and result.satisfied
