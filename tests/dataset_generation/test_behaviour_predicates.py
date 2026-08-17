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
    check_pause,
    check_side_step,
    check_stand_to_walk,
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
