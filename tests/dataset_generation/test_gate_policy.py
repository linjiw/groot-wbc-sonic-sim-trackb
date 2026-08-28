"""Tests for grading an episode according to what its label actually is."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.gate_policy import (
    DEFAULT_MAX_DRIFT_RATE_MPS,
    REFERENCE_GATES,
    SCENE_AROUND_MOTION,
    SCENE_FIRST,
    SETTLING_TIME_S,
    GatePolicyError,
    apply_policy,
    drift_rate_mps,
    endpoint_error_m,
)

FPS = 50.0


def payload(drift_per_second: float = 0.0, frames: int = 200, offset: float = 0.0) -> dict:
    seconds = np.arange(frames) / FPS
    executed = np.zeros((frames, 3))
    executed[:, 0] = seconds * 0.8
    # Error grows sideways at the requested rate, on top of a constant offset.
    executed[:, 1] = offset + drift_per_second * seconds
    reference = np.zeros((frames, 36))
    reference[:, 0] = seconds * 0.8
    return {
        "root_pos_w": executed,
        "reference_g1_qpos": reference,
        "fps": FPS,
        "total_frames": frames,
    }


def test_stable_tracking_has_a_drift_rate_near_zero():
    data = payload(drift_per_second=0.0)
    rate = drift_rate_mps(data["root_pos_w"][:, :2], data["reference_g1_qpos"][:, :2], FPS)
    assert rate == pytest.approx(0.0, abs=1e-9)


def test_drift_rate_recovers_the_constructed_slope():
    data = payload(drift_per_second=0.3)
    rate = drift_rate_mps(data["root_pos_w"][:, :2], data["reference_g1_qpos"][:, :2], FPS)
    assert rate == pytest.approx(0.3, rel=1e-6)


def test_a_constant_offset_is_not_drift():
    """An episode that settles off-reference and stays there is stable, not diverging.

    This is the case the whole-episode slope handles badly and the reason the fit uses the
    second half only.
    """
    data = payload(drift_per_second=0.0, offset=0.4)
    rate = drift_rate_mps(data["root_pos_w"][:, :2], data["reference_g1_qpos"][:, :2], FPS)
    assert rate == pytest.approx(0.0, abs=1e-9)


def test_drift_rate_is_independent_of_episode_length():
    """The whole point: the metric must not punish a prompt for being long.

    Both clips must clear the settling window plus the minimum fit window, which is why
    the shorter one is 3 s rather than 2 s.
    """
    short = payload(drift_per_second=0.2, frames=150)
    long = payload(drift_per_second=0.2, frames=400)
    rate_short = drift_rate_mps(short["root_pos_w"][:, :2], short["reference_g1_qpos"][:, :2], FPS)
    rate_long = drift_rate_mps(long["root_pos_w"][:, :2], long["reference_g1_qpos"][:, :2], FPS)
    assert rate_short == pytest.approx(rate_long, rel=1e-6)


def test_too_short_to_fit_is_an_error_not_a_zero():
    """A 1 s clip has no post-settling window, so there is nothing to fit."""
    with pytest.raises(GatePolicyError, match="settling window"):
        drift_rate_mps(np.zeros((50, 2)), np.zeros((50, 2)), FPS)


def test_the_fit_starts_after_the_measured_settling_time():
    """Error that grows only during settling and then holds is stable, not drifting.

    Fitting from the midpoint instead reported drift here, which is how the gate acquired
    a short-episode penalty of its own.
    """
    frames = 300
    seconds = np.arange(frames) / FPS
    executed = np.zeros((frames, 3))
    executed[:, 0] = seconds * 0.8
    executed[:, 1] = np.minimum(seconds, SETTLING_TIME_S) * 0.2  # ramps, then flat
    reference = np.zeros((frames, 36))
    reference[:, 0] = seconds * 0.8
    rate = drift_rate_mps(executed[:, :2], reference[:, :2], FPS)
    assert rate == pytest.approx(0.0, abs=1e-6)


def test_bad_fps_is_rejected():
    with pytest.raises(GatePolicyError, match="fps must be positive"):
        drift_rate_mps(np.zeros((40, 2)), np.zeros((40, 2)), 0.0)


def test_scene_around_motion_demotes_reference_gates():
    """The room was built around the executed corridor, so the reference is not the label."""
    outcome = apply_policy(
        ["reference_path_tracking_error", "reference_endpoint_tracking_error"],
        payload(),
        SCENE_AROUND_MOTION,
    )
    assert outcome.accepted
    assert outcome.rejection_reasons == ()
    assert set(outcome.demoted_failures) == REFERENCE_GATES


def test_safety_gates_still_bind_under_every_policy():
    """Nothing here relaxes what the corpus promises about collisions."""
    for policy, extra in (
        (SCENE_AROUND_MOTION, {}),
        (SCENE_FIRST, {"planned_goal_xy": np.array([3.2, 0.0])}),
    ):
        outcome = apply_policy(
            ["reference_path_tracking_error", "disallowed_robot_contact"],
            payload(),
            policy,
            **extra,
        )
        assert not outcome.accepted
        assert "disallowed_robot_contact" in outcome.rejection_reasons


def test_the_demoted_failure_is_recorded_not_discarded():
    """Provenance must still say the episode departed from its reference."""
    outcome = apply_policy(["reference_path_tracking_error"], payload(), SCENE_AROUND_MOTION)
    assert outcome.demoted_failures == ("reference_path_tracking_error",)
    assert "drift_rate_mps" in outcome.diagnostics


def test_runaway_divergence_is_caught_even_with_no_other_failure():
    outcome = apply_policy([], payload(drift_per_second=0.9), SCENE_AROUND_MOTION)
    assert not outcome.accepted
    assert "unstable_reference_drift" in outcome.rejection_reasons


def test_the_backstop_admits_everything_the_corpus_measured_as_clean():
    """Otherwise-clean episodes reached at most 0.139 m/s; safety-rejected reached 0.638."""
    assert 0.139 < DEFAULT_MAX_DRIFT_RATE_MPS < 0.638
    assert apply_policy([], payload(drift_per_second=0.139), SCENE_AROUND_MOTION).accepted


def test_a_clip_too_short_to_assess_abstains_rather_than_rejecting():
    """Rejecting a 1 s clip for instability it has not had time to show would reintroduce
    exactly the length dependence this policy exists to remove.

    Measured: the midpoint-fit version rejected 49 episodes at a 1.2 s horizon, inverting
    the duration bias instead of removing it. Safety gates still apply to short clips.
    """
    outcome = apply_policy([], payload(frames=50), SCENE_AROUND_MOTION)
    assert outcome.accepted
    assert "drift_rate_not_assessable" in outcome.diagnostics
    assert "drift_rate_mps" not in outcome.diagnostics


def test_a_short_clip_is_still_subject_to_safety_gates():
    outcome = apply_policy(["disallowed_robot_contact"], payload(frames=50), SCENE_AROUND_MOTION)
    assert not outcome.accepted


def test_scene_first_grades_against_the_planned_goal():
    goal = np.array([3.184, 0.0])  # where a 200-frame, 0.8 m/s run ends
    reached = apply_policy([], payload(), SCENE_FIRST, planned_goal_xy=goal)
    assert reached.accepted
    assert reached.diagnostics["goal_error_m"] == pytest.approx(0.0, abs=0.02)

    missed = apply_policy([], payload(), SCENE_FIRST, planned_goal_xy=np.array([9.0, 0.0]))
    assert not missed.accepted
    assert "planned_goal_not_reached" in missed.rejection_reasons


def test_scene_first_still_ignores_path_hugging():
    """Even where the route is the label, corridor-hugging over a whole episode is the
    wrong instrument -- only the endpoint and stability are graded."""
    outcome = apply_policy(
        ["reference_path_tracking_error"],
        payload(),
        SCENE_FIRST,
        planned_goal_xy=np.array([3.184, 0.0]),
    )
    assert outcome.accepted
    assert outcome.demoted_failures == ("reference_path_tracking_error",)


def test_scene_first_without_a_goal_is_an_error_not_a_silent_pass():
    with pytest.raises(GatePolicyError, match="planned goal"):
        apply_policy([], payload(), SCENE_FIRST)


def test_endpoint_error_measures_from_the_last_frame():
    executed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert endpoint_error_m(executed, np.array([2.5, 0.0])) == pytest.approx(0.5)


def test_reasons_are_deduplicated():
    outcome = apply_policy(
        ["disallowed_robot_contact", "disallowed_robot_contact"],
        payload(),
        SCENE_AROUND_MOTION,
    )
    assert outcome.rejection_reasons.count("disallowed_robot_contact") == 1
