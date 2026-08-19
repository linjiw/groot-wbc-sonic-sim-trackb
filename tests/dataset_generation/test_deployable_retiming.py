"""Tests for asking a crouched robot for a crouched pace.

The property under test throughout is that retiming changes *when* the robot is somewhere and
never *where*: the route, the start, the goal, the joint angles at each route position and the
obstacle's station in route progress all survive, while duration does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.deployable_retiming import (
    MIN_RETIME_RATIO,
    RETIME_SAFETY,
    ROUTE_TOLERANCE_M,
    departure_profile,
    deployable_clip,
    resample_clip,
    retime_ratio,
    route_deviation_m,
    time_map,
    weighted_route_length,
)


def walk(frames: int = 120, *, turn: float = 0.0) -> np.ndarray:
    """A body walking along +x, optionally curving, with a small leg swing."""
    qpos = np.zeros((frames, 36))
    heading = np.linspace(0.0, turn, frames)
    step = 4.0 / (frames - 1)
    qpos[:, 0] = np.cumsum(step * np.cos(heading)) - step * np.cos(heading[0])
    qpos[:, 1] = np.cumsum(step * np.sin(heading)) - step * np.sin(heading[0])
    qpos[:, 2] = 0.74
    qpos[:, 3] = np.cos(0.5 * heading)
    qpos[:, 6] = np.sin(0.5 * heading)
    phase = np.linspace(0.0, 8 * np.pi, frames)
    qpos[:, 7] = 0.25 * np.sin(phase)
    qpos[:, 10] = 0.30 + 0.20 * np.sin(phase)
    qpos[:, 13] = 0.25 * np.sin(phase + np.pi)
    qpos[:, 16] = 0.30 + 0.20 * np.sin(phase + np.pi)
    return qpos


def progress(root_xy: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(root_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return cumulative / cumulative[-1]


def crouched(nominal: np.ndarray, station: float = 0.55, *, window: float = 0.18) -> np.ndarray:
    """A stand-in for a local adaptation: a smooth leg departure centred on one station.

    Deliberately not `local_crouch`, so these tests exercise the retiming rather than the MJCF and
    run wherever the model is absent.
    """
    alpha = np.clip((window - np.abs(progress(nominal[:, :2]) - station)) / (window * 0.55), 0, 1)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    adapted = nominal.copy()
    adapted[:, 7] -= alpha * 0.5
    adapted[:, 10] += alpha * 1.0
    adapted[:, 13] -= alpha * 0.5
    adapted[:, 16] += alpha * 1.0
    return adapted


# ---- the pace a measurement asks for -------------------------------------------------------


def test_a_measured_shortfall_sets_the_pace():
    """The correction is arithmetic, not a model: a clip that arrived a tenth of its window
    short is commanded a tenth slower, times the safety factor."""
    ratio, floored = retime_ratio(0.10, 1.00, safety=1.0)
    assert ratio == pytest.approx(0.90)
    assert not floored


def test_the_safety_factor_overshoots_because_undershooting_costs_a_rollout():
    plain, _ = retime_ratio(0.10, 1.00, safety=1.0)
    safe, _ = retime_ratio(0.10, 1.00, safety=RETIME_SAFETY)
    assert safe < plain, "the safety factor must slow the clip further, not less"


def test_an_adaptation_that_tracked_better_than_its_nominal_is_not_slowed():
    """The arm tuck passes through at 88-105% and the crouch at 46-67%. A negative excess is a
    tuck that beat its nominal, and slowing it would be inventing a cost it did not incur."""
    ratio, floored = retime_ratio(-0.04, 1.00)
    assert ratio == 1.0
    assert not floored


def test_a_shortfall_larger_than_its_window_is_floored_and_says_so():
    """Below the floor the clip is a different behaviour, not a slower walk. The caller has to
    know the request was not met, or it will read the resulting rejection as a fact about the
    operator rather than about the request."""
    ratio, floored = retime_ratio(2.0, 1.0)
    assert ratio == MIN_RETIME_RATIO
    assert floored


def test_a_window_of_no_length_cannot_set_a_pace():
    with pytest.raises(ValueError, match="no distance"):
        retime_ratio(0.1, 0.0)


def test_the_window_is_weighted_by_how_active_the_adaptation_is():
    """Dividing the shortfall by the whole route would include the stretches walked upright and
    under-correct. Only the active arclength is where the shortfall accrued."""
    xy = np.stack([np.linspace(0.0, 4.0, 101), np.zeros(101)], axis=1)
    everywhere = weighted_route_length(xy, np.ones(101))
    half = weighted_route_length(xy, np.r_[np.ones(51), np.zeros(50)])
    assert everywhere == pytest.approx(4.0)
    assert half == pytest.approx(2.0, abs=0.05)


# ---- the clock ------------------------------------------------------------------------------


def test_an_unwarped_clock_ticks_once_per_source_frame():
    tau = time_map(np.ones(50))
    assert len(tau) == 50
    assert tau == pytest.approx(np.arange(50.0))


def test_the_clock_is_monotone_and_covers_the_whole_clip():
    rate = np.linspace(1.0, 0.4, 80)
    tau = time_map(rate)
    assert np.all(np.diff(tau) > 0), "a non-monotone clock would replay frames backwards"
    assert tau[0] == pytest.approx(0.0)
    assert tau[-1] == pytest.approx(79.0), "the goal frame must be reached, not approached"


def test_halving_the_pace_roughly_doubles_the_frames():
    tau = time_map(np.full(101, 0.5))
    assert len(tau) == pytest.approx(201, abs=1)


def test_the_clock_refuses_to_speed_a_clip_up():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        time_map(np.full(20, 1.5))


# ---- resampling -----------------------------------------------------------------------------


def test_sampling_at_whole_frames_returns_those_frames_exactly():
    clip = walk(60, turn=0.6)
    assert resample_clip(clip, np.arange(60.0)) == pytest.approx(clip)


def test_the_root_quaternion_stays_on_the_unit_sphere():
    """Componentwise interpolation leaves the sphere, and a reference the tracker cannot read as
    an orientation is worse than one that is merely wrong."""
    clip = walk(60, turn=1.2)
    resampled = resample_clip(clip, np.linspace(0.0, 59.0, 173))
    assert np.linalg.norm(resampled[:, 3:7], axis=1) == pytest.approx(np.ones(173), abs=1e-9)


def test_interpolation_takes_the_short_way_round_a_sign_flip():
    """A quaternion and its negation are the same rotation. Interpolating between them naively
    swings the heading through half a circle inside one frame."""
    clip = np.zeros((2, 36))
    clip[0, 3] = 1.0
    clip[1, 3] = -1.0  # the same orientation, written with the opposite sign
    midpoint = resample_clip(clip, np.array([0.5]))[0, 3:7]
    assert abs(abs(midpoint[0]) - 1.0) < 1e-9, "the midpoint of a rotation with itself is itself"


def test_a_single_frame_cannot_be_retimed():
    with pytest.raises(ValueError, match="two frames"):
        resample_clip(np.zeros((1, 36)), np.array([0.0]))


# ---- what retiming holds --------------------------------------------------------------------


def test_a_clip_asked_for_nominal_pace_comes_back_untouched():
    nominal = walk()
    adapted = crouched(nominal)
    retimed, report = deployable_clip(nominal, adapted, ratio=1.0)
    assert retimed == pytest.approx(adapted)
    assert report.frames == report.source_frames
    assert report.duration_scale == pytest.approx(1.0)


def test_the_start_and_the_goal_are_reached_rather_than_approached():
    """Rounding the clip to a whole number of frames must not cost the endpoint, because the
    endpoint is what the gate measures and what the family's scene was built around."""
    nominal = walk(turn=0.8)
    retimed, report = deployable_clip(nominal, crouched(nominal), excess_lag_m=0.27)
    assert report.start_shift_m == pytest.approx(0.0, abs=1e-9)
    assert report.goal_shift_m == pytest.approx(0.0, abs=1e-9)


def test_the_robot_still_walks_the_same_line_through_the_room():
    """The obstacle was placed against the nominal route. If retiming moved the route, the family's
    scene would no longer be testing the thing it was built to test."""
    nominal = walk(turn=0.9)
    adapted = crouched(nominal)
    retimed, report = deployable_clip(nominal, adapted, excess_lag_m=0.30)
    assert report.route_deviation_m < ROUTE_TOLERANCE_M
    assert report.holds_route


def test_the_obstacle_station_lands_at_the_same_place_on_the_route():
    """Route progress is normalised arclength, so a station is a place. Retiming must move the
    frame at which the robot arrives there without moving where 'there' is."""
    nominal = walk(turn=0.5)
    adapted = crouched(nominal, station=0.55)
    retimed, _ = deployable_clip(nominal, adapted, excess_lag_m=0.30)
    for station in (0.25, 0.55, 0.80):
        before = np.array(
            [np.interp(station, progress(adapted[:, :2]), adapted[:, axis]) for axis in (0, 1)]
        )
        after = np.array(
            [np.interp(station, progress(retimed[:, :2]), retimed[:, axis]) for axis in (0, 1)]
        )
        assert np.linalg.norm(before - after) < 5e-3, f"station {station} moved"


def test_the_joint_angles_survive_at_the_route_position_the_operator_chose():
    """Retiming is a change of clock. The crouch must be exactly as deep where the shelf is."""
    nominal = walk()
    adapted = crouched(nominal, station=0.55)
    retimed, _ = deployable_clip(nominal, adapted, excess_lag_m=0.25)
    knee_before = np.interp(0.55, progress(adapted[:, :2]), adapted[:, 10])
    knee_after = np.interp(0.55, progress(retimed[:, :2]), retimed[:, 10])
    assert knee_after == pytest.approx(knee_before, abs=2e-3)


def test_only_the_stretch_the_adaptation_occupies_is_slowed():
    """A clip that crawled from frame zero would be a slow walk, not a walk that slows for an
    obstacle, and it would spend duration where nothing is being cleared.

    Asked as *where the added frames went*: rounding the clip to a whole number of frames is
    spread across it, so the approach is not bit-identical, but almost none of the added duration
    may fall before the adaptation begins.
    """
    nominal = walk()
    adapted = crouched(nominal, station=0.55, window=0.12)
    retimed, report = deployable_clip(nominal, adapted, excess_lag_m=0.25)
    assert report.active_fraction < 0.5

    added = report.frames - report.source_frames
    assert added > 0
    reached = np.searchsorted(progress(retimed[:, :2]), 0.43)
    unadapted = np.searchsorted(progress(adapted[:, :2]), 0.43)
    assert reached - unadapted <= max(1, 0.1 * added), "the approach absorbed the slowdown"


def test_retiming_only_ever_slows_a_clip_down():
    nominal = walk()
    retimed, report = deployable_clip(nominal, crouched(nominal), excess_lag_m=0.35)
    assert report.duration_scale >= 1.0
    assert len(retimed) >= len(nominal)
    assert report.realised_ratio <= report.commanded_ratio + 1e-12


def test_a_deeper_shortfall_buys_a_longer_clip():
    """The relationship the register measured -- lag rising monotonically with crouch depth --
    has to come back out as duration, or the correction is not tracking the cause."""
    nominal = walk()
    adapted = crouched(nominal)
    lengths = [
        len(deployable_clip(nominal, adapted, excess_lag_m=lag)[0]) for lag in (0.05, 0.15, 0.30)
    ]
    assert lengths[0] < lengths[1] < lengths[2]


# ---- reading the adaptation off the clips ----------------------------------------------------


def test_the_active_stretch_is_read_from_the_clips_not_from_the_parameters():
    """A profile recomputed from the station and window drifts away from the clip whenever a
    bisection, a clamp or an excursion cap changed what was actually applied."""
    nominal = walk()
    adapted = crouched(nominal, station=0.30)
    alpha = departure_profile(nominal, adapted)
    assert alpha.max() == pytest.approx(1.0)
    # The centre of activity, not its first frame: a fully-active plateau has many frames at the
    # peak and `argmax` would report whichever edge came first.
    route = progress(nominal[:, :2])
    centre = float((route * alpha).sum() / alpha.sum())
    assert centre == pytest.approx(0.30, abs=0.05)


def test_two_identical_clips_have_no_adaptation_to_retime():
    nominal = walk()
    with pytest.raises(ValueError, match="identical"):
        departure_profile(nominal, nominal.copy())


def test_a_deployable_skill_is_retimed_from_its_own_causal_reference():
    """Pairing a clip with something that is not the reference it was bent from would reintroduce
    exactly the two-different-journeys problem the matched construction removed."""
    with pytest.raises(ValueError, match="same shape"):
        departure_profile(walk(120), walk(90))


def test_the_caller_must_choose_between_a_measurement_and_a_sweep():
    nominal = walk()
    adapted = crouched(nominal)
    with pytest.raises(ValueError, match="exactly one"):
        deployable_clip(nominal, adapted, excess_lag_m=0.2, ratio=0.8)
    with pytest.raises(ValueError, match="exactly one"):
        deployable_clip(nominal, adapted)


# ---- route deviation ------------------------------------------------------------------------


def test_a_point_beside_the_route_is_measured_from_the_route_not_from_a_frame():
    path = np.stack([np.linspace(0.0, 4.0, 41), np.zeros(41)], axis=1)
    shifted = path.copy()
    shifted[:, 0] += 0.9  # far along the route, but exactly on it
    assert route_deviation_m(path, shifted[:-10]) < 1e-9
    beside = path.copy()
    beside[:, 1] += 0.05
    assert route_deviation_m(path, beside) == pytest.approx(0.05)
