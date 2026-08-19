"""Tests for joining two clips where the gait already agrees.

The properties under test are the ones a tracker feels at a seam: that the phase estimate orders
the stride, that the planner finds the frames where two clips are at the same point in it, that
re-anchoring moves the target clip without tipping or lifting it, and that the stitched result has
no step in commanded velocity.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.motion_transitions import (
    DEFAULT_BLEND_FRAMES,
    anchor_to,
    gait_phase,
    holds_route,
    join_discontinuity,
    phase_gap,
    plan_transition,
    route_progress,
    stitch,
    support_profile,
)

STRIDES = 4
FRAMES = 120


def swing(frames: int = FRAMES, *, offset: float = 0.0) -> np.ndarray:
    return np.linspace(0.0, STRIDES * 2 * np.pi, frames) + offset


def soles(phase: np.ndarray) -> np.ndarray:
    """Two feet alternately lifted, the shape the phase estimator is meant to read."""
    return np.stack(
        [
            0.03 * np.clip(np.sin(phase), 0.0, None),
            0.03 * np.clip(np.sin(phase + np.pi), 0.0, None),
        ],
        axis=1,
    )


def walk(frames: int = FRAMES, *, offset: float = 0.0, x1: float = 4.0) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 0] = np.linspace(0.0, x1, frames)
    qpos[:, 2] = 0.74
    qpos[:, 3] = 1.0
    phase = swing(frames, offset=offset)
    qpos[:, 7] = 0.25 * np.sin(phase)
    qpos[:, 10] = 0.30 + 0.20 * np.sin(phase)
    qpos[:, 13] = 0.25 * np.sin(phase + np.pi)
    qpos[:, 16] = 0.30 + 0.20 * np.sin(phase + np.pi)
    return qpos


# ---- reading the stride ---------------------------------------------------------------------


def test_the_phase_wraps_exactly_once_per_stride():
    """Four strides in the clip must give four wraps. A phase that wraps twice per stride cannot
    tell stepping left from stepping right, which is the whole point of pairing the height
    difference with its own rate."""
    phase = gait_phase(soles(swing()))
    wraps = int(np.sum(np.abs(np.diff(phase)) > np.pi))
    assert wraps == STRIDES


def test_the_phase_advances_monotonically_within_a_stride():
    phase = np.unwrap(gait_phase(soles(swing())))
    assert np.all(np.diff(phase) > 0), "the stride must not run backwards"


def test_a_clip_with_no_stride_reports_no_phase():
    """Standing still has no point in a cycle, and inventing one would let the planner match a
    stationary clip to any frame of a walking one."""
    still = np.full((60, 2), 0.01)
    assert gait_phase(still) == pytest.approx(np.zeros(60))


def test_two_clips_offset_in_the_stride_are_offset_in_phase():
    early = gait_phase(soles(swing()))
    late = gait_phase(soles(swing(offset=np.pi / 2)))
    middle = len(early) // 2
    assert phase_gap(float(early[middle]), float(late[middle])) > 1.0


def test_phase_distance_wraps_the_short_way():
    assert phase_gap(3.0, -3.0) == pytest.approx(2 * np.pi - 6.0, abs=1e-9)
    assert phase_gap(0.2, -0.2) == pytest.approx(0.4)


def test_both_feet_down_scores_full_support():
    down = np.zeros((10, 2))
    one_up = np.stack([np.zeros(10), np.full(10, 0.2)], axis=1)
    assert support_profile(down) == pytest.approx(np.ones(10))
    assert support_profile(one_up) == pytest.approx(np.full(10, 0.5))


def test_a_stride_cannot_be_read_from_two_frames():
    with pytest.raises(ValueError, match="three frames"):
        gait_phase(np.zeros((2, 2)))


# ---- choosing the join ----------------------------------------------------------------------


def test_the_planner_finds_the_frames_where_the_two_strides_agree():
    """The target clip is shifted 0.7 rad into its stride, which at this cadence is about three
    frames. The planner must pay that offset back rather than joining at equal frame indices."""
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.55,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
    )
    assert point.phase_gap_rad < 0.2, "the join was placed at a mismatched point in the stride"
    assert point.source_frame - point.target_frame == pytest.approx(3, abs=1)


def test_the_join_lands_where_the_obstacle_is():
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.30,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
        window=0.05,
    )
    assert abs(point.route_progress - 0.30) <= 0.05


def test_a_switch_that_cannot_be_placed_at_the_obstacle_is_refused():
    """Silently joining somewhere else would produce a clip that looks fine and adapts in the
    wrong room."""
    source, target = walk(), walk()
    with pytest.raises(ValueError, match="cannot be placed"):
        plan_transition(
            source,
            target,
            at_progress=0.5,
            source_soles=soles(swing()),
            target_soles=soles(swing()),
            window=1e-6,
        )


def test_double_support_is_preferred_when_everything_else_ties():
    """Switching mid-flight retargets a leg that is swinging with nothing to push against."""
    source, target = walk(), walk()
    grounded = np.zeros((FRAMES, 2))
    airborne = np.stack([np.zeros(FRAMES), np.full(FRAMES, 0.2)], axis=1)
    both_down = plan_transition(
        source, target, at_progress=0.5, source_soles=grounded, target_soles=grounded
    )
    one_up = plan_transition(
        source, target, at_progress=0.5, source_soles=grounded, target_soles=airborne
    )
    assert both_down.support > one_up.support
    assert both_down.cost < one_up.cost


def test_two_clips_of_different_bodies_cannot_be_joined():
    with pytest.raises(ValueError, match="same body"):
        plan_transition(
            walk(),
            np.zeros((FRAMES, 30)),
            at_progress=0.5,
            source_soles=soles(swing()),
            target_soles=soles(swing()),
        )


# ---- re-anchoring ---------------------------------------------------------------------------


def test_the_anchored_frame_lands_exactly_where_the_robot_is():
    target = walk()
    pose = np.array([2.5, -1.25, 0.74])
    yaw = 0.6
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    moved = anchor_to(target, 40, pose, quat)
    assert moved[40, :2] == pytest.approx(pose[:2])


def test_anchoring_turns_the_clip_without_tipping_it():
    """Matching roll or pitch would lean a reference the robot is expected to hold upright."""
    target = walk()
    yaw = 1.1
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    moved = anchor_to(target, 40, np.array([0.0, 0.0, 0.74]), quat)
    assert moved[:, 4] == pytest.approx(np.zeros(len(moved)), abs=1e-12), "roll appeared"
    assert moved[:, 5] == pytest.approx(np.zeros(len(moved)), abs=1e-12), "pitch appeared"


def test_anchoring_leaves_height_alone_so_a_crouch_stays_crouched():
    crouched = walk()
    crouched[:, 2] = 0.60
    moved = anchor_to(crouched, 40, np.array([1.0, 1.0, 0.74]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert moved[:, 2] == pytest.approx(crouched[:, 2])


def test_anchoring_preserves_the_shape_of_the_route():
    """Only a rigid move is allowed: the path may be placed anywhere, but its length is the
    journey the clip was verified on."""
    target = walk()
    yaw = 0.9
    moved = anchor_to(
        target, 30, np.array([5.0, 2.0, 0.74]), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    )
    before = np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1).sum()
    after = np.linalg.norm(np.diff(moved[:, :2], axis=0), axis=1).sum()
    assert after == pytest.approx(before)


# ---- stitching ------------------------------------------------------------------------------


def test_the_seam_has_no_step_in_commanded_velocity():
    """A position step is a bounded error the tracker closes. A velocity step asks for an
    acceleration no actuator has, which is why the blend exists at all."""
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.55,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
    )
    joined = stitch(source, target, point)
    ordinary = np.abs(np.diff(np.diff(source[:, 7:], axis=0), axis=0)).max()
    for frame in range(point.source_frame - 1, point.source_frame + DEFAULT_BLEND_FRAMES + 1):
        _, velocity = join_discontinuity(joined, frame)
        assert velocity < 10 * ordinary + 1e-3, f"velocity step at frame {frame}"


def test_the_stitched_clip_still_walks_the_source_line():
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.55,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
    )
    assert holds_route(source, stitch(source, target, point))


def test_joining_a_clip_to_itself_returns_that_clip():
    """The degenerate case has to be the identity, or every non-degenerate result is suspect."""
    source = walk()
    point = plan_transition(
        source,
        source.copy(),
        at_progress=0.5,
        source_soles=soles(swing()),
        target_soles=soles(swing()),
    )
    assert point.source_frame == point.target_frame
    joined = stitch(source, source.copy(), point)
    assert len(joined) == len(source)
    assert joined == pytest.approx(source, abs=1e-9)


def test_a_zero_length_blend_is_a_plain_concatenation():
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.55,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
    )
    joined = stitch(source, target, point, blend_frames=0)
    assert len(joined) == point.source_frame + (len(target) - point.target_frame)


def test_a_negative_blend_is_refused():
    source, target = walk(), walk()
    point = plan_transition(
        source, target, at_progress=0.5, source_soles=soles(swing()), target_soles=soles(swing())
    )
    with pytest.raises(ValueError, match="not be negative"):
        stitch(source, target, point, blend_frames=-1)


def test_the_stitched_route_progress_still_runs_from_nought_to_one():
    source, target = walk(), walk(offset=0.7)
    point = plan_transition(
        source,
        target,
        at_progress=0.55,
        source_soles=soles(swing()),
        target_soles=soles(swing(offset=0.7)),
    )
    progress = route_progress(stitch(source, target, point)[:, :2])
    assert progress[0] == pytest.approx(0.0)
    assert progress[-1] == pytest.approx(1.0)
    assert np.all(np.diff(progress) >= -1e-12)


def test_a_frame_with_no_neighbours_has_no_discontinuity_to_report():
    with pytest.raises(ValueError, match="no neighbours"):
        join_discontinuity(walk(), 0)
