"""Tests for splitting a capture that spans an environment reset."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.trajectory_segments import (
    SegmentError,
    best_evaluable_payload,
    find_reset_boundaries,
    segment_spans,
    spans_a_reset,
    split_payload,
)


def payload(times: list[float], **extra) -> dict:
    frames = len(times)
    base = {
        "motion_time_s": np.asarray(times, dtype=np.float64),
        "root_pos_w": np.arange(frames * 3, dtype=np.float64).reshape(frames, 3),
        "body_pos_w": np.zeros((frames, 30, 3)),
        "total_frames": frames,
        "fps": 50.0,
        "body_names": ["pelvis"],
        "tracking_metrics": {"error_body_pos": np.zeros((frames, 4))},
    }
    base.update(extra)
    return base


def ramp(start: float, count: int, step: float = 0.02) -> list[float]:
    return [start + i * step for i in range(count)]


def test_single_pass_has_no_boundary():
    assert find_reset_boundaries(np.asarray(ramp(0.0, 50))) == []
    assert not spans_a_reset(payload(ramp(0.0, 50)))


def test_boundary_is_the_frame_where_the_clock_drops():
    times = ramp(0.0, 60) + ramp(0.0, 40)
    assert find_reset_boundaries(np.asarray(times)) == [60]


def test_repeated_timestamps_are_not_a_reset():
    """A held or paused reference repeats a timestamp; that is not a new pass.

    Treating equality as a boundary would shred a stand-still motion into single frames.
    """
    times = [0.0, 0.02, 0.02, 0.02, 0.04, 0.06]
    assert find_reset_boundaries(np.asarray(times)) == []


def test_multiple_resets_produce_multiple_spans():
    times = ramp(0.0, 50) + ramp(0.0, 50) + ramp(0.0, 50)
    spans = segment_spans(np.asarray(times))
    assert [(s.start, s.end) for s in spans] == [(0, 50), (50, 100), (100, 150)]
    assert [s.frames for s in spans] == [50, 50, 50]


def test_empty_clock_is_an_error():
    with pytest.raises(SegmentError, match="empty"):
        find_reset_boundaries(np.asarray([]))


def test_split_slices_per_frame_arrays_and_copies_metadata():
    data = payload(ramp(0.0, 60) + ramp(0.0, 50))
    segments = split_payload(data)
    assert [s["total_frames"] for s in segments] == [60, 50]
    for segment in segments:
        assert segment["root_pos_w"].shape[0] == segment["total_frames"]
        assert segment["body_pos_w"].shape[0] == segment["total_frames"]
        assert segment["fps"] == 50.0
        assert segment["body_names"] == ["pelvis"]


def test_split_slices_arrays_nested_one_level_down():
    """tracking_metrics holds per-frame arrays; copying it unchanged leaves the segment
    internally inconsistent in a way most consumers never check."""
    data = payload(ramp(0.0, 60) + ramp(0.0, 50))
    for segment in split_payload(data):
        nested = segment["tracking_metrics"]["error_body_pos"]
        assert nested.shape[0] == segment["total_frames"]


def test_total_frames_is_rewritten_not_inherited():
    data = payload(ramp(0.0, 60) + ramp(0.0, 50))
    assert data["total_frames"] == 110
    assert sum(s["total_frames"] for s in split_payload(data)) == 110
    assert all(s["total_frames"] != 110 for s in split_payload(data))


def test_segments_come_back_longest_first():
    data = payload(ramp(0.0, 45) + ramp(0.0, 90))
    assert [s["total_frames"] for s in split_payload(data)] == [90, 45]


def test_short_tail_is_dropped():
    """The trailing partial pass was cut by the recorder, not the environment, so it has
    no endpoint to compare against."""
    data = payload(ramp(0.0, 100) + ramp(0.0, 5))
    segments = split_payload(data, min_frames=40)
    assert [s["total_frames"] for s in segments] == [100]


def test_all_segments_too_short_is_an_error_naming_the_lengths():
    data = payload(ramp(0.0, 10) + ramp(0.0, 12))
    with pytest.raises(SegmentError, match=r"\[10, 12\]"):
        split_payload(data, min_frames=40)


def test_best_evaluable_returns_the_input_untouched_for_a_single_pass():
    data = payload(ramp(0.0, 80))
    best, was_split = best_evaluable_payload(data)
    assert was_split is False
    assert best is data


def test_best_evaluable_picks_the_longest_pass():
    data = payload(ramp(0.0, 40) + ramp(0.0, 95))
    best, was_split = best_evaluable_payload(data)
    assert was_split is True
    assert best["total_frames"] == 95
    assert best["root_pos_w"].shape[0] == 95


def test_arrays_that_merely_share_the_frame_count_by_coincidence_are_still_sliced():
    """A per-body array whose length equals the frame count is indistinguishable from a
    per-frame one, so the split must not be trusted to guess -- it slices on the leading
    axis only, which is the documented contract."""
    data = payload(ramp(0.0, 60) + ramp(0.0, 30), extra_array=np.zeros((90, 2)))
    best, _ = best_evaluable_payload(data)
    assert best["extra_array"].shape == (60, 2)


def test_scalars_and_non_matching_arrays_survive_unchanged():
    data = payload(ramp(0.0, 60) + ramp(0.0, 30), dof_order=np.arange(29))
    best, _ = best_evaluable_payload(data)
    assert best["dof_order"].shape == (29,)
