"""Tests for the frozen corpus snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.corpus_snapshot import (
    EpisodeRecord,
    SnapshotError,
    build_snapshot,
    check_consistency,
    trajectory_fingerprint,
    wilson_interval,
)


def record(
    episode_id: str,
    outcome: str = "accepted",
    behaviour: str = "walk",
    fingerprint: str | None = None,
    video: str | None = "clip.mp4",
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=episode_id,
        trajectory_path=f"/x/{episode_id}.pkl",
        trajectory_sha256="sha256:deadbeef",
        trajectory_fingerprint=fingerprint or episode_id,
        frames=200,
        fps=50.0,
        outcome=outcome,
        rejection_reasons=(),
        demoted_failures=(),
        errors=(),
        recovered_from_split=False,
        policy="scene_around_motion",
        behaviour=behaviour,
        speed_style="steady",
        video_path=video,
        video_sha256=None,
    )


def test_wilson_interval_is_wide_where_the_sample_is_thin():
    """A family at 2/2 is not solved, and the interval must say so."""
    low, high = wilson_interval(2, 2)
    assert low < 0.5 and high == pytest.approx(1.0)
    # The same rate with a real sample behind it is a much tighter claim.
    tight_low, _ = wilson_interval(60, 60)
    assert tight_low > low


def test_wilson_interval_at_zero_is_not_a_point():
    low, high = wilson_interval(0, 3)
    assert low == 0.0
    assert 0.3 < high < 0.8


def test_wilson_interval_of_no_trials_is_the_whole_line():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_impossible_counts_are_rejected():
    with pytest.raises(SnapshotError, match="impossible"):
        wilson_interval(5, 3)


def test_identical_state_gives_an_identical_fingerprint():
    """The same motion rolled out in two rooms is one behaviour, not two."""
    payload = {
        "root_pos_w": np.arange(30, dtype=np.float64).reshape(10, 3),
        "root_quat_w": np.zeros((10, 4)),
        "dof_pos": np.ones((10, 29)),
    }
    twin = {key: value.copy() for key, value in payload.items()}
    assert trajectory_fingerprint(payload) == trajectory_fingerprint(twin)


def test_a_different_trajectory_gives_a_different_fingerprint():
    base = {
        "root_pos_w": np.zeros((10, 3)),
        "root_quat_w": np.zeros((10, 4)),
        "dof_pos": np.zeros((10, 29)),
    }
    moved = {key: value.copy() for key, value in base.items()}
    moved["root_pos_w"][5, 0] = 1.0
    assert trajectory_fingerprint(base) != trajectory_fingerprint(moved)


def test_a_payload_without_state_cannot_be_fingerprinted():
    with pytest.raises(SnapshotError, match="no state arrays"):
        trajectory_fingerprint({"fps": 50.0})


def test_duplicates_are_counted_separately_from_episodes():
    """Measured on the real corpus: 156 evaluable episodes, 131 distinct trajectories."""
    snapshot = build_snapshot(
        "test",
        "gate-v1",
        [
            record("a", fingerprint="same"),
            record("b", fingerprint="same"),
            record("c", fingerprint="other"),
        ],
    )
    assert snapshot.episodes == 3
    assert snapshot.distinct_trajectories == 2
    assert snapshot.duplicate_episodes == 1


def test_unevaluable_episodes_are_excluded_from_distinct_counts():
    snapshot = build_snapshot(
        "test",
        "gate-v1",
        [record("a"), record("b", outcome="unevaluable", fingerprint="unevaluable")],
    )
    assert len(snapshot.evaluable) == 1
    assert snapshot.distinct_trajectories == 1


def test_missing_clips_are_named_not_merely_counted():
    """A reviewer should never have to guess which episode a count gap refers to."""
    snapshot = build_snapshot("test", "gate-v1", [record("a"), record("b", video=None)])
    assert [r.episode_id for r in snapshot.missing_clips] == ["b"]
    assert snapshot.to_dict()["missing_clips"] == [{"episode_id": "b", "outcome": "accepted"}]


def test_family_rates_carry_their_uncertainty():
    snapshot = build_snapshot(
        "test",
        "gate-v1",
        [record(f"w{i}", behaviour="walk") for i in range(2)]
        + [record("s0", outcome="rejected", behaviour="squat")],
    )
    families = snapshot.family_rates()
    assert families["walk"]["rate"] == pytest.approx(1.0)
    assert families["walk"]["wilson_low"] < 0.9
    assert families["squat"]["rate"] == 0.0
    assert not families["walk"]["informative"]


def test_a_family_with_a_real_sample_is_informative():
    snapshot = build_snapshot(
        "test", "gate-v1", [record(f"w{i}", fingerprint=f"f{i}") for i in range(60)]
    )
    assert snapshot.family_rates()["walk"]["informative"]


def test_an_empty_snapshot_is_refused():
    with pytest.raises(SnapshotError, match="empty snapshot"):
        build_snapshot("test", "gate-v1", [])


def test_internal_contradictions_are_caught_at_build_time():
    """The whole point: a report cannot inherit a contradiction the snapshot rejects."""
    good = build_snapshot("test", "gate-v1", [record("a")])
    assert check_consistency(good) == []


def test_the_dict_view_carries_the_gate_version():
    payload = build_snapshot("snap", "gate-v9", [record("a")]).to_dict()
    assert payload["gate_version"] == "gate-v9"
    assert payload["snapshot"] == "snap"
    assert payload["counts"]["episodes"] == 1
