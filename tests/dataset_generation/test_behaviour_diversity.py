"""Tests for the behaviour-diversity measurement."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.behaviour_diversity import (
    build_diversity_report,
    effective_rank,
    summarise_episode,
)

FPS = 50.0
DIM = 8


def straight_payload(speed_mps: float, frames: int = 100, height: float = 0.75, seed: int = 0):
    """A straight-line walk at a fixed speed, with a low-rank action block."""
    t = np.arange(frames)
    root = np.zeros((frames, 3))
    root[:, 0] = t * speed_mps / FPS
    root[:, 2] = height
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1))
    rng = np.random.default_rng(seed)
    # Each episode's own variance sits on the same two directions, so within-episode rank
    # is ~2 for all of them. The per-episode offset goes on a *different* dimension per
    # episode, which is what makes episodes occupy different parts of the space. Offsetting
    # every dimension equally instead would move all episodes along one shared direction,
    # and the pooled rank would correctly collapse to ~1.
    action = np.zeros((frames, DIM))
    action[:, 0] = np.sin(t / 7.0)
    action[:, 1] = np.cos(t / 7.0)
    action[:, 2 + seed % (DIM - 2)] += 5.0 * (1 + seed)
    action += rng.normal(scale=1e-4, size=action.shape)
    return {
        "root_pos_w": root,
        "root_quat_w": quat,
        "action_motion_token": action,
        "fps": FPS,
        "total_frames": frames,
    }


def test_effective_rank_is_one_for_a_single_direction():
    data = np.outer(np.linspace(-1, 1, 200), np.array([1.0, 2.0, -0.5, 0.0]))
    assert effective_rank(data) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_equals_dimension_for_an_isotropic_cloud():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(20000, 4))
    assert effective_rank(data) == pytest.approx(4.0, rel=0.05)


def test_effective_rank_needs_two_samples():
    assert effective_rank(np.zeros((1, 4))) == 0.0


def test_effective_rank_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2-D"):
        effective_rank(np.zeros(10))


def test_mean_speed_matches_the_constructed_speed():
    episode = summarise_episode("e", straight_payload(speed_mps=0.9))
    assert episode.mean_speed_mps == pytest.approx(0.9, rel=1e-6)


def test_straight_walk_has_unit_tortuosity_and_no_heading_change():
    episode = summarise_episode("e", straight_payload(speed_mps=0.7))
    assert episode.tortuosity == pytest.approx(1.0, rel=1e-6)
    assert episode.heading_change_rad == pytest.approx(0.0, abs=1e-9)


def test_heading_change_accumulates_through_a_turn():
    payload = straight_payload(speed_mps=0.7)
    frames = len(payload["root_pos_w"])
    angle = np.linspace(0.0, math.pi / 2, frames)
    radius = 2.0
    payload["root_pos_w"][:, 0] = radius * np.sin(angle)
    payload["root_pos_w"][:, 1] = radius * (1 - np.cos(angle))
    episode = summarise_episode("turn", payload)
    assert episode.heading_change_rad == pytest.approx(math.pi / 2, abs=0.05)
    assert episode.tortuosity > 1.0


def test_returning_to_the_start_does_not_divide_by_zero():
    payload = straight_payload(speed_mps=0.7)
    payload["root_pos_w"][:, 0] = np.concatenate(
        [np.linspace(0, 1, 50), np.linspace(1, 0, 50)]
    )
    episode = summarise_episode("loop", payload)
    assert math.isinf(episode.tortuosity)


def test_the_three_ranks_measure_different_things():
    """A corpus of distinct low-rank episodes has low within-rank but higher pooled rank.

    This is the distinction an earlier datasheet collapsed, quoting the within-episode
    figure as the corpus's coverage.
    """
    payloads = [straight_payload(speed_mps=0.6 + 0.1 * i, seed=i) for i in range(6)]
    episodes = [summarise_episode(f"e{i}", p) for i, p in enumerate(payloads)]
    actions = [p["action_motion_token"] for p in payloads]
    report = build_diversity_report(episodes, actions)
    assert report.within_episode_rank_mean == pytest.approx(2.0, abs=0.3)
    assert report.pooled_rank > report.within_episode_rank_mean
    assert report.between_episode_rank > 0.0


def test_identical_episodes_do_not_raise_the_pooled_rank():
    """Repeating one behaviour adds rows, not coverage."""
    payloads = [straight_payload(speed_mps=0.7, seed=0) for _ in range(8)]
    episodes = [summarise_episode(f"e{i}", p) for i, p in enumerate(payloads)]
    report = build_diversity_report(episodes, [p["action_motion_token"] for p in payloads])
    assert report.pooled_rank == pytest.approx(report.within_episode_rank_mean, abs=0.2)


def test_speed_spread_cv_separates_a_degenerate_corpus_from_a_varied_one():
    same = [straight_payload(speed_mps=0.70, seed=i) for i in range(5)]
    varied = [straight_payload(speed_mps=s, seed=i) for i, s in enumerate([0.3, 0.6, 0.9, 1.2, 1.5])]
    for payloads, expect_low in ((same, True), (varied, False)):
        episodes = [summarise_episode(f"e{i}", p) for i, p in enumerate(payloads)]
        report = build_diversity_report(episodes, [p["action_motion_token"] for p in payloads])
        cv = report.spreads["mean_speed_mps"]["cv"]
        assert (cv < 0.02) if expect_low else (cv > 0.3)


def test_crouching_shows_up_in_root_height():
    walk = straight_payload(speed_mps=0.7, height=0.75, seed=0)
    crouch = straight_payload(speed_mps=0.5, height=0.48, seed=1)
    episodes = [summarise_episode("walk", walk), summarise_episode("crouch", crouch)]
    report = build_diversity_report(
        episodes, [walk["action_motion_token"], crouch["action_motion_token"]]
    )
    assert report.spreads["root_height_min_m"]["min"] == pytest.approx(0.48)
    assert report.spreads["root_height_min_m"]["max"] == pytest.approx(0.75)


def test_mismatched_inputs_raise():
    payload = straight_payload(speed_mps=0.7)
    episode = summarise_episode("e", payload)
    with pytest.raises(ValueError, match="action blocks"):
        build_diversity_report([episode], [])
    with pytest.raises(ValueError, match="zero episodes"):
        build_diversity_report([], [])


def test_inconsistent_action_dimensions_raise():
    a = straight_payload(speed_mps=0.7, seed=0)
    b = straight_payload(speed_mps=0.8, seed=1)
    episodes = [summarise_episode("a", a), summarise_episode("b", b)]
    with pytest.raises(ValueError, match="inconsistent action dimensions"):
        build_diversity_report(
            episodes, [a["action_motion_token"], b["action_motion_token"][:, :4]]
        )
