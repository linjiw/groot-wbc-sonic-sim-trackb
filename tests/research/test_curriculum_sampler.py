from __future__ import annotations

import random

import pytest

from scripts.research.curriculum_sampler import (
    StageState,
    learning_progress,
    sample_stage,
    stage_sampling_weights,
    update_difficulty,
)


def test_learning_progress_uses_absolute_ema_delta() -> None:
    state = StageState(stage_id="S2", ema_success=0.40, prev_ema_success=0.65)

    assert learning_progress(state) == pytest.approx(0.25)


def test_sampling_weights_include_anchor_replay_and_normalize() -> None:
    states = {
        "S1": StageState(
            stage_id="S1",
            unlocked=True,
            ema_success=0.80,
            prev_ema_success=0.70,
        ),
        "S2": StageState(
            stage_id="S2",
            unlocked=True,
            ema_success=0.50,
            prev_ema_success=0.10,
        ),
        "S3": StageState(stage_id="S3", unlocked=False),
    }

    weights = stage_sampling_weights(states, anchor_stage_ids=["S1"], beta=2.0, anchor_mass=0.20)

    assert set(weights) == {"S1", "S2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["S1"] >= 0.20
    assert weights["S2"] > weights["S1"] - 0.20


def test_sampling_requires_unlocked_stage() -> None:
    states = {"S1": StageState(stage_id="S1", unlocked=False)}

    with pytest.raises(ValueError, match="No unlocked"):
        stage_sampling_weights(states, anchor_stage_ids=[])


def test_sample_stage_uses_provided_rng() -> None:
    sampled = sample_stage({"S1": 1.0, "S2": 0.0}, rng=random.Random(7))

    assert sampled == "S1"


def test_update_difficulty_is_conservative() -> None:
    assert update_difficulty(0.50, {"success_rate": 0.95, "fall_rate": 0.0}) == pytest.approx(
        0.55
    )
    assert update_difficulty(0.50, {"success_rate": 0.40, "fall_rate": 0.0}) == pytest.approx(
        0.45
    )
    assert update_difficulty(
        0.50,
        {"success_rate": 0.95, "fall_rate": 0.09, "drop_rate": 0.0},
    ) == pytest.approx(0.40)
    assert update_difficulty(0.03, {"success_rate": 0.20}) == pytest.approx(0.0)
