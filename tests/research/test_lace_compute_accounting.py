from __future__ import annotations

import pytest

from gear_sonic.research.lace.compute import (
    ComputeBudget,
    assert_headline_matched_training_budget,
    assert_matched_training_budget,
)


def test_fixed_horizon_budget_counts_physics_and_optimizer_steps() -> None:
    budget = ComputeBudget(
        num_envs_per_rank=512,
        world_size=1,
        rollout_steps_per_iteration=24,
        iterations=200,
        ppo_epochs=5,
        minibatches_per_epoch=4,
        microbatches_per_minibatch=2,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=2,
    )

    assert budget.control_transitions == 2_457_600
    assert budget.physics_substeps == 9_830_400
    assert budget.physics_steps == budget.physics_substeps
    assert budget.planned_optimizer_step_calls == 8_000
    assert budget.planned_parameter_update_steps == 4_000
    assert not budget.as_manifest_record()["optimizer_step_count_is_observed"]


def test_matched_budget_rejects_different_factorization_with_same_totals() -> None:
    left = ComputeBudget(256, 1, 24, 100, 5, 4)
    right = ComputeBudget(128, 1, 24, 200, 5, 2)

    with pytest.raises(ValueError, match="training-schedule mismatch"):
        assert_matched_training_budget(left, right)


def test_matched_budget_accepts_identical_schedule_and_totals() -> None:
    left = ComputeBudget(256, 1, 24, 100, 5, 4)
    right = ComputeBudget(256, 1, 24, 100, 5, 4)

    assert_matched_training_budget(left, right)


def test_budget_comparison_rejects_physics_or_realized_update_mismatch() -> None:
    reference = ComputeBudget(256, 1, 24, 100, 5, 4, realized_optimizer_step_calls=2_000)
    physics_mismatch = ComputeBudget(128, 1, 24, 100, 5, 4)
    with pytest.raises(ValueError, match="training-schedule mismatch"):
        assert_matched_training_budget(reference, physics_mismatch)

    realized_mismatch = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        realized_optimizer_step_calls=1_999,
    )
    with pytest.raises(ValueError, match="realized_optimizer_step_calls mismatch"):
        assert_matched_training_budget(reference, realized_mismatch)


def test_budget_rejects_decimation_accumulation_and_config_hash_mismatches() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    reference = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=2,
        resolved_training_config_sha256=digest_a,
    )
    different_decimation = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=2,
        gradient_accumulation_steps=2,
        resolved_training_config_sha256=digest_a,
    )
    with pytest.raises(ValueError, match="physics_substeps_per_control_step"):
        assert_matched_training_budget(reference, different_decimation)

    different_accumulation = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=1,
        resolved_training_config_sha256=digest_a,
    )
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        assert_matched_training_budget(reference, different_accumulation)

    different_config = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=2,
        resolved_training_config_sha256=digest_b,
    )
    with pytest.raises(ValueError, match="resolved training-config mismatch"):
        assert_matched_training_budget(reference, different_config)


def test_headline_budget_requires_complete_realized_counters() -> None:
    digest = "a" * 64
    complete = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=2,
        realized_optimizer_step_calls=2_000,
        realized_parameter_update_steps=1_000,
        realized_skipped_optimizer_step_calls=0,
        resolved_training_config_sha256=digest,
    )
    assert_headline_matched_training_budget(complete, complete)

    unobserved = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        physics_substeps_per_control_step=4,
        gradient_accumulation_steps=2,
        resolved_training_config_sha256=digest,
    )
    with pytest.raises(ValueError, match="realized optimizer counters are required"):
        assert_headline_matched_training_budget(unobserved, unobserved)


def test_headline_budget_rejects_skipped_or_missing_parameter_updates() -> None:
    digest = "a" * 64
    skipped = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        realized_optimizer_step_calls=1_999,
        realized_parameter_update_steps=1_999,
        realized_skipped_optimizer_step_calls=1,
        resolved_training_config_sha256=digest,
    )
    with pytest.raises(ValueError, match="skipped optimizer calls"):
        assert_headline_matched_training_budget(skipped, skipped)

    incomplete = ComputeBudget(
        256,
        1,
        24,
        100,
        5,
        4,
        realized_optimizer_step_calls=2_000,
        realized_parameter_update_steps=1_999,
        realized_skipped_optimizer_step_calls=0,
        resolved_training_config_sha256=digest,
    )
    with pytest.raises(ValueError, match="realized parameter updates"):
        assert_headline_matched_training_budget(incomplete, incomplete)


def test_budget_rejects_nonintegral_accumulation_schedule_and_bad_digest() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        ComputeBudget(1, 1, 1, 1, 1, 3, gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ComputeBudget(1, 1, 1, 1, 1, 1, resolved_training_config_sha256="not-a-digest")
