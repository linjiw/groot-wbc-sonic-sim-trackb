"""Explicit interaction and optimizer accounting for LACE comparisons.

SONIC advances the simulator ``decimation`` times for every policy/control
transition.  Keeping those quantities separate avoids calling control
transitions "physics steps" and makes fixed-budget comparisons auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ComputeBudget:
    """One fixed-horizon SONIC training or shadow-update budget."""

    num_envs_per_rank: int
    world_size: int
    rollout_steps_per_iteration: int
    iterations: int
    ppo_epochs: int
    minibatches_per_epoch: int
    microbatches_per_minibatch: int = 1
    physics_substeps_per_control_step: int = 1
    gradient_accumulation_steps: int = 1
    realized_optimizer_step_calls: int | None = None
    realized_parameter_update_steps: int | None = None
    realized_skipped_optimizer_step_calls: int | None = None
    resolved_training_config_sha256: str | None = None

    def __post_init__(self) -> None:
        positive_fields = (
            "num_envs_per_rank",
            "world_size",
            "rollout_steps_per_iteration",
            "iterations",
            "ppo_epochs",
            "minibatches_per_epoch",
            "microbatches_per_minibatch",
            "physics_substeps_per_control_step",
            "gradient_accumulation_steps",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.planned_optimizer_step_calls % self.gradient_accumulation_steps:
            raise ValueError(
                "planned_optimizer_step_calls must be divisible by " "gradient_accumulation_steps"
            )
        if self.realized_optimizer_step_calls is not None:
            if not 0 <= self.realized_optimizer_step_calls <= self.planned_optimizer_step_calls:
                raise ValueError(
                    "realized_optimizer_step_calls must be between zero and the planned calls"
                )
        if self.realized_parameter_update_steps is not None:
            if not 0 <= self.realized_parameter_update_steps <= self.planned_parameter_update_steps:
                raise ValueError(
                    "realized_parameter_update_steps must be between zero and the planned updates"
                )
        if self.realized_skipped_optimizer_step_calls is not None:
            if (
                not 0
                <= self.realized_skipped_optimizer_step_calls
                <= (self.planned_optimizer_step_calls)
            ):
                raise ValueError(
                    "realized_skipped_optimizer_step_calls must be between zero and the "
                    "planned calls"
                )
        observed_call_counts = (
            self.realized_optimizer_step_calls,
            self.realized_skipped_optimizer_step_calls,
        )
        if all(value is not None for value in observed_call_counts):
            calls, skipped = observed_call_counts
            assert calls is not None and skipped is not None
            if calls + skipped > self.planned_optimizer_step_calls:
                raise ValueError(
                    "realized optimizer calls plus skipped calls exceed the planned calls"
                )
        if self.resolved_training_config_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.resolved_training_config_sha256
        ):
            raise ValueError("resolved_training_config_sha256 must be a lowercase SHA-256 digest")

    @property
    def control_transitions(self) -> int:
        return (
            self.num_envs_per_rank
            * self.world_size
            * self.rollout_steps_per_iteration
            * self.iterations
        )

    @property
    def physics_substeps(self) -> int:
        return self.control_transitions * self.physics_substeps_per_control_step

    @property
    def physics_steps(self) -> int:
        """Backward-compatible alias for actual simulator substeps."""

        return self.physics_substeps

    @property
    def planned_optimizer_step_calls(self) -> int:
        return (
            self.iterations
            * self.ppo_epochs
            * self.minibatches_per_epoch
            * self.microbatches_per_minibatch
        )

    @property
    def planned_parameter_update_steps(self) -> int:
        return self.planned_optimizer_step_calls // self.gradient_accumulation_steps

    def as_manifest_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["control_transitions"] = self.control_transitions
        record["physics_substeps"] = self.physics_substeps
        record["physics_steps"] = self.physics_steps
        record["physics_steps_semantics"] = "alias_of_physics_substeps"
        record["planned_optimizer_step_calls"] = self.planned_optimizer_step_calls
        record["planned_parameter_update_steps"] = self.planned_parameter_update_steps
        record["optimizer_step_count_is_observed"] = all(
            value is not None
            for value in (
                self.realized_optimizer_step_calls,
                self.realized_parameter_update_steps,
                self.realized_skipped_optimizer_step_calls,
            )
        )
        return record


def assert_matched_training_budget(left: ComputeBudget, right: ComputeBudget) -> None:
    """Reject comparisons that change either totals or the update schedule.

    LACE headline arms hold the measured throughput-selected environment count
    and the complete PPO schedule fixed. Equal aggregate totals are insufficient:
    changing rollout batch size, update cadence, or minibatch factorization can
    change PPO optimization even when the two headline counters coincide.
    """

    schedule_fields = (
        "num_envs_per_rank",
        "world_size",
        "rollout_steps_per_iteration",
        "iterations",
        "ppo_epochs",
        "minibatches_per_epoch",
        "microbatches_per_minibatch",
        "physics_substeps_per_control_step",
        "gradient_accumulation_steps",
    )
    mismatched_fields = [
        name for name in schedule_fields if getattr(left, name) != getattr(right, name)
    ]
    if mismatched_fields:
        raise ValueError(
            "training-schedule mismatch: "
            + ", ".join(
                f"{name}={getattr(left, name)} vs {getattr(right, name)}"
                for name in mismatched_fields
            )
        )

    if left.control_transitions != right.control_transitions:
        raise ValueError(
            "control-transition mismatch: "
            f"left={left.control_transitions}, right={right.control_transitions}"
        )
    if left.physics_substeps != right.physics_substeps:
        raise ValueError(
            f"physics-substep mismatch: left={left.physics_substeps}, "
            f"right={right.physics_substeps}"
        )
    if left.planned_optimizer_step_calls != right.planned_optimizer_step_calls:
        raise ValueError(
            "planned optimizer-step mismatch: "
            f"left={left.planned_optimizer_step_calls}, "
            f"right={right.planned_optimizer_step_calls}"
        )
    if left.planned_parameter_update_steps != right.planned_parameter_update_steps:
        raise ValueError(
            "planned parameter-update mismatch: "
            f"left={left.planned_parameter_update_steps}, "
            f"right={right.planned_parameter_update_steps}"
        )

    observed_fields = (
        "realized_optimizer_step_calls",
        "realized_parameter_update_steps",
        "realized_skipped_optimizer_step_calls",
    )
    for name in observed_fields:
        observed = (getattr(left, name), getattr(right, name))
        if (observed[0] is None) != (observed[1] is None):
            raise ValueError(f"{name} observation status differs between arms")
        if observed[0] is not None and observed[0] != observed[1]:
            raise ValueError(f"{name} mismatch: left={observed[0]}, right={observed[1]}")

    hashes = (
        left.resolved_training_config_sha256,
        right.resolved_training_config_sha256,
    )
    if (hashes[0] is None) != (hashes[1] is None):
        raise ValueError("resolved training-config hash status differs between arms")
    if hashes[0] is not None and hashes[0] != hashes[1]:
        raise ValueError(f"resolved training-config mismatch: left={hashes[0]}, right={hashes[1]}")


def assert_headline_matched_training_budget(left: ComputeBudget, right: ComputeBudget) -> None:
    """Apply the stricter completed-run contract used by headline comparisons.

    A headline result needs an identical resolved training configuration and
    measured counters proving that every planned optimizer call and synchronized
    parameter update occurred.  A run with a non-finite-gradient skip is not a
    fixed-update-budget comparison and fails closed.
    """

    assert_matched_training_budget(left, right)
    for side, budget in (("left", left), ("right", right)):
        if budget.resolved_training_config_sha256 is None:
            raise ValueError(f"{side} resolved training-config hash is required")
        observed = (
            budget.realized_optimizer_step_calls,
            budget.realized_parameter_update_steps,
            budget.realized_skipped_optimizer_step_calls,
        )
        if any(value is None for value in observed):
            raise ValueError(f"{side} realized optimizer counters are required")
        calls, updates, skipped = observed
        assert calls is not None and updates is not None and skipped is not None
        if skipped:
            raise ValueError(f"{side} has {skipped} skipped optimizer calls")
        if calls != budget.planned_optimizer_step_calls:
            raise ValueError(
                f"{side} realized optimizer calls={calls}, "
                f"planned={budget.planned_optimizer_step_calls}"
            )
        if updates != budget.planned_parameter_update_steps:
            raise ValueError(
                f"{side} realized parameter updates={updates}, "
                f"planned={budget.planned_parameter_update_steps}"
            )
