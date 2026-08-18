"""Fail-closed Isaac Lab boundary for recording LACE probe episodes.

This module is safe to import in an ordinary CPU process: neither :mod:`torch`
nor :mod:`isaaclab` is imported at module import time.  The live recorder types
are constructed only by :func:`load_isaac_recorder_types`, which must be called
after Isaac Lab's ``AppLauncher`` has started the application.

Hydra can opt in after launch with either of these targets::

    _target_: gear_sonic.research.lace.isaac_recorder.create_isaac_recorder_cfg

or, through the lazy module attribute::

    _target_: gear_sonic.research.lace.isaac_recorder.LaceIsaacRecorderCfg

The adapter intentionally has no fallback tensor names.  A missing sensor,
body, torque tensor, termination term, or metadata field raises immediately;
an invalid atlas channel must never be recorded as an innocent column of zeros.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import textwrap
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gear_sonic.research.lace.atlas import SCIENTIFIC_SEED_SEMANTICS
from gear_sonic.research.lace.atlas_probe_mode import (
    AtlasProbeBatch,
    require_runtime_rng_seed_readback,
)
from gear_sonic.research.lace.probes import ProbeThresholds, compute_episode_probe
from gear_sonic.research.lace.schedule import DOMAIN_RANDOMIZATION_SEED_SEMANTICS
from gear_sonic.research.lace.schema import (
    ATLAS_V1_INTERVAL_EVENT_POLICY,
    ROBOT_CONTRACT_READBACK_KIND,
    ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
    RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
    RUNTIME_REALIZATION_KIND,
    RUNTIME_REALIZATION_SCHEMA_VERSION,
    TERMINATION_TRACE_ALGORITHM,
    TERMINATION_TRACE_KIND,
    TERMINATION_TRACE_SCHEMA_VERSION,
    canonical_sha256,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_ROBOT_RANDOMIZATION_FIELDS = (
    "material_properties",
    "masses",
    "inertias",
    "centers_of_mass",
    "default_joint_position",
    "joint_action_offset",
)
_POST_RESET_STATE_FIELDS = (
    "root_position_w",
    "root_quaternion_wxyz",
    "root_linear_velocity_w",
    "root_angular_velocity_w",
    "joint_position",
    "joint_velocity",
)
_SCHEDULED_REFERENCE_STATE_FIELDS = (
    "command_time_step",
    "motion_id",
    "anchor_position_w",
    "anchor_quaternion_wxyz",
    "body_position_w",
    "body_quaternion_wxyz",
    "body_linear_velocity_w",
    "body_angular_velocity_w",
    "joint_position",
    "joint_velocity",
    "left_foot_contact",
    "right_foot_contact",
)
_ROBOT_CONTRACT_FIELDS = (
    "joint_pos_limits",
    "soft_joint_pos_limits",
    "joint_vel_limits",
    "soft_joint_vel_limits",
)

EXPECTED_ISAACLAB_TERMINATION_COMPUTE_SHA256 = (
    "d6e3e411f73f62aa302422f6df617d48ce9065f0182b29a6642860b8a1cb7f7f"
)


@dataclass(frozen=True)
class IsaacBindings:
    """Explicit names needed to read one SONIC post-step frame."""

    command_name: str = "motion"
    robot_name: str = "robot"
    joint_action_name: str = "joint_pos"
    contact_sensor_name: str = "contact_forces"
    foot_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    contact_force_threshold: float = 10.0
    ground_normal_axis: int = 2
    fall_termination_terms: tuple[str, ...] = ()
    timeout_termination_terms: tuple[str, ...] = ("time_out",)

    def validate(self) -> None:
        for name, value in (
            ("command_name", self.command_name),
            ("robot_name", self.robot_name),
            ("joint_action_name", self.joint_action_name),
            ("contact_sensor_name", self.contact_sensor_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.foot_body_names) != 2 or any(not name for name in self.foot_body_names):
            raise ValueError("foot_body_names must contain exactly two non-empty names")
        if len(set(self.foot_body_names)) != 2:
            raise ValueError("foot_body_names must be distinct")
        if not math.isfinite(self.contact_force_threshold) or self.contact_force_threshold <= 0.0:
            raise ValueError("contact_force_threshold must be finite and positive")
        if (
            not isinstance(self.ground_normal_axis, int)
            or isinstance(self.ground_normal_axis, bool)
            or self.ground_normal_axis not in (0, 1, 2)
        ):
            raise ValueError("ground_normal_axis must be one of 0, 1, or 2")
        if any(not isinstance(name, str) or not name for name in self.fall_termination_terms):
            raise ValueError("fall_termination_terms must contain non-empty strings")
        if len(set(self.fall_termination_terms)) != len(self.fall_termination_terms):
            raise ValueError("fall_termination_terms must be unique")
        if any(not isinstance(name, str) or not name for name in self.timeout_termination_terms):
            raise ValueError("timeout_termination_terms must contain non-empty strings")
        if len(set(self.timeout_termination_terms)) != len(self.timeout_termination_terms):
            raise ValueError("timeout_termination_terms must be unique")


@dataclass(frozen=True)
class RolloutRun:
    """Run-level identity shared by every vectorized environment."""

    policy_id: str
    domain_randomization_seed: int
    partition: str = "D_atlas"
    rollout_id_prefix: str = "lace"
    repeat_index: int = 0
    repeat_indices_by_env: tuple[int, ...] | None = None

    def validate(self, num_envs: int) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(self.domain_randomization_seed, int) or isinstance(
            self.domain_randomization_seed, bool
        ):
            raise ValueError("domain_randomization_seed must be an integer")
        if self.partition != "D_atlas":
            raise ValueError("the failure-atlas recorder partition must be 'D_atlas'")
        if not isinstance(self.rollout_id_prefix, str) or not self.rollout_id_prefix:
            raise ValueError("rollout_id_prefix must be a non-empty string")
        if not isinstance(self.repeat_index, int) or isinstance(self.repeat_index, bool):
            raise ValueError("repeat_index must be an integer")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be nonnegative")
        if self.repeat_indices_by_env is not None:
            if len(self.repeat_indices_by_env) != num_envs:
                raise ValueError("repeat_indices_by_env length must match num_envs")
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in self.repeat_indices_by_env
            ):
                raise ValueError("repeat_indices_by_env must contain nonnegative integers")

    def repeats(self, num_envs: int) -> tuple[int, ...]:
        self.validate(num_envs)
        if self.repeat_indices_by_env is not None:
            return self.repeat_indices_by_env
        return (self.repeat_index,) * num_envs


@dataclass(frozen=True)
class RolloutMetadata:
    rollout_id: str
    motion_key: str
    policy_id: str
    domain_randomization_seed: int
    partition: str
    reference_start_step: int
    reference_num_steps: int
    initial_phase: float
    repeat_index: int
    env_index: int
    motion_id: int | None = None
    checkpoint_sha256: str | None = None
    runtime_rng_seed: int | None = None
    runtime_rng_seed_readback: int | None = None
    phase_id: str | None = None
    target_fraction: float | None = None
    realized_fraction: float | None = None
    split_sha256: str | None = None
    split_selection_sha256: str | None = None
    schedule_sha256: str | None = None
    schedule_entry_id: str | None = None
    domain_randomization_realization: Mapping[str, Any] | None = None
    domain_randomization_realization_sha256: str | None = None
    runtime_rng_seed_semantics: str | None = None
    robot_contract_readback: Mapping[str, Any] | None = None
    robot_contract_readback_sha256: str | None = None

    @property
    def scientific_identity(self) -> tuple[Any, ...]:
        return (
            self.rollout_id,
            self.motion_key,
            self.policy_id,
            self.domain_randomization_seed,
            self.partition,
            self.reference_start_step,
            self.reference_num_steps,
            self.repeat_index,
            self.motion_id,
            self.checkpoint_sha256,
            self.runtime_rng_seed,
            self.runtime_rng_seed_readback,
            self.phase_id,
            self.target_fraction,
            self.realized_fraction,
            self.split_sha256,
            self.split_selection_sha256,
            self.schedule_sha256,
            self.schedule_entry_id,
            self.domain_randomization_realization_sha256,
            self.runtime_rng_seed_semantics,
            self.robot_contract_readback_sha256,
        )


@dataclass(frozen=True)
class RuntimeRealization:
    """Canonical post-reset state bound to one frozen rollout row."""

    env_index: int
    rollout_id: str
    record: Mapping[str, Any]
    sha256: str
    robot_contract_readback: Mapping[str, Any]
    robot_contract_readback_sha256: str

    def validate(self) -> None:
        if not isinstance(self.env_index, int) or isinstance(self.env_index, bool):
            raise ValueError("runtime realization env_index must be an integer")
        if self.env_index < 0:
            raise ValueError("runtime realization env_index must be nonnegative")
        if not isinstance(self.rollout_id, str) or not self.rollout_id:
            raise ValueError("runtime realization rollout_id must be a non-empty string")
        if not isinstance(self.record, Mapping):
            raise TypeError("runtime realization record must be a mapping")
        if canonical_sha256(self.record) != self.sha256:
            raise ValueError("runtime realization SHA-256 does not match its canonical record")
        if not isinstance(self.robot_contract_readback, Mapping):
            raise TypeError("robot contract readback must be a mapping")
        if canonical_sha256(self.robot_contract_readback) != self.robot_contract_readback_sha256:
            raise ValueError("robot contract readback SHA-256 does not match its record")


@dataclass(frozen=True)
class TerminationTraceSnapshot:
    """One fresh, independently retained raw termination-term matrix."""

    generation: int
    common_step_counter: int
    term_names: tuple[str, ...]
    time_out_flags: tuple[bool, ...]
    values: BoolArray
    contract: Mapping[str, Any]
    contract_sha256: str

    def validate(self) -> None:
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation <= 0
        ):
            raise ValueError("termination trace generation must be a positive integer")
        if (
            not isinstance(self.common_step_counter, int)
            or isinstance(self.common_step_counter, bool)
            or self.common_step_counter < 0
        ):
            raise ValueError("termination trace common_step_counter must be nonnegative")
        if not self.term_names or len(self.term_names) != len(set(self.term_names)):
            raise ValueError("termination trace term_names must be non-empty and unique")
        if len(self.time_out_flags) != len(self.term_names) or any(
            not isinstance(value, bool) for value in self.time_out_flags
        ):
            raise ValueError("termination trace time_out_flags must align with term_names")
        values = _binary_array(self.values, "termination trace values", 2)
        if values.shape[1] != len(self.term_names):
            raise ValueError("termination trace values must align with term_names")
        if not isinstance(self.contract, Mapping):
            raise TypeError("termination trace contract must be a mapping")
        if canonical_sha256(self.contract) != self.contract_sha256:
            raise ValueError("termination trace contract SHA-256 mismatch")


@dataclass(frozen=True)
class ProbeFrameBatch:
    """One pre-reset post-step frame for every vectorized environment."""

    reference_contacts: BoolArray
    actual_contacts: BoolArray
    foot_tangential_speed: FloatArray
    base_translation_error: FloatArray
    base_orientation_error: FloatArray
    base_tilt: FloatArray
    requested_torque: FloatArray
    applied_torque: FloatArray
    effort_limits: FloatArray
    reference_joint_position: FloatArray
    joint_position: FloatArray
    joint_soft_lower_limits: FloatArray
    joint_soft_upper_limits: FloatArray
    local_pose_error: FloatArray
    episode_end_mask: BoolArray
    failure_mask: BoolArray
    fall_mask: BoolArray
    termination_terms: dict[str, BoolArray]
    fall_termination_terms: tuple[str, ...]
    timeout_termination_terms: tuple[str, ...]
    termination_semantics: str
    termination_multi_hot_available: bool
    termination_trace_contract: Mapping[str, Any] | None
    termination_trace_contract_sha256: str | None
    termination_trace_generation: int | None
    termination_trace_step_counter: int | None

    @property
    def num_envs(self) -> int:
        return int(self.reference_contacts.shape[0])


def _numeric_array(values: ArrayLike, name: str, ndim: int) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _binary_array(values: ArrayLike, name: str, ndim: int) -> BoolArray:
    result = np.asarray(values)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if np.issubdtype(result.dtype, np.bool_):
        return result.astype(np.bool_, copy=False)
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must be boolean or binary numeric")
    numeric = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must contain only boolean or binary values")
    return numeric.astype(np.bool_)


def _quaternion_array(values: ArrayLike, name: str, num_envs: int) -> FloatArray:
    result = _numeric_array(values, name, 2)
    if result.shape != (num_envs, 4):
        raise ValueError(f"{name} must have shape [{num_envs}, 4]")
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError(f"{name} contains a zero-norm quaternion")
    return result / norm


def _quaternion_error_radians(reference: FloatArray, actual: FloatArray) -> FloatArray:
    # q and -q encode the same rotation, hence abs(dot).
    cosine_half_angle = np.clip(np.abs(np.sum(reference * actual, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(cosine_half_angle)


def _gravity_vector_in_body(quaternion_wxyz: FloatArray) -> FloatArray:
    """Rotate unit world gravity into each body frame for ``wxyz`` quaternions."""

    w = quaternion_wxyz[:, 0]
    x = quaternion_wxyz[:, 1]
    y = quaternion_wxyz[:, 2]
    z = quaternion_wxyz[:, 3]
    return np.stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        axis=1,
    )


def _gravity_residual_radians(reference: FloatArray, actual: FloatArray) -> FloatArray:
    reference_gravity = _gravity_vector_in_body(reference)
    actual_gravity = _gravity_vector_in_body(actual)
    cosine = np.clip(np.sum(reference_gravity * actual_gravity, axis=1), -1.0, 1.0)
    return np.arccos(cosine)


def assemble_probe_frame_batch(
    *,
    reference_left_contact: ArrayLike,
    reference_right_contact: ArrayLike,
    foot_contact_force_w: ArrayLike,
    foot_linear_velocity_w: ArrayLike,
    reference_anchor_position_w: ArrayLike,
    actual_anchor_position_w: ArrayLike,
    reference_anchor_quaternion_wxyz: ArrayLike,
    actual_anchor_quaternion_wxyz: ArrayLike,
    requested_torque: ArrayLike,
    applied_torque: ArrayLike,
    effort_limits: ArrayLike,
    reference_joint_position: ArrayLike,
    joint_position: ArrayLike,
    joint_soft_limits: ArrayLike,
    aligned_reference_body_position_w: ArrayLike,
    actual_body_position_w: ArrayLike,
    episode_end_mask: ArrayLike,
    failure_mask: ArrayLike,
    fall_mask: ArrayLike,
    termination_terms: Mapping[str, ArrayLike],
    fall_termination_terms: Sequence[str] = (),
    timeout_termination_terms: Sequence[str] = ("time_out",),
    termination_semantics: str = "manager_last_trigger_wins_masked_to_current_end",
    termination_multi_hot_available: bool = False,
    termination_trace_contract: Mapping[str, Any] | None = None,
    termination_trace_contract_sha256: str | None = None,
    termination_trace_generation: int | None = None,
    termination_trace_step_counter: int | None = None,
    contact_force_threshold: float,
    ground_normal_axis: int = 2,
) -> ProbeFrameBatch:
    """Map explicit Isaac tensors to the pure episode-probe frame contract.

    Isaac Lab quaternions are required in ``wxyz`` order.  The reference body
    positions must already be SONIC's anchor/heading-aligned
    ``body_pos_relative_w`` values; substituting raw world-frame reference poses
    would conflate global drift with local pose divergence and is rejected only
    by this documented binding contract, since the two tensors share a shape.
    """

    if not math.isfinite(contact_force_threshold) or contact_force_threshold <= 0.0:
        raise ValueError("contact_force_threshold must be finite and positive")
    if (
        not isinstance(ground_normal_axis, int)
        or isinstance(ground_normal_axis, bool)
        or ground_normal_axis not in (0, 1, 2)
    ):
        raise ValueError("ground_normal_axis must be one of 0, 1, or 2")

    left_contact = _binary_array(reference_left_contact, "reference_left_contact", 1)
    num_envs = int(left_contact.shape[0])
    right_contact = _binary_array(reference_right_contact, "reference_right_contact", 1)
    if right_contact.shape != (num_envs,):
        raise ValueError("reference_right_contact shape must match reference_left_contact")
    reference_contacts = np.stack((left_contact, right_contact), axis=1)

    contact_force = _numeric_array(foot_contact_force_w, "foot_contact_force_w", 3)
    foot_velocity = _numeric_array(foot_linear_velocity_w, "foot_linear_velocity_w", 3)
    expected_feet_shape = (num_envs, 2, 3)
    if contact_force.shape != expected_feet_shape:
        raise ValueError(f"foot_contact_force_w must have shape {expected_feet_shape}")
    if foot_velocity.shape != expected_feet_shape:
        raise ValueError(f"foot_linear_velocity_w must have shape {expected_feet_shape}")
    force_norm = np.linalg.norm(contact_force, axis=2)
    actual_contacts = force_norm > contact_force_threshold
    # The primary atlas is flat-plane only. Resultant contact force contains
    # friction and is not a ground-normal estimate, so removing velocity along
    # that vector would erase part of the slip signal itself.
    tangential_velocity = foot_velocity.copy()
    tangential_velocity[:, :, ground_normal_axis] = 0.0
    foot_tangential_speed = np.linalg.norm(tangential_velocity, axis=2)
    foot_tangential_speed[~actual_contacts] = 0.0

    reference_anchor_position = _numeric_array(
        reference_anchor_position_w,
        "reference_anchor_position_w",
        2,
    )
    actual_anchor_position = _numeric_array(
        actual_anchor_position_w,
        "actual_anchor_position_w",
        2,
    )
    if reference_anchor_position.shape != (num_envs, 3):
        raise ValueError(f"reference_anchor_position_w must have shape [{num_envs}, 3]")
    if actual_anchor_position.shape != (num_envs, 3):
        raise ValueError(f"actual_anchor_position_w must have shape [{num_envs}, 3]")
    base_translation_error = reference_anchor_position - actual_anchor_position

    reference_quaternion = _quaternion_array(
        reference_anchor_quaternion_wxyz,
        "reference_anchor_quaternion_wxyz",
        num_envs,
    )
    actual_quaternion = _quaternion_array(
        actual_anchor_quaternion_wxyz,
        "actual_anchor_quaternion_wxyz",
        num_envs,
    )
    base_orientation_error = _quaternion_error_radians(reference_quaternion, actual_quaternion)
    base_tilt = _gravity_residual_radians(reference_quaternion, actual_quaternion)

    requested = _numeric_array(requested_torque, "requested_torque", 2)
    torque = _numeric_array(applied_torque, "applied_torque", 2)
    effort = _numeric_array(effort_limits, "effort_limits", 2)
    reference_position = _numeric_array(
        reference_joint_position,
        "reference_joint_position",
        2,
    )
    position = _numeric_array(joint_position, "joint_position", 2)
    soft_limits = _numeric_array(joint_soft_limits, "joint_soft_limits", 3)
    if requested.shape[0] != num_envs:
        raise ValueError("requested_torque first dimension must match num_envs")
    joint_count = int(requested.shape[1])
    if torque.shape != (num_envs, joint_count):
        raise ValueError("applied_torque shape must match requested_torque")
    if effort.shape != (num_envs, joint_count):
        raise ValueError("effort_limits shape must match requested_torque")
    if np.any(effort <= 0.0):
        raise ValueError("effort_limits must be strictly positive")
    if reference_position.shape != (num_envs, joint_count):
        raise ValueError("reference_joint_position shape must match requested_torque")
    if position.shape != (num_envs, joint_count):
        raise ValueError("joint_position shape must match requested_torque")
    if soft_limits.shape != (num_envs, joint_count, 2):
        raise ValueError(f"joint_soft_limits must have shape [{num_envs}, {joint_count}, 2]")
    if np.any(soft_limits[:, :, 1] <= soft_limits[:, :, 0]):
        raise ValueError("joint_soft_limits upper values must exceed lower values")

    reference_body_position = _numeric_array(
        aligned_reference_body_position_w,
        "aligned_reference_body_position_w",
        3,
    )
    actual_body_position = _numeric_array(
        actual_body_position_w,
        "actual_body_position_w",
        3,
    )
    if reference_body_position.shape != actual_body_position.shape:
        raise ValueError("reference and actual body-position shapes must match")
    if reference_body_position.shape[0] != num_envs or reference_body_position.shape[2] != 3:
        raise ValueError("body positions must have shape [num_envs, num_bodies, 3]")
    local_pose_error = (reference_body_position - actual_body_position).reshape(num_envs, -1)

    episode_end = _binary_array(episode_end_mask, "episode_end_mask", 1)
    failures = _binary_array(failure_mask, "failure_mask", 1)
    falls = _binary_array(fall_mask, "fall_mask", 1)
    for array, name in (
        (episode_end, "episode_end_mask"),
        (failures, "failure_mask"),
        (falls, "fall_mask"),
    ):
        if array.shape != (num_envs,):
            raise ValueError(f"{name} must have shape [{num_envs}]")
    if np.any(failures & ~episode_end):
        raise ValueError("failure_mask may only be true at an episode end")
    if np.any(falls & ~failures):
        raise ValueError("fall_mask may only be true where failure_mask is true")

    if not isinstance(termination_terms, Mapping) or not termination_terms:
        raise ValueError("termination_terms must contain every active termination term")
    term_arrays: dict[str, BoolArray] = {}
    for name, values in termination_terms.items():
        if not isinstance(name, str) or not name:
            raise ValueError("termination term names must be non-empty strings")
        array = _binary_array(values, f"termination_terms[{name!r}]", 1)
        if array.shape != (num_envs,):
            raise ValueError(f"termination term {name!r} must have shape [{num_envs}]")
        term_arrays[name] = array
    fall_term_names = tuple(fall_termination_terms)
    if any(not isinstance(name, str) or not name for name in fall_term_names):
        raise ValueError("fall_termination_terms must contain non-empty strings")
    if len(set(fall_term_names)) != len(fall_term_names):
        raise ValueError("fall_termination_terms must be unique")
    missing_fall_terms = sorted(set(fall_term_names) - set(term_arrays))
    if missing_fall_terms:
        raise ValueError(
            f"fall termination terms are absent from termination_terms: {missing_fall_terms}"
        )
    timeout_term_names = tuple(timeout_termination_terms)
    if any(not isinstance(name, str) or not name for name in timeout_term_names):
        raise ValueError("timeout_termination_terms must contain non-empty strings")
    if len(set(timeout_term_names)) != len(timeout_term_names):
        raise ValueError("timeout_termination_terms must be unique")
    missing_timeout_terms = sorted(set(timeout_term_names) - set(term_arrays))
    if missing_timeout_terms:
        raise ValueError(
            f"timeout termination terms are absent from termination_terms: {missing_timeout_terms}"
        )
    if not isinstance(termination_semantics, str) or not termination_semantics:
        raise ValueError("termination_semantics must be a non-empty string")
    if not isinstance(termination_multi_hot_available, bool):
        raise TypeError("termination_multi_hot_available must be boolean")
    if termination_multi_hot_available:
        if not isinstance(termination_trace_contract, Mapping):
            raise ValueError("multi-hot termination frames require a trace contract")
        if termination_trace_contract.get("kind") != TERMINATION_TRACE_KIND:
            raise ValueError("termination trace contract kind is invalid")
        if termination_trace_contract.get("schema_version") != TERMINATION_TRACE_SCHEMA_VERSION:
            raise ValueError("termination trace contract schema version is invalid")
        if canonical_sha256(termination_trace_contract) != termination_trace_contract_sha256:
            raise ValueError("termination trace contract SHA-256 mismatch")
        if termination_trace_contract.get("term_names") != list(term_arrays):
            raise ValueError("termination trace term order drifted from termination_terms")
        expected_timeout_flags = [name in timeout_term_names for name in term_arrays]
        if termination_trace_contract.get("time_out_flags") != expected_timeout_flags:
            raise ValueError("termination trace timeout flags drifted from configured terms")
        for name, value in (
            ("termination_trace_generation", termination_trace_generation),
            ("termination_trace_step_counter", termination_trace_step_counter),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if name.endswith("generation") else 0)
            ):
                raise ValueError(f"{name} is invalid")
    elif any(
        value is not None
        for value in (
            termination_trace_contract,
            termination_trace_contract_sha256,
            termination_trace_generation,
            termination_trace_step_counter,
        )
    ):
        raise ValueError("non-multi-hot frames may not claim termination trace provenance")
    for name, array in term_arrays.items():
        # TerminationManager.get_term is a cached last-trigger signal. Restrict
        # it to the current end/failure rather than treating it as a fresh
        # multi-hot predicate on every post-step.
        scope = episode_end if name in timeout_term_names else failures
        term_arrays[name] = array & scope

    return ProbeFrameBatch(
        reference_contacts=reference_contacts,
        actual_contacts=actual_contacts,
        foot_tangential_speed=foot_tangential_speed,
        base_translation_error=base_translation_error,
        base_orientation_error=base_orientation_error,
        base_tilt=base_tilt,
        requested_torque=requested,
        applied_torque=torque,
        effort_limits=effort,
        reference_joint_position=reference_position,
        joint_position=position,
        joint_soft_lower_limits=soft_limits[:, :, 0],
        joint_soft_upper_limits=soft_limits[:, :, 1],
        local_pose_error=local_pose_error,
        episode_end_mask=episode_end,
        failure_mask=failures,
        fall_mask=falls,
        termination_terms=term_arrays,
        fall_termination_terms=fall_term_names,
        timeout_termination_terms=timeout_term_names,
        termination_semantics=termination_semantics,
        termination_multi_hot_available=termination_multi_hot_available,
        termination_trace_contract=termination_trace_contract,
        termination_trace_contract_sha256=termination_trace_contract_sha256,
        termination_trace_generation=termination_trace_generation,
        termination_trace_step_counter=termination_trace_step_counter,
    )


def assemble_rollout_metadata(
    *,
    motion_ids: ArrayLike,
    motion_start_steps: ArrayLike,
    motion_num_steps: ArrayLike,
    motion_keys: Sequence[str],
    run: RolloutRun,
) -> list[RolloutMetadata]:
    """Build deterministic per-environment rollout identities."""

    ids = _numeric_array(motion_ids, "motion_ids", 1)
    starts = _numeric_array(motion_start_steps, "motion_start_steps", 1)
    lengths = _numeric_array(motion_num_steps, "motion_num_steps", 1)
    num_envs = int(ids.size)
    if starts.shape != (num_envs,) or lengths.shape != (num_envs,):
        raise ValueError("motion metadata arrays must have matching shapes")
    if not np.all(ids == np.floor(ids)):
        raise ValueError("motion_ids must contain integers")
    integer_ids = ids.astype(np.int64)
    if np.any(integer_ids < 0) or np.any(integer_ids >= len(motion_keys)):
        raise ValueError("motion_ids reference an unavailable motion key")
    if any(not isinstance(key, str) or not key for key in motion_keys):
        raise ValueError("motion_keys must contain non-empty strings")
    if np.any(lengths < 1.0) or not np.all(lengths == np.floor(lengths)):
        raise ValueError("motion_num_steps must contain positive integers")
    maximum_start = np.maximum(lengths - 1.0, 0.0)
    if np.any(starts < 0.0) or np.any(starts > maximum_start):
        raise ValueError("motion_start_steps must lie inside each reference clip")
    if not np.all(starts == np.floor(starts)):
        raise ValueError("motion_start_steps must contain integers")

    repeats = run.repeats(num_envs)
    metadata: list[RolloutMetadata] = []
    for env_index in range(num_envs):
        motion_key = motion_keys[int(integer_ids[env_index])]
        denominator = max(int(lengths[env_index]) - 1, 1)
        phase = float(starts[env_index] / denominator)
        identity = {
            "motion_key": motion_key,
            "policy_id": run.policy_id,
            "domain_randomization_seed": run.domain_randomization_seed,
            "partition": run.partition,
            "reference_start_step": int(starts[env_index]),
            "reference_num_steps": int(lengths[env_index]),
            "initial_phase": phase,
            "repeat_index": repeats[env_index],
        }
        digest = hashlib.sha256(
            json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        metadata.append(
            RolloutMetadata(
                rollout_id=f"{run.rollout_id_prefix}:{digest}",
                motion_key=motion_key,
                policy_id=run.policy_id,
                domain_randomization_seed=run.domain_randomization_seed,
                partition=run.partition,
                reference_start_step=int(starts[env_index]),
                reference_num_steps=int(lengths[env_index]),
                initial_phase=phase,
                repeat_index=repeats[env_index],
                env_index=env_index,
                motion_id=int(integer_ids[env_index]),
            )
        )
    if len({item.rollout_id for item in metadata}) != len(metadata):
        raise ValueError(
            "duplicate rollout schedules across vectorized environments; assign distinct "
            "repeat_indices_by_env"
        )
    return metadata


def assemble_atlas_probe_metadata(
    *,
    batch: AtlasProbeBatch,
    motion_ids: ArrayLike,
    motion_start_steps: ArrayLike,
    motion_num_steps: ArrayLike,
    motion_keys: Sequence[str],
    process_seed_readback: Any,
) -> list[RolloutMetadata]:
    """Bind live command state to exact rows from a frozen atlas batch."""

    if not isinstance(batch, AtlasProbeBatch):
        raise TypeError("batch must be an AtlasProbeBatch")
    applied_seed = require_runtime_rng_seed_readback(batch, process_seed_readback)
    ids = _numeric_array(motion_ids, "motion_ids", 1)
    starts = _numeric_array(motion_start_steps, "motion_start_steps", 1)
    lengths = _numeric_array(motion_num_steps, "motion_num_steps", 1)
    expected_shape = (batch.num_envs,)
    if (
        ids.shape != expected_shape
        or starts.shape != expected_shape
        or lengths.shape != expected_shape
    ):
        raise ValueError(
            "live atlas motion metadata arrays must each match AtlasProbeBatch.num_envs"
        )
    if not np.all(ids == np.floor(ids)):
        raise ValueError("live atlas motion_ids must contain integers")
    if not np.all(starts == np.floor(starts)):
        raise ValueError("live atlas motion_start_steps must contain integers")
    if not np.all(lengths == np.floor(lengths)):
        raise ValueError("live atlas motion_num_steps must contain integers")
    live_ids = tuple(int(value) for value in ids)
    live_starts = tuple(int(value) for value in starts)
    live_lengths = tuple(int(value) for value in lengths)
    if live_ids != batch.motion_ids:
        raise ValueError("live atlas motion_ids drifted from the frozen AtlasProbeBatch")
    if live_starts != batch.start_steps:
        raise ValueError("live atlas start steps drifted from the frozen AtlasProbeBatch")
    if live_lengths != batch.reference_num_steps:
        raise ValueError("live atlas reference lengths drifted from the frozen AtlasProbeBatch")
    loaded_keys = tuple(motion_keys)
    if any(not isinstance(key, str) or not key for key in loaded_keys):
        raise ValueError("motion_keys must contain non-empty strings")
    try:
        resolved_keys = tuple(loaded_keys[motion_id] for motion_id in live_ids)
    except IndexError as error:
        raise ValueError("live atlas motion_ids reference unavailable motion keys") from error
    if resolved_keys != batch.motion_keys:
        raise ValueError("live atlas motion keys drifted from the frozen AtlasProbeBatch")

    metadata: list[RolloutMetadata] = []
    for env_index in range(batch.num_envs):
        denominator = batch.reference_num_steps[env_index] - 1
        realized_fraction = batch.start_steps[env_index] / denominator
        metadata.append(
            RolloutMetadata(
                rollout_id=batch.rollout_ids[env_index],
                motion_key=batch.motion_keys[env_index],
                policy_id=batch.probe_policy_id,
                domain_randomization_seed=batch.domain_randomization_seed,
                partition="D_atlas",
                reference_start_step=batch.start_steps[env_index],
                reference_num_steps=batch.reference_num_steps[env_index],
                initial_phase=realized_fraction,
                repeat_index=batch.repeat_index,
                env_index=env_index,
                motion_id=live_ids[env_index],
                checkpoint_sha256=batch.checkpoint_sha256,
                runtime_rng_seed=batch.runtime_rng_seed,
                runtime_rng_seed_readback=applied_seed,
                phase_id=batch.phase_id,
                target_fraction=batch.target_fraction,
                realized_fraction=realized_fraction,
                split_sha256=batch.split_sha256,
                split_selection_sha256=batch.split_selection_sha256,
                schedule_sha256=batch.schedule_sha256,
                schedule_entry_id=batch.rollout_ids[env_index],
            )
        )
    return metadata


def _to_numpy(value: Any, name: str) -> np.ndarray:
    """Convert a NumPy array or a torch-like tensor without importing torch."""

    if value is None:
        raise RuntimeError(f"required live tensor {name!r} is None")
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        result = np.asarray(converted)
    except Exception as error:  # noqa: BLE001 - third-party tensor conversion boundary
        raise RuntimeError(f"could not copy live tensor {name!r} to NumPy") from error
    if result.dtype == object:
        raise RuntimeError(f"live tensor {name!r} produced an object array")
    return result.copy()


def _callable_identifier(value: Any, name: str) -> str:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    if isinstance(value, functools.partial):
        return _callable_identifier(value.func, f"{name}.func")
    if isinstance(value, type):
        owner = value
    elif hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        owner = value
    else:
        owner = type(value)
    module = getattr(owner, "__module__", None)
    qualname = getattr(owner, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise TypeError(f"{name} has no stable module-qualified identity")
    return f"{module}:{qualname}"


def _array_payload(value: Any, name: str, *, require_nonempty: bool = True) -> dict[str, Any]:
    """Copy one numeric tensor into a canonical, JSON-ready payload."""

    array = _to_numpy(value, name)
    if require_nonempty and array.size == 0:
        raise RuntimeError(f"live tensor {name!r} must be non-empty")
    if array.dtype.kind not in "biuf":
        raise RuntimeError(f"live tensor {name!r} must have a boolean, integer, or floating dtype")
    if array.dtype.kind in "iuf" and not np.all(np.isfinite(array)):
        raise RuntimeError(f"live tensor {name!r} contains non-finite values")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _canonicalize_runtime_value(
    value: Any,
    name: str,
    *,
    _seen: set[int] | None = None,
) -> Any:
    """Strictly canonicalize resolved config without importing Isaac or torch."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return _canonicalize_runtime_value(value.value, f"{name}.value", _seen=_seen)
    if isinstance(value, np.generic):
        return _canonicalize_runtime_value(value.item(), name, _seen=_seen)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": _canonicalize_runtime_value(value.start, f"{name}.start", _seen=_seen),
            "stop": _canonicalize_runtime_value(value.stop, f"{name}.stop", _seen=_seen),
            "step": _canonicalize_runtime_value(value.step, f"{name}.step", _seen=_seen),
        }
    if isinstance(value, functools.partial):
        return {
            "type": "partial",
            "callable": _callable_identifier(value.func, f"{name}.func"),
            "args": _canonicalize_runtime_value(list(value.args), f"{name}.args", _seen=_seen),
            "keywords": _canonicalize_runtime_value(
                value.keywords or {},
                f"{name}.keywords",
                _seen=_seen,
            ),
        }
    if callable(value):
        return {"type": "callable", "identifier": _callable_identifier(value, name)}

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        raise ValueError(f"{name} contains a reference cycle")
    seen.add(identity)
    try:
        if isinstance(value, np.ndarray) or all(
            callable(getattr(value, method, None)) for method in ("detach", "cpu", "numpy")
        ):
            return {"type": "tensor", **_array_payload(value, name, require_nonempty=False)}
        if isinstance(value, Mapping):
            canonical: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(f"{name} mapping keys must be non-empty strings")
                canonical[key] = _canonicalize_runtime_value(
                    item,
                    f"{name}.{key}",
                    _seen=seen,
                )
            return canonical
        if isinstance(value, (list, tuple)):
            return [
                _canonicalize_runtime_value(item, f"{name}[{index}]", _seen=seen)
                for index, item in enumerate(value)
            ]
        if is_dataclass(value):
            field_names = tuple(getattr(value, "__dataclass_fields__", {}))
            return {
                field_name: _canonicalize_runtime_value(
                    getattr(value, field_name),
                    f"{name}.{field_name}",
                    _seen=seen,
                )
                for field_name in field_names
            }
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                converted = to_dict()
            except Exception as error:  # noqa: BLE001 - third-party config boundary
                raise RuntimeError(f"could not serialize resolved config {name}") from error
            return _canonicalize_runtime_value(converted, f"{name}.to_dict", _seen=seen)
    finally:
        seen.remove(identity)
    raise TypeError(
        f"{name} has unsupported runtime-config type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def capture_resolved_event_configuration(event_manager: Any) -> dict[str, Any]:
    """Capture active event order/config and reject untraced interval events."""

    active_terms = _required_attr(event_manager, "active_terms", "event_manager")
    if not isinstance(active_terms, Mapping):
        raise RuntimeError("event_manager.active_terms must be an ordered mode mapping")
    getter = _required_attr(event_manager, "get_term_cfg", "event_manager")
    if not callable(getter):
        raise RuntimeError("event_manager.get_term_cfg must be callable")

    seen_names: set[str] = set()
    modes: list[dict[str, Any]] = []
    for mode, raw_names in active_terms.items():
        if not isinstance(mode, str) or not mode:
            raise RuntimeError("event mode names must be non-empty strings")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise RuntimeError(f"event mode {mode!r} must expose an ordered term sequence")
        term_names = tuple(raw_names)
        if any(not isinstance(term_name, str) or not term_name for term_name in term_names):
            raise RuntimeError(f"event mode {mode!r} contains an invalid term name")
        duplicate = seen_names.intersection(term_names)
        if duplicate:
            raise RuntimeError(
                "event term names must be unique across modes for unambiguous readback: "
                f"{sorted(duplicate)}"
            )
        seen_names.update(term_names)
        if mode == "interval" and term_names:
            raise RuntimeError(
                "LACE atlas v1 rejects active interval events because realized event times and "
                f"impulses are not instrumented: {list(term_names)}"
            )
        terms: list[dict[str, Any]] = []
        for term_name in term_names:
            try:
                cfg = getter(term_name)
            except Exception as error:  # noqa: BLE001 - live manager boundary
                raise RuntimeError(
                    f"could not read resolved event config for {term_name!r}"
                ) from error
            cfg_mode = _required_attr(cfg, "mode", f"event_cfg[{term_name!r}]")
            if cfg_mode != mode:
                raise RuntimeError(
                    f"event term {term_name!r} is indexed under mode {mode!r} but its config "
                    f"declares {cfg_mode!r}"
                )
            func = _required_attr(cfg, "func", f"event_cfg[{term_name!r}]")
            terms.append(
                {
                    "term_name": term_name,
                    "callable": _callable_identifier(func, f"event_cfg[{term_name!r}].func"),
                    "resolved_config": _canonicalize_runtime_value(
                        cfg,
                        f"event_cfg[{term_name!r}]",
                    ),
                }
            )
        modes.append({"mode": mode, "terms": terms})
    record = {
        "mode_order": [entry["mode"] for entry in modes],
        "modes": modes,
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
        "interval_events_instrumented": False,
    }
    # Reject NaN, unsupported objects, and accidental non-JSON config values.
    json.dumps(record, allow_nan=False, sort_keys=True)
    return record


def _validated_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique and ordered")
    return result


def _validated_tensor_group(
    raw: Mapping[str, Any],
    expected_fields: Sequence[str],
    *,
    num_envs: int,
    name: str,
) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != set(expected_fields):
        raise ValueError(f"{name} must exactly contain {list(expected_fields)}")
    arrays: dict[str, np.ndarray] = {}
    for field_name in expected_fields:
        array = _to_numpy(raw[field_name], f"{name}.{field_name}")
        if array.ndim < 1 or array.shape[0] != num_envs or array.size == 0:
            raise ValueError(
                f"{name}.{field_name} must be non-empty with leading dimension {num_envs}"
            )
        if array.dtype.kind not in "biuf":
            raise ValueError(f"{name}.{field_name} must be boolean or real numeric")
        if array.dtype.kind in "iuf" and not np.all(np.isfinite(array)):
            raise ValueError(f"{name}.{field_name} must contain only finite values")
        arrays[field_name] = array
    return arrays


def _per_env_tensor_record(array: np.ndarray, env_index: int, name: str) -> dict[str, Any]:
    return _array_payload(array[env_index], f"{name}[{env_index}]")


def build_runtime_realizations(
    *,
    metadata: Sequence[RolloutMetadata],
    event_configuration: Mapping[str, Any],
    body_names: Sequence[str],
    joint_names: Sequence[str],
    action_joint_names: Sequence[str],
    reference_body_names: Sequence[str],
    robot_contract_state: Mapping[str, Any],
    robot_randomization: Mapping[str, Any],
    post_reset_state: Mapping[str, Any],
    scheduled_reference_state: Mapping[str, Any],
    env_indices: Sequence[int] | None = None,
) -> list[RuntimeRealization]:
    """Build exact per-environment post-reset provenance using CPU arrays only."""

    num_envs = len(metadata)
    if num_envs <= 0:
        raise ValueError("metadata must be non-empty")
    if any(item.env_index != index for index, item in enumerate(metadata)):
        raise ValueError("metadata must be ordered by contiguous env_index")
    bodies = _validated_names(body_names, "body_names")
    joints = _validated_names(joint_names, "joint_names")
    action_joints = _validated_names(action_joint_names, "action_joint_names")
    reference_bodies = _validated_names(reference_body_names, "reference_body_names")
    canonical_events = _canonicalize_runtime_value(event_configuration, "event_configuration")
    if not isinstance(canonical_events, Mapping):
        raise TypeError("event_configuration must canonicalize to a mapping")
    if canonical_events.get("interval_events_instrumented") is not False:
        raise ValueError("atlas v1 requires interval_events_instrumented=false")
    if canonical_events.get("interval_event_policy") != ATLAS_V1_INTERVAL_EVENT_POLICY:
        raise ValueError("event_configuration uses an unknown interval-event policy")

    randomization = _validated_tensor_group(
        robot_randomization,
        _ROBOT_RANDOMIZATION_FIELDS,
        num_envs=num_envs,
        name="robot_randomization",
    )
    state = _validated_tensor_group(
        post_reset_state,
        _POST_RESET_STATE_FIELDS,
        num_envs=num_envs,
        name="post_reset_state",
    )
    reference = _validated_tensor_group(
        scheduled_reference_state,
        _SCHEDULED_REFERENCE_STATE_FIELDS,
        num_envs=num_envs,
        name="scheduled_reference_state",
    )
    robot_contract_arrays = _validated_tensor_group(
        robot_contract_state,
        _ROBOT_CONTRACT_FIELDS,
        num_envs=num_envs,
        name="robot_contract_state",
    )

    if randomization["masses"].shape != (num_envs, len(bodies)):
        raise ValueError("robot_randomization.masses must align with body_names")
    for field_name in ("inertias", "centers_of_mass"):
        array = randomization[field_name]
        if array.ndim < 3 or array.shape[1] != len(bodies):
            raise ValueError(f"robot_randomization.{field_name} must align with body_names")
    if randomization["default_joint_position"].shape != (num_envs, len(joints)):
        raise ValueError("robot_randomization.default_joint_position must align with joint_names")
    if randomization["joint_action_offset"].shape != (num_envs, len(action_joints)):
        raise ValueError(
            "robot_randomization.joint_action_offset must align with action_joint_names"
        )
    expected_state_shapes = {
        "root_position_w": (num_envs, 3),
        "root_quaternion_wxyz": (num_envs, 4),
        "root_linear_velocity_w": (num_envs, 3),
        "root_angular_velocity_w": (num_envs, 3),
        "joint_position": (num_envs, len(joints)),
        "joint_velocity": (num_envs, len(joints)),
    }
    for field_name, expected_shape in expected_state_shapes.items():
        if state[field_name].shape != expected_shape:
            raise ValueError(f"post_reset_state.{field_name} must have shape {expected_shape}")
    expected_reference_shapes = {
        "command_time_step": (num_envs,),
        "motion_id": (num_envs,),
        "anchor_position_w": (num_envs, 3),
        "anchor_quaternion_wxyz": (num_envs, 4),
        "body_position_w": (num_envs, len(reference_bodies), 3),
        "body_quaternion_wxyz": (num_envs, len(reference_bodies), 4),
        "body_linear_velocity_w": (num_envs, len(reference_bodies), 3),
        "body_angular_velocity_w": (num_envs, len(reference_bodies), 3),
        "joint_position": (num_envs, len(joints)),
        "joint_velocity": (num_envs, len(joints)),
    }
    for field_name, expected_shape in expected_reference_shapes.items():
        if reference[field_name].shape != expected_shape:
            raise ValueError(
                f"scheduled_reference_state.{field_name} must have shape {expected_shape}"
            )
    for field_name in ("left_foot_contact", "right_foot_contact"):
        if reference[field_name].shape != (num_envs,):
            raise ValueError(
                f"scheduled_reference_state.{field_name} must have shape ({num_envs},)"
            )
    for field_name in ("command_time_step", "motion_id"):
        values = reference[field_name]
        if not np.all(values == np.floor(values)):
            raise ValueError(f"scheduled_reference_state.{field_name} must contain integers")
    expected_contract_shapes = {
        "joint_pos_limits": (num_envs, len(joints), 2),
        "soft_joint_pos_limits": (num_envs, len(joints), 2),
        "joint_vel_limits": (num_envs, len(joints)),
        "soft_joint_vel_limits": (num_envs, len(joints)),
    }
    for field_name, expected_shape in expected_contract_shapes.items():
        array = robot_contract_arrays[field_name]
        if array.shape != expected_shape:
            raise ValueError(f"robot_contract_state.{field_name} must have shape {expected_shape}")
        if not np.all(array == array[0]):
            raise ValueError(
                f"robot_contract_state.{field_name} varies by environment and cannot be "
                "represented as a global contract"
            )

    if env_indices is None:
        selected = tuple(range(num_envs))
    else:
        selected = tuple(env_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
            raise TypeError("env_indices must contain integers")
        if len(selected) != len(set(selected)):
            raise ValueError("env_indices must be unique")
        if any(index < 0 or index >= num_envs for index in selected):
            raise ValueError("env_indices contain an out-of-range environment")

    event_sha256 = canonical_sha256(canonical_events)
    robot_contract: dict[str, Any] = {
        "kind": ROBOT_CONTRACT_READBACK_KIND,
        "schema_version": ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "ordered_body_names": list(bodies),
        "ordered_joint_names": list(joints),
        "source_properties": {
            "joint_names": "robot.joint_names",
            "joint_pos_limits": "robot.data.joint_pos_limits[0]",
            "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
            "joint_vel_limits": "robot.data.joint_vel_limits[0]",
            "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
        },
        "limits": {
            field_name: _per_env_tensor_record(
                robot_contract_arrays[field_name],
                0,
                f"robot_contract_state.{field_name}",
            )
            for field_name in _ROBOT_CONTRACT_FIELDS
        },
        "environment_invariance_verified": True,
    }
    frozen_robot_contract = json.loads(json.dumps(robot_contract, allow_nan=False, sort_keys=True))
    robot_contract_sha256 = canonical_sha256(frozen_robot_contract)
    realizations: list[RuntimeRealization] = []
    for env_index in selected:
        item = metadata[env_index]
        scheduled_motion_id = int(reference["motion_id"][env_index])
        if item.motion_id is None or scheduled_motion_id != item.motion_id:
            raise ValueError(
                "scheduled_reference_state.motion_id drifted from live rollout metadata"
            )
        if item.runtime_rng_seed is None or item.runtime_rng_seed_readback is None:
            raise ValueError("runtime realization requires exact scheduled runtime RNG metadata")
        if int(reference["command_time_step"][env_index]) != 0:
            raise ValueError(
                "record_post_reset must observe command_time_step=0 before episode collection"
            )
        record: dict[str, Any] = {
            "kind": RUNTIME_REALIZATION_KIND,
            "schema_version": RUNTIME_REALIZATION_SCHEMA_VERSION,
            "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
            "runtime_rng_seed_semantics": SCIENTIFIC_SEED_SEMANTICS,
            "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
            "resolved_event_configuration_sha256": event_sha256,
            "resolved_event_configuration": canonical_events,
            "ordered_names": {
                "body_names": list(bodies),
                "joint_names": list(joints),
                "action_joint_names": list(action_joints),
                "reference_body_names": list(reference_bodies),
            },
            "realized_parameters": {
                field_name: _per_env_tensor_record(
                    randomization[field_name],
                    env_index,
                    f"robot_randomization.{field_name}",
                )
                for field_name in _ROBOT_RANDOMIZATION_FIELDS
            },
            "post_reset_state": {
                field_name: _per_env_tensor_record(
                    state[field_name],
                    env_index,
                    f"post_reset_state.{field_name}",
                )
                for field_name in _POST_RESET_STATE_FIELDS
            },
            "scheduled_reference": {
                "identity": {
                    "schedule_sha256": item.schedule_sha256,
                    "motion_key": item.motion_key,
                    "motion_id": scheduled_motion_id,
                    "domain_randomization_seed": item.domain_randomization_seed,
                    "runtime_rng_seed": item.runtime_rng_seed,
                    "runtime_rng_seed_readback": item.runtime_rng_seed_readback,
                    "phase_id": item.phase_id,
                    "target_fraction": item.target_fraction,
                    "realized_fraction": item.realized_fraction,
                    "reference_start_step": item.reference_start_step,
                    "reference_num_steps": item.reference_num_steps,
                    "repeat_index": item.repeat_index,
                },
                "state": {
                    field_name: _per_env_tensor_record(
                        reference[field_name],
                        env_index,
                        f"scheduled_reference_state.{field_name}",
                    )
                    for field_name in _SCHEDULED_REFERENCE_STATE_FIELDS
                },
            },
        }
        # A round-trip proves that the retained payload, not an incidental
        # tensor object, is exactly what was hashed.
        frozen_record = json.loads(json.dumps(record, allow_nan=False, sort_keys=True))
        realization = RuntimeRealization(
            env_index=env_index,
            rollout_id=item.rollout_id,
            record=frozen_record,
            sha256=canonical_sha256(frozen_record),
            robot_contract_readback=frozen_robot_contract,
            robot_contract_readback_sha256=robot_contract_sha256,
        )
        realization.validate()
        realizations.append(realization)
    return realizations


def bind_runtime_realizations(
    metadata: Sequence[RolloutMetadata],
    realizations: Sequence[RuntimeRealization | None],
    *,
    env_indices: Sequence[int] | None = None,
) -> list[RolloutMetadata]:
    """Attach previously captured post-reset provenance to live metadata."""

    if len(metadata) != len(realizations):
        raise ValueError("runtime realization count must match metadata")
    for env_index, item in enumerate(metadata):
        if item.env_index != env_index:
            raise ValueError("metadata must be ordered by contiguous env_index")
    if env_indices is None:
        selected = tuple(range(len(metadata)))
    else:
        selected = tuple(env_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
            raise TypeError("runtime realization env_indices must contain integers")
        if len(selected) != len(set(selected)):
            raise ValueError("runtime realization env_indices must be unique")
        if any(index < 0 or index >= len(metadata) for index in selected):
            raise ValueError("runtime realization env_indices contain an out-of-range environment")
    bound = list(metadata)
    for env_index in selected:
        item = metadata[env_index]
        realization = realizations[env_index]
        if realization is None:
            raise RuntimeError(
                f"environment {env_index} has no record_post_reset runtime realization"
            )
        realization.validate()
        if realization.env_index != env_index or realization.rollout_id != item.rollout_id:
            raise RuntimeError(
                f"environment {env_index} runtime realization does not match its rollout row"
            )
        identity = realization.record.get("scheduled_reference", {}).get("identity", {})
        if (
            identity.get("motion_key") != item.motion_key
            or identity.get("schedule_sha256") != item.schedule_sha256
            or identity.get("reference_start_step") != item.reference_start_step
        ):
            raise RuntimeError(
                f"environment {env_index} runtime realization schedule coordinates drifted"
            )
        bound[env_index] = replace(
            item,
            domain_randomization_realization=realization.record,
            domain_randomization_realization_sha256=realization.sha256,
            runtime_rng_seed_semantics=SCIENTIFIC_SEED_SEMANTICS,
            robot_contract_readback=realization.robot_contract_readback,
            robot_contract_readback_sha256=realization.robot_contract_readback_sha256,
        )
    return bound


def _callable_source_sha256(value: Any, name: str) -> str:
    try:
        source_lines, _ = inspect.getsourcelines(value)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"could not inspect exact source for {name}") from error
    canonical = textwrap.dedent("".join(source_lines)).replace("\r\n", "\n").strip() + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def termination_compute_source_sha256(manager: Any) -> str:
    """Hash the class-level compute implementation that would be replaced."""

    instance_dict = getattr(manager, "__dict__", {})
    if "compute" in instance_dict:
        raise RuntimeError("termination manager already has an instance-level compute override")
    manager_type = type(manager)
    compute = getattr(manager_type, "compute", None)
    if not callable(compute):
        raise RuntimeError("termination manager class has no callable compute method")
    return _callable_source_sha256(
        compute, f"{manager_type.__module__}.{manager_type.__qualname__}.compute"
    )


def _termination_manager_private_contract(
    manager: Any,
) -> tuple[tuple[str, ...], tuple[Any, ...], tuple[bool, ...], int]:
    try:
        names = tuple(manager._term_names)
        configs = tuple(manager._term_cfgs)
        name_to_index = dict(manager._term_name_to_term_idx)
        term_dones = manager._term_dones
        truncated = manager._truncated_buf
        terminated = manager._terminated_buf
        num_envs = manager.num_envs
        managed_env = manager._env
    except Exception as error:  # noqa: BLE001 - version-locked Isaac private API
        raise RuntimeError(
            "termination manager private API does not match the pinned contract"
        ) from error
    if (
        not isinstance(num_envs, int)
        or isinstance(num_envs, bool)
        or num_envs <= 0
        or managed_env is None
    ):
        raise RuntimeError("termination manager num_envs/_env contract is invalid")
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("termination manager term names must be non-empty strings")
    if len(names) != len(set(names)) or len(configs) != len(names):
        raise RuntimeError("termination manager term name/config ordering is invalid")
    if name_to_index != {name: index for index, name in enumerate(names)}:
        raise RuntimeError("termination manager term index mapping drifted")
    try:
        active_terms = tuple(manager.active_terms)
    except Exception as error:  # noqa: BLE001 - live manager boundary
        raise RuntimeError("termination manager active_terms is unavailable") from error
    if active_terms != names:
        raise RuntimeError("termination manager active_terms drifted from private term order")
    expected_shapes = (
        (term_dones, (num_envs, len(names)), "_term_dones"),
        (truncated, (num_envs,), "_truncated_buf"),
        (terminated, (num_envs,), "_terminated_buf"),
    )
    for tensor, expected_shape, field_name in expected_shapes:
        if tuple(getattr(tensor, "shape", ())) != expected_shape:
            raise RuntimeError(
                f"termination manager {field_name} shape drifted from {expected_shape}"
            )
        if "bool" not in str(getattr(tensor, "dtype", "")):
            raise RuntimeError(f"termination manager {field_name} must have boolean dtype")
    timeout_flags: list[bool] = []
    for index, cfg in enumerate(configs):
        func = getattr(cfg, "func", None)
        params = getattr(cfg, "params", None)
        time_out = getattr(cfg, "time_out", None)
        if not callable(func):
            raise RuntimeError(f"termination config {names[index]!r} func is not callable")
        if not isinstance(params, Mapping):
            raise RuntimeError(f"termination config {names[index]!r} params must be a mapping")
        if not isinstance(time_out, bool):
            raise RuntimeError(f"termination config {names[index]!r} time_out must be boolean")
        timeout_flags.append(time_out)
    return names, configs, tuple(timeout_flags), num_envs


def _termination_term_config_record(
    names: Sequence[str],
    configs: Sequence[Any],
) -> list[dict[str, Any]]:
    return [
        {
            "term_name": name,
            "callable": _callable_identifier(cfg.func, f"termination_cfg[{name!r}].func"),
            "time_out": cfg.time_out,
            "params": _canonicalize_runtime_value(
                cfg.params,
                f"termination_cfg[{name!r}].params",
            ),
        }
        for name, cfg in zip(names, configs, strict=True)
    ]


def _termination_compute_with_lace_trace(manager: Any) -> Any:
    """Pinned Isaac ``compute`` semantics plus one independent raw-value matrix."""

    if getattr(manager, "_lace_trace_installed", False) is not True:
        raise RuntimeError("LACE termination trace wrapper marker is missing")
    if manager._lace_trace_generation != manager._lace_trace_consumed_generation:
        raise RuntimeError(
            "previous LACE termination trace was not consumed before the next compute"
        )
    names, configs, timeout_flags, num_envs = _termination_manager_private_contract(manager)
    contract = manager._lace_trace_contract
    if canonical_sha256(contract) != manager._lace_trace_contract_sha256:
        raise RuntimeError("LACE termination trace contract was mutated after installation")
    current_term_record = _termination_term_config_record(names, configs)
    if (
        names != tuple(contract["term_names"])
        or timeout_flags != tuple(contract["time_out_flags"])
        or canonical_sha256({"terms": current_term_record}) != contract["term_config_sha256"]
    ):
        raise RuntimeError("termination terms/config changed after LACE trace installation")

    raw_matrix = manager._term_dones.clone()
    raw_matrix[:] = False
    manager._truncated_buf[:] = False
    manager._terminated_buf[:] = False
    for index, term_cfg in enumerate(configs):
        # Exactly one invocation, matching the pinned Isaac implementation.
        value = term_cfg.func(manager._env, **term_cfg.params)
        if tuple(getattr(value, "shape", ())) != (num_envs,):
            raise RuntimeError(
                f"termination term {names[index]!r} returned shape "
                f"{tuple(getattr(value, 'shape', ()))}, expected {(num_envs,)}"
            )
        if "bool" not in str(getattr(value, "dtype", "")):
            raise RuntimeError(f"termination term {names[index]!r} must return boolean values")
        raw_matrix[:, index] = value
        if term_cfg.time_out:
            manager._truncated_buf |= value
        else:
            manager._terminated_buf |= value
        rows = value.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            manager._term_dones[rows] = False
            manager._term_dones[rows, index] = True

    step_counter = getattr(manager._env, "common_step_counter", None)
    if not isinstance(step_counter, int) or isinstance(step_counter, bool) or step_counter < 0:
        raise RuntimeError("env.common_step_counter must be a nonnegative integer")
    generation = manager._lace_trace_generation + 1
    manager._lace_trace = {
        "generation": generation,
        "common_step_counter": step_counter,
        "values": raw_matrix,
    }
    manager._lace_trace_generation = generation
    return manager._truncated_buf | manager._terminated_buf


def install_termination_trace(
    manager: Any,
    *,
    expected_compute_source_sha256: str = EXPECTED_ISAACLAB_TERMINATION_COMPUTE_SHA256,
) -> tuple[dict[str, Any], str]:
    """Install the version-locked one-pass trace on one manager instance."""

    if hasattr(manager, "_lace_trace_installed"):
        raise RuntimeError("LACE termination trace is already installed on this manager")
    if (
        not isinstance(expected_compute_source_sha256, str)
        or len(expected_compute_source_sha256) != 64
    ):
        raise ValueError("expected termination compute source digest must be a SHA-256")
    try:
        int(expected_compute_source_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "expected termination compute source digest must be hexadecimal"
        ) from error
    actual_source_sha256 = termination_compute_source_sha256(manager)
    if actual_source_sha256 != expected_compute_source_sha256:
        raise RuntimeError(
            "termination manager compute source drifted: expected "
            f"{expected_compute_source_sha256}, observed {actual_source_sha256}"
        )
    names, configs, timeout_flags, _ = _termination_manager_private_contract(manager)
    term_record = _termination_term_config_record(names, configs)
    manager_type = type(manager)
    contract = {
        "kind": TERMINATION_TRACE_KIND,
        "schema_version": TERMINATION_TRACE_SCHEMA_VERSION,
        "algorithm": TERMINATION_TRACE_ALGORITHM,
        "manager_type": f"{manager_type.__module__}:{manager_type.__qualname__}",
        "manager_compute_source_sha256": actual_source_sha256,
        "instrument_compute_source_sha256": _callable_source_sha256(
            _termination_compute_with_lace_trace,
            "_termination_compute_with_lace_trace",
        ),
        "term_names": list(names),
        "time_out_flags": list(timeout_flags),
        "term_config_sha256": canonical_sha256({"terms": term_record}),
        "term_configs": term_record,
        "legacy_term_dones_semantics": "last_trigger_wins_stale_rows_preserved",
        "raw_trace_semantics": "ordered_independent_values_from_single_manager_evaluation",
        "freshness_semantics": "one_compute_one_consume_common_step_counter_bound",
    }
    frozen_contract = json.loads(json.dumps(contract, allow_nan=False, sort_keys=True))
    contract_sha256 = canonical_sha256(frozen_contract)
    manager._lace_trace_contract = frozen_contract
    manager._lace_trace_contract_sha256 = contract_sha256
    manager._lace_trace_generation = 0
    manager._lace_trace_consumed_generation = 0
    manager._lace_trace = None
    manager._lace_trace_installed = True
    try:
        manager.compute = MethodType(_termination_compute_with_lace_trace, manager)
    except Exception:
        for field_name in (
            "_lace_trace_contract",
            "_lace_trace_contract_sha256",
            "_lace_trace_generation",
            "_lace_trace_consumed_generation",
            "_lace_trace",
            "_lace_trace_installed",
        ):
            delattr(manager, field_name)
        raise
    return frozen_contract, contract_sha256


def consume_termination_trace(
    manager: Any,
    *,
    expected_common_step_counter: int,
) -> TerminationTraceSnapshot:
    """Consume exactly one fresh trace produced by the current manager compute."""

    if getattr(manager, "_lace_trace_installed", False) is not True:
        raise RuntimeError("LACE termination trace is not installed")
    bound_compute = getattr(manager, "compute", None)
    if getattr(bound_compute, "__func__", None) is not _termination_compute_with_lace_trace:
        raise RuntimeError("LACE termination manager compute wrapper was replaced")
    generation = getattr(manager, "_lace_trace_generation", None)
    consumed = getattr(manager, "_lace_trace_consumed_generation", None)
    trace = getattr(manager, "_lace_trace", None)
    if not isinstance(generation, int) or generation <= 0 or generation == consumed:
        raise RuntimeError("LACE termination trace is stale or has already been consumed")
    if generation != consumed + 1 or not isinstance(trace, Mapping):
        raise RuntimeError("LACE termination trace generation sequence drifted")
    if (
        not isinstance(expected_common_step_counter, int)
        or isinstance(expected_common_step_counter, bool)
        or trace.get("common_step_counter") != expected_common_step_counter
    ):
        raise RuntimeError("LACE termination trace is stale for env.common_step_counter")
    names, configs, timeout_flags, num_envs = _termination_manager_private_contract(manager)
    contract = manager._lace_trace_contract
    if canonical_sha256(contract) != manager._lace_trace_contract_sha256:
        raise RuntimeError("LACE termination trace contract was mutated")
    if canonical_sha256({"terms": _termination_term_config_record(names, configs)}) != contract.get(
        "term_config_sha256"
    ):
        raise RuntimeError("termination config drifted before trace consumption")
    values = _binary_array(_to_numpy(trace.get("values"), "termination raw trace"), "trace", 2)
    if values.shape != (num_envs, len(names)):
        raise RuntimeError("LACE termination raw trace shape drifted")
    snapshot = TerminationTraceSnapshot(
        generation=generation,
        common_step_counter=expected_common_step_counter,
        term_names=names,
        time_out_flags=timeout_flags,
        values=values,
        contract=contract,
        contract_sha256=manager._lace_trace_contract_sha256,
    )
    snapshot.validate()
    manager._lace_trace_consumed_generation = generation
    return snapshot


def _validate_installed_termination_trace(
    manager: Any,
    *,
    expected_contract: Mapping[str, Any],
    expected_contract_sha256: str,
) -> None:
    """Fail closed if the installed per-instance tap was replaced or mutated."""

    if getattr(manager, "_lace_trace_installed", False) is not True:
        raise RuntimeError("LACE termination trace is not installed")
    bound_compute = getattr(manager, "compute", None)
    if getattr(bound_compute, "__func__", None) is not _termination_compute_with_lace_trace:
        raise RuntimeError("LACE termination manager compute wrapper was replaced")
    actual_contract = getattr(manager, "_lace_trace_contract", None)
    actual_sha256 = getattr(manager, "_lace_trace_contract_sha256", None)
    if (
        not isinstance(actual_contract, Mapping)
        or actual_contract != expected_contract
        or actual_sha256 != expected_contract_sha256
        or canonical_sha256(actual_contract) != expected_contract_sha256
    ):
        raise RuntimeError("LACE termination trace installation contract drifted")
    generation = getattr(manager, "_lace_trace_generation", None)
    consumed = getattr(manager, "_lace_trace_consumed_generation", None)
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or not isinstance(consumed, int)
        or isinstance(consumed, bool)
        or generation < 0
        or consumed < 0
        or generation != consumed
    ):
        raise RuntimeError("LACE termination trace has an unconsumed or invalid generation")


def validate_termination_trace_unions(
    snapshot: TerminationTraceSnapshot,
    *,
    reset_terminated: ArrayLike,
    reset_time_outs: ArrayLike,
    reset_buf: ArrayLike,
) -> dict[str, BoolArray]:
    """Verify raw trace unions against the environment's exact done buffers."""

    snapshot.validate()
    values = _binary_array(snapshot.values, "termination trace values", 2)
    num_envs = values.shape[0]
    terminated = _binary_array(reset_terminated, "reset_terminated", 1)
    time_outs = _binary_array(reset_time_outs, "reset_time_outs", 1)
    dones = _binary_array(reset_buf, "reset_buf", 1)
    if any(array.shape != (num_envs,) for array in (terminated, time_outs, dones)):
        raise ValueError("termination trace and environment done-buffer shapes must match")
    timeout_columns = np.asarray(snapshot.time_out_flags, dtype=bool)
    timeout_union = (
        np.any(values[:, timeout_columns], axis=1)
        if np.any(timeout_columns)
        else np.zeros(num_envs, dtype=bool)
    )
    terminated_union = (
        np.any(values[:, ~timeout_columns], axis=1)
        if np.any(~timeout_columns)
        else np.zeros(num_envs, dtype=bool)
    )
    if not np.array_equal(terminated_union, terminated):
        raise RuntimeError("raw non-timeout termination union does not match env.reset_terminated")
    if not np.array_equal(timeout_union, time_outs):
        raise RuntimeError("raw timeout termination union does not match env.reset_time_outs")
    if not np.array_equal(terminated_union | timeout_union, dones):
        raise RuntimeError("raw termination union does not match env.reset_buf")
    return {name: values[:, index].copy() for index, name in enumerate(snapshot.term_names)}


def _sonic_reference_foot_contact(value: Any, name: str) -> np.ndarray:
    """Normalize SONIC's one-body-per-side contact label to ``[num_envs]``.

    The released G1 motion library retains a singleton configured-foot axis, so
    ``TrackingCommand.feet_l/feet_r`` are ``[N, 1]``. A future multi-body foot
    representation needs an explicit aggregation rule and therefore fails here.
    """

    contact = _to_numpy(value, name)
    if contact.ndim == 2 and contact.shape[1] == 1:
        contact = contact[:, 0]
    if contact.ndim != 1 or contact.size == 0:
        raise ValueError(f"{name} must have shape [num_envs] or [num_envs, 1]")
    return contact


def _required_attr(owner: Any, name: str, context: str) -> Any:
    try:
        value = getattr(owner, name)
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(f"LACE live binding missing {context}.{name}") from error
    if value is None:
        raise RuntimeError(f"LACE live binding {context}.{name} is None")
    return value


def _scene_entity(env: Any, name: str) -> Any:
    try:
        return env.scene[name]
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(f"LACE live binding could not resolve scene entity {name!r}") from error


def _exact_body_indices(entity: Any, body_names: tuple[str, str], context: str) -> list[int]:
    finder = _required_attr(entity, "find_bodies", context)
    try:
        indices, resolved_names = finder(list(body_names), preserve_order=True)
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(f"LACE could not resolve {context} foot bodies {body_names}") from error
    if tuple(resolved_names) != body_names or len(indices) != 2:
        raise RuntimeError(
            f"LACE {context} foot-body mismatch: requested={body_names}, "
            f"resolved={tuple(resolved_names)}"
        )
    return [int(index) for index in indices]


def extract_isaac_post_step(
    env: Any,
    bindings: IsaacBindings,
    *,
    termination_trace: TerminationTraceSnapshot | None = None,
) -> ProbeFrameBatch:
    """Copy the verified pre-reset tensors available in ``record_post_step``.

    The function is duck-typed so CPU fakes can exercise every mapping.  On a
    live environment it must run after termination computation and before Isaac
    Lab resets ``reset_buf`` environments, which is exactly the custom
    ``RecorderTerm.record_post_step`` lifecycle.
    """

    bindings.validate()
    try:
        command = env.command_manager.get_term(bindings.command_name)
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(
            f"LACE could not resolve command term {bindings.command_name!r}"
        ) from error
    robot = _scene_entity(env, bindings.robot_name)
    contact_sensor = _scene_entity(env, bindings.contact_sensor_name)
    robot_foot_indices = _exact_body_indices(robot, bindings.foot_body_names, "robot")
    sensor_foot_indices = _exact_body_indices(
        contact_sensor,
        bindings.foot_body_names,
        "contact sensor",
    )

    robot_data = _required_attr(robot, "data", "robot")
    sensor_data = _required_attr(contact_sensor, "data", "contact_sensor")
    termination_manager = _required_attr(env, "termination_manager", "env")
    active_terms = tuple(_required_attr(termination_manager, "active_terms", "termination_manager"))
    if not active_terms:
        raise RuntimeError("LACE requires at least one active termination term")
    missing_fall_terms = sorted(set(bindings.fall_termination_terms) - set(active_terms))
    if missing_fall_terms:
        raise RuntimeError(f"configured fall termination terms are inactive: {missing_fall_terms}")
    missing_timeout_terms = sorted(set(bindings.timeout_termination_terms) - set(active_terms))
    if missing_timeout_terms:
        raise RuntimeError(
            f"configured timeout termination terms are inactive: {missing_timeout_terms}"
        )
    failure_mask = _to_numpy(
        _required_attr(env, "reset_terminated", "env"),
        "env.reset_terminated",
    )
    if termination_trace is None:
        term_arrays = {
            name: _to_numpy(termination_manager.get_term(name), f"termination_manager.{name}")
            for name in active_terms
        }
        termination_semantics = "manager_last_trigger_wins_masked_to_current_end"
        termination_multi_hot_available = False
        trace_contract = None
        trace_contract_sha256 = None
        trace_generation = None
        trace_step_counter = None
    else:
        if termination_trace.term_names != active_terms:
            raise RuntimeError("termination trace term order drifted from active_terms")
        traced_timeout_terms = tuple(
            name
            for name, time_out in zip(
                termination_trace.term_names,
                termination_trace.time_out_flags,
                strict=True,
            )
            if time_out
        )
        if traced_timeout_terms != bindings.timeout_termination_terms:
            raise RuntimeError(
                "termination trace timeout terms do not match the explicit recorder binding"
            )
        term_arrays = validate_termination_trace_unions(
            termination_trace,
            reset_terminated=failure_mask,
            reset_time_outs=_to_numpy(
                _required_attr(env, "reset_time_outs", "env"),
                "env.reset_time_outs",
            ),
            reset_buf=_to_numpy(_required_attr(env, "reset_buf", "env"), "env.reset_buf"),
        )
        termination_semantics = "instrumented_single_evaluation_ordered_raw_boolean_matrix"
        termination_multi_hot_available = True
        trace_contract = termination_trace.contract
        trace_contract_sha256 = termination_trace.contract_sha256
        trace_generation = termination_trace.generation
        trace_step_counter = termination_trace.common_step_counter
    fall_mask = np.zeros_like(next(iter(term_arrays.values())), dtype=bool)
    for name in bindings.fall_termination_terms:
        fall_mask |= np.asarray(term_arrays[name], dtype=bool)
    fall_mask &= np.asarray(failure_mask, dtype=bool)

    contact_forces = _to_numpy(
        _required_attr(sensor_data, "net_forces_w", "contact_sensor.data"),
        "contact_sensor.data.net_forces_w",
    )
    body_velocity = _to_numpy(
        _required_attr(robot_data, "body_link_lin_vel_w", "robot.data"),
        "robot.data.body_link_lin_vel_w",
    )
    soft_limits = _to_numpy(
        _required_attr(robot_data, "soft_joint_pos_limits", "robot.data"),
        "robot.data.soft_joint_pos_limits",
    )

    return assemble_probe_frame_batch(
        reference_left_contact=_sonic_reference_foot_contact(
            _required_attr(command, "feet_l", "motion_command"),
            "motion_command.feet_l",
        ),
        reference_right_contact=_sonic_reference_foot_contact(
            _required_attr(command, "feet_r", "motion_command"),
            "motion_command.feet_r",
        ),
        foot_contact_force_w=contact_forces[:, sensor_foot_indices, :],
        foot_linear_velocity_w=body_velocity[:, robot_foot_indices, :],
        reference_anchor_position_w=_to_numpy(
            _required_attr(command, "anchor_pos_w", "motion_command"),
            "motion_command.anchor_pos_w",
        ),
        actual_anchor_position_w=_to_numpy(
            _required_attr(command, "robot_anchor_pos_w", "motion_command"),
            "motion_command.robot_anchor_pos_w",
        ),
        reference_anchor_quaternion_wxyz=_to_numpy(
            _required_attr(command, "anchor_quat_w", "motion_command"),
            "motion_command.anchor_quat_w",
        ),
        actual_anchor_quaternion_wxyz=_to_numpy(
            _required_attr(command, "robot_anchor_quat_w", "motion_command"),
            "motion_command.robot_anchor_quat_w",
        ),
        requested_torque=_to_numpy(
            _required_attr(robot_data, "computed_torque", "robot.data"),
            "robot.data.computed_torque",
        ),
        applied_torque=_to_numpy(
            _required_attr(robot_data, "applied_torque", "robot.data"),
            "robot.data.applied_torque",
        ),
        effort_limits=_to_numpy(
            _required_attr(robot_data, "joint_effort_limits", "robot.data"),
            "robot.data.joint_effort_limits",
        ),
        reference_joint_position=_to_numpy(
            _required_attr(command, "joint_pos", "motion_command"),
            "motion_command.joint_pos",
        ),
        joint_position=_to_numpy(
            _required_attr(robot_data, "joint_pos", "robot.data"),
            "robot.data.joint_pos",
        ),
        joint_soft_limits=soft_limits,
        aligned_reference_body_position_w=_to_numpy(
            _required_attr(command, "body_pos_relative_w", "motion_command"),
            "motion_command.body_pos_relative_w",
        ),
        actual_body_position_w=_to_numpy(
            _required_attr(command, "robot_body_pos_w", "motion_command"),
            "motion_command.robot_body_pos_w",
        ),
        episode_end_mask=_to_numpy(
            _required_attr(env, "reset_buf", "env"),
            "env.reset_buf",
        ),
        failure_mask=failure_mask,
        fall_mask=fall_mask,
        termination_terms=term_arrays,
        fall_termination_terms=bindings.fall_termination_terms,
        timeout_termination_terms=bindings.timeout_termination_terms,
        termination_semantics=termination_semantics,
        termination_multi_hot_available=termination_multi_hot_available,
        termination_trace_contract=trace_contract,
        termination_trace_contract_sha256=trace_contract_sha256,
        termination_trace_generation=trace_generation,
        termination_trace_step_counter=trace_step_counter,
        contact_force_threshold=bindings.contact_force_threshold,
        ground_normal_axis=bindings.ground_normal_axis,
    )


def extract_isaac_rollout_metadata(
    env: Any,
    run: RolloutRun | None = None,
    *,
    command_name: str = "motion",
) -> list[RolloutMetadata]:
    """Copy motion assignment metadata from the live SONIC command term."""

    if not isinstance(command_name, str) or not command_name:
        raise ValueError("command_name must be a non-empty string")
    try:
        command = env.command_manager.get_term(command_name)
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(f"LACE could not resolve command term {command_name!r}") from error
    motion_lib = _required_attr(command, "motion_lib", "motion_command")
    motion_keys = _required_attr(motion_lib, "curr_motion_keys", "motion_command.motion_lib")
    atlas_batch = getattr(command, "atlas_probe_batch", None)
    if atlas_batch is not None:
        env_cfg = _required_attr(env, "cfg", "env")
        if isinstance(env_cfg, Mapping):
            if "seed" not in env_cfg:
                raise RuntimeError("LACE live binding missing env.cfg.seed")
            process_seed_readback = env_cfg["seed"]
        else:
            process_seed_readback = _required_attr(env_cfg, "seed", "env.cfg")
        return assemble_atlas_probe_metadata(
            batch=atlas_batch,
            motion_ids=_to_numpy(
                _required_attr(command, "motion_ids", "motion_command"),
                "motion_command.motion_ids",
            ),
            motion_start_steps=_to_numpy(
                _required_attr(command, "motion_start_time_steps", "motion_command"),
                "motion_command.motion_start_time_steps",
            ),
            motion_num_steps=_to_numpy(
                _required_attr(command, "motion_num_steps", "motion_command"),
                "motion_command.motion_num_steps",
            ),
            motion_keys=tuple(motion_keys),
            process_seed_readback=process_seed_readback,
        )
    if bool(getattr(command, "_atlas_probe_enabled", False)):
        raise RuntimeError("atlas_probe_mode is enabled but its frozen AtlasProbeBatch is missing")
    if run is None:
        raise RuntimeError(
            "the live LACE recorder requires atlas_probe_mode and an exact AtlasProbeBatch"
        )
    return assemble_rollout_metadata(
        motion_ids=_to_numpy(
            _required_attr(command, "motion_ids", "motion_command"),
            "motion_command.motion_ids",
        ),
        motion_start_steps=_to_numpy(
            _required_attr(command, "motion_start_time_steps", "motion_command"),
            "motion_command.motion_start_time_steps",
        ),
        motion_num_steps=_to_numpy(
            _required_attr(command, "motion_num_steps", "motion_command"),
            "motion_command.motion_num_steps",
        ),
        motion_keys=tuple(motion_keys),
        run=run,
    )


def _normalized_env_indices(env_ids: Any, num_envs: int) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(num_envs))
    values = _to_numpy(env_ids, "record_post_reset.env_ids")
    if values.ndim != 1:
        raise ValueError("record_post_reset env_ids must be one-dimensional")
    if values.dtype.kind not in "iu" or not np.all(values == np.floor(values)):
        raise ValueError("record_post_reset env_ids must contain integers")
    result = tuple(int(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError("record_post_reset env_ids must be unique")
    if any(index < 0 or index >= num_envs for index in result):
        raise ValueError("record_post_reset env_ids contain an out-of-range environment")
    return result


def _physx_readback(view: Any, method_name: str) -> Any:
    method = _required_attr(view, method_name, "robot.root_physx_view")
    if not callable(method):
        raise RuntimeError(f"robot.root_physx_view.{method_name} must be callable")
    try:
        return method()
    except Exception as error:  # noqa: BLE001 - live simulator boundary
        raise RuntimeError(f"could not read robot.root_physx_view.{method_name}()") from error


def extract_isaac_post_reset_realizations(
    env: Any,
    bindings: IsaacBindings,
    metadata: Sequence[RolloutMetadata],
    env_ids: Any = None,
) -> list[RuntimeRealization]:
    """Read exact post-reset DR/state provenance through public live buffers.

    The sole version-locked exception is the joint action term's ``_offset``:
    the released SONIC randomizer mutates that exact private tensor and the
    installed Isaac Lab action term exposes no public per-environment readback.
    Missing or reshaped state fails closed.
    """

    bindings.validate()
    num_envs = len(metadata)
    if int(_required_attr(env, "num_envs", "env")) != num_envs:
        raise ValueError("live env.num_envs must match rollout metadata")
    selected = _normalized_env_indices(env_ids, num_envs)
    event_manager = _required_attr(env, "event_manager", "env")
    event_configuration = capture_resolved_event_configuration(event_manager)
    robot = _scene_entity(env, bindings.robot_name)
    robot_data = _required_attr(robot, "data", "robot")
    physx_view = _required_attr(robot, "root_physx_view", "robot")
    action_manager = _required_attr(env, "action_manager", "env")
    get_action_term = _required_attr(action_manager, "get_term", "action_manager")
    if not callable(get_action_term):
        raise RuntimeError("action_manager.get_term must be callable")
    try:
        action_term = get_action_term(bindings.joint_action_name)
    except Exception as error:  # noqa: BLE001 - live manager boundary
        raise RuntimeError(
            f"LACE could not resolve joint action term {bindings.joint_action_name!r}"
        ) from error
    try:
        command = env.command_manager.get_term(bindings.command_name)
    except Exception as error:  # noqa: BLE001 - live manager boundary
        raise RuntimeError(
            f"LACE could not resolve command term {bindings.command_name!r}"
        ) from error

    body_names = tuple(_required_attr(robot, "body_names", "robot"))
    joint_names = tuple(_required_attr(robot, "joint_names", "robot"))
    action_joint_names = tuple(_required_attr(action_term, "_joint_names", "joint_action_term"))
    reference_body_names = tuple(_required_attr(command, "cmd_body_names", "motion_command"))
    robot_contract_state = {
        "joint_pos_limits": _required_attr(robot_data, "joint_pos_limits", "robot.data"),
        "soft_joint_pos_limits": _required_attr(
            robot_data,
            "soft_joint_pos_limits",
            "robot.data",
        ),
        "joint_vel_limits": _required_attr(robot_data, "joint_vel_limits", "robot.data"),
        "soft_joint_vel_limits": _required_attr(
            robot_data,
            "soft_joint_vel_limits",
            "robot.data",
        ),
    }
    robot_randomization = {
        "material_properties": _physx_readback(physx_view, "get_material_properties"),
        "masses": _physx_readback(physx_view, "get_masses"),
        "inertias": _physx_readback(physx_view, "get_inertias"),
        "centers_of_mass": _physx_readback(physx_view, "get_coms"),
        "default_joint_position": _required_attr(
            robot_data,
            "default_joint_pos",
            "robot.data",
        ),
        "joint_action_offset": _required_attr(
            action_term,
            "_offset",
            "joint_action_term",
        ),
    }
    post_reset_state = {
        "root_position_w": _required_attr(robot_data, "root_pos_w", "robot.data"),
        "root_quaternion_wxyz": _required_attr(robot_data, "root_quat_w", "robot.data"),
        "root_linear_velocity_w": _required_attr(
            robot_data,
            "root_lin_vel_w",
            "robot.data",
        ),
        "root_angular_velocity_w": _required_attr(
            robot_data,
            "root_ang_vel_w",
            "robot.data",
        ),
        "joint_position": _required_attr(robot_data, "joint_pos", "robot.data"),
        "joint_velocity": _required_attr(robot_data, "joint_vel", "robot.data"),
    }
    scheduled_reference_state = {
        "command_time_step": _required_attr(command, "time_steps", "motion_command"),
        "motion_id": _required_attr(command, "motion_ids", "motion_command"),
        "anchor_position_w": _required_attr(command, "anchor_pos_w", "motion_command"),
        "anchor_quaternion_wxyz": _required_attr(
            command,
            "anchor_quat_w",
            "motion_command",
        ),
        "body_position_w": _required_attr(command, "body_pos_w", "motion_command"),
        "body_quaternion_wxyz": _required_attr(
            command,
            "body_quat_w",
            "motion_command",
        ),
        "body_linear_velocity_w": _required_attr(
            command,
            "body_lin_vel_w",
            "motion_command",
        ),
        "body_angular_velocity_w": _required_attr(
            command,
            "body_ang_vel_w",
            "motion_command",
        ),
        "joint_position": _required_attr(command, "joint_pos", "motion_command"),
        "joint_velocity": _required_attr(command, "joint_vel", "motion_command"),
        "left_foot_contact": _sonic_reference_foot_contact(
            _required_attr(command, "feet_l", "motion_command"),
            "motion_command.feet_l",
        ),
        "right_foot_contact": _sonic_reference_foot_contact(
            _required_attr(command, "feet_r", "motion_command"),
            "motion_command.feet_r",
        ),
    }
    return build_runtime_realizations(
        metadata=metadata,
        event_configuration=event_configuration,
        body_names=body_names,
        joint_names=joint_names,
        action_joint_names=action_joint_names,
        reference_body_names=reference_body_names,
        robot_contract_state=robot_contract_state,
        robot_randomization=robot_randomization,
        post_reset_state=post_reset_state,
        scheduled_reference_state=scheduled_reference_state,
        env_indices=selected,
    )


@dataclass
class _EpisodeBuffer:
    metadata: RolloutMetadata
    fields: dict[str, list[np.ndarray]] = field(default_factory=dict)
    termination_terms: dict[str, list[bool]] = field(default_factory=dict)
    fall_termination_terms: tuple[str, ...] = ()
    timeout_termination_terms: tuple[str, ...] = ()
    termination_semantics: str = ""
    termination_multi_hot_available: bool = False
    termination_trace_contract: Mapping[str, Any] | None = None
    termination_trace_contract_sha256: str | None = None
    termination_trace_generations: list[int] = field(default_factory=list)
    termination_trace_step_counters: list[int] = field(default_factory=list)


class EpisodeAssembler:
    """Accumulate bounded smoke-run frames and finalize JSON-ready episodes.

    This is deliberately a full-episode CPU buffer so the pure probe contract
    can be audited before an optimized online/event-window recorder is built.
    ``max_episode_steps`` makes that limitation explicit and prevents an
    accidentally unbounded atlas collection. Scientific atlas mode enables
    ``quiesce_after_first_completion`` so Isaac auto-resets cannot reopen a
    frozen schedule row while slower vectorized environments are unfinished.
    """

    _ARRAY_FIELDS = (
        "reference_contacts",
        "actual_contacts",
        "foot_tangential_speed",
        "base_translation_error",
        "base_orientation_error",
        "base_tilt",
        "requested_torque",
        "applied_torque",
        "effort_limits",
        "reference_joint_position",
        "joint_position",
        "joint_soft_lower_limits",
        "joint_soft_upper_limits",
        "local_pose_error",
        "failure_mask",
        "fall_mask",
    )

    def __init__(
        self,
        *,
        num_envs: int,
        timestep_seconds: float,
        thresholds: ProbeThresholds | None = None,
        max_episode_steps: int = 10_000,
        ground_normal_axis: int = 2,
        quiesce_after_first_completion: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not math.isfinite(timestep_seconds) or timestep_seconds <= 0.0:
            raise ValueError("timestep_seconds must be finite and positive")
        if (
            not isinstance(max_episode_steps, int)
            or isinstance(max_episode_steps, bool)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        self.num_envs = int(num_envs)
        self.timestep_seconds = float(timestep_seconds)
        self.thresholds = thresholds if thresholds is not None else ProbeThresholds()
        if not isinstance(self.thresholds, ProbeThresholds):
            raise TypeError("thresholds must be a ProbeThresholds instance")
        self.max_episode_steps = max_episode_steps
        if not isinstance(quiesce_after_first_completion, bool):
            raise TypeError("quiesce_after_first_completion must be boolean")
        self.quiesce_after_first_completion = quiesce_after_first_completion
        if (
            not isinstance(ground_normal_axis, int)
            or isinstance(ground_normal_axis, bool)
            or ground_normal_axis not in (0, 1, 2)
        ):
            raise ValueError("ground_normal_axis must be one of 0, 1, or 2")
        self.ground_normal_axis = ground_normal_axis
        self._threshold_record = asdict(self.thresholds)
        threshold_payload = json.dumps(
            self._threshold_record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._threshold_sha256 = hashlib.sha256(threshold_payload).hexdigest()
        self._buffers: list[_EpisodeBuffer | None] = [None] * self.num_envs
        self._completed_ids: set[str] = set()
        self._quiescent_env_indices: set[int] = set()

    def active_env_indices(self) -> tuple[int, ...]:
        return tuple(index for index, buffer in enumerate(self._buffers) if buffer is not None)

    def quiescent_env_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._quiescent_env_indices))

    def unfinished_env_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index in range(self.num_envs) if index not in self._quiescent_env_indices
        )

    def assert_empty(self, env_indices: Sequence[int]) -> None:
        stale = [int(index) for index in env_indices if self._buffers[int(index)] is not None]
        if stale:
            raise RuntimeError(
                "LACE observed a reset without a post-step episode_end_mask; refusing to mix "
                f"pre- and post-reset frames for environments {stale}"
            )

    def append(
        self,
        frame: ProbeFrameBatch,
        metadata: Sequence[RolloutMetadata],
    ) -> list[dict[str, Any]]:
        """Append one frame and return episodes completed at this post-step."""

        if frame.num_envs != self.num_envs:
            raise ValueError("frame num_envs does not match EpisodeAssembler")
        if len(metadata) != self.num_envs:
            raise ValueError("metadata length must match num_envs")

        completed: list[dict[str, Any]] = []
        for env_index, current_metadata in enumerate(metadata):
            if current_metadata.env_index != env_index:
                raise ValueError("metadata env_index ordering must match the vectorized frame")
            if env_index in self._quiescent_env_indices:
                if self._buffers[env_index] is not None:
                    raise RuntimeError("quiescent LACE environment retained an episode buffer")
                continue
            buffer = self._buffers[env_index]
            if (
                buffer is not None
                and buffer.metadata.scientific_identity != current_metadata.scientific_identity
            ):
                raise RuntimeError(
                    "motion assignment changed before an explicit episode end for environment "
                    f"{env_index}; LACE cannot distinguish reference completion from unexpected "
                    "reassignment and refuses to label a partial rollout"
                )
            if buffer is None:
                buffer = _EpisodeBuffer(
                    metadata=current_metadata,
                    fields={name: [] for name in self._ARRAY_FIELDS},
                    termination_terms={name: [] for name in frame.termination_terms},
                    fall_termination_terms=frame.fall_termination_terms,
                    timeout_termination_terms=frame.timeout_termination_terms,
                    termination_semantics=frame.termination_semantics,
                    termination_multi_hot_available=frame.termination_multi_hot_available,
                    termination_trace_contract=frame.termination_trace_contract,
                    termination_trace_contract_sha256=(frame.termination_trace_contract_sha256),
                )
                self._buffers[env_index] = buffer
            if set(buffer.termination_terms) != set(frame.termination_terms):
                raise RuntimeError("active termination terms changed during a LACE episode")
            if buffer.fall_termination_terms != frame.fall_termination_terms:
                raise RuntimeError("fall termination terms changed during a LACE episode")
            if buffer.timeout_termination_terms != frame.timeout_termination_terms:
                raise RuntimeError("timeout termination terms changed during a LACE episode")
            if buffer.termination_semantics != frame.termination_semantics:
                raise RuntimeError("termination semantics changed during a LACE episode")
            if buffer.termination_multi_hot_available != frame.termination_multi_hot_available:
                raise RuntimeError("termination multi-hot availability changed during an episode")
            if (
                buffer.termination_trace_contract_sha256 != frame.termination_trace_contract_sha256
                or buffer.termination_trace_contract != frame.termination_trace_contract
            ):
                raise RuntimeError("termination trace contract changed during a LACE episode")
            buffered_steps = len(buffer.fields[self._ARRAY_FIELDS[0]])
            if buffered_steps >= self.max_episode_steps:
                raise RuntimeError(
                    f"LACE episode in environment {env_index} exceeded max_episode_steps="
                    f"{self.max_episode_steps}; use a calibrated bound or an online recorder"
                )

            for name in self._ARRAY_FIELDS:
                values = np.asarray(getattr(frame, name))
                buffer.fields[name].append(values[env_index].copy())
            for name, values in frame.termination_terms.items():
                buffer.termination_terms[name].append(bool(values[env_index]))
            if frame.termination_multi_hot_available:
                assert frame.termination_trace_generation is not None
                assert frame.termination_trace_step_counter is not None
                if (
                    buffer.termination_trace_generations
                    and frame.termination_trace_generation
                    <= buffer.termination_trace_generations[-1]
                ):
                    raise RuntimeError("termination trace generation is not strictly increasing")
                if (
                    buffer.termination_trace_step_counters
                    and frame.termination_trace_step_counter
                    <= buffer.termination_trace_step_counters[-1]
                ):
                    raise RuntimeError("termination trace step counter is not strictly increasing")
                buffer.termination_trace_generations.append(frame.termination_trace_generation)
                buffer.termination_trace_step_counters.append(frame.termination_trace_step_counter)

            if bool(frame.episode_end_mask[env_index]):
                completed.append(self._finalize(env_index, "episode_end"))
                if self.quiesce_after_first_completion:
                    self._quiescent_env_indices.add(env_index)
        return completed

    def _finalize(self, env_index: int, completion_reason: str) -> dict[str, Any]:
        buffer = self._buffers[env_index]
        if buffer is None:
            raise RuntimeError(f"environment {env_index} has no active LACE episode")
        if buffer.metadata.rollout_id in self._completed_ids:
            raise RuntimeError(
                f"duplicate completed rollout_id {buffer.metadata.rollout_id!r}; increment repeat_index"
            )
        arrays = {name: np.stack(values) for name, values in buffer.fields.items()}
        result = compute_episode_probe(
            reference_contacts=arrays["reference_contacts"],
            actual_contacts=arrays["actual_contacts"],
            foot_tangential_speed=arrays["foot_tangential_speed"],
            base_translation_error=arrays["base_translation_error"],
            base_orientation_error=arrays["base_orientation_error"],
            base_tilt=arrays["base_tilt"],
            requested_torque=arrays["requested_torque"],
            applied_torque=arrays["applied_torque"],
            effort_limits=arrays["effort_limits"],
            reference_joint_position=arrays["reference_joint_position"],
            joint_position=arrays["joint_position"],
            joint_soft_lower_limits=arrays["joint_soft_lower_limits"],
            joint_soft_upper_limits=arrays["joint_soft_upper_limits"],
            local_pose_error=arrays["local_pose_error"],
            fall_mask=arrays["fall_mask"],
            failure_mask=arrays["failure_mask"],
            timestep_seconds=self.timestep_seconds,
            thresholds=self.thresholds,
        )
        term_diagnostics: dict[str, dict[str, float | int | bool | None]] = {}
        for name, values in buffer.termination_terms.items():
            mask = np.asarray(values, dtype=bool)
            indices = np.flatnonzero(mask)
            onset_index = int(indices[0]) if indices.size else None
            term_diagnostics[name] = {
                "occurred": bool(indices.size),
                "incidence": float(mask.mean()),
                "onset_index": onset_index,
                "onset_time_seconds": (
                    float(onset_index * self.timestep_seconds) if onset_index is not None else None
                ),
            }

        metadata = buffer.metadata
        episode_diagnostics = dict(result.episode_diagnostics)
        observed_steps = int(arrays["failure_mask"].shape[0])
        reference_denominator = max(metadata.reference_num_steps - 1, 1)
        first_failure_index = episode_diagnostics["first_failure_index"]
        reference_failure_step = (
            metadata.reference_start_step + int(first_failure_index)
            if first_failure_index is not None
            else None
        )
        reference_end_step = min(
            metadata.reference_start_step + observed_steps - 1,
            metadata.reference_num_steps - 1,
        )
        episode_diagnostics.update(
            {
                "reference_start_step": metadata.reference_start_step,
                "reference_num_steps": metadata.reference_num_steps,
                "reference_end_step": reference_end_step,
                "reference_progress_at_end": float(reference_end_step / reference_denominator),
                "reference_failure_step": reference_failure_step,
                "reference_progress_to_failure": (
                    float(reference_failure_step / reference_denominator)
                    if reference_failure_step is not None
                    else None
                ),
                "failure_progress_censored": reference_failure_step is None,
            }
        )
        record: dict[str, Any] = {
            "rollout_id": metadata.rollout_id,
            "motion_key": metadata.motion_key,
            # build_atlas_manifest consumes probe_policy_id. policy_id is retained
            # as an explicit human-facing alias in raw rollout records.
            "probe_policy_id": metadata.policy_id,
            "policy_id": metadata.policy_id,
            "domain_randomization_seed": metadata.domain_randomization_seed,
            "domain_randomization_seed_semantics": (
                DOMAIN_RANDOMIZATION_SEED_SEMANTICS
                if metadata.runtime_rng_seed is not None
                else "declared_run_seed_not_runtime_verified"
            ),
            "partition": metadata.partition,
            "reference_start_step": metadata.reference_start_step,
            "reference_num_steps": metadata.reference_num_steps,
            "initial_phase": metadata.initial_phase,
            "repeat_index": metadata.repeat_index,
            "failed": bool(result.episode_diagnostics["failed"]),
            "mechanism_scores": dict(result.mechanism_scores),
            "completion_reason": completion_reason,
            "termination_terms": term_diagnostics,
            "termination_semantics": buffer.termination_semantics,
            "termination_multi_hot_available": buffer.termination_multi_hot_available,
            "fall_termination_terms": list(buffer.fall_termination_terms),
            "fall_signal_available": bool(buffer.fall_termination_terms),
            "fall_signal_semantics": (
                buffer.termination_semantics
                if buffer.fall_termination_terms
                else "unavailable_explicit_false"
            ),
            "timeout_termination_terms": list(buffer.timeout_termination_terms),
            "foot_slip_velocity_proxy": "link_origin_velocity_tangent_to_configured_plane",
            "ground_normal_axis": self.ground_normal_axis,
            "buffering_semantics": "bounded_full_episode_cpu_smoke",
            "buffered_step_count": observed_steps,
            "max_episode_steps": self.max_episode_steps,
            "probe_thresholds": dict(self._threshold_record),
            "probe_thresholds_sha256": self._threshold_sha256,
            "probe": {
                "scores": dict(result.mechanism_scores),
                "onsets": dict(result.onset_times_seconds),
                "onset_unit": "seconds",
                "diagnostics": {name: dict(values) for name, values in result.diagnostics.items()},
                "episode_diagnostics": episode_diagnostics,
            },
        }
        if buffer.termination_multi_hot_available:
            if (
                not isinstance(buffer.termination_trace_contract, Mapping)
                or not isinstance(buffer.termination_trace_contract_sha256, str)
                or canonical_sha256(buffer.termination_trace_contract)
                != buffer.termination_trace_contract_sha256
            ):
                raise RuntimeError("episode termination trace contract is invalid")
            multi_hot_values = np.column_stack(
                [np.asarray(values, dtype=bool) for values in buffer.termination_terms.values()]
            )
            record["termination_multi_hot"] = {
                "term_names": list(buffer.termination_terms),
                "time_out_flags": [
                    name in buffer.timeout_termination_terms for name in buffer.termination_terms
                ],
                "values": multi_hot_values.tolist(),
                "trace_generations": list(buffer.termination_trace_generations),
                "common_step_counters": list(buffer.termination_trace_step_counters),
                "contract": dict(buffer.termination_trace_contract),
                "contract_sha256": buffer.termination_trace_contract_sha256,
            }
        if metadata.runtime_rng_seed is not None:
            realization = metadata.domain_randomization_realization
            realization_sha256 = metadata.domain_randomization_realization_sha256
            if not isinstance(realization, Mapping) or not isinstance(realization_sha256, str):
                raise RuntimeError(
                    "scheduled atlas metadata is missing its record_post_reset runtime realization"
                )
            if canonical_sha256(realization) != realization_sha256:
                raise RuntimeError(
                    "scheduled atlas runtime realization changed after record_post_reset"
                )
            if metadata.runtime_rng_seed_semantics != SCIENTIFIC_SEED_SEMANTICS:
                raise RuntimeError("scheduled atlas runtime RNG semantics are not scientific")
            robot_contract = metadata.robot_contract_readback
            robot_contract_sha256 = metadata.robot_contract_readback_sha256
            if not isinstance(robot_contract, Mapping) or not isinstance(
                robot_contract_sha256,
                str,
            ):
                raise RuntimeError("scheduled atlas metadata is missing robot contract readback")
            if canonical_sha256(robot_contract) != robot_contract_sha256:
                raise RuntimeError("scheduled atlas robot contract readback changed during episode")
            record.update(
                {
                    "checkpoint_sha256": metadata.checkpoint_sha256,
                    "runtime_rng_seed": metadata.runtime_rng_seed,
                    "runtime_rng_seed_readback": metadata.runtime_rng_seed_readback,
                    "runtime_rng_seed_readback_source": "env.cfg.seed",
                    "runtime_rng_seed_semantics": metadata.runtime_rng_seed_semantics,
                    "phase_id": metadata.phase_id,
                    "target_fraction": metadata.target_fraction,
                    "realized_fraction": metadata.realized_fraction,
                    "split_sha256": metadata.split_sha256,
                    "split_selection_sha256": metadata.split_selection_sha256,
                    "schedule_sha256": metadata.schedule_sha256,
                    "schedule_entry_id": metadata.schedule_entry_id,
                    "domain_randomization_realization": dict(realization),
                    "domain_randomization_realization_sha256": realization_sha256,
                    "robot_contract_readback": dict(robot_contract),
                    "robot_contract_readback_sha256": robot_contract_sha256,
                    "scientific_runtime_ready": buffer.termination_multi_hot_available,
                    "scientific_runtime_blockers": (
                        []
                        if buffer.termination_multi_hot_available
                        else ["independent_termination_multi_hot_unavailable"]
                    ),
                }
            )
        # This is both a serialization check and a guard against NaN/Inf leaking
        # out of a third-party simulator tensor.
        json.dumps(record, allow_nan=False, sort_keys=True)
        self._completed_ids.add(metadata.rollout_id)
        self._buffers[env_index] = None
        return record


_RUNTIME_TYPES: tuple[type[Any], type[Any], type[Any]] | None = None


def _append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_isaac_recorder_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Return term, term-config, and manager-config types after Isaac launches."""

    global _RUNTIME_TYPES
    if _RUNTIME_TYPES is not None:
        return _RUNTIME_TYPES
    try:
        from isaaclab.managers import manager_term_cfg, recorder_manager
        from isaaclab.utils import configclass
    except Exception as error:  # noqa: BLE001 - AppLauncher import boundary
        raise RuntimeError(
            "Isaac Lab recorder types are unavailable. Call this only after AppLauncher "
            "has started the Isaac application."
        ) from error

    class LaceIsaacRecorderTerm(recorder_manager.RecorderTerm):
        """Opt-in RecorderTerm that captures the pre-reset SONIC post-step."""

        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            self.cfg = cfg
            self.env = env
            self._enabled = bool(cfg.enabled)
            self._assembler: EpisodeAssembler | None = None
            self._runtime_realizations: list[RuntimeRealization | None] | None = None
            self._termination_trace_contract: Mapping[str, Any] | None = None
            self._termination_trace_contract_sha256: str | None = None
            self._record_count = 0
            if not self._enabled:
                return
            if str(cfg.partition) != "D_atlas":
                raise ValueError("the failure-atlas recorder partition must be 'D_atlas'")
            output_path = Path(str(cfg.output_path)).expanduser()
            if not str(cfg.output_path):
                raise ValueError("LACE output_path is required when the recorder is enabled")
            if (
                output_path.exists()
                and output_path.stat().st_size > 0
                and not cfg.allow_append_existing
            ):
                raise FileExistsError(
                    f"LACE output already exists: {output_path}; use a new run path or explicitly "
                    "set allow_append_existing=true"
                )
            self._output_path = output_path
            self._bindings = IsaacBindings(
                command_name=str(cfg.command_name),
                robot_name=str(cfg.robot_name),
                joint_action_name=str(cfg.joint_action_name),
                contact_sensor_name=str(cfg.contact_sensor_name),
                foot_body_names=tuple(cfg.foot_body_names),
                contact_force_threshold=float(cfg.contact_force_threshold),
                ground_normal_axis=int(cfg.ground_normal_axis),
                fall_termination_terms=tuple(cfg.fall_termination_terms),
                timeout_termination_terms=tuple(cfg.timeout_termination_terms),
            )
            self._bindings.validate()
            self._assembler = EpisodeAssembler(
                num_envs=int(env.num_envs),
                timestep_seconds=float(env.step_dt),
                thresholds=ProbeThresholds(**dict(cfg.probe_thresholds)),
                max_episode_steps=int(cfg.max_episode_steps),
                ground_normal_axis=self._bindings.ground_normal_axis,
                quiesce_after_first_completion=True,
            )
            self._runtime_realizations = [None] * int(env.num_envs)

        def _ensure_termination_trace_installed(self) -> None:
            """Install after manager construction and before termination compute."""

            if not self._enabled:
                return
            termination_manager = _required_attr(self.env, "termination_manager", "env")
            if self._termination_trace_contract is None:
                if self._termination_trace_contract_sha256 is not None:
                    raise RuntimeError("partial LACE termination trace installation state")
                (
                    self._termination_trace_contract,
                    self._termination_trace_contract_sha256,
                ) = install_termination_trace(
                    termination_manager,
                    expected_compute_source_sha256=str(
                        self.cfg.expected_termination_compute_source_sha256
                    ),
                )
                return
            if not isinstance(self._termination_trace_contract_sha256, str):
                raise RuntimeError("partial LACE termination trace installation state")
            _validate_installed_termination_trace(
                termination_manager,
                expected_contract=self._termination_trace_contract,
                expected_contract_sha256=self._termination_trace_contract_sha256,
            )

        def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
            del env_ids
            self._ensure_termination_trace_installed()
            return None, None

        def record_pre_step(self) -> tuple[None, None]:
            # This is the mandatory safeguard for callers that violate Isaac's
            # recommended reset-before-first-step lifecycle.
            self._ensure_termination_trace_installed()
            return None, None

        def record_post_step(self) -> tuple[None, None]:
            if not self._enabled:
                return None, None
            assert self._assembler is not None
            assert self._runtime_realizations is not None
            if (
                self._termination_trace_contract is None
                or self._termination_trace_contract_sha256 is None
            ):
                raise RuntimeError(
                    "LACE termination trace was not installed before termination computation"
                )
            termination_manager = _required_attr(self.env, "termination_manager", "env")
            common_step_counter = _required_attr(
                self.env,
                "common_step_counter",
                "env",
            )
            trace = consume_termination_trace(
                termination_manager,
                expected_common_step_counter=common_step_counter,
            )
            if (
                trace.contract != self._termination_trace_contract
                or trace.contract_sha256 != self._termination_trace_contract_sha256
            ):
                raise RuntimeError("LACE termination trace contract drifted before post-step")
            frame = extract_isaac_post_step(
                self.env,
                self._bindings,
                termination_trace=trace,
            )
            metadata = extract_isaac_rollout_metadata(
                self.env,
                command_name=self._bindings.command_name,
            )
            metadata = bind_runtime_realizations(
                metadata,
                self._runtime_realizations,
                env_indices=self._assembler.unfinished_env_indices(),
            )
            completed = self._assembler.append(frame, metadata)
            _append_jsonl(self._output_path, completed)
            self._record_count += len(completed)
            return None, None

        def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[None, None]:
            if self._enabled and self._assembler is not None:
                self._ensure_termination_trace_installed()
                assert self._runtime_realizations is not None
                indices = _normalized_env_indices(env_ids, int(self.env.num_envs))
                unfinished = set(self._assembler.unfinished_env_indices())
                indices = tuple(index for index in indices if index in unfinished)
                if not indices:
                    return None, None
                self._assembler.assert_empty(indices)
                metadata = extract_isaac_rollout_metadata(
                    self.env,
                    command_name=self._bindings.command_name,
                )
                realizations = extract_isaac_post_reset_realizations(
                    self.env,
                    self._bindings,
                    metadata,
                    indices,
                )
                if tuple(item.env_index for item in realizations) != indices:
                    raise RuntimeError("post-reset realization order drifted from env_ids")
                for realization in realizations:
                    self._runtime_realizations[realization.env_index] = realization
            return None, None

        @property
        def completed_record_count(self) -> int:
            return self._record_count

    @configclass
    class LaceIsaacRecorderCfg(manager_term_cfg.RecorderTermCfg):
        """Hydra/Isaac config; every live write remains disabled by default."""

        class_type = LaceIsaacRecorderTerm
        enabled: bool = False
        output_path: str = ""
        allow_append_existing: bool = False
        partition: str = "D_atlas"
        command_name: str = "motion"
        robot_name: str = "robot"
        joint_action_name: str = "joint_pos"
        contact_sensor_name: str = "contact_forces"
        foot_body_names: list[str] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        contact_force_threshold: float = 10.0
        ground_normal_axis: int = 2
        fall_termination_terms: list[str] = []
        timeout_termination_terms: list[str] = ["time_out"]
        expected_termination_compute_source_sha256: str = (
            EXPECTED_ISAACLAB_TERMINATION_COMPUTE_SHA256
        )
        probe_thresholds: dict[str, float] = {}
        max_episode_steps: int = 10_000

    @configclass
    class LaceRecorderManagerCfg(recorder_manager.RecorderManagerBaseCfg):
        """Manager config with an opt-in failure-atlas term and no HDF5 export."""

        dataset_export_mode = recorder_manager.DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset: bool = False
        # ``Any`` avoids a local-class forward reference during configclass
        # field resolution while retaining the concrete term config at runtime.
        failure_atlas: Any = None

    LaceIsaacRecorderTerm.__module__ = __name__
    LaceIsaacRecorderCfg.__module__ = __name__
    LaceRecorderManagerCfg.__module__ = __name__
    _RUNTIME_TYPES = LaceIsaacRecorderTerm, LaceIsaacRecorderCfg, LaceRecorderManagerCfg
    return _RUNTIME_TYPES


def create_isaac_recorder_cfg(**kwargs: Any) -> Any:
    """Hydra-callable lazy factory for the opt-in recorder configuration."""

    _, config_type, _ = load_isaac_recorder_types()
    return config_type(**kwargs)


def create_isaac_recorder_manager_cfg(**kwargs: Any) -> Any:
    """Hydra-callable lazy factory for the recorder-manager configuration."""

    _, _, manager_config_type = load_isaac_recorder_types()
    return manager_config_type(**kwargs)


def __getattr__(name: str) -> Any:
    """Resolve Hydra's optional class targets without eager Isaac imports."""

    if name not in {
        "LaceIsaacRecorderTerm",
        "LaceIsaacRecorderCfg",
        "LaceRecorderManagerCfg",
    }:
        raise AttributeError(name)
    term_type, config_type, manager_config_type = load_isaac_recorder_types()
    if name == "LaceIsaacRecorderTerm":
        return term_type
    if name == "LaceIsaacRecorderCfg":
        return config_type
    return manager_config_type
