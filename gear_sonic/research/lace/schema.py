"""Versioned JSON artifact contracts for LACE experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

SPLIT_SCHEMA_VERSION = 1
ATLAS_SCHEMA_VERSION = 5
SPLIT_KIND = "lace_source_disjoint_split"
ATLAS_KIND = "lace_failure_atlas"
RUNTIME_REALIZATION_KIND = "lace_runtime_realization"
RUNTIME_REALIZATION_SCHEMA_VERSION = 1
RUNTIME_REALIZATION_CAPTURE_LIFECYCLE = (
    "RecorderTerm.record_post_reset_after_scene_reset_events_action_reset_and_command_reset"
)
ATLAS_V1_INTERVAL_EVENT_POLICY = "reject_active_interval_events_without_realized_event_trace"
ROBOT_CONTRACT_READBACK_KIND = "lace_robot_contract_readback"
ROBOT_CONTRACT_READBACK_SCHEMA_VERSION = 1
ROBOT_CONTRACT_LIMIT_FIELDS = (
    "joint_pos_limits",
    "soft_joint_pos_limits",
    "joint_vel_limits",
    "soft_joint_vel_limits",
)
ROBOT_CONTRACT_SOURCE_PROPERTIES = {
    "joint_names": "robot.joint_names",
    "joint_pos_limits": "robot.data.joint_pos_limits[0]",
    "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
    "joint_vel_limits": "robot.data.joint_vel_limits[0]",
    "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
}
TERMINATION_TRACE_KIND = "lace_termination_multi_hot_trace"
TERMINATION_TRACE_SCHEMA_VERSION = 1
TERMINATION_TRACE_ALGORITHM = "isaaclab_termination_manager_compute_single_evaluation_raw_matrix_v1"
SCIENTIFIC_NORMALIZER_METHOD = "failed_positive_score_q90_linear_v1"
SCIENTIFIC_NORMALIZER_MINIMUM_POSITIVE_OBSERVATIONS = 20
SPLIT_NAMES = (
    "D_atlas",
    "D_curriculum",
    "D_geometry",
    "D_controller",
    "D_test",
)

PROBE_RECORD_FIELDS = {
    "scores",
    "onsets",
    "onset_unit",
    "diagnostics",
    "episode_diagnostics",
}
_PROBE_STREAM_FIELDS = {
    "mean",
    "std",
    "q90",
    "max",
    "incidence",
    "onset_index",
    "onset_time_seconds",
    "pre_failure_slope_per_second",
    "onset_index_within_score_window",
    "onset_time_within_score_window_seconds",
}
_PROBE_DIAGNOSTIC_FIELDS = {
    "contact_timing": _PROBE_STREAM_FIELDS
    | {
        "mismatch_sample_fraction",
        "normalized_onset_offset_component",
        "reference_onset_count",
        "actual_onset_count",
        "matched_onset_pair_count",
        "unmatched_onset_count",
        "onset_matching",
        "mean_absolute_onset_offset_seconds",
        "max_absolute_onset_offset_seconds",
    },
    "foot_slip": _PROBE_STREAM_FIELDS
    | {
        "contact_sample_count",
        "slip_contact_sample_fraction",
        "q90_contact_speed",
        "slip_duration_seconds",
    },
    "base_drift": _PROBE_STREAM_FIELDS
    | {
        "initial_absolute_residual",
        "max_absolute_residual",
        "max_residual_displacement_from_initial",
    },
    "balance_orientation": _PROBE_STREAM_FIELDS
    | {
        "orientation_rms_radians",
        "orientation_q90_radians",
        "max_tilt_radians",
        "fall_incidence",
    },
    "actuation_saturation": _PROBE_STREAM_FIELDS
    | {
        "clipped_joint_fraction",
        "clipped_duration_fraction",
        "q90_requested_torque_ratio",
        "max_requested_torque_ratio",
        "max_applied_torque_ratio",
        "q90_clip_gap_ratio",
        "max_clip_gap_ratio",
        "torque_semantics",
    },
    "joint_pose_constraint": _PROBE_STREAM_FIELDS
    | {
        "excess_near_limit_joint_fraction",
        "excess_near_limit_duration_fraction",
        "q90_excess_limit_severity",
        "q90_actual_limit_severity",
        "q90_reference_limit_severity",
        "q90_pose_rms",
        "max_pose_rms",
        "pose_residual_growth",
    },
}
_PROBE_EPISODE_DIAGNOSTIC_FIELDS = {
    "num_steps",
    "num_feet",
    "num_joints",
    "timestep_seconds",
    "episode_duration_seconds",
    "failed",
    "first_failure_index",
    "first_failure_time_seconds",
    "score_window_rule",
    "score_window_requested_seconds",
    "score_window_start_index",
    "score_window_end_index_exclusive",
    "score_window_num_samples",
    "score_window_duration_seconds",
    "score_window_left_censored",
    "observed_buffer_fraction_to_failure",
    "failure_time_censored",
    "reference_start_step",
    "reference_num_steps",
    "reference_end_step",
    "reference_progress_at_end",
    "reference_failure_step",
    "reference_progress_to_failure",
    "failure_progress_censored",
}


def canonical_sha256(payload: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    """Return a stable SHA-256 over a JSON mapping.

    ``digest_field`` is removed only from the top level. This lets an artifact
    contain and verify its own digest without mutating the caller's object.
    """

    canonical_payload = dict(payload)
    if digest_field is not None:
        canonical_payload.pop(digest_field, None)
    encoded = json.dumps(
        canonical_payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_numeric_values_shape(value: Any, shape: Sequence[int], name: str) -> None:
    if not shape:
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"{name} must contain finite real numbers",
        )
        return
    _require(isinstance(value, list), f"{name} values do not match declared shape")
    _require(len(value) == shape[0], f"{name} values do not match declared shape")
    for index, child in enumerate(value):
        _validate_numeric_values_shape(child, shape[1:], f"{name}[{index}]")


def validate_robot_contract_readback(raw: Mapping[str, Any]) -> None:
    """Validate the exact non-empty robot-contract payload emitted by Isaac."""

    _require(isinstance(raw, Mapping), "robot contract readback must be a mapping")
    contract = dict(raw)
    _require(
        set(contract)
        == {
            "kind",
            "schema_version",
            "capture_lifecycle",
            "ordered_body_names",
            "ordered_joint_names",
            "source_properties",
            "limits",
            "environment_invariance_verified",
        },
        "robot contract readback fields are invalid",
    )
    _require(contract.get("kind") == ROBOT_CONTRACT_READBACK_KIND, "robot contract kind invalid")
    _require(
        contract.get("schema_version") == ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
        "robot contract schema version invalid",
    )
    _require(
        contract.get("capture_lifecycle") == RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "robot contract capture lifecycle invalid",
    )
    for field in ("ordered_body_names", "ordered_joint_names"):
        names = contract.get(field)
        _require(
            isinstance(names, list)
            and bool(names)
            and all(isinstance(name, str) and name for name in names)
            and len(names) == len(set(names)),
            f"robot contract {field} must be non-empty, unique, and ordered",
        )
    _require(
        contract.get("source_properties") == ROBOT_CONTRACT_SOURCE_PROPERTIES,
        "robot contract source properties drifted",
    )
    limits = contract.get("limits")
    _require(
        isinstance(limits, Mapping) and set(limits) == set(ROBOT_CONTRACT_LIMIT_FIELDS),
        "robot contract limit fields are invalid",
    )
    joint_count = len(contract["ordered_joint_names"])
    for field in ROBOT_CONTRACT_LIMIT_FIELDS:
        payload = limits[field]
        _require(
            isinstance(payload, Mapping) and set(payload) == {"dtype", "shape", "values"},
            f"robot contract limit {field!r} payload is invalid",
        )
        expected_shape = [joint_count, 2] if field.endswith("pos_limits") else [joint_count]
        _require(
            isinstance(payload.get("dtype"), str) and bool(payload["dtype"]),
            f"robot contract limit {field!r} dtype is invalid",
        )
        _require(
            payload.get("shape") == expected_shape,
            f"robot contract limit {field!r} shape is invalid",
        )
        _validate_numeric_values_shape(
            payload.get("values"),
            expected_shape,
            f"robot contract limit {field!r}",
        )
    _require(
        contract.get("environment_invariance_verified") is True,
        "robot contract environment invariance is unverified",
    )


def _probe_real(value: Any, name: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite real number",
    )
    result = float(value)
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def _probe_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return int(value)


def _probe_optional_onset(
    index: Any,
    seconds: Any,
    *,
    timestep_seconds: float,
    name: str,
) -> tuple[int | None, float | None]:
    _require((index is None) is (seconds is None), f"{name} index/time nullability drifted")
    if index is None:
        return None, None
    index_value = _probe_integer(index, f"{name}.index")
    seconds_value = _probe_real(seconds, f"{name}.seconds", minimum=0.0)
    _require(
        math.isclose(seconds_value, index_value * timestep_seconds, abs_tol=1e-10),
        f"{name} seconds do not match index and timestep",
    )
    return index_value, seconds_value


def validate_scientific_probe_episode(
    record: Mapping[str, Any],
    *,
    score_window_config: Mapping[str, Any],
    probe_thresholds: Mapping[str, Any],
) -> None:
    """Validate the exact live recorder probe payload and all duplicated fields.

    This deliberately validates the values consumed by atlas geometry rather
    than trusting an independently editable ``mechanism_scores`` mapping.
    """

    _require(isinstance(record, Mapping), "scientific probe episode must be a mapping")
    mechanism_scores = record.get("mechanism_scores")
    _require(
        isinstance(mechanism_scores, Mapping) and set(mechanism_scores) == set(DEFAULT_MECHANISMS),
        "scientific mechanism_scores must exactly cover the frozen channels",
    )
    for name in DEFAULT_MECHANISMS:
        score = _probe_real(mechanism_scores[name], f"mechanism_scores.{name}")
        _require(0.0 <= score <= 1.0, f"mechanism_scores.{name} must be in [0, 1]")

    probe = record.get("probe")
    _require(
        isinstance(probe, Mapping) and set(probe) == PROBE_RECORD_FIELDS,
        "scientific episode probe fields are invalid",
    )
    _require(probe.get("scores") == mechanism_scores, "probe scores drifted from mechanism_scores")
    _require(probe.get("onset_unit") == "seconds", "probe onset unit must be seconds")
    _require(
        isinstance(probe.get("onsets"), Mapping)
        and set(probe["onsets"]) == set(DEFAULT_MECHANISMS),
        "probe onsets must exactly cover the frozen channels",
    )
    _require(
        isinstance(probe.get("diagnostics"), Mapping)
        and set(probe["diagnostics"]) == set(DEFAULT_MECHANISMS),
        "probe diagnostics must exactly cover the frozen channels",
    )

    diagnostics = probe["diagnostics"]
    onsets = probe["onsets"]
    episode = probe.get("episode_diagnostics")
    _require(
        isinstance(episode, Mapping) and set(episode) == _PROBE_EPISODE_DIAGNOSTIC_FIELDS,
        "probe episode diagnostics fields are invalid",
    )
    failed = record.get("failed")
    _require(isinstance(failed, bool), "scientific episode failed must be boolean")
    _require(episode.get("failed") is failed, "probe failed status drifted from episode failed")

    num_steps = _probe_integer(episode.get("num_steps"), "probe.num_steps", minimum=1)
    num_feet = _probe_integer(episode.get("num_feet"), "probe.num_feet", minimum=1)
    _probe_integer(episode.get("num_joints"), "probe.num_joints", minimum=1)
    timestep = _probe_real(episode.get("timestep_seconds"), "probe.timestep_seconds", minimum=0.0)
    _require(timestep > 0.0, "probe.timestep_seconds must be positive")
    _require(
        score_window_config
        == {
            "rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
            "score_window_seconds": probe_thresholds.get("score_window_seconds"),
            "timestep_seconds": timestep,
            "sample_count_rule": ("max(1,floor(score_window_seconds/timestep_seconds+1e-12))"),
        },
        "probe score-window instrument contract drifted",
    )
    requested_window = _probe_real(
        episode.get("score_window_requested_seconds"),
        "probe.score_window_requested_seconds",
        minimum=0.0,
    )
    _require(
        requested_window > 0.0 and requested_window == probe_thresholds.get("score_window_seconds"),
        "probe requested score window drifted from thresholds",
    )
    _require(
        episode.get("score_window_rule") == score_window_config["rule"],
        "probe score window rule drifted",
    )
    _require(
        math.isclose(
            _probe_real(
                episode.get("episode_duration_seconds"),
                "probe.episode_duration_seconds",
                minimum=0.0,
            ),
            num_steps * timestep,
            abs_tol=1e-10,
        ),
        "probe episode duration does not match steps and timestep",
    )
    _require(
        record.get("buffering_semantics") == "bounded_full_episode_cpu_smoke",
        "scientific probe buffering semantics drifted",
    )
    _require(
        record.get("foot_slip_velocity_proxy")
        == "link_origin_velocity_tangent_to_configured_plane",
        "scientific foot-slip proxy drifted",
    )
    _require(
        record.get("ground_normal_axis") in (0, 1, 2)
        and not isinstance(record.get("ground_normal_axis"), bool),
        "scientific ground-normal axis is invalid",
    )
    _require(
        record.get("buffered_step_count") == num_steps,
        "probe num_steps drifted from buffered_step_count",
    )
    max_episode_steps = _probe_integer(
        record.get("max_episode_steps"), "max_episode_steps", minimum=1
    )
    _require(num_steps <= max_episode_steps, "buffered probe exceeds max_episode_steps")

    first_failure_index = episode.get("first_failure_index")
    first_failure_seconds = episode.get("first_failure_time_seconds")
    if failed:
        failure_index, _ = _probe_optional_onset(
            first_failure_index,
            first_failure_seconds,
            timestep_seconds=timestep,
            name="probe.first_failure",
        )
        assert failure_index is not None
        _require(failure_index < num_steps, "probe first failure is outside the episode")
    else:
        _require(
            first_failure_index is None and first_failure_seconds is None,
            "censored probe must not claim a failure onset",
        )
        failure_index = None
    _require(
        episode.get("failure_time_censored") is (not failed),
        "probe failure-time censoring drifted",
    )
    window_start = _probe_integer(
        episode.get("score_window_start_index"), "probe.score_window_start_index"
    )
    window_end = _probe_integer(
        episode.get("score_window_end_index_exclusive"),
        "probe.score_window_end_index_exclusive",
        minimum=1,
    )
    window_count = _probe_integer(
        episode.get("score_window_num_samples"),
        "probe.score_window_num_samples",
        minimum=1,
    )
    max_window_samples = max(1, int(math.floor(requested_window / timestep + 1e-12)))
    expected_window_end = failure_index + 1 if failure_index is not None else num_steps
    expected_window_start = max(0, expected_window_end - max_window_samples)
    _require(
        (window_start, window_end, window_count)
        == (
            expected_window_start,
            expected_window_end,
            expected_window_end - expected_window_start,
        ),
        "probe score-window indices are inconsistent",
    )
    _require(
        math.isclose(
            _probe_real(
                episode.get("score_window_duration_seconds"),
                "probe.score_window_duration_seconds",
                minimum=0.0,
            ),
            window_count * timestep,
            abs_tol=1e-10,
        ),
        "probe score-window duration is inconsistent",
    )
    _require(
        episode.get("score_window_left_censored")
        is (window_start == 0 and window_end < max_window_samples),
        "probe score-window left-censoring drifted",
    )
    observed_fraction = episode.get("observed_buffer_fraction_to_failure")
    if failure_index is None:
        _require(observed_fraction is None, "censored probe has a failure fraction")
    else:
        _require(
            math.isclose(
                _probe_real(observed_fraction, "probe.observed_buffer_fraction_to_failure"),
                failure_index / max(num_steps - 1, 1),
                abs_tol=1e-10,
            ),
            "probe observed failure fraction is inconsistent",
        )

    reference_start = _probe_integer(
        episode.get("reference_start_step"), "probe.reference_start_step"
    )
    reference_steps = _probe_integer(
        episode.get("reference_num_steps"), "probe.reference_num_steps", minimum=2
    )
    _require(
        reference_start == record.get("reference_start_step")
        and reference_steps == record.get("reference_num_steps"),
        "probe reference coordinates drifted from episode coordinates",
    )
    reference_end = min(reference_start + num_steps - 1, reference_steps - 1)
    denominator = max(reference_steps - 1, 1)
    _require(
        episode.get("reference_end_step") == reference_end
        and math.isclose(
            _probe_real(
                episode.get("reference_progress_at_end"),
                "probe.reference_progress_at_end",
            ),
            reference_end / denominator,
            abs_tol=1e-10,
        ),
        "probe reference end/progress is inconsistent",
    )
    expected_reference_failure = (
        reference_start + failure_index if failure_index is not None else None
    )
    _require(
        episode.get("reference_failure_step") == expected_reference_failure
        and episode.get("failure_progress_censored") is (failure_index is None),
        "probe reference failure/censoring is inconsistent",
    )
    if failure_index is None:
        _require(
            episode.get("reference_progress_to_failure") is None,
            "censored probe has reference failure progress",
        )
    else:
        _require(
            math.isclose(
                _probe_real(
                    episode.get("reference_progress_to_failure"),
                    "probe.reference_progress_to_failure",
                ),
                expected_reference_failure / denominator,
                abs_tol=1e-10,
            ),
            "probe reference failure progress is inconsistent",
        )

    probability_fields = {
        "incidence",
        "mismatch_sample_fraction",
        "slip_contact_sample_fraction",
        "fall_incidence",
        "clipped_joint_fraction",
        "clipped_duration_fraction",
        "excess_near_limit_joint_fraction",
        "excess_near_limit_duration_fraction",
    }
    integer_fields = {
        "reference_onset_count",
        "actual_onset_count",
        "matched_onset_pair_count",
        "unmatched_onset_count",
        "contact_sample_count",
    }
    for name in DEFAULT_MECHANISMS:
        diagnostic = diagnostics[name]
        _require(
            isinstance(diagnostic, Mapping) and set(diagnostic) == _PROBE_DIAGNOSTIC_FIELDS[name],
            f"probe diagnostics.{name} fields are invalid",
        )
        for field, value in diagnostic.items():
            qualified = f"probe.diagnostics.{name}.{field}"
            if field in {
                "onset_index",
                "onset_time_seconds",
                "onset_index_within_score_window",
                "onset_time_within_score_window_seconds",
            }:
                continue
            if field == "onset_matching":
                _require(
                    value == "ordered_one_to_one_minimum_cost_v1",
                    "contact onset matching semantics drifted",
                )
            elif field == "torque_semantics":
                _require(
                    value == "implicit_actuator_requested_pd_proxy",
                    "actuation torque semantics drifted",
                )
            elif field in integer_fields:
                _probe_integer(value, qualified)
            elif field in probability_fields:
                probability = _probe_real(value, qualified)
                _require(0.0 <= probability <= 1.0, f"{qualified} must be in [0, 1]")
            elif field == "pre_failure_slope_per_second":
                _probe_real(value, qualified)
            else:
                _probe_real(value, qualified, minimum=0.0)
        _require(
            diagnostic["mean"] <= diagnostic["max"] + 1e-12
            and diagnostic["q90"] <= diagnostic["max"] + 1e-12,
            f"probe diagnostics.{name} summary order is invalid",
        )
        absolute_index, absolute_seconds = _probe_optional_onset(
            diagnostic["onset_index"],
            diagnostic["onset_time_seconds"],
            timestep_seconds=timestep,
            name=f"probe.diagnostics.{name}.onset",
        )
        relative_index, _ = _probe_optional_onset(
            diagnostic["onset_index_within_score_window"],
            diagnostic["onset_time_within_score_window_seconds"],
            timestep_seconds=timestep,
            name=f"probe.diagnostics.{name}.window_onset",
        )
        _require(
            (absolute_index is None and relative_index is None)
            or absolute_index == window_start + relative_index,
            f"probe diagnostics.{name} onset coordinates are inconsistent",
        )
        _require(
            onsets[name] == absolute_seconds,
            f"probe onset for {name} drifted from channel diagnostics",
        )
    contact = diagnostics["contact_timing"]
    _require(
        contact["reference_onset_count"] + contact["actual_onset_count"]
        == 2 * contact["matched_onset_pair_count"] + contact["unmatched_onset_count"],
        "contact onset counts are inconsistent",
    )
    _require(
        diagnostics["foot_slip"]["contact_sample_count"] <= window_count * num_feet,
        "foot-slip contact sample count exceeds the score window",
    )

    def bounded_excess(value: Any, threshold_name: str, saturation_name: str) -> float:
        raw_value = _probe_real(value, f"probe score input {threshold_name}", minimum=0.0)
        threshold = _probe_real(
            probe_thresholds.get(threshold_name),
            f"probe_thresholds.{threshold_name}",
            minimum=0.0,
        )
        saturation = _probe_real(
            probe_thresholds.get(saturation_name),
            f"probe_thresholds.{saturation_name}",
            minimum=0.0,
        )
        _require(
            threshold < saturation,
            f"probe thresholds {threshold_name}/{saturation_name} are inconsistent",
        )
        return min(max((raw_value - threshold) / (saturation - threshold), 0.0), 1.0)

    def component_mean(*values: float) -> float:
        _require(bool(values), "scientific probe score requires components")
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
            "scientific probe score component is outside [0, 1]",
        )
        return sum(values) / len(values)

    onset_component = _probe_real(
        contact["normalized_onset_offset_component"],
        "probe.diagnostics.contact_timing.normalized_onset_offset_component",
    )
    _require(
        0.0 <= onset_component <= 1.0,
        "contact normalized onset component must be in [0, 1]",
    )
    if contact["reference_onset_count"] == 0 and contact["actual_onset_count"] == 0:
        _require(
            onset_component == 0.0,
            "contact normalized onset component must be zero without onsets",
        )

    slip = diagnostics["foot_slip"]
    drift = diagnostics["base_drift"]
    balance = diagnostics["balance_orientation"]
    actuation = diagnostics["actuation_saturation"]
    joint_pose = diagnostics["joint_pose_constraint"]
    reconstructed_scores = {
        "contact_timing": component_mean(
            float(contact["mismatch_sample_fraction"]),
            onset_component,
        ),
        "foot_slip": component_mean(
            float(slip["slip_contact_sample_fraction"]),
            bounded_excess(
                slip["q90_contact_speed"],
                "slip_speed_threshold",
                "slip_speed_saturation",
            ),
        ),
        "base_drift": bounded_excess(
            drift["max_residual_displacement_from_initial"],
            "base_drift_threshold",
            "base_drift_saturation",
        ),
        "balance_orientation": component_mean(
            bounded_excess(
                balance["orientation_rms_radians"],
                "orientation_error_threshold",
                "orientation_error_saturation",
            ),
            bounded_excess(
                balance["max_tilt_radians"],
                "base_tilt_threshold",
                "base_tilt_saturation",
            ),
        ),
        "actuation_saturation": component_mean(
            float(actuation["clipped_joint_fraction"]),
            float(actuation["clipped_duration_fraction"]),
            bounded_excess(
                actuation["q90_requested_torque_ratio"],
                "torque_ratio_threshold",
                "torque_ratio_saturation",
            ),
            bounded_excess(
                actuation["q90_clip_gap_ratio"],
                "torque_clip_gap_ratio_threshold",
                "torque_clip_gap_ratio_saturation",
            ),
        ),
        "joint_pose_constraint": component_mean(
            float(joint_pose["excess_near_limit_joint_fraction"]),
            float(joint_pose["excess_near_limit_duration_fraction"]),
            bounded_excess(
                joint_pose["q90_excess_limit_severity"],
                "joint_limit_excess_threshold",
                "joint_limit_excess_saturation",
            ),
            bounded_excess(
                joint_pose["q90_pose_rms"],
                "pose_error_threshold",
                "pose_error_saturation",
            ),
            bounded_excess(
                joint_pose["pose_residual_growth"],
                "pose_error_threshold",
                "pose_error_saturation",
            ),
        ),
    }
    for name in DEFAULT_MECHANISMS:
        _require(
            math.isclose(
                float(mechanism_scores[name]),
                reconstructed_scores[name],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"mechanism_scores.{name} does not match frozen diagnostic reconstruction",
        )


def _require_probability(value: Any, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{name} must be in [0, 1]")
    return result


def _verify_self_digest(manifest: Mapping[str, Any], field: str) -> None:
    expected = manifest.get(field)
    _require(isinstance(expected, str) and len(expected) == 64, f"{field} must be a SHA-256")
    actual = canonical_sha256(manifest, digest_field=field)
    _require(expected == actual, f"{field} mismatch: expected {expected}, computed {actual}")


def _unmirrored_motion_key(motion_key: str) -> str:
    return motion_key[:-2] if motion_key.endswith("_M") else motion_key


def split_selection_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash only scientific split membership, independent of machine paths."""

    motions = manifest.get("motions", [])
    selection = {
        "schema_version": manifest.get("schema_version"),
        "seed": manifest.get("seed"),
        "ratios": manifest.get("ratios"),
        "grouping_rule": manifest.get("grouping_rule"),
        "motions": [
            {
                "motion_key": record.get("motion_key"),
                "source_group_id": record.get("source_group_id"),
                "partition": record.get("partition"),
            }
            for record in motions
        ],
    }
    return canonical_sha256(selection)


def validate_split_manifest(manifest: Mapping[str, Any], *, verify_digest: bool = True) -> None:
    """Validate the frozen, source-disjoint LACE partition manifest."""

    _require(manifest.get("kind") == SPLIT_KIND, f"kind must be {SPLIT_KIND!r}")
    _require(
        manifest.get("schema_version") == SPLIT_SCHEMA_VERSION,
        f"unsupported split schema version: {manifest.get('schema_version')!r}",
    )
    motions = manifest.get("motions")
    _require(isinstance(motions, list) and motions, "motions must be a non-empty list")

    seen_motion_keys: set[str] = set()
    source_partitions: dict[str, str] = {}
    mirror_partitions: dict[str, str] = {}
    partition_counts = {name: 0 for name in SPLIT_NAMES}

    for index, record in enumerate(motions):
        _require(isinstance(record, Mapping), f"motions[{index}] must be a mapping")
        motion_key = record.get("motion_key")
        source_group_id = record.get("source_group_id")
        partition = record.get("partition")
        _require(isinstance(motion_key, str) and motion_key, f"motions[{index}].motion_key missing")
        _require(
            isinstance(source_group_id, str) and source_group_id,
            f"motions[{index}].source_group_id missing",
        )
        _require(partition in SPLIT_NAMES, f"motions[{index}].partition is invalid: {partition!r}")
        _require(motion_key not in seen_motion_keys, f"duplicate motion_key: {motion_key}")
        seen_motion_keys.add(motion_key)
        partition_counts[partition] += 1

        prior_partition = source_partitions.setdefault(source_group_id, partition)
        _require(
            prior_partition == partition,
            f"source group {source_group_id!r} leaks across {prior_partition} and {partition}",
        )

        base_key = _unmirrored_motion_key(motion_key)
        prior_mirror_partition = mirror_partitions.setdefault(base_key, partition)
        _require(
            prior_mirror_partition == partition,
            f"mirror family {base_key!r} leaks across partitions",
        )

    missing_partitions = [name for name, count in partition_counts.items() if count == 0]
    _require(not missing_partitions, f"empty required partitions: {missing_partitions}")

    summary = manifest.get("partition_summary")
    _require(isinstance(summary, Mapping), "partition_summary must be a mapping")
    for partition, observed_count in partition_counts.items():
        summary_record = summary.get(partition)
        _require(isinstance(summary_record, Mapping), f"partition_summary.{partition} missing")
        _require(
            int(summary_record.get("motion_count", -1)) == observed_count,
            f"partition_summary.{partition}.motion_count does not match motions",
        )

    if verify_digest:
        selection_digest = manifest.get("selection_sha256")
        _require(
            selection_digest == split_selection_sha256(manifest),
            "selection_sha256 does not match scientific split membership",
        )
        _verify_self_digest(manifest, "split_sha256")


def _validate_signature(
    signature: Mapping[str, Any], mechanism_count: int, *, index: int
) -> tuple[str, str]:
    prefix = f"signatures[{index}]"
    motion_key = signature.get("motion_key")
    probe_policy_id = signature.get("probe_policy_id")
    _require(isinstance(motion_key, str) and motion_key, f"{prefix}.motion_key missing")
    _require(
        isinstance(probe_policy_id, str) and probe_policy_id,
        f"{prefix}.probe_policy_id missing",
    )
    num_rollouts = int(signature.get("num_rollouts", -1))
    num_failures = int(signature.get("num_failures", -1))
    num_resolved_failures = int(signature.get("num_resolved_failures", -1))
    _require(num_rollouts > 0, f"{prefix}.num_rollouts must be positive")
    _require(0 <= num_failures <= num_rollouts, f"{prefix}.num_failures is inconsistent")
    _require(
        0 <= num_resolved_failures <= num_failures,
        f"{prefix}.num_resolved_failures is inconsistent",
    )
    difficulty = _require_probability(signature.get("difficulty"), f"{prefix}.difficulty")
    attributed_failure_rate = _require_probability(
        signature.get("attributed_failure_rate"),
        f"{prefix}.attributed_failure_rate",
    )
    unresolved_failure_probability = _require_probability(
        signature.get("unresolved_failure_probability"),
        f"{prefix}.unresolved_failure_probability",
    )
    _require(
        math.isclose(attributed_failure_rate, num_resolved_failures / num_rollouts, abs_tol=1e-8),
        f"{prefix}.attributed_failure_rate does not match counts",
    )
    _require(
        math.isclose(
            unresolved_failure_probability,
            (num_failures - num_resolved_failures) / num_rollouts,
            abs_tol=1e-8,
        ),
        f"{prefix}.unresolved_failure_probability does not match counts",
    )
    _require(
        math.isclose(
            difficulty,
            attributed_failure_rate + unresolved_failure_probability,
            abs_tol=1e-8,
        ),
        f"{prefix}.failure mass does not sum to difficulty",
    )

    q = signature.get("q")
    f = signature.get("f")
    if q is None:
        _require(f is None, f"{prefix}.f must be null when q is null")
        for field in (
            "mechanism_posterior_alpha",
            "q_posterior_mean",
            "q_credible_interval",
            "f_posterior_mean",
            "f_all_failure_mar_sensitivity",
        ):
            _require(signature.get(field) is None, f"{prefix}.{field} must be null when q is null")
    else:
        _require(
            isinstance(q, Sequence) and not isinstance(q, (str, bytes)),
            f"{prefix}.q must be a sequence or null",
        )
        _require(len(q) == mechanism_count, f"{prefix}.q has the wrong length")
        q_values = [float(value) for value in q]
        _require(all(math.isfinite(x) and x >= 0.0 for x in q_values), f"{prefix}.q is invalid")
        _require(math.isclose(sum(q_values), 1.0, abs_tol=1e-8), f"{prefix}.q must sum to one")
        _require(
            isinstance(f, Sequence) and not isinstance(f, (str, bytes)),
            f"{prefix}.f must be a sequence",
        )
        _require(len(f) == mechanism_count, f"{prefix}.f has the wrong length")
        for mechanism_index, (q_value, f_value) in enumerate(zip(q_values, f, strict=True)):
            _require(
                math.isclose(float(f_value), attributed_failure_rate * q_value, abs_tol=1e-8),
                f"{prefix}.f[{mechanism_index}] must equal attributed_failure_rate * q",
            )
        posterior_alpha = signature.get("mechanism_posterior_alpha")
        posterior_mean = signature.get("q_posterior_mean")
        intervals = signature.get("q_credible_interval")
        f_posterior = signature.get("f_posterior_mean")
        mar_sensitivity = signature.get("f_all_failure_mar_sensitivity")
        for field, values in (
            ("mechanism_posterior_alpha", posterior_alpha),
            ("q_posterior_mean", posterior_mean),
            ("q_credible_interval", intervals),
            ("f_posterior_mean", f_posterior),
            ("f_all_failure_mar_sensitivity", mar_sensitivity),
        ):
            _require(
                isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
                f"{prefix}.{field} must be a sequence",
            )
            _require(len(values) == mechanism_count, f"{prefix}.{field} has the wrong length")
        posterior_values = [float(value) for value in posterior_mean]
        posterior_alpha_values = [float(value) for value in posterior_alpha]
        _require(
            all(math.isfinite(value) and value > 0.0 for value in posterior_alpha_values),
            f"{prefix}.mechanism_posterior_alpha is invalid",
        )
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in posterior_values),
            f"{prefix}.q_posterior_mean is invalid",
        )
        _require(
            math.isclose(sum(posterior_values), 1.0, abs_tol=1e-8),
            f"{prefix}.q_posterior_mean must sum to one",
        )
        for mechanism_index, interval in enumerate(intervals):
            _require(
                isinstance(interval, Sequence)
                and not isinstance(interval, (str, bytes))
                and len(interval) == 2,
                f"{prefix}.q_credible_interval[{mechanism_index}] must be [lower, upper]",
            )
            lower = _require_probability(
                interval[0], f"{prefix}.q_credible_interval[{mechanism_index}][0]"
            )
            upper = _require_probability(
                interval[1], f"{prefix}.q_credible_interval[{mechanism_index}][1]"
            )
            _require(lower <= upper, f"{prefix}.q_credible_interval bounds are reversed")
        for mechanism_index, (posterior_value, f_value, sensitivity_value) in enumerate(
            zip(posterior_values, f_posterior, mar_sensitivity, strict=True)
        ):
            _require(
                math.isclose(
                    float(f_value),
                    attributed_failure_rate * posterior_value,
                    abs_tol=1e-8,
                ),
                f"{prefix}.f_posterior_mean[{mechanism_index}] is inconsistent",
            )
            _require(
                math.isclose(
                    float(sensitivity_value),
                    difficulty * q_values[mechanism_index],
                    abs_tol=1e-8,
                ),
                f"{prefix}.f_all_failure_mar_sensitivity[{mechanism_index}] is inconsistent",
            )
    return motion_key, probe_policy_id


def validate_atlas_manifest(manifest: Mapping[str, Any], *, verify_digest: bool = True) -> None:
    """Validate a frozen LACE failure-atlas aggregate artifact."""

    _require(manifest.get("kind") == ATLAS_KIND, f"kind must be {ATLAS_KIND!r}")
    artifact_mode = manifest.get("artifact_mode")
    _require(
        artifact_mode in {"scientific", "contract_smoke"},
        "artifact_mode must be 'scientific' or 'contract_smoke'",
    )
    _require(
        manifest.get("scientific_use") is (artifact_mode == "scientific"),
        "scientific_use must agree with artifact_mode",
    )
    schema_version = manifest.get("schema_version")
    if artifact_mode == "scientific":
        _require(
            schema_version == ATLAS_SCHEMA_VERSION,
            f"scientific atlas requires schema version {ATLAS_SCHEMA_VERSION}",
        )
    else:
        _require(
            schema_version in {3, 4},
            f"unsupported historical contract-smoke atlas schema: {schema_version!r}",
        )
    measurement_family = manifest.get("measurement_family")
    measurement_family_sha256 = manifest.get("measurement_family_sha256")
    cell_instruments = manifest.get("cell_instruments")
    cell_instrument_count = manifest.get("cell_instrument_count")
    rollout_collection_sha256 = manifest.get("rollout_collection_sha256")
    rollout_receipts = manifest.get("rollout_receipts")
    rollout_receipt_count = manifest.get("rollout_receipt_count")
    analysis_protocol = manifest.get("analysis_protocol")
    analysis_protocol_sha256 = manifest.get("analysis_protocol_sha256")
    analysis_protocol_path = manifest.get("analysis_protocol_path")
    analysis_protocol_file_sha256 = manifest.get("analysis_protocol_file_sha256")
    analysis_protocol_lock = manifest.get("analysis_protocol_lock")
    analysis_protocol_lock_path = manifest.get("analysis_protocol_lock_path")
    analysis_protocol_lock_file_sha256 = manifest.get("analysis_protocol_lock_file_sha256")
    analysis_protocol_lock_sha256 = manifest.get("analysis_protocol_lock_sha256")
    if artifact_mode == "scientific":
        _require(
            manifest.get("instrument") is None and manifest.get("instrument_sha256") is None,
            "scientific atlas schema v5 forbids one global cell instrument",
        )
        from gear_sonic.research.lace.analysis_protocol import (
            ANALYSIS_PROTOCOL_DIGEST_FIELD,
            validate_analysis_protocol,
            validate_instrument_against_analysis_protocol,
        )

        _require(isinstance(analysis_protocol, Mapping), "scientific atlas protocol missing")
        validate_analysis_protocol(analysis_protocol)
        _require(
            analysis_protocol_sha256 == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
            "scientific atlas protocol digest drifted",
        )
        protocol_path = Path(str(analysis_protocol_path))
        _require(
            protocol_path.is_absolute()
            and not protocol_path.is_symlink()
            and protocol_path.resolve() == protocol_path
            and protocol_path.is_file(),
            "scientific atlas protocol path is invalid",
        )
        _require(
            isinstance(analysis_protocol_file_sha256, str)
            and len(analysis_protocol_file_sha256) == 64
            and hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            == analysis_protocol_file_sha256,
            "scientific atlas protocol file digest drifted",
        )
        from gear_sonic.research.lace.instrument_runtime import _strict_json_loads

        _require(
            _strict_json_loads(
                protocol_path.read_text(encoding="utf-8"),
                "atlas analysis protocol",
            )
            == dict(analysis_protocol),
            "scientific atlas protocol bytes differ from embedded protocol",
        )
        _require(
            manifest.get("data_origin") == analysis_protocol["data_origin"],
            "scientific atlas data origin drifted from analysis protocol",
        )
        from gear_sonic.research.lace.analysis_protocol_lock import (
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
            validate_analysis_protocol_lock,
        )

        _require(
            isinstance(analysis_protocol_lock, Mapping),
            "scientific atlas protocol preregistration lock missing",
        )
        lock_path = Path(str(analysis_protocol_lock_path))
        _require(
            lock_path.is_absolute()
            and not lock_path.is_symlink()
            and lock_path.resolve() == lock_path
            and lock_path.is_file(),
            "scientific atlas protocol preregistration lock path is invalid",
        )
        loaded_lock = _strict_json_loads(
            lock_path.read_text(encoding="utf-8"),
            "atlas analysis protocol lock",
        )
        _require(isinstance(loaded_lock, Mapping), "scientific atlas protocol lock is invalid")
        validate_analysis_protocol_lock(
            loaded_lock,
            repo_root=Path.cwd(),
            verify_files=False,
        )
        _require(
            loaded_lock == dict(analysis_protocol_lock)
            and str(lock_path) == analysis_protocol_lock_path
            and hashlib.sha256(lock_path.read_bytes()).hexdigest()
            == analysis_protocol_lock_file_sha256
            and loaded_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD] == analysis_protocol_lock_sha256
            and loaded_lock["analysis_protocol"]
            == {
                "path": str(protocol_path),
                "file_sha256": analysis_protocol_file_sha256,
                ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
            },
            "scientific atlas protocol preregistration lock drifted",
        )
        _require(
            isinstance(measurement_family, Mapping),
            "scientific atlas measurement_family missing",
        )
        _require(
            measurement_family.get("kind") == "lace_scientific_measurement_family"
            and measurement_family.get("schema_version") == 1
            and measurement_family.get("termination_multi_hot_available") is True,
            "scientific atlas requires a single-evaluation multi-hot termination trace",
        )
        _require(
            measurement_family_sha256 == canonical_sha256(measurement_family),
            "measurement_family_sha256 does not match the scientific family",
        )
        _require(
            isinstance(cell_instruments, Mapping) and bool(cell_instruments),
            "scientific atlas cell_instruments missing",
        )
        _require(
            cell_instrument_count == len(cell_instruments),
            "cell_instrument_count does not match cell_instruments",
        )
        _require(
            isinstance(rollout_collection_sha256, str) and len(rollout_collection_sha256) == 64,
            "scientific atlas rollout_collection_sha256 missing",
        )
        _require(
            isinstance(rollout_receipts, list) and rollout_receipts,
            "scientific atlas rollout receipts missing",
        )
        _require(
            rollout_receipt_count == len(rollout_receipts) == cell_instrument_count,
            "scientific atlas receipt count does not match cell instruments",
        )
        receipt_digests: set[str] = set()
        receipt_paths: set[str] = set()
        for index, receipt in enumerate(rollout_receipts):
            _require(
                isinstance(receipt, Mapping)
                and set(receipt)
                == {
                    "cell",
                    "receipt_path",
                    "receipt_file_sha256",
                    "rollout_binding_sha256",
                    "episode_instrument_sha256",
                    "rollout_count",
                    "rollout_ids_sha256",
                },
                f"rollout_receipts[{index}] fields are invalid",
            )
            path = receipt.get("receipt_path")
            _require(
                isinstance(path, str) and path.startswith("/"),
                f"rollout_receipts[{index}].receipt_path is invalid",
            )
            _require(path not in receipt_paths, "scientific atlas receipt paths are duplicated")
            receipt_paths.add(path)
            for field in (
                "receipt_file_sha256",
                "rollout_binding_sha256",
                "episode_instrument_sha256",
                "rollout_ids_sha256",
            ):
                digest = receipt.get(field)
                _require(
                    isinstance(digest, str) and len(digest) == 64,
                    f"rollout_receipts[{index}].{field} is invalid",
                )
            binding_digest = str(receipt["rollout_binding_sha256"])
            _require(
                binding_digest not in receipt_digests,
                "scientific atlas receipt digests are duplicated",
            )
            receipt_digests.add(binding_digest)
            _require(
                isinstance(receipt.get("rollout_count"), int)
                and not isinstance(receipt.get("rollout_count"), bool)
                and receipt["rollout_count"] > 0,
                f"rollout_receipts[{index}].rollout_count is invalid",
            )
        for digest, cell_instrument in cell_instruments.items():
            _require(
                isinstance(digest, str) and len(digest) == 64,
                "cell instrument key must be a SHA-256",
            )
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError("cell instrument key must be hexadecimal") from error
            _require(
                isinstance(cell_instrument, Mapping)
                and digest == canonical_sha256(cell_instrument),
                "cell instrument key does not bind its exact manifest",
            )
        from gear_sonic.research.lace.instrument import (
            validate_measurement_family,
            validate_scientific_instrument,
        )

        for cell_instrument in cell_instruments.values():
            validate_scientific_instrument(cell_instrument)
            validate_instrument_against_analysis_protocol(
                cell_instrument,
                analysis_protocol,
            )
        validate_measurement_family(
            measurement_family,
            instruments=list(cell_instruments.values()),
        )
        _require(
            {receipt["episode_instrument_sha256"] for receipt in rollout_receipts}
            == set(cell_instruments),
            "scientific atlas receipts do not map one-to-one to cell instruments",
        )
        cell_fields = {
            "probe_policy_id",
            "checkpoint_sha256",
            "domain_randomization_seed",
            "runtime_rng_seed",
            "phase_id",
            "target_fraction",
            "repeat_index",
        }
        for index, receipt in enumerate(rollout_receipts):
            cell = receipt["cell"]
            _require(
                isinstance(cell, Mapping) and set(cell) == cell_fields,
                f"rollout_receipts[{index}].cell is invalid",
            )
            instrument = cell_instruments[receipt["episode_instrument_sha256"]]
            config = instrument["resolved_hydra_config"]
            assignments = (
                config.get("manager_env", {})
                .get("commands", {})
                .get("motion", {})
                .get("atlas_probe_assignments")
            )
            _require(
                isinstance(assignments, list)
                and bool(assignments)
                and all(isinstance(row, Mapping) for row in assignments),
                f"rollout_receipts[{index}] instrument assignments are invalid",
            )
            _require(
                all(
                    all(row.get(field) == cell[field] for field in cell_fields)
                    for row in assignments
                ),
                f"rollout_receipts[{index}] cell is mapped to the wrong instrument",
            )
            rollout_ids = [row.get("rollout_id") for row in assignments]
            _require(
                len(rollout_ids) == receipt["rollout_count"]
                and canonical_sha256({"rollout_ids": rollout_ids}) == receipt["rollout_ids_sha256"],
                f"rollout_receipts[{index}] rollout identity drifted",
            )
        _require(
            sum(receipt["rollout_count"] for receipt in rollout_receipts)
            == manifest.get("rollout_count"),
            "scientific atlas receipt rollout counts do not match rollout_count",
        )
    else:
        _require(
            measurement_family is None
            and measurement_family_sha256 is None
            and cell_instruments in (None, {})
            and cell_instrument_count in (None, 0)
            and rollout_collection_sha256 is None
            and rollout_receipts in (None, [])
            and rollout_receipt_count in (None, 0),
            "contract-smoke atlas cannot claim scientific cell instruments",
        )
        _require(
            analysis_protocol is None
            and analysis_protocol_sha256 is None
            and analysis_protocol_path is None
            and analysis_protocol_file_sha256 is None
            and analysis_protocol_lock is None
            and analysis_protocol_lock_path is None
            and analysis_protocol_lock_file_sha256 is None
            and analysis_protocol_lock_sha256 is None,
            "contract-smoke atlas cannot claim a scientific analysis protocol",
        )
    split_sha256 = manifest.get("split_sha256")
    _require(isinstance(split_sha256, str) and len(split_sha256) == 64, "split_sha256 missing")
    split_selection_sha256 = manifest.get("split_selection_sha256")
    _require(
        isinstance(split_selection_sha256, str) and len(split_selection_sha256) == 64,
        "split_selection_sha256 missing",
    )
    mechanism_names = manifest.get("mechanism_names")
    _require(
        isinstance(mechanism_names, list) and mechanism_names,
        "mechanism_names must be a non-empty list",
    )
    _require(
        all(isinstance(name, str) and name for name in mechanism_names),
        "mechanism_names must contain non-empty strings",
    )
    _require(len(set(mechanism_names)) == len(mechanism_names), "mechanism_names must be unique")
    if artifact_mode == "scientific":
        _require(
            mechanism_names == list(DEFAULT_MECHANISMS)
            and analysis_protocol["mechanism_names"] == mechanism_names,
            "scientific mechanism order drifted from analysis protocol",
        )

    selected_motion_keys = manifest.get("selected_motion_keys")
    _require(
        isinstance(selected_motion_keys, list) and selected_motion_keys,
        "selected_motion_keys must be a non-empty list",
    )
    _require(
        all(isinstance(key, str) and key for key in selected_motion_keys)
        and len(selected_motion_keys) == len(set(selected_motion_keys)),
        "selected_motion_keys must contain unique non-empty strings",
    )
    _require(
        int(manifest.get("motion_count", -1)) == len(selected_motion_keys),
        "motion_count must match selected_motion_keys",
    )
    _require(
        manifest.get("selection_complete_for_d_atlas") is (artifact_mode == "scientific"),
        "selection completeness must agree with artifact_mode",
    )
    rollout_schedule = manifest.get("rollout_schedule")
    rollout_schedule_sha256 = manifest.get("rollout_schedule_sha256")
    rollout_schedule_summary = manifest.get("rollout_schedule_summary")
    if artifact_mode == "scientific":
        _require(
            rollout_schedule is None,
            "scientific atlas cannot embed a self-attested legacy rollout_schedule",
        )
        _require(
            isinstance(rollout_schedule_sha256, str) and len(rollout_schedule_sha256) == 64,
            "scientific atlas rollout_schedule_sha256 missing",
        )
        _require(
            isinstance(rollout_schedule_summary, Mapping),
            "scientific atlas rollout_schedule_summary missing",
        )
        _require(
            rollout_schedule_summary.get("schedule_sha256") == rollout_schedule_sha256,
            "rollout schedule summary digest mismatch",
        )
        _require(
            rollout_schedule_summary.get("artifact_mode") == "scientific"
            and rollout_schedule_summary.get("scientific_use") is True
            and rollout_schedule_summary.get("partition") == "D_atlas",
            "rollout schedule summary is not scientific D_atlas",
        )
        _require(
            rollout_schedule_summary.get("selected_motion_keys") == selected_motion_keys,
            "rollout schedule summary selection mismatch",
        )
        _require(
            measurement_family.get("schedule_sha256") == rollout_schedule_sha256,
            "scientific measurement family and rollout schedule digests differ",
        )
    else:
        _require(
            isinstance(rollout_schedule, list) and rollout_schedule,
            "contract-smoke rollout_schedule must be a non-empty list",
        )
        _require(
            rollout_schedule_sha256 is None and rollout_schedule_summary is None,
            "contract-smoke atlas cannot claim a frozen scientific schedule",
        )
    signature_config = manifest.get("signature_config")
    _require(isinstance(signature_config, Mapping), "signature_config must be a mapping")
    if artifact_mode == "scientific":
        _require(
            signature_config == analysis_protocol["signature_config"],
            "scientific signature config drifted from analysis protocol",
        )
    minimum_resolved = signature_config.get("minimum_resolved_failures")
    _require(
        isinstance(minimum_resolved, int)
        and not isinstance(minimum_resolved, bool)
        and minimum_resolved >= (3 if artifact_mode == "scientific" else 1),
        "signature_config.minimum_resolved_failures is invalid",
    )

    normalizer = manifest.get("normalizer")
    _require(isinstance(normalizer, Mapping), "normalizer must be a mapping")
    _require(normalizer.get("frozen") is True, "normalizer.frozen must be true")
    _require(normalizer.get("fit_partition") == "D_atlas", "normalizer must be fit on D_atlas")
    scales = normalizer.get("mechanism_scales")
    _require(isinstance(scales, Mapping), "normalizer.mechanism_scales must be a mapping")
    _require(set(scales) == set(mechanism_names), "normalizer scales must match mechanisms")
    if artifact_mode == "scientific":
        _require(
            normalizer.get("method") == SCIENTIFIC_NORMALIZER_METHOD,
            "scientific normalizer method is invalid",
        )
        _require(
            normalizer.get("mechanism_names") == mechanism_names,
            "scientific normalizer channel order is invalid",
        )
        input_digest = normalizer.get("input_sha256")
        _require(
            isinstance(input_digest, str) and len(input_digest) == 64,
            "scientific normalizer input_sha256 missing",
        )
        counts = normalizer.get("positive_observation_counts")
        minimums = normalizer.get("minimum_positive_observations")
        _require(
            isinstance(counts, Mapping)
            and isinstance(minimums, Mapping)
            and set(counts) == set(mechanism_names)
            and set(minimums) == set(mechanism_names),
            "scientific normalizer evidence maps are invalid",
        )
        for name in mechanism_names:
            _require(
                int(minimums[name]) >= SCIENTIFIC_NORMALIZER_MINIMUM_POSITIVE_OBSERVATIONS
                and int(counts[name]) >= int(minimums[name]),
                f"scientific normalizer evidence for {name!r} is insufficient",
            )
        normalizer_protocol = analysis_protocol["normalizer_config"]
        _require(
            normalizer.get("method") == normalizer_protocol["method"]
            and normalizer.get("quantile") == normalizer_protocol["quantile"]
            and normalizer.get("quantile_method") == normalizer_protocol["quantile_method"]
            and normalizer.get("fit_partition") == normalizer_protocol["fit_partition"]
            and set(minimums.values()) == {normalizer_protocol["minimum_positive_observations"]},
            "scientific normalizer settings drifted from analysis protocol",
        )
    else:
        _require(
            normalizer.get("kind") == "identity_contract_smoke",
            "contract-smoke atlas requires an identity_contract_smoke normalizer",
        )

    probe_policies = manifest.get("probe_policies")
    _require(
        isinstance(probe_policies, list) and probe_policies, "probe_policies must be non-empty"
    )
    policy_ids: set[str] = set()
    for index, policy in enumerate(probe_policies):
        _require(isinstance(policy, Mapping), f"probe_policies[{index}] must be a mapping")
        policy_id = policy.get("id")
        checkpoint_sha256 = policy.get("checkpoint_sha256")
        _require(isinstance(policy_id, str) and policy_id, f"probe_policies[{index}].id missing")
        _require(policy_id not in policy_ids, f"duplicate probe policy id: {policy_id}")
        policy_ids.add(policy_id)
        _require(
            isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64,
            f"probe_policies[{index}].checkpoint_sha256 missing",
        )

    dr_seeds = manifest.get("domain_randomization_seeds")
    _require(
        isinstance(dr_seeds, list) and dr_seeds, "domain_randomization_seeds must be non-empty"
    )
    _require(len(dr_seeds) == len(set(dr_seeds)), "domain_randomization_seeds must be unique")
    if artifact_mode == "scientific":
        _require(
            rollout_schedule_summary.get("probe_policies") == probe_policies,
            "rollout schedule summary probe-policy axis mismatch",
        )
        _require(
            rollout_schedule_summary.get("domain_randomization_seeds") == dr_seeds,
            "rollout schedule summary DR-condition axis mismatch",
        )
        _require(
            int(rollout_schedule_summary.get("rollout_count", -1))
            == int(manifest.get("rollout_count", -2)),
            "rollout schedule summary count mismatch",
        )

    signatures = manifest.get("signatures")
    _require(isinstance(signatures, list) and signatures, "signatures must be non-empty")
    seen_pairs: set[tuple[str, str]] = set()
    for index, signature in enumerate(signatures):
        _require(isinstance(signature, Mapping), f"signatures[{index}] must be a mapping")
        pair = _validate_signature(signature, len(mechanism_names), index=index)
        _require(pair[1] in policy_ids, f"signature references undeclared policy: {pair[1]}")
        _require(pair not in seen_pairs, f"duplicate motion/policy signature: {pair}")
        seen_pairs.add(pair)
        if artifact_mode == "contract_smoke":
            expected_signature_rollouts = len(rollout_schedule)
        else:
            pair_count = len(selected_motion_keys) * len(policy_ids)
            schedule_rollout_count = int(rollout_schedule_summary.get("rollout_count", -1))
            _require(
                pair_count > 0 and schedule_rollout_count % pair_count == 0,
                "scientific rollout schedule count is not divisible by motion/policy pairs",
            )
            expected_signature_rollouts = schedule_rollout_count // pair_count
        _require(
            int(signature.get("num_rollouts", -1)) == expected_signature_rollouts,
            f"signature {pair} does not match rollout schedule cardinality",
        )

    expected_pairs = {
        (motion_key, policy_id) for motion_key in selected_motion_keys for policy_id in policy_ids
    }
    _require(
        seen_pairs == expected_pairs,
        "signatures must exactly cover selected motions crossed with probe policies",
    )
    expected_rollout_count = (
        len(expected_pairs) * len(rollout_schedule)
        if artifact_mode == "contract_smoke"
        else int(rollout_schedule_summary["rollout_count"])
    )
    _require(
        int(manifest.get("rollout_count", -1)) == expected_rollout_count,
        "rollout_count does not match selected motions, policies, and schedule",
    )

    if verify_digest:
        _verify_self_digest(manifest, "atlas_sha256")
