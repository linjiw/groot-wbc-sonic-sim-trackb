"""Simulator-independent episode probes for the LACE failure atlas.

The live Isaac Lab adapter is intentionally kept out of this module.  It must
resolve simulator tensors into the explicit arrays accepted by
``compute_episode_probe``; missing tensors therefore cannot silently become
zero-valued failure channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
MECHANISM_NAMES = DEFAULT_MECHANISMS


@dataclass(frozen=True)
class ProbeThresholds:
    """Physical thresholds used to turn residual streams into bounded scores.

    Values between a threshold and its saturation value are mapped linearly to
    ``[0, 1]``.  Thresholds are configuration, not quantities to tune on test
    motions; atlas code should freeze them alongside its fitted normalizers.
    """

    contact_onset_tolerance_seconds: float = 0.04
    contact_onset_saturation_seconds: float = 0.20
    # This is a foot-link tangential-velocity proxy, not contact-point slip.
    # SONIC already uses 0.15 m/s for a related foot-velocity diagnostic; the
    # atlas must validate this initial calibration before scientific use.
    slip_speed_threshold: float = 0.15
    slip_speed_saturation: float = 0.75
    base_drift_threshold: float = 0.05
    base_drift_saturation: float = 0.50
    orientation_error_threshold: float = 0.15
    orientation_error_saturation: float = 0.80
    base_tilt_threshold: float = 0.35
    base_tilt_saturation: float = 1.00
    torque_ratio_threshold: float = 0.95
    torque_ratio_saturation: float = 1.20
    torque_clip_gap_ratio_threshold: float = 0.01
    torque_clip_gap_ratio_saturation: float = 0.25
    joint_limit_margin_fraction: float = 0.05
    joint_limit_excess_threshold: float = 0.05
    joint_limit_excess_saturation: float = 0.50
    pose_error_threshold: float = 0.10
    pose_error_saturation: float = 0.50
    # Primary severities use a fixed window ending at first failure (or episode
    # end for a censored success) so longer-surviving episodes do not receive a
    # different time-at-risk definition.
    score_window_seconds: float = 2.0

    def __post_init__(self) -> None:
        pairs = (
            (
                "contact_onset",
                self.contact_onset_tolerance_seconds,
                self.contact_onset_saturation_seconds,
            ),
            ("slip_speed", self.slip_speed_threshold, self.slip_speed_saturation),
            ("base_drift", self.base_drift_threshold, self.base_drift_saturation),
            (
                "orientation_error",
                self.orientation_error_threshold,
                self.orientation_error_saturation,
            ),
            ("base_tilt", self.base_tilt_threshold, self.base_tilt_saturation),
            ("torque_ratio", self.torque_ratio_threshold, self.torque_ratio_saturation),
            (
                "torque_clip_gap_ratio",
                self.torque_clip_gap_ratio_threshold,
                self.torque_clip_gap_ratio_saturation,
            ),
            (
                "joint_limit_excess",
                self.joint_limit_excess_threshold,
                self.joint_limit_excess_saturation,
            ),
            ("pose_error", self.pose_error_threshold, self.pose_error_saturation),
        )
        for name, threshold, saturation in pairs:
            if not (
                math.isfinite(threshold)
                and math.isfinite(saturation)
                and 0.0 <= threshold < saturation
            ):
                raise ValueError(
                    f"{name} threshold and saturation must be finite with "
                    "0 <= threshold < saturation"
                )
        margin = self.joint_limit_margin_fraction
        if not math.isfinite(margin) or not 0.0 < margin < 0.5:
            raise ValueError("joint_limit_margin_fraction must be finite and in (0, 0.5)")
        if not math.isfinite(self.score_window_seconds) or self.score_window_seconds <= 0.0:
            raise ValueError("score_window_seconds must be finite and positive")


@dataclass(frozen=True)
class EpisodeProbeResult:
    """Compact, JSON-ready probe output for one episode."""

    mechanism_names: tuple[str, ...]
    mechanism_scores: dict[str, float]
    onset_times_seconds: dict[str, float | None]
    diagnostics: dict[str, dict[str, float | int | None]]
    episode_diagnostics: dict[str, float | int | bool | None]

    def as_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for an atlas rollout artifact."""

        return {
            "mechanism_names": list(self.mechanism_names),
            "mechanism_scores": dict(self.mechanism_scores),
            "onset_times_seconds": dict(self.onset_times_seconds),
            "diagnostics": {name: dict(values) for name, values in self.diagnostics.items()},
            "episode_diagnostics": dict(self.episode_diagnostics),
        }

    def as_signature_episode(
        self,
        *,
        motion_key: str,
        probe_policy_id: str,
    ) -> dict[str, Any]:
        """Return the fields consumed by ``build_factorized_signatures``."""

        if not isinstance(motion_key, str) or not motion_key:
            raise ValueError("motion_key must be a non-empty string")
        if not isinstance(probe_policy_id, str) or not probe_policy_id:
            raise ValueError("probe_policy_id must be a non-empty string")
        return {
            "motion_key": motion_key,
            "probe_policy_id": probe_policy_id,
            "failed": bool(self.episode_diagnostics["failed"]),
            "mechanism_scores": dict(self.mechanism_scores),
        }


def _float_array(values: ArrayLike, name: str, ndim: int) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _bool_array(values: ArrayLike, name: str, ndim: int) -> BoolArray:
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


def _require_time(array: np.ndarray, name: str, time_steps: int) -> None:
    if array.shape[0] != time_steps:
        raise ValueError(
            f"{name} time dimension mismatch: expected {time_steps}, got {array.shape[0]}"
        )


def _bounded_excess(values: ArrayLike | float, threshold: float, saturation: float) -> FloatArray:
    return np.clip(
        (np.asarray(values, dtype=np.float64) - threshold) / (saturation - threshold),
        0.0,
        1.0,
    )


def _bounded_component_mean(*components: float) -> float:
    """Combine normalized evidence without rewarding component multiplicity.

    A probabilistic-union transform grows mechanically with the number of
    components, but LACE mechanisms expose different numbers of diagnostics.
    Their unweighted mean preserves a common ``[0, 1]`` scale: repeating the
    same evidence does not make a mechanism look more severe.  D-atlas channel
    normalization still calibrates the empirical scale between mechanisms.
    """

    if not components:
        return 0.0
    values = np.asarray(components, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("internal severity components must be finite and in [0, 1]")
    return float(values.mean())


def _first_onset_seconds(event_mask: BoolArray, timestep_seconds: float) -> float | None:
    indices = np.flatnonzero(event_mask)
    return float(indices[0] * timestep_seconds) if indices.size else None


def _pre_failure_slope(
    values: FloatArray,
    failure_mask: BoolArray,
    timestep_seconds: float,
) -> float:
    failure_indices = np.flatnonzero(failure_mask)
    end = int(failure_indices[0]) + 1 if failure_indices.size else values.size
    if end < 2:
        return 0.0
    y = values[:end]
    x = np.arange(end, dtype=np.float64) * timestep_seconds
    centered_x = x - float(x.mean())
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered_x, y - float(y.mean())) / denominator)


def _stream_summary(
    values: FloatArray,
    event_mask: BoolArray,
    failure_mask: BoolArray,
    timestep_seconds: float,
) -> dict[str, float | int | None]:
    if values.ndim != 1 or event_mask.shape != values.shape or failure_mask.shape != values.shape:
        raise ValueError("internal stream summary shape mismatch")
    onset = _first_onset_seconds(event_mask, timestep_seconds)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "incidence": float(event_mask.mean()),
        "onset_index": int(round(onset / timestep_seconds)) if onset is not None else None,
        "onset_time_seconds": onset,
        "pre_failure_slope_per_second": _pre_failure_slope(
            values,
            failure_mask,
            timestep_seconds,
        ),
    }


def _rising_edges(contact: BoolArray) -> list[NDArray[np.int64]]:
    # Frame zero is an initial state, not an observed contact transition.
    return [
        np.flatnonzero((~contact[:-1, foot]) & contact[1:, foot]) + 1
        for foot in range(contact.shape[1])
    ]


def _contact_onset_offsets(
    reference_contacts: BoolArray,
    actual_contacts: BoolArray,
    timestep_seconds: float,
    missing_offset_seconds: float,
) -> tuple[list[float], int, int, int, int]:
    """Return optimal ordered one-to-one onset offsets.

    Insertions and deletions receive ``missing_offset_seconds``.  Dynamic
    programming prevents one transition from being reused as the nearest
    neighbor of several transitions, which otherwise understates contact
    chatter and missed contacts.
    """

    reference_edges = _rising_edges(reference_contacts)
    actual_edges = _rising_edges(actual_contacts)
    offsets: list[float] = []
    reference_count = 0
    actual_count = 0
    matched_count = 0
    unmatched_count = 0
    for reference, actual in zip(reference_edges, actual_edges, strict=True):
        reference_count += int(reference.size)
        actual_count += int(actual.size)
        n_reference = int(reference.size)
        n_actual = int(actual.size)
        costs = np.full((n_reference + 1, n_actual + 1), np.inf, dtype=np.float64)
        operations = np.full((n_reference + 1, n_actual + 1), -1, dtype=np.int8)
        costs[0, 0] = 0.0
        for i in range(1, n_reference + 1):
            costs[i, 0] = i * missing_offset_seconds
            operations[i, 0] = 1  # unmatched reference onset
        for j in range(1, n_actual + 1):
            costs[0, j] = j * missing_offset_seconds
            operations[0, j] = 2  # unmatched actual onset
        for i in range(1, n_reference + 1):
            for j in range(1, n_actual + 1):
                match_offset = min(
                    abs(int(reference[i - 1]) - int(actual[j - 1])) * timestep_seconds,
                    missing_offset_seconds,
                )
                candidates = (
                    costs[i - 1, j - 1] + match_offset,
                    costs[i - 1, j] + missing_offset_seconds,
                    costs[i, j - 1] + missing_offset_seconds,
                )
                # np.argmin breaks exact ties toward a match, then a missing
                # reference, for deterministic maximum-cardinality pairing.
                operation = int(np.argmin(candidates))
                costs[i, j] = candidates[operation]
                operations[i, j] = operation

        foot_offsets: list[float] = []
        i, j = n_reference, n_actual
        while i or j:
            operation = int(operations[i, j])
            if operation == 0:
                foot_offsets.append(
                    min(
                        abs(int(reference[i - 1]) - int(actual[j - 1])) * timestep_seconds,
                        missing_offset_seconds,
                    )
                )
                matched_count += 1
                i -= 1
                j -= 1
            elif operation == 1:
                foot_offsets.append(missing_offset_seconds)
                unmatched_count += 1
                i -= 1
            elif operation == 2:
                foot_offsets.append(missing_offset_seconds)
                unmatched_count += 1
                j -= 1
            else:  # pragma: no cover - guards internal DP corruption
                raise RuntimeError("invalid contact-onset matching operation")
        offsets.extend(reversed(foot_offsets))
    return offsets, reference_count, actual_count, matched_count, unmatched_count


def _limit_matrix(values: ArrayLike, name: str, time_steps: int, joint_count: int) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        if result.shape != (joint_count,):
            raise ValueError(
                f"{name} must have shape [{joint_count}] or [{time_steps}, {joint_count}]"
            )
        result = np.broadcast_to(result, (time_steps, joint_count))
    elif result.shape != (time_steps, joint_count):
        raise ValueError(f"{name} must have shape [{joint_count}] or [{time_steps}, {joint_count}]")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(result, dtype=np.float64)


def compute_episode_probe(
    *,
    reference_contacts: ArrayLike,
    actual_contacts: ArrayLike,
    foot_tangential_speed: ArrayLike,
    base_translation_error: ArrayLike,
    base_orientation_error: ArrayLike,
    base_tilt: ArrayLike,
    requested_torque: ArrayLike,
    applied_torque: ArrayLike,
    effort_limits: ArrayLike,
    reference_joint_position: ArrayLike,
    joint_position: ArrayLike,
    joint_soft_lower_limits: ArrayLike,
    joint_soft_upper_limits: ArrayLike,
    local_pose_error: ArrayLike,
    fall_mask: ArrayLike,
    failure_mask: ArrayLike,
    timestep_seconds: float,
    thresholds: ProbeThresholds | None = None,
) -> EpisodeProbeResult:
    """Compute six bounded LACE mechanism severities for one episode.

    Every simulator-derived quantity is a required keyword-only argument.  The
    function validates finite values and exact time/feature agreement before it
    computes anything.  Scores preserve co-occurrence and lie in ``[0, 1]``;
    they are not a forced single-cause classification.

    ``base_translation_error`` has shape ``[T, 3]``. Contact arrays and foot
    speed share ``[T, F]``. Requested/applied torque and reference/actual joint
    arrays use ``[T, J]``; limits may be static ``[J]`` or time-varying
    ``[T, J]``. ``local_pose_error`` may contain one or more signed residual
    components in shape ``[T, P]``.
    """

    config = thresholds if thresholds is not None else ProbeThresholds()
    if not isinstance(config, ProbeThresholds):
        raise TypeError("thresholds must be a ProbeThresholds instance")
    if not math.isfinite(timestep_seconds) or timestep_seconds <= 0.0:
        raise ValueError("timestep_seconds must be finite and positive")

    reference_contact = _bool_array(reference_contacts, "reference_contacts", 2)
    time_steps, foot_count = reference_contact.shape
    actual_contact = _bool_array(actual_contacts, "actual_contacts", 2)
    if actual_contact.shape != (time_steps, foot_count):
        raise ValueError(
            "actual_contacts shape mismatch: expected "
            f"{(time_steps, foot_count)}, got {actual_contact.shape}"
        )
    foot_speed = _float_array(foot_tangential_speed, "foot_tangential_speed", 2)
    if foot_speed.shape != (time_steps, foot_count):
        raise ValueError(
            "foot_tangential_speed shape mismatch: expected "
            f"{(time_steps, foot_count)}, got {foot_speed.shape}"
        )
    if np.any(foot_speed < 0.0):
        raise ValueError("foot_tangential_speed must be nonnegative")

    translation_error = _float_array(base_translation_error, "base_translation_error", 2)
    if translation_error.shape != (time_steps, 3):
        raise ValueError(
            f"base_translation_error must have shape [{time_steps}, 3], "
            f"got {translation_error.shape}"
        )
    orientation_error = _float_array(base_orientation_error, "base_orientation_error", 1)
    tilt = _float_array(base_tilt, "base_tilt", 1)
    for array, name in ((orientation_error, "base_orientation_error"), (tilt, "base_tilt")):
        _require_time(array, name, time_steps)
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be an angular magnitude and therefore nonnegative")

    requested = _float_array(requested_torque, "requested_torque", 2)
    _require_time(requested, "requested_torque", time_steps)
    joint_count = requested.shape[1]
    applied = _float_array(applied_torque, "applied_torque", 2)
    if applied.shape != (time_steps, joint_count):
        raise ValueError(
            f"applied_torque shape mismatch: expected {(time_steps, joint_count)}, "
            f"got {applied.shape}"
        )
    effort_limit = _limit_matrix(effort_limits, "effort_limits", time_steps, joint_count)
    if np.any(effort_limit <= 0.0):
        raise ValueError("effort_limits must be strictly positive")

    reference_position = _float_array(
        reference_joint_position,
        "reference_joint_position",
        2,
    )
    if reference_position.shape != (time_steps, joint_count):
        raise ValueError(
            "reference_joint_position shape mismatch: expected "
            f"{(time_steps, joint_count)}, got {reference_position.shape}"
        )
    position = _float_array(joint_position, "joint_position", 2)
    if position.shape != (time_steps, joint_count):
        raise ValueError(
            f"joint_position shape mismatch: expected {(time_steps, joint_count)}, "
            f"got {position.shape}"
        )
    lower_limit = _limit_matrix(
        joint_soft_lower_limits,
        "joint_soft_lower_limits",
        time_steps,
        joint_count,
    )
    upper_limit = _limit_matrix(
        joint_soft_upper_limits,
        "joint_soft_upper_limits",
        time_steps,
        joint_count,
    )
    if np.any(upper_limit <= lower_limit):
        raise ValueError("joint soft upper limits must be strictly greater than lower limits")

    pose_error = _float_array(local_pose_error, "local_pose_error", 2)
    _require_time(pose_error, "local_pose_error", time_steps)
    falls = _bool_array(fall_mask, "fall_mask", 1)
    failures = _bool_array(failure_mask, "failure_mask", 1)
    _require_time(falls, "fall_mask", time_steps)
    _require_time(failures, "failure_mask", time_steps)
    if np.any(falls & ~failures):
        raise ValueError("fall_mask may only be true where failure_mask is true")

    failure_indices = np.flatnonzero(failures)
    first_failure_index = int(failure_indices[0]) if failure_indices.size else None
    score_window_end = first_failure_index + 1 if first_failure_index is not None else time_steps
    maximum_window_samples = max(
        1,
        int(math.floor(config.score_window_seconds / timestep_seconds + 1e-12)),
    )
    score_window_start = max(0, score_window_end - maximum_window_samples)
    score_window = slice(score_window_start, score_window_end)
    for_window = (
        reference_contact,
        actual_contact,
        foot_speed,
        translation_error,
        orientation_error,
        tilt,
        requested,
        applied,
        effort_limit,
        reference_position,
        position,
        lower_limit,
        upper_limit,
        pose_error,
        falls,
        failures,
    )
    (
        reference_contact,
        actual_contact,
        foot_speed,
        translation_error,
        orientation_error,
        tilt,
        requested,
        applied,
        effort_limit,
        reference_position,
        position,
        lower_limit,
        upper_limit,
        pose_error,
        falls,
        failures,
    ) = tuple(array[score_window] for array in for_window)

    # Contact timing: time in a mismatched state plus symmetric onset offsets.
    contact_mismatch = reference_contact ^ actual_contact
    mismatch_trace = contact_mismatch.mean(axis=1, dtype=np.float64)
    mismatch_event = np.any(contact_mismatch, axis=1)
    onset_offsets, reference_onsets, actual_onsets, matched_onsets, unmatched_onsets = (
        _contact_onset_offsets(
            reference_contact,
            actual_contact,
            timestep_seconds,
            config.contact_onset_saturation_seconds,
        )
    )
    if onset_offsets:
        normalized_offsets = _bounded_excess(
            onset_offsets,
            config.contact_onset_tolerance_seconds,
            config.contact_onset_saturation_seconds,
        )
        onset_component = float(np.mean(normalized_offsets))
        mean_onset_offset = float(np.mean(onset_offsets))
        max_onset_offset = float(np.max(onset_offsets))
    else:
        onset_component = 0.0
        mean_onset_offset = 0.0
        max_onset_offset = 0.0
    contact_score = _bounded_component_mean(float(contact_mismatch.mean()), onset_component)
    contact_diagnostics = _stream_summary(
        mismatch_trace,
        mismatch_event,
        failures,
        timestep_seconds,
    )
    contact_diagnostics.update(
        {
            "mismatch_sample_fraction": float(contact_mismatch.mean()),
            # The ordered onset offsets are intentionally not retained as a
            # raw per-transition trace.  Persist the exact normalized
            # component consumed by the score so downstream scientific
            # validation can reconstruct the score without trusting either
            # of its duplicated aliases.
            "normalized_onset_offset_component": onset_component,
            "reference_onset_count": reference_onsets,
            "actual_onset_count": actual_onsets,
            "matched_onset_pair_count": matched_onsets,
            "unmatched_onset_count": unmatched_onsets,
            "onset_matching": "ordered_one_to_one_minimum_cost_v1",
            "mean_absolute_onset_offset_seconds": mean_onset_offset,
            "max_absolute_onset_offset_seconds": max_onset_offset,
        }
    )

    # Foot slip: duration above the speed threshold and the q90 speed in contact.
    active_speed = np.where(actual_contact, foot_speed, 0.0)
    slip_event_by_foot = actual_contact & (foot_speed > config.slip_speed_threshold)
    slip_event = np.any(slip_event_by_foot, axis=1)
    slip_trace = active_speed.max(axis=1)
    contact_samples = int(actual_contact.sum())
    slip_fraction = float(slip_event_by_foot.sum()) / contact_samples if contact_samples else 0.0
    contact_speeds = foot_speed[actual_contact]
    slip_q90 = float(np.quantile(contact_speeds, 0.90)) if contact_speeds.size else 0.0
    slip_speed_component = float(
        _bounded_excess(
            slip_q90,
            config.slip_speed_threshold,
            config.slip_speed_saturation,
        )
    )
    slip_score = _bounded_component_mean(slip_fraction, slip_speed_component)
    slip_diagnostics = _stream_summary(slip_trace, slip_event, failures, timestep_seconds)
    slip_diagnostics.update(
        {
            "contact_sample_count": contact_samples,
            "slip_contact_sample_fraction": slip_fraction,
            "q90_contact_speed": slip_q90,
            "slip_duration_seconds": float(slip_event.sum()) * timestep_seconds,
        }
    )

    # Base drift: displacement of the residual vector away from the reset state.
    # The absolute reset residual is retained as a diagnostic but must not make
    # reset noise look like policy-induced drift.
    absolute_drift = np.linalg.norm(translation_error, axis=1)
    drift = np.linalg.norm(translation_error - translation_error[0], axis=1)
    drift_event = drift > config.base_drift_threshold
    drift_max = float(drift.max())
    drift_score = float(
        _bounded_excess(
            drift_max,
            config.base_drift_threshold,
            config.base_drift_saturation,
        )
    )
    drift_diagnostics = _stream_summary(drift, drift_event, failures, timestep_seconds)
    drift_diagnostics.update(
        {
            "initial_absolute_residual": float(absolute_drift[0]),
            "max_absolute_residual": float(absolute_drift.max()),
            "max_residual_displacement_from_initial": drift_max,
        }
    )

    # Balance/orientation: angular RMS, peak tilt, and explicit fall termination.
    orientation_event = orientation_error > config.orientation_error_threshold
    tilt_event = tilt > config.base_tilt_threshold
    balance_event = orientation_event | tilt_event | falls
    orientation_rms = float(np.sqrt(np.mean(np.square(orientation_error))))
    balance_trace = np.maximum(
        _bounded_excess(
            orientation_error,
            config.orientation_error_threshold,
            config.orientation_error_saturation,
        ),
        _bounded_excess(
            tilt,
            config.base_tilt_threshold,
            config.base_tilt_saturation,
        ),
    )
    balance_trace = np.maximum(balance_trace, falls.astype(np.float64))
    balance_score = _bounded_component_mean(
        float(
            _bounded_excess(
                orientation_rms,
                config.orientation_error_threshold,
                config.orientation_error_saturation,
            )
        ),
        float(
            _bounded_excess(
                float(tilt.max()),
                config.base_tilt_threshold,
                config.base_tilt_saturation,
            )
        ),
    )
    balance_diagnostics = _stream_summary(
        balance_trace,
        balance_event,
        failures,
        timestep_seconds,
    )
    balance_diagnostics.update(
        {
            "orientation_rms_radians": orientation_rms,
            "orientation_q90_radians": float(np.quantile(orientation_error, 0.90)),
            "max_tilt_radians": float(tilt.max()),
            "fall_incidence": float(np.any(falls)),
        }
    )

    # Implicit actuators expose a requested-PD estimate and its clipped command,
    # not measured PhysX torque. Preserve both so clipping beyond the limit is
    # not hidden by looking only at the applied value.
    requested_ratio = np.abs(requested) / effort_limit
    applied_ratio = np.abs(applied) / effort_limit
    clip_gap_ratio = np.abs(requested - applied) / effort_limit
    clipped = (requested_ratio >= config.torque_ratio_threshold) | (
        clip_gap_ratio > config.torque_clip_gap_ratio_threshold
    )
    clipped_event = np.any(clipped, axis=1)
    torque_q90 = float(np.quantile(requested_ratio, 0.90))
    clip_gap_q90 = float(np.quantile(clip_gap_ratio, 0.90))
    clipped_joint_fraction = float(clipped.mean())
    clipped_duration_fraction = float(clipped_event.mean())
    actuation_score = _bounded_component_mean(
        clipped_joint_fraction,
        clipped_duration_fraction,
        float(
            _bounded_excess(
                torque_q90,
                config.torque_ratio_threshold,
                config.torque_ratio_saturation,
            )
        ),
        float(
            _bounded_excess(
                clip_gap_q90,
                config.torque_clip_gap_ratio_threshold,
                config.torque_clip_gap_ratio_saturation,
            )
        ),
    )
    actuation_diagnostics = _stream_summary(
        requested_ratio.max(axis=1),
        clipped_event,
        failures,
        timestep_seconds,
    )
    actuation_diagnostics.update(
        {
            "clipped_joint_fraction": clipped_joint_fraction,
            "clipped_duration_fraction": clipped_duration_fraction,
            "q90_requested_torque_ratio": torque_q90,
            "max_requested_torque_ratio": float(requested_ratio.max()),
            "max_applied_torque_ratio": float(applied_ratio.max()),
            "q90_clip_gap_ratio": clip_gap_q90,
            "max_clip_gap_ratio": float(clip_gap_ratio.max()),
            "torque_semantics": "implicit_actuator_requested_pd_proxy",
        }
    )

    # Joint/pose constraint: policy-induced proximity beyond the reference plus
    # local-pose divergence. Absolute reference proximity is exported only as a
    # feasibility covariate; otherwise a reference near a joint limit would be
    # mislabeled as a policy failure mechanism.
    joint_range = upper_limit - lower_limit
    normalized_position = (position - lower_limit) / joint_range
    normalized_reference = (reference_position - lower_limit) / joint_range
    margin = config.joint_limit_margin_fraction
    lower_proximity = np.clip((margin - normalized_position) / margin, 0.0, 1.0)
    upper_proximity = np.clip((normalized_position - (1.0 - margin)) / margin, 0.0, 1.0)
    limit_severity = np.maximum(lower_proximity, upper_proximity)
    reference_lower_proximity = np.clip(
        (margin - normalized_reference) / margin,
        0.0,
        1.0,
    )
    reference_upper_proximity = np.clip(
        (normalized_reference - (1.0 - margin)) / margin,
        0.0,
        1.0,
    )
    reference_limit_severity = np.maximum(
        reference_lower_proximity,
        reference_upper_proximity,
    )
    excess_limit_severity = np.maximum(limit_severity - reference_limit_severity, 0.0)
    excess_near_limit = excess_limit_severity > config.joint_limit_excess_threshold
    excess_near_limit_event = np.any(excess_near_limit, axis=1)
    pose_rms = np.sqrt(np.mean(np.square(pose_error), axis=1))
    pose_event = pose_rms > config.pose_error_threshold
    pose_growth = max(0.0, float(pose_rms.max()) - float(pose_rms[0]))
    pose_component = float(
        _bounded_excess(
            float(np.quantile(pose_rms, 0.90)),
            config.pose_error_threshold,
            config.pose_error_saturation,
        )
    )
    pose_growth_component = float(
        _bounded_excess(
            pose_growth,
            config.pose_error_threshold,
            config.pose_error_saturation,
        )
    )
    excess_near_limit_joint_fraction = float(excess_near_limit.mean())
    excess_near_limit_duration_fraction = float(excess_near_limit_event.mean())
    excess_limit_q90 = float(np.quantile(excess_limit_severity, 0.90))
    joint_pose_score = _bounded_component_mean(
        excess_near_limit_joint_fraction,
        excess_near_limit_duration_fraction,
        float(
            _bounded_excess(
                excess_limit_q90,
                config.joint_limit_excess_threshold,
                config.joint_limit_excess_saturation,
            )
        ),
        pose_component,
        pose_growth_component,
    )
    joint_pose_event = excess_near_limit_event | pose_event
    joint_pose_trace = np.maximum(
        _bounded_excess(
            excess_limit_severity.max(axis=1),
            config.joint_limit_excess_threshold,
            config.joint_limit_excess_saturation,
        ),
        _bounded_excess(
            pose_rms,
            config.pose_error_threshold,
            config.pose_error_saturation,
        ),
    )
    joint_pose_diagnostics = _stream_summary(
        joint_pose_trace,
        joint_pose_event,
        failures,
        timestep_seconds,
    )
    joint_pose_diagnostics.update(
        {
            "excess_near_limit_joint_fraction": excess_near_limit_joint_fraction,
            "excess_near_limit_duration_fraction": excess_near_limit_duration_fraction,
            "q90_excess_limit_severity": excess_limit_q90,
            "q90_actual_limit_severity": float(np.quantile(limit_severity, 0.90)),
            "q90_reference_limit_severity": float(np.quantile(reference_limit_severity, 0.90)),
            "q90_pose_rms": float(np.quantile(pose_rms, 0.90)),
            "max_pose_rms": float(pose_rms.max()),
            "pose_residual_growth": pose_growth,
        }
    )

    mechanism_scores = dict(
        zip(
            MECHANISM_NAMES,
            (
                contact_score,
                slip_score,
                drift_score,
                balance_score,
                actuation_score,
                joint_pose_score,
            ),
            strict=True,
        )
    )
    diagnostics = dict(
        zip(
            MECHANISM_NAMES,
            (
                contact_diagnostics,
                slip_diagnostics,
                drift_diagnostics,
                balance_diagnostics,
                actuation_diagnostics,
                joint_pose_diagnostics,
            ),
            strict=True,
        )
    )
    onset_times_seconds: dict[str, float | None] = {}
    for name in MECHANISM_NAMES:
        relative_index = diagnostics[name]["onset_index"]
        relative_time = diagnostics[name]["onset_time_seconds"]
        diagnostics[name]["onset_index_within_score_window"] = relative_index
        diagnostics[name]["onset_time_within_score_window_seconds"] = relative_time
        absolute_index = (
            score_window_start + int(relative_index) if relative_index is not None else None
        )
        absolute_time = (
            float(absolute_index * timestep_seconds) if absolute_index is not None else None
        )
        diagnostics[name]["onset_index"] = absolute_index
        diagnostics[name]["onset_time_seconds"] = absolute_time
        onset_times_seconds[name] = absolute_time
    first_failure_time = (
        float(first_failure_index * timestep_seconds) if first_failure_index is not None else None
    )
    episode_diagnostics: dict[str, float | int | bool | None] = {
        "num_steps": time_steps,
        "num_feet": foot_count,
        "num_joints": joint_count,
        "timestep_seconds": float(timestep_seconds),
        "episode_duration_seconds": float(time_steps * timestep_seconds),
        "failed": bool(failure_indices.size),
        "first_failure_index": first_failure_index,
        "first_failure_time_seconds": first_failure_time,
        "score_window_rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
        "score_window_requested_seconds": float(config.score_window_seconds),
        "score_window_start_index": score_window_start,
        "score_window_end_index_exclusive": score_window_end,
        "score_window_num_samples": score_window_end - score_window_start,
        "score_window_duration_seconds": float(
            (score_window_end - score_window_start) * timestep_seconds
        ),
        "score_window_left_censored": score_window_start == 0
        and score_window_end < maximum_window_samples,
        # This fraction is only relative to the observed buffer. The live
        # recorder adds absolute reference-clip progress from frozen start/length
        # metadata; do not treat this censoring diagnostic as motion progress.
        "observed_buffer_fraction_to_failure": (
            float(first_failure_index / max(time_steps - 1, 1))
            if first_failure_index is not None
            else None
        ),
        "failure_time_censored": first_failure_index is None,
    }
    return EpisodeProbeResult(
        mechanism_names=MECHANISM_NAMES,
        mechanism_scores=mechanism_scores,
        onset_times_seconds=onset_times_seconds,
        diagnostics=diagnostics,
        episode_diagnostics=episode_diagnostics,
    )
