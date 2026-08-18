from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.research.lace.probes import (
    MECHANISM_NAMES,
    ProbeThresholds,
    _bounded_component_mean,
    _contact_onset_offsets,
    compute_episode_probe,
)
from gear_sonic.research.lace.signatures import (
    DEFAULT_MECHANISMS,
    build_factorized_signatures,
)


def _nominal_inputs(time_steps: int = 20) -> dict[str, object]:
    contacts = np.zeros((time_steps, 2), dtype=bool)
    contacts[5:12] = True
    return {
        "reference_contacts": contacts,
        "actual_contacts": contacts.copy(),
        "foot_tangential_speed": np.zeros((time_steps, 2)),
        "base_translation_error": np.zeros((time_steps, 3)),
        "base_orientation_error": np.zeros(time_steps),
        "base_tilt": np.zeros(time_steps),
        "requested_torque": np.zeros((time_steps, 4)),
        "applied_torque": np.zeros((time_steps, 4)),
        "effort_limits": np.full(4, 100.0),
        "reference_joint_position": np.zeros((time_steps, 4)),
        "joint_position": np.zeros((time_steps, 4)),
        "joint_soft_lower_limits": np.full(4, -1.0),
        "joint_soft_upper_limits": np.full(4, 1.0),
        "local_pose_error": np.zeros((time_steps, 6)),
        "fall_mask": np.zeros(time_steps, dtype=bool),
        "failure_mask": np.zeros(time_steps, dtype=bool),
        "timestep_seconds": 0.02,
    }


def test_nominal_episode_preserves_six_channel_contract() -> None:
    result = compute_episode_probe(**_nominal_inputs())

    assert result.mechanism_names == DEFAULT_MECHANISMS == MECHANISM_NAMES
    assert tuple(result.mechanism_scores) == DEFAULT_MECHANISMS
    assert tuple(result.onset_times_seconds) == DEFAULT_MECHANISMS
    assert tuple(result.diagnostics) == DEFAULT_MECHANISMS
    assert all(score == pytest.approx(0.0) for score in result.mechanism_scores.values())
    assert all(onset is None for onset in result.onset_times_seconds.values())
    assert result.episode_diagnostics["failed"] is False
    assert result.episode_diagnostics["observed_buffer_fraction_to_failure"] is None
    assert result.episode_diagnostics["failure_time_censored"] is True


def test_each_physical_failure_channel_produces_bounded_nonzero_severity() -> None:
    values = _nominal_inputs()
    actual_contacts = np.asarray(values["actual_contacts"]).copy()
    actual_contacts[5:8, 0] = False
    values["actual_contacts"] = actual_contacts
    speed = np.asarray(values["foot_tangential_speed"]).copy()
    speed[8:12, 1] = 0.8
    values["foot_tangential_speed"] = speed
    translation = np.asarray(values["base_translation_error"]).copy()
    translation[:, 0] = np.linspace(0.0, 1.0, translation.shape[0])
    values["base_translation_error"] = translation
    orientation = np.asarray(values["base_orientation_error"]).copy()
    orientation[10:] = 0.9
    values["base_orientation_error"] = orientation
    requested_torque = np.asarray(values["requested_torque"]).copy()
    requested_torque[8:15, :2] = 130.0
    values["requested_torque"] = requested_torque
    applied_torque = np.asarray(values["applied_torque"]).copy()
    applied_torque[8:15, :2] = 100.0
    values["applied_torque"] = applied_torque
    position = np.asarray(values["joint_position"]).copy()
    position[10:, 0] = 0.99
    values["joint_position"] = position
    pose = np.asarray(values["local_pose_error"]).copy()
    pose[10:] = 0.7
    values["local_pose_error"] = pose

    result = compute_episode_probe(**values)

    assert set(result.mechanism_scores) == set(DEFAULT_MECHANISMS)
    assert all(0.0 < score <= 1.0 for score in result.mechanism_scores.values())
    assert all(onset is not None for onset in result.onset_times_seconds.values())


def test_delayed_contact_exports_mismatch_and_onset_offset() -> None:
    values = _nominal_inputs()
    actual = np.zeros((20, 2), dtype=bool)
    actual[8:15] = True
    values["actual_contacts"] = actual

    result = compute_episode_probe(**values)
    diagnostics = result.diagnostics["contact_timing"]

    assert result.mechanism_scores["contact_timing"] > 0.0
    assert result.onset_times_seconds["contact_timing"] == pytest.approx(0.10)
    assert diagnostics["reference_onset_count"] == 2
    assert diagnostics["actual_onset_count"] == 2
    assert diagnostics["mean_absolute_onset_offset_seconds"] == pytest.approx(0.06)


def test_contact_onsets_are_matched_one_to_one_without_reuse() -> None:
    reference = np.zeros((12, 1), dtype=bool)
    actual = np.zeros((12, 1), dtype=bool)
    reference[2:4] = True
    reference[6:8] = True
    actual[2:8] = True

    offsets, reference_count, actual_count, matched_count, unmatched_count = _contact_onset_offsets(
        reference, actual, 0.02, 0.20
    )

    assert reference_count == 2
    assert actual_count == 1
    assert matched_count == 1
    assert unmatched_count == 1
    assert sorted(offsets) == pytest.approx([0.0, 0.20])


def test_component_composition_does_not_reward_duplicate_evidence() -> None:
    assert _bounded_component_mean(0.4) == pytest.approx(0.4)
    assert _bounded_component_mean(0.4, 0.4, 0.4, 0.4) == pytest.approx(0.4)


def test_transient_diagnostics_stop_pre_failure_slope_at_first_failure() -> None:
    values = _nominal_inputs(time_steps=10)
    translation = np.asarray(values["base_translation_error"]).copy()
    translation[:, 0] = np.arange(10) * 0.02
    values["base_translation_error"] = translation
    failure = np.zeros(10, dtype=bool)
    failure[5] = True
    values["failure_mask"] = failure

    result = compute_episode_probe(**values)
    diagnostics = result.diagnostics["base_drift"]

    assert result.onset_times_seconds["base_drift"] == pytest.approx(0.06)
    assert diagnostics["pre_failure_slope_per_second"] == pytest.approx(1.0)
    assert result.episode_diagnostics["first_failure_time_seconds"] == pytest.approx(0.10)
    assert result.episode_diagnostics["observed_buffer_fraction_to_failure"] == pytest.approx(5 / 9)
    assert result.episode_diagnostics["failure_time_censored"] is False


def test_extreme_finite_inputs_remain_bounded() -> None:
    values = _nominal_inputs()
    values["actual_contacts"] = ~np.asarray(values["reference_contacts"])
    values["foot_tangential_speed"] = np.full((20, 2), 1e12)
    values["base_translation_error"] = np.full((20, 3), 1e12)
    values["base_orientation_error"] = np.full(20, 1e12)
    values["base_tilt"] = np.full(20, 1e12)
    values["requested_torque"] = np.full((20, 4), 1e12)
    values["applied_torque"] = np.full((20, 4), 100.0)
    values["joint_position"] = np.full((20, 4), 1e12)
    values["local_pose_error"] = np.full((20, 6), 1e12)
    failure = np.zeros(20, dtype=bool)
    failure[-1] = True
    values["failure_mask"] = failure
    values["fall_mask"] = failure.copy()

    result = compute_episode_probe(**values)

    assert all(np.isfinite(score) for score in result.mechanism_scores.values())
    assert all(0.0 <= score <= 1.0 for score in result.mechanism_scores.values())


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("foot_tangential_speed", np.full((20, 2), np.nan), "finite"),
        ("actual_contacts", np.zeros((19, 2), dtype=bool), "shape mismatch"),
        ("base_translation_error", np.zeros((20, 2)), r"shape \[20, 3\]"),
        ("effort_limits", np.zeros(4), "strictly positive"),
        ("failure_mask", np.zeros(19, dtype=bool), "time dimension mismatch"),
    ),
)
def test_missing_or_invalid_simulator_data_fails_loudly(
    field: str,
    bad_value: object,
    message: str,
) -> None:
    values = _nominal_inputs()
    values[field] = bad_value

    with pytest.raises(ValueError, match=message):
        compute_episode_probe(**values)


def test_fall_must_be_an_explicit_failure_term() -> None:
    values = _nominal_inputs()
    fall = np.zeros(20, dtype=bool)
    fall[-1] = True
    values["fall_mask"] = fall

    with pytest.raises(ValueError, match="fall_mask"):
        compute_episode_probe(**values)


def test_threshold_contract_rejects_degenerate_scales() -> None:
    with pytest.raises(ValueError, match="slip_speed"):
        ProbeThresholds(slip_speed_threshold=0.5, slip_speed_saturation=0.5)
    with pytest.raises(ValueError, match="joint_limit_margin_fraction"):
        ProbeThresholds(joint_limit_margin_fraction=0.5)
    with pytest.raises(ValueError, match="score_window_seconds"):
        ProbeThresholds(score_window_seconds=0.0)


def test_primary_scores_use_fixed_window_and_export_absolute_onset_time() -> None:
    values = _nominal_inputs(time_steps=200)
    values["timestep_seconds"] = 0.02
    translation = np.asarray(values["base_translation_error"]).copy()
    translation[:100, 0] = np.linspace(0.0, 1.0, 100)
    values["base_translation_error"] = translation
    actual = np.asarray(values["actual_contacts"]).copy()
    actual[150:155, 0] = ~actual[150:155, 0]
    values["actual_contacts"] = actual
    failure = np.zeros(200, dtype=bool)
    failure[-1] = True
    values["failure_mask"] = failure

    result = compute_episode_probe(
        **values,
        thresholds=ProbeThresholds(score_window_seconds=2.0),
    )

    assert result.episode_diagnostics["score_window_start_index"] == 100
    assert result.episode_diagnostics["score_window_num_samples"] == 100
    assert result.mechanism_scores["base_drift"] == pytest.approx(0.0)
    assert result.onset_times_seconds["contact_timing"] == pytest.approx(3.0)
    assert result.diagnostics["contact_timing"][
        "onset_time_within_score_window_seconds"
    ] == pytest.approx(1.0)


def test_joint_constraint_subtracts_reference_limit_proximity() -> None:
    values = _nominal_inputs()
    position = np.asarray(values["joint_position"]).copy()
    reference = np.asarray(values["reference_joint_position"]).copy()
    position[:, 0] = 0.99
    reference[:, 0] = 0.99
    values["joint_position"] = position
    values["reference_joint_position"] = reference

    matched = compute_episode_probe(**values)
    assert matched.mechanism_scores["joint_pose_constraint"] == pytest.approx(0.0)
    assert matched.diagnostics["joint_pose_constraint"]["q90_reference_limit_severity"] > 0.0

    reference[:, 0] = 0.0
    values["reference_joint_position"] = reference
    mismatched = compute_episode_probe(**values)
    assert mismatched.mechanism_scores["joint_pose_constraint"] > 0.0


def test_actuation_uses_requested_torque_and_clip_gap() -> None:
    values = _nominal_inputs()
    requested = np.asarray(values["requested_torque"]).copy()
    applied = np.asarray(values["applied_torque"]).copy()
    requested[:, 0] = 130.0
    applied[:, 0] = 100.0
    values["requested_torque"] = requested
    values["applied_torque"] = applied

    result = compute_episode_probe(**values)

    assert result.mechanism_scores["actuation_saturation"] > 0.0
    diagnostics = result.diagnostics["actuation_saturation"]
    assert diagnostics["max_requested_torque_ratio"] == pytest.approx(1.3)
    assert diagnostics["max_applied_torque_ratio"] == pytest.approx(1.0)
    assert diagnostics["max_clip_gap_ratio"] == pytest.approx(0.3)


def test_probe_output_feeds_factorized_signature_builder_without_renaming() -> None:
    values = _nominal_inputs()
    torque = np.asarray(values["requested_torque"]).copy()
    torque[:, 0] = 100.0
    values["requested_torque"] = torque
    values["applied_torque"] = torque.copy()
    failure = np.zeros(20, dtype=bool)
    failure[-1] = True
    values["failure_mask"] = failure
    probe = compute_episode_probe(**values)

    signatures = build_factorized_signatures(
        [
            probe.as_signature_episode(
                motion_key="walk_001",
                probe_policy_id="lite_early",
            )
        ],
        minimum_resolved_failures=1,
    )

    assert signatures[0]["difficulty"] == 1.0
    assert signatures[0]["q"] is not None
    actuation_index = DEFAULT_MECHANISMS.index("actuation_saturation")
    assert signatures[0]["q"][actuation_index] == pytest.approx(1.0)
