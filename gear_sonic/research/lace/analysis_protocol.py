"""Pre-outcome, self-hashed analysis protocol for a scientific LACE atlas.

The protocol is deliberately external to rollout and atlas outputs.  It freezes
operator-selectable measurement and aggregation choices before any episode
outcomes exist, then lets collectors prove that every live cell instrument
implements those choices exactly.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Mapping

from gear_sonic.research.lace.instrument import (
    TERMINATION_MULTI_HOT_SEMANTICS,
    _validate_termination_predicates,
    validate_scientific_instrument,
)
from gear_sonic.research.lace.normalizer import (
    DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
    FIT_PARTITION,
    NORMALIZER_METHOD,
    Q90,
    QUANTILE_METHOD,
)
from gear_sonic.research.lace.probes import ProbeThresholds
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import validate_rollout_schedule
from gear_sonic.research.lace.schema import (
    canonical_sha256,
    split_selection_sha256,
    validate_split_manifest,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

ANALYSIS_PROTOCOL_KIND = "lace_frozen_atlas_analysis_protocol"
ANALYSIS_PROTOCOL_SCHEMA_VERSION = 1
ANALYSIS_PROTOCOL_DIGEST_FIELD = "analysis_protocol_sha256"
SCIENTIFIC_ROLLOUT_DATA_ORIGIN = "receipt_committed_isaaclab_lace_atlas_probe_v1"
SCIENTIFIC_SIGNATURE_CONFIG = {
    "difficulty_beta_prior": [0.5, 0.5],
    "minimum_resolved_failures": 3,
    "mechanism_dirichlet_prior": 0.5,
    "credible_interval_level": 0.95,
}
SCIENTIFIC_NORMALIZER_CONFIG = {
    "method": NORMALIZER_METHOD,
    "quantile": Q90,
    "quantile_method": QUANTILE_METHOD,
    "fit_partition": FIT_PARTITION,
    "minimum_positive_observations": DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
    "evidence_rule": "failed_episodes_positive_scores_only_v1",
}
SCIENTIFIC_SCORE_WINDOW_RULE = "fixed_window_ending_at_first_failure_or_censored_end_v1"
SCIENTIFIC_SCORE_WINDOW_SAMPLE_RULE = "max(1,floor(score_window_seconds/timestep_seconds+1e-12))"
SCIENTIFIC_SENSOR_FIELDS = {
    "command_name",
    "robot_name",
    "joint_action_name",
    "contact_sensor_name",
    "foot_body_names",
    "contact_force_threshold",
    "ground_normal_axis",
    "actual_contact",
    "reference_contact",
    "foot_slip",
    "torque",
    "joint_limits",
}

_PROTOCOL_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_use",
    "schedule_sha256",
    "split_sha256",
    "split_selection_sha256",
    "reference_length_inventory_sha256",
    "mechanism_names",
    "probe_thresholds",
    "probe_thresholds_sha256",
    "score_window_config",
    "score_window_config_sha256",
    "sensor_semantics",
    "sensor_semantics_sha256",
    "termination_predicates",
    "termination_predicates_sha256",
    "termination_semantics",
    "termination_multi_hot_available",
    "domain_randomization_config",
    "domain_randomization_config_sha256",
    "primary_terrain_type",
    "eval_entrypoint",
    "scientific_passthrough_hydra_args",
    "signature_config",
    "normalizer_config",
    "data_origin",
    "git_commit",
    "source_bundle_sha256",
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _git_commit(value: Any) -> str:
    _require(isinstance(value, str) and len(value) == 40, "git_commit must have 40 hex chars")
    _require(value == value.lower(), "git_commit must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("git_commit must be hexadecimal") from error
    return value


def _validate_thresholds(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "probe_thresholds must be a mapping")
    thresholds = dict(raw)
    expected_fields = set(asdict(ProbeThresholds()))
    _require(set(thresholds) == expected_fields, "probe_thresholds fields are incomplete")
    try:
        validated = ProbeThresholds(**thresholds)
    except (TypeError, ValueError) as error:
        raise ValueError(f"probe_thresholds are invalid: {error}") from error
    return asdict(validated)


def _validate_score_window(raw: Any, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "score_window_config must be a mapping")
    config = dict(raw)
    _require(
        set(config) == {"rule", "score_window_seconds", "timestep_seconds", "sample_count_rule"},
        "score_window_config fields are invalid",
    )
    timestep = config.get("timestep_seconds")
    _require(
        isinstance(timestep, (int, float))
        and not isinstance(timestep, bool)
        and math.isfinite(float(timestep))
        and float(timestep) > 0.0,
        "score_window_config timestep must be finite and positive",
    )
    _require(
        config
        == {
            "rule": SCIENTIFIC_SCORE_WINDOW_RULE,
            "score_window_seconds": thresholds["score_window_seconds"],
            "timestep_seconds": float(timestep),
            "sample_count_rule": SCIENTIFIC_SCORE_WINDOW_SAMPLE_RULE,
        },
        "score_window_config does not match the frozen scientific rule",
    )
    return config


def _validate_sensor_semantics(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "sensor_semantics must be a mapping")
    semantics = dict(raw)
    _require(set(semantics) == SCIENTIFIC_SENSOR_FIELDS, "sensor_semantics fields are invalid")
    for name in (
        "command_name",
        "robot_name",
        "joint_action_name",
        "contact_sensor_name",
    ):
        _require(isinstance(semantics[name], str) and semantics[name], f"{name} is missing")
    feet = semantics["foot_body_names"]
    _require(
        isinstance(feet, list)
        and bool(feet)
        and all(isinstance(name, str) and name for name in feet)
        and len(feet) == len(set(feet)),
        "foot_body_names must be non-empty and unique",
    )
    threshold = semantics["contact_force_threshold"]
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and float(threshold) >= 0.0,
        "contact_force_threshold is invalid",
    )
    _require(
        semantics["ground_normal_axis"] in (0, 1, 2)
        and not isinstance(semantics["ground_normal_axis"], bool),
        "ground_normal_axis is invalid",
    )
    _require(
        semantics["actual_contact"] == "latest_net_forces_w_history_norm_threshold"
        and semantics["reference_contact"] == "sonic_feet_channel_ground_height_rule"
        and semantics["foot_slip"] == "link_origin_velocity_tangent_to_configured_plane"
        and semantics["torque"]
        == "requested_action_target_minus_joint_state_and_applied_joint_effort"
        and semantics["joint_limits"] == "live_soft_joint_position_limits",
        "sensor semantics algorithms drifted",
    )
    return semantics


def build_analysis_protocol(
    *,
    schedule_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    probe_thresholds: Mapping[str, Any],
    score_window_config: Mapping[str, Any],
    sensor_semantics: Mapping[str, Any],
    termination_predicates: Mapping[str, Any],
    domain_randomization_config: Mapping[str, Any],
    git_commit: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    """Build a frozen protocol from explicit, outcome-free payloads."""

    validate_split_manifest(split_manifest)
    validate_reference_length_inventory(reference_length_inventory)
    validate_rollout_schedule(
        schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
    )
    thresholds = _validate_thresholds(probe_thresholds)
    score_window = _validate_score_window(score_window_config, thresholds)
    sensors = _validate_sensor_semantics(sensor_semantics)
    _require(isinstance(termination_predicates, Mapping), "termination_predicates missing")
    termination = dict(termination_predicates)
    _validate_termination_predicates(termination)
    _require(
        isinstance(domain_randomization_config, Mapping),
        "domain_randomization_config must be a mapping",
    )
    domain_randomization = dict(domain_randomization_config)
    protocol: dict[str, Any] = {
        "kind": ANALYSIS_PROTOCOL_KIND,
        "schema_version": ANALYSIS_PROTOCOL_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "schedule_sha256": schedule_manifest["schedule_sha256"],
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_selection_sha256(split_manifest),
        "reference_length_inventory_sha256": reference_length_inventory[
            REFERENCE_LENGTH_DIGEST_FIELD
        ],
        "mechanism_names": list(DEFAULT_MECHANISMS),
        "probe_thresholds": thresholds,
        "probe_thresholds_sha256": canonical_sha256(thresholds),
        "score_window_config": score_window,
        "score_window_config_sha256": canonical_sha256(score_window),
        "sensor_semantics": sensors,
        "sensor_semantics_sha256": canonical_sha256(sensors),
        "termination_predicates": termination,
        "termination_predicates_sha256": canonical_sha256(termination),
        "termination_semantics": TERMINATION_MULTI_HOT_SEMANTICS,
        "termination_multi_hot_available": True,
        "domain_randomization_config": domain_randomization,
        "domain_randomization_config_sha256": canonical_sha256(domain_randomization),
        "primary_terrain_type": "plane",
        "eval_entrypoint": "gear_sonic/eval_agent_trl.py",
        "scientific_passthrough_hydra_args": [],
        "signature_config": dict(SCIENTIFIC_SIGNATURE_CONFIG),
        "normalizer_config": dict(SCIENTIFIC_NORMALIZER_CONFIG),
        "data_origin": SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
        "git_commit": _git_commit(git_commit),
        "source_bundle_sha256": _sha256(source_bundle_sha256, "source_bundle_sha256"),
    }
    protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=ANALYSIS_PROTOCOL_DIGEST_FIELD,
    )
    validate_analysis_protocol(
        protocol,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
    )
    return protocol


def validate_analysis_protocol(
    raw: Mapping[str, Any],
    *,
    schedule_manifest: Mapping[str, Any] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    reference_length_inventory: Mapping[str, Any] | None = None,
) -> None:
    _require(isinstance(raw, Mapping), "analysis protocol must be a mapping")
    protocol = dict(raw)
    _require(set(protocol) == _PROTOCOL_FIELDS, "analysis protocol fields are invalid")
    _require(
        protocol["kind"] == ANALYSIS_PROTOCOL_KIND
        and protocol["schema_version"] == ANALYSIS_PROTOCOL_SCHEMA_VERSION
        and protocol["artifact_mode"] == "scientific"
        and protocol["scientific_use"] is True,
        "analysis protocol identity is invalid",
    )
    thresholds = _validate_thresholds(protocol["probe_thresholds"])
    score_window = _validate_score_window(protocol["score_window_config"], thresholds)
    sensors = _validate_sensor_semantics(protocol["sensor_semantics"])
    _validate_termination_predicates(protocol["termination_predicates"])
    for payload, field in (
        (thresholds, "probe_thresholds_sha256"),
        (score_window, "score_window_config_sha256"),
        (sensors, "sensor_semantics_sha256"),
        (protocol["termination_predicates"], "termination_predicates_sha256"),
        (protocol["domain_randomization_config"], "domain_randomization_config_sha256"),
    ):
        _require(protocol[field] == canonical_sha256(payload), f"{field} payload drifted")
    for field in (
        "schedule_sha256",
        "split_sha256",
        "split_selection_sha256",
        "reference_length_inventory_sha256",
        "source_bundle_sha256",
    ):
        _sha256(protocol[field], field)
    _git_commit(protocol["git_commit"])
    _require(
        protocol["mechanism_names"] == list(DEFAULT_MECHANISMS)
        and protocol["signature_config"] == SCIENTIFIC_SIGNATURE_CONFIG
        and protocol["normalizer_config"] == SCIENTIFIC_NORMALIZER_CONFIG
        and protocol["data_origin"] == SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
        "analysis/normalization protocol drifted",
    )
    _require(
        protocol["termination_semantics"] == TERMINATION_MULTI_HOT_SEMANTICS
        and protocol["termination_multi_hot_available"] is True
        and protocol["primary_terrain_type"] == "plane"
        and protocol["eval_entrypoint"] == "gear_sonic/eval_agent_trl.py"
        and protocol["scientific_passthrough_hydra_args"] == [],
        "analysis runtime invariants drifted",
    )
    _require(
        protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        == canonical_sha256(protocol, digest_field=ANALYSIS_PROTOCOL_DIGEST_FIELD),
        "analysis_protocol_sha256 mismatch",
    )
    supplied = (
        schedule_manifest,
        split_manifest,
        reference_length_inventory,
    )
    _require(
        all(item is None for item in supplied) or all(item is not None for item in supplied),
        "analysis protocol external inputs must be supplied together",
    )
    if schedule_manifest is not None:
        assert split_manifest is not None and reference_length_inventory is not None
        validate_split_manifest(split_manifest)
        validate_reference_length_inventory(reference_length_inventory)
        validate_rollout_schedule(
            schedule_manifest,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
        )
        _require(
            protocol["schedule_sha256"] == schedule_manifest["schedule_sha256"]
            and protocol["split_sha256"] == split_manifest["split_sha256"]
            and protocol["split_selection_sha256"] == split_selection_sha256(split_manifest)
            and protocol["reference_length_inventory_sha256"]
            == reference_length_inventory[REFERENCE_LENGTH_DIGEST_FIELD],
            "analysis protocol frozen data identities drifted",
        )


def validate_instrument_against_analysis_protocol(
    instrument: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Prove one live cell instrument implements the pre-outcome protocol."""

    validate_scientific_instrument(instrument)
    validate_analysis_protocol(protocol)
    for field in (
        "schedule_sha256",
        "probe_thresholds",
        "probe_thresholds_sha256",
        "score_window_config",
        "score_window_config_sha256",
        "sensor_semantics",
        "sensor_semantics_sha256",
        "termination_predicates",
        "termination_predicates_sha256",
        "termination_semantics",
        "termination_multi_hot_available",
        "domain_randomization_config",
        "domain_randomization_config_sha256",
        "git_commit",
        "source_bundle_sha256",
    ):
        _require(
            instrument.get(field) == protocol.get(field),
            f"cell instrument {field} drifted from the frozen analysis protocol",
        )
    resolved = instrument["resolved_hydra_config"]
    terrain = resolved.get("manager_env", {}).get("config", {}).get("terrain_type")
    _require(terrain == protocol["primary_terrain_type"], "cell terrain drifted from protocol")


def build_analysis_protocol_from_instrument(
    instrument: Mapping[str, Any],
    *,
    schedule_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Convenience for a pre-rollout live-instrument preflight."""

    validate_scientific_instrument(instrument)
    return build_analysis_protocol(
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        probe_thresholds=instrument["probe_thresholds"],
        score_window_config=instrument["score_window_config"],
        sensor_semantics=instrument["sensor_semantics"],
        termination_predicates=instrument["termination_predicates"],
        domain_randomization_config=instrument["domain_randomization_config"],
        git_commit=instrument["git_commit"],
        source_bundle_sha256=instrument["source_bundle_sha256"],
    )
