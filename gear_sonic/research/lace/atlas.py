"""Build a frozen LACE atlas from validated episode-level probe records."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
    SCIENTIFIC_SIGNATURE_CONFIG,
    validate_analysis_protocol,
)
from gear_sonic.research.lace.analysis_protocol_lock import (
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
)
from gear_sonic.research.lace.instrument import (
    MEASUREMENT_FAMILY_DIGEST_FIELD,
    episode_measurement_family_sha256,
    validate_measurement_family,
    validate_scientific_instrument,
)
from gear_sonic.research.lace.instrument_collection import (
    ROLLOUT_COLLECTION_DIGEST_FIELD,
    validate_rollout_collection,
)
from gear_sonic.research.lace.normalizer import (
    DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
    NORMALIZER_METHOD,
    fit_d_atlas_normalizer,
    normalizer_input_sha256,
)
from gear_sonic.research.lace.schedule import (
    DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
    SCHEDULE_KIND,
    SCHEDULE_SCHEMA_VERSION,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import (
    ATLAS_KIND,
    ATLAS_SCHEMA_VERSION,
    ATLAS_V1_INTERVAL_EVENT_POLICY,
    RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
    RUNTIME_REALIZATION_KIND,
    RUNTIME_REALIZATION_SCHEMA_VERSION,
    TERMINATION_TRACE_ALGORITHM,
    TERMINATION_TRACE_KIND,
    TERMINATION_TRACE_SCHEMA_VERSION,
    canonical_sha256,
    validate_atlas_manifest,
    validate_split_manifest,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS, build_factorized_signatures

ROLLOUT_KIND = "lace_probe_rollouts"
ROLLOUT_SCHEMA_VERSION = 4
ATLAS_MODES = ("scientific", "contract_smoke")
SCIENTIFIC_SEED_SEMANTICS = "runtime_rng_seed_applied_and_realization_hash_verified"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _validate_probe_policies(raw: Any) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    _require(isinstance(raw, list) and raw, "probe_policies must be a non-empty list")
    policies: list[dict[str, Any]] = []
    policy_ids: list[str] = []
    for index, item in enumerate(raw):
        _require(isinstance(item, Mapping), f"probe_policies[{index}] must be a mapping")
        policy_id = item.get("id")
        _require(isinstance(policy_id, str) and policy_id, f"probe_policies[{index}].id missing")
        checkpoint_sha256 = _validate_sha256(
            item.get("checkpoint_sha256"),
            f"probe_policies[{index}].checkpoint_sha256",
        )
        policy_ids.append(policy_id)
        policies.append({**dict(item), "id": policy_id, "checkpoint_sha256": checkpoint_sha256})
    _require(len(policy_ids) == len(set(policy_ids)), "probe policy ids must be unique")
    return policies, tuple(policy_ids)


def _validate_normalizer(
    raw: Any,
    mechanism_names: Sequence[str],
    *,
    artifact_mode: str,
) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "normalizer must be a mapping")
    normalizer = dict(raw)
    _require(normalizer.get("frozen") is True, "normalizer.frozen must be true")
    _require(
        normalizer.get("fit_partition") == "D_atlas",
        "normalizer.fit_partition must be D_atlas",
    )
    scales = normalizer.get("mechanism_scales")
    _require(isinstance(scales, Mapping), "normalizer.mechanism_scales must be a mapping")
    _require(
        set(scales) == set(mechanism_names),
        "normalizer.mechanism_scales must exactly match mechanism_names",
    )
    for name, value in scales.items():
        numeric = float(value)
        _require(
            math.isfinite(numeric) and numeric > 0.0,
            f"normalizer.mechanism_scales[{name!r}] must be finite and positive",
        )
    if artifact_mode == "scientific":
        _require(normalizer.get("schema_version") == 1, "normalizer.schema_version must be 1")
        _require(
            normalizer.get("method") == NORMALIZER_METHOD,
            f"scientific normalizer.method must be {NORMALIZER_METHOD!r}",
        )
        _require(
            normalizer.get("mechanism_names") == list(mechanism_names),
            "normalizer.mechanism_names must preserve the declared channel order",
        )
        input_sha256 = normalizer.get("input_sha256")
        _validate_sha256(input_sha256, "normalizer.input_sha256")
        counts = normalizer.get("positive_observation_counts")
        minimums = normalizer.get("minimum_positive_observations")
        _require(isinstance(counts, Mapping), "normalizer positive counts must be a mapping")
        _require(isinstance(minimums, Mapping), "normalizer evidence floors must be a mapping")
        _require(
            set(counts) == set(mechanism_names) and set(minimums) == set(mechanism_names),
            "normalizer evidence maps must exactly match mechanism_names",
        )
        for name in mechanism_names:
            observed = counts[name]
            required = minimums[name]
            _require(
                isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0,
                f"normalizer positive count for {name!r} is invalid",
            )
            _require(
                isinstance(required, int)
                and not isinstance(required, bool)
                and required >= DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
                f"scientific normalizer evidence floor for {name!r} must be at least "
                f"{DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS}",
            )
            _require(observed >= required, f"normalizer evidence for {name!r} is insufficient")
    else:
        _require(
            normalizer.get("kind") == "identity_contract_smoke",
            "contract-smoke normalizer must be explicitly identity_contract_smoke",
        )
    return normalizer


def _validate_domain_randomization_seeds(raw: Any) -> tuple[int, ...]:
    _require(isinstance(raw, list) and raw, "domain_randomization_seeds must be non-empty")
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in raw),
        "DR seeds must be integers",
    )
    seeds = tuple(raw)
    _require(len(seeds) == len(set(seeds)), "domain_randomization_seeds must be unique")
    return seeds


def _validate_rollout_schedule(
    raw: Any,
    *,
    declared_seeds: Sequence[int],
) -> tuple[tuple[int, float, int], ...]:
    _require(isinstance(raw, list) and raw, "rollout_schedule must be a non-empty list")
    schedule: list[tuple[int, float, int]] = []
    for index, item in enumerate(raw):
        _require(isinstance(item, Mapping), f"rollout_schedule[{index}] must be a mapping")
        seed = item.get("domain_randomization_seed")
        phase = item.get("initial_phase")
        repeat = item.get("repeat_index")
        _require(seed in declared_seeds, f"rollout_schedule[{index}] seed is undeclared")
        phase_value = float(phase)
        _require(
            math.isfinite(phase_value) and 0.0 <= phase_value <= 1.0,
            f"rollout_schedule[{index}].initial_phase is invalid",
        )
        _require(
            isinstance(repeat, int) and not isinstance(repeat, bool) and repeat >= 0,
            f"rollout_schedule[{index}].repeat_index must be a nonnegative integer",
        )
        entry = (int(seed), phase_value, repeat)
        _require(entry not in schedule, f"duplicate rollout_schedule entry: {entry}")
        schedule.append(entry)
    _require(
        {entry[0] for entry in schedule} == set(declared_seeds),
        "rollout_schedule must cover every declared DR seed",
    )
    return tuple(schedule)


def _validate_signature_config(raw: Any, *, artifact_mode: str) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), "signature_config must be a mapping")
    config = dict(raw)
    minimum = config.get("minimum_resolved_failures")
    _require(
        isinstance(minimum, int) and not isinstance(minimum, bool) and minimum >= 1,
        "signature_config.minimum_resolved_failures must be a positive integer",
    )
    if artifact_mode == "scientific":
        beta_prior = config.get("difficulty_beta_prior")
        _require(
            isinstance(beta_prior, list)
            and len(beta_prior) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in beta_prior
            ),
            "scientific signature_config.difficulty_beta_prior is invalid",
        )
        _require(
            minimum >= 3,
            "scientific signature_config.minimum_resolved_failures must be at least 3",
        )
    prior = float(config.get("mechanism_dirichlet_prior"))
    interval = float(config.get("credible_interval_level"))
    _require(
        math.isfinite(prior) and prior > 0.0,
        "signature_config.mechanism_dirichlet_prior must be finite and positive",
    )
    _require(
        math.isfinite(interval) and 0.0 < interval < 1.0,
        "signature_config.credible_interval_level must lie inside (0,1)",
    )
    return config


def _validate_runtime_realization(
    raw: Any,
    expected_sha256: Any,
    *,
    episode: Mapping[str, Any],
    index: int,
) -> None:
    prefix = f"episodes[{index}].domain_randomization_realization"
    _require(isinstance(raw, Mapping), f"{prefix} must be a mapping")
    realization = dict(raw)
    _require(
        realization.get("kind") == RUNTIME_REALIZATION_KIND,
        f"{prefix}.kind must be {RUNTIME_REALIZATION_KIND!r}",
    )
    _require(
        realization.get("schema_version") == RUNTIME_REALIZATION_SCHEMA_VERSION,
        f"{prefix}.schema_version must be {RUNTIME_REALIZATION_SCHEMA_VERSION}",
    )
    _require(
        realization.get("capture_lifecycle") == RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        f"{prefix}.capture_lifecycle is not the post-reset scientific boundary",
    )
    _require(
        realization.get("runtime_rng_seed_semantics") == SCIENTIFIC_SEED_SEMANTICS,
        f"{prefix}.runtime_rng_seed_semantics is invalid",
    )
    _require(
        realization.get("interval_event_policy") == ATLAS_V1_INTERVAL_EVENT_POLICY,
        f"{prefix}.interval_event_policy is invalid",
    )
    event_config = realization.get("resolved_event_configuration")
    _require(isinstance(event_config, Mapping), f"{prefix}.resolved_event_configuration missing")
    _require(
        event_config.get("interval_events_instrumented") is False,
        f"{prefix} must explicitly reject uninstrumented interval events",
    )
    _require(
        event_config.get("interval_event_policy") == ATLAS_V1_INTERVAL_EVENT_POLICY,
        f"{prefix}.resolved_event_configuration interval policy is invalid",
    )
    _require(
        realization.get("resolved_event_configuration_sha256") == canonical_sha256(event_config),
        f"{prefix}.resolved_event_configuration_sha256 mismatch",
    )
    for field in ("ordered_names", "realized_parameters", "post_reset_state"):
        _require(isinstance(realization.get(field), Mapping), f"{prefix}.{field} missing")
    scheduled_reference = realization.get("scheduled_reference")
    _require(isinstance(scheduled_reference, Mapping), f"{prefix}.scheduled_reference missing")
    identity = scheduled_reference.get("identity")
    _require(isinstance(identity, Mapping), f"{prefix}.scheduled_reference.identity missing")
    _require(
        isinstance(scheduled_reference.get("state"), Mapping),
        f"{prefix}.scheduled_reference.state missing",
    )
    for realization_field, episode_field in (
        ("schedule_sha256", "schedule_sha256"),
        ("motion_key", "motion_key"),
        ("domain_randomization_seed", "domain_randomization_seed"),
        ("runtime_rng_seed", "runtime_rng_seed"),
        ("phase_id", "phase_id"),
        ("target_fraction", "target_fraction"),
        ("realized_fraction", "realized_fraction"),
        ("reference_start_step", "reference_start_step"),
        ("reference_num_steps", "reference_num_steps"),
        ("repeat_index", "repeat_index"),
    ):
        _require(
            identity.get(realization_field) == episode.get(episode_field),
            f"{prefix}.scheduled_reference.identity.{realization_field} drifted from episode",
        )
    digest = _validate_sha256(expected_sha256, f"{prefix}_sha256")
    _require(digest == canonical_sha256(realization), f"{prefix}_sha256 mismatch")


def _validate_termination_multi_hot(episode: Mapping[str, Any], *, index: int) -> None:
    prefix = f"episodes[{index}].termination_multi_hot"
    _require(
        episode.get("termination_multi_hot_available") is True,
        f"episodes[{index}] lacks an independent termination multi-hot trace",
    )
    _require(
        episode.get("termination_semantics")
        == "instrumented_single_evaluation_ordered_raw_boolean_matrix",
        f"episodes[{index}].termination_semantics is not independently instrumented",
    )
    raw = episode.get("termination_multi_hot")
    _require(isinstance(raw, Mapping), f"{prefix} must be a mapping")
    term_names = raw.get("term_names")
    timeout_flags = raw.get("time_out_flags")
    values = raw.get("values")
    generations = raw.get("trace_generations")
    step_counters = raw.get("common_step_counters")
    _require(
        isinstance(term_names, list)
        and term_names
        and all(isinstance(name, str) and name for name in term_names)
        and len(term_names) == len(set(term_names)),
        f"{prefix}.term_names must be non-empty and unique",
    )
    _require(
        isinstance(timeout_flags, list)
        and len(timeout_flags) == len(term_names)
        and all(isinstance(flag, bool) for flag in timeout_flags),
        f"{prefix}.time_out_flags must align with term_names",
    )
    _require(
        isinstance(values, list)
        and values
        and all(
            isinstance(row, list)
            and len(row) == len(term_names)
            and all(isinstance(value, bool) for value in row)
            for row in values
        ),
        f"{prefix}.values must be a non-empty boolean matrix",
    )
    _require(
        isinstance(generations, list)
        and len(generations) == len(values)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in generations
        )
        and all(right - left == 1 for left, right in zip(generations, generations[1:])),
        f"{prefix}.trace_generations must be consecutive positive integers",
    )
    _require(
        isinstance(step_counters, list)
        and len(step_counters) == len(values)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in step_counters
        )
        and all(right - left == 1 for left, right in zip(step_counters, step_counters[1:])),
        f"{prefix}.common_step_counters must be consecutive",
    )
    _require(
        len(
            {
                generation - step_counter
                for generation, step_counter in zip(generations, step_counters, strict=True)
            }
        )
        == 1,
        f"{prefix} generation/common-step offset drifted",
    )
    _require(
        episode.get("buffered_step_count") == len(values),
        f"{prefix}.values length does not match buffered_step_count",
    )
    contract = raw.get("contract")
    _require(isinstance(contract, Mapping), f"{prefix}.contract must be a mapping")
    _require(contract.get("kind") == TERMINATION_TRACE_KIND, f"{prefix}.contract kind is invalid")
    _require(
        contract.get("schema_version") == TERMINATION_TRACE_SCHEMA_VERSION,
        f"{prefix}.contract schema version is invalid",
    )
    _require(
        contract.get("algorithm") == TERMINATION_TRACE_ALGORITHM,
        f"{prefix}.contract algorithm is invalid",
    )
    _require(
        contract.get("term_names") == term_names
        and contract.get("time_out_flags") == timeout_flags,
        f"{prefix}.contract term order/timeout flags drifted",
    )
    _require(
        contract.get("raw_trace_semantics")
        == "ordered_independent_values_from_single_manager_evaluation",
        f"{prefix}.contract raw trace semantics are invalid",
    )
    _require(
        contract.get("legacy_term_dones_semantics") == "last_trigger_wins_stale_rows_preserved",
        f"{prefix}.contract legacy semantics are invalid",
    )
    _require(
        contract.get("freshness_semantics") == "one_compute_one_consume_common_step_counter_bound",
        f"{prefix}.contract freshness semantics are invalid",
    )
    term_configs = contract.get("term_configs")
    _require(isinstance(term_configs, list), f"{prefix}.contract.term_configs must be a list")
    _require(
        contract.get("term_config_sha256") == canonical_sha256({"terms": term_configs}),
        f"{prefix}.contract.term_config_sha256 mismatch",
    )
    for field in (
        "manager_compute_source_sha256",
        "instrument_compute_source_sha256",
        "term_config_sha256",
    ):
        _validate_sha256(contract.get(field), f"{prefix}.contract.{field}")
    contract_sha256 = _validate_sha256(raw.get("contract_sha256"), f"{prefix}.contract_sha256")
    _require(contract_sha256 == canonical_sha256(contract), f"{prefix}.contract_sha256 mismatch")
    # A true predicate ends the environment immediately, so only the final
    # buffered row may contain causes. Simultaneous final causes are retained.
    _require(
        not any(any(row) for row in values[:-1]),
        f"{prefix}.values contains a cause before the episode-ending row",
    )
    _require(
        any(values[-1]),
        f"{prefix}.values final row has no episode-ending predicate",
    )
    final_non_timeout = any(
        value for value, is_timeout in zip(values[-1], timeout_flags) if not is_timeout
    )
    _require(
        final_non_timeout is (episode.get("failed") is True),
        f"{prefix} non-timeout union disagrees with failed",
    )
    diagnostics = episode.get("termination_terms")
    _require(
        isinstance(diagnostics, Mapping),
        f"episodes[{index}].termination_terms must be a mapping",
    )
    _require(
        len(diagnostics) == len(term_names) and set(diagnostics) == set(term_names),
        f"episodes[{index}].termination_terms must exactly match the multi-hot term-name set",
    )
    # JSON objects are unordered scientific data. Iterate by the explicit
    # column order carried beside the raw matrix, never mapping insertion order.
    for column, name in enumerate(term_names):
        term = diagnostics[name]
        _require(
            isinstance(term, Mapping), f"episodes[{index}].termination_terms[{name!r}] invalid"
        )
        occurred = any(row[column] for row in values)
        _require(
            term.get("occurred") is occurred,
            f"episodes[{index}].termination_terms[{name!r}].occurred disagrees with raw trace",
        )


def _validate_scientific_instrument(
    raw: Any,
    *,
    episodes: Sequence[Mapping[str, Any]],
    expected_schedule_sha256: str,
) -> tuple[dict[str, Any], str]:
    _require(isinstance(raw, Mapping), "scientific instrument must be a mapping")
    instrument = dict(raw)
    validate_scientific_instrument(instrument)
    _require(instrument.get("schema_version") == 1, "instrument.schema_version must be 1")
    digest_fields = (
        "probe_thresholds_sha256",
        "recorder_config_sha256",
        "resolved_hydra_config_sha256",
        "environment_sha256",
        "sensor_semantics_sha256",
        "termination_predicates_sha256",
        "score_window_config_sha256",
        "domain_randomization_config_sha256",
        "schedule_sha256",
    )
    for field in digest_fields:
        _validate_sha256(instrument.get(field), f"instrument.{field}")
    _require(
        instrument["schedule_sha256"] == expected_schedule_sha256,
        "instrument.schedule_sha256 does not match the frozen rollout schedule",
    )
    git_commit = instrument.get("git_commit")
    _require(
        isinstance(git_commit, str) and len(git_commit) == 40,
        "instrument.git_commit must be a full 40-character commit id",
    )
    try:
        int(git_commit, 16)
    except ValueError as exc:
        raise ValueError("instrument.git_commit must be hexadecimal") from exc
    _require(
        instrument.get("domain_randomization_seed_semantics")
        == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "scientific declared DR-seed semantics do not match the schedule contract",
    )
    _require(
        instrument.get("runtime_rng_seed_semantics") == SCIENTIFIC_SEED_SEMANTICS,
        "scientific runtime RNG seeds must be applied and verified for each tuple",
    )
    _require(
        instrument.get("termination_multi_hot_available") is True,
        "scientific instrument requires an independently retained single-evaluation "
        "multi-hot termination trace",
    )
    instrument_sha256 = canonical_sha256(instrument)
    realization_sha256_by_common_random_cell: dict[tuple[Any, ...], str] = {}
    for index, episode in enumerate(episodes):
        _require(
            episode.get("instrument_sha256") == instrument_sha256,
            f"episodes[{index}].instrument_sha256 does not match the frozen instrument",
        )
        _require(
            episode.get("probe_thresholds_sha256") == instrument["probe_thresholds_sha256"],
            f"episodes[{index}] uses a different probe threshold contract",
        )
        _require(
            episode.get("domain_randomization_seed_semantics")
            == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
            f"episodes[{index}] declared DR-seed semantics do not match the schedule",
        )
        _require(
            episode.get("runtime_rng_seed_semantics") == SCIENTIFIC_SEED_SEMANTICS,
            f"episodes[{index}] does not verify its applied runtime RNG seed",
        )
        _validate_runtime_realization(
            episode.get("domain_randomization_realization"),
            episode.get("domain_randomization_realization_sha256"),
            episode=episode,
            index=index,
        )
        _validate_termination_multi_hot(episode, index=index)
        _require(
            episode["termination_multi_hot"].get("contract")
            == instrument["termination_predicates"],
            f"episodes[{index}] termination contract differs from its cell instrument",
        )
        common_random_cell = (
            episode.get("motion_key"),
            episode.get("domain_randomization_seed"),
            episode.get("runtime_rng_seed"),
            episode.get("phase_id"),
            episode.get("target_fraction"),
            episode.get("realized_fraction"),
            episode.get("reference_start_step"),
            episode.get("reference_num_steps"),
            episode.get("repeat_index"),
        )
        realization_sha256 = str(episode["domain_randomization_realization_sha256"])
        prior_sha256 = realization_sha256_by_common_random_cell.setdefault(
            common_random_cell,
            realization_sha256,
        )
        _require(
            prior_sha256 == realization_sha256,
            "scientific common-random-number cell has policy-dependent runtime realization: "
            f"{common_random_cell}",
        )
        schedule_entry_id = episode.get("schedule_entry_id")
        _require(
            isinstance(schedule_entry_id, str) and schedule_entry_id,
            f"episodes[{index}].schedule_entry_id missing",
        )
    return instrument, instrument_sha256


def _required_mapping_child(raw: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    child = raw.get(key)
    _require(isinstance(child, Mapping), f"{name}.{key} must be a mapping")
    return child


def _validate_cell_instrument_binding(
    instrument: Mapping[str, Any],
    *,
    scheduled_rows: Sequence[Mapping[str, Any]],
    schedule_sha256: str,
) -> None:
    """Prove every normalized cell coordinate against its frozen schedule cell."""

    _require(bool(scheduled_rows), "instrument execution cell must not be empty")
    expected_assignments = [dict(row) for row in scheduled_rows]
    expected_ids = [str(row["rollout_id"]) for row in scheduled_rows]
    runtime_seeds = {row["runtime_rng_seed"] for row in scheduled_rows}
    policy_ids = {row["probe_policy_id"] for row in scheduled_rows}
    checkpoint_digests = {row["checkpoint_sha256"] for row in scheduled_rows}
    _require(
        len(runtime_seeds) == len(policy_ids) == len(checkpoint_digests) == 1,
        "instrument execution cell has heterogeneous policy/checkpoint/runtime seed",
    )

    config = instrument["resolved_hydra_config"]
    _require(
        config.get("lace_scientific_instrument_required") is True, "cell config is not scientific"
    )
    _require(config.get("num_envs") == len(scheduled_rows), "cell config num_envs drifted")
    _require(config.get("seed") == next(iter(runtime_seeds)), "cell config seed drifted")
    _require(config.get("headless") is True, "cell config must be headless")
    _require(config.get("run_eval_loop") is True, "cell config disables policy rollout")
    _require(config.get("run_once") is True, "cell config is not one episode per environment")
    _require(config.get("use_encoder") == "g1", "cell config does not select the G1 encoder")
    _require(config.get("eval_callbacks") == [], "cell config eval callbacks are not frozen empty")
    expected_max_steps = (
        max(int(row["reference_num_steps"]) - int(row["start_step"]) for row in scheduled_rows) + 2
    )
    _require(config.get("max_render_steps") == expected_max_steps, "cell max_render_steps drifted")

    manager_env = _required_mapping_child(config, "manager_env", "resolved_hydra_config")
    manager_config = _required_mapping_child(manager_env, "config", "manager_env")
    _require(manager_config.get("terrain_type") == "plane", "cell terrain is not the frozen plane")
    _require(manager_config.get("render_results") is False, "cell rendering must be disabled")
    commands = _required_mapping_child(manager_env, "commands", "manager_env")
    motion = _required_mapping_child(commands, "motion", "manager_env.commands")
    _require(motion.get("atlas_probe_mode") is True, "cell atlas_probe_mode is not enabled")
    _require(
        motion.get("atlas_probe_schedule_sha256") == schedule_sha256,
        "cell config schedule digest drifted",
    )
    _require(
        motion.get("atlas_probe_assignments") == expected_assignments,
        "cell config assignments do not exactly match the frozen schedule cell",
    )
    expected_motion_keys = [str(row["motion_key"]) for row in scheduled_rows]
    _require(
        motion.get("filter_motion_keys") == expected_motion_keys,
        "cell config motion filter order drifted",
    )
    motion_lib = _required_mapping_child(motion, "motion_lib_cfg", "manager_env.commands.motion")
    adaptive = _required_mapping_child(
        motion_lib,
        "adaptive_sampling",
        "manager_env.commands.motion.motion_lib_cfg",
    )
    _require(adaptive.get("enable") is False, "cell native adaptive sampling is not disabled")
    _require(
        motion_lib.get("filter_motion_keys") == expected_motion_keys,
        "cell motion-library filter order drifted",
    )
    for field in ("motion_file", "smpl_motion_file"):
        value = motion_lib.get(field)
        _require(
            isinstance(value, str)
            and bool(value)
            and value != "dummy"
            and Path(value).is_absolute()
            and Path(value).resolve() == Path(value),
            f"cell motion-library {field} is not a canonical scientific dataset root",
        )
    observations = _required_mapping_child(manager_env, "observations", "manager_env")
    for group_name in ("policy", "tokenizer"):
        group = _required_mapping_child(observations, group_name, "manager_env.observations")
        _require(
            group.get("enable_corruption") is False, f"cell {group_name} corruption is enabled"
        )

    recorders = _required_mapping_child(manager_env, "recorders", "manager_env")
    failure_atlas = _required_mapping_child(recorders, "failure_atlas", "manager_env.recorders")
    _require(failure_atlas.get("enabled") is True, "cell failure-atlas recorder is disabled")
    _require(
        failure_atlas.get("allow_append_existing") is False,
        "cell failure-atlas recorder allows append",
    )
    recorder = instrument["recorder_config"]
    _require(
        recorder.get("resolved_hydra_term") == failure_atlas,
        "cell recorder payload drifted from its resolved Hydra term",
    )
    recorder_runtime = _required_mapping_child(recorder, "runtime", "recorder_config")
    _require(recorder_runtime.get("num_envs") == len(scheduled_rows), "recorder num_envs drifted")
    _require(
        recorder_runtime.get("cell_identity")
        == {
            "schedule_sha256": schedule_sha256,
            "checkpoint_sha256": next(iter(checkpoint_digests)),
            "probe_policy_id": next(iter(policy_ids)),
            "rollout_ids": expected_ids,
        },
        "recorder cell identity does not bind its frozen execution cell",
    )


def _validate_scientific_instruments(
    raw_family: Any,
    raw_instruments: Any,
    *,
    episodes: Sequence[Mapping[str, Any]],
    expected_schedule_sha256: str,
    scheduled_by_rollout_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    """Validate per-cell instruments plus their common measurement family."""

    _require(isinstance(raw_family, Mapping), "scientific measurement_family must be a mapping")
    family = dict(raw_family)
    _require(
        isinstance(raw_instruments, Mapping) and bool(raw_instruments),
        "scientific cell_instruments must be a non-empty digest map",
    )
    instruments: dict[str, dict[str, Any]] = {}
    episodes_by_instrument: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    instrument_digests_by_cell: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    scheduled_rows_by_cell: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for scheduled in scheduled_by_rollout_id.values():
        cell = (
            scheduled["probe_policy_id"],
            scheduled["checkpoint_sha256"],
            scheduled["domain_randomization_seed"],
            scheduled["runtime_rng_seed"],
            scheduled["phase_id"],
            scheduled["target_fraction"],
            scheduled["repeat_index"],
        )
        scheduled_rows_by_cell[cell].append(scheduled)
    for index, episode in enumerate(episodes):
        digest = _validate_sha256(
            episode.get("instrument_sha256"),
            f"episodes[{index}].instrument_sha256",
        )
        episodes_by_instrument[digest].append(episode)
        scheduled = scheduled_by_rollout_id[str(episode["rollout_id"])]
        cell = (
            scheduled["probe_policy_id"],
            scheduled["checkpoint_sha256"],
            scheduled["domain_randomization_seed"],
            scheduled["runtime_rng_seed"],
            scheduled["phase_id"],
            scheduled["target_fraction"],
            scheduled["repeat_index"],
        )
        instrument_digests_by_cell[cell].add(digest)
        _require(
            episode.get("measurement_family_sha256") == episode_measurement_family_sha256(family),
            f"episodes[{index}].measurement_family_sha256 mismatch",
        )
        _require(
            episode.get("measurement_family_manifest_sha256")
            == family.get(MEASUREMENT_FAMILY_DIGEST_FIELD),
            f"episodes[{index}].measurement_family_manifest_sha256 mismatch",
        )

    for raw_digest, raw_instrument in raw_instruments.items():
        digest = _validate_sha256(raw_digest, "cell_instruments key")
        _require(isinstance(raw_instrument, Mapping), f"cell_instruments[{digest}] is invalid")
        instrument, computed_digest = _validate_scientific_instrument(
            raw_instrument,
            episodes=episodes_by_instrument.get(digest, []),
            expected_schedule_sha256=expected_schedule_sha256,
        )
        _require(digest == computed_digest, f"cell_instruments[{digest}] key does not bind value")
        instruments[digest] = instrument

    _require(
        set(episodes_by_instrument) == set(instruments),
        "scientific episodes and cell_instruments must reference exactly the same digests",
    )
    _require(
        set(instrument_digests_by_cell) == set(scheduled_rows_by_cell),
        "scientific instrument cells do not exactly cover the frozen schedule cells",
    )
    _require(
        all(len(digests) == 1 for digests in instrument_digests_by_cell.values()),
        "each exact execution cell must use exactly one cell instrument digest",
    )
    digest_by_cell = {
        cell: next(iter(digests)) for cell, digests in instrument_digests_by_cell.items()
    }
    _require(
        len(set(digest_by_cell.values())) == len(digest_by_cell),
        "one cell instrument digest may not be reused across execution cells",
    )
    _require(
        len(instruments) == len(scheduled_rows_by_cell),
        "cell instrument count does not match the frozen execution-cell count",
    )
    for cell, scheduled_rows in scheduled_rows_by_cell.items():
        _validate_cell_instrument_binding(
            instruments[digest_by_cell[cell]],
            scheduled_rows=scheduled_rows,
            schedule_sha256=expected_schedule_sha256,
        )
    validate_measurement_family(family, instruments=list(instruments.values()))

    # Common-random-number realizations must remain policy independent even
    # though policies necessarily live in different cell instruments.
    realization_by_common_cell: dict[tuple[Any, ...], str] = {}
    for episode in episodes:
        common_random_cell = (
            episode.get("motion_key"),
            episode.get("domain_randomization_seed"),
            episode.get("runtime_rng_seed"),
            episode.get("phase_id"),
            episode.get("target_fraction"),
            episode.get("realized_fraction"),
            episode.get("reference_start_step"),
            episode.get("reference_num_steps"),
            episode.get("repeat_index"),
        )
        realization_digest = str(episode["domain_randomization_realization_sha256"])
        previous = realization_by_common_cell.setdefault(common_random_cell, realization_digest)
        _require(
            previous == realization_digest,
            "scientific common-random-number cell has policy-dependent runtime realization: "
            f"{common_random_cell}",
        )
    family_digest = episode_measurement_family_sha256(family)
    return family, family_digest, {key: instruments[key] for key in sorted(instruments)}


def _scientific_schedule_summary(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Retain immutable axes while leaving tuple rows in their locked artifact."""

    fields = (
        "kind",
        "schema_version",
        "artifact_mode",
        "scientific_use",
        "schedule_sha256",
        "split_sha256",
        "split_selection_sha256",
        "partition",
        "selected_motion_keys",
        "motion_count",
        "reference_length_inventory_binding",
        "probe_policies",
        "policy_order",
        "domain_randomization_seeds",
        "domain_randomization_seed_semantics",
        "phase_targets",
        "repeat_count",
        "repeat_indices",
        "runtime_rng_seed_derivation",
        "runtime_rng_seed_schedule",
        "quantization",
        "cartesian_order",
        "rollout_count",
    )
    return {field: schedule[field] for field in fields}


def _validate_episode_against_schedule(
    episode: Mapping[str, Any],
    scheduled: Mapping[str, Any],
    *,
    schedule_sha256: str,
    index: int,
) -> None:
    """Fail if runtime output drifts from any precommitted tuple coordinate."""

    exact_fields = (
        "rollout_id",
        "partition",
        "motion_key",
        "probe_policy_id",
        "checkpoint_sha256",
        "domain_randomization_seed",
        "runtime_rng_seed",
        "phase_id",
        "target_fraction",
        "reference_num_steps",
        "repeat_index",
    )
    for field in exact_fields:
        _require(
            episode.get(field) == scheduled[field],
            f"episodes[{index}].{field} does not match its frozen schedule row",
        )
    _require(
        episode.get("reference_start_step") == scheduled["start_step"],
        f"episodes[{index}].reference_start_step does not match its frozen schedule row",
    )
    for field, expected in (
        ("realized_fraction", scheduled["realized_fraction"]),
        ("initial_phase", scheduled["realized_fraction"]),
    ):
        value = episode.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-15),
            f"episodes[{index}].{field} does not match its frozen schedule row",
        )
    _require(
        episode.get("schedule_sha256") == schedule_sha256,
        f"episodes[{index}].schedule_sha256 does not match the frozen schedule",
    )
    _require(
        episode.get("schedule_entry_id") == scheduled["rollout_id"],
        f"episodes[{index}].schedule_entry_id does not match its rollout_id",
    )


def build_atlas_manifest(
    rollout_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    schedule_manifest: Mapping[str, Any] | None = None,
    reference_length_inventory: Mapping[str, Any] | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate raw probe episodes and return a frozen atlas artifact.

    Scientific construction requires the independently frozen schedule artifact;
    the rollout manifest cannot self-attest its own treatment grid. Contract-smoke
    inputs retain a small explicit legacy grid for CPU-only schema testing.
    """

    validate_split_manifest(split_manifest)
    _require(rollout_manifest.get("kind") == ROLLOUT_KIND, f"kind must be {ROLLOUT_KIND!r}")
    artifact_mode = rollout_manifest.get("artifact_mode")
    _require(artifact_mode in ATLAS_MODES, f"artifact_mode must be one of {ATLAS_MODES}")
    rollout_schema_version = rollout_manifest.get("schema_version")
    if artifact_mode == "scientific":
        _require(
            rollout_schema_version == ROLLOUT_SCHEMA_VERSION,
            f"scientific rollout requires schema {ROLLOUT_SCHEMA_VERSION}",
        )
    else:
        _require(
            rollout_schema_version == 3,
            "historical contract-smoke rollout must remain schema 3",
        )
    _require(
        rollout_manifest.get("split_sha256") == split_manifest["split_sha256"],
        "rollout split_sha256 does not match split manifest",
    )

    mechanism_names_raw = rollout_manifest.get("mechanism_names")
    _require(
        isinstance(mechanism_names_raw, list) and mechanism_names_raw,
        "mechanism_names must be a non-empty list",
    )
    mechanism_names = tuple(mechanism_names_raw)
    _require(
        all(isinstance(name, str) and name for name in mechanism_names),
        "mechanism_names must contain non-empty strings",
    )
    _require(len(mechanism_names) == len(set(mechanism_names)), "mechanism_names must be unique")
    if artifact_mode == "scientific":
        _require(
            mechanism_names == DEFAULT_MECHANISMS,
            "scientific mechanism_names must equal the frozen six-channel protocol",
        )
    probe_policies, policy_ids = _validate_probe_policies(rollout_manifest.get("probe_policies"))
    dr_seeds = _validate_domain_randomization_seeds(
        rollout_manifest.get("domain_randomization_seeds")
    )
    frozen_schedule: dict[str, Any] | None = None
    frozen_schedule_sha256: str | None = None
    scheduled_by_rollout_id: dict[str, Mapping[str, Any]] = {}
    declared_schedule: tuple[tuple[int, float, int], ...] | None = None
    analysis_protocol: dict[str, Any] | None = None
    analysis_protocol_sha256: str | None = None
    if artifact_mode == "scientific":
        _require(isinstance(schedule_manifest, Mapping), "scientific atlas requires --schedule")
        _require(
            isinstance(reference_length_inventory, Mapping),
            "scientific atlas requires --reference-length-inventory",
        )
        validate_rollout_schedule(
            schedule_manifest,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
        )
        raw_protocol = rollout_manifest.get("analysis_protocol")
        _require(isinstance(raw_protocol, Mapping), "scientific rollouts require analysis protocol")
        validate_analysis_protocol(
            raw_protocol,
            schedule_manifest=schedule_manifest,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
        )
        analysis_protocol = dict(raw_protocol)
        analysis_protocol_sha256 = analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        _require(
            rollout_manifest.get(ANALYSIS_PROTOCOL_DIGEST_FIELD) == analysis_protocol_sha256,
            "rollout analysis protocol digest drifted",
        )
        _require(
            schedule_manifest.get("kind") == SCHEDULE_KIND
            and schedule_manifest.get("schema_version") == SCHEDULE_SCHEMA_VERSION
            and schedule_manifest.get("artifact_mode") == "scientific"
            and schedule_manifest.get("scientific_use") is True,
            "scientific atlas requires a scientific rollout schedule",
        )
        _require(
            rollout_manifest.get("rollout_schedule") is None,
            "scientific rollouts cannot carry a self-attested legacy rollout_schedule",
        )
        _require(
            rollout_manifest.get("probe_policies") == schedule_manifest["probe_policies"],
            "rollout probe_policies do not exactly match the frozen schedule",
        )
        _require(
            rollout_manifest.get("domain_randomization_seeds")
            == schedule_manifest["domain_randomization_seeds"],
            "rollout DR conditions do not exactly match the frozen schedule",
        )
        frozen_schedule = dict(schedule_manifest)
        frozen_schedule_sha256 = str(schedule_manifest["schedule_sha256"])
        scheduled_by_rollout_id = {
            str(row["rollout_id"]): row for row in schedule_manifest["rollouts"]
        }
    else:
        _require(
            schedule_manifest is None,
            "contract_smoke atlas must not be supplied a scientific schedule",
        )
        _require(
            reference_length_inventory is None,
            "contract_smoke atlas must not be supplied a scientific reference-length inventory",
        )
        declared_schedule = _validate_rollout_schedule(
            rollout_manifest.get("rollout_schedule"),
            declared_seeds=dr_seeds,
        )
    signature_config = _validate_signature_config(
        rollout_manifest.get("signature_config"),
        artifact_mode=str(artifact_mode),
    )
    if artifact_mode == "scientific":
        assert analysis_protocol is not None
        _require(
            signature_config == SCIENTIFIC_SIGNATURE_CONFIG
            and signature_config == analysis_protocol["signature_config"],
            "scientific signature config drifted from analysis protocol",
        )
    normalizer = _validate_normalizer(
        rollout_manifest.get("normalizer"),
        mechanism_names,
        artifact_mode=str(artifact_mode),
    )
    data_origin = rollout_manifest.get("data_origin")
    _require(isinstance(data_origin, str) and data_origin, "data_origin must be a non-empty string")
    if artifact_mode == "scientific":
        assert analysis_protocol is not None
        _require(
            data_origin == SCIENTIFIC_ROLLOUT_DATA_ORIGIN
            and data_origin == analysis_protocol["data_origin"],
            "scientific data origin drifted from analysis protocol",
        )

    split_records = {record["motion_key"]: record for record in split_manifest["motions"]}
    expected_partition_motions = {
        record["motion_key"]
        for record in split_manifest["motions"]
        if record["partition"] == "D_atlas"
    }
    selected_motion_keys_raw = rollout_manifest.get("selected_motion_keys")
    _require(
        isinstance(selected_motion_keys_raw, list) and selected_motion_keys_raw,
        "selected_motion_keys must be a non-empty list",
    )
    _require(
        all(isinstance(key, str) and key for key in selected_motion_keys_raw),
        "selected_motion_keys must contain non-empty strings",
    )
    _require(
        len(selected_motion_keys_raw) == len(set(selected_motion_keys_raw)),
        "selected_motion_keys must be unique",
    )
    selected_motion_keys = set(selected_motion_keys_raw)
    _require(
        selected_motion_keys <= expected_partition_motions,
        "selected_motion_keys must be a subset of the frozen D_atlas partition",
    )
    if artifact_mode == "scientific":
        _require(
            selected_motion_keys == expected_partition_motions,
            "scientific atlas selection must exactly cover the frozen D_atlas partition",
        )
        assert frozen_schedule is not None
        _require(
            sorted(selected_motion_keys) == frozen_schedule["selected_motion_keys"],
            "rollout selected_motion_keys do not exactly match the frozen schedule",
        )
    episodes_raw = rollout_manifest.get("episodes")
    _require(isinstance(episodes_raw, list) and episodes_raw, "episodes must be a non-empty list")
    episodes: list[dict[str, Any]] = []
    rollout_ids: set[str] = set()
    coverage: dict[tuple[str, str], set[tuple[int, float, int]]] = defaultdict(set)

    for index, item in enumerate(episodes_raw):
        _require(isinstance(item, Mapping), f"episodes[{index}] must be a mapping")
        episode = dict(item)
        rollout_id = episode.get("rollout_id")
        _require(
            isinstance(rollout_id, str) and rollout_id, f"episodes[{index}].rollout_id missing"
        )
        _require(rollout_id not in rollout_ids, f"duplicate rollout_id: {rollout_id}")
        rollout_ids.add(rollout_id)
        if artifact_mode == "scientific":
            scheduled = scheduled_by_rollout_id.get(rollout_id)
            _require(scheduled is not None, f"episodes[{index}] rollout_id is not scheduled")
            assert frozen_schedule_sha256 is not None
            _validate_episode_against_schedule(
                episode,
                scheduled,
                schedule_sha256=frozen_schedule_sha256,
                index=index,
            )

        motion_key = episode.get("motion_key")
        _require(
            motion_key in split_records,
            f"episodes[{index}] motion is absent from split: {motion_key!r}",
        )
        split_record = split_records[motion_key]
        _require(
            split_record["partition"] == "D_atlas",
            f"motion {motion_key!r} belongs to {split_record['partition']}, not D_atlas",
        )
        _require(
            motion_key in selected_motion_keys,
            f"motion {motion_key!r} is absent from selected_motion_keys",
        )
        _require(
            episode.get("partition") == "D_atlas",
            f"episodes[{index}].partition must be D_atlas",
        )
        policy_id = episode.get("probe_policy_id")
        _require(
            policy_id in policy_ids, f"episodes[{index}] has undeclared probe policy: {policy_id!r}"
        )
        seed = episode.get("domain_randomization_seed")
        _require(seed in dr_seeds, f"episodes[{index}] has undeclared DR seed: {seed!r}")
        phase = float(episode.get("initial_phase", -1.0))
        _require(
            math.isfinite(phase) and 0.0 <= phase <= 1.0, f"episodes[{index}].initial_phase invalid"
        )
        repeat_index = episode.get("repeat_index")
        _require(
            isinstance(repeat_index, int)
            and not isinstance(repeat_index, bool)
            and repeat_index >= 0,
            f"episodes[{index}].repeat_index must be a nonnegative integer",
        )
        coverage_key = (str(motion_key), str(policy_id))
        if artifact_mode == "contract_smoke":
            schedule_key = (int(seed), phase, repeat_index)
            _require(
                schedule_key not in coverage[coverage_key],
                f"duplicate rollout schedule for motion/policy {coverage_key}: {schedule_key}",
            )
            coverage[coverage_key].add(schedule_key)
        else:
            coverage[coverage_key].add((int(seed), phase, repeat_index))
        episodes.append(episode)

    measurement_family: dict[str, Any] | None = None
    measurement_family_sha256: str | None = None
    cell_instruments: dict[str, dict[str, Any]] = {}
    rollout_collection_sha256: str | None = None
    rollout_receipts: list[dict[str, Any]] = []
    if artifact_mode == "scientific":
        assert frozen_schedule_sha256 is not None
        _require(
            rollout_manifest.get("instrument") is None,
            "scientific rollouts must use per-cell instruments, not one global instrument",
        )
        collection = rollout_manifest.get("rollout_collection")
        _require(
            isinstance(collection, Mapping),
            "scientific rollouts require a receipt-driven rollout_collection",
        )
        assert frozen_schedule is not None
        assert reference_length_inventory is not None
        validate_rollout_collection(
            collection,
            schedule_manifest=frozen_schedule,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
            analysis_protocol=analysis_protocol,
            analysis_protocol_path=rollout_manifest.get("analysis_protocol_path"),
            analysis_protocol_lock_path=rollout_manifest.get("analysis_protocol_lock_path"),
            expected_analysis_protocol_lock_sha256=rollout_manifest.get(
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ),
            repo_root=repo_root,
        )
        _require(
            collection.get("episodes") == episodes,
            "rollout episodes differ from the receipt-driven collection",
        )
        _require(
            collection.get("analysis_protocol") == analysis_protocol
            and collection.get(ANALYSIS_PROTOCOL_DIGEST_FIELD) == analysis_protocol_sha256
            and collection.get("analysis_protocol_path")
            == rollout_manifest.get("analysis_protocol_path")
            and collection.get("analysis_protocol_file_sha256")
            == rollout_manifest.get("analysis_protocol_file_sha256"),
            "rollout collection analysis protocol drifted",
        )
        _require(
            collection.get("analysis_protocol_lock")
            == rollout_manifest.get("analysis_protocol_lock")
            and collection.get("analysis_protocol_lock_path")
            == rollout_manifest.get("analysis_protocol_lock_path")
            and collection.get("analysis_protocol_lock_file_sha256")
            == rollout_manifest.get("analysis_protocol_lock_file_sha256")
            and collection.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD)
            == rollout_manifest.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD),
            "rollout collection analysis protocol preregistration lock drifted",
        )
        _require(
            rollout_manifest.get("measurement_family") is None
            and rollout_manifest.get("cell_instruments") in (None, {}),
            "scientific rollout identity must come only from rollout_collection",
        )
        (
            measurement_family,
            measurement_family_sha256,
            cell_instruments,
        ) = _validate_scientific_instruments(
            collection.get("measurement_family"),
            collection.get("cell_instruments"),
            episodes=episodes,
            expected_schedule_sha256=frozen_schedule_sha256,
            scheduled_by_rollout_id=scheduled_by_rollout_id,
        )
        rollout_collection_sha256 = str(collection[ROLLOUT_COLLECTION_DIGEST_FIELD])
        rollout_receipts = [
            {
                "cell": dict(item["cell"]),
                "receipt_path": item["receipt_path"],
                "receipt_file_sha256": item["receipt_file_sha256"],
                "rollout_binding_sha256": item["rollout_binding_sha256"],
                "episode_instrument_sha256": item["receipt"]["episode_instrument_sha256"],
                "rollout_count": item["receipt"]["rollout_count"],
                "rollout_ids_sha256": canonical_sha256(
                    {"rollout_ids": item["receipt"]["rollout_ids"]}
                ),
            }
            for item in collection["receipts"]
        ]
    else:
        _require(
            rollout_manifest.get("instrument") is None,
            "contract_smoke rollouts must not carry a scientific instrument claim",
        )
        _require(
            rollout_manifest.get("measurement_family") is None
            and rollout_manifest.get("cell_instruments") in (None, {})
            and rollout_manifest.get("rollout_collection") is None,
            "contract_smoke rollouts must not carry scientific cell instruments",
        )

    motion_keys = sorted({motion_key for motion_key, _ in coverage})
    _require(
        set(motion_keys) == selected_motion_keys,
        "episode motion coverage must exactly match selected_motion_keys",
    )
    if artifact_mode == "scientific":
        _require(
            rollout_ids == set(scheduled_by_rollout_id),
            "scientific episodes must exactly cover every frozen schedule rollout_id once",
        )
    else:
        assert declared_schedule is not None
        expected_schedule = set(declared_schedule)
        for motion_key in sorted(selected_motion_keys):
            schedules = []
            for policy_id in policy_ids:
                schedule = coverage.get((motion_key, policy_id))
                _require(
                    schedule is not None,
                    f"motion {motion_key!r} is missing policy {policy_id!r}",
                )
                _require(
                    schedule == expected_schedule,
                    f"motion {motion_key!r}, policy {policy_id!r} does not exactly match the "
                    "declared rollout_schedule",
                )
                schedules.append(schedule)
            first_schedule = schedules[0]
            _require(
                all(schedule == first_schedule for schedule in schedules[1:]),
                f"motion {motion_key!r} does not use common random numbers across policies",
            )

    if artifact_mode == "scientific":
        expected_normalizer_input = normalizer_input_sha256(episodes, mechanism_names)
        _require(
            normalizer["input_sha256"] == expected_normalizer_input,
            "normalizer.input_sha256 does not bind the exact rollout episodes",
        )
        expected_normalizer = fit_d_atlas_normalizer(
            episodes,
            mechanism_names=DEFAULT_MECHANISMS,
            minimum_positive_observations=DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
            quantile=0.90,
        )
        _require(
            normalizer == expected_normalizer,
            "scientific normalizer is not the deterministic frozen-protocol fit",
        )

    signatures = build_factorized_signatures(
        episodes,
        mechanism_names=mechanism_names,
        mechanism_scales=normalizer["mechanism_scales"],
        beta_prior=tuple(signature_config.get("difficulty_beta_prior", (0.5, 0.5))),
        minimum_resolved_failures=int(signature_config["minimum_resolved_failures"]),
        mechanism_dirichlet_prior=float(signature_config["mechanism_dirichlet_prior"]),
        credible_interval_level=float(signature_config["credible_interval_level"]),
    )
    for signature in signatures:
        split_record = split_records[signature["motion_key"]]
        signature["source_group_id"] = split_record["source_group_id"]
        signature["partition"] = split_record["partition"]

    atlas: dict[str, Any] = {
        "kind": ATLAS_KIND,
        "schema_version": ATLAS_SCHEMA_VERSION if artifact_mode == "scientific" else 4,
        "artifact_mode": artifact_mode,
        "scientific_use": artifact_mode == "scientific",
        "data_origin": data_origin,
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "rollout_manifest_sha256": canonical_sha256(rollout_manifest),
        "mechanism_names": list(mechanism_names),
        "normalizer": normalizer,
        "signature_config": signature_config,
        "analysis_protocol": analysis_protocol,
        ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol_sha256,
        "analysis_protocol_path": (
            rollout_manifest.get("analysis_protocol_path")
            if artifact_mode == "scientific"
            else None
        ),
        "analysis_protocol_file_sha256": (
            rollout_manifest.get("analysis_protocol_file_sha256")
            if artifact_mode == "scientific"
            else None
        ),
        "analysis_protocol_lock": (
            rollout_manifest.get("analysis_protocol_lock")
            if artifact_mode == "scientific"
            else None
        ),
        "analysis_protocol_lock_path": (
            rollout_manifest.get("analysis_protocol_lock_path")
            if artifact_mode == "scientific"
            else None
        ),
        "analysis_protocol_lock_file_sha256": (
            rollout_manifest.get("analysis_protocol_lock_file_sha256")
            if artifact_mode == "scientific"
            else None
        ),
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: (
            rollout_manifest.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD)
            if artifact_mode == "scientific"
            else None
        ),
        "measurement_family": measurement_family,
        "measurement_family_sha256": measurement_family_sha256,
        "cell_instruments": cell_instruments,
        "cell_instrument_count": len(cell_instruments),
        "rollout_collection_sha256": rollout_collection_sha256,
        "rollout_receipts": rollout_receipts,
        "rollout_receipt_count": len(rollout_receipts),
        "probe_policies": probe_policies,
        "domain_randomization_seeds": list(dr_seeds),
        "rollout_schedule": (
            [
                {
                    "domain_randomization_seed": seed,
                    "initial_phase": phase,
                    "repeat_index": repeat,
                }
                for seed, phase, repeat in declared_schedule
            ]
            if declared_schedule is not None
            else None
        ),
        "rollout_schedule_sha256": frozen_schedule_sha256,
        "rollout_schedule_summary": (
            _scientific_schedule_summary(frozen_schedule) if frozen_schedule is not None else None
        ),
        "selected_motion_keys": sorted(selected_motion_keys),
        "selection_complete_for_d_atlas": artifact_mode == "scientific",
        "motion_count": len(motion_keys),
        "rollout_count": len(episodes),
        "signatures": signatures,
    }
    atlas["atlas_sha256"] = canonical_sha256(atlas)
    validate_atlas_manifest(atlas)
    return atlas


def validate_scientific_atlas_deep(
    atlas_manifest: Mapping[str, Any],
    rollout_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    schedule_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    *,
    analysis_protocol: Mapping[str, Any],
    analysis_protocol_path: str | Path,
    analysis_protocol_lock_path: str | Path,
    expected_analysis_protocol_lock_sha256: str,
    repo_root: str | Path,
) -> None:
    """Rebuild a scientific atlas from its source artifacts and compare all fields.

    The standalone schema validator is intentionally structural.  Scientific
    consumers must use this source-driven verifier so a self-rehashed q,
    evidence, posterior, or signature substitution cannot pass.
    """

    _require(
        atlas_manifest.get("artifact_mode") == "scientific",
        "deep atlas verification requires a scientific atlas",
    )
    collection = rollout_manifest.get("rollout_collection")
    _require(
        isinstance(collection, Mapping),
        "deep atlas verification requires a receipt-driven rollout collection",
    )
    validate_rollout_collection(
        collection,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        analysis_protocol=analysis_protocol,
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_lock_path=analysis_protocol_lock_path,
        expected_analysis_protocol_lock_sha256=(expected_analysis_protocol_lock_sha256),
        verify_artifacts=True,
        repo_root=repo_root,
    )
    _require(
        rollout_manifest.get("analysis_protocol") == dict(analysis_protocol)
        and rollout_manifest.get("analysis_protocol_path")
        == str(Path(analysis_protocol_path).expanduser().resolve()),
        "rollout wrapper drifted from the external analysis protocol binding",
    )
    rebuilt = build_atlas_manifest(
        rollout_manifest,
        split_manifest,
        schedule_manifest,
        reference_length_inventory,
        repo_root=repo_root,
    )
    _require(dict(atlas_manifest) == rebuilt, "stored atlas differs from source-driven rebuild")
