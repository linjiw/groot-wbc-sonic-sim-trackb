"""CPU-only construction of a frozen LACE scientific instrument manifest.

The atlas consumes rollout records produced by live Isaac Lab code, but the
identity of that measurement instrument can be assembled and audited without
starting a simulator.  This module deliberately accepts explicit JSON
payloads: it never probes the current machine, guesses a resolved Hydra value,
or upgrades an unverified termination stream to an independent multi-hot
trace.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.schedule import DOMAIN_RANDOMIZATION_SEED_SEMANTICS
from gear_sonic.research.lace.schema import (
    TERMINATION_TRACE_ALGORITHM,
    TERMINATION_TRACE_KIND,
    TERMINATION_TRACE_SCHEMA_VERSION,
    canonical_sha256,
)

INSTRUMENT_KIND = "lace_scientific_instrument"
INSTRUMENT_SCHEMA_VERSION = 1
INSTRUMENT_DIGEST_FIELD = "instrument_manifest_sha256"
SOURCE_BUNDLE_KIND = "lace_repository_source_bundle"
SOURCE_BUNDLE_SCHEMA_VERSION = 1
SOURCE_BUNDLE_DIGEST_FIELD = "source_bundle_sha256"
MEASUREMENT_FAMILY_KIND = "lace_scientific_measurement_family"
MEASUREMENT_FAMILY_SCHEMA_VERSION = 1
MEASUREMENT_FAMILY_DIGEST_FIELD = "measurement_family_manifest_sha256"

RUNTIME_RNG_SEED_SEMANTICS = "runtime_rng_seed_applied_and_realization_hash_verified"
RESOLVED_HYDRA_CONFIG_SEMANTICS = "fully_resolved_after_checkpoint_merge_and_command_line_overrides"
TERMINATION_MULTI_HOT_SEMANTICS = "instrumented_single_evaluation_ordered_raw_boolean_matrix"
TERMINATION_RAW_TRACE_SEMANTICS = "ordered_independent_values_from_single_manager_evaluation"
TERMINATION_FRESHNESS_SEMANTICS = "one_compute_one_consume_common_step_counter_bound"
TERMINATION_LEGACY_DIAGNOSTIC_SEMANTICS = "last_trigger_wins_stale_rows_preserved"
SCIENTIFIC_CACHE_ENVIRONMENT_KEYS = (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "ISAACLAB_USD_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TORCH_HOME",
    "OMNI_USER_CACHE_DIR",
)
SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS = (
    "one_canonical_launch_token_root_with_explicit_no_home_cache_fallback_v1"
)
ENVIRONMENT_CACHE_NORMALIZATION_SEMANTICS = (
    "validated_exact_per_launch_cache_paths_masked_by_environment_key_v1"
)
ENVIRONMENT_CELL_COORDINATE_PATHS = ("process_argv",)

# A scientific atlas is collected as separate policy/DR/phase/repeat cells.
# These are the only resolved-config coordinates allowed to differ while still
# belonging to one measurement family.  Missing paths are not invented; they
# remain missing and therefore affect the normalized digest.
RESOLVED_CONFIG_CELL_COORDINATE_PATHS = (
    "checkpoint",
    "manager_env.commands.motion.atlas_probe_assignments",
    "max_render_steps",
    "seed",
)
RESOLVED_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS = (
    "eval_log_dir",
    "eval_timestamp",
    "experiment_dir",
    "lace_instrument_handshake_file_sha256",
    "lace_instrument_handshake_path",
    "lace_launch_plan_path",
    "manager_env.config.experiment_dir",
    "manager_env.config.save_rendering_dir",
    "manager_env.recorders.failure_atlas.output_path",
    "output_dir",
    "timestamp",
)
RECORDER_CONFIG_CELL_COORDINATE_PATHS = ("runtime.cell_identity",)
RECORDER_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS = (
    "output_path",
    "resolved_hydra_term.output_path",
)

_PAYLOAD_DIGEST_FIELDS = {
    "probe_thresholds": "probe_thresholds_sha256",
    "recorder_config": "recorder_config_sha256",
    "resolved_hydra_config": "resolved_hydra_config_sha256",
    "environment_fingerprint": "environment_sha256",
    "sensor_semantics": "sensor_semantics_sha256",
    "termination_predicates": "termination_predicates_sha256",
    "score_window_config": "score_window_config_sha256",
    "domain_randomization_config": "domain_randomization_config_sha256",
}

_INSTRUMENT_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_use",
    *_PAYLOAD_DIGEST_FIELDS,
    *_PAYLOAD_DIGEST_FIELDS.values(),
    "resolved_hydra_config_semantics",
    "termination_semantics",
    "termination_multi_hot_available",
    "schedule_sha256",
    "git_commit",
    "domain_randomization_seed_semantics",
    "runtime_rng_seed_semantics",
    "source_bundle",
    "source_bundle_sha256",
    INSTRUMENT_DIGEST_FIELD,
}

_SOURCE_BUNDLE_FIELDS = {
    "kind",
    "schema_version",
    "hash_algorithm",
    "path_semantics",
    "ordering",
    "files",
    SOURCE_BUNDLE_DIGEST_FIELD,
}

_SOURCE_FILE_FIELDS = {"path", "size_bytes", "sha256"}

_TERMINATION_CONTRACT_FIELDS = {
    "kind",
    "schema_version",
    "algorithm",
    "manager_type",
    "manager_compute_source_sha256",
    "instrument_compute_source_sha256",
    "term_names",
    "time_out_flags",
    "term_config_sha256",
    "term_configs",
    "legacy_term_dones_semantics",
    "raw_trace_semantics",
    "freshness_semantics",
}

_MEASUREMENT_FAMILY_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_use",
    "schedule_sha256",
    "probe_thresholds_sha256",
    "recorder_config_common_sha256",
    "resolved_hydra_config_common_sha256",
    "environment_sha256",
    "sensor_semantics_sha256",
    "termination_predicates_sha256",
    "score_window_config_sha256",
    "domain_randomization_config_sha256",
    "git_commit",
    "source_bundle_sha256",
    "domain_randomization_seed_semantics",
    "runtime_rng_seed_semantics",
    "termination_semantics",
    "termination_multi_hot_available",
    "resolved_config_cell_coordinate_paths",
    "resolved_config_noncausal_bookkeeping_paths",
    "recorder_config_cell_coordinate_paths",
    "recorder_config_noncausal_bookkeeping_paths",
    "environment_cache_coordinate_keys",
    "environment_cache_normalization_semantics",
    "environment_cell_coordinate_paths",
    MEASUREMENT_FAMILY_DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_sha256(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64,
        f"{name} must be a 64-character SHA-256",
    )
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _validate_git_commit(value: Any) -> str:
    _require(
        isinstance(value, str) and len(value) == 40,
        "git_commit must be a full 40-character commit id",
    )
    _require(value == value.lower(), "git_commit must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("git_commit must be hexadecimal") from error
    return value


def _canonical_mapping(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    _require(isinstance(raw, Mapping), f"{name} must be a JSON mapping")
    _require(bool(raw), f"{name} must not be empty")
    _validate_json_tree(raw, name)
    try:
        encoded = json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    _require(isinstance(canonical, dict), f"{name} must canonicalize to a JSON object")
    _require(
        all(isinstance(key, str) for key in canonical),
        f"{name} keys must be strings",
    )
    return canonical


def _validate_json_tree(raw: Any, name: str) -> None:
    """Reject key coercion and non-finite/non-JSON runtime values before hashing."""

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            _require(isinstance(key, str), f"{name} keys must be strings")
            _validate_json_tree(value, f"{name}.{key}")
        return
    if isinstance(raw, (list, tuple)):
        for index, value in enumerate(raw):
            _validate_json_tree(value, f"{name}[{index}]")
        return
    _require(
        raw is None or isinstance(raw, (str, int, float, bool)),
        f"{name} contains a non-JSON value",
    )


def _contains_hydra_interpolation(raw: Any) -> bool:
    if isinstance(raw, Mapping):
        return any(_contains_hydra_interpolation(value) for value in raw.values())
    if isinstance(raw, (list, tuple)):
        return any(_contains_hydra_interpolation(value) for value in raw)
    return isinstance(raw, str) and "${" in raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_repository_root(repo_root: str | Path) -> Path:
    root = Path(repo_root)
    _require(root.exists(), f"repository root does not exist: {root}")
    _require(root.is_dir(), f"repository root is not a directory: {root}")
    _require(not root.is_symlink(), "repository root may not be a symlink")
    return root.resolve(strict=True)


def _resolve_source_path(repo_root: Path, raw_path: str | Path) -> tuple[Path, str]:
    path = Path(raw_path)
    _require(str(path) not in {"", "."}, "source path must name a file")
    _require(".." not in path.parts, f"source path may not contain '..': {raw_path}")

    candidate = path if path.is_absolute() else repo_root / path
    try:
        lexical_relative = candidate.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"source path is outside repository: {raw_path}") from error

    current = repo_root
    for component in lexical_relative.parts:
        current = current / component
        _require(not current.is_symlink(), f"source path traverses a symlink: {raw_path}")

    _require(candidate.exists(), f"source path does not exist: {raw_path}")
    _require(candidate.is_file(), f"source path is not a regular file: {raw_path}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"source path resolves outside repository: {raw_path}") from error

    relative_posix = relative.as_posix()
    _validate_relative_source_path(relative_posix, "source path")
    return resolved, relative_posix


def _validate_relative_source_path(value: Any, name: str) -> str:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty string")
    _require("\\" not in value, f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{name} must be repository-relative")
    _require(".." not in path.parts, f"{name} may not traverse outside the repository")
    _require(path.as_posix() == value and value != ".", f"{name} is not canonical")
    return value


def build_source_bundle(
    repo_root: str | Path,
    source_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Bind exact bytes for an explicitly selected repository source set.

    Inputs may be repository-relative paths or absolute paths inside the
    repository.  The artifact always stores unique, sorted POSIX-relative
    paths.  Symlinks are rejected so the binding cannot silently depend on
    bytes outside the declared repository root.
    """

    root = _validated_repository_root(repo_root)
    _require(
        isinstance(source_paths, Sequence) and not isinstance(source_paths, (str, bytes)),
        "source_paths must be a non-empty path sequence",
    )
    _require(bool(source_paths), "source bundle must contain at least one file")

    resolved_by_relative: dict[str, Path] = {}
    for raw_path in source_paths:
        resolved, relative = _resolve_source_path(root, raw_path)
        _require(relative not in resolved_by_relative, f"duplicate source path: {relative}")
        resolved_by_relative[relative] = resolved

    files = [
        {
            "path": relative,
            "size_bytes": resolved_by_relative[relative].stat().st_size,
            "sha256": _file_sha256(resolved_by_relative[relative]),
        }
        for relative in sorted(resolved_by_relative)
    ]
    bundle: dict[str, Any] = {
        "kind": SOURCE_BUNDLE_KIND,
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "path_semantics": "repository_relative_posix_regular_files_no_symlinks",
        "ordering": "ascending_lexicographic_repository_relative_path",
        "files": files,
    }
    bundle[SOURCE_BUNDLE_DIGEST_FIELD] = canonical_sha256(
        bundle,
        digest_field=SOURCE_BUNDLE_DIGEST_FIELD,
    )
    validate_source_bundle(bundle, repo_root=root)
    return bundle


def validate_source_bundle(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Validate bundle structure, self-digest, and optionally live file bytes."""

    _require(isinstance(raw, Mapping), "source_bundle must be a mapping")
    bundle = dict(raw)
    _require(
        set(bundle) == _SOURCE_BUNDLE_FIELDS,
        "source_bundle fields do not match schema v1",
    )
    _require(bundle.get("kind") == SOURCE_BUNDLE_KIND, "source_bundle.kind is invalid")
    _require(
        bundle.get("schema_version") == SOURCE_BUNDLE_SCHEMA_VERSION,
        "source_bundle.schema_version is invalid",
    )
    _require(bundle.get("hash_algorithm") == "sha256", "source_bundle hash is invalid")
    _require(
        bundle.get("path_semantics") == "repository_relative_posix_regular_files_no_symlinks",
        "source_bundle path semantics are invalid",
    )
    _require(
        bundle.get("ordering") == "ascending_lexicographic_repository_relative_path",
        "source_bundle ordering semantics are invalid",
    )
    files = bundle.get("files")
    _require(isinstance(files, list) and files, "source_bundle.files must be non-empty")

    paths: list[str] = []
    for index, raw_record in enumerate(files):
        _require(isinstance(raw_record, Mapping), f"source_bundle.files[{index}] is invalid")
        record = dict(raw_record)
        _require(
            set(record) == _SOURCE_FILE_FIELDS,
            f"source_bundle.files[{index}] fields are invalid",
        )
        path = _validate_relative_source_path(
            record.get("path"),
            f"source_bundle.files[{index}].path",
        )
        size_bytes = record.get("size_bytes")
        _require(
            isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0,
            f"source_bundle.files[{index}].size_bytes is invalid",
        )
        _validate_sha256(record.get("sha256"), f"source_bundle.files[{index}].sha256")
        paths.append(path)

    _require(len(paths) == len(set(paths)), "source_bundle contains duplicate paths")
    _require(paths == sorted(paths), "source_bundle paths are not in canonical order")
    declared_digest = _validate_sha256(
        bundle.get(SOURCE_BUNDLE_DIGEST_FIELD),
        f"source_bundle.{SOURCE_BUNDLE_DIGEST_FIELD}",
    )
    computed_digest = canonical_sha256(bundle, digest_field=SOURCE_BUNDLE_DIGEST_FIELD)
    _require(declared_digest == computed_digest, "source_bundle_sha256 mismatch")

    if repo_root is None:
        return
    root = _validated_repository_root(repo_root)
    for index, record in enumerate(files):
        path, relative = _resolve_source_path(root, record["path"])
        _require(relative == record["path"], f"source_bundle.files[{index}].path drifted")
        _require(
            path.stat().st_size == record["size_bytes"],
            f"source bundle file size drifted: {relative}",
        )
        _require(
            _file_sha256(path) == record["sha256"],
            f"source bundle file digest drifted: {relative}",
        )


def _validate_termination_predicates(raw: Mapping[str, Any]) -> None:
    contract = dict(raw)
    _require(
        set(contract) == _TERMINATION_CONTRACT_FIELDS,
        "termination_predicates fields do not match the verified trace schema",
    )
    _require(
        contract.get("kind") == TERMINATION_TRACE_KIND,
        "termination_predicates.kind is invalid",
    )
    _require(
        contract.get("schema_version") == TERMINATION_TRACE_SCHEMA_VERSION,
        "termination_predicates.schema_version is invalid",
    )
    _require(
        contract.get("algorithm") == TERMINATION_TRACE_ALGORITHM,
        "termination predicate algorithm is not the pinned single-evaluation implementation",
    )
    _require(
        isinstance(contract.get("manager_type"), str) and bool(contract["manager_type"]),
        "termination_predicates.manager_type must be explicit",
    )
    for field in (
        "manager_compute_source_sha256",
        "instrument_compute_source_sha256",
        "term_config_sha256",
    ):
        _validate_sha256(contract.get(field), f"termination_predicates.{field}")

    names = contract.get("term_names")
    timeout_flags = contract.get("time_out_flags")
    configs = contract.get("term_configs")
    _require(
        isinstance(names, list)
        and bool(names)
        and all(isinstance(name, str) and name for name in names)
        and len(names) == len(set(names)),
        "termination_predicates.term_names must be non-empty, ordered, and unique",
    )
    _require(
        isinstance(timeout_flags, list)
        and len(timeout_flags) == len(names)
        and all(isinstance(flag, bool) for flag in timeout_flags),
        "termination_predicates.time_out_flags must align with term_names",
    )
    _require(
        isinstance(configs, list) and len(configs) == len(names),
        "termination_predicates.term_configs must align with term_names",
    )
    for index, (name, timeout, raw_config) in enumerate(
        zip(names, timeout_flags, configs, strict=True)
    ):
        _require(
            isinstance(raw_config, Mapping),
            f"termination_predicates.term_configs[{index}] must be a mapping",
        )
        config = dict(raw_config)
        _require(
            set(config) == {"term_name", "callable", "time_out", "params"},
            f"termination_predicates.term_configs[{index}] fields are invalid",
        )
        _require(
            config.get("term_name") == name,
            f"termination_predicates.term_configs[{index}] order drifted",
        )
        _require(
            config.get("time_out") is timeout,
            f"termination_predicates.term_configs[{index}].time_out drifted",
        )
        _require(
            isinstance(config.get("callable"), str) and bool(config["callable"]),
            f"termination_predicates.term_configs[{index}].callable is missing",
        )
        _require(
            isinstance(config.get("params"), Mapping),
            f"termination_predicates.term_configs[{index}].params must be a mapping",
        )
    _require(
        contract["term_config_sha256"] == canonical_sha256({"terms": configs}),
        "termination_predicates.term_config_sha256 mismatch",
    )
    _require(
        contract.get("legacy_term_dones_semantics") == TERMINATION_LEGACY_DIAGNOSTIC_SEMANTICS,
        "termination predicate legacy diagnostic semantics are invalid",
    )
    _require(
        contract.get("raw_trace_semantics") == TERMINATION_RAW_TRACE_SEMANTICS,
        "termination predicates do not declare an ordered single-evaluation raw trace",
    )
    _require(
        contract.get("freshness_semantics") == TERMINATION_FRESHNESS_SEMANTICS,
        "termination predicate trace freshness semantics are invalid",
    )


def build_scientific_instrument(
    *,
    schedule_sha256: str,
    probe_thresholds: Mapping[str, Any],
    recorder_config: Mapping[str, Any],
    resolved_hydra_config: Mapping[str, Any],
    environment_fingerprint: Mapping[str, Any],
    sensor_semantics: Mapping[str, Any],
    termination_predicates: Mapping[str, Any],
    score_window_config: Mapping[str, Any],
    domain_randomization_config: Mapping[str, Any],
    git_commit: str,
    repo_root: str | Path,
    source_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Build a deterministic, source-bound schema-v1 scientific instrument."""

    schedule_digest = _validate_sha256(schedule_sha256, "schedule_sha256")
    commit = _validate_git_commit(git_commit)
    payload_inputs = {
        "probe_thresholds": probe_thresholds,
        "recorder_config": recorder_config,
        "resolved_hydra_config": resolved_hydra_config,
        "environment_fingerprint": environment_fingerprint,
        "sensor_semantics": sensor_semantics,
        "termination_predicates": termination_predicates,
        "score_window_config": score_window_config,
        "domain_randomization_config": domain_randomization_config,
    }
    payloads = {name: _canonical_mapping(value, name) for name, value in payload_inputs.items()}
    _validate_termination_predicates(payloads["termination_predicates"])
    source_bundle = build_source_bundle(repo_root, source_paths)

    instrument: dict[str, Any] = {
        "kind": INSTRUMENT_KIND,
        "schema_version": INSTRUMENT_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        **payloads,
        "resolved_hydra_config_semantics": RESOLVED_HYDRA_CONFIG_SEMANTICS,
        "termination_semantics": TERMINATION_MULTI_HOT_SEMANTICS,
        "termination_multi_hot_available": True,
        "schedule_sha256": schedule_digest,
        "git_commit": commit,
        "domain_randomization_seed_semantics": DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "runtime_rng_seed_semantics": RUNTIME_RNG_SEED_SEMANTICS,
        "source_bundle": source_bundle,
        "source_bundle_sha256": source_bundle[SOURCE_BUNDLE_DIGEST_FIELD],
    }
    for payload_name, digest_name in _PAYLOAD_DIGEST_FIELDS.items():
        instrument[digest_name] = canonical_sha256(instrument[payload_name])
    instrument[INSTRUMENT_DIGEST_FIELD] = canonical_sha256(
        instrument,
        digest_field=INSTRUMENT_DIGEST_FIELD,
    )
    validate_scientific_instrument(instrument, repo_root=repo_root)
    return instrument


def validate_scientific_instrument(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Validate a scientific instrument and optionally re-hash repository files."""

    _require(isinstance(raw, Mapping), "instrument must be a mapping")
    instrument = dict(raw)
    _require(
        set(instrument) == _INSTRUMENT_FIELDS,
        "instrument fields do not match scientific schema v1",
    )
    _require(instrument.get("kind") == INSTRUMENT_KIND, "instrument.kind is invalid")
    _require(
        instrument.get("schema_version") == INSTRUMENT_SCHEMA_VERSION,
        "instrument.schema_version must be 1",
    )
    _require(instrument.get("artifact_mode") == "scientific", "instrument mode is invalid")
    _require(instrument.get("scientific_use") is True, "instrument.scientific_use must be true")

    for payload_name, digest_name in _PAYLOAD_DIGEST_FIELDS.items():
        payload = _canonical_mapping(instrument.get(payload_name), payload_name)
        declared = _validate_sha256(instrument.get(digest_name), f"instrument.{digest_name}")
        _require(
            declared == canonical_sha256(payload),
            f"instrument.{digest_name} does not bind {payload_name}",
        )
    _require(
        not _contains_hydra_interpolation(instrument["resolved_hydra_config"]),
        "resolved_hydra_config still contains an OmegaConf/Hydra interpolation",
    )
    _validate_termination_predicates(instrument["termination_predicates"])
    _require(
        instrument.get("resolved_hydra_config_semantics") == RESOLVED_HYDRA_CONFIG_SEMANTICS,
        "resolved Hydra config is not declared post-checkpoint and post-CLI",
    )
    _require(
        instrument.get("termination_semantics") == TERMINATION_MULTI_HOT_SEMANTICS,
        "instrument termination semantics are not an ordered raw boolean matrix",
    )
    _require(
        instrument.get("termination_multi_hot_available") is True,
        "scientific instrument requires verified independent termination multi-hot",
    )
    _validate_sha256(instrument.get("schedule_sha256"), "instrument.schedule_sha256")
    _validate_git_commit(instrument.get("git_commit"))
    _require(
        instrument.get("domain_randomization_seed_semantics")
        == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "instrument domain-randomization seed semantics do not match the schedule schema",
    )
    _require(
        instrument.get("runtime_rng_seed_semantics") == RUNTIME_RNG_SEED_SEMANTICS,
        "instrument runtime RNG seed semantics are invalid",
    )

    source_bundle = instrument.get("source_bundle")
    _require(
        isinstance(source_bundle, Mapping),
        "instrument requires a repository source bundle; git commit alone is insufficient",
    )
    validate_source_bundle(source_bundle, repo_root=repo_root)
    source_digest = _validate_sha256(
        instrument.get("source_bundle_sha256"),
        "instrument.source_bundle_sha256",
    )
    _require(
        source_digest == source_bundle[SOURCE_BUNDLE_DIGEST_FIELD],
        "instrument.source_bundle_sha256 does not bind source_bundle",
    )
    self_digest = _validate_sha256(
        instrument.get(INSTRUMENT_DIGEST_FIELD),
        f"instrument.{INSTRUMENT_DIGEST_FIELD}",
    )
    _require(
        self_digest == canonical_sha256(instrument, digest_field=INSTRUMENT_DIGEST_FIELD),
        f"instrument.{INSTRUMENT_DIGEST_FIELD} mismatch",
    )


def episode_instrument_sha256(instrument: Mapping[str, Any]) -> str:
    """Return the digest that scientific episode records must carry.

    The embedded self-digest excludes only its own field.  The episode binding
    follows :func:`atlas._validate_scientific_instrument` and hashes the entire
    frozen mapping, including that self-digest.
    """

    validate_scientific_instrument(instrument)
    return canonical_sha256(instrument)


def _normalize_cell_coordinates(
    raw: Mapping[str, Any],
    paths: Sequence[str],
    *,
    name: str,
) -> dict[str, Any]:
    normalized = _canonical_mapping(raw, name)
    for dotted_path in paths:
        components = dotted_path.split(".")
        parent: Any = normalized
        for component in components[:-1]:
            if not isinstance(parent, dict) or component not in parent:
                parent = None
                break
            parent = parent[component]
        leaf = components[-1]
        if isinstance(parent, dict) and leaf in parent:
            parent[leaf] = {
                "kind": "lace_cell_coordinate",
                "path": dotted_path,
            }
    return normalized


def _measurement_family_projection(instrument: Mapping[str, Any]) -> dict[str, Any]:
    recorder_common = _normalize_cell_coordinates(
        instrument["recorder_config"],
        (
            *RECORDER_CONFIG_CELL_COORDINATE_PATHS,
            *RECORDER_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS,
        ),
        name="recorder_config",
    )
    resolved_common = _normalize_cell_coordinates(
        instrument["resolved_hydra_config"],
        (
            *RESOLVED_CONFIG_CELL_COORDINATE_PATHS,
            *RESOLVED_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS,
        ),
        name="resolved_hydra_config",
    )
    environment_common = _canonical_mapping(
        instrument["environment_fingerprint"],
        "environment_fingerprint",
    )
    cache_environment = environment_common.get("scientific_cache_environment")
    _require(
        isinstance(cache_environment, dict)
        and set(cache_environment) == set(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS),
        "environment fingerprint cache-coordinate fields are invalid",
    )
    _require(
        environment_common.get("scientific_cache_environment_semantics")
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "environment fingerprint cache semantics are invalid",
    )
    for key in SCIENTIFIC_CACHE_ENVIRONMENT_KEYS:
        value = cache_environment[key]
        _require(
            isinstance(value, str)
            and bool(value)
            and Path(value).is_absolute()
            and str(Path(value)) == value,
            f"environment fingerprint cache path {key} is not canonical absolute text",
        )
        cache_environment[key] = {
            "kind": "lace_cell_coordinate",
            "environment_key": key,
        }
    process_argv = environment_common.get("process_argv")
    _require(
        isinstance(process_argv, list)
        and bool(process_argv)
        and all(isinstance(argument, str) and argument for argument in process_argv),
        "environment fingerprint process_argv is invalid",
    )
    environment_common["process_argv"] = {
        "kind": "lace_cell_coordinate",
        "path": "process_argv",
    }
    family: dict[str, Any] = {
        "kind": MEASUREMENT_FAMILY_KIND,
        "schema_version": MEASUREMENT_FAMILY_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "schedule_sha256": instrument["schedule_sha256"],
        "probe_thresholds_sha256": instrument["probe_thresholds_sha256"],
        "recorder_config_common_sha256": canonical_sha256(recorder_common),
        "resolved_hydra_config_common_sha256": canonical_sha256(resolved_common),
        "environment_sha256": canonical_sha256(environment_common),
        "sensor_semantics_sha256": instrument["sensor_semantics_sha256"],
        "termination_predicates_sha256": instrument["termination_predicates_sha256"],
        "score_window_config_sha256": instrument["score_window_config_sha256"],
        "domain_randomization_config_sha256": instrument["domain_randomization_config_sha256"],
        "git_commit": instrument["git_commit"],
        "source_bundle_sha256": instrument["source_bundle_sha256"],
        "domain_randomization_seed_semantics": instrument["domain_randomization_seed_semantics"],
        "runtime_rng_seed_semantics": instrument["runtime_rng_seed_semantics"],
        "termination_semantics": instrument["termination_semantics"],
        "termination_multi_hot_available": instrument["termination_multi_hot_available"],
        "resolved_config_cell_coordinate_paths": list(RESOLVED_CONFIG_CELL_COORDINATE_PATHS),
        "resolved_config_noncausal_bookkeeping_paths": list(
            RESOLVED_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS
        ),
        "recorder_config_cell_coordinate_paths": list(RECORDER_CONFIG_CELL_COORDINATE_PATHS),
        "recorder_config_noncausal_bookkeeping_paths": list(
            RECORDER_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS
        ),
        "environment_cache_coordinate_keys": list(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS),
        "environment_cache_normalization_semantics": ENVIRONMENT_CACHE_NORMALIZATION_SEMANTICS,
        "environment_cell_coordinate_paths": list(ENVIRONMENT_CELL_COORDINATE_PATHS),
    }
    family[MEASUREMENT_FAMILY_DIGEST_FIELD] = canonical_sha256(
        family,
        digest_field=MEASUREMENT_FAMILY_DIGEST_FIELD,
    )
    return family


def build_measurement_family(instrument: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the common measurement contract from one exact cell instrument."""

    validate_scientific_instrument(instrument)
    family = _measurement_family_projection(instrument)
    validate_measurement_family(family, instruments=[instrument])
    return family


def validate_measurement_family(
    raw: Mapping[str, Any],
    *,
    instruments: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Validate a family and prove each exact cell instrument projects to it."""

    _require(isinstance(raw, Mapping), "measurement_family must be a mapping")
    family = dict(raw)
    _require(
        set(family) == _MEASUREMENT_FAMILY_FIELDS,
        "measurement_family fields do not match schema v1",
    )
    _require(family.get("kind") == MEASUREMENT_FAMILY_KIND, "measurement_family kind is invalid")
    _require(
        family.get("schema_version") == MEASUREMENT_FAMILY_SCHEMA_VERSION,
        "measurement_family schema_version is invalid",
    )
    _require(
        family.get("artifact_mode") == "scientific" and family.get("scientific_use") is True,
        "measurement_family must be scientific",
    )
    for field in (
        "schedule_sha256",
        "probe_thresholds_sha256",
        "recorder_config_common_sha256",
        "resolved_hydra_config_common_sha256",
        "environment_sha256",
        "sensor_semantics_sha256",
        "termination_predicates_sha256",
        "score_window_config_sha256",
        "domain_randomization_config_sha256",
        "source_bundle_sha256",
    ):
        _validate_sha256(family.get(field), f"measurement_family.{field}")
    _validate_git_commit(family.get("git_commit"))
    _require(
        family.get("domain_randomization_seed_semantics") == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "measurement_family DR-seed semantics are invalid",
    )
    _require(
        family.get("runtime_rng_seed_semantics") == RUNTIME_RNG_SEED_SEMANTICS,
        "measurement_family runtime RNG semantics are invalid",
    )
    _require(
        family.get("termination_semantics") == TERMINATION_MULTI_HOT_SEMANTICS
        and family.get("termination_multi_hot_available") is True,
        "measurement_family termination semantics are invalid",
    )
    _require(
        family.get("resolved_config_cell_coordinate_paths")
        == list(RESOLVED_CONFIG_CELL_COORDINATE_PATHS),
        "measurement_family resolved-config normalization rules drifted",
    )
    _require(
        family.get("recorder_config_cell_coordinate_paths")
        == list(RECORDER_CONFIG_CELL_COORDINATE_PATHS),
        "measurement_family recorder-config normalization rules drifted",
    )
    _require(
        family.get("resolved_config_noncausal_bookkeeping_paths")
        == list(RESOLVED_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS),
        "measurement_family resolved-config bookkeeping rules drifted",
    )
    _require(
        family.get("recorder_config_noncausal_bookkeeping_paths")
        == list(RECORDER_CONFIG_NONCAUSAL_BOOKKEEPING_PATHS),
        "measurement_family recorder-config bookkeeping rules drifted",
    )
    _require(
        family.get("environment_cache_coordinate_keys") == list(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS)
        and family.get("environment_cache_normalization_semantics")
        == ENVIRONMENT_CACHE_NORMALIZATION_SEMANTICS,
        "measurement_family environment-cache normalization rules drifted",
    )
    _require(
        family.get("environment_cell_coordinate_paths") == list(ENVIRONMENT_CELL_COORDINATE_PATHS),
        "measurement_family environment cell-coordinate rules drifted",
    )
    declared = _validate_sha256(
        family.get(MEASUREMENT_FAMILY_DIGEST_FIELD),
        f"measurement_family.{MEASUREMENT_FAMILY_DIGEST_FIELD}",
    )
    _require(
        declared == canonical_sha256(family, digest_field=MEASUREMENT_FAMILY_DIGEST_FIELD),
        f"measurement_family.{MEASUREMENT_FAMILY_DIGEST_FIELD} mismatch",
    )
    for index, instrument in enumerate(instruments):
        validate_scientific_instrument(instrument)
        _require(
            _measurement_family_projection(instrument) == family,
            f"cell instrument {index} does not belong to the declared measurement family",
        )


def episode_measurement_family_sha256(family: Mapping[str, Any]) -> str:
    """Return the whole-family digest carried by every scientific episode."""

    validate_measurement_family(family)
    return canonical_sha256(family)
