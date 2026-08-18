"""Immutable pre-outcome commitment for a LACE analysis protocol.

The protocol describes live measurement semantics.  Its self digest alone is
not a preregistration anchor because the protocol and every downstream artifact
could be consistently replaced after outcomes are observed.  This separate
lock is published with exclusive-create semantics before a scientific launch
and pins both the already-locked schedule and the exact external protocol
bytes.  A run must consume this lock, never an uncommitted protocol path.

Filesystem immutability remains an external operational boundary: preserve the
lock in append-only/versioned storage or commit its digest in the experiment
registry before launching a policy rollout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    validate_analysis_protocol,
)
from gear_sonic.research.lace.instrument_runtime import file_sha256
from gear_sonic.research.lace.schedule_batch_loader import load_locked_rollout_schedule
from gear_sonic.research.lace.schema import canonical_sha256

ANALYSIS_PROTOCOL_LOCK_KIND = "lace_analysis_protocol_preregistration_lock"
ANALYSIS_PROTOCOL_LOCK_SCHEMA_VERSION = 3
ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD = "analysis_protocol_lock_sha256"
ANALYSIS_PROTOCOL_LOCK_SEMANTICS = (
    "exclusive_create_before_outcomes_then_external_append_only_or_registry_pinning_v1"
)

_LOCK_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "declared_before_outcomes",
    "commitment_semantics",
    "analysis_protocol",
    "analysis_protocol_preflight",
    "schedule_lock",
    "probe_policy_checkpoint_configs",
    "execution_registry",
    "code_identity",
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
}
_PROTOCOL_BINDING_FIELDS = {
    "path",
    "file_sha256",
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
}
_PREFLIGHT_BINDING_FIELDS = {
    "path",
    "file_sha256",
    "protocol_preflight_receipt_sha256",
}
_SCHEDULE_BINDING_FIELDS = {"path", "file_sha256", "schedule_sha256"}
_CODE_IDENTITY_FIELDS = {"git_commit", "source_bundle_sha256"}
_CHECKPOINT_CONFIG_FIELDS = {
    "probe_policy_id",
    "checkpoint_sha256",
    "config_path",
    "config_file_sha256",
}
_CELL_FIELDS = (
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "repeat_index",
)
_EXECUTION_REGISTRY_FIELDS = {
    "execution_root",
    "runtime_storage_root",
    "attempt_semantics",
    "cells",
}
_EXECUTION_CELL_FIELDS = {
    "cell",
    "cell_token",
    "cell_directory",
    "plan_output_path",
    "rollout_output_path",
    "instrumented_rollout_output_path",
    "instrument_output_path",
    "runtime_handshake_output_path",
    "instrument_binding_output_path",
    "rollout_binding_output_path",
}
EXECUTION_ATTEMPT_SEMANTICS = (
    "one_exclusive_attempt_per_frozen_cell_no_retry_partial_artifacts_are_tombstones_v1"
)


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


def _canonical_file(path_value: str | Path, name: str) -> Path:
    candidate = Path(path_value).expanduser()
    _require(candidate.is_absolute(), f"{name} must be absolute")
    _require(not candidate.is_symlink(), f"{name} may not be a symlink")
    resolved = candidate.resolve()
    _require(candidate == resolved, f"{name} must be canonical")
    _require(resolved.is_file(), f"{name} is missing: {resolved}")
    return resolved


def _canonical_directory(path_value: str | Path, name: str) -> Path:
    candidate = Path(path_value).expanduser()
    _require(candidate.is_absolute(), f"{name} must be absolute")
    _require(not candidate.is_symlink(), f"{name} may not be a symlink")
    resolved = candidate.resolve()
    _require(candidate == resolved, f"{name} must be canonical")
    _require(resolved.is_dir(), f"{name} is missing: {resolved}")
    return resolved


def _schedule_cells(schedule_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in schedule_manifest["rollouts"]:
        cell = {field: row[field] for field in _CELL_FIELDS}
        token = canonical_sha256(cell)
        if token not in seen:
            seen.add(token)
            cells.append(cell)
    return cells


def _execution_registry(
    schedule_manifest: Mapping[str, Any],
    *,
    execution_root: Path,
    runtime_storage_root: Path,
) -> dict[str, Any]:
    cells = []
    for index, cell in enumerate(_schedule_cells(schedule_manifest)):
        token = canonical_sha256(
            {"schedule_sha256": schedule_manifest["schedule_sha256"], "cell": cell}
        )
        cell_directory = execution_root / f"cell-{index:03d}-{token[:16]}"
        instrument = cell_directory / "instrument.json"
        rollout = cell_directory / "cell.jsonl"
        cells.append(
            {
                "cell": cell,
                "cell_token": token,
                "cell_directory": str(cell_directory),
                "plan_output_path": str(cell_directory / "plan.json"),
                "rollout_output_path": str(rollout),
                "instrumented_rollout_output_path": str(cell_directory / "cell.instrumented.jsonl"),
                "instrument_output_path": str(instrument),
                "runtime_handshake_output_path": str(
                    cell_directory / "instrument.runtime-handshake.json"
                ),
                "instrument_binding_output_path": str(
                    cell_directory / "instrument.runtime-binding.json"
                ),
                "rollout_binding_output_path": str(
                    cell_directory / "instrument.rollout-binding.json"
                ),
            }
        )
    return {
        "execution_root": str(execution_root),
        "runtime_storage_root": str(runtime_storage_root),
        "attempt_semantics": EXECUTION_ATTEMPT_SEMANTICS,
        "cells": cells,
    }


def build_analysis_protocol_lock(
    *,
    analysis_protocol_path: str | Path,
    analysis_protocol_preflight_receipt_path: str | Path,
    schedule_lock_path: str | Path,
    checkpoint_config_paths: Mapping[str, str | Path],
    execution_root: str | Path,
    runtime_storage_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a self-hashed lock over exact pre-existing protocol/schedule bytes."""

    protocol_path = _canonical_file(analysis_protocol_path, "analysis protocol path")
    preflight_receipt_path = _canonical_file(
        analysis_protocol_preflight_receipt_path,
        "analysis protocol preflight receipt path",
    )
    schedule_path = _canonical_file(schedule_lock_path, "schedule lock path")
    from gear_sonic.research.lace.instrument_runtime import _load_json_object

    protocol = _load_json_object(protocol_path, "analysis protocol")
    preflight_receipt = _load_json_object(
        preflight_receipt_path,
        "analysis protocol preflight receipt",
    )
    from gear_sonic.research.lace.protocol_preflight import (
        PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD,
        validate_protocol_preflight_receipt,
    )

    validate_protocol_preflight_receipt(
        preflight_receipt,
        repo_root=repo_root,
        verify_files=True,
    )
    locked_schedule = load_locked_rollout_schedule(schedule_path, repo_root=repo_root)
    validate_analysis_protocol(
        protocol,
        schedule_manifest=locked_schedule.manifest,
        split_manifest=locked_schedule.split_manifest,
        reference_length_inventory=locked_schedule.reference_length_inventory,
    )
    _require(
        preflight_receipt["analysis_protocol"]
        == {
            "path": str(protocol_path),
            "file_sha256": file_sha256(protocol_path),
            "self_sha256": protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        }
        and preflight_receipt["schedule_sha256"] == locked_schedule.manifest["schedule_sha256"]
        and preflight_receipt["git_commit"] == protocol["git_commit"]
        and preflight_receipt["source_bundle_sha256"] == protocol["source_bundle_sha256"],
        "analysis protocol does not match its live zero-outcome preflight receipt",
    )
    _require(
        isinstance(checkpoint_config_paths, Mapping),
        "checkpoint_config_paths must be a mapping by probe policy id",
    )
    scheduled_policies = locked_schedule.manifest["probe_policies"]
    expected_policy_ids = [str(policy["id"]) for policy in scheduled_policies]
    _require(
        set(checkpoint_config_paths) == set(expected_policy_ids),
        "checkpoint_config_paths must exactly cover the frozen probe policies",
    )
    checkpoint_configs = []
    for policy in scheduled_policies:
        policy_id = str(policy["id"])
        config_path = _canonical_file(
            checkpoint_config_paths[policy_id],
            f"checkpoint config for policy {policy_id!r}",
        )
        checkpoint_configs.append(
            {
                "probe_policy_id": policy_id,
                "checkpoint_sha256": policy["checkpoint_sha256"],
                "config_path": str(config_path),
                "config_file_sha256": file_sha256(config_path),
            }
        )
    preflight_cell = preflight_receipt["cell"]
    preflight_matches = [
        record
        for record in checkpoint_configs
        if record["probe_policy_id"] == preflight_cell["probe_policy_id"]
    ]
    _require(
        len(preflight_matches) == 1
        and preflight_matches[0]["checkpoint_sha256"]
        == preflight_cell["checkpoint_sha256"]
        == preflight_receipt["checkpoint_bundle"]["checkpoint_sha256"]
        and preflight_matches[0]["config_path"]
        == preflight_receipt["checkpoint_bundle"]["config_path"]
        and preflight_matches[0]["config_file_sha256"]
        == preflight_receipt["checkpoint_bundle"]["config_sha256"],
        "live preflight checkpoint/config differs from the frozen policy bundle",
    )
    execution_directory = _canonical_directory(execution_root, "execution root")
    runtime_directory = _canonical_directory(runtime_storage_root, "runtime storage root")
    _require(
        not any(execution_directory.iterdir()),
        "execution root must be empty when the preregistration lock is built",
    )
    _require(
        not any(runtime_directory.iterdir()),
        "runtime storage root must be empty when the preregistration lock is built",
    )
    _require(
        execution_directory != runtime_directory
        and not execution_directory.is_relative_to(runtime_directory)
        and not runtime_directory.is_relative_to(execution_directory),
        "execution and runtime-storage roots must not overlap",
    )
    lock: dict[str, Any] = {
        "kind": ANALYSIS_PROTOCOL_LOCK_KIND,
        "schema_version": ANALYSIS_PROTOCOL_LOCK_SCHEMA_VERSION,
        "scientific_use": True,
        "declared_before_outcomes": True,
        "commitment_semantics": ANALYSIS_PROTOCOL_LOCK_SEMANTICS,
        "analysis_protocol": {
            "path": str(protocol_path),
            "file_sha256": file_sha256(protocol_path),
            ANALYSIS_PROTOCOL_DIGEST_FIELD: protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        },
        "analysis_protocol_preflight": {
            "path": str(preflight_receipt_path),
            "file_sha256": file_sha256(preflight_receipt_path),
            PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD: preflight_receipt[
                PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD
            ],
        },
        "schedule_lock": {
            "path": str(schedule_path),
            "file_sha256": file_sha256(schedule_path),
            "schedule_sha256": locked_schedule.manifest["schedule_sha256"],
        },
        "probe_policy_checkpoint_configs": checkpoint_configs,
        "execution_registry": _execution_registry(
            locked_schedule.manifest,
            execution_root=execution_directory,
            runtime_storage_root=runtime_directory,
        ),
        "code_identity": {
            "git_commit": protocol["git_commit"],
            "source_bundle_sha256": protocol["source_bundle_sha256"],
        },
    }
    lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD] = canonical_sha256(
        lock,
        digest_field=ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    )
    validate_analysis_protocol_lock(lock, repo_root=repo_root, verify_files=True)
    return lock


def validate_analysis_protocol_lock(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_files: bool = True,
) -> None:
    _require(isinstance(raw, Mapping), "analysis protocol lock must be a mapping")
    lock = dict(raw)
    _require(set(lock) == _LOCK_FIELDS, "analysis protocol lock fields are invalid")
    _require(
        lock.get("kind") == ANALYSIS_PROTOCOL_LOCK_KIND
        and lock.get("schema_version") == ANALYSIS_PROTOCOL_LOCK_SCHEMA_VERSION
        and lock.get("scientific_use") is True
        and lock.get("declared_before_outcomes") is True
        and lock.get("commitment_semantics") == ANALYSIS_PROTOCOL_LOCK_SEMANTICS,
        "analysis protocol lock identity is invalid",
    )
    protocol_binding = lock.get("analysis_protocol")
    preflight_binding = lock.get("analysis_protocol_preflight")
    schedule_binding = lock.get("schedule_lock")
    code_identity = lock.get("code_identity")
    checkpoint_configs = lock.get("probe_policy_checkpoint_configs")
    execution_registry = lock.get("execution_registry")
    _require(
        isinstance(protocol_binding, Mapping) and set(protocol_binding) == _PROTOCOL_BINDING_FIELDS,
        "analysis protocol lock protocol binding is invalid",
    )
    _require(
        isinstance(preflight_binding, Mapping)
        and set(preflight_binding) == _PREFLIGHT_BINDING_FIELDS,
        "analysis protocol lock preflight binding is invalid",
    )
    _require(
        isinstance(schedule_binding, Mapping) and set(schedule_binding) == _SCHEDULE_BINDING_FIELDS,
        "analysis protocol lock schedule binding is invalid",
    )
    _require(
        isinstance(code_identity, Mapping) and set(code_identity) == _CODE_IDENTITY_FIELDS,
        "analysis protocol lock code identity is invalid",
    )
    _require(
        isinstance(checkpoint_configs, list) and checkpoint_configs,
        "analysis protocol lock checkpoint configs are missing",
    )
    _require(
        isinstance(execution_registry, Mapping)
        and set(execution_registry) == _EXECUTION_REGISTRY_FIELDS
        and execution_registry.get("attempt_semantics") == EXECUTION_ATTEMPT_SEMANTICS,
        "analysis protocol lock execution registry is invalid",
    )
    execution_cells = execution_registry.get("cells")
    _require(
        isinstance(execution_cells, list) and execution_cells,
        "analysis protocol lock execution cells are missing",
    )
    for index, raw_cell in enumerate(execution_cells):
        _require(
            isinstance(raw_cell, Mapping) and set(raw_cell) == _EXECUTION_CELL_FIELDS,
            f"analysis protocol lock execution cell {index} is invalid",
        )
        cell = raw_cell.get("cell")
        _require(
            isinstance(cell, Mapping) and set(cell) == set(_CELL_FIELDS),
            f"analysis protocol lock execution cell {index} coordinates are invalid",
        )
        _sha256(raw_cell.get("cell_token"), f"execution cell {index} token")
    seen_policy_ids: set[str] = set()
    normalized_checkpoint_configs: list[Mapping[str, Any]] = []
    for index, raw_config in enumerate(checkpoint_configs):
        _require(
            isinstance(raw_config, Mapping) and set(raw_config) == _CHECKPOINT_CONFIG_FIELDS,
            f"analysis protocol lock checkpoint config {index} is invalid",
        )
        policy_id = raw_config.get("probe_policy_id")
        _require(
            isinstance(policy_id, str) and policy_id and policy_id not in seen_policy_ids,
            f"analysis protocol lock checkpoint config {index} policy id is invalid",
        )
        seen_policy_ids.add(policy_id)
        _sha256(raw_config.get("checkpoint_sha256"), f"policy {policy_id} checkpoint")
        _sha256(raw_config.get("config_file_sha256"), f"policy {policy_id} config file")
        _require(
            isinstance(raw_config.get("config_path"), str) and bool(raw_config["config_path"]),
            f"policy {policy_id} config path is invalid",
        )
        normalized_checkpoint_configs.append(raw_config)
    for value, name in (
        (protocol_binding["file_sha256"], "analysis protocol file"),
        (protocol_binding[ANALYSIS_PROTOCOL_DIGEST_FIELD], "analysis protocol"),
        (preflight_binding["file_sha256"], "analysis protocol preflight file"),
        (
            preflight_binding["protocol_preflight_receipt_sha256"],
            "analysis protocol preflight receipt",
        ),
        (schedule_binding["file_sha256"], "schedule lock file"),
        (schedule_binding["schedule_sha256"], "schedule"),
        (code_identity["source_bundle_sha256"], "source bundle"),
        (lock.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD), "analysis protocol lock"),
    ):
        _sha256(value, name)
    git_commit = code_identity.get("git_commit")
    _require(isinstance(git_commit, str) and len(git_commit) == 40, "code git commit invalid")
    try:
        int(git_commit, 16)
    except ValueError as error:
        raise ValueError("code git commit must be hexadecimal") from error
    _require(
        lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
        == canonical_sha256(lock, digest_field=ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD),
        "analysis_protocol_lock_sha256 mismatch",
    )
    if not verify_files:
        return
    protocol_path = _canonical_file(protocol_binding["path"], "analysis protocol path")
    preflight_path = _canonical_file(
        preflight_binding["path"],
        "analysis protocol preflight receipt path",
    )
    schedule_path = _canonical_file(schedule_binding["path"], "schedule lock path")
    _require(
        file_sha256(protocol_path) == protocol_binding["file_sha256"],
        "analysis protocol lock file digest drifted",
    )
    _require(
        file_sha256(preflight_path) == preflight_binding["file_sha256"],
        "analysis protocol preflight receipt file digest drifted",
    )
    _require(
        file_sha256(schedule_path) == schedule_binding["file_sha256"],
        "schedule lock file digest drifted",
    )
    from gear_sonic.research.lace.instrument_runtime import _load_json_object

    protocol = _load_json_object(protocol_path, "analysis protocol")
    preflight_receipt = _load_json_object(
        preflight_path,
        "analysis protocol preflight receipt",
    )
    from gear_sonic.research.lace.protocol_preflight import (
        PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD,
        validate_protocol_preflight_receipt,
    )

    validate_protocol_preflight_receipt(
        preflight_receipt,
        repo_root=repo_root,
        verify_files=True,
    )
    locked_schedule = load_locked_rollout_schedule(schedule_path, repo_root=repo_root)
    validate_analysis_protocol(
        protocol,
        schedule_manifest=locked_schedule.manifest,
        split_manifest=locked_schedule.split_manifest,
        reference_length_inventory=locked_schedule.reference_length_inventory,
    )
    _require(
        protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD] == protocol_binding[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        and protocol["git_commit"] == code_identity["git_commit"]
        and protocol["source_bundle_sha256"] == code_identity["source_bundle_sha256"],
        "analysis protocol lock code/protocol identity drifted",
    )
    _require(
        preflight_receipt[PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD]
        == preflight_binding[PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD]
        and preflight_receipt["analysis_protocol"]
        == {
            "path": str(protocol_path),
            "file_sha256": protocol_binding["file_sha256"],
            "self_sha256": protocol_binding[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        }
        and preflight_receipt["schedule_sha256"] == schedule_binding["schedule_sha256"],
        "analysis protocol lock live-preflight provenance drifted",
    )
    _require(
        str(locked_schedule.lock_path) == str(schedule_path)
        and locked_schedule.manifest["schedule_sha256"] == schedule_binding["schedule_sha256"],
        "analysis protocol lock schedule identity drifted",
    )
    expected_policies = locked_schedule.manifest["probe_policies"]
    _require(
        [record["probe_policy_id"] for record in normalized_checkpoint_configs]
        == [policy["id"] for policy in expected_policies],
        "analysis protocol lock checkpoint config order/coverage drifted from schedule",
    )
    for record, policy in zip(
        normalized_checkpoint_configs,
        expected_policies,
        strict=True,
    ):
        _require(
            record["checkpoint_sha256"] == policy["checkpoint_sha256"],
            f"policy {policy['id']} checkpoint/config binding drifted from schedule",
        )
        if verify_files:
            config_path = _canonical_file(
                record["config_path"],
                f"checkpoint config for policy {policy['id']!r}",
            )
            _require(
                file_sha256(config_path) == record["config_file_sha256"],
                f"checkpoint config bytes drifted for policy {policy['id']!r}",
            )
    preflight_config_matches = [
        record
        for record in normalized_checkpoint_configs
        if record["probe_policy_id"] == preflight_receipt["cell"]["probe_policy_id"]
    ]
    _require(
        len(preflight_config_matches) == 1
        and preflight_config_matches[0]["checkpoint_sha256"]
        == preflight_receipt["cell"]["checkpoint_sha256"]
        == preflight_receipt["checkpoint_bundle"]["checkpoint_sha256"]
        and preflight_config_matches[0]["config_path"]
        == preflight_receipt["checkpoint_bundle"]["config_path"]
        and preflight_config_matches[0]["config_file_sha256"]
        == preflight_receipt["checkpoint_bundle"]["config_sha256"],
        "analysis protocol lock preflight checkpoint/config binding drifted",
    )
    execution_root = _canonical_directory(
        execution_registry["execution_root"],
        "execution registry root",
    )
    runtime_storage_root = _canonical_directory(
        execution_registry["runtime_storage_root"],
        "execution registry runtime storage root",
    )
    _require(
        dict(execution_registry)
        == _execution_registry(
            locked_schedule.manifest,
            execution_root=execution_root,
            runtime_storage_root=runtime_storage_root,
        ),
        "analysis protocol lock execution registry is not the deterministic schedule layout",
    )


def checkpoint_config_binding(
    lock: Mapping[str, Any],
    *,
    probe_policy_id: str,
    checkpoint_sha256: str,
) -> dict[str, str]:
    """Return the one pre-outcome config binding for a scheduled policy."""

    configs = lock.get("probe_policy_checkpoint_configs")
    _require(isinstance(configs, list), "analysis protocol lock checkpoint configs missing")
    matches = [
        record
        for record in configs
        if isinstance(record, Mapping) and record.get("probe_policy_id") == probe_policy_id
    ]
    _require(len(matches) == 1, "probe policy has no unique checkpoint config binding")
    record = matches[0]
    _require(
        record.get("checkpoint_sha256") == checkpoint_sha256,
        "probe policy checkpoint digest differs from its preregistered config binding",
    )
    return {
        "config_path": str(record["config_path"]),
        "config_sha256": str(record["config_file_sha256"]),
    }


def execution_cell_binding(
    lock: Mapping[str, Any],
    *,
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only preregistered output layout for an execution cell."""

    normalized = {field: cell.get(field) for field in _CELL_FIELDS}
    registry = lock.get("execution_registry")
    _require(isinstance(registry, Mapping), "analysis protocol lock execution registry missing")
    cells = registry.get("cells")
    _require(isinstance(cells, list), "analysis protocol lock execution cells missing")
    matches = [record for record in cells if record.get("cell") == normalized]
    _require(len(matches) == 1, "execution cell has no unique preregistered output binding")
    return dict(matches[0])


def load_analysis_protocol_lock(
    path_value: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Load one canonical lock and return it with its exact protocol payload."""

    lock_path = _canonical_file(path_value, "analysis protocol lock path")
    from gear_sonic.research.lace.instrument_runtime import _load_json_object

    lock = _load_json_object(lock_path, "analysis protocol lock")
    validate_analysis_protocol_lock(lock, repo_root=repo_root, verify_files=True)
    protocol_path = Path(lock["analysis_protocol"]["path"])
    protocol = _load_json_object(protocol_path, "analysis protocol")
    return lock_path, lock, protocol_path, protocol
