"""Fail-closed runtime binding for frozen LACE RQ1 motion distributions.

RQ1 changes motion exposure over a fixed, representation-blind source panel.
The probabilities are reconstructed from a pre-outcome artifact lock before
training and are never solved, tuned, or inferred by the runtime sampler.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.intervention_plan import (
    INTERVENTION_PLAN_DIGEST_FIELD,
    INTERVENTION_PLAN_KIND,
    INTERVENTION_PLAN_SCHEMA_VERSION,
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    validate_intervention_plan,
    validate_intervention_protocol,
)
from gear_sonic.research.lace.panels import validate_panel_manifest
from gear_sonic.research.lace.schema import canonical_sha256, validate_split_manifest

FIXED_DISTRIBUTION_BINDING_KIND = "lace_fixed_motion_distribution_runtime_binding"
FIXED_DISTRIBUTION_BINDING_SCHEMA_VERSION = 2
FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD = "runtime_binding_sha256"
FIXED_DISTRIBUTION_STATE_KEY = "lace_fixed_distribution"
FIXED_DISTRIBUTION_CHECKPOINT_KIND = "lace_fixed_motion_distribution_checkpoint_state"
FIXED_DISTRIBUTION_CHECKPOINT_SCHEMA_VERSION = 1
FIXED_DISTRIBUTION_DRAW_REPORT_KIND = "lace_fixed_motion_distribution_draw_report"
FIXED_DISTRIBUTION_DRAW_REPORT_SCHEMA_VERSION = 1
FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD = "draw_report_sha256"
INTERVENTION_PLAN_LOCK_KIND = "lace_rq1_intervention_plan_artifact_lock"
INTERVENTION_PLAN_LOCK_SCHEMA_VERSION = 1
BASE_DISTRIBUTION_ID = "base"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FIELDS = {
    "enable",
    "plan_lock_path",
    "plan_lock_file_sha256",
    "distribution_id",
}
_LOCK_FIELDS = {
    "artifact",
    "declared_before_transfer_outcomes",
    "inputs",
    "kind",
    "schema_version",
    "scientific_use",
}
_LOCK_ARTIFACT_FIELDS = {
    "file_sha256",
    INTERVENTION_PLAN_DIGEST_FIELD,
    "path",
}
_LOCK_INPUT_FIELDS = {
    "split_manifest",
    "split_manifest_sha256",
    "split_sha256",
    "split_selection_sha256",
    "panel_manifest",
    "panel_manifest_sha256",
    "panel_sha256",
    "intervention_protocol",
    "intervention_protocol_sha256",
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
}
_BINDING_FIELDS = {
    "kind",
    "schema_version",
    "frozen",
    "plan_lock_path",
    "plan_lock_file_sha256",
    "plan_path",
    "plan_file_sha256",
    INTERVENTION_PLAN_DIGEST_FIELD,
    "split_manifest_path",
    "split_manifest_file_sha256",
    "split_sha256",
    "split_selection_sha256",
    "panel_manifest_path",
    "panel_manifest_file_sha256",
    "panel_sha256",
    "intervention_protocol_path",
    "intervention_protocol_file_sha256",
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    "fixed_distribution_config_sha256",
    "arm_id",
    "distribution_id",
    "distribution_kind",
    "distribution_sha256",
    "motion_keys",
    "motion_count",
    "motion_order_sha256",
    "probabilities",
    "probability_dtype",
    "resident_identity_dtype",
    "adaptive_sampling_enabled",
    "all_motions_loaded",
    "accounting_scope",
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
}
_CHECKPOINT_FIELDS = {
    "kind",
    "schema_version",
    "runtime_binding",
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    "fixed_distribution_config_sha256",
    "arm_id",
    INTERVENTION_PLAN_DIGEST_FIELD,
    "plan_lock_file_sha256",
    "probabilities",
    "draw_counts",
    "total_draw_count",
    FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
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


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"locked JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_path(value: Any, name: str, *, allow_repo_relative: bool) -> Path:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty string")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        absolute = candidate
    else:
        _require(allow_repo_relative, f"{name} must be absolute")
        _require(
            candidate.parts and all(part not in ("", ".", "..") for part in candidate.parts),
            f"{name} must be a canonical repository-relative path",
        )
        _require(value == candidate.as_posix(), f"{name} must use canonical POSIX syntax")
        absolute = _REPO_ROOT / candidate
    _require(not absolute.is_symlink(), f"{name} may not be a symlink")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} is missing: {absolute}") from error
    _require(absolute == resolved, f"{name} must be canonical")
    _require(resolved.is_file(), f"{name} is not a regular file: {resolved}")
    if not candidate.is_absolute():
        _require(
            resolved.is_relative_to(_REPO_ROOT),
            f"{name} escapes the repository root",
        )
    return resolved


def _read_json_once(path: Path, name: str) -> tuple[dict[str, Any], str]:
    """Hash and parse one immutable byte buffer from one no-follow file handle."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"could not open {name} {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{name} must be a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload_bytes = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"{name} changed while it was being read")
    _require(len(payload_bytes) == before.st_size, f"{name} byte count changed while reading")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    try:
        decoded = payload_bytes.decode("utf-8")
        payload = json.loads(decoded, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not parse {name} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload, digest


def _read_locked_reference(
    section: Mapping[str, Any],
    *,
    path_field: str,
    digest_field: str,
    name: str,
) -> tuple[dict[str, Any], Path, str]:
    path = _canonical_path(section.get(path_field), path_field, allow_repo_relative=True)
    expected_digest = _sha256(section.get(digest_field), digest_field)
    payload, actual_digest = _read_json_once(path, name)
    _require(actual_digest == expected_digest, f"{name} bytes drifted from its lock")
    return payload, path, actual_digest


def _load_and_validate_plan_lock(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    lock_path = _canonical_path(
        config.get("plan_lock_path"),
        "plan_lock_path",
        allow_repo_relative=False,
    )
    expected_lock_digest = _sha256(
        config.get("plan_lock_file_sha256"),
        "plan_lock_file_sha256",
    )
    lock, actual_lock_digest = _read_json_once(lock_path, "intervention-plan lock")
    _require(actual_lock_digest == expected_lock_digest, "intervention-plan lock bytes drifted")
    _require(set(lock) == _LOCK_FIELDS, "intervention-plan lock fields are invalid")
    _require(lock.get("kind") == INTERVENTION_PLAN_LOCK_KIND, "plan lock kind mismatch")
    _require(
        lock.get("schema_version") == INTERVENTION_PLAN_LOCK_SCHEMA_VERSION,
        "plan lock schema version mismatch",
    )
    _require(lock.get("scientific_use") is True, "plan lock must be scientific")
    _require(
        lock.get("declared_before_transfer_outcomes") is True,
        "plan lock must be declared before transfer outcomes",
    )
    artifact = lock.get("artifact")
    inputs = lock.get("inputs")
    _require(isinstance(artifact, Mapping), "plan lock artifact must be a mapping")
    _require(isinstance(inputs, Mapping), "plan lock inputs must be a mapping")
    _require(set(artifact) == _LOCK_ARTIFACT_FIELDS, "plan lock artifact fields are invalid")
    _require(set(inputs) == _LOCK_INPUT_FIELDS, "plan lock input fields are invalid")

    plan, plan_path, plan_file_digest = _read_locked_reference(
        artifact,
        path_field="path",
        digest_field="file_sha256",
        name="intervention plan",
    )
    split, split_path, split_file_digest = _read_locked_reference(
        inputs,
        path_field="split_manifest",
        digest_field="split_manifest_sha256",
        name="split manifest",
    )
    panels, panel_path, panel_file_digest = _read_locked_reference(
        inputs,
        path_field="panel_manifest",
        digest_field="panel_manifest_sha256",
        name="panel manifest",
    )
    protocol, protocol_path, protocol_file_digest = _read_locked_reference(
        inputs,
        path_field="intervention_protocol",
        digest_field="intervention_protocol_sha256",
        name="intervention protocol",
    )

    expected_plan_digest = _sha256(
        artifact.get(INTERVENTION_PLAN_DIGEST_FIELD),
        f"artifact.{INTERVENTION_PLAN_DIGEST_FIELD}",
    )
    _require(
        plan.get(INTERVENTION_PLAN_DIGEST_FIELD) == expected_plan_digest,
        "locked intervention-plan digest does not match the artifact",
    )
    locked_split_digest = _sha256(inputs.get("split_sha256"), "inputs.split_sha256")
    locked_selection_digest = _sha256(
        inputs.get("split_selection_sha256"),
        "inputs.split_selection_sha256",
    )
    locked_panel_digest = _sha256(inputs.get("panel_sha256"), "inputs.panel_sha256")
    locked_protocol_digest = _sha256(
        inputs.get(INTERVENTION_PROTOCOL_DIGEST_FIELD),
        f"inputs.{INTERVENTION_PROTOCOL_DIGEST_FIELD}",
    )
    _require(split.get("split_sha256") == locked_split_digest, "locked split digest mismatch")
    _require(
        split.get("selection_sha256") == locked_selection_digest,
        "locked split-selection digest mismatch",
    )
    _require(panels.get("panel_sha256") == locked_panel_digest, "locked panel digest mismatch")
    _require(
        protocol.get(INTERVENTION_PROTOCOL_DIGEST_FIELD) == locked_protocol_digest,
        "locked intervention-protocol digest mismatch",
    )
    try:
        validate_split_manifest(split, verify_digest=True)
        validate_panel_manifest(panels, split_manifest=split, verify_digest=True)
        validate_intervention_protocol(protocol, verify_digest=True)
        validate_intervention_plan(
            plan,
            split_manifest=split,
            panel_manifest=panels,
            protocol=protocol,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"intervention-plan lock deep validation failed: {error}") from error

    provenance = {
        "plan_lock_path": str(lock_path),
        "plan_lock_file_sha256": actual_lock_digest,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_file_digest,
        INTERVENTION_PLAN_DIGEST_FIELD: expected_plan_digest,
        "split_manifest_path": str(split_path),
        "split_manifest_file_sha256": split_file_digest,
        "split_sha256": locked_split_digest,
        "split_selection_sha256": locked_selection_digest,
        "panel_manifest_path": str(panel_path),
        "panel_manifest_file_sha256": panel_file_digest,
        "panel_sha256": locked_panel_digest,
        "intervention_protocol_path": str(protocol_path),
        "intervention_protocol_file_sha256": protocol_file_digest,
        INTERVENTION_PROTOCOL_DIGEST_FIELD: locked_protocol_digest,
    }
    return plan, lock, provenance


def _probability_vector(raw: Any, expected_count: int, name: str) -> tuple[float, ...]:
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
        f"{name} must be a sequence",
    )
    _require(len(raw) == expected_count, f"{name} length does not match motion count")
    probabilities: list[float] = []
    for index, value in enumerate(raw):
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name}[{index}] must be numeric",
        )
        probability = float(value)
        _require(
            math.isfinite(probability) and probability > 0.0,
            f"{name}[{index}] must be finite and strictly positive",
        )
        probabilities.append(probability)
    _require(abs(math.fsum(probabilities) - 1.0) <= 1e-12, f"{name} must sum to one")
    return tuple(probabilities)


def _validate_plan_identity(plan: Mapping[str, Any]) -> tuple[str, ...]:
    _require(plan.get("kind") == INTERVENTION_PLAN_KIND, "intervention plan kind mismatch")
    _require(
        plan.get("schema_version") == INTERVENTION_PLAN_SCHEMA_VERSION,
        "intervention plan schema version mismatch",
    )
    _require(plan.get("frozen") is True, "intervention plan must be frozen")
    _require(plan.get("scientific_use") is True, "intervention plan must be scientific")
    raw_keys = plan.get("motion_keys")
    _require(isinstance(raw_keys, list) and raw_keys, "intervention plan motion_keys missing")
    _require(all(isinstance(value, str) and value for value in raw_keys), "motion key invalid")
    motion_keys = tuple(raw_keys)
    _require(len(set(motion_keys)) == len(motion_keys), "motion keys must be unique")
    _require(tuple(sorted(motion_keys)) == motion_keys, "motion keys must be canonically sorted")
    _require(plan.get("motion_count") == len(motion_keys), "motion_count mismatch")
    _require(
        plan.get("motion_order_sha256") == canonical_sha256({"motion_keys": list(motion_keys)}),
        "motion order digest mismatch",
    )
    return motion_keys


def _select_probabilities(
    plan: Mapping[str, Any],
    distribution_id: str,
    motion_keys: tuple[str, ...],
) -> tuple[tuple[float, ...], str, str]:
    base = _probability_vector(plan.get("base_probabilities"), len(motion_keys), "base")
    expected_base_digest = canonical_sha256(
        {"motion_keys": list(motion_keys), "probabilities": list(base)}
    )
    _require(
        plan.get("base_distribution_sha256") == expected_base_digest,
        "base distribution digest mismatch",
    )
    if distribution_id == BASE_DISTRIBUTION_ID:
        return base, "base", expected_base_digest

    rows = plan.get("interventions")
    _require(isinstance(rows, list), "intervention rows missing")
    matches = [
        row for row in rows if isinstance(row, Mapping) and row.get("panel_id") == distribution_id
    ]
    _require(len(matches) == 1, f"distribution_id {distribution_id!r} is not a unique panel")
    intervention = matches[0].get("intervention")
    _require(isinstance(intervention, Mapping), "panel intervention payload missing")
    intervention_digest = _sha256(intervention.get("intervention_sha256"), "intervention_sha256")
    _require(
        canonical_sha256(intervention, digest_field="intervention_sha256") == intervention_digest,
        "panel intervention self digest mismatch",
    )
    probabilities = _probability_vector(
        intervention.get("p_plus"), len(motion_keys), "panel p_plus"
    )
    distribution_digest = canonical_sha256(
        {
            "motion_keys": list(motion_keys),
            "probabilities": list(probabilities),
            "panel_id": distribution_id,
            "intervention_sha256": intervention_digest,
        }
    )
    return probabilities, "panel_upweight", distribution_digest


def resolve_fixed_distribution(
    config: Mapping[str, Any],
    *,
    loaded_motion_keys: Sequence[str],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Resolve one exact runtime distribution and its full provenance binding."""

    _require(isinstance(config, Mapping), "fixed distribution config must be a mapping")
    _require(set(config) == _CONFIG_FIELDS, "fixed distribution config fields are invalid")
    _require(config.get("enable") is True, "fixed distribution config is not enabled")
    distribution_id = config.get("distribution_id")
    _require(
        isinstance(distribution_id, str) and distribution_id,
        "distribution_id must be non-empty",
    )
    plan, _lock, provenance = _load_and_validate_plan_lock(config)
    motion_keys = _validate_plan_identity(plan)
    _require(
        isinstance(loaded_motion_keys, Sequence)
        and not isinstance(loaded_motion_keys, (str, bytes)),
        "resident motion keys must be a sequence",
    )
    _require(
        all(isinstance(value, str) and value for value in loaded_motion_keys),
        "resident motion keys must be non-empty strings",
    )
    runtime_keys = tuple(loaded_motion_keys)
    _require(runtime_keys == motion_keys, "resident motion order differs from intervention plan")
    probabilities, distribution_kind, distribution_digest = _select_probabilities(
        plan, distribution_id, motion_keys
    )
    normalized_config = {
        "enable": True,
        "plan_lock_path": provenance["plan_lock_path"],
        "plan_lock_file_sha256": provenance["plan_lock_file_sha256"],
        "distribution_id": distribution_id,
    }
    binding: dict[str, Any] = {
        "kind": FIXED_DISTRIBUTION_BINDING_KIND,
        "schema_version": FIXED_DISTRIBUTION_BINDING_SCHEMA_VERSION,
        "frozen": True,
        **provenance,
        "fixed_distribution_config_sha256": canonical_sha256(normalized_config),
        "arm_id": distribution_id,
        "distribution_id": distribution_id,
        "distribution_kind": distribution_kind,
        "distribution_sha256": distribution_digest,
        "motion_keys": list(motion_keys),
        "motion_count": len(motion_keys),
        "motion_order_sha256": plan["motion_order_sha256"],
        "probabilities": list(probabilities),
        "probability_dtype": "torch.float64",
        "resident_identity_dtype": "torch.int64",
        "adaptive_sampling_enabled": False,
        "all_motions_loaded": True,
        "accounting_scope": "motion_draw_counts_only_v1;occupancy_and_optimizer_updates_runner_required",
    }
    binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD] = canonical_sha256(
        binding,
        digest_field=FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    )
    return probabilities, binding


def validate_fixed_distribution_command_modes(
    *,
    use_paired_motions: bool,
    sample_unique_motions: bool,
    is_evaluating: bool,
    atlas_probe_mode: bool,
    multi_object_mode: bool,
) -> None:
    """Reject command paths that do not draw independently from the frozen arm."""

    forbidden = [
        name
        for name, active in (
            ("paired-motion assignment", use_paired_motions),
            ("unique-motion assignment", sample_unique_motions),
            ("evaluation sequencing", is_evaluating),
            ("atlas assignment", atlas_probe_mode),
            ("multi-object shared-motion assignment", multi_object_mode),
        )
        if active
    ]
    _require(
        not forbidden,
        "RQ1 fixed distribution forbids sampler bypass mode(s): " + ", ".join(forbidden),
    )


def _binding_is_configured(motion_lib: Any) -> bool:
    # This marker is set atomically with installation, after MotionLib's one
    # permitted initial resident load.  It therefore detects a deleted binding
    # without mistaking that initial construction load for an illicit reload.
    return getattr(motion_lib, "_lace_fixed_distribution_required", False) is True


def _validate_binding(binding: Any) -> Mapping[str, Any]:
    _require(isinstance(binding, Mapping), "fixed-distribution binding is missing")
    _require(set(binding) == _BINDING_FIELDS, "fixed-distribution binding fields drifted")
    _require(binding.get("kind") == FIXED_DISTRIBUTION_BINDING_KIND, "binding kind drifted")
    _require(
        binding.get("schema_version") == FIXED_DISTRIBUTION_BINDING_SCHEMA_VERSION,
        "binding schema version drifted",
    )
    _require(binding.get("frozen") is True, "fixed-distribution binding is not frozen")
    binding_digest = _sha256(
        binding.get(FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD),
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    )
    _require(
        binding_digest
        == canonical_sha256(binding, digest_field=FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD),
        "fixed-distribution runtime binding drifted",
    )
    return binding


def _validate_accounting(motion_lib: Any, motion_count: int) -> tuple[list[int], int]:
    import torch

    counts = getattr(motion_lib, "_lace_fixed_distribution_draw_counts", None)
    _require(isinstance(counts, torch.Tensor), "fixed-distribution draw counts are missing")
    _require(counts.dtype == torch.long, "fixed-distribution draw counts must use torch.int64")
    _require(counts.ndim == 1 and counts.numel() == motion_count, "draw-count shape drifted")
    _require(bool(torch.all(counts >= 0).item()), "draw counts must be nonnegative")
    total = getattr(motion_lib, "_lace_fixed_distribution_total_draws", None)
    _require(
        isinstance(total, Integral) and not isinstance(total, bool) and int(total) >= 0,
        "total draw count must be a nonnegative integer",
    )
    count_values = [int(value) for value in counts.detach().cpu().tolist()]
    _require(sum(count_values) == int(total), "draw-count sum does not equal total draws")
    return count_values, int(total)


def assert_fixed_distribution_integrity(motion_lib: Any) -> bool:
    """Revalidate the complete resident sampler state before every draw."""

    binding_raw = getattr(motion_lib, "lace_fixed_distribution_binding", None)
    if binding_raw is None:
        _require(
            not _binding_is_configured(motion_lib),
            "fixed distribution is configured but its runtime binding is missing",
        )
        return False
    binding = _validate_binding(binding_raw)
    expected_binding_digest = getattr(
        motion_lib,
        "_lace_fixed_distribution_binding_sha256",
        None,
    )
    _require(
        expected_binding_digest == binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD],
        "installed fixed-distribution binding identity drifted",
    )
    _require(
        getattr(motion_lib, "_lace_fixed_distribution_config_sha256", None)
        == binding["fixed_distribution_config_sha256"],
        "installed fixed-distribution config identity drifted",
    )
    _require(
        getattr(motion_lib, "use_adaptive_sampling", None) is False,
        "native adaptive sampling was enabled after fixed-distribution installation",
    )
    _require(
        getattr(motion_lib, "all_motions_loaded", None) is True,
        "resident all-motion state drifted after fixed-distribution installation",
    )
    motion_keys = binding.get("motion_keys")
    _require(isinstance(motion_keys, list) and motion_keys, "binding motion keys are invalid")
    _require(
        all(isinstance(value, str) and value for value in motion_keys),
        "binding motion keys are invalid",
    )
    motion_count = binding.get("motion_count")
    _require(
        isinstance(motion_count, Integral)
        and not isinstance(motion_count, bool)
        and int(motion_count) == len(motion_keys),
        "binding motion count drifted",
    )
    motion_count = int(motion_count)
    resident_keys = getattr(motion_lib, "curr_motion_keys", None)
    _require(
        isinstance(resident_keys, Sequence) and not isinstance(resident_keys, (str, bytes)),
        "resident motion keys are unavailable",
    )
    _require(list(resident_keys) == motion_keys, "resident motion keys/order drifted")
    _require(
        getattr(motion_lib, "_num_unique_motions", None) == motion_count,
        "resident unique-motion count drifted",
    )
    _require(
        getattr(motion_lib, "_num_motions", None) == motion_count,
        "resident loaded-motion count drifted",
    )

    import torch

    canonical_ids = torch.arange(motion_count, dtype=torch.long)
    for field in ("_curr_motion_ids", "motion_ids"):
        actual_ids = getattr(motion_lib, field, None)
        _require(isinstance(actual_ids, torch.Tensor), f"{field} is unavailable")
        _require(actual_ids.dtype == torch.long, f"{field} must use torch.int64")
        _require(
            actual_ids.ndim == 1 and actual_ids.numel() == motion_count,
            f"{field} resident identity shape drifted",
        )
        _require(
            actual_ids.device == torch.device(motion_lib._device),
            f"{field} device drifted from the resident motion library",
        )
        _require(
            torch.equal(actual_ids.detach().cpu(), canonical_ids),
            f"{field} is not the canonical all-motion identity mapping",
        )

    probabilities = _probability_vector(binding.get("probabilities"), motion_count, "binding")
    expected = torch.tensor(probabilities, dtype=torch.float64)
    for field in ("_sampling_prob", "_sampling_batch_prob"):
        actual = getattr(motion_lib, field, None)
        _require(isinstance(actual, torch.Tensor), f"{field} is no longer a tensor")
        _require(actual.dtype == torch.float64, f"{field} must use torch.float64")
        _require(
            actual.ndim == 1 and actual.numel() == motion_count,
            f"{field} shape drifted from the frozen distribution",
        )
        _require(
            actual.device == torch.device(motion_lib._device),
            f"{field} device drifted from the resident motion library",
        )
        _require(
            torch.equal(actual.detach().cpu(), expected),
            f"{field} drifted from the frozen distribution",
        )
    _validate_accounting(motion_lib, motion_count)
    return True


def apply_fixed_distribution(motion_lib: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Install a frozen, deeply locked plan distribution into resident MotionLib."""

    _require(
        getattr(motion_lib, "lace_fixed_distribution_binding", None) is None
        and not getattr(motion_lib, "_lace_fixed_distribution_required", False),
        "fixed distribution is already installed",
    )
    _require(
        getattr(motion_lib, "use_adaptive_sampling", None) is False,
        "native adaptive sampling must be disabled for an RQ1 fixed distribution",
    )
    _require(
        getattr(motion_lib, "all_motions_loaded", None) is True,
        "every planned motion must be resident before fixed sampling is installed",
    )
    loaded_keys = getattr(motion_lib, "curr_motion_keys", None)
    _require(
        isinstance(loaded_keys, Sequence) and not isinstance(loaded_keys, (str, bytes)),
        "resident motion keys are unavailable",
    )
    probabilities, binding = resolve_fixed_distribution(
        config,
        loaded_motion_keys=loaded_keys,
    )
    import torch

    motion_count = len(probabilities)
    _require(
        getattr(motion_lib, "_num_unique_motions", None) == motion_count,
        "resident motion count does not equal the locked motion universe",
    )
    _require(
        getattr(motion_lib, "_num_motions", None) == motion_count,
        "loaded motion count does not equal the locked motion universe",
    )
    canonical_ids = torch.arange(motion_count, dtype=torch.long)
    for field in ("_curr_motion_ids", "motion_ids"):
        actual_ids = getattr(motion_lib, field, None)
        _require(isinstance(actual_ids, torch.Tensor), f"{field} is unavailable")
        _require(actual_ids.dtype == torch.long, f"{field} must use torch.int64")
        _require(
            actual_ids.ndim == 1
            and actual_ids.numel() == motion_count
            and torch.equal(actual_ids.detach().cpu(), canonical_ids),
            f"{field} is not the canonical all-motion identity mapping",
        )
    expected = torch.tensor(probabilities, dtype=torch.float64, device=motion_lib._device)
    motion_lib._sampling_prob = expected.clone()
    motion_lib._sampling_batch_prob = expected.clone()
    motion_lib.lace_fixed_distribution_binding = deepcopy(binding)
    motion_lib._lace_fixed_distribution_required = True
    motion_lib._lace_fixed_distribution_binding_sha256 = binding[
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD
    ]
    motion_lib._lace_fixed_distribution_config_sha256 = binding["fixed_distribution_config_sha256"]
    motion_lib._lace_fixed_distribution_draw_counts = torch.zeros(
        len(probabilities), dtype=torch.long, device=motion_lib._device
    )
    motion_lib._lace_fixed_distribution_total_draws = 0
    assert_fixed_distribution_integrity(motion_lib)
    return deepcopy(binding)


def record_fixed_distribution_draws(motion_lib: Any, motion_ids: Any) -> None:
    """Record one fixed-sampler call after validating exact drawn ID semantics."""

    active = assert_fixed_distribution_integrity(motion_lib)
    _require(active, "cannot record draws without an active fixed distribution")
    import torch

    _require(isinstance(motion_ids, torch.Tensor), "drawn motion IDs must be a tensor")
    _require(motion_ids.dtype == torch.long, "drawn motion IDs must use torch.int64")
    _require(motion_ids.ndim == 1, "drawn motion IDs must be one-dimensional")
    motion_count = int(motion_lib.lace_fixed_distribution_binding["motion_count"])
    if motion_ids.numel() > 0:
        _require(
            bool(torch.all((motion_ids >= 0) & (motion_ids < motion_count)).item()),
            "drawn motion ID is outside the resident universe",
        )
    increments = torch.bincount(motion_ids, minlength=motion_count).to(
        device=motion_lib._lace_fixed_distribution_draw_counts.device,
        dtype=torch.long,
    )
    motion_lib._lace_fixed_distribution_draw_counts += increments
    motion_lib._lace_fixed_distribution_total_draws += int(motion_ids.numel())
    _validate_accounting(motion_lib, motion_count)


def _build_draw_report(
    binding: Mapping[str, Any],
    draw_counts: Sequence[int],
    total_draw_count: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": FIXED_DISTRIBUTION_DRAW_REPORT_KIND,
        "schema_version": FIXED_DISTRIBUTION_DRAW_REPORT_SCHEMA_VERSION,
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD: binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD],
        "fixed_distribution_config_sha256": binding["fixed_distribution_config_sha256"],
        "arm_id": binding["arm_id"],
        INTERVENTION_PLAN_DIGEST_FIELD: binding[INTERVENTION_PLAN_DIGEST_FIELD],
        "plan_lock_file_sha256": binding["plan_lock_file_sha256"],
        "distribution_sha256": binding["distribution_sha256"],
        "motion_keys": list(binding["motion_keys"]),
        "motion_count": int(binding["motion_count"]),
        "draw_counts": [int(value) for value in draw_counts],
        "total_draw_count": int(total_draw_count),
        "counter_sum": sum(int(value) for value in draw_counts),
        "exact_invariants_pass": True,
        "accounting_scope": binding["accounting_scope"],
    }
    report[FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD] = canonical_sha256(
        report,
        digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    )
    return report


def fixed_distribution_draw_report(motion_lib: Any) -> dict[str, Any]:
    """Return a self-hashed exact motion-draw report for run receipts."""

    active = assert_fixed_distribution_integrity(motion_lib)
    _require(active, "fixed-distribution draw report requires an active binding")
    binding = motion_lib.lace_fixed_distribution_binding
    counts, total = _validate_accounting(motion_lib, int(binding["motion_count"]))
    return _build_draw_report(binding, counts, total)


def fixed_distribution_checkpoint_state(motion_lib: Any) -> dict[str, Any]:
    """Serialize fixed-arm identity and counters for an exact full resume."""

    report = fixed_distribution_draw_report(motion_lib)
    binding = deepcopy(dict(motion_lib.lace_fixed_distribution_binding))
    state = {
        "kind": FIXED_DISTRIBUTION_CHECKPOINT_KIND,
        "schema_version": FIXED_DISTRIBUTION_CHECKPOINT_SCHEMA_VERSION,
        "runtime_binding": binding,
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD: binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD],
        "fixed_distribution_config_sha256": binding["fixed_distribution_config_sha256"],
        "arm_id": binding["arm_id"],
        INTERVENTION_PLAN_DIGEST_FIELD: binding[INTERVENTION_PLAN_DIGEST_FIELD],
        "plan_lock_file_sha256": binding["plan_lock_file_sha256"],
        "probabilities": list(binding["probabilities"]),
        "draw_counts": list(report["draw_counts"]),
        "total_draw_count": report["total_draw_count"],
        FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD: report[
            FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD
        ],
    }
    return state


def restore_fixed_distribution_checkpoint_state(
    motion_lib: Any,
    state: Mapping[str, Any],
) -> None:
    """Restore only a checkpoint from the exact same locked arm and config."""

    active = assert_fixed_distribution_integrity(motion_lib)
    _require(active, "fixed-distribution checkpoint restore requires an active binding")
    _require(isinstance(state, Mapping), "fixed-distribution checkpoint state must be a mapping")
    _require(set(state) == _CHECKPOINT_FIELDS, "fixed-distribution checkpoint fields are invalid")
    _require(state.get("kind") == FIXED_DISTRIBUTION_CHECKPOINT_KIND, "checkpoint kind mismatch")
    _require(
        state.get("schema_version") == FIXED_DISTRIBUTION_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint schema version mismatch",
    )
    binding = motion_lib.lace_fixed_distribution_binding
    saved_binding = state.get("runtime_binding")
    _require(isinstance(saved_binding, Mapping), "checkpoint runtime binding is invalid")
    _validate_binding(saved_binding)
    _require(dict(saved_binding) == dict(binding), "checkpoint fixed-arm runtime binding mismatch")
    for field in (
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
        "fixed_distribution_config_sha256",
        "arm_id",
        INTERVENTION_PLAN_DIGEST_FIELD,
        "plan_lock_file_sha256",
    ):
        _require(state.get(field) == binding[field], f"checkpoint {field} mismatch")
    _require(
        state.get("probabilities") == binding["probabilities"],
        "checkpoint probability vector mismatch",
    )
    raw_counts = state.get("draw_counts")
    _require(isinstance(raw_counts, list), "checkpoint draw_counts must be a list")
    _require(
        len(raw_counts) == binding["motion_count"]
        and all(
            isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0
            for value in raw_counts
        ),
        "checkpoint draw_counts are invalid",
    )
    total = state.get("total_draw_count")
    _require(
        isinstance(total, Integral) and not isinstance(total, bool) and int(total) >= 0,
        "checkpoint total_draw_count is invalid",
    )
    counts = [int(value) for value in raw_counts]
    total = int(total)
    _require(sum(counts) == total, "checkpoint draw-count sum does not equal total")
    expected_report = _build_draw_report(binding, counts, total)
    _require(
        state.get(FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD)
        == expected_report[FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD],
        "checkpoint draw-report digest mismatch",
    )

    import torch

    motion_lib._lace_fixed_distribution_draw_counts = torch.tensor(
        counts,
        dtype=torch.long,
        device=motion_lib._device,
    )
    motion_lib._lace_fixed_distribution_total_draws = total
    assert_fixed_distribution_integrity(motion_lib)


def assert_fixed_distribution_reload_allowed(motion_lib: Any, api_name: str) -> None:
    """Fail closed if a resident reload API is invoked after arm installation."""

    if assert_fixed_distribution_integrity(motion_lib):
        raise RuntimeError(
            f"{api_name} is forbidden after an RQ1 fixed distribution is installed; "
            "resident motion identity and exposure must remain immutable"
        )
