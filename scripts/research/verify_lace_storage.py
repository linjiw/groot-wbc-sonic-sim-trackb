#!/usr/bin/env python3
"""Verify LACE storage capacity and locked research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.analysis_protocol import (  # noqa: E402
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    validate_analysis_protocol,
)
from gear_sonic.research.lace.analysis_protocol_lock import (  # noqa: E402
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    load_analysis_protocol_lock,
)
from gear_sonic.research.lace.atlas import validate_scientific_atlas_deep  # noqa: E402
from gear_sonic.research.lace.instrument_runtime import (  # noqa: E402
    _strict_json_loads,
)
from gear_sonic.research.lace.intervention_plan import (  # noqa: E402
    INTERVENTION_PLAN_DIGEST_FIELD,
    INTERVENTION_PLAN_KIND,
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    validate_intervention_plan,
    validate_intervention_protocol,
)
from gear_sonic.research.lace.panels import (  # noqa: E402
    PANEL_KIND,
    validate_panel_manifest,
)
from gear_sonic.research.lace.reference_lengths import (  # noqa: E402
    REFERENCE_LENGTH_DIGEST_FIELD,
    reference_num_steps_by_motion,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import (  # noqa: E402
    SCHEDULE_DIGEST_FIELD,
    SCHEDULE_KIND,
    build_rollout_schedule,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import (  # noqa: E402
    ATLAS_KIND,
    SPLIT_KIND,
    SPLIT_NAMES,
    canonical_sha256,
    validate_atlas_manifest,
    validate_split_manifest,
)
from gear_sonic.research.lace.split import (  # noqa: E402
    build_source_disjoint_split,
)
from scripts.research.build_bones_seed_official_cohort import (  # noqa: E402
    build_cohort,
)
from scripts.research.build_bones_seed_paired_manifest import (  # noqa: E402
    build_manifest as build_paired_manifest,
)

DEFAULT_STORAGE_CONFIG = REPO_ROOT / "configs/research/lace/storage_5090.json"
DEFAULT_MINIMUM_FREE_BYTES = 100_000_000_000
_SPLIT_LOCK_KINDS = {"lace_split_artifact_lock"}
_ATLAS_LOCK_KINDS = {"lace_atlas_artifact_lock", "lace_atlas_contract_smoke_lock"}
_PANEL_LOCK_KINDS = {"lace_source_panel_artifact_lock"}
_SCHEDULE_LOCK_KINDS = {"lace_probe_rollout_schedule_lock"}
_INTERVENTION_PLAN_LOCK_KINDS = {"lace_rq1_intervention_plan_artifact_lock"}
_REQUIRED_SCHEDULE_SPEC_FIELDS = {
    "schema_version",
    "reference_length_inventory_sha256",
    "probe_policies",
    "domain_randomization_seeds",
    "phase_targets",
    "repeats",
}
_OPTIONAL_SCHEDULE_SPEC_FIELDS = {
    "rollout_id_prefix",
}
_HEADLINE_SELECTION_PROTOCOL_KIND = "lace_headline_cohort_selection_protocol"
_HEADLINE_SPLIT_LOCK_TOP_FIELDS = {
    "schema_version",
    "kind",
    "scientific_use",
    "declared_before_headline_candidate_policy_outcomes",
    "prior_rollout_disposition",
    "seed",
    "artifact",
    "inputs",
    "partition_summary",
    "scope",
}
_HEADLINE_SPLIT_LOCK_ARTIFACT_FIELDS = {
    "path",
    "file_sha256",
    "selection_sha256",
    "split_sha256",
}
_HEADLINE_SPLIT_LOCK_INPUT_FIELDS = {
    "selection_protocol",
    "selection_protocol_sha256",
    "selection_protocol_self_sha256",
    "cohort_manifest",
    "cohort_manifest_sha256",
    "materialized_manifest",
    "materialized_manifest_sha256",
    "paired_dataset_sha256",
}
_HEADLINE_PROTOCOL_TOP_FIELDS = {
    "schema_version",
    "kind",
    "frozen",
    "scientific_use",
    "declared_before_headline_candidate_policy_outcomes",
    "outcome_access_permitted",
    "prior_rollout_disposition",
    "source",
    "selection",
    "split_preview",
    "artifacts",
    "scope",
    "protocol_sha256",
}
_HEADLINE_PROTOCOL_SOURCE_FIELDS = {
    "repo_id",
    "revision",
    "metadata_path",
    "metadata_bytes",
    "metadata_sha256",
    "published_motion_count",
    "eligible_motion_count",
    "eligibility_rule",
}
_HEADLINE_PROTOCOL_SELECTION_FIELDS = {
    "builder",
    "builder_sha256",
    "filter_implementation",
    "filter_implementation_sha256",
    "seed",
    "duration_bins",
    "stratification",
    "tie_breaker",
    "candidate_size_grid",
    "pre_outcome_constraints",
    "selection_rule",
    "selected_size",
    "selection_sha256",
}
_HEADLINE_PROTOCOL_PREVIEW_FIELDS = {
    "builder",
    "builder_sha256",
    "seed",
    "source_group_count",
    "partition_summary",
    "d_geometry_plus_d_test_source_groups",
    "note",
}
_HEADLINE_PROTOCOL_ARTIFACT_FIELDS = {
    "cohort_path",
    "cohort_file_sha256",
    "g1_member_list_path",
    "g1_member_list_file_sha256",
    "smpl_member_list_path",
    "smpl_member_list_file_sha256",
    "materializer_path",
    "materializer_file_sha256",
    "paired_manifest_builder_path",
    "paired_manifest_builder_file_sha256",
}


class ReadinessError(RuntimeError):
    """Raised when storage or a locked artifact is not ready for an experiment."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise ValueError("JSON input is blank")
        payload = _strict_json_loads(raw, f"JSON object {path}")
    except (OSError, ValueError) as exc:
        raise ReadinessError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReadinessError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 of a file's exact bytes."""

    resolved = Path(path)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReadinessError(f"cannot hash {resolved}: {exc}") from exc
    return digest.hexdigest()


def _expected_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReadinessError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ReadinessError(f"{field} must be a hexadecimal SHA-256 digest") from exc
    return value.lower()


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise ReadinessError(
            f"{field} fields differ from the frozen schema: "
            f"missing={missing}, unknown={unknown}"
        )


def _lexical_absolute_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        if relative_to is None:
            relative_to = Path.cwd()
        path = relative_to / path
    return Path(os.path.abspath(path))


def _reject_symlink_components(path: Path, field: str) -> None:
    """Reject an existing symlink anywhere in an externally bound path.

    ``Path.resolve`` would otherwise erase the alias before provenance checks.
    Missing components are left to the caller's normal missing-file error.
    """

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise ReadinessError(f"{field} must not contain symlink components: {current}")
        except OSError as exc:
            raise ReadinessError(f"cannot inspect {field} path component {current}: {exc}") from exc
        if not current.exists():
            break


def _canonical_existing_path(
    value: Any, *, relative_to: Path, field: str, expect_directory: bool = False
) -> Path:
    if not isinstance(value, str) or not value:
        raise ReadinessError(f"{field} must be a non-empty path string")
    raw = Path(value).expanduser()
    if any(part in {".", ".."} for part in raw.parts):
        raise ReadinessError(f"{field} must not contain '.' or '..' components")
    path = _lexical_absolute_path(raw, relative_to=relative_to)
    _reject_symlink_components(path, field)
    if expect_directory:
        if not path.is_dir():
            raise ReadinessError(f"{field} is not a directory: {path}")
    elif not path.is_file():
        raise ReadinessError(f"{field} is not a file: {path}")
    return path


def _checked_hash(path: Path, expected: Any, field: str) -> str:
    expected_digest = _expected_sha256(expected, field)
    if not path.is_file():
        raise ReadinessError(f"{field} references a missing file: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ReadinessError(
            f"{field} mismatch for {path}: expected {expected_digest}, computed {actual_digest}"
        )
    return actual_digest


def _resolve_storage_path(value: Any, *, config_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReadinessError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_lock_path(value: Any, *, repo_root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReadinessError(f"{field} must be a non-empty path string")
    raw = Path(value).expanduser()
    if any(part in {".", ".."} for part in raw.parts):
        raise ReadinessError(f"{field} must not contain '.' or '..' components")
    path = _lexical_absolute_path(raw, relative_to=repo_root)
    _reject_symlink_components(path, field)
    return path.resolve()


def _minimum_free_bytes(config: Mapping[str, Any], override: int | None) -> int:
    value: Any = config.get("minimum_free_bytes", DEFAULT_MINIMUM_FREE_BYTES)
    if override is not None:
        value = override
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReadinessError("minimum_free_bytes must be a non-negative integer")
    return value


def verify_storage_roots(
    storage_config_path: str | Path,
    *,
    minimum_free_bytes: int | None = None,
) -> dict[str, Any]:
    """Verify every configured dataset/artifact root and report its free bytes.

    Relative roots are resolved next to the storage configuration. Tests and
    alternate machines can therefore supply a fully temporary configuration.
    """

    config_path = Path(storage_config_path).expanduser().resolve()
    config = _load_json_object(config_path)
    required_free = _minimum_free_bytes(config, minimum_free_bytes)

    roots: list[tuple[str, Path]] = [
        (
            "storage_root",
            _resolve_storage_path(
                config.get("storage_root"), config_path=config_path, field="storage_root"
            ),
        ),
        (
            "dataset_root",
            _resolve_storage_path(
                config.get("dataset_root"), config_path=config_path, field="dataset_root"
            ),
        ),
    ]
    artifact_roots = config.get("artifact_roots")
    if not isinstance(artifact_roots, Mapping) or not artifact_roots:
        raise ReadinessError("artifact_roots must be a non-empty mapping")
    for name in sorted(artifact_roots):
        roots.append(
            (
                f"artifact_roots.{name}",
                _resolve_storage_path(
                    artifact_roots[name],
                    config_path=config_path,
                    field=f"artifact_roots.{name}",
                ),
            )
        )

    results: list[dict[str, Any]] = []
    for name, root in roots:
        if not root.is_dir():
            raise ReadinessError(
                f"configured storage root does not exist or is not a directory: {name}={root}"
            )
        try:
            free_bytes = shutil.disk_usage(root).free
        except OSError as exc:
            raise ReadinessError(f"cannot inspect free space for {name}={root}: {exc}") from exc
        if free_bytes < required_free:
            raise ReadinessError(
                f"insufficient free space for {name}={root}: "
                f"requires {required_free} bytes, found {free_bytes}"
            )
        results.append({"name": name, "path": str(root), "free_bytes": free_bytes})

    return {
        "config_path": str(config_path),
        "minimum_free_bytes": required_free,
        "roots": results,
    }


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{field} must be a mapping")
    return value


def _verify_locked_self_digests(
    manifest: Mapping[str, Any],
    artifact_lock: Mapping[str, Any],
    *,
    lock_kind: Any,
) -> tuple[str, dict[str, str]]:
    manifest_kind = manifest.get("kind")
    if manifest_kind == SPLIT_KIND:
        if lock_kind not in _SPLIT_LOCK_KINDS:
            raise ReadinessError(
                f"lock kind {lock_kind!r} cannot lock split artifact kind {SPLIT_KIND!r}"
            )
        locked = {
            "selection_sha256": _expected_sha256(
                artifact_lock.get("selection_sha256"), "artifact.selection_sha256"
            ),
            "split_sha256": _expected_sha256(
                artifact_lock.get("split_sha256"), "artifact.split_sha256"
            ),
        }
        for field, expected in locked.items():
            if manifest.get(field) != expected:
                raise ReadinessError(
                    f"locked {field} mismatch: expected {expected}, artifact contains {manifest.get(field)!r}"
                )
        try:
            validate_split_manifest(manifest, verify_digest=True)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"split artifact validation failed: {exc}") from exc
        return "split", locked

    if manifest_kind == ATLAS_KIND:
        if lock_kind not in _ATLAS_LOCK_KINDS:
            raise ReadinessError(
                f"lock kind {lock_kind!r} cannot lock atlas artifact kind {ATLAS_KIND!r}"
            )
        locked = {
            "atlas_sha256": _expected_sha256(
                artifact_lock.get("atlas_sha256"), "artifact.atlas_sha256"
            )
        }
        if manifest.get("atlas_sha256") != locked["atlas_sha256"]:
            raise ReadinessError(
                "locked atlas_sha256 mismatch: "
                f"expected {locked['atlas_sha256']}, artifact contains {manifest.get('atlas_sha256')!r}"
            )
        try:
            validate_atlas_manifest(manifest, verify_digest=True)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"atlas artifact validation failed: {exc}") from exc
        return "atlas", locked

    if manifest_kind == PANEL_KIND:
        if lock_kind not in _PANEL_LOCK_KINDS:
            raise ReadinessError(
                f"lock kind {lock_kind!r} cannot lock panel artifact kind {PANEL_KIND!r}"
            )
        locked = {
            "panel_sha256": _expected_sha256(
                artifact_lock.get("panel_sha256"), "artifact.panel_sha256"
            ),
            "split_sha256": _expected_sha256(
                artifact_lock.get("split_sha256"), "artifact.split_sha256"
            ),
            "split_selection_sha256": _expected_sha256(
                artifact_lock.get("split_selection_sha256"),
                "artifact.split_selection_sha256",
            ),
        }
        for field, expected in locked.items():
            if manifest.get(field) != expected:
                raise ReadinessError(
                    f"locked {field} mismatch: expected {expected}, "
                    f"artifact contains {manifest.get(field)!r}"
                )
        try:
            validate_panel_manifest(manifest, verify_digest=True)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"panel artifact validation failed: {exc}") from exc
        return "source_panels", locked

    if manifest_kind == SCHEDULE_KIND:
        if lock_kind not in _SCHEDULE_LOCK_KINDS:
            raise ReadinessError(
                f"lock kind {lock_kind!r} cannot lock schedule artifact kind {SCHEDULE_KIND!r}"
            )
        locked = {
            SCHEDULE_DIGEST_FIELD: _expected_sha256(
                artifact_lock.get(SCHEDULE_DIGEST_FIELD),
                f"artifact.{SCHEDULE_DIGEST_FIELD}",
            )
        }
        if manifest.get(SCHEDULE_DIGEST_FIELD) != locked[SCHEDULE_DIGEST_FIELD]:
            raise ReadinessError(
                f"locked {SCHEDULE_DIGEST_FIELD} mismatch: "
                f"expected {locked[SCHEDULE_DIGEST_FIELD]}, "
                f"artifact contains {manifest.get(SCHEDULE_DIGEST_FIELD)!r}"
            )
        schedule_schema_version = artifact_lock.get("schedule_schema_version")
        if schedule_schema_version != manifest.get("schema_version"):
            raise ReadinessError(
                "artifact.schedule_schema_version does not match the schedule artifact"
            )
        try:
            validate_rollout_schedule(manifest, verify_digest=True)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"schedule artifact validation failed: {exc}") from exc
        return "rollout_schedule", locked

    if manifest_kind == INTERVENTION_PLAN_KIND:
        if lock_kind not in _INTERVENTION_PLAN_LOCK_KINDS:
            raise ReadinessError(
                f"lock kind {lock_kind!r} cannot lock intervention plan kind "
                f"{INTERVENTION_PLAN_KIND!r}"
            )
        locked = {
            INTERVENTION_PLAN_DIGEST_FIELD: _expected_sha256(
                artifact_lock.get(INTERVENTION_PLAN_DIGEST_FIELD),
                f"artifact.{INTERVENTION_PLAN_DIGEST_FIELD}",
            )
        }
        if manifest.get(INTERVENTION_PLAN_DIGEST_FIELD) != locked[INTERVENTION_PLAN_DIGEST_FIELD]:
            raise ReadinessError(f"locked {INTERVENTION_PLAN_DIGEST_FIELD} does not match artifact")
        return "rq1_intervention_plan", locked

    raise ReadinessError(
        "locked artifact kind must be "
        f"{SPLIT_KIND!r}, {ATLAS_KIND!r}, {PANEL_KIND!r}, {SCHEDULE_KIND!r}, "
        f"or {INTERVENTION_PLAN_KIND!r}, "
        f"got {manifest_kind!r}"
    )


def _require_strict_bool(value: Any, field: str, *, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ReadinessError(f"{field} must be {str(expected).lower()}")


def _count_summary(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    summary = _require_mapping(manifest.get("partition_summary"), "partition_summary")
    if set(summary) != set(SPLIT_NAMES):
        raise ReadinessError(f"partition_summary must have exactly these keys: {list(SPLIT_NAMES)}")
    result: dict[str, dict[str, int]] = {}
    for partition in SPLIT_NAMES:
        record = _require_mapping(summary[partition], f"partition_summary.{partition}")
        result[partition] = {}
        for field in ("motion_count", "source_group_count"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ReadinessError(
                    f"partition_summary.{partition}.{field} must be a positive integer"
                )
            result[partition][field] = value
    return result


def _strict_flat_dataset_directory(
    root: Path, expected_names: Sequence[str], *, field: str
) -> None:
    _reject_symlink_components(root, field)
    if not root.is_dir():
        raise ReadinessError(f"{field} is not a directory: {root}")
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ReadinessError(f"{field} must contain only regular files: {entry}")
    names = sorted(entry.name for entry in entries)
    expected = sorted(expected_names)
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extras = sorted(set(names) - set(expected))
        raise ReadinessError(
            f"{field} inventory mismatch: missing={missing[:10]}, extras={extras[:10]}"
        )


def _validate_headline_materialized_manifest(
    materialized: Mapping[str, Any],
    *,
    materialized_path: Path,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    protocol: Mapping[str, Any],
    repo_root: Path,
) -> None:
    expected_top = {
        "schema_version",
        "kind",
        "generator",
        "source",
        "output",
        "materialization",
    }
    _require_exact_fields(materialized, expected_top, "materialized_manifest")
    if materialized.get("schema_version") != 1:
        raise ReadinessError("materialized_manifest.schema_version must be 1")
    if materialized.get("kind") != "official_bones_seed_paired_sonic_cohort":
        raise ReadinessError("materialized_manifest kind is invalid")

    artifacts = _require_mapping(protocol.get("artifacts"), "selection_protocol.artifacts")
    for path_field, digest_field in (
        ("materializer_path", "materializer_file_sha256"),
        ("paired_manifest_builder_path", "paired_manifest_builder_file_sha256"),
    ):
        implementation_path = _canonical_existing_path(
            artifacts.get(path_field),
            relative_to=repo_root,
            field=f"selection_protocol.artifacts.{path_field}",
        )
        _checked_hash(
            implementation_path,
            artifacts.get(digest_field),
            f"selection_protocol.artifacts.{digest_field}",
        )

    dataset_root = materialized_path.parent
    _reject_symlink_components(dataset_root, "materialized dataset root")
    expected_top_entries = {"dataset_manifest.json", "robot_filtered", "smpl_filtered"}
    observed_top_entries = {entry.name for entry in dataset_root.iterdir()}
    if observed_top_entries != expected_top_entries:
        raise ReadinessError(
            "materialized dataset root inventory mismatch: "
            f"expected={sorted(expected_top_entries)}, observed={sorted(observed_top_entries)}"
        )
    for entry in dataset_root.iterdir():
        if entry.is_symlink():
            raise ReadinessError(f"materialized dataset entries must not be symlinks: {entry}")

    output = _require_mapping(materialized.get("output"), "materialized_manifest.output")
    motion_keys = output.get("motion_keys")
    if not isinstance(motion_keys, list) or any(
        not isinstance(key, str) or not key for key in motion_keys
    ):
        raise ReadinessError("materialized_manifest.output.motion_keys must be strings")
    cohort_motions = cohort.get("motions")
    if not isinstance(cohort_motions, list):
        raise ReadinessError("cohort motions must be a list")
    expected_keys = sorted(str(record["motion_key"]) for record in cohort_motions)
    if motion_keys != expected_keys or len(set(motion_keys)) != len(motion_keys):
        raise ReadinessError("materialized motion keys do not exactly match the selected cohort")
    if output.get("motion_count") != len(expected_keys):
        raise ReadinessError("materialized motion_count does not match the cohort")
    if output.get("robot_dir") != "robot_filtered" or output.get("smpl_dir") != "smpl_filtered":
        raise ReadinessError("materialized output directories must use the frozen flat layout")

    robot_dir = dataset_root / "robot_filtered"
    smpl_dir = dataset_root / "smpl_filtered"
    expected_files = [f"{key}.pkl" for key in expected_keys]
    _strict_flat_dataset_directory(robot_dir, expected_files, field="materialized robot data")
    _strict_flat_dataset_directory(smpl_dir, expected_files, field="materialized SMPL data")
    try:
        rebuilt = build_paired_manifest(cohort_path, robot_dir, smpl_dir)
    except (OSError, TypeError, ValueError) as exc:
        raise ReadinessError(f"materialized dataset rebuild failed: {exc}") from exc
    observed_base = {field: materialized.get(field) for field in rebuilt}
    if observed_base != rebuilt:
        raise ReadinessError(
            "materialized manifest is not the exact deterministic rebuild of its files"
        )

    materialization = _require_mapping(
        materialized.get("materialization"), "materialized_manifest.materialization"
    )
    expected_materialization_fields = {
        "schema_version",
        "generator",
        "source_lock",
        "archives",
        "member_manifests",
        "robot_transform",
        "selective_regular_files_only",
        "source_lock_schema_version",
    }
    _require_exact_fields(
        materialization,
        expected_materialization_fields,
        "materialized_manifest.materialization",
    )
    if materialization.get("schema_version") != 1:
        raise ReadinessError("materialized materialization schema_version must be 1")
    if materialization.get("generator") != artifacts.get("materializer_path"):
        raise ReadinessError("materialized generator differs from the frozen protocol")
    _require_strict_bool(
        materialization.get("selective_regular_files_only"),
        "materialized selective_regular_files_only",
        expected=True,
    )

    source_lock_record = _require_mapping(
        materialization.get("source_lock"), "materialized_manifest.materialization.source_lock"
    )
    _require_exact_fields(source_lock_record, {"path", "sha256"}, "materialized source_lock")
    source_lock_path = _canonical_existing_path(
        source_lock_record.get("path"),
        relative_to=repo_root,
        field="materialized source_lock.path",
    )
    _checked_hash(
        source_lock_path,
        source_lock_record.get("sha256"),
        "materialized source_lock.sha256",
    )
    source_lock = _load_json_object(source_lock_path)
    if source_lock.get("schema_version") != materialization.get("source_lock_schema_version"):
        raise ReadinessError("materialized source-lock schema version mismatch")
    bones_lock = _require_mapping(source_lock.get("bones_seed"), "source_lock.bones_seed")
    smpl_lock = _require_mapping(source_lock.get("smpl"), "source_lock.smpl")
    archives = _require_mapping(
        materialization.get("archives"), "materialized_manifest.materialization.archives"
    )
    _require_strict_bool(
        archives.get("all_full_archives_verified_before_extraction"),
        "materialized archive verification declaration",
        expected=True,
    )
    bones_files = _require_mapping(bones_lock.get("files"), "source_lock.bones_seed.files")
    expected_g1 = {
        "path": "g1.tar.gz",
        **_require_mapping(bones_files.get("g1.tar.gz"), "G1 archive"),
    }
    if archives.get("g1") != expected_g1:
        raise ReadinessError("materialized G1 archive record differs from the source lock")
    if archives.get("smpl_parts") != smpl_lock.get("parts"):
        raise ReadinessError("materialized SMPL archive records differ from the source lock")

    source = _require_mapping(protocol.get("source"), "selection_protocol.source")
    metadata_record = _require_mapping(
        bones_files.get("metadata/seed_metadata_v004.csv"),
        "source_lock metadata CSV",
    )
    if metadata_record != {
        "bytes": source.get("metadata_bytes"),
        "sha256": source.get("metadata_sha256"),
    }:
        raise ReadinessError("selection metadata record differs from the source lock")
    if bones_lock.get("repo_id") != source.get("repo_id") or bones_lock.get(
        "revision"
    ) != source.get("revision"):
        raise ReadinessError("selection protocol repository identity differs from source lock")

    members = _require_mapping(
        materialization.get("member_manifests"),
        "materialized_manifest.materialization.member_manifests",
    )
    _require_strict_bool(
        members.get("exact_cohort_order"),
        "materialized exact_cohort_order",
        expected=True,
    )
    if members.get("g1_sha256") != artifacts.get("g1_member_list_file_sha256"):
        raise ReadinessError("materialized G1 member digest differs from selection protocol")
    if members.get("smpl_sha256") != artifacts.get("smpl_member_list_file_sha256"):
        raise ReadinessError("materialized SMPL member digest differs from selection protocol")


def _validate_headline_selection_chain(
    lock: Mapping[str, Any],
    *,
    repo_root: Path,
    split_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    materialized: Mapping[str, Any],
    materialized_path: Path,
) -> None:
    _require_exact_fields(lock, _HEADLINE_SPLIT_LOCK_TOP_FIELDS, "headline split lock")
    if lock.get("schema_version") != 2:
        raise ReadinessError("headline split lock schema_version must be 2")
    if lock.get("kind") != "lace_split_artifact_lock":
        raise ReadinessError("headline split lock kind is invalid")
    _require_strict_bool(lock.get("scientific_use"), "lock.scientific_use", expected=True)
    _require_strict_bool(
        lock.get("declared_before_headline_candidate_policy_outcomes"),
        "lock.declared_before_headline_candidate_policy_outcomes",
        expected=True,
    )
    if not isinstance(lock.get("prior_rollout_disposition"), str) or not lock.get(
        "prior_rollout_disposition"
    ):
        raise ReadinessError("lock.prior_rollout_disposition must be non-empty")
    artifact = _require_mapping(lock.get("artifact"), "artifact")
    inputs = _require_mapping(lock.get("inputs"), "inputs")
    _require_exact_fields(artifact, _HEADLINE_SPLIT_LOCK_ARTIFACT_FIELDS, "artifact")
    _require_exact_fields(inputs, _HEADLINE_SPLIT_LOCK_INPUT_FIELDS, "inputs")

    _require_exact_fields(protocol, _HEADLINE_PROTOCOL_TOP_FIELDS, "selection_protocol")
    if protocol.get("schema_version") != 2:
        raise ReadinessError("selection protocol schema_version must be 2")
    if protocol.get("kind") != _HEADLINE_SELECTION_PROTOCOL_KIND:
        raise ReadinessError("selection protocol kind is invalid")
    for field, expected in (
        ("frozen", True),
        ("scientific_use", True),
        ("declared_before_headline_candidate_policy_outcomes", True),
        ("outcome_access_permitted", False),
    ):
        _require_strict_bool(protocol.get(field), f"selection_protocol.{field}", expected=expected)
    if protocol.get("prior_rollout_disposition") != lock.get("prior_rollout_disposition"):
        raise ReadinessError("selection protocol and lock prior-rollout dispositions differ")
    expected_protocol_digest = _expected_sha256(
        inputs.get("selection_protocol_self_sha256"),
        "inputs.selection_protocol_self_sha256",
    )
    if protocol.get("protocol_sha256") != expected_protocol_digest:
        raise ReadinessError("selection protocol self digest does not match the split lock")
    if canonical_sha256(protocol, digest_field="protocol_sha256") != expected_protocol_digest:
        raise ReadinessError("selection protocol self digest is invalid")
    if (
        _resolve_lock_path(
            inputs.get("selection_protocol"),
            repo_root=repo_root,
            field="inputs.selection_protocol",
        )
        != protocol_path
    ):
        raise ReadinessError("selection protocol path binding changed during verification")

    source = _require_mapping(protocol.get("source"), "selection_protocol.source")
    selection = _require_mapping(protocol.get("selection"), "selection_protocol.selection")
    preview = _require_mapping(protocol.get("split_preview"), "selection_protocol.split_preview")
    artifacts = _require_mapping(protocol.get("artifacts"), "selection_protocol.artifacts")
    _require_exact_fields(source, _HEADLINE_PROTOCOL_SOURCE_FIELDS, "selection_protocol.source")
    _require_exact_fields(
        selection, _HEADLINE_PROTOCOL_SELECTION_FIELDS, "selection_protocol.selection"
    )
    _require_exact_fields(
        preview, _HEADLINE_PROTOCOL_PREVIEW_FIELDS, "selection_protocol.split_preview"
    )
    _require_exact_fields(
        artifacts, _HEADLINE_PROTOCOL_ARTIFACT_FIELDS, "selection_protocol.artifacts"
    )

    metadata_path = _canonical_existing_path(
        source.get("metadata_path"),
        relative_to=repo_root,
        field="selection_protocol.source.metadata_path",
    )
    metadata_bytes = source.get("metadata_bytes")
    if (
        isinstance(metadata_bytes, bool)
        or not isinstance(metadata_bytes, int)
        or metadata_bytes <= 0
    ):
        raise ReadinessError("selection protocol metadata_bytes must be positive")
    if metadata_path.stat().st_size != metadata_bytes:
        raise ReadinessError("selection metadata byte count mismatch")
    _checked_hash(
        metadata_path,
        source.get("metadata_sha256"),
        "selection_protocol.source.metadata_sha256",
    )

    for path_field, digest_field in (
        ("builder", "builder_sha256"),
        ("filter_implementation", "filter_implementation_sha256"),
    ):
        implementation = _canonical_existing_path(
            selection.get(path_field),
            relative_to=repo_root,
            field=f"selection_protocol.selection.{path_field}",
        )
        _checked_hash(
            implementation,
            selection.get(digest_field),
            f"selection_protocol.selection.{digest_field}",
        )
    if preview.get("builder") != "gear_sonic.research.lace.split:build_source_disjoint_split":
        raise ReadinessError("selection split-preview builder is invalid")
    split_builder_path = repo_root / "gear_sonic/research/lace/split.py"
    _checked_hash(
        split_builder_path,
        preview.get("builder_sha256"),
        "selection_protocol.split_preview.builder_sha256",
    )

    if (
        _canonical_existing_path(
            artifacts.get("cohort_path"),
            relative_to=repo_root,
            field="selection_protocol.artifacts.cohort_path",
        )
        != cohort_path
    ):
        raise ReadinessError("selection protocol cohort path does not match split-lock input")
    cohort_file_digest = _expected_sha256(
        artifacts.get("cohort_file_sha256"),
        "selection_protocol.artifacts.cohort_file_sha256",
    )
    if cohort_file_digest != _expected_sha256(
        inputs.get("cohort_manifest_sha256"), "inputs.cohort_manifest_sha256"
    ):
        raise ReadinessError("selection protocol and split lock cohort digests differ")

    seed = selection.get("seed")
    duration_bins = selection.get("duration_bins")
    selected_size = selection.get("selected_size")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (seed, duration_bins, selected_size)
    ):
        raise ReadinessError(
            "selection seed, duration_bins, and selected_size must be positive integers"
        )
    try:
        rebuilt_selected = build_cohort(
            metadata_path,
            size=selected_size,
            seed=seed,
            duration_bins=duration_bins,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ReadinessError(f"selected cohort deterministic rebuild failed: {exc}") from exc
    if rebuilt_selected != cohort:
        raise ReadinessError("cohort manifest is not the exact deterministic protocol output")
    cohort_selection = _require_mapping(cohort.get("selection"), "cohort.selection")
    if cohort_selection.get("selection_sha256") != selection.get("selection_sha256"):
        raise ReadinessError("selection protocol selection digest differs from cohort")
    if len(cohort.get("motions", [])) != selected_size:
        raise ReadinessError("selection protocol selected_size differs from cohort")
    cohort_source = _require_mapping(cohort.get("source"), "cohort.source")
    eligibility = _require_mapping(cohort.get("eligibility"), "cohort.eligibility")
    for field in ("repo_id", "revision", "published_motion_count"):
        if cohort_source.get(field) != source.get(field):
            raise ReadinessError(f"selection protocol source.{field} differs from cohort")
    if eligibility.get("eligible_motion_count") != source.get("eligible_motion_count"):
        raise ReadinessError("selection protocol eligible count differs from cohort")
    if eligibility.get("rule") != source.get("eligibility_rule"):
        raise ReadinessError("selection protocol eligibility rule differs from cohort")
    for protocol_field, cohort_field in (
        ("seed", "seed"),
        ("duration_bins", "duration_bins"),
        ("stratification", "stratification"),
        ("tie_breaker", "tie_breaker"),
    ):
        if selection.get(protocol_field) != cohort_selection.get(cohort_field):
            raise ReadinessError(f"selection protocol {protocol_field} differs from rebuilt cohort")

    for modality, row_field in (
        ("g1", "g1_archive_member"),
        ("smpl", "smpl_archive_member"),
    ):
        path_field = f"{modality}_member_list_path"
        digest_field = f"{modality}_member_list_file_sha256"
        member_path = _canonical_existing_path(
            artifacts.get(path_field),
            relative_to=repo_root,
            field=f"selection_protocol.artifacts.{path_field}",
        )
        _checked_hash(
            member_path,
            artifacts.get(digest_field),
            f"selection_protocol.artifacts.{digest_field}",
        )
        expected_lines = [str(record[row_field]) for record in cohort["motions"]]
        try:
            observed_lines = member_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReadinessError(f"cannot read frozen {modality} member list: {exc}") from exc
        if observed_lines != expected_lines or len(set(observed_lines)) != len(observed_lines):
            raise ReadinessError(
                f"frozen {modality} member list does not exactly match cohort order"
            )

    grid = _require_mapping(
        selection.get("candidate_size_grid"),
        "selection_protocol.selection.candidate_size_grid",
    )
    _require_exact_fields(
        grid,
        {"start_inclusive", "stop_inclusive", "step"},
        "selection_protocol.selection.candidate_size_grid",
    )
    start = grid.get("start_inclusive")
    stop = grid.get("stop_inclusive")
    step = grid.get("step")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, stop, step)):
        raise ReadinessError("candidate grid values must be integers")
    if start <= 0 or stop < start or step <= 0 or (stop - start) % step:
        raise ReadinessError("candidate grid must be a positive exact inclusive range")
    constraints = _require_mapping(
        selection.get("pre_outcome_constraints"),
        "selection_protocol.selection.pre_outcome_constraints",
    )
    _require_exact_fields(
        constraints,
        {
            "minimum_d_atlas_motion_count",
            "maximum_d_atlas_motion_count",
            "minimum_d_geometry_plus_d_test_source_groups",
        },
        "selection_protocol.selection.pre_outcome_constraints",
    )
    minimum_atlas = constraints.get("minimum_d_atlas_motion_count")
    maximum_atlas = constraints.get("maximum_d_atlas_motion_count")
    minimum_tail_groups = constraints.get("minimum_d_geometry_plus_d_test_source_groups")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (minimum_atlas, maximum_atlas, minimum_tail_groups)
        )
        or minimum_atlas > maximum_atlas
    ):
        raise ReadinessError("pre-outcome cohort constraints are invalid")
    if (
        selection.get("selection_rule")
        != "unique_grid_candidate_satisfying_all_pre_outcome_constraints"
    ):
        raise ReadinessError("selection rule is not the frozen unique-grid rule")

    split_seed = preview.get("seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ReadinessError("split preview seed must be an integer")
    candidate_cache: dict[int, Mapping[str, Any]] = {selected_size: rebuilt_selected}
    passing_sizes: list[int] = []
    selected_preview: Mapping[str, Any] | None = None
    for candidate_size in range(start, stop + 1, step):
        candidate = candidate_cache.get(candidate_size)
        if candidate is None:
            try:
                candidate = build_cohort(
                    metadata_path,
                    size=candidate_size,
                    seed=seed,
                    duration_bins=duration_bins,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ReadinessError(
                    f"candidate cohort {candidate_size} rebuild failed: {exc}"
                ) from exc
        try:
            candidate_split = build_source_disjoint_split(candidate["motions"], seed=split_seed)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(
                f"candidate cohort {candidate_size} split rebuild failed: {exc}"
            ) from exc
        candidate_summary = _count_summary(candidate_split)
        atlas_count = candidate_summary["D_atlas"]["motion_count"]
        tail_groups = sum(
            candidate_summary[name]["source_group_count"] for name in ("D_geometry", "D_test")
        )
        if minimum_atlas <= atlas_count <= maximum_atlas and tail_groups >= minimum_tail_groups:
            passing_sizes.append(candidate_size)
        if candidate_size == selected_size:
            selected_preview = candidate_split
    if passing_sizes != [selected_size]:
        raise ReadinessError(
            "selection protocol unique-grid claim failed: "
            f"selected={selected_size}, passing={passing_sizes}"
        )
    assert selected_preview is not None
    preview_summary = _count_summary(selected_preview)
    declared_preview_summary = _count_summary(
        {"partition_summary": preview.get("partition_summary")}
    )
    if declared_preview_summary != preview_summary:
        raise ReadinessError("selection protocol split preview summary is not deterministic")
    source_group_count = sum(record["source_group_count"] for record in preview_summary.values())
    if preview.get("source_group_count") != source_group_count:
        raise ReadinessError("selection protocol split-preview source-group count is invalid")
    tail_groups = sum(
        preview_summary[name]["source_group_count"] for name in ("D_geometry", "D_test")
    )
    if preview.get("d_geometry_plus_d_test_source_groups") != tail_groups:
        raise ReadinessError("selection protocol split-preview tail group count is invalid")

    locked_summary = _count_summary({"partition_summary": lock.get("partition_summary")})
    final_summary = _count_summary(split_manifest)
    if locked_summary != preview_summary or final_summary != preview_summary:
        raise ReadinessError("headline lock, preview, and final split partition summaries differ")
    if lock.get("seed") != split_seed or split_manifest.get("seed") != split_seed:
        raise ReadinessError("headline lock, preview, and final split seeds differ")

    _validate_headline_materialized_manifest(
        materialized,
        materialized_path=materialized_path,
        cohort=cohort,
        cohort_path=cohort_path,
        protocol=protocol,
        repo_root=repo_root,
    )
    materialized_digest = _expected_sha256(
        inputs.get("materialized_manifest_sha256"),
        "inputs.materialized_manifest_sha256",
    )
    paired_digest = _expected_sha256(
        inputs.get("paired_dataset_sha256"), "inputs.paired_dataset_sha256"
    )
    expected_dataset = {
        "cohort_manifest": str(cohort_path),
        "cohort_manifest_sha256": cohort_file_digest,
        "dataset_root": str(materialized_path.parent),
        "materialized_manifest": str(materialized_path),
        "materialized_manifest_sha256": materialized_digest,
        "motion_count": selected_size,
        "paired_dataset_sha256": paired_digest,
    }
    if split_manifest.get("dataset") != expected_dataset:
        raise ReadinessError("headline split dataset provenance does not exactly bind lock inputs")

    output = _require_mapping(materialized.get("output"), "materialized_manifest.output")
    variants = output.get("variants")
    if not isinstance(variants, list) or len(variants) != selected_size:
        raise ReadinessError("materialized variants do not cover the headline cohort")
    variants_by_key = {
        str(record.get("motion_key")): record for record in variants if isinstance(record, Mapping)
    }
    if len(variants_by_key) != selected_size:
        raise ReadinessError("materialized variants contain duplicate or malformed keys")
    enriched: list[dict[str, Any]] = []
    for source_record in cohort["motions"]:
        motion_key = str(source_record["motion_key"])
        variant = _require_mapping(
            variants_by_key.get(motion_key), f"materialized variant {motion_key}"
        )
        robot = _require_mapping(variant.get("robot"), f"materialized {motion_key}.robot")
        smpl = _require_mapping(variant.get("smpl"), f"materialized {motion_key}.smpl")
        record = dict(source_record)
        record.update(
            {
                "robot_path": str(materialized_path.parent / str(robot.get("path"))),
                "smpl_path": str(materialized_path.parent / str(smpl.get("path"))),
                "available_modalities": ["g1", "smpl"],
            }
        )
        enriched.append(record)
    try:
        rebuilt_split = build_source_disjoint_split(
            enriched,
            seed=split_seed,
            ratios=_require_mapping(split_manifest.get("ratios"), "split.ratios"),
            dataset=expected_dataset,
        )
    except (TypeError, ValueError) as exc:
        raise ReadinessError(f"headline split deterministic rebuild failed: {exc}") from exc
    if rebuilt_split != split_manifest:
        raise ReadinessError(
            "headline split is not the exact deterministic cohort/materialization output"
        )


def _verify_referenced_manifests(
    lock: Mapping[str, Any],
    *,
    repo_root: Path,
    artifact_manifest: Mapping[str, Any],
    expected_analysis_protocol_lock_sha256: str | None,
) -> list[dict[str, str]]:
    inputs = lock.get("inputs")
    if inputs is None:
        if artifact_manifest.get("kind") == SCHEDULE_KIND:
            raise ReadinessError("schedule locks require an inputs mapping")
        return []
    inputs = _require_mapping(inputs, "inputs")
    references: list[dict[str, str]] = []
    loaded: dict[str, Mapping[str, Any]] = {}
    loaded_paths: dict[str, Path] = {}
    for name in (
        "cohort_manifest",
        "materialized_manifest",
        "split_manifest",
        "schedule_spec",
        "panel_manifest",
        "intervention_protocol",
        "rollout_manifest",
        "schedule_manifest",
    ):
        digest_name = f"{name}_sha256"
        path_value = inputs.get(name)
        digest_value = inputs.get(digest_name)
        if path_value is None and digest_value is None:
            continue
        if path_value is None or digest_value is None:
            raise ReadinessError(
                f"inputs.{name} and inputs.{digest_name} must be provided together"
            )
        path = _resolve_lock_path(path_value, repo_root=repo_root, field=f"inputs.{name}")
        digest = _checked_hash(path, digest_value, f"inputs.{digest_name}")
        loaded[name] = _load_json_object(path)
        loaded_paths[name] = path
        references.append({"name": name, "path": str(path), "file_sha256": digest})

    selection_protocol_path_value = inputs.get("selection_protocol")
    selection_protocol_file_digest_value = inputs.get("selection_protocol_sha256")
    selection_protocol_self_digest_value = inputs.get("selection_protocol_self_sha256")
    selection_protocol_values = (
        selection_protocol_path_value,
        selection_protocol_file_digest_value,
        selection_protocol_self_digest_value,
    )
    if any(value is not None for value in selection_protocol_values):
        if any(value is None for value in selection_protocol_values):
            raise ReadinessError(
                "inputs.selection_protocol, inputs.selection_protocol_sha256, and "
                "inputs.selection_protocol_self_sha256 must be provided together"
            )
        selection_protocol_path = _resolve_lock_path(
            selection_protocol_path_value,
            repo_root=repo_root,
            field="inputs.selection_protocol",
        )
        selection_protocol_file_digest = _checked_hash(
            selection_protocol_path,
            selection_protocol_file_digest_value,
            "inputs.selection_protocol_sha256",
        )
        selection_protocol = _load_json_object(selection_protocol_path)
        selection_protocol_self_digest = _expected_sha256(
            selection_protocol_self_digest_value,
            "inputs.selection_protocol_self_sha256",
        )
        if selection_protocol.get("protocol_sha256") != selection_protocol_self_digest:
            raise ReadinessError(
                "inputs.selection_protocol_self_sha256 does not match the protocol"
            )
        loaded["selection_protocol"] = selection_protocol
        loaded_paths["selection_protocol"] = selection_protocol_path
        references.append(
            {
                "name": "selection_protocol",
                "path": str(selection_protocol_path),
                "file_sha256": selection_protocol_file_digest,
                "protocol_sha256": selection_protocol_self_digest,
            }
        )

    inventory_path_value = inputs.get("reference_length_inventory")
    inventory_file_digest_value = inputs.get("reference_length_inventory_file_sha256")
    inventory_self_digest_value = inputs.get("reference_length_inventory_sha256")
    inventory_values = (
        inventory_path_value,
        inventory_file_digest_value,
        inventory_self_digest_value,
    )
    if any(value is not None for value in inventory_values):
        if any(value is None for value in inventory_values):
            raise ReadinessError(
                "inputs.reference_length_inventory, "
                "inputs.reference_length_inventory_file_sha256, and "
                "inputs.reference_length_inventory_sha256 must be provided together"
            )
        inventory_path = _resolve_lock_path(
            inventory_path_value,
            repo_root=repo_root,
            field="inputs.reference_length_inventory",
        )
        inventory_file_digest = _checked_hash(
            inventory_path,
            inventory_file_digest_value,
            "inputs.reference_length_inventory_file_sha256",
        )
        inventory = _load_json_object(inventory_path)
        inventory_self_digest = _expected_sha256(
            inventory_self_digest_value,
            "inputs.reference_length_inventory_sha256",
        )
        if inventory.get(REFERENCE_LENGTH_DIGEST_FIELD) != inventory_self_digest:
            raise ReadinessError(
                "inputs.reference_length_inventory_sha256 does not match the "
                f"inventory's {REFERENCE_LENGTH_DIGEST_FIELD}"
            )
        loaded["reference_length_inventory"] = inventory
        references.append(
            {
                "name": "reference_length_inventory",
                "path": str(inventory_path),
                "file_sha256": inventory_file_digest,
                REFERENCE_LENGTH_DIGEST_FIELD: inventory_self_digest,
            }
        )

    protocol_path_value = inputs.get("analysis_protocol")
    protocol_file_digest_value = inputs.get("analysis_protocol_file_sha256")
    protocol_self_digest_value = inputs.get(ANALYSIS_PROTOCOL_DIGEST_FIELD)
    protocol_values = (
        protocol_path_value,
        protocol_file_digest_value,
        protocol_self_digest_value,
    )
    if any(value is not None for value in protocol_values):
        if any(value is None for value in protocol_values):
            raise ReadinessError(
                "inputs.analysis_protocol, inputs.analysis_protocol_file_sha256, and "
                f"inputs.{ANALYSIS_PROTOCOL_DIGEST_FIELD} must be provided together"
            )
        protocol_path = _resolve_lock_path(
            protocol_path_value,
            repo_root=repo_root,
            field="inputs.analysis_protocol",
        )
        protocol_file_digest = _checked_hash(
            protocol_path,
            protocol_file_digest_value,
            "inputs.analysis_protocol_file_sha256",
        )
        protocol = _load_json_object(protocol_path)
        protocol_self_digest = _expected_sha256(
            protocol_self_digest_value,
            f"inputs.{ANALYSIS_PROTOCOL_DIGEST_FIELD}",
        )
        if protocol.get(ANALYSIS_PROTOCOL_DIGEST_FIELD) != protocol_self_digest:
            raise ReadinessError(
                f"inputs.{ANALYSIS_PROTOCOL_DIGEST_FIELD} does not match the protocol"
            )
        loaded["analysis_protocol"] = protocol
        loaded["analysis_protocol_path"] = {"path": str(protocol_path)}
        references.append(
            {
                "name": "analysis_protocol",
                "path": str(protocol_path),
                "file_sha256": protocol_file_digest,
                ANALYSIS_PROTOCOL_DIGEST_FIELD: protocol_self_digest,
            }
        )

    protocol_lock_path_value = inputs.get("analysis_protocol_lock")
    protocol_lock_file_digest_value = inputs.get("analysis_protocol_lock_file_sha256")
    protocol_lock_self_digest_value = inputs.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD)
    protocol_lock_values = (
        protocol_lock_path_value,
        protocol_lock_file_digest_value,
        protocol_lock_self_digest_value,
    )
    if any(value is not None for value in protocol_lock_values):
        if any(value is None for value in protocol_lock_values):
            raise ReadinessError(
                "inputs.analysis_protocol_lock, inputs.analysis_protocol_lock_file_sha256, "
                f"and inputs.{ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD} must be provided together"
            )
        protocol_lock_path = _resolve_lock_path(
            protocol_lock_path_value,
            repo_root=repo_root,
            field="inputs.analysis_protocol_lock",
        )
        protocol_lock_file_digest = _checked_hash(
            protocol_lock_path,
            protocol_lock_file_digest_value,
            "inputs.analysis_protocol_lock_file_sha256",
        )
        try:
            _, protocol_lock, _, _ = load_analysis_protocol_lock(
                protocol_lock_path,
                repo_root=repo_root,
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"analysis protocol lock validation failed: {exc}") from exc
        protocol_lock_self_digest = _expected_sha256(
            protocol_lock_self_digest_value,
            f"inputs.{ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD}",
        )
        if protocol_lock.get(ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD) != protocol_lock_self_digest:
            raise ReadinessError("analysis protocol lock self digest does not match input")
        loaded["analysis_protocol_lock"] = protocol_lock
        loaded["analysis_protocol_lock_path"] = {"path": str(protocol_lock_path)}
        references.append(
            {
                "name": "analysis_protocol_lock",
                "path": str(protocol_lock_path),
                "file_sha256": protocol_lock_file_digest,
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: protocol_lock_self_digest,
            }
        )

    expected_paired = inputs.get("paired_dataset_sha256")
    if expected_paired is not None:
        expected_paired = _expected_sha256(expected_paired, "inputs.paired_dataset_sha256")
        materialized = loaded.get("materialized_manifest")
        if materialized is None:
            raise ReadinessError(
                "inputs.paired_dataset_sha256 requires a referenced materialized_manifest"
            )
        output = _require_mapping(materialized.get("output"), "materialized_manifest.output")
        if output.get("paired_dataset_sha256") != expected_paired:
            raise ReadinessError(
                "inputs.paired_dataset_sha256 does not match "
                "materialized_manifest.output.paired_dataset_sha256"
            )
    split_manifest = loaded.get("split_manifest")
    if split_manifest is not None:
        try:
            validate_split_manifest(split_manifest, verify_digest=True)
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"referenced split manifest validation failed: {exc}") from exc
        if artifact_manifest.get("kind") in {ATLAS_KIND, PANEL_KIND}:
            for artifact_field, split_field in (
                ("split_sha256", "split_sha256"),
                ("split_selection_sha256", "selection_sha256"),
            ):
                if artifact_manifest.get(artifact_field) != split_manifest.get(split_field):
                    raise ReadinessError(
                        f"artifact {artifact_field} does not bind the referenced split manifest"
                    )
    if artifact_manifest.get("kind") == SPLIT_KIND and lock.get("scientific_use") is True:
        required_headline_inputs = {
            "selection_protocol",
            "cohort_manifest",
            "materialized_manifest",
        }
        missing_headline_inputs = sorted(required_headline_inputs - set(loaded))
        if missing_headline_inputs:
            raise ReadinessError(
                "scientific headline split lock is missing required inputs: "
                f"{missing_headline_inputs}"
            )
        _validate_headline_selection_chain(
            lock,
            repo_root=repo_root,
            split_manifest=artifact_manifest,
            protocol=loaded["selection_protocol"],
            protocol_path=loaded_paths["selection_protocol"],
            cohort=loaded["cohort_manifest"],
            cohort_path=loaded_paths["cohort_manifest"],
            materialized=loaded["materialized_manifest"],
            materialized_path=loaded_paths["materialized_manifest"],
        )
    if artifact_manifest.get("kind") == SCHEDULE_KIND:
        required_references = {
            "split_manifest",
            "schedule_spec",
            "reference_length_inventory",
        }
        missing_references = sorted(required_references - set(loaded))
        if missing_references:
            raise ReadinessError(
                f"schedule lock is missing required referenced inputs: {missing_references}"
            )
        if lock.get("schema_version") != 1:
            raise ReadinessError("schedule lock schema_version must be 1")
        scientific_use = lock.get("scientific_use")
        if not isinstance(scientific_use, bool):
            raise ReadinessError("schedule lock scientific_use must be boolean")
        if artifact_manifest.get("scientific_use") is not scientific_use:
            raise ReadinessError(
                "schedule lock scientific_use does not match the schedule artifact"
            )
        schedule_spec = loaded["schedule_spec"]
        inventory = loaded["reference_length_inventory"]
        assert split_manifest is not None
        try:
            validate_reference_length_inventory(
                inventory,
                split_manifest=split_manifest,
                verify_digest=True,
                verify_source_files=True,
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"referenced length inventory validation failed: {exc}") from exc
        try:
            validate_rollout_schedule(
                artifact_manifest,
                split_manifest=split_manifest,
                reference_length_inventory=inventory,
                verify_digest=True,
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(
                f"schedule artifact does not bind the referenced split: {exc}"
            ) from exc

        spec_fields = set(schedule_spec)
        missing_spec_fields = sorted(_REQUIRED_SCHEDULE_SPEC_FIELDS - spec_fields)
        unknown_spec_fields = sorted(
            spec_fields - _REQUIRED_SCHEDULE_SPEC_FIELDS - _OPTIONAL_SCHEDULE_SPEC_FIELDS
        )
        if missing_spec_fields:
            raise ReadinessError(
                f"referenced schedule spec is missing required fields: {missing_spec_fields}"
            )
        if unknown_spec_fields:
            raise ReadinessError(
                f"referenced schedule spec contains unknown fields: {unknown_spec_fields}"
            )
        if schedule_spec.get("schema_version") != 2:
            raise ReadinessError("referenced schedule spec schema_version must be 2")
        if (
            schedule_spec.get("reference_length_inventory_sha256")
            != inventory[REFERENCE_LENGTH_DIGEST_FIELD]
        ):
            raise ReadinessError(
                "referenced schedule spec does not bind the referenced length inventory"
            )
        try:
            expected_schedule = build_rollout_schedule(
                split_manifest,
                reference_length_inventory=inventory,
                probe_policies=schedule_spec["probe_policies"],
                domain_randomization_seeds=schedule_spec["domain_randomization_seeds"],
                phase_targets=schedule_spec["phase_targets"],
                repeats=schedule_spec["repeats"],
                rollout_id_prefix=schedule_spec.get("rollout_id_prefix", "lace-rollout"),
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(f"referenced schedule spec is invalid: {exc}") from exc
        if expected_schedule != artifact_manifest:
            raise ReadinessError(
                "schedule artifact is not the exact deterministic output of the referenced spec"
            )

        for field in ("artifact_mode", "scientific_use", "selected_motion_keys"):
            if inventory.get(field) != artifact_manifest.get(field):
                raise ReadinessError(
                    f"reference length inventory {field} does not bind the schedule artifact"
                )
        for inventory_field, schedule_field in (
            ("split_sha256", "split_sha256"),
            ("split_selection_sha256", "split_selection_sha256"),
        ):
            if inventory.get(inventory_field) != artifact_manifest.get(schedule_field):
                raise ReadinessError(
                    f"reference length inventory {inventory_field} does not bind the schedule"
                )
        schedule_steps = {
            str(record["motion_key"]): int(record["reference_num_steps"])
            for record in artifact_manifest["reference_num_steps"]
        }
        if reference_num_steps_by_motion(inventory) != schedule_steps:
            raise ReadinessError(
                "reference length inventory target frame counts do not bind schedule steps"
            )
    if artifact_manifest.get("kind") == INTERVENTION_PLAN_KIND:
        required_references = {
            "split_manifest",
            "panel_manifest",
            "intervention_protocol",
        }
        missing_references = sorted(required_references - set(loaded))
        if missing_references:
            raise ReadinessError(
                "intervention-plan lock is missing required referenced inputs: "
                f"{missing_references}"
            )
        if lock.get("schema_version") != 1:
            raise ReadinessError("intervention-plan lock schema_version must be 1")
        if lock.get("scientific_use") is not True:
            raise ReadinessError("intervention-plan lock scientific_use must be true")
        assert split_manifest is not None
        panel_manifest = loaded["panel_manifest"]
        protocol = loaded["intervention_protocol"]
        inputs = _require_mapping(lock.get("inputs"), "inputs")
        locked_panel_digest = _expected_sha256(
            inputs.get("panel_sha256"),
            "inputs.panel_sha256",
        )
        if panel_manifest.get("panel_sha256") != locked_panel_digest:
            raise ReadinessError("inputs.panel_sha256 does not match the panel artifact")
        locked_protocol_digest = _expected_sha256(
            inputs.get(INTERVENTION_PROTOCOL_DIGEST_FIELD),
            f"inputs.{INTERVENTION_PROTOCOL_DIGEST_FIELD}",
        )
        if protocol.get(INTERVENTION_PROTOCOL_DIGEST_FIELD) != locked_protocol_digest:
            raise ReadinessError(
                f"inputs.{INTERVENTION_PROTOCOL_DIGEST_FIELD} does not match the protocol"
            )
        try:
            validate_panel_manifest(
                panel_manifest,
                split_manifest=split_manifest,
                verify_digest=True,
            )
            validate_intervention_protocol(protocol, verify_digest=True)
            validate_intervention_plan(
                artifact_manifest,
                split_manifest=split_manifest,
                panel_manifest=panel_manifest,
                protocol=protocol,
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(
                f"intervention plan deterministic validation failed: {exc}"
            ) from exc
    if (
        artifact_manifest.get("kind") == ATLAS_KIND
        and artifact_manifest.get("artifact_mode") == "scientific"
    ):
        required_references = {
            "rollout_manifest",
            "split_manifest",
            "schedule_manifest",
            "reference_length_inventory",
            "analysis_protocol",
            "analysis_protocol_path",
            "analysis_protocol_lock",
            "analysis_protocol_lock_path",
        }
        missing_references = sorted(required_references - set(loaded))
        if missing_references:
            raise ReadinessError(
                "scientific atlas lock is missing deep-verification inputs: "
                f"{missing_references}"
            )
        if lock.get("kind") != "lace_atlas_artifact_lock" or lock.get("schema_version") != 2:
            raise ReadinessError(
                "scientific atlas requires lace_atlas_artifact_lock schema_version 2"
            )
        rollout_manifest = loaded["rollout_manifest"]
        schedule_manifest = loaded["schedule_manifest"]
        inventory = loaded["reference_length_inventory"]
        protocol = loaded["analysis_protocol"]
        protocol_path = loaded["analysis_protocol_path"]["path"]
        protocol_lock_path = loaded["analysis_protocol_lock_path"]["path"]
        if expected_analysis_protocol_lock_sha256 is None:
            raise ReadinessError(
                "scientific atlas readiness requires an independently supplied expected "
                "analysis protocol lock digest"
            )
        expected_protocol_lock_digest = _expected_sha256(
            expected_analysis_protocol_lock_sha256,
            "expected_analysis_protocol_lock_sha256",
        )
        if (
            loaded["analysis_protocol_lock"][ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
            != expected_protocol_lock_digest
        ):
            raise ReadinessError(
                "analysis protocol lock differs from the independently supplied "
                "preregistration digest"
            )
        assert split_manifest is not None
        try:
            validate_analysis_protocol(
                protocol,
                schedule_manifest=schedule_manifest,
                split_manifest=split_manifest,
                reference_length_inventory=inventory,
            )
            validate_scientific_atlas_deep(
                artifact_manifest,
                rollout_manifest,
                split_manifest,
                schedule_manifest,
                inventory,
                analysis_protocol=protocol,
                analysis_protocol_path=protocol_path,
                analysis_protocol_lock_path=protocol_lock_path,
                expected_analysis_protocol_lock_sha256=(expected_protocol_lock_digest),
                repo_root=repo_root,
            )
        except (TypeError, ValueError) as exc:
            raise ReadinessError(
                f"scientific atlas receipt-driven validation failed: {exc}"
            ) from exc
    return references


def verify_artifact_lock(
    lock_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    verify_referenced_manifests: bool = False,
    expected_analysis_protocol_lock_sha256: str | None = None,
    expected_headline_split_lock_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a supported lock against exact file and semantic digests."""

    unresolved_lock = _lexical_absolute_path(Path(lock_path).expanduser())
    _reject_symlink_components(unresolved_lock, "lock_path")
    resolved_lock = unresolved_lock.resolve()
    resolved_repo_root = Path(repo_root).expanduser().resolve()
    lock = _load_json_object(resolved_lock)
    lock_file_digest = sha256_file(resolved_lock)
    if lock.get("kind") == "lace_split_artifact_lock" and lock.get("scientific_use") is True:
        if expected_headline_split_lock_file_sha256 is None:
            raise ReadinessError(
                "scientific headline split readiness requires an independently retained "
                "expected split-lock file digest"
            )
        expected_lock_file_digest = _expected_sha256(
            expected_headline_split_lock_file_sha256,
            "expected_headline_split_lock_file_sha256",
        )
        if lock_file_digest != expected_lock_file_digest:
            raise ReadinessError(
                "headline split lock differs from the independently retained "
                "preregistration file digest"
            )
    artifact_lock = _require_mapping(lock.get("artifact"), "artifact")
    artifact_path = _resolve_lock_path(
        artifact_lock.get("path"), repo_root=resolved_repo_root, field="artifact.path"
    )
    file_digest = _checked_hash(
        artifact_path, artifact_lock.get("file_sha256"), "artifact.file_sha256"
    )
    manifest = _load_json_object(artifact_path)
    artifact_type, self_digests = _verify_locked_self_digests(
        manifest, artifact_lock, lock_kind=lock.get("kind")
    )
    references_required = (
        manifest.get("kind")
        in {
            SCHEDULE_KIND,
            INTERVENTION_PLAN_KIND,
        }
        or (manifest.get("kind") == ATLAS_KIND and manifest.get("artifact_mode") == "scientific")
        or (manifest.get("kind") == SPLIT_KIND and lock.get("scientific_use") is True)
    )
    references_verified = verify_referenced_manifests or references_required
    references = (
        _verify_referenced_manifests(
            lock,
            repo_root=resolved_repo_root,
            artifact_manifest=manifest,
            expected_analysis_protocol_lock_sha256=(expected_analysis_protocol_lock_sha256),
        )
        if references_verified
        else []
    )
    return {
        "lock_path": str(resolved_lock),
        "lock_file_sha256": lock_file_digest,
        "lock_kind": lock.get("kind"),
        "artifact_type": artifact_type,
        "artifact_path": str(artifact_path),
        "file_sha256": file_digest,
        "self_digests": self_digests,
        "referenced_manifests_verified": references_verified,
        "referenced_manifests_required": references_required,
        "referenced_manifests": references,
    }


def verify_readiness(
    storage_config_path: str | Path,
    lock_path: str | Path,
    *,
    minimum_free_bytes: int | None = None,
    repo_root: str | Path = REPO_ROOT,
    verify_referenced_manifests: bool = False,
    expected_analysis_protocol_lock_sha256: str | None = None,
    expected_headline_split_lock_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the complete storage and artifact readiness check."""

    storage = verify_storage_roots(
        storage_config_path,
        minimum_free_bytes=minimum_free_bytes,
    )
    artifact = verify_artifact_lock(
        lock_path,
        repo_root=repo_root,
        verify_referenced_manifests=verify_referenced_manifests,
        expected_analysis_protocol_lock_sha256=(expected_analysis_protocol_lock_sha256),
        expected_headline_split_lock_file_sha256=(expected_headline_split_lock_file_sha256),
    )
    return {"ready": True, "storage": storage, "artifact": artifact}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-config",
        type=Path,
        default=DEFAULT_STORAGE_CONFIG,
        help=f"Storage configuration (default: {DEFAULT_STORAGE_CONFIG})",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        required=True,
        help=(
            "Split, atlas, source-panel, rollout-schedule, or RQ1-intervention-plan "
            "artifact lock JSON"
        ),
    )
    parser.add_argument(
        "--minimum-free-bytes",
        type=int,
        help=f"Override the configured minimum (default fallback: {DEFAULT_MINIMUM_FREE_BYTES})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Base for repository-relative paths in the lock",
    )
    parser.add_argument(
        "--verify-referenced-manifests",
        action="store_true",
        help=(
            "Also verify referenced cohort/materialized/split/spec/inventory hashes "
            "and semantic bindings"
        ),
    )
    parser.add_argument(
        "--expected-analysis-protocol-lock-sha256",
        help=(
            "Independently retained preregistration digest for a scientific atlas "
            "protocol lock. Required when verifying a scientific atlas lock; this "
            "value must come from an external experiment registry or immutable run "
            "record, not from the lock being verified."
        ),
    )
    parser.add_argument(
        "--expected-headline-split-lock-file-sha256",
        help=(
            "Independently retained exact-file digest for a scientific headline split "
            "lock. Required for that lock type; obtain it from an external pre-outcome "
            "registry rather than the lock being verified."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = verify_readiness(
            args.storage_config,
            args.lock,
            minimum_free_bytes=args.minimum_free_bytes,
            repo_root=args.repo_root,
            verify_referenced_manifests=args.verify_referenced_manifests,
            expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
            expected_headline_split_lock_file_sha256=(
                args.expected_headline_split_lock_file_sha256
            ),
        )
    except (ReadinessError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
