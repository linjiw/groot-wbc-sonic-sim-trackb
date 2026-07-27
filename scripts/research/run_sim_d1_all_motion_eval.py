#!/usr/bin/env python3
"""Run the fail-closed SIM-D1 all-motion evaluation and classification.

This launcher is intentionally separate from the bounded SIM-M1 smoke harness.
SIM-D1 must evaluate every motion from its first frame to natural completion:
there is no motion-count limiter and ``max_eval_steps`` is forced to ``null``.
Expected coverage comes from a frozen, hashed manifest outside the dataset
directories; it is never inferred from the evaluation output.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from scripts.research.classify_sim_d1_headroom import classify_sim_d1  # noqa: E402, I001


_SCHEMA_VERSION = 1
_DATASET_KINDS = {"real", "sample_data", "synthetic"}
_DATASET_MANIFEST_SHA256_KINDS = {
    "file_bytes",
    "canonical_json_without_source_locations_v1",
}
_MOTION_ORDER_MODES = {"coverage_manifest"}
_REQUIRED_SPEC_KEYS = (
    "name",
    "dataset",
    "dataset_kind",
    "checkpoint",
    "checkpoint_sha256",
    "dataset_robot",
    "dataset_smpl",
    "coverage_manifest",
    "output_dir",
    "num_envs",
    "timeout_seconds",
)
_FORBIDDEN_EXTRA_OVERRIDE_FRAGMENTS = (
    "callbacks.im_eval.max_eval_steps",
    "debug=",
    "eval_callbacks=",
    "eval_output_dir=",
    "filter_motion_keys",
    "headless=",
    "max_unique_motions",
    "motion_file=",
    "num_envs=",
    "override_num_motions_to_load",
    "remove_motion_keys",
    "run_eval_loop=",
    "smpl_motion_file=",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc
    return value.lower()


def _dataset_hash(records: Sequence[tuple[str, str]]) -> str:
    """Hash a file inventory using the synthetic dataset builder's canonical format."""
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(records):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_dataset_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash semantic manifest content while excluding host-specific source locations."""
    canonical = json.loads(json.dumps(manifest))
    source = canonical.get("source")
    if isinstance(source, dict):
        for key in ("robot_dir", "smpl_dir"):
            if key in source:
                source[key] = f"<{key}>"
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_spec(spec: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_SPEC_KEYS if spec.get(key) in (None, "")]
    if missing:
        raise ValueError("spec missing required field(s): " + ", ".join(missing))
    if spec["dataset_kind"] not in _DATASET_KINDS:
        raise ValueError("dataset_kind must be one of: " + ", ".join(sorted(_DATASET_KINDS)))
    if isinstance(spec["num_envs"], bool) or int(spec["num_envs"]) <= 0:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(spec["timeout_seconds"], bool) or int(spec["timeout_seconds"]) <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    _validate_sha256(spec["checkpoint_sha256"], field="checkpoint_sha256")

    motion_order = spec.get("motion_order")
    if motion_order is not None:
        if motion_order not in _MOTION_ORDER_MODES:
            raise ValueError(
                "motion_order must be one of: "
                + ", ".join(sorted(_MOTION_ORDER_MODES))
            )
        if int(spec["num_envs"]) != 1:
            raise ValueError(
                "motion_order=coverage_manifest requires num_envs=1 so every "
                "motion uses the same evaluation slot"
            )

    dataset_manifest = spec.get("dataset_manifest")
    if dataset_manifest is not None:
        if not isinstance(dataset_manifest, dict):
            raise ValueError("dataset_manifest must be an object when provided")
        missing_manifest_fields = [
            key
            for key in ("path", "sha256", "paired_dataset_sha256")
            if dataset_manifest.get(key) in (None, "")
        ]
        if missing_manifest_fields:
            raise ValueError(
                "dataset_manifest missing required field(s): "
                + ", ".join(missing_manifest_fields)
            )
        if not isinstance(dataset_manifest["path"], str):
            raise ValueError("dataset_manifest.path must be a non-empty string")
        _validate_sha256(dataset_manifest["sha256"], field="dataset_manifest.sha256")
        sha256_kind = dataset_manifest.get("sha256_kind", "file_bytes")
        if sha256_kind not in _DATASET_MANIFEST_SHA256_KINDS:
            raise ValueError(
                "dataset_manifest.sha256_kind must be one of: "
                + ", ".join(sorted(_DATASET_MANIFEST_SHA256_KINDS))
            )
        _validate_sha256(
            dataset_manifest["paired_dataset_sha256"],
            field="dataset_manifest.paired_dataset_sha256",
        )

    extra_overrides = spec.get("extra_overrides", [])
    if not isinstance(extra_overrides, list) or not all(
        isinstance(value, str) and value for value in extra_overrides
    ):
        raise ValueError("extra_overrides must be a list of non-empty strings")
    for override in extra_overrides:
        lowered = override.lower()
        forbidden = [fragment for fragment in _FORBIDDEN_EXTRA_OVERRIDE_FRAGMENTS if fragment in lowered]
        if forbidden:
            raise ValueError(
                f"extra override {override!r} violates the all-motion contract: " + ", ".join(forbidden)
            )
        if motion_order == "coverage_manifest" and "sort_motion_keys" in lowered:
            raise ValueError(
                "motion_order=coverage_manifest owns sort_motion_keys; remove the "
                f"conflicting extra override {override!r}"
            )


def _discover_directory_motion_keys(directory: Path) -> list[str]:
    """Mirror MotionLib's recursive PKL key discovery without loading payloads."""
    paths = sorted(
        path
        for path in directory.rglob("*.pkl")
        if path.name != "metadata.pkl"
    )
    keys = [path.stem for path in paths]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            f"dataset contains duplicate motion-key stems: {duplicate_keys}"
        )
    return keys


def _coverage_manifest_order_overrides(
    *,
    coverage: dict[str, Any],
    dataset_robot: Path,
    dataset_smpl: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Build an exact-list ordering override after proving it selects all data."""
    expected_keys = list(coverage["motion_keys"])
    expected_set = set(expected_keys)
    for modality, directory in (
        ("robot", dataset_robot),
        ("smpl", dataset_smpl),
    ):
        keys = _discover_directory_motion_keys(directory)
        if len(keys) != len(expected_keys) or set(keys) != expected_set:
            missing = sorted(expected_set - set(keys))
            unexpected = sorted(set(keys) - expected_set)
            raise ValueError(
                f"{modality} dataset does not exactly match the frozen coverage "
                f"manifest for ordered evaluation: missing={missing}, "
                f"unexpected={unexpected}, discovered_count={len(keys)}, "
                f"expected_count={len(expected_keys)}"
            )

    encoded_order = json.dumps(expected_keys, separators=(",", ":"))
    overrides = [
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
        + encoded_order,
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=false",
    ]
    provenance = {
        "mode": "coverage_manifest",
        "motion_keys": expected_keys,
        "motion_count": len(expected_keys),
        "coverage_manifest_path": coverage["path"],
        "coverage_manifest_sha256": coverage["sha256"],
        "robot_exact_key_set_verified": True,
        "smpl_exact_key_set_verified": True,
        "exact_list_filter_is_order_only": True,
    }
    return overrides, provenance


def _load_coverage_manifest(path: Path, dataset: str) -> dict[str, Any]:
    manifest = _load_json_object(path)
    if manifest.get("dataset") != dataset:
        raise ValueError(
            f"coverage manifest dataset {manifest.get('dataset')!r} does not match spec dataset {dataset!r}"
        )
    keys = manifest.get("motion_keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("coverage manifest motion_keys must be a non-empty list")
    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError("coverage manifest motion_keys must contain non-empty strings")
    if len(keys) != len(set(keys)):
        raise ValueError("coverage manifest motion_keys contains duplicates")
    expected_count = manifest.get("expected_motion_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("coverage manifest expected_motion_count must be an integer")
    if expected_count != len(keys):
        raise ValueError("coverage manifest expected_motion_count does not match motion_keys length")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("coverage manifest provenance must be an object")
    if provenance.get("independent_of_metrics_eval") is not True:
        raise ValueError("coverage manifest must assert independent_of_metrics_eval=true")
    source = provenance.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("coverage manifest provenance.source must be non-empty")
    dataset_manifest = provenance.get("dataset_manifest")
    if dataset_manifest is not None:
        if not isinstance(dataset_manifest, dict):
            raise ValueError("coverage provenance.dataset_manifest must be an object")
        missing = [
            key
            for key in ("path", "sha256", "paired_dataset_sha256")
            if dataset_manifest.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "coverage provenance.dataset_manifest missing required field(s): "
                + ", ".join(missing)
            )
        if not isinstance(dataset_manifest["path"], str):
            raise ValueError("coverage provenance.dataset_manifest.path must be a string")
        _validate_sha256(
            dataset_manifest["sha256"],
            field="coverage provenance.dataset_manifest.sha256",
        )
        sha256_kind = dataset_manifest.get("sha256_kind", "file_bytes")
        if sha256_kind not in _DATASET_MANIFEST_SHA256_KINDS:
            raise ValueError(
                "coverage provenance.dataset_manifest.sha256_kind must be one of: "
                + ", ".join(sorted(_DATASET_MANIFEST_SHA256_KINDS))
            )
        _validate_sha256(
            dataset_manifest["paired_dataset_sha256"],
            field="coverage provenance.dataset_manifest.paired_dataset_sha256",
        )
    return {
        "motion_keys": list(keys),
        "expected_motion_count": expected_count,
        "path": str(path),
        "sha256": _sha256(path),
        "provenance": provenance,
    }


def _load_and_verify_dataset_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
    manifest_sha256_kind: str,
    expected_paired_dataset_sha256: str,
    dataset_robot: Path,
    dataset_smpl: Path,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Verify a paired robot/SMPL inventory against the files to be evaluated."""
    manifest = _load_json_object(path)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"dataset manifest schema_version must be {_SCHEMA_VERSION}"
        )
    if not isinstance(manifest.get("kind"), str) or not manifest["kind"]:
        raise ValueError("dataset manifest kind must be a non-empty string")
    if not isinstance(manifest.get("generator"), str) or not manifest["generator"]:
        raise ValueError("dataset manifest generator must be a non-empty string")
    if manifest_sha256_kind == "file_bytes":
        actual_manifest_sha256 = _sha256(path)
    elif manifest_sha256_kind == "canonical_json_without_source_locations_v1":
        actual_manifest_sha256 = _canonical_dataset_manifest_sha256(manifest)
    else:  # guarded by _validate_spec; defensive for direct helper callers
        raise ValueError(f"unsupported dataset manifest SHA-256 kind: {manifest_sha256_kind}")
    expected_manifest_sha256 = _validate_sha256(
        expected_manifest_sha256, field="dataset_manifest.sha256"
    )
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "dataset manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("dataset manifest output must be an object")
    motion_keys = output.get("motion_keys")
    variants = output.get("variants")
    motion_count = output.get("motion_count")
    if not isinstance(motion_keys, list) or not motion_keys:
        raise ValueError("dataset manifest output.motion_keys must be a non-empty list")
    if not all(isinstance(key, str) and key for key in motion_keys):
        raise ValueError("dataset manifest output.motion_keys must contain non-empty strings")
    if len(motion_keys) != len(set(motion_keys)):
        raise ValueError("dataset manifest output.motion_keys contains duplicates")
    if isinstance(motion_count, bool) or not isinstance(motion_count, int):
        raise ValueError("dataset manifest output.motion_count must be an integer")
    if motion_count != len(motion_keys):
        raise ValueError("dataset manifest output.motion_count does not match motion_keys")
    if not isinstance(variants, list) or len(variants) != motion_count:
        raise ValueError("dataset manifest output.variants must cover every motion")
    if set(motion_keys) != set(coverage["motion_keys"]):
        raise ValueError("dataset manifest motion keys do not match the coverage manifest")
    if motion_count != coverage["expected_motion_count"]:
        raise ValueError("dataset manifest motion count does not match the coverage manifest")

    roots = {"robot": dataset_robot.resolve(), "smpl": dataset_smpl.resolve()}
    manifest_dirs: dict[str, Path] = {}
    for modality in roots:
        relative_dir = output.get(f"{modality}_dir")
        if not isinstance(relative_dir, str) or not relative_dir:
            raise ValueError(f"dataset manifest output.{modality}_dir must be non-empty")
        relative_dir_path = Path(relative_dir)
        if relative_dir_path.is_absolute() or ".." in relative_dir_path.parts:
            raise ValueError(f"dataset manifest output.{modality}_dir must be relative")
        manifest_dirs[modality] = relative_dir_path

    records: dict[str, list[tuple[str, str]]] = {"robot": [], "smpl": []}
    variant_keys: list[str] = []
    seen_paths: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"dataset manifest output.variants[{index}] must be an object")
        motion_key = variant.get("motion_key")
        if not isinstance(motion_key, str) or not motion_key:
            raise ValueError(
                f"dataset manifest output.variants[{index}].motion_key must be non-empty"
            )
        variant_keys.append(motion_key)
        for modality in roots:
            file_record = variant.get(modality)
            if not isinstance(file_record, dict):
                raise ValueError(
                    f"dataset manifest output.variants[{index}].{modality} must be an object"
                )
            relative_value = file_record.get("path")
            if not isinstance(relative_value, str) or not relative_value:
                raise ValueError(
                    f"dataset manifest output.variants[{index}].{modality}.path must be non-empty"
                )
            relative_path = Path(relative_value)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("dataset manifest file paths must be relative and non-traversing")
            relative_posix = relative_path.as_posix()
            if relative_posix in seen_paths:
                raise ValueError(f"dataset manifest contains duplicate file path: {relative_posix}")
            seen_paths.add(relative_posix)
            try:
                dataset_relative_path = relative_path.relative_to(manifest_dirs[modality])
            except ValueError as exc:
                raise ValueError(
                    f"dataset manifest {modality} file is outside output.{modality}_dir: "
                    f"{relative_posix}"
                ) from exc
            actual_path = (roots[modality] / dataset_relative_path).resolve()
            if not _is_within(actual_path, roots[modality]) or not actual_path.is_file():
                raise ValueError(f"dataset manifest file is missing or outside its dataset: {actual_path}")
            expected_file_hash = _validate_sha256(
                file_record.get("sha256"),
                field=f"dataset manifest {relative_posix} sha256",
            )
            actual_file_hash = _sha256(actual_path)
            if actual_file_hash != expected_file_hash:
                raise ValueError(
                    f"dataset file SHA-256 mismatch for {actual_path}: "
                    f"expected {expected_file_hash}, got {actual_file_hash}"
                )
            records[modality].append((relative_posix, actual_file_hash))

    if variant_keys != motion_keys:
        raise ValueError("dataset manifest variant order does not match output.motion_keys")

    actual_hashes = {
        "robot_dataset_sha256": _dataset_hash(records["robot"]),
        "smpl_dataset_sha256": _dataset_hash(records["smpl"]),
        "paired_dataset_sha256": _dataset_hash(records["robot"] + records["smpl"]),
    }
    for field, actual_hash in actual_hashes.items():
        declared_hash = _validate_sha256(
            output.get(field), field=f"dataset manifest output.{field}"
        )
        if declared_hash != actual_hash:
            raise ValueError(
                f"dataset manifest output.{field} mismatch: "
                f"declared {declared_hash}, actual {actual_hash}"
            )

    expected_paired_dataset_sha256 = _validate_sha256(
        expected_paired_dataset_sha256,
        field="dataset_manifest.paired_dataset_sha256",
    )
    if actual_hashes["paired_dataset_sha256"] != expected_paired_dataset_sha256:
        raise ValueError(
            "paired dataset SHA-256 mismatch: "
            f"expected {expected_paired_dataset_sha256}, "
            f"got {actual_hashes['paired_dataset_sha256']}"
        )

    return {
        "path": str(path),
        "sha256": actual_manifest_sha256,
        "sha256_kind": manifest_sha256_kind,
        "kind": manifest["kind"],
        "generator": manifest["generator"],
        "paired_dataset_sha256": actual_hashes["paired_dataset_sha256"],
        "robot_dataset_sha256": actual_hashes["robot_dataset_sha256"],
        "smpl_dataset_sha256": actual_hashes["smpl_dataset_sha256"],
        "motion_count": motion_count,
        "motion_keys": list(motion_keys),
        "verified_file_count": len(records["robot"]) + len(records["smpl"]),
        "content_hashes_verified": True,
    }


def build_eval_command(
    *,
    python_executable: str,
    repo_root: Path,
    checkpoint: Path,
    dataset_robot: Path,
    dataset_smpl: Path,
    eval_output_dir: Path,
    num_envs: int,
    seed: int,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build the fixed all-motion/full-sequence Isaac Lab evaluation command."""
    command = [
        python_executable,
        str(repo_root / "gear_sonic/eval_agent_trl.py"),
        f"+checkpoint={checkpoint}",
        "+headless=True",
        f"++seed={seed}",
        "++eval_callbacks=im_eval",
        "++run_eval_loop=False",
        f"++num_envs={num_envs}",
        f"++eval_output_dir={eval_output_dir}",
        "++callbacks.im_eval.max_eval_steps=null",
        (f"++manager_env.commands.motion.motion_lib_cfg.motion_file={dataset_robot}"),
        (f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={dataset_smpl}"),
        "+manager_env/terminations=tracking/eval",
        *extra_overrides,
    ]
    joined = " ".join(command).lower()
    if "max_unique_motions" in joined:
        raise ValueError("SIM-D1 command must never set max_unique_motions")
    finite_step_overrides = [
        value for value in command if "max_eval_steps" in value.lower() and not value.lower().endswith("=null")
    ]
    if finite_step_overrides:
        raise ValueError("SIM-D1 command must never set a finite max_eval_steps")
    return command


def _file_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _fresh_snapshot(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> bool:
    # A before/after snapshot is stronger than comparing wall-clock timestamps:
    # some filesystems quantize mtime a few milliseconds behind time.time_ns().
    if after is None:
        return False
    if before is None:
        return True
    return after != before


def _new_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _archive_file(
    path: Path,
    *,
    archive_root: Path,
    attempt_id: str,
    category: str,
    label: str,
    reason: str,
) -> dict[str, Any] | None:
    """Atomically preserve a canonical output and record why it moved."""
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"refusing to archive non-file canonical output: {path}")

    snapshot = _file_snapshot(path)
    destination_dir = archive_root / attempt_id / category
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{label}__{path.name}"
    if destination.exists():
        raise ValueError(f"refusing to overwrite archived output: {destination}")
    path.rename(destination)

    record = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "sim_d1_archived_output",
        "attempt_id": attempt_id,
        "category": category,
        "label": label,
        "reason": reason,
        "original_path": str(path),
        "archived_path": str(destination),
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
    }
    provenance_path = destination.with_name(f"{destination.name}.provenance.json")
    record["provenance_path"] = str(provenance_path)
    _write_json(provenance_path, record)
    return record


def _archive_existing_canonical_outputs(
    *,
    output_dir: Path,
    attempt_id: str,
    classification_path: Path,
    result_path: Path,
    eval_log_path: Path,
    metrics_eval_path: Path,
    plan_path: Path,
) -> list[dict[str, Any]]:
    archive_root = output_dir / "archive"
    records: list[dict[str, Any]] = []
    for label, path in (
        ("classification", classification_path),
        ("result", result_path),
        ("eval_log", eval_log_path),
        ("metrics_eval", metrics_eval_path),
        ("run_plan", plan_path),
    ):
        record = _archive_file(
            path,
            archive_root=archive_root,
            attempt_id=attempt_id,
            category="prior",
            label=label,
            reason="superseded_by_new_execute_attempt",
        )
        if record is not None:
            records.append(record)
    return records


def _quarantine_failed_metrics(
    metrics_eval_path: Path,
    *,
    output_dir: Path,
    attempt_id: str,
    reason: str,
) -> list[dict[str, Any]]:
    record = _archive_file(
        metrics_eval_path,
        archive_root=output_dir / "archive",
        attempt_id=attempt_id,
        category="failed_attempt",
        label="metrics_eval",
        reason=reason,
    )
    return [] if record is None else [record]


def _apply_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(spec)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def run_sim_d1_all_motion_eval(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    python_executable: str = sys.executable,
    dry_run: bool,
) -> dict[str, Any]:
    """Materialize a dry-run plan or execute one SIM-D1 evaluation."""
    _validate_spec(spec)
    repo_root = repo_root.expanduser().resolve()
    checkpoint = _resolve_path(spec["checkpoint"], repo_root)
    dataset_robot = _resolve_path(spec["dataset_robot"], repo_root)
    dataset_smpl = _resolve_path(spec["dataset_smpl"], repo_root)
    coverage_path = _resolve_path(spec["coverage_manifest"], repo_root)
    output_dir = _resolve_path(spec["output_dir"], repo_root)
    eval_output_dir = output_dir / "eval_metrics"
    metrics_eval_path = eval_output_dir / "metrics_eval.json"
    classification_path = output_dir / "classification.json"
    plan_path = output_dir / ("dry_run_plan.json" if dry_run else "run_plan.json")
    result_path = output_dir / "result.json"
    eval_log_path = output_dir / "eval.log"
    attempt_id = None if dry_run else _new_attempt_id()

    # This entire phase is read-only. Canonical outputs are not archived until
    # every path, coverage, checkpoint, and optional paired-dataset hash passes.
    resource_errors: list[str] = []
    if not checkpoint.is_file():
        resource_errors.append(f"checkpoint is not a file: {checkpoint}")
    if not dataset_robot.is_dir():
        resource_errors.append(f"robot dataset is not a directory: {dataset_robot}")
    if not dataset_smpl.is_dir():
        resource_errors.append(f"SMPL dataset is not a directory: {dataset_smpl}")

    coverage: dict[str, Any] | None = None
    if not coverage_path.is_file():
        resource_errors.append(f"coverage manifest is not a file: {coverage_path}")
    else:
        try:
            coverage = _load_coverage_manifest(coverage_path, str(spec["dataset"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            resource_errors.append(f"coverage manifest validation failed: {exc}")
        if coverage is not None:
            for dataset_dir in (dataset_robot, dataset_smpl):
                if _is_within(coverage_path, dataset_dir):
                    resource_errors.append(
                        "coverage manifest must live outside both dataset directories"
                    )
                    break
            if _is_within(coverage_path, output_dir):
                resource_errors.append(
                    "coverage manifest must not be generated inside the run output"
                )

    actual_checkpoint_hash: str | None = None
    if checkpoint.is_file():
        try:
            actual_checkpoint_hash = _sha256(checkpoint)
        except OSError as exc:
            resource_errors.append(f"checkpoint hashing failed: {exc}")
        else:
            expected_checkpoint_hash = spec["checkpoint_sha256"].lower()
            if actual_checkpoint_hash != expected_checkpoint_hash:
                resource_errors.append(
                    "checkpoint SHA-256 mismatch: "
                    f"expected {expected_checkpoint_hash}, got {actual_checkpoint_hash}"
                )

    dataset_manifest_verification: dict[str, Any] | None = None
    dataset_manifest_spec = spec.get("dataset_manifest")
    dataset_manifest_path: Path | None = None
    if dataset_manifest_spec is not None:
        dataset_manifest_path = _resolve_path(dataset_manifest_spec["path"], repo_root)
        if not dataset_manifest_path.is_file():
            resource_errors.append(
                f"dataset manifest is not a file: {dataset_manifest_path}"
            )
        elif coverage is not None and dataset_robot.is_dir() and dataset_smpl.is_dir():
            try:
                dataset_manifest_verification = _load_and_verify_dataset_manifest(
                    dataset_manifest_path,
                    expected_manifest_sha256=dataset_manifest_spec["sha256"],
                    manifest_sha256_kind=dataset_manifest_spec.get(
                        "sha256_kind", "file_bytes"
                    ),
                    expected_paired_dataset_sha256=dataset_manifest_spec[
                        "paired_dataset_sha256"
                    ],
                    dataset_robot=dataset_robot,
                    dataset_smpl=dataset_smpl,
                    coverage=coverage,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                resource_errors.append(f"dataset manifest verification failed: {exc}")

    coverage_binding = (
        coverage.get("provenance", {}).get("dataset_manifest")
        if coverage is not None
        else None
    )
    if coverage_binding is not None:
        if dataset_manifest_spec is None or dataset_manifest_path is None:
            resource_errors.append(
                "coverage binds a dataset manifest but the eval spec does not"
            )
        else:
            binding_path = _resolve_path(coverage_binding["path"], repo_root)
            comparisons = (
                ("path", str(binding_path), str(dataset_manifest_path)),
                (
                    "sha256",
                    coverage_binding["sha256"].lower(),
                    dataset_manifest_spec["sha256"].lower(),
                ),
                (
                    "sha256_kind",
                    coverage_binding.get("sha256_kind", "file_bytes"),
                    dataset_manifest_spec.get("sha256_kind", "file_bytes"),
                ),
                (
                    "paired_dataset_sha256",
                    coverage_binding["paired_dataset_sha256"].lower(),
                    dataset_manifest_spec["paired_dataset_sha256"].lower(),
                ),
            )
            for field, coverage_value, spec_value in comparisons:
                if coverage_value != spec_value:
                    resource_errors.append(
                        f"coverage/spec dataset manifest {field} mismatch: "
                        f"coverage={coverage_value}, spec={spec_value}"
                    )

    motion_order_overrides: list[str] = []
    motion_order_provenance: dict[str, Any] | None = None
    if (
        spec.get("motion_order") == "coverage_manifest"
        and coverage is not None
        and dataset_robot.is_dir()
        and dataset_smpl.is_dir()
    ):
        try:
            (
                motion_order_overrides,
                motion_order_provenance,
            ) = _coverage_manifest_order_overrides(
                coverage=coverage,
                dataset_robot=dataset_robot,
                dataset_smpl=dataset_smpl,
            )
        except (OSError, ValueError) as exc:
            resource_errors.append(f"motion-order validation failed: {exc}")

    command = build_eval_command(
        python_executable=python_executable,
        repo_root=repo_root,
        checkpoint=checkpoint,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
        eval_output_dir=eval_output_dir,
        num_envs=int(spec["num_envs"]),
        seed=int(spec.get("seed", 0)),
        extra_overrides=[
            *spec.get("extra_overrides", []),
            *motion_order_overrides,
        ],
    )
    plan: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "sim_d1_all_motion_eval_plan",
        "name": spec["name"],
        "dataset": spec["dataset"],
        "dataset_kind": spec["dataset_kind"],
        "mode": "dry_run" if dry_run else "execute",
        "attempt_id": attempt_id,
        "archived_prior_outputs": [],
        "repo_root": str(repo_root),
        "command": command,
        "command_display": shlex.join(command),
        "environment": {
            "set": {
                "HYDRA_FULL_ERROR": "1",
                "LOGURU_LEVEL": "ERROR",
                "PYTHONNOUSERSITE": "1",
                "WANDB_MODE": "disabled",
            },
            "unset": ["PYTHONPATH"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "expected_sha256": spec["checkpoint_sha256"].lower(),
            "actual_sha256": actual_checkpoint_hash,
        },
        "datasets": {
            "robot": str(dataset_robot),
            "smpl": str(dataset_smpl),
        },
        "coverage_manifest": coverage,
        "motion_order": motion_order_provenance,
        "dataset_manifest": dataset_manifest_verification,
        "outputs": {
            "directory": str(output_dir),
            "eval_log": str(eval_log_path),
            "metrics_eval_json": str(metrics_eval_path),
            "classification_json": str(classification_path),
            "result_json": str(result_path),
        },
        "safeguards": {
            "all_motion_limiter_absent": not any("max_unique_motions" in value.lower() for value in command),
            "coverage_order_exact_filter_verified": (
                spec.get("motion_order") != "coverage_manifest"
                or motion_order_provenance is not None
            ),
            "full_sequences": all(
                "max_eval_steps" not in value.lower() or value.lower().endswith("=null") for value in command
            ),
            "classification_requires_fresh_metrics": True,
            "coverage_is_independent_and_hashed": coverage is not None,
            "paired_dataset_content_is_hashed": (
                dataset_manifest_spec is not None
                and dataset_manifest_verification is not None
            ),
            "canonical_classification_only_after_success": True,
            "dry_run_leaves_canonical_outputs_untouched": dry_run,
            "read_only_preflight_precedes_canonical_archive": True,
            "canonical_outputs_preserved_on_preflight_failure": True,
            "wandb_disabled": True,
        },
        "preflight_errors": resource_errors,
        "ready_to_execute": (
            not resource_errors
            and coverage is not None
            and actual_checkpoint_hash is not None
            and (
                dataset_manifest_spec is None
                or dataset_manifest_verification is not None
            )
        ),
    }
    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(plan_path, plan)
        return {**plan, "exit_code": 0, "plan_path": str(plan_path)}
    assert attempt_id is not None

    if not plan["ready_to_execute"]:
        # Do not write run_plan.json or result.json here: either may be a prior
        # canonical scientific artifact that this invalid attempt must preserve.
        return {
            "schema_version": _SCHEMA_VERSION,
            "kind": "sim_d1_all_motion_eval_result",
            "ok": False,
            "status": "preflight_failed",
            "errors": resource_errors,
            "attempt_id": attempt_id,
            "archived_prior_outputs": [],
            "plan_path": None,
            "result_json": None,
            "canonical_outputs_preserved": True,
            "exit_code": 2,
        }

    assert coverage is not None
    assert actual_checkpoint_hash is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _file_snapshot(metrics_eval_path)
    archived_prior_outputs = _archive_existing_canonical_outputs(
        output_dir=output_dir,
        attempt_id=attempt_id,
        classification_path=classification_path,
        result_path=result_path,
        eval_log_path=eval_log_path,
        metrics_eval_path=metrics_eval_path,
        plan_path=plan_path,
    )
    plan["archived_prior_outputs"] = archived_prior_outputs
    _write_json(plan_path, plan)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HYDRA_FULL_ERROR": "1",
            "LOGURU_LEVEL": "ERROR",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    timed_out = False
    returncode: int
    with eval_log_path.open("w", encoding="utf-8") as eval_log:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(spec["timeout_seconds"]),
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            eval_log.write(f"\n[SIM_D1_TIMEOUT] timed out after {spec['timeout_seconds']}s: {exc}\n")

    base_result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "sim_d1_all_motion_eval_result",
        "name": spec["name"],
        "dataset": spec["dataset"],
        "attempt_id": attempt_id,
        "archived_prior_outputs": archived_prior_outputs,
        "eval_returncode": returncode,
        "timed_out": timed_out,
        "plan_path": str(plan_path),
        "eval_log": str(eval_log_path),
        "metrics_eval_json": str(metrics_eval_path),
        "classification_json": str(classification_path),
        "result_json": str(result_path),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": actual_checkpoint_hash,
        },
        "coverage_manifest": coverage,
        "dataset_manifest": dataset_manifest_verification,
    }
    if returncode != 0:
        quarantined_outputs = _quarantine_failed_metrics(
            metrics_eval_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="evaluation_returned_nonzero_or_timed_out",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "evaluation_failed",
            "quarantined_attempt_outputs": quarantined_outputs,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    after = _file_snapshot(metrics_eval_path)
    if not _fresh_snapshot(before, after):
        quarantined_outputs = _quarantine_failed_metrics(
            metrics_eval_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="metrics_missing_or_stale",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "metrics_missing_or_stale",
            "metrics_before": before,
            "metrics_after": after,
            "quarantined_attempt_outputs": quarantined_outputs,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    try:
        metrics_eval = _load_json_object(metrics_eval_path)
        classification = classify_sim_d1(
            metrics_eval,
            dataset=str(spec["dataset"]),
            dataset_kind=str(spec["dataset_kind"]),
            expected_motion_count=coverage["expected_motion_count"],
            expected_motion_keys=coverage["motion_keys"],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        quarantined_outputs = _quarantine_failed_metrics(
            metrics_eval_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="metrics_failed_sim_d1_classification",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "classification_failed",
            "error": str(exc),
            "metrics_after": after,
            "quarantined_attempt_outputs": quarantined_outputs,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    classification["source"] = {
        "metrics_eval_path": str(metrics_eval_path),
        "metrics_eval_sha256": after["sha256"],
        "eval_log_path": str(eval_log_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_hash,
        "coverage_manifest_path": coverage["path"],
        "coverage_manifest_sha256": coverage["sha256"],
        "expected_motion_count": coverage["expected_motion_count"],
        "dataset_manifest": dataset_manifest_verification,
        "dataset_robot": str(dataset_robot),
        "dataset_smpl": str(dataset_smpl),
        "execute_attempt_id": attempt_id,
    }
    _write_json(classification_path, classification)
    result = {
        **base_result,
        "ok": True,
        "status": "classified",
        "verdict": classification["verdict"],
        "eligible_for_effect_experiment": classification["eligible_for_effect_experiment"],
        "metrics_after": after,
        "exit_code": 0,
    }
    _write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-kind", choices=sorted(_DATASET_KINDS))
    parser.add_argument("--dataset-robot")
    parser.add_argument("--dataset-smpl")
    parser.add_argument("--coverage-manifest")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args()

    try:
        spec = _apply_overrides(
            _load_json_object(args.spec),
            {
                "output_dir": args.output_dir,
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": args.checkpoint_sha256,
                "dataset": args.dataset,
                "dataset_kind": args.dataset_kind,
                "dataset_robot": args.dataset_robot,
                "dataset_smpl": args.dataset_smpl,
                "coverage_manifest": args.coverage_manifest,
                "num_envs": args.num_envs,
                "timeout_seconds": args.timeout_seconds,
            },
        )
        result = run_sim_d1_all_motion_eval(
            spec,
            repo_root=args.repo_root,
            python_executable=args.python_executable,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"wrote SIM-D1 dry-run plan to {result['plan_path']}")
        print(f"ready_to_execute={result['ready_to_execute']}")
        for error in result["preflight_errors"]:
            print(f"preflight: {error}")
    else:
        result_location = result.get("result_json") or "not written (canonical artifacts preserved)"
        print(f"SIM-D1 status={result['status']} ok={result['ok']} result={result_location}")
        if result.get("verdict") is not None:
            print(f"scientific verdict={result['verdict']}")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
