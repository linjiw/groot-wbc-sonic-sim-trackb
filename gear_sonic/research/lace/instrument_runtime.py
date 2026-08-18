"""Runtime handshake between a locked LACE launch plan and its instrument.

Some scientific inputs do not exist before Isaac Lab starts: the checkpoint-
merged Hydra configuration, installed one-pass termination contract, resolved
event-manager configuration, and live package/runtime fingerprint.  The CPU
launcher therefore freezes a source-bound handshake.  The eval process verifies
that handshake after initialization, materializes the instrument before policy
rollout, and finally binds completed recorder rows atomically.

This module has no eager Isaac dependency.  The explicit-payload path and all
artifact validation remain CPU-testable; only ``materialize_live_instrument``
introspects a running environment.

Crash recovery is deliberately narrow. Exact instrument/binding sidecars and
an exact enriched-without-receipt half-state are recoverable. A partial raw
recorder JSONL is immutable and is never resumed or truncated; that launch is
an irreversible tombstone for its preregistered cell. It is preserved for
forensic inspection and invalidates the study execution; scientific cells are
never retried under alternate paths or cherry-picked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    validate_analysis_protocol,
    validate_instrument_against_analysis_protocol,
)
from gear_sonic.research.lace.instrument import (
    INSTRUMENT_DIGEST_FIELD,
    MEASUREMENT_FAMILY_DIGEST_FIELD,
    RUNTIME_RNG_SEED_SEMANTICS,
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    build_measurement_family,
    build_scientific_instrument,
    build_source_bundle,
    episode_instrument_sha256,
    episode_measurement_family_sha256,
    validate_source_bundle,
)
from gear_sonic.research.lace.schedule import DOMAIN_RANDOMIZATION_SEED_SEMANTICS
from gear_sonic.research.lace.schema import (
    canonical_sha256,
    validate_robot_contract_readback,
    validate_scientific_probe_episode,
)

RUNTIME_HANDSHAKE_KIND = "lace_scientific_instrument_runtime_handshake"
RUNTIME_HANDSHAKE_SCHEMA_VERSION = 1
RUNTIME_HANDSHAKE_DIGEST_FIELD = "runtime_handshake_sha256"
PLAN_BINDING_KIND = "lace_scientific_instrument_plan_binding"
PLAN_BINDING_SCHEMA_VERSION = 1
INSTRUMENT_BINDING_KIND = "lace_scientific_instrument_runtime_binding"
INSTRUMENT_BINDING_SCHEMA_VERSION = 1
INSTRUMENT_BINDING_DIGEST_FIELD = "instrument_binding_sha256"
ROLLOUT_BINDING_KIND = "lace_scientific_rollout_binding"
ROLLOUT_BINDING_SCHEMA_VERSION = 1
ROLLOUT_BINDING_DIGEST_FIELD = "rollout_binding_sha256"

_INSTRUMENT_BINDING_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "launch_plan_sha256",
    "launch_plan_file_sha256",
    "runtime_handshake_sha256",
    "schedule_sha256",
    "checkpoint_sha256",
    "loaded_checkpoint_bundle",
    "dataset_binding_sha256",
    "rollout_output_path",
    "instrumented_rollout_output_path",
    "instrument_output_path",
    "instrument_file_sha256",
    "instrument_manifest_sha256",
    "episode_instrument_sha256",
    "measurement_family",
    "measurement_family_manifest_sha256",
    "episode_measurement_family_sha256",
    "source_bundle_sha256",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    "analysis_protocol_sha256",
    INSTRUMENT_BINDING_DIGEST_FIELD,
}
_CHECKPOINT_BUNDLE_FIELDS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "config_path",
    "config_sha256",
}
_ARTIFACT_BINDING_FIELDS = {"path", "file_sha256"}
_CELL_FIELDS = {
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "repeat_index",
}
_ROLLOUT_ARTIFACT_NAMES = {
    "runtime_handshake",
    "launch_plan",
    "checkpoint",
    "checkpoint_config",
    "instrument",
    "instrument_binding",
    "raw_rollouts",
    "instrumented_rollouts",
    "schedule_lock",
    "schedule",
    "schedule_spec",
    "split_manifest",
    "reference_length_inventory",
    "analysis_protocol",
}
_ROLLOUT_BINDING_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "cell",
    "schedule_sha256",
    "split_sha256",
    "split_selection_sha256",
    "rollout_ids",
    "rollout_count",
    "rollout_order",
    "launch_plan_sha256",
    "runtime_handshake_sha256",
    "instrument_binding_sha256",
    "instrument_manifest_sha256",
    "episode_instrument_sha256",
    "measurement_family",
    "measurement_family_manifest_sha256",
    "episode_measurement_family_sha256",
    "source_bundle_sha256",
    "dataset_binding_sha256",
    "robot_contract_readback_sha256",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    "analysis_protocol_sha256",
    "artifacts",
    ROLLOUT_BINDING_DIGEST_FIELD,
}

_IGNORED_SOURCE_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
_IGNORED_SOURCE_SUFFIXES = {".pyc", ".pyo"}
_HANDSHAKE_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "schedule_sha256",
    "checkpoint_sha256",
    "probe_policy_id",
    "rollout_ids",
    "motion_count",
    "rollout_output_path",
    "instrumented_rollout_output_path",
    "launch_plan_path",
    "instrument_output_path",
    "instrument_binding_output_path",
    "rollout_binding_output_path",
    "git_commit",
    "source_bundle",
    "source_bundle_sha256",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    "analysis_protocol_sha256",
    RUNTIME_HANDSHAKE_DIGEST_FIELD,
}
_PLAN_BINDING_FIELDS = {
    "kind",
    "schema_version",
    "required",
    "state",
    "handshake_path",
    "handshake_file_sha256",
    "runtime_handshake_sha256",
    "instrument_output_path",
    "instrument_binding_output_path",
    "instrumented_rollout_output_path",
    "rollout_binding_output_path",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    "analysis_protocol_sha256",
}
_SCIENTIFIC_LAUNCH_PLAN_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "schedule_lock_path",
    "schedule_path",
    "schedule_file_sha256",
    "schedule_sha256",
    "schedule_spec_path",
    "cell",
    "num_envs",
    "motion_keys",
    "reference_num_steps",
    "rollout_ids",
    "atlas_probe_assignments",
    "checkpoint_path",
    "rollout_output_path",
    "dataset_binding",
    "instrumentation_invariants",
    "eval_entrypoint",
    "hydra_overrides",
    "instrument_runtime",
    "passthrough_hydra_args",
    "command",
    "checkpoint_bundle",
    "launch_environment",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    "analysis_protocol_sha256",
    "analysis_protocol_lock_path",
    "analysis_protocol_lock_file_sha256",
    "analysis_protocol_lock_sha256",
    "launch_plan_sha256",
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


def validate_scientific_cache_environment(
    raw: Mapping[str, Any],
    *,
    launch_cache_token: str | None = None,
) -> None:
    """Validate the exact process-cache paths bound by a scientific launch plan."""

    _require(isinstance(raw, Mapping), "scientific cache environment must be a mapping")
    environment = dict(raw)
    _require(
        set(environment) == set(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS),
        "scientific cache environment fields are invalid",
    )
    suffixes = {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "xdg-cache",
        "ISAACLAB_USD_CACHE_DIR": "isaaclab-usd-cache",
        "CUDA_CACHE_PATH": "cuda-cache",
        "TORCH_HOME": "torch-home",
        "OMNI_USER_CACHE_DIR": "omni-user-cache",
    }
    if launch_cache_token is not None:
        _sha256(launch_cache_token, "launch_cache_token")
    resolved_paths = []
    for key in SCIENTIFIC_CACHE_ENVIRONMENT_KEYS:
        value = environment.get(key)
        _require(isinstance(value, str) and value, f"scientific cache environment {key} missing")
        path = Path(value)
        _require(
            path.is_absolute()
            and not path.is_symlink()
            and path.resolve() == path
            and path.is_dir(),
            f"scientific cache environment {key} is not a canonical directory",
        )
        _require(path.name == suffixes[key], f"scientific cache environment {key} suffix drifted")
        resolved_paths.append(path)
    _require(
        len(set(resolved_paths)) == len(resolved_paths),
        "scientific cache environment paths must be distinct",
    )
    launch_roots = {path.parent for path in resolved_paths}
    _require(len(launch_roots) == 1, "scientific cache paths do not share one launch-token root")
    launch_root = next(iter(launch_roots))
    if launch_cache_token is not None:
        _require(
            launch_root.name == launch_cache_token,
            "scientific cache launch-token directory drifted",
        )
    _require(
        not (
            ".cache" in launch_root.parts
            and ("home" in launch_root.parts or "root" in launch_root.parts)
        ),
        "scientific cache environment may not use a home-directory cache fallback",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdirs_durable(path: Path) -> None:
    """Create missing directories and durably publish every parent entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    _require(cursor.is_dir() and not cursor.is_symlink(), "artifact parent is not canonical")
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _require(directory.is_dir() and not directory.is_symlink(), "artifact parent drifted")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _publish_new_bytes(path: str | Path, payload: bytes, *, label: str) -> Path:
    """Publish bytes atomically without clobbering an existing artifact.

    The temporary file is fsynced in the destination directory, then linked to
    the final name with no-replace semantics.  A directory fsync makes the
    publication durable before a later receipt can act as the commit marker.
    """

    destination = Path(path).expanduser().resolve()
    _mkdirs_durable(destination.parent)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".staging",
        dir=destination.parent,
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite runtime artifact: {destination}"
            ) from error
        _fsync_directory(destination.parent)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    _require(destination.read_bytes() == payload, f"{label} bytes drifted after publication")
    return destination


def write_new_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write deterministic pretty JSON with exclusive-create semantics."""

    return _publish_new_bytes(path, _pretty_json_bytes(payload), label="runtime JSON artifact")


def _publish_or_recover_exact_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """Publish a runtime sidecar or recover only an exact prior publication."""

    destination = Path(path).expanduser().resolve()
    expected = _pretty_json_bytes(payload)
    if destination.exists():
        _require(
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == expected,
            f"existing {label} is not the deterministic expected artifact",
        )
        return destination
    return _publish_new_bytes(destination, expected, label=label)


def _strict_json_loads(raw: str, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {name}: {error}") from error

    def reject_nonfinite_tree(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_nonfinite_tree(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_nonfinite_tree(child, f"{location}[{index}]")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} contains non-finite number at {location}")

    reject_nonfinite_tree(value, "$")
    return value


def _load_json_object(path: str | Path, name: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read {name} {resolved}: {error}") from error
    _require(bool(raw.strip()), f"{name} must not be blank")
    payload = _strict_json_loads(raw, name)
    _require(isinstance(payload, dict), f"{name} must contain a JSON object")
    return payload


def _load_bound_analysis_protocol(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).expanduser()
    _require(candidate.is_absolute(), "analysis protocol path must be absolute")
    _require(not candidate.is_symlink(), "analysis protocol path may not be a symlink")
    resolved = candidate.resolve()
    _require(
        resolved == candidate and resolved.is_file(), "analysis protocol path is not canonical"
    )
    file_digest = file_sha256(resolved)
    if expected_file_sha256 is not None:
        _require(
            file_digest == _sha256(expected_file_sha256, "analysis protocol file SHA-256"),
            "analysis protocol file bytes drifted",
        )
    protocol = _load_json_object(resolved, "analysis protocol")
    validate_analysis_protocol(protocol)
    protocol_digest = protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
    if expected_protocol_sha256 is not None:
        _require(
            protocol_digest == _sha256(expected_protocol_sha256, "analysis protocol SHA-256"),
            "analysis protocol self digest drifted",
        )
    return resolved, protocol, file_digest


def _absolute_new_json_path(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{name} must be an absolute path")
    resolved = path.resolve()
    _require(resolved.suffix == ".json", f"{name} must use the .json suffix")
    _require(not resolved.exists(), f"{name} already exists: {resolved}")
    return resolved


def discover_scientific_source_paths(
    repo_root: str | Path,
    *,
    extra_paths: Sequence[str | Path] = (),
) -> tuple[str, ...]:
    """Return the deterministic SONIC source/asset closure used by atlas eval.

    Every regular file under ``gear_sonic`` is included except generated Python
    bytecode and known cache directories.  This is intentionally broader than
    the LACE package: dirty policy/environment code *and* local USD, URDF,
    collision-mesh, or learned preprocessing assets can change the measurement
    instrument even when the git commit remains fixed.

    Symlinks fail closed.  A link can make the live instrument depend on bytes
    outside the repository, so silently omitting it would create a false claim
    of closure.
    """

    root = Path(repo_root).expanduser().resolve()
    package_root = root / "gear_sonic"
    _require(package_root.is_dir(), f"missing gear_sonic package under {root}")
    paths: set[str] = set()
    for path in package_root.rglob("*"):
        _require(not path.is_symlink(), f"scientific source closure contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in _IGNORED_SOURCE_SUFFIXES:
            continue
        paths.add(relative.as_posix())
    for required in (
        "scripts/research/run_lace_atlas_probe_cell.py",
        "scripts/research/build_lace_instrument.py",
    ):
        _require((root / required).is_file(), f"missing required instrument source: {required}")
        paths.add(required)
    for raw in extra_paths:
        path = Path(raw)
        candidate = path if path.is_absolute() else root / path
        try:
            lexical_relative = candidate.relative_to(root)
            current = root
            for component in lexical_relative.parts:
                current = current / component
                _require(
                    not current.is_symlink(),
                    f"extra instrument source traverses a symlink: {raw}",
                )
            relative = candidate.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as error:
            raise ValueError(f"extra instrument source is outside/missing: {raw}") from error
        _require(candidate.is_file(), f"extra instrument source is not a file: {raw}")
        _require(relative not in paths, f"duplicate instrument source path: {relative}")
        paths.add(relative)
    return tuple(sorted(paths))


def build_runtime_handshake(
    *,
    schedule_sha256: str,
    checkpoint_sha256: str,
    probe_policy_id: str,
    rollout_ids: Sequence[str],
    rollout_output_path: str | Path,
    instrumented_rollout_output_path: str | Path,
    launch_plan_path: str | Path,
    instrument_output_path: str | Path,
    instrument_binding_output_path: str | Path,
    rollout_binding_output_path: str | Path,
    analysis_protocol_path: str | Path,
    git_commit: str,
    repo_root: str | Path,
    source_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Freeze every prelaunch input to runtime instrument materialization."""

    schedule_digest = _sha256(schedule_sha256, "schedule_sha256")
    checkpoint_digest = _sha256(checkpoint_sha256, "checkpoint_sha256")
    commit = _git_commit(git_commit)
    _require(isinstance(probe_policy_id, str) and probe_policy_id, "probe_policy_id is invalid")
    ids = tuple(rollout_ids)
    _require(
        bool(ids)
        and all(isinstance(item, str) and item for item in ids)
        and len(ids) == len(set(ids)),
        "rollout_ids must be non-empty and unique",
    )
    rollout_output = Path(rollout_output_path).expanduser().resolve()
    _require(rollout_output.suffix == ".jsonl", "rollout_output_path must use .jsonl")
    _require(not rollout_output.exists(), f"rollout output already exists: {rollout_output}")
    instrumented_rollout_output = Path(instrumented_rollout_output_path).expanduser().resolve()
    _require(
        instrumented_rollout_output.suffix == ".jsonl",
        "instrumented_rollout_output_path must use .jsonl",
    )
    _require(
        not instrumented_rollout_output.exists(),
        f"instrumented rollout output already exists: {instrumented_rollout_output}",
    )
    launch_plan = _absolute_new_json_path(launch_plan_path, "launch_plan_path")
    instrument_output = _absolute_new_json_path(
        instrument_output_path,
        "instrument_output_path",
    )
    instrument_binding = _absolute_new_json_path(
        instrument_binding_output_path,
        "instrument_binding_output_path",
    )
    rollout_binding = _absolute_new_json_path(
        rollout_binding_output_path,
        "rollout_binding_output_path",
    )
    output_paths = {
        launch_plan,
        instrument_output,
        instrument_binding,
        rollout_binding,
        rollout_output,
        instrumented_rollout_output,
    }
    _require(len(output_paths) == 6, "runtime output paths must be distinct")

    source_bundle = build_source_bundle(repo_root, source_paths)
    protocol_path, protocol, protocol_file_digest = _load_bound_analysis_protocol(
        analysis_protocol_path
    )
    _require(
        protocol["schedule_sha256"] == schedule_digest
        and protocol["git_commit"] == commit
        and protocol["source_bundle_sha256"] == source_bundle["source_bundle_sha256"],
        "analysis protocol does not bind this schedule/source freeze",
    )
    handshake: dict[str, Any] = {
        "kind": RUNTIME_HANDSHAKE_KIND,
        "schema_version": RUNTIME_HANDSHAKE_SCHEMA_VERSION,
        "scientific_use": True,
        "schedule_sha256": schedule_digest,
        "checkpoint_sha256": checkpoint_digest,
        "probe_policy_id": probe_policy_id,
        "rollout_ids": list(ids),
        "motion_count": len(ids),
        "rollout_output_path": str(rollout_output),
        "instrumented_rollout_output_path": str(instrumented_rollout_output),
        "launch_plan_path": str(launch_plan),
        "instrument_output_path": str(instrument_output),
        "instrument_binding_output_path": str(instrument_binding),
        "rollout_binding_output_path": str(rollout_binding),
        "git_commit": commit,
        "source_bundle": source_bundle,
        "source_bundle_sha256": source_bundle["source_bundle_sha256"],
        "analysis_protocol_path": str(protocol_path),
        "analysis_protocol_file_sha256": protocol_file_digest,
        "analysis_protocol_sha256": protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
    }
    handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD] = canonical_sha256(
        handshake,
        digest_field=RUNTIME_HANDSHAKE_DIGEST_FIELD,
    )
    validate_runtime_handshake(handshake, repo_root=repo_root, require_new_outputs=True)
    return handshake


def validate_runtime_handshake(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    require_new_outputs: bool = False,
) -> None:
    _require(isinstance(raw, Mapping), "runtime handshake must be a mapping")
    handshake = dict(raw)
    _require(set(handshake) == _HANDSHAKE_FIELDS, "runtime handshake fields are invalid")
    _require(handshake.get("kind") == RUNTIME_HANDSHAKE_KIND, "runtime handshake kind is invalid")
    _require(
        handshake.get("schema_version") == RUNTIME_HANDSHAKE_SCHEMA_VERSION,
        "runtime handshake schema_version is invalid",
    )
    _require(handshake.get("scientific_use") is True, "runtime handshake must be scientific")
    _sha256(handshake.get("schedule_sha256"), "runtime handshake schedule_sha256")
    _sha256(handshake.get("checkpoint_sha256"), "runtime handshake checkpoint_sha256")
    _git_commit(handshake.get("git_commit"))
    _require(
        isinstance(handshake.get("probe_policy_id"), str) and handshake["probe_policy_id"],
        "runtime handshake probe_policy_id is invalid",
    )
    ids = handshake.get("rollout_ids")
    _require(
        isinstance(ids, list)
        and bool(ids)
        and all(isinstance(item, str) and item for item in ids)
        and len(ids) == len(set(ids)),
        "runtime handshake rollout_ids are invalid",
    )
    _require(handshake.get("motion_count") == len(ids), "runtime handshake motion_count drifted")
    rollout_output = Path(str(handshake.get("rollout_output_path", "")))
    _require(
        rollout_output.is_absolute() and rollout_output.suffix == ".jsonl", "invalid rollout path"
    )
    instrumented_rollout_output = Path(str(handshake.get("instrumented_rollout_output_path", "")))
    _require(
        instrumented_rollout_output.is_absolute()
        and instrumented_rollout_output.suffix == ".jsonl",
        "invalid instrumented rollout path",
    )
    output_names = (
        "launch_plan_path",
        "instrument_output_path",
        "instrument_binding_output_path",
        "rollout_binding_output_path",
    )
    output_paths = []
    for name in output_names:
        path = Path(str(handshake.get(name, "")))
        _require(path.is_absolute() and path.suffix == ".json", f"runtime handshake {name} invalid")
        output_paths.append(path)
    _require(
        len(set(output_paths + [rollout_output, instrumented_rollout_output])) == 6,
        "runtime output paths are not distinct",
    )
    if require_new_outputs:
        _require(not rollout_output.exists(), f"rollout output already exists: {rollout_output}")
        _require(
            not instrumented_rollout_output.exists(),
            f"instrumented rollout output already exists: {instrumented_rollout_output}",
        )
        for path in output_paths:
            _require(not path.exists(), f"runtime output already exists: {path}")

    source_bundle = handshake.get("source_bundle")
    _require(isinstance(source_bundle, Mapping), "runtime handshake source_bundle is missing")
    validate_source_bundle(source_bundle, repo_root=repo_root)
    _require(
        handshake.get("source_bundle_sha256") == source_bundle["source_bundle_sha256"],
        "runtime handshake source bundle binding drifted",
    )
    _, protocol, _ = _load_bound_analysis_protocol(
        handshake.get("analysis_protocol_path"),
        expected_file_sha256=handshake.get("analysis_protocol_file_sha256"),
        expected_protocol_sha256=handshake.get("analysis_protocol_sha256"),
    )
    _require(
        protocol["schedule_sha256"] == handshake["schedule_sha256"]
        and protocol["git_commit"] == handshake["git_commit"]
        and protocol["source_bundle_sha256"] == handshake["source_bundle_sha256"],
        "runtime handshake analysis protocol provenance drifted",
    )
    digest = _sha256(
        handshake.get(RUNTIME_HANDSHAKE_DIGEST_FIELD),
        RUNTIME_HANDSHAKE_DIGEST_FIELD,
    )
    _require(
        digest == canonical_sha256(handshake, digest_field=RUNTIME_HANDSHAKE_DIGEST_FIELD),
        "runtime_handshake_sha256 mismatch",
    )


def build_plan_binding(handshake_path: str | Path, handshake: Mapping[str, Any]) -> dict[str, Any]:
    """Bind exact handshake bytes and all promised outputs into a launch plan."""

    path = Path(handshake_path).expanduser().resolve()
    _require(path.is_file(), f"runtime handshake file is missing: {path}")
    validate_runtime_handshake(handshake)
    binding = {
        "kind": PLAN_BINDING_KIND,
        "schema_version": PLAN_BINDING_SCHEMA_VERSION,
        "required": True,
        "state": "runtime_materialization_required_before_policy_rollout",
        "handshake_path": str(path),
        "handshake_file_sha256": file_sha256(path),
        "runtime_handshake_sha256": handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD],
        "instrument_output_path": handshake["instrument_output_path"],
        "instrument_binding_output_path": handshake["instrument_binding_output_path"],
        "instrumented_rollout_output_path": handshake["instrumented_rollout_output_path"],
        "rollout_binding_output_path": handshake["rollout_binding_output_path"],
        "analysis_protocol_path": handshake["analysis_protocol_path"],
        "analysis_protocol_file_sha256": handshake["analysis_protocol_file_sha256"],
        "analysis_protocol_sha256": handshake["analysis_protocol_sha256"],
    }
    validate_plan_binding(binding)
    return binding


def validate_plan_binding(raw: Mapping[str, Any]) -> None:
    _require(isinstance(raw, Mapping), "instrument plan binding must be a mapping")
    binding = dict(raw)
    _require(set(binding) == _PLAN_BINDING_FIELDS, "instrument plan binding fields are invalid")
    _require(binding.get("kind") == PLAN_BINDING_KIND, "instrument plan binding kind is invalid")
    _require(
        binding.get("schema_version") == PLAN_BINDING_SCHEMA_VERSION,
        "instrument plan binding schema_version is invalid",
    )
    _require(binding.get("required") is True, "scientific instrument plan binding is not required")
    _require(
        binding.get("state") == "runtime_materialization_required_before_policy_rollout",
        "scientific instrument plan binding state is invalid",
    )
    _sha256(binding.get("handshake_file_sha256"), "handshake_file_sha256")
    _sha256(binding.get("runtime_handshake_sha256"), "runtime_handshake_sha256")
    _sha256(binding.get("analysis_protocol_file_sha256"), "analysis_protocol_file_sha256")
    _sha256(binding.get("analysis_protocol_sha256"), "analysis_protocol_sha256")
    for name in (
        "handshake_path",
        "instrument_output_path",
        "instrument_binding_output_path",
        "instrumented_rollout_output_path",
        "rollout_binding_output_path",
        "analysis_protocol_path",
    ):
        _require(Path(str(binding.get(name, ""))).is_absolute(), f"{name} must be absolute")


def validate_instrument_binding(raw: Mapping[str, Any]) -> None:
    """Validate the post-load runtime binding written beside an instrument."""

    _require(isinstance(raw, Mapping), "instrument binding must be a mapping")
    binding = dict(raw)
    _require(
        set(binding) == _INSTRUMENT_BINDING_FIELDS,
        "instrument binding fields are invalid",
    )
    _require(binding.get("kind") == INSTRUMENT_BINDING_KIND, "instrument binding kind is invalid")
    _require(
        binding.get("schema_version") == INSTRUMENT_BINDING_SCHEMA_VERSION,
        "instrument binding schema_version is invalid",
    )
    _require(binding.get("scientific_use") is True, "instrument binding must be scientific")
    for field in (
        "launch_plan_sha256",
        "launch_plan_file_sha256",
        "runtime_handshake_sha256",
        "schedule_sha256",
        "checkpoint_sha256",
        "dataset_binding_sha256",
        "instrument_file_sha256",
        "instrument_manifest_sha256",
        "episode_instrument_sha256",
        "measurement_family_manifest_sha256",
        "episode_measurement_family_sha256",
        "source_bundle_sha256",
        "analysis_protocol_file_sha256",
        "analysis_protocol_sha256",
        INSTRUMENT_BINDING_DIGEST_FIELD,
    ):
        _sha256(binding.get(field), f"instrument_binding.{field}")
    loaded = binding.get("loaded_checkpoint_bundle")
    _require(
        isinstance(loaded, Mapping) and set(loaded) == _CHECKPOINT_BUNDLE_FIELDS,
        "instrument binding loaded_checkpoint_bundle is invalid",
    )
    _sha256(loaded.get("checkpoint_sha256"), "loaded checkpoint sha256")
    _sha256(loaded.get("config_sha256"), "loaded checkpoint config sha256")
    _require(
        loaded.get("checkpoint_sha256") == binding["checkpoint_sha256"],
        "instrument binding checkpoint digest drifted",
    )
    for field in ("checkpoint_path", "config_path"):
        _require(Path(str(loaded.get(field, ""))).is_absolute(), f"loaded {field} is invalid")
    for field in (
        "rollout_output_path",
        "instrumented_rollout_output_path",
        "instrument_output_path",
        "analysis_protocol_path",
    ):
        _require(Path(str(binding.get(field, ""))).is_absolute(), f"{field} must be absolute")
    _load_bound_analysis_protocol(
        binding["analysis_protocol_path"],
        expected_file_sha256=binding["analysis_protocol_file_sha256"],
        expected_protocol_sha256=binding["analysis_protocol_sha256"],
    )
    family = binding.get("measurement_family")
    _require(isinstance(family, Mapping), "instrument binding measurement_family is missing")
    from gear_sonic.research.lace.instrument import validate_measurement_family

    validate_measurement_family(family)
    _require(
        family[MEASUREMENT_FAMILY_DIGEST_FIELD] == binding["measurement_family_manifest_sha256"],
        "instrument binding family manifest digest drifted",
    )
    _require(
        episode_measurement_family_sha256(family) == binding["episode_measurement_family_sha256"],
        "instrument binding episode family digest drifted",
    )
    _require(
        binding["source_bundle_sha256"] == family["source_bundle_sha256"],
        "instrument binding source bundle drifted from its measurement family",
    )
    _require(
        binding[INSTRUMENT_BINDING_DIGEST_FIELD]
        == canonical_sha256(binding, digest_field=INSTRUMENT_BINDING_DIGEST_FIELD),
        "instrument_binding_sha256 mismatch",
    )


def _current_git_commit(repo_root: Path) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve runtime git commit: {completed.stderr.strip()}")
    return _git_commit(completed.stdout.strip())


def _validate_bound_regular_file(path_value: Any, digest_value: Any, name: str) -> Path:
    _sha256(digest_value, f"{name}.sha256")
    _require(isinstance(path_value, str) and path_value, f"{name}.path is invalid")
    path = Path(path_value)
    _require(path.is_absolute(), f"{name}.path must be absolute")
    _require(not path.is_symlink(), f"{name}.path may not be a symlink")
    _require(path.is_file(), f"{name}.path is not a file: {path}")
    _require(path.resolve() == path, f"{name}.path must be canonical")
    _require(file_sha256(path) == digest_value, f"{name} file digest drifted")
    return path


def _validate_checkpoint_bundle(
    plan: Mapping[str, Any], handshake: Mapping[str, Any]
) -> dict[str, str]:
    bundle = plan.get("checkpoint_bundle")
    _require(isinstance(bundle, Mapping), "launch plan checkpoint_bundle is missing")
    _require(
        set(bundle) == _CHECKPOINT_BUNDLE_FIELDS,
        "launch plan checkpoint_bundle fields are invalid",
    )
    checkpoint = _validate_bound_regular_file(
        bundle.get("checkpoint_path"),
        bundle.get("checkpoint_sha256"),
        "checkpoint_bundle.checkpoint",
    )
    _require(
        str(checkpoint) == plan.get("checkpoint_path"),
        "checkpoint bundle path drifted from launch plan",
    )
    _require(
        bundle.get("checkpoint_sha256") == handshake["checkpoint_sha256"],
        "checkpoint bundle digest drifted from runtime handshake",
    )
    config_path = _validate_bound_regular_file(
        bundle.get("config_path"),
        bundle.get("config_sha256"),
        "checkpoint_bundle.config",
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": str(bundle["checkpoint_sha256"]),
        "config_path": str(config_path),
        "config_sha256": str(bundle["config_sha256"]),
    }


def _validate_loaded_checkpoint_bundle(
    raw: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> dict[str, str]:
    _require(isinstance(raw, Mapping), "loaded_checkpoint_bundle must be a mapping")
    bundle = dict(raw)
    _require(
        set(bundle) == _CHECKPOINT_BUNDLE_FIELDS,
        "loaded_checkpoint_bundle fields are invalid",
    )
    checkpoint = _validate_bound_regular_file(
        bundle.get("checkpoint_path"),
        bundle.get("checkpoint_sha256"),
        "loaded_checkpoint_bundle.checkpoint",
    )
    config = _validate_bound_regular_file(
        bundle.get("config_path"),
        bundle.get("config_sha256"),
        "loaded_checkpoint_bundle.config",
    )
    normalized = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": str(bundle["checkpoint_sha256"]),
        "config_path": str(config),
        "config_sha256": str(bundle["config_sha256"]),
    }
    _require(normalized == dict(expected), "loaded checkpoint bundle drifted from launch plan")
    return normalized


def _validate_dataset_binding(plan: Mapping[str, Any]) -> None:
    binding = plan.get("dataset_binding")
    _require(isinstance(binding, Mapping), "launch plan dataset_binding is missing")
    _require(set(binding) == {"robot", "smpl"}, "launch plan dataset binding fields are invalid")
    expected_motion_keys = plan.get("motion_keys")
    _require(
        isinstance(expected_motion_keys, list)
        and expected_motion_keys
        and len(expected_motion_keys) == len(set(expected_motion_keys)),
        "launch plan motion_keys are invalid",
    )
    for group_name in ("robot", "smpl"):
        group = binding.get(group_name)
        _require(isinstance(group, Mapping), f"dataset_binding.{group_name} is invalid")
        if group_name == "smpl":
            _require(group.get("mode") == "real", "scientific dataset binding requires real SMPL")
        root = Path(str(group.get("motion_root", "")))
        _require(
            root.is_absolute()
            and not root.is_symlink()
            and root.resolve() == root
            and root.is_dir(),
            f"dataset_binding.{group_name} root invalid",
        )
        files = group.get("files")
        _require(
            isinstance(files, list) and len(files) == len(expected_motion_keys),
            f"dataset_binding.{group_name} files do not cover the cell",
        )
        _require(
            [record.get("motion_key") for record in files] == expected_motion_keys,
            f"dataset_binding.{group_name} file order drifted",
        )
        for index, record in enumerate(files):
            _require(
                isinstance(record, Mapping) and set(record) == {"motion_key", "path", "sha256"},
                f"dataset_binding.{group_name}.files[{index}] is invalid",
            )
            path = _validate_bound_regular_file(
                record.get("path"),
                record.get("sha256"),
                f"dataset_binding.{group_name}.files[{index}]",
            )
            _require(
                path.parent == root
                and path.name == f"{Path(expected_motion_keys[index]).name}.pkl",
                f"dataset_binding.{group_name}.files[{index}] path drifted",
            )
        _require(
            group.get("selected_file_set_sha256") == canonical_sha256({"files": files}),
            f"dataset_binding.{group_name} aggregate digest drifted",
        )


def _validate_plan_and_handshake(
    *,
    handshake_path: str | Path,
    expected_handshake_file_sha256: str,
    launch_plan_path: str | Path,
    repo_root: str | Path,
    verify_git_head: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = Path(repo_root).expanduser().resolve()
    handshake_file = Path(handshake_path).expanduser().resolve()
    expected_file_digest = _sha256(
        expected_handshake_file_sha256,
        "expected_handshake_file_sha256",
    )
    _require(
        file_sha256(handshake_file) == expected_file_digest,
        "runtime handshake file SHA-256 drifted from the Hydra/plan binding",
    )
    handshake = _load_json_object(handshake_file, "runtime handshake")
    validate_runtime_handshake(handshake, repo_root=root)
    plan_file = Path(launch_plan_path).expanduser().resolve()
    _require(
        str(plan_file) == handshake["launch_plan_path"],
        "launch plan path drifted from runtime handshake",
    )
    plan = _load_json_object(plan_file, "launch plan")
    _require(
        set(plan) == _SCIENTIFIC_LAUNCH_PLAN_FIELDS,
        "scientific launch plan fields are invalid",
    )
    from gear_sonic.research.lace.schedule_batch_loader import (
        LAUNCH_PLAN_KIND,
        LAUNCH_PLAN_SCHEMA_VERSION,
        load_locked_rollout_schedule,
        rebuild_eval_hydra_overrides_from_launch_plan,
    )

    _require(plan.get("kind") == LAUNCH_PLAN_KIND, "scientific launch plan kind is invalid")
    _require(
        plan.get("schema_version") == LAUNCH_PLAN_SCHEMA_VERSION,
        "scientific launch plan schema_version is invalid",
    )
    plan_digest = _sha256(plan.get("launch_plan_sha256"), "launch_plan_sha256")
    _require(
        plan_digest == canonical_sha256(plan, digest_field="launch_plan_sha256"),
        "launch_plan_sha256 mismatch",
    )
    _require(plan.get("scientific_use") is True, "runtime instrument requires a scientific plan")
    _require(plan.get("schedule_sha256") == handshake["schedule_sha256"], "plan schedule drifted")
    _require(
        plan.get("rollout_output_path") == handshake["rollout_output_path"],
        "plan rollout output drifted",
    )
    cell = plan.get("cell")
    _require(isinstance(cell, Mapping), "launch plan cell is missing")
    _require(
        cell.get("checkpoint_sha256") == handshake["checkpoint_sha256"],
        "plan checkpoint drifted",
    )
    _require(cell.get("probe_policy_id") == handshake["probe_policy_id"], "plan policy drifted")
    _require(plan.get("rollout_ids") == handshake["rollout_ids"], "plan rollout ids drifted")
    _require(plan.get("num_envs") == handshake["motion_count"], "plan num_envs drifted")
    raw_assignments = plan.get("atlas_probe_assignments")
    _require(
        isinstance(raw_assignments, list)
        and all(isinstance(row, Mapping) for row in raw_assignments)
        and [row.get("rollout_id") for row in raw_assignments] == handshake["rollout_ids"],
        "plan atlas assignments drifted",
    )
    locked = load_locked_rollout_schedule(plan.get("schedule_lock_path"), repo_root=root)
    _require(
        plan.get("schedule_path") == str(locked.schedule_path)
        and plan.get("schedule_file_sha256") == locked.schedule_file_sha256
        and plan.get("schedule_spec_path") == str(locked.schedule_spec_path)
        and plan.get("schedule_sha256") == locked.manifest["schedule_sha256"],
        "launch plan schedule lock provenance drifted",
    )
    protocol_path, protocol, protocol_file_digest = _load_bound_analysis_protocol(
        plan.get("analysis_protocol_path"),
        expected_file_sha256=plan.get("analysis_protocol_file_sha256"),
        expected_protocol_sha256=plan.get("analysis_protocol_sha256"),
    )
    validate_analysis_protocol(
        protocol,
        schedule_manifest=locked.manifest,
        split_manifest=locked.split_manifest,
        reference_length_inventory=locked.reference_length_inventory,
    )
    from gear_sonic.research.lace.analysis_protocol_lock import (
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
        checkpoint_config_binding,
        execution_cell_binding,
        load_analysis_protocol_lock,
    )

    protocol_lock_path, protocol_lock, locked_protocol_path, locked_protocol = (
        load_analysis_protocol_lock(
            plan.get("analysis_protocol_lock_path"),
            repo_root=root,
        )
    )
    _require(
        file_sha256(protocol_lock_path) == plan.get("analysis_protocol_lock_file_sha256")
        and protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
        == plan.get("analysis_protocol_lock_sha256")
        and locked_protocol_path == protocol_path
        and locked_protocol == protocol
        and protocol_lock["schedule_lock"]["path"] == str(locked.lock_path)
        and protocol_lock["schedule_lock"]["schedule_sha256"] == locked.manifest["schedule_sha256"],
        "launch plan analysis protocol preregistration lock drifted",
    )
    preregistered_config = checkpoint_config_binding(
        protocol_lock,
        probe_policy_id=str(cell["probe_policy_id"]),
        checkpoint_sha256=str(cell["checkpoint_sha256"]),
    )
    plan_checkpoint_bundle = plan.get("checkpoint_bundle")
    _require(
        isinstance(plan_checkpoint_bundle, Mapping)
        and plan_checkpoint_bundle.get("config_path") == preregistered_config["config_path"]
        and plan_checkpoint_bundle.get("config_sha256") == preregistered_config["config_sha256"],
        "launch plan checkpoint companion config drifted from preregistration",
    )
    registered_execution = execution_cell_binding(protocol_lock, cell=cell)
    _require(
        registered_execution["plan_output_path"] == str(plan_file)
        and registered_execution["rollout_output_path"]
        == handshake["rollout_output_path"]
        == plan["rollout_output_path"]
        and registered_execution["instrumented_rollout_output_path"]
        == handshake["instrumented_rollout_output_path"]
        and registered_execution["instrument_output_path"] == handshake["instrument_output_path"]
        and registered_execution["runtime_handshake_output_path"] == str(handshake_file)
        and registered_execution["instrument_binding_output_path"]
        == handshake["instrument_binding_output_path"]
        and registered_execution["rollout_binding_output_path"]
        == handshake["rollout_binding_output_path"],
        "launch plan/runtime outputs drifted from the preregistered execution cell",
    )
    _require(
        str(protocol_path) == handshake["analysis_protocol_path"]
        and protocol_file_digest == handshake["analysis_protocol_file_sha256"]
        and protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD] == handshake["analysis_protocol_sha256"]
        and protocol["source_bundle_sha256"] == handshake["source_bundle_sha256"],
        "launch plan analysis protocol binding drifted",
    )
    expected_rows = [
        dict(row)
        for row in locked.manifest["rollouts"]
        if all(row[field] == cell[field] for field in _CELL_FIELDS)
    ]
    _require(expected_rows == raw_assignments, "launch plan rows drifted from its schedule lock")
    _require(
        plan.get("motion_keys") == [row["motion_key"] for row in expected_rows]
        and plan.get("reference_num_steps") == [row["reference_num_steps"] for row in expected_rows]
        and plan.get("num_envs") == len(expected_rows),
        "launch plan motion/reference batch drifted from its locked rows",
    )
    expected_max_render_steps = (
        max(int(row["reference_num_steps"]) - int(row["start_step"]) for row in expected_rows) + 2
    )
    _require(
        plan.get("instrumentation_invariants")
        == {
            "headless": True,
            "terrain_type": "plane",
            "render_results": False,
            "policy_enable_corruption": False,
            "tokenizer_enable_corruption": False,
            "use_encoder": "g1",
            "eval_callbacks": [],
            "run_once": True,
            "max_render_steps": expected_max_render_steps,
            "native_adaptive_sampling": False,
        },
        "launch plan instrumentation invariants drifted",
    )
    canonical_entrypoint = (root / "gear_sonic/eval_agent_trl.py").resolve()
    _require(
        canonical_entrypoint.is_file()
        and plan.get("eval_entrypoint") == str(canonical_entrypoint)
        and "gear_sonic/eval_agent_trl.py"
        in {record["path"] for record in handshake["source_bundle"]["files"]},
        "launch plan eval entrypoint is not the canonical source-bound file",
    )
    runtime_overrides = (
        "++lace_scientific_instrument_required=true",
        "++lace_instrument_handshake_path=" + json.dumps(str(handshake_file)),
        "++lace_instrument_handshake_file_sha256=" + file_sha256(handshake_file),
        "++lace_launch_plan_path=" + json.dumps(str(plan_file)),
        "++lace_analysis_protocol_path=" + json.dumps(str(protocol_path)),
        "++lace_analysis_protocol_file_sha256=" + protocol_file_digest,
        "++lace_analysis_protocol_sha256=" + protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "++lace_analysis_protocol_lock_path=" + json.dumps(plan["analysis_protocol_lock_path"]),
        "++lace_analysis_protocol_lock_file_sha256=" + plan["analysis_protocol_lock_file_sha256"],
        "++lace_analysis_protocol_lock_sha256=" + plan["analysis_protocol_lock_sha256"],
    )
    expected_hydra_overrides = [
        *rebuild_eval_hydra_overrides_from_launch_plan(plan),
        *runtime_overrides,
    ]
    _require(
        plan.get("hydra_overrides") == expected_hydra_overrides,
        "launch plan Hydra overrides are not the deterministic frozen set",
    )
    passthrough = plan.get("passthrough_hydra_args")
    _require(
        passthrough == [],
        "scientific launch plan must not contain passthrough Hydra arguments",
    )
    expected_command = [
        str(Path(sys.executable).resolve()),
        str(canonical_entrypoint),
        *passthrough,
        *expected_hydra_overrides,
    ]
    _require(plan.get("command") == expected_command, "launch plan command drifted")
    launch_environment = plan.get("launch_environment")
    _require(
        isinstance(launch_environment, Mapping)
        and set(launch_environment)
        == {
            "PYTHONPATH",
            "semantics",
            "scientific_cache_environment",
            "scientific_cache_environment_semantics",
            "launch_cache_token",
        },
        "launch plan environment fields are invalid",
    )
    _require(
        launch_environment.get("semantics")
        == "repository_root_first_preserve_inherited_unique_entries_v1",
        "launch plan PYTHONPATH semantics drifted",
    )
    pythonpath = launch_environment.get("PYTHONPATH")
    _require(isinstance(pythonpath, str) and pythonpath, "launch plan PYTHONPATH is missing")
    pythonpath_entries = pythonpath.split(os.pathsep)
    _require(
        pythonpath_entries[0] == str(root)
        and len(pythonpath_entries) == len(set(pythonpath_entries))
        and all(pythonpath_entries),
        "launch plan PYTHONPATH is not repository-first and unique",
    )
    cache_environment = launch_environment.get("scientific_cache_environment")
    _require(isinstance(cache_environment, Mapping), "launch plan cache environment is missing")
    validate_scientific_cache_environment(
        cache_environment,
        launch_cache_token=launch_environment.get("launch_cache_token"),
    )
    registered_runtime_root = Path(str(protocol_lock["execution_registry"]["runtime_storage_root"]))
    _require(
        all(
            Path(str(path)).parents[1] == registered_runtime_root
            for path in cache_environment.values()
        ),
        "launch cache paths drifted from the preregistered runtime storage root",
    )
    _validate_checkpoint_bundle(plan, handshake)
    _validate_dataset_binding(plan)
    plan_binding = plan.get("instrument_runtime")
    _require(isinstance(plan_binding, Mapping), "launch plan lacks instrument runtime binding")
    validate_plan_binding(plan_binding)
    expected_binding = build_plan_binding(handshake_file, handshake)
    _require(dict(plan_binding) == expected_binding, "launch plan instrument binding drifted")
    if verify_git_head:
        _require(
            _current_git_commit(root) == handshake["git_commit"],
            "runtime git HEAD drifted from instrument handshake",
        )
    return handshake, plan, plan_digest


@dataclass(frozen=True)
class MaterializedInstrument:
    repo_root: Path
    handshake: Mapping[str, Any]
    handshake_path: Path
    handshake_file_sha256: str
    launch_plan: Mapping[str, Any]
    launch_plan_path: Path
    launch_plan_sha256: str
    launch_plan_file_sha256: str
    loaded_checkpoint_bundle: Mapping[str, str]
    instrument: Mapping[str, Any]
    instrument_file_sha256: str
    episode_instrument_sha256: str
    measurement_family: Mapping[str, Any]
    episode_measurement_family_sha256: str
    binding: Mapping[str, Any]
    binding_file_sha256: str


def materialize_instrument_from_payloads(
    *,
    handshake_path: str | Path,
    expected_handshake_file_sha256: str,
    launch_plan_path: str | Path,
    repo_root: str | Path,
    resolved_hydra_config: Mapping[str, Any],
    environment_fingerprint: Mapping[str, Any],
    probe_thresholds: Mapping[str, Any],
    recorder_config: Mapping[str, Any],
    sensor_semantics: Mapping[str, Any],
    termination_predicates: Mapping[str, Any],
    score_window_config: Mapping[str, Any],
    domain_randomization_config: Mapping[str, Any],
    loaded_checkpoint_bundle: Mapping[str, Any],
    verify_git_head: bool = True,
) -> MaterializedInstrument:
    """Materialize the real instrument from explicit post-startup payloads."""

    root = Path(repo_root).expanduser().resolve()
    handshake, plan, plan_digest = _validate_plan_and_handshake(
        handshake_path=handshake_path,
        expected_handshake_file_sha256=expected_handshake_file_sha256,
        launch_plan_path=launch_plan_path,
        repo_root=root,
        verify_git_head=verify_git_head,
    )
    expected_checkpoint_bundle = _validate_checkpoint_bundle(plan, handshake)
    loaded_bundle = _validate_loaded_checkpoint_bundle(
        loaded_checkpoint_bundle,
        expected=expected_checkpoint_bundle,
    )
    resolved_checkpoint = resolved_hydra_config.get("checkpoint")
    _require(
        isinstance(resolved_checkpoint, str)
        and str(Path(resolved_checkpoint).expanduser().resolve())
        == loaded_bundle["checkpoint_path"],
        "resolved Hydra checkpoint does not match the loaded checkpoint",
    )
    _require(
        resolved_hydra_config.get("lace_instrument_handshake_path")
        == str(Path(handshake_path).expanduser().resolve()),
        "resolved Hydra handshake path drifted",
    )
    _require(
        resolved_hydra_config.get("lace_instrument_handshake_file_sha256")
        == expected_handshake_file_sha256,
        "resolved Hydra handshake file digest drifted",
    )
    _require(
        resolved_hydra_config.get("lace_launch_plan_path")
        == str(Path(launch_plan_path).expanduser().resolve()),
        "resolved Hydra launch plan path drifted",
    )
    protocol_path, analysis_protocol, protocol_file_digest = _load_bound_analysis_protocol(
        handshake["analysis_protocol_path"],
        expected_file_sha256=handshake["analysis_protocol_file_sha256"],
        expected_protocol_sha256=handshake["analysis_protocol_sha256"],
    )
    _require(
        resolved_hydra_config.get("lace_analysis_protocol_path") == str(protocol_path)
        and resolved_hydra_config.get("lace_analysis_protocol_file_sha256") == protocol_file_digest
        and resolved_hydra_config.get("lace_analysis_protocol_sha256")
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "resolved Hydra analysis protocol binding drifted",
    )
    _require(
        resolved_hydra_config.get("lace_analysis_protocol_lock_path")
        == plan["analysis_protocol_lock_path"]
        and resolved_hydra_config.get("lace_analysis_protocol_lock_file_sha256")
        == plan["analysis_protocol_lock_file_sha256"]
        and resolved_hydra_config.get("lace_analysis_protocol_lock_sha256")
        == plan["analysis_protocol_lock_sha256"],
        "resolved Hydra analysis protocol preregistration lock drifted",
    )
    launch_environment = plan.get("launch_environment")
    _require(isinstance(launch_environment, Mapping), "launch plan environment is missing")
    cache_environment = launch_environment.get("scientific_cache_environment")
    _require(isinstance(cache_environment, Mapping), "launch plan cache environment is missing")
    expected_cache_token = canonical_sha256(
        {
            "schedule_sha256": plan["schedule_sha256"],
            "cell": dict(plan["cell"]),
            "plan_output": str(Path(launch_plan_path).expanduser().resolve()),
            "rollout_output": plan["rollout_output_path"],
        }
    )
    _require(
        launch_environment.get("launch_cache_token") == expected_cache_token,
        "launch plan cache token drifted from its exact execution cell",
    )
    validate_scientific_cache_environment(
        cache_environment,
        launch_cache_token=expected_cache_token,
    )
    _require(
        launch_environment.get("scientific_cache_environment_semantics")
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "launch plan cache environment semantics drifted",
    )
    _require(
        environment_fingerprint.get("scientific_cache_environment") == cache_environment
        and environment_fingerprint.get("scientific_cache_environment_semantics")
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "live cache environment drifted from the launch plan",
    )
    _require(
        environment_fingerprint.get("process_executable") == plan["command"][0]
        and environment_fingerprint.get("process_argv") == plan["command"][1:]
        and environment_fingerprint.get("pythonpath") == launch_environment["PYTHONPATH"],
        "live process invocation drifted from the launch plan",
    )
    source_paths = [record["path"] for record in handshake["source_bundle"]["files"]]
    instrument = build_scientific_instrument(
        schedule_sha256=handshake["schedule_sha256"],
        probe_thresholds=probe_thresholds,
        recorder_config=recorder_config,
        resolved_hydra_config=resolved_hydra_config,
        environment_fingerprint=environment_fingerprint,
        sensor_semantics=sensor_semantics,
        termination_predicates=termination_predicates,
        score_window_config=score_window_config,
        domain_randomization_config=domain_randomization_config,
        git_commit=handshake["git_commit"],
        repo_root=root,
        source_paths=source_paths,
    )
    validate_instrument_against_analysis_protocol(instrument, analysis_protocol)
    _require(
        instrument["source_bundle"] == handshake["source_bundle"],
        "runtime instrument source bytes drifted from prelaunch handshake",
    )
    from gear_sonic.research.lace.atlas import _validate_cell_instrument_binding

    assignments = plan["atlas_probe_assignments"]
    _validate_cell_instrument_binding(
        instrument,
        scheduled_rows=assignments,
        schedule_sha256=handshake["schedule_sha256"],
    )
    failure_term = resolved_hydra_config["manager_env"]["recorders"]["failure_atlas"]
    _require(
        failure_term.get("output_path") == handshake["rollout_output_path"],
        "resolved recorder output path drifted from runtime handshake",
    )

    episode_digest = episode_instrument_sha256(instrument)
    measurement_family = build_measurement_family(instrument)
    family_digest = episode_measurement_family_sha256(measurement_family)
    instrument_path = _publish_or_recover_exact_json(
        handshake["instrument_output_path"],
        instrument,
        label="scientific instrument",
    )
    instrument_file_digest = file_sha256(instrument_path)
    plan_path = Path(launch_plan_path).expanduser().resolve()
    plan_file_digest = file_sha256(plan_path)
    binding: dict[str, Any] = {
        "kind": INSTRUMENT_BINDING_KIND,
        "schema_version": INSTRUMENT_BINDING_SCHEMA_VERSION,
        "scientific_use": True,
        "launch_plan_sha256": plan_digest,
        "launch_plan_file_sha256": plan_file_digest,
        "runtime_handshake_sha256": handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD],
        "schedule_sha256": handshake["schedule_sha256"],
        "checkpoint_sha256": handshake["checkpoint_sha256"],
        "loaded_checkpoint_bundle": loaded_bundle,
        "dataset_binding_sha256": canonical_sha256(plan["dataset_binding"]),
        "rollout_output_path": handshake["rollout_output_path"],
        "instrumented_rollout_output_path": handshake["instrumented_rollout_output_path"],
        "instrument_output_path": str(instrument_path),
        "instrument_file_sha256": instrument_file_digest,
        "instrument_manifest_sha256": instrument[INSTRUMENT_DIGEST_FIELD],
        "episode_instrument_sha256": episode_digest,
        "measurement_family": measurement_family,
        "measurement_family_manifest_sha256": measurement_family[MEASUREMENT_FAMILY_DIGEST_FIELD],
        "episode_measurement_family_sha256": family_digest,
        "source_bundle_sha256": instrument["source_bundle_sha256"],
        "analysis_protocol_path": str(protocol_path),
        "analysis_protocol_file_sha256": protocol_file_digest,
        "analysis_protocol_sha256": analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
    }
    binding[INSTRUMENT_BINDING_DIGEST_FIELD] = canonical_sha256(
        binding,
        digest_field=INSTRUMENT_BINDING_DIGEST_FIELD,
    )
    validate_instrument_binding(binding)
    binding_path = _publish_or_recover_exact_json(
        handshake["instrument_binding_output_path"],
        binding,
        label="scientific instrument binding",
    )
    return MaterializedInstrument(
        repo_root=root,
        handshake=handshake,
        handshake_path=Path(handshake_path).expanduser().resolve(),
        handshake_file_sha256=file_sha256(handshake_path),
        launch_plan=plan,
        launch_plan_path=plan_path,
        launch_plan_sha256=plan_digest,
        launch_plan_file_sha256=plan_file_digest,
        loaded_checkpoint_bundle=loaded_bundle,
        instrument=instrument,
        instrument_file_sha256=instrument_file_digest,
        episode_instrument_sha256=episode_digest,
        measurement_family=measurement_family,
        episode_measurement_family_sha256=family_digest,
        binding=binding,
        binding_file_sha256=file_sha256(binding_path),
    )


def capture_environment_fingerprint(raw_env: Any) -> dict[str, Any]:
    """Capture explicit runtime versions/files without inventing missing values."""

    module_records = []
    distributions = importlib.metadata.packages_distributions()
    for name in ("hydra", "isaaclab", "numpy", "omegaconf", "torch"):
        module = importlib.import_module(name)
        module_path_value = getattr(module, "__file__", None)
        module_path = (
            Path(module_path_value).resolve()
            if isinstance(module_path_value, str) and Path(module_path_value).is_file()
            else None
        )
        distribution_versions = {}
        for distribution in sorted(distributions.get(name, [])):
            try:
                distribution_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        module_records.append(
            {
                "module": name,
                "module_version": getattr(module, "__version__", None),
                "module_file": str(module_path) if module_path is not None else None,
                "module_file_sha256": file_sha256(module_path) if module_path is not None else None,
                "distribution_versions": distribution_versions,
            }
        )
    torch_module = importlib.import_module("torch")
    cuda_available = bool(torch_module.cuda.is_available())
    cuda_record: dict[str, Any] = {
        "available": cuda_available,
        "compiled_version": getattr(torch_module.version, "cuda", None),
    }
    if cuda_available:
        device_index = int(torch_module.cuda.current_device())
        properties = torch_module.cuda.get_device_properties(device_index)
        cuda_record.update(
            {
                "device_index": device_index,
                "device_name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "capability": list(torch_module.cuda.get_device_capability(device_index)),
            }
        )
    cache_environment = {key: os.environ.get(key) for key in SCIENTIFIC_CACHE_ENVIRONMENT_KEYS}
    validate_scientific_cache_environment(cache_environment)
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "modules": module_records,
        "cuda": cuda_record,
        "scientific_cache_environment": cache_environment,
        "scientific_cache_environment_semantics": SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "process_executable": str(Path(sys.executable).resolve()),
        "process_argv": list(sys.argv),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "runtime": {
            "environment_type": f"{type(raw_env).__module__}:{type(raw_env).__qualname__}",
            "termination_manager_type": (
                f"{type(raw_env.termination_manager).__module__}:"
                f"{type(raw_env.termination_manager).__qualname__}"
            ),
            "event_manager_type": (
                f"{type(raw_env.event_manager).__module__}:"
                f"{type(raw_env.event_manager).__qualname__}"
            ),
            "device": str(raw_env.device),
            "num_envs": int(raw_env.num_envs),
            "step_dt_seconds": float(raw_env.step_dt),
        },
    }


def capture_live_analysis_protocol_inputs(
    *,
    wrapped_env: Any,
    resolved_hydra_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture outcome-free live measurement semantics from initialized managers.

    This function does not reset or step the environment.  It requires the
    scientific recorder to have emitted zero rows and returns only JSON-ready
    payloads needed to freeze the analysis protocol (plus an auditable runtime
    fingerprint).  The caller must still exit before policy evaluation.
    """

    raw_env = getattr(wrapped_env, "env", None)
    _require(raw_env is not None, "wrapped environment does not expose its manager environment")
    recorder_manager = getattr(raw_env, "recorder_manager", None)
    terms = getattr(recorder_manager, "_terms", None)
    _require(isinstance(terms, Mapping), "live recorder manager does not expose ordered terms")
    term = terms.get("failure_atlas")
    _require(
        term is not None and getattr(term, "_enabled", False) is True,
        "atlas recorder is not enabled",
    )
    _require(
        getattr(term, "_record_count", None) == 0,
        "live protocol inputs must be captured before recorder outcomes",
    )
    termination_contract = getattr(term, "_termination_trace_contract", None)
    termination_digest = getattr(term, "_termination_trace_contract_sha256", None)
    _require(isinstance(termination_contract, Mapping), "termination trace is not installed")
    _require(
        canonical_sha256(termination_contract) == termination_digest,
        "live termination trace contract digest is invalid",
    )
    assembler = getattr(term, "_assembler", None)
    thresholds = getattr(assembler, "_threshold_record", None)
    thresholds_digest = getattr(assembler, "_threshold_sha256", None)
    _require(isinstance(thresholds, Mapping), "live recorder thresholds are unavailable")
    _require(
        canonical_sha256(thresholds) == thresholds_digest,
        "live recorder threshold digest is invalid",
    )
    bindings = getattr(term, "_bindings", None)
    _require(bindings is not None, "live recorder sensor bindings are unavailable")
    manager_env = resolved_hydra_config.get("manager_env")
    _require(isinstance(manager_env, Mapping), "resolved config lacks manager_env")
    recorder_configs = manager_env.get("recorders")
    _require(isinstance(recorder_configs, Mapping), "resolved config lacks recorder config")
    failure_config = recorder_configs.get("failure_atlas")
    _require(isinstance(failure_config, Mapping), "resolved config lacks failure_atlas recorder")

    from gear_sonic.research.lace.isaac_recorder import capture_resolved_event_configuration

    event_config = capture_resolved_event_configuration(raw_env.event_manager)
    sensor_semantics = {
        "command_name": str(bindings.command_name),
        "robot_name": str(bindings.robot_name),
        "joint_action_name": str(bindings.joint_action_name),
        "contact_sensor_name": str(bindings.contact_sensor_name),
        "foot_body_names": list(bindings.foot_body_names),
        "contact_force_threshold": float(bindings.contact_force_threshold),
        "ground_normal_axis": int(bindings.ground_normal_axis),
        "actual_contact": "latest_net_forces_w_history_norm_threshold",
        "reference_contact": "sonic_feet_channel_ground_height_rule",
        "foot_slip": "link_origin_velocity_tangent_to_configured_plane",
        "torque": "requested_action_target_minus_joint_state_and_applied_joint_effort",
        "joint_limits": "live_soft_joint_position_limits",
    }
    score_window_config = {
        "rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
        "score_window_seconds": float(thresholds["score_window_seconds"]),
        "timestep_seconds": float(raw_env.step_dt),
        "sample_count_rule": "max(1,floor(score_window_seconds/timestep_seconds+1e-12))",
    }
    result = {
        "probe_thresholds": dict(thresholds),
        "probe_thresholds_sha256": str(thresholds_digest),
        "termination_predicates": dict(termination_contract),
        "termination_predicates_sha256": str(termination_digest),
        "sensor_semantics": sensor_semantics,
        "score_window_config": score_window_config,
        "domain_randomization_config": event_config,
        "resolved_recorder_config": dict(failure_config),
        "recorder_term_type": f"{type(term).__module__}:{type(term).__qualname__}",
        "recorder_record_count": 0,
        "num_envs": int(raw_env.num_envs),
        "step_dt_seconds": float(raw_env.step_dt),
        "environment_fingerprint": capture_environment_fingerprint(raw_env),
    }
    _require(
        getattr(term, "_record_count", None) == 0,
        "recorder emitted an outcome during live protocol capture",
    )
    return result


def materialize_live_instrument(
    *,
    handshake_path: str | Path,
    expected_handshake_file_sha256: str,
    launch_plan_path: str | Path,
    repo_root: str | Path,
    resolved_hydra_config: Mapping[str, Any],
    loaded_checkpoint_bundle: Mapping[str, Any],
    wrapped_env: Any,
) -> MaterializedInstrument:
    """Extract verified live recorder inputs and materialize before rollout."""

    raw_env = getattr(wrapped_env, "env", None)
    _require(raw_env is not None, "wrapped environment does not expose its manager environment")
    live_inputs = capture_live_analysis_protocol_inputs(
        wrapped_env=wrapped_env,
        resolved_hydra_config=resolved_hydra_config,
    )

    command = raw_env.command_manager.get_term(str(live_inputs["sensor_semantics"]["command_name"]))
    batch = getattr(command, "atlas_probe_batch", None)
    _require(batch is not None, "live motion command lacks its locked atlas batch")
    handshake = _load_json_object(handshake_path, "runtime handshake")
    _require(batch.schedule_sha256 == handshake.get("schedule_sha256"), "live schedule drifted")
    _require(
        batch.checkpoint_sha256 == handshake.get("checkpoint_sha256"), "live checkpoint drifted"
    )
    _require(list(batch.rollout_ids) == handshake.get("rollout_ids"), "live rollout ids drifted")

    recorder_config = {
        "resolved_hydra_term": dict(live_inputs["resolved_recorder_config"]),
        "runtime": {
            "term_type": live_inputs["recorder_term_type"],
            "num_envs": live_inputs["num_envs"],
            "step_dt_seconds": live_inputs["step_dt_seconds"],
            "effective_probe_thresholds_sha256": live_inputs["probe_thresholds_sha256"],
            "cell_identity": {
                "schedule_sha256": handshake["schedule_sha256"],
                "checkpoint_sha256": handshake["checkpoint_sha256"],
                "probe_policy_id": handshake["probe_policy_id"],
                "rollout_ids": list(handshake["rollout_ids"]),
            },
        },
    }
    return materialize_instrument_from_payloads(
        handshake_path=handshake_path,
        expected_handshake_file_sha256=expected_handshake_file_sha256,
        launch_plan_path=launch_plan_path,
        repo_root=repo_root,
        resolved_hydra_config=resolved_hydra_config,
        environment_fingerprint=live_inputs["environment_fingerprint"],
        probe_thresholds=live_inputs["probe_thresholds"],
        recorder_config=recorder_config,
        sensor_semantics=live_inputs["sensor_semantics"],
        termination_predicates=live_inputs["termination_predicates"],
        score_window_config=live_inputs["score_window_config"],
        domain_randomization_config=live_inputs["domain_randomization_config"],
        loaded_checkpoint_bundle=loaded_checkpoint_bundle,
    )


def _artifact_binding(path: Any, name: str) -> dict[str, str]:
    _require(isinstance(path, (str, Path)) and str(path), f"{name} path is invalid")
    candidate = Path(path).expanduser()
    _require(candidate.is_absolute(), f"{name} path must be absolute")
    _require(not candidate.is_symlink(), f"{name} path may not be a symlink")
    resolved = candidate.resolve()
    _require(candidate == resolved, f"{name} path must be canonical")
    _require(resolved.is_file(), f"{name} artifact is missing: {resolved}")
    return {"path": str(resolved), "file_sha256": file_sha256(resolved)}


def _resolve_lock_input_path(value: Any, *, repo_root: Path, name: str) -> Path:
    _require(isinstance(value, str) and value, f"{name} path is invalid")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _build_rollout_artifacts(
    materialized: MaterializedInstrument,
    *,
    raw_output: Path,
    instrumented_output: Path,
) -> dict[str, dict[str, str]]:
    plan = materialized.launch_plan
    lock_path = Path(str(plan.get("schedule_lock_path", ""))).expanduser()
    _require(lock_path.is_absolute(), "launch plan schedule_lock_path must be absolute")
    lock = _load_json_object(lock_path, "schedule lock")
    inputs = lock.get("inputs")
    _require(isinstance(inputs, Mapping), "schedule lock inputs are missing")
    split_path = _resolve_lock_input_path(
        inputs.get("split_manifest"),
        repo_root=materialized.repo_root,
        name="schedule lock split manifest",
    )
    inventory_path = _resolve_lock_input_path(
        inputs.get("reference_length_inventory"),
        repo_root=materialized.repo_root,
        name="schedule lock reference-length inventory",
    )
    checkpoint_bundle = materialized.loaded_checkpoint_bundle
    artifacts = {
        "runtime_handshake": _artifact_binding(
            materialized.handshake_path,
            "runtime handshake",
        ),
        "launch_plan": _artifact_binding(materialized.launch_plan_path, "launch plan"),
        "checkpoint": _artifact_binding(
            checkpoint_bundle["checkpoint_path"],
            "checkpoint",
        ),
        "checkpoint_config": _artifact_binding(
            checkpoint_bundle["config_path"],
            "checkpoint config",
        ),
        "instrument": _artifact_binding(
            materialized.handshake["instrument_output_path"],
            "instrument",
        ),
        "instrument_binding": _artifact_binding(
            materialized.handshake["instrument_binding_output_path"],
            "instrument binding",
        ),
        "raw_rollouts": _artifact_binding(raw_output, "raw rollouts"),
        "instrumented_rollouts": _artifact_binding(
            instrumented_output,
            "instrumented rollouts",
        ),
        "schedule_lock": _artifact_binding(lock_path.resolve(), "schedule lock"),
        "schedule": _artifact_binding(plan.get("schedule_path"), "schedule"),
        "schedule_spec": _artifact_binding(plan.get("schedule_spec_path"), "schedule spec"),
        "split_manifest": _artifact_binding(split_path, "split manifest"),
        "reference_length_inventory": _artifact_binding(
            inventory_path,
            "reference-length inventory",
        ),
        "analysis_protocol": _artifact_binding(
            materialized.handshake["analysis_protocol_path"],
            "analysis protocol",
        ),
    }
    _require(set(artifacts) == _ROLLOUT_ARTIFACT_NAMES, "rollout artifact set is incomplete")
    return artifacts


def validate_rollout_binding(
    raw: Mapping[str, Any],
    *,
    verify_artifacts: bool = False,
) -> None:
    """Validate a receipt commit marker and optionally every bound file byte."""

    _require(isinstance(raw, Mapping), "rollout binding must be a mapping")
    receipt = dict(raw)
    _require(set(receipt) == _ROLLOUT_BINDING_FIELDS, "rollout binding fields are invalid")
    _require(receipt.get("kind") == ROLLOUT_BINDING_KIND, "rollout binding kind is invalid")
    _require(
        receipt.get("schema_version") == ROLLOUT_BINDING_SCHEMA_VERSION,
        "rollout binding schema_version is invalid",
    )
    _require(receipt.get("scientific_use") is True, "rollout binding must be scientific")
    cell = receipt.get("cell")
    _require(isinstance(cell, Mapping) and set(cell) == _CELL_FIELDS, "receipt cell is invalid")
    _require(
        isinstance(cell.get("probe_policy_id"), str) and cell["probe_policy_id"],
        "receipt cell policy is invalid",
    )
    _sha256(cell.get("checkpoint_sha256"), "receipt cell checkpoint_sha256")
    _require(isinstance(cell.get("phase_id"), str) and cell["phase_id"], "receipt phase invalid")
    for field in (
        "schedule_sha256",
        "split_sha256",
        "split_selection_sha256",
        "launch_plan_sha256",
        "runtime_handshake_sha256",
        "instrument_binding_sha256",
        "instrument_manifest_sha256",
        "episode_instrument_sha256",
        "measurement_family_manifest_sha256",
        "episode_measurement_family_sha256",
        "source_bundle_sha256",
        "dataset_binding_sha256",
        "robot_contract_readback_sha256",
        "analysis_protocol_file_sha256",
        "analysis_protocol_sha256",
        ROLLOUT_BINDING_DIGEST_FIELD,
    ):
        _sha256(receipt.get(field), f"rollout_binding.{field}")
    ids = receipt.get("rollout_ids")
    _require(
        isinstance(ids, list)
        and bool(ids)
        and all(isinstance(item, str) and item for item in ids)
        and len(ids) == len(set(ids)),
        "rollout binding rollout_ids are invalid",
    )
    _require(receipt.get("rollout_count") == len(ids), "rollout binding count drifted")
    _require(
        receipt.get("rollout_order") == "frozen_handshake_rollout_id_order",
        "rollout binding order semantics are invalid",
    )
    family = receipt.get("measurement_family")
    _require(isinstance(family, Mapping), "rollout binding measurement_family is missing")
    from gear_sonic.research.lace.instrument import validate_measurement_family

    validate_measurement_family(family)
    _require(
        family[MEASUREMENT_FAMILY_DIGEST_FIELD] == receipt["measurement_family_manifest_sha256"],
        "rollout binding family manifest digest drifted",
    )
    _require(
        episode_measurement_family_sha256(family) == receipt["episode_measurement_family_sha256"],
        "rollout binding episode family digest drifted",
    )
    _require(
        receipt["source_bundle_sha256"] == family["source_bundle_sha256"],
        "rollout binding source bundle drifted from its measurement family",
    )
    artifacts = receipt.get("artifacts")
    _require(
        isinstance(artifacts, Mapping) and set(artifacts) == _ROLLOUT_ARTIFACT_NAMES,
        "rollout binding artifacts are incomplete",
    )
    for name, raw_binding in artifacts.items():
        _require(
            isinstance(raw_binding, Mapping) and set(raw_binding) == _ARTIFACT_BINDING_FIELDS,
            f"rollout binding artifact {name!r} is invalid",
        )
        path = Path(str(raw_binding.get("path", "")))
        _require(path.is_absolute(), f"rollout binding artifact {name!r} path is invalid")
        _sha256(raw_binding.get("file_sha256"), f"rollout binding artifact {name!r}")
        if verify_artifacts:
            _require(not path.is_symlink(), f"rollout binding artifact {name!r} is a symlink")
            _require(path.is_file(), f"rollout binding artifact {name!r} is missing")
            _require(
                path.resolve() == path,
                f"rollout binding artifact {name!r} path is not canonical",
            )
            _require(
                file_sha256(path) == raw_binding["file_sha256"],
                f"rollout binding artifact {name!r} digest drifted",
            )
    protocol_path = Path(str(receipt.get("analysis_protocol_path", "")))
    _require(
        protocol_path.is_absolute()
        and artifacts["analysis_protocol"]["path"] == str(protocol_path)
        and artifacts["analysis_protocol"]["file_sha256"]
        == receipt["analysis_protocol_file_sha256"],
        "rollout binding analysis protocol artifact drifted",
    )
    _load_bound_analysis_protocol(
        protocol_path,
        expected_file_sha256=receipt["analysis_protocol_file_sha256"],
        expected_protocol_sha256=receipt["analysis_protocol_sha256"],
    )
    _require(
        receipt[ROLLOUT_BINDING_DIGEST_FIELD]
        == canonical_sha256(receipt, digest_field=ROLLOUT_BINDING_DIGEST_FIELD),
        "rollout_binding_sha256 mismatch",
    )


def finalize_scientific_rollout(materialized: MaterializedInstrument) -> dict[str, Any]:
    """Verify complete recorder output, add instrument identity, and bind bytes."""

    handshake = dict(materialized.handshake)
    raw_output = Path(handshake["rollout_output_path"])
    instrumented_output = Path(handshake["instrumented_rollout_output_path"])
    receipt_output = Path(handshake["rollout_binding_output_path"])
    _require(raw_output.is_file(), f"scientific recorder output is missing: {raw_output}")
    if receipt_output.exists():
        raise FileExistsError(f"refusing to overwrite runtime artifact: {receipt_output}")
    records = []
    with raw_output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"rollout JSONL line {line_number} is blank")
            record = _strict_json_loads(line, f"rollout JSONL line {line_number}")
            _require(isinstance(record, dict), f"rollout line {line_number} is not an object")
            records.append(record)
    expected_ids = list(handshake["rollout_ids"])
    assignments = materialized.launch_plan.get("atlas_probe_assignments")
    _require(
        isinstance(assignments, list) and len(assignments) == len(expected_ids),
        "launch plan atlas assignments do not exactly cover the cell",
    )
    _require(
        all(isinstance(item, Mapping) for item in assignments),
        "launch plan atlas assignments must be mappings",
    )
    scheduled_by_id = {item.get("rollout_id"): item for item in assignments}
    _require(
        list(scheduled_by_id) == expected_ids and len(scheduled_by_id) == len(expected_ids),
        "launch plan atlas assignment order/identity drifted",
    )
    from gear_sonic.research.lace.atlas import (
        _validate_episode_against_schedule,
        _validate_runtime_realization,
        _validate_termination_multi_hot,
    )

    by_id: dict[str, dict[str, Any]] = {}
    robot_contract_digests: set[str] = set()
    for index, record in enumerate(records):
        rollout_id = record.get("rollout_id")
        _require(rollout_id in expected_ids, f"unexpected scientific rollout id: {rollout_id!r}")
        _require(rollout_id not in by_id, f"duplicate scientific rollout id: {rollout_id!r}")
        scheduled = scheduled_by_id[rollout_id]
        _validate_episode_against_schedule(
            record,
            scheduled,
            schedule_sha256=handshake["schedule_sha256"],
            index=index,
        )
        for episode_field, schedule_field in (
            ("partition", "partition"),
            ("motion_key", "motion_key"),
            ("probe_policy_id", "probe_policy_id"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("domain_randomization_seed", "domain_randomization_seed"),
            ("runtime_rng_seed", "runtime_rng_seed"),
            ("phase_id", "phase_id"),
            ("target_fraction", "target_fraction"),
            ("realized_fraction", "realized_fraction"),
            ("repeat_index", "repeat_index"),
            ("reference_num_steps", "reference_num_steps"),
            ("reference_start_step", "start_step"),
            ("split_sha256", "split_sha256"),
            ("split_selection_sha256", "split_selection_sha256"),
        ):
            _require(
                record.get(episode_field) == scheduled.get(schedule_field),
                f"record {episode_field} drifted from its launch-plan assignment",
            )
        _require(record.get("scientific_runtime_ready") is True, "record is not runtime-ready")
        _require(
            record.get("scientific_runtime_blockers") == [],
            "record has scientific runtime blockers",
        )
        _require(record.get("completion_reason") == "episode_end", "record is not a full episode")
        _require(
            record.get("policy_id") == record.get("probe_policy_id"),
            "record policy alias drifted",
        )
        _require(
            record.get("domain_randomization_seed_semantics")
            == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
            "record DR seed semantics drifted",
        )
        _require(
            record.get("runtime_rng_seed_semantics") == RUNTIME_RNG_SEED_SEMANTICS,
            "record runtime RNG semantics drifted",
        )
        _require(
            record.get("runtime_rng_seed_readback") == scheduled["runtime_rng_seed"],
            "record runtime RNG seed readback drifted",
        )
        _require(
            record.get("runtime_rng_seed_readback_source") == "env.cfg.seed",
            "record runtime RNG seed readback source drifted",
        )
        _require(record.get("termination_multi_hot_available") is True, "record lacks multi-hot")
        _require(
            record.get("termination_semantics") == materialized.instrument["termination_semantics"],
            "record termination semantics drifted",
        )
        _require(
            record.get("probe_thresholds") == materialized.instrument["probe_thresholds"],
            "record threshold payload drifted",
        )
        _require(
            record.get("probe_thresholds_sha256")
            == materialized.instrument["probe_thresholds_sha256"],
            "record threshold contract drifted",
        )
        _validate_runtime_realization(
            record.get("domain_randomization_realization"),
            record.get("domain_randomization_realization_sha256"),
            episode=record,
            index=index,
        )
        realization = record["domain_randomization_realization"]
        _require(
            realization.get("resolved_event_configuration")
            == materialized.instrument["domain_randomization_config"],
            "record resolved DR event configuration drifted",
        )
        _validate_termination_multi_hot(record, index=index)
        trace = record.get("termination_multi_hot")
        _require(isinstance(trace, Mapping), "record lacks termination trace")
        _require(
            trace.get("contract_sha256")
            == materialized.instrument["termination_predicates_sha256"],
            "record termination contract drifted",
        )
        _require(
            trace.get("contract") == materialized.instrument["termination_predicates"],
            "record termination contract payload drifted",
        )
        validate_scientific_probe_episode(
            record,
            score_window_config=materialized.instrument["score_window_config"],
            probe_thresholds=materialized.instrument["probe_thresholds"],
        )
        robot_contract = record.get("robot_contract_readback")
        robot_contract_digest = record.get("robot_contract_readback_sha256")
        _require(isinstance(robot_contract, Mapping), "record robot contract readback is missing")
        digest = _sha256(robot_contract_digest, "record robot_contract_readback_sha256")
        validate_robot_contract_readback(robot_contract)
        _require(
            canonical_sha256(robot_contract) == digest,
            "record robot contract readback digest drifted",
        )
        robot_contract_digests.add(digest)
        for field in (
            "instrument_sha256",
            "instrument_manifest_sha256",
            "instrument_sidecar_file_sha256",
            "instrument_binding_sha256",
            "measurement_family_sha256",
            "measurement_family_manifest_sha256",
        ):
            _require(
                field not in record, f"recorder output already contains reserved field {field}"
            )
        record.update(
            {
                "instrument_sha256": materialized.episode_instrument_sha256,
                "instrument_manifest_sha256": materialized.instrument[INSTRUMENT_DIGEST_FIELD],
                "instrument_sidecar_file_sha256": materialized.instrument_file_sha256,
                "instrument_binding_sha256": materialized.binding[INSTRUMENT_BINDING_DIGEST_FIELD],
                "measurement_family_sha256": (materialized.episode_measurement_family_sha256),
                "measurement_family_manifest_sha256": materialized.measurement_family[
                    MEASUREMENT_FAMILY_DIGEST_FIELD
                ],
            }
        )
        by_id[str(rollout_id)] = record
    _require(set(by_id) == set(expected_ids), "scientific recorder output is incomplete")
    _require(
        len(robot_contract_digests) == 1,
        "scientific cell contains heterogeneous robot contracts",
    )

    instrumented_bytes = b"".join(
        (
            json.dumps(
                by_id[rollout_id],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for rollout_id in expected_ids
    )
    if instrumented_output.exists():
        _require(
            instrumented_output.is_file()
            and not instrumented_output.is_symlink()
            and instrumented_output.read_bytes() == instrumented_bytes,
            "existing instrumented rollout is not the deterministic enrichment of immutable raw",
        )
    else:
        _publish_new_bytes(
            instrumented_output,
            instrumented_bytes,
            label="instrumented rollout",
        )
    artifacts = _build_rollout_artifacts(
        materialized,
        raw_output=raw_output,
        instrumented_output=instrumented_output,
    )
    assignments_by_id = {
        str(item["rollout_id"]): item
        for item in materialized.launch_plan["atlas_probe_assignments"]
    }
    split_sha256_values = {item["split_sha256"] for item in assignments_by_id.values()}
    split_selection_values = {item["split_selection_sha256"] for item in assignments_by_id.values()}
    _require(len(split_sha256_values) == 1, "cell has heterogeneous split digests")
    _require(len(split_selection_values) == 1, "cell has heterogeneous split selections")
    receipt: dict[str, Any] = {
        "kind": ROLLOUT_BINDING_KIND,
        "schema_version": ROLLOUT_BINDING_SCHEMA_VERSION,
        "scientific_use": True,
        "cell": {field: materialized.launch_plan["cell"][field] for field in _CELL_FIELDS},
        "schedule_sha256": handshake["schedule_sha256"],
        "split_sha256": next(iter(split_sha256_values)),
        "split_selection_sha256": next(iter(split_selection_values)),
        "rollout_ids": expected_ids,
        "launch_plan_sha256": materialized.launch_plan_sha256,
        "runtime_handshake_sha256": handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD],
        "instrument_binding_sha256": materialized.binding[INSTRUMENT_BINDING_DIGEST_FIELD],
        "instrument_manifest_sha256": materialized.instrument[INSTRUMENT_DIGEST_FIELD],
        "episode_instrument_sha256": materialized.episode_instrument_sha256,
        "measurement_family": materialized.measurement_family,
        "measurement_family_manifest_sha256": materialized.measurement_family[
            MEASUREMENT_FAMILY_DIGEST_FIELD
        ],
        "episode_measurement_family_sha256": (materialized.episode_measurement_family_sha256),
        "source_bundle_sha256": materialized.instrument["source_bundle_sha256"],
        "dataset_binding_sha256": canonical_sha256(materialized.launch_plan["dataset_binding"]),
        "robot_contract_readback_sha256": next(iter(robot_contract_digests)),
        "analysis_protocol_path": handshake["analysis_protocol_path"],
        "analysis_protocol_file_sha256": handshake["analysis_protocol_file_sha256"],
        "analysis_protocol_sha256": handshake["analysis_protocol_sha256"],
        "artifacts": artifacts,
        "rollout_count": len(expected_ids),
        "rollout_order": "frozen_handshake_rollout_id_order",
    }
    receipt[ROLLOUT_BINDING_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
    )
    validate_rollout_binding(receipt, verify_artifacts=True)
    # The receipt is the last-write commit marker. Raw recorder bytes are never
    # rewritten; an instrumented file without this receipt is an explicit,
    # collector-rejected half-state after a crash or failed finalization.
    receipt_path = _publish_new_bytes(
        receipt_output,
        _pretty_json_bytes(receipt),
        label="scientific rollout receipt",
    )
    return {
        "rollout_binding_path": str(receipt_path),
        "rollout_binding_file_sha256": file_sha256(receipt_path),
        **receipt,
    }
