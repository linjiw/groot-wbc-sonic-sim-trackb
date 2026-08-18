"""Fail-closed loading of one vectorized atlas-probe schedule cell.

The simulator-facing command accepts exact rollout rows, but manually copying a
list of rows into Hydra is both fragile and unauditable.  This module keeps the
selection path CPU-only: it validates a locked schema-v2 schedule and all of its
bound inputs, selects one complete common-random-number cell, and resolves the
rows to an :class:`AtlasProbeBatch` before an Isaac process is launched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.research.lace.atlas_probe_mode import (
    AtlasProbeBatch,
    resolve_atlas_probe_batch,
)
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    reference_num_steps_by_motion,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import (
    SCHEDULE_DIGEST_FIELD,
    SCHEDULE_SCHEMA_VERSION,
    build_rollout_schedule,
    derive_runtime_rng_seed,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import canonical_sha256

SCHEDULE_LOCK_KIND = "lace_probe_rollout_schedule_lock"
SCHEDULE_LOCK_SCHEMA_VERSION = 1
LAUNCH_PLAN_KIND = "lace_atlas_probe_launch_plan"
LAUNCH_PLAN_SCHEMA_VERSION = 1
LAUNCH_PLAN_DIGEST_FIELD = "launch_plan_sha256"

_SCHEDULE_SPEC_REQUIRED_FIELDS = {
    "schema_version",
    "reference_length_inventory_sha256",
    "probe_policies",
    "domain_randomization_seeds",
    "phase_targets",
    "repeats",
}
_SCHEDULE_SPEC_OPTIONAL_FIELDS = {"rollout_id_prefix"}

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LockedRolloutSchedule:
    """A schedule whose bytes, self-digest, split, and length inventory agree."""

    lock_path: Path
    schedule_path: Path
    schedule_file_sha256: str
    manifest: dict[str, Any]
    split_path: Path
    split_manifest: dict[str, Any]
    reference_length_inventory_path: Path
    reference_length_inventory: dict[str, Any]
    schedule_spec_path: Path
    schedule_spec: dict[str, Any]


@dataclass(frozen=True)
class AtlasProbeSelection:
    """One full schedule cell and its CPU-resolved vectorized assignment."""

    locked_schedule: LockedRolloutSchedule
    assignments: tuple[dict[str, Any], ...]
    batch: AtlasProbeBatch


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    return value


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without importing Torch or Isaac."""

    resolved = Path(path)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash file {resolved}: {error}") from error
    return digest.hexdigest()


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read {name} JSON object {path}: {error}") from error
    _require(bool(raw.strip()), f"{name} at {path} must not be blank")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"cannot read {name} JSON object {path}: {error}") from error

    def reject_nonfinite(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_nonfinite(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_nonfinite(child, f"{location}[{index}]")
        elif isinstance(item, float):
            _require(math.isfinite(item), f"{name} contains non-finite number at {location}")

    reject_nonfinite(payload, "$")
    _require(isinstance(payload, dict), f"{name} at {path} must contain a JSON object")
    return payload


def _resolve_path(value: Any, *, repo_root: Path, name: str) -> Path:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _require_file_hash(path: Path, expected: Any, name: str) -> str:
    expected_digest = _sha256(expected, name)
    _require(path.is_file(), f"{name} references a missing file: {path}")
    actual_digest = sha256_file(path)
    _require(
        actual_digest == expected_digest,
        f"{name} mismatch for {path}: expected {expected_digest}, computed {actual_digest}",
    )
    return actual_digest


def load_locked_rollout_schedule(
    lock_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> LockedRolloutSchedule:
    """Load and externally revalidate a schema-v2 rollout schedule lock.

    Unlike schedule self-validation alone, the lock pins the artifact's exact
    bytes and the exact split and reference-length inventory files.  A launch
    therefore cannot silently follow a schedule whose self-digest was simply
    recomputed after an edit.
    """

    root = _REPO_ROOT if repo_root is None else Path(repo_root).expanduser().resolve()
    resolved_lock = Path(lock_path).expanduser().resolve()
    lock = _load_json_object(resolved_lock, "schedule lock")
    _require(lock.get("kind") == SCHEDULE_LOCK_KIND, "invalid schedule lock kind")
    _require(
        lock.get("schema_version") == SCHEDULE_LOCK_SCHEMA_VERSION,
        "unsupported schedule lock schema_version",
    )
    _require(isinstance(lock.get("scientific_use"), bool), "lock scientific_use must be boolean")

    artifact = lock.get("artifact")
    _require(isinstance(artifact, Mapping), "schedule lock artifact must be a mapping")
    _require(
        artifact.get("schedule_schema_version") == SCHEDULE_SCHEMA_VERSION,
        "schedule lock must pin schema version 2",
    )
    schedule_path = _resolve_path(artifact.get("path"), repo_root=root, name="artifact.path")
    schedule_file_sha256 = _require_file_hash(
        schedule_path,
        artifact.get("file_sha256"),
        "artifact.file_sha256",
    )
    schedule = _load_json_object(schedule_path, "rollout schedule")
    _require(
        schedule.get("schema_version") == SCHEDULE_SCHEMA_VERSION,
        "atlas batch loading requires a schema-v2 rollout schedule",
    )
    expected_schedule_sha256 = _sha256(
        artifact.get(SCHEDULE_DIGEST_FIELD),
        f"artifact.{SCHEDULE_DIGEST_FIELD}",
    )
    _require(
        schedule.get(SCHEDULE_DIGEST_FIELD) == expected_schedule_sha256,
        "locked schedule_sha256 does not match the schedule artifact",
    )
    _require(
        schedule.get("scientific_use") is lock["scientific_use"],
        "schedule lock scientific_use does not match the artifact",
    )

    inputs = lock.get("inputs")
    _require(isinstance(inputs, Mapping), "schedule lock inputs must be a mapping")
    split_path = _resolve_path(
        inputs.get("split_manifest"),
        repo_root=root,
        name="inputs.split_manifest",
    )
    _require_file_hash(
        split_path,
        inputs.get("split_manifest_sha256"),
        "inputs.split_manifest_sha256",
    )
    split_manifest = _load_json_object(split_path, "split manifest")

    inventory_path = _resolve_path(
        inputs.get("reference_length_inventory"),
        repo_root=root,
        name="inputs.reference_length_inventory",
    )
    _require_file_hash(
        inventory_path,
        inputs.get("reference_length_inventory_file_sha256"),
        "inputs.reference_length_inventory_file_sha256",
    )
    inventory = _load_json_object(inventory_path, "reference-length inventory")
    expected_inventory_sha256 = _sha256(
        inputs.get("reference_length_inventory_sha256"),
        "inputs.reference_length_inventory_sha256",
    )
    _require(
        inventory.get(REFERENCE_LENGTH_DIGEST_FIELD) == expected_inventory_sha256,
        "locked reference-length inventory digest does not match the artifact",
    )

    schedule_spec_path = _resolve_path(
        inputs.get("schedule_spec"),
        repo_root=root,
        name="inputs.schedule_spec",
    )
    _require_file_hash(
        schedule_spec_path,
        inputs.get("schedule_spec_sha256"),
        "inputs.schedule_spec_sha256",
    )
    schedule_spec = _load_json_object(schedule_spec_path, "schedule spec")
    _require(
        set(schedule_spec) == _SCHEDULE_SPEC_REQUIRED_FIELDS | _SCHEDULE_SPEC_OPTIONAL_FIELDS
        or set(schedule_spec) == _SCHEDULE_SPEC_REQUIRED_FIELDS,
        "schedule spec fields do not match the schema-v2 build contract",
    )
    _require(schedule_spec.get("schema_version") == 2, "schedule spec schema_version must be 2")
    _require(
        schedule_spec.get("reference_length_inventory_sha256") == expected_inventory_sha256,
        "schedule spec reference-length digest does not match the locked inventory",
    )

    validate_reference_length_inventory(
        inventory,
        split_manifest=split_manifest,
        verify_digest=True,
    )
    validate_rollout_schedule(
        schedule,
        split_manifest=split_manifest,
        reference_length_inventory=inventory,
        verify_digest=True,
    )
    rebuilt_schedule = build_rollout_schedule(
        split_manifest,
        reference_length_inventory=inventory,
        probe_policies=schedule_spec["probe_policies"],
        domain_randomization_seeds=schedule_spec["domain_randomization_seeds"],
        phase_targets=schedule_spec["phase_targets"],
        repeats=schedule_spec["repeats"],
        rollout_id_prefix=schedule_spec.get("rollout_id_prefix", "lace-rollout"),
    )
    _require(
        rebuilt_schedule == schedule,
        "locked schedule artifact does not exactly reproduce from its schedule spec",
    )
    return LockedRolloutSchedule(
        lock_path=resolved_lock,
        schedule_path=schedule_path,
        schedule_file_sha256=schedule_file_sha256,
        manifest=schedule,
        split_path=split_path,
        split_manifest=split_manifest,
        reference_length_inventory_path=inventory_path,
        reference_length_inventory=inventory,
        schedule_spec_path=schedule_spec_path,
        schedule_spec=schedule_spec,
    )


def select_locked_atlas_probe_cell(
    locked_schedule: LockedRolloutSchedule,
    *,
    probe_policy_id: str,
    domain_randomization_seed: int,
    phase_id: str,
    repeat_index: int,
    expected_checkpoint_sha256: str,
    loaded_motion_keys: Sequence[str],
    loaded_reference_num_steps: Sequence[int],
    num_envs: int,
) -> AtlasProbeSelection:
    """Select every motion row in exactly one homogeneous schedule cell."""

    _require(
        isinstance(locked_schedule, LockedRolloutSchedule),
        "locked_schedule must come from load_locked_rollout_schedule",
    )
    _require(isinstance(probe_policy_id, str) and probe_policy_id, "policy id is invalid")
    _require(
        isinstance(domain_randomization_seed, int)
        and not isinstance(domain_randomization_seed, bool),
        "domain_randomization_seed must be an integer",
    )
    _require(isinstance(phase_id, str) and phase_id, "phase_id is invalid")
    _require(
        isinstance(repeat_index, int) and not isinstance(repeat_index, bool) and repeat_index >= 0,
        "repeat_index must be a nonnegative integer",
    )
    checkpoint_sha256 = _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    manifest = locked_schedule.manifest
    validate_rollout_schedule(manifest, verify_digest=True)

    policies = {
        str(record["id"]): str(record["checkpoint_sha256"]) for record in manifest["probe_policies"]
    }
    _require(
        probe_policy_id in policies,
        f"probe policy {probe_policy_id!r} is not declared by the schedule",
    )
    _require(
        policies[probe_policy_id] == checkpoint_sha256,
        f"checkpoint SHA-256 does not match probe policy {probe_policy_id!r}",
    )
    _require(
        domain_randomization_seed in manifest["domain_randomization_seeds"],
        "domain_randomization_seed is not declared by the schedule",
    )
    phase_ids = {record["phase_id"] for record in manifest["phase_targets"]}
    _require(phase_id in phase_ids, "phase_id is not declared by the schedule")
    _require(repeat_index in manifest["repeat_indices"], "repeat_index is not scheduled")

    rows = [
        dict(row)
        for row in manifest["rollouts"]
        if row["probe_policy_id"] == probe_policy_id
        and row["domain_randomization_seed"] == domain_randomization_seed
        and row["phase_id"] == phase_id
        and row["repeat_index"] == repeat_index
    ]
    expected_motion_order = list(manifest["selected_motion_keys"])
    observed_motion_order = [row["motion_key"] for row in rows]
    _require(
        observed_motion_order == expected_motion_order,
        "selected cell does not exactly cover every scheduled motion in frozen row order",
    )
    _require(
        len(rows) == manifest["motion_count"],
        "selected cell row count does not match schedule motion_count",
    )
    _require(
        num_envs == len(rows),
        f"num_envs {num_envs} does not match selected cell size {len(rows)}",
    )

    runtime_seeds = {row["runtime_rng_seed"] for row in rows}
    _require(len(runtime_seeds) == 1, "selected cell has heterogeneous runtime RNG seeds")
    expected_runtime_seed = derive_runtime_rng_seed(
        domain_randomization_seed,
        phase_id,
        repeat_index,
    )
    _require(
        runtime_seeds == {expected_runtime_seed},
        "selected cell runtime RNG seed violates the frozen derivation",
    )
    _require(
        {row["checkpoint_sha256"] for row in rows} == {checkpoint_sha256},
        "selected cell checkpoint rows do not match the requested policy checkpoint",
    )

    batch = resolve_atlas_probe_batch(
        rows,
        schedule_sha256=manifest[SCHEDULE_DIGEST_FIELD],
        loaded_motion_keys=loaded_motion_keys,
        loaded_reference_num_steps=loaded_reference_num_steps,
        num_envs=num_envs,
    )
    return AtlasProbeSelection(
        locked_schedule=locked_schedule,
        assignments=tuple(rows),
        batch=batch,
    )


def select_locked_atlas_probe_cell_for_filtered_library(
    locked_schedule: LockedRolloutSchedule,
    *,
    probe_policy_id: str,
    domain_randomization_seed: int,
    phase_id: str,
    repeat_index: int,
    expected_checkpoint_sha256: str,
    num_envs: int,
) -> AtlasProbeSelection:
    """Resolve a cell against the exact filtered library emitted for Hydra.

    The simulator re-resolves the same rows against its live loaded ordering.
    This planning resolution uses the schedule's canonical motion order and the
    independently locked reference-length inventory; both are injected as a
    filter contract by :func:`build_eval_hydra_overrides`.
    """

    motion_keys = list(locked_schedule.manifest["selected_motion_keys"])
    lengths_by_motion = reference_num_steps_by_motion(locked_schedule.reference_length_inventory)
    return select_locked_atlas_probe_cell(
        locked_schedule,
        probe_policy_id=probe_policy_id,
        domain_randomization_seed=domain_randomization_seed,
        phase_id=phase_id,
        repeat_index=repeat_index,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        loaded_motion_keys=motion_keys,
        loaded_reference_num_steps=[lengths_by_motion[key] for key in motion_keys],
        num_envs=num_envs,
    )


def _hydra_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hydra_value(value: Any) -> str:
    """Encode nested primitive data in Hydra's override grammar.

    JSON object keys are quoted, which Hydra's override parser rejects. Values
    retain JSON quoting, while mapping keys use validated schedule names.
    """

    if isinstance(value, Mapping):
        parts = []
        for key in sorted(value):
            item = value[key]
            _require(
                isinstance(key, str) and key.replace("_", "").isalnum(),
                f"cannot encode Hydra mapping key {key!r}",
            )
            parts.append(f"{key}:{_hydra_value(item)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return _hydra_json(value)


def validate_motion_dataset_binding(
    selection: AtlasProbeSelection,
    *,
    robot_motion_root: str | Path,
    smpl_motion_root: str | Path | None,
    use_dummy_smpl: bool,
) -> dict[str, Any]:
    """Validate the motion roots that replace stale release-checkpoint paths.

    Robot files are byte-checked against the frozen reference-length inventory.
    A real SMPL directory is checked for direct ``<motion_key>.pkl`` coverage and
    byte-bound in the launch plan.  Pilot runs may explicitly use SONIC's
    ``dummy`` SMPL mode because the frozen policy selector is G1-only; scientific
    schedules must provide the real SMPL files.
    """

    _require(isinstance(selection, AtlasProbeSelection), "selection is invalid")
    _require(
        (smpl_motion_root is None) is bool(use_dummy_smpl),
        "choose exactly one of a real SMPL root or explicit dummy SMPL mode",
    )
    _require(
        not (selection.locked_schedule.manifest["scientific_use"] and use_dummy_smpl),
        "scientific schedules require a real SMPL motion root",
    )
    robot_root = Path(robot_motion_root).expanduser().resolve()
    _require(robot_root.is_dir(), f"robot motion root is not a directory: {robot_root}")
    inventory_rows = {
        str(record["motion_key"]): record
        for record in selection.locked_schedule.reference_length_inventory["motions"]
    }
    robot_files: list[dict[str, str]] = []
    for motion_key in selection.batch.motion_keys:
        record = inventory_rows[motion_key]
        source_path = Path(record["source_path"]).expanduser().resolve()
        _require(
            source_path.parent == robot_root,
            f"frozen robot source for {motion_key!r} is not directly under robot motion root",
        )
        _require(
            source_path.name == f"{Path(motion_key).name}.pkl",
            f"frozen robot source filename does not match motion key {motion_key!r}",
        )
        actual_digest = sha256_file(source_path)
        _require(
            actual_digest == record["source_file_sha256"],
            f"robot motion SHA-256 mismatch for {motion_key!r}",
        )
        robot_files.append(
            {
                "motion_key": motion_key,
                "path": str(source_path),
                "sha256": actual_digest,
            }
        )

    if use_dummy_smpl:
        smpl_binding: dict[str, Any] = {
            "mode": "dummy",
            "motion_root": "dummy",
            "selected_file_set_sha256": None,
            "files": [],
        }
    else:
        assert smpl_motion_root is not None
        smpl_root = Path(smpl_motion_root).expanduser().resolve()
        _require(smpl_root.is_dir(), f"SMPL motion root is not a directory: {smpl_root}")
        smpl_files = []
        for motion_key in selection.batch.motion_keys:
            path = smpl_root / f"{Path(motion_key).name}.pkl"
            _require(path.is_file(), f"SMPL root is missing scheduled motion {path.name}")
            smpl_files.append(
                {
                    "motion_key": motion_key,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
        smpl_binding = {
            "mode": "real",
            "motion_root": str(smpl_root),
            "selected_file_set_sha256": canonical_sha256({"files": smpl_files}),
            "files": smpl_files,
        }
    binding = {
        "robot": {
            "motion_root": str(robot_root),
            "selected_file_set_sha256": canonical_sha256({"files": robot_files}),
            "files": robot_files,
        },
        "smpl": smpl_binding,
    }
    if selection.locked_schedule.manifest["scientific_use"]:
        validate_scientific_materialized_dataset_binding(
            selection.locked_schedule.reference_length_inventory,
            motion_keys=selection.batch.motion_keys,
            dataset_binding=binding,
        )
    return binding


def validate_scientific_materialized_dataset_binding(
    reference_length_inventory: Mapping[str, Any],
    *,
    motion_keys: Sequence[str],
    dataset_binding: Mapping[str, Any],
) -> str:
    """Prove both live modalities against the split-frozen materialized manifest.

    A per-cell hash only proves that a cell used *some* stable files.  Scientific
    collection additionally needs the robot and SMPL bytes selected before
    outcomes in the materialized paired-dataset manifest.  The returned digest
    is the manifest's full paired-dataset identity and is common to every cell.
    """

    _require(
        reference_length_inventory.get("artifact_mode") == "scientific"
        and reference_length_inventory.get("scientific_use") is True,
        "materialized dataset validation requires a scientific inventory",
    )
    keys = list(motion_keys)
    _require(
        bool(keys)
        and all(isinstance(key, str) and key for key in keys)
        and len(keys) == len(set(keys)),
        "scientific materialized motion keys are invalid",
    )
    provenance = reference_length_inventory.get("dataset_provenance")
    _require(isinstance(provenance, Mapping), "scientific dataset provenance is missing")
    manifest_value = provenance.get("materialized_manifest")
    _require(
        isinstance(manifest_value, str) and bool(manifest_value),
        "scientific inventory requires frozen materialized-manifest provenance",
    )
    manifest_candidate = Path(manifest_value).expanduser()
    _require(
        manifest_candidate.is_absolute() and not manifest_candidate.is_symlink(),
        "scientific materialized manifest must be an absolute non-symlink path",
    )
    manifest_path = manifest_candidate.resolve()
    _require(
        manifest_path == manifest_candidate and manifest_path.is_file(),
        "scientific materialized manifest must be canonical and existing",
    )
    _require(
        sha256_file(manifest_path) == provenance.get("materialized_manifest_sha256"),
        "scientific materialized manifest bytes drifted from the frozen inventory",
    )
    manifest = _load_json_object(manifest_path, "materialized dataset manifest")
    output = manifest.get("output")
    _require(isinstance(output, Mapping), "materialized dataset output is missing")
    paired_digest = _sha256(
        provenance.get("paired_dataset_sha256"),
        "inventory paired_dataset_sha256",
    )
    _require(
        output.get("paired_dataset_sha256") == paired_digest,
        "materialized paired-dataset digest drifted from the frozen inventory",
    )
    variants = output.get("variants")
    _require(isinstance(variants, list) and variants, "materialized variants are missing")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw_variant in enumerate(variants):
        _require(
            isinstance(raw_variant, Mapping),
            f"materialized variants[{index}] is invalid",
        )
        key = raw_variant.get("motion_key")
        _require(
            isinstance(key, str) and key and key not in indexed,
            f"materialized variants[{index}] has an invalid or duplicate motion key",
        )
        indexed[key] = raw_variant

    _require(
        isinstance(dataset_binding, Mapping) and set(dataset_binding) == {"robot", "smpl"},
        "scientific dataset binding fields are invalid",
    )
    manifest_root = manifest_path.parent
    for group_name, output_root_field in (("robot", "robot_dir"), ("smpl", "smpl_dir")):
        group = dataset_binding.get(group_name)
        _require(isinstance(group, Mapping), f"scientific {group_name} binding is invalid")
        if group_name == "smpl":
            _require(group.get("mode") == "real", "scientific SMPL binding must be real")
        relative_root = output.get(output_root_field)
        _require(
            isinstance(relative_root, str) and bool(relative_root),
            f"materialized output.{output_root_field} is invalid",
        )
        root_fragment = Path(relative_root)
        _require(
            not root_fragment.is_absolute() and ".." not in root_fragment.parts,
            f"materialized output.{output_root_field} must be a safe relative path",
        )
        root_candidate = manifest_root / root_fragment
        expected_root = root_candidate.resolve()
        _require(
            not root_candidate.is_symlink()
            and root_candidate == expected_root
            and expected_root.is_dir()
            and Path(str(group.get("motion_root", ""))) == expected_root,
            f"scientific {group_name} root differs from the frozen materialized manifest",
        )
        expected_files: list[dict[str, str]] = []
        for motion_key in keys:
            _require(
                motion_key in indexed,
                f"materialized manifest is missing scheduled motion {motion_key!r}",
            )
            record = indexed[motion_key].get(group_name)
            _require(
                isinstance(record, Mapping),
                f"materialized {motion_key}.{group_name} is invalid",
            )
            relative_path = record.get("path")
            _require(
                isinstance(relative_path, str) and bool(relative_path),
                f"materialized {motion_key}.{group_name}.path is invalid",
            )
            path_fragment = Path(relative_path)
            _require(
                not path_fragment.is_absolute() and ".." not in path_fragment.parts,
                f"materialized {motion_key}.{group_name}.path must be safe and relative",
            )
            path_candidate = manifest_root / path_fragment
            expected_path = path_candidate.resolve()
            _require(
                not path_candidate.is_symlink()
                and path_candidate == expected_path
                and expected_path.parent == expected_root
                and expected_path.name == f"{Path(motion_key).name}.pkl"
                and expected_path.is_file(),
                f"materialized {motion_key}.{group_name} path is not a canonical direct member",
            )
            expected_digest = _sha256(
                record.get("sha256"),
                f"materialized {motion_key}.{group_name}.sha256",
            )
            _require(
                sha256_file(expected_path) == expected_digest,
                f"materialized {motion_key}.{group_name} bytes drifted",
            )
            expected_files.append(
                {
                    "motion_key": motion_key,
                    "path": str(expected_path),
                    "sha256": expected_digest,
                }
            )
        _require(
            group.get("files") == expected_files
            and group.get("selected_file_set_sha256")
            == canonical_sha256({"files": expected_files}),
            f"scientific {group_name} binding differs from frozen materialized variants",
        )
    return paired_digest


def build_eval_hydra_overrides(
    selection: AtlasProbeSelection,
    *,
    checkpoint_path: str | Path,
    rollout_output_path: str | Path,
    robot_motion_root: str | Path,
    smpl_motion_root: str | Path | None,
    use_dummy_smpl: bool,
) -> tuple[str, ...]:
    """Build non-shell-expanded Hydra arguments for the SONIC eval entrypoint."""

    _require(isinstance(selection, AtlasProbeSelection), "selection is invalid")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    _require(checkpoint.is_file(), f"checkpoint is not a file: {checkpoint}")
    checkpoint_digest = sha256_file(checkpoint)
    _require(
        checkpoint_digest == selection.batch.checkpoint_sha256,
        "checkpoint file SHA-256 does not match the selected probe policy",
    )
    rollout_output = Path(rollout_output_path).expanduser().resolve()
    _require(
        rollout_output.suffix == ".jsonl",
        "rollout_output_path must use the .jsonl suffix",
    )
    _require(
        not rollout_output.exists(),
        f"refusing an existing rollout output path: {rollout_output}",
    )
    if not bool(selection.locked_schedule.manifest["scientific_use"]):
        _require(
            rollout_output.parent.is_dir(),
            f"rollout output parent does not exist: {rollout_output.parent}",
        )

    batch = selection.batch
    assignments = [dict(row) for row in selection.assignments]
    motion_keys = list(batch.motion_keys)
    dataset_binding = validate_motion_dataset_binding(
        selection,
        robot_motion_root=robot_motion_root,
        smpl_motion_root=smpl_motion_root,
        use_dummy_smpl=use_dummy_smpl,
    )
    max_render_steps = (
        max(
            reference_steps - start_step
            for reference_steps, start_step in zip(
                batch.reference_num_steps,
                batch.start_steps,
                strict=True,
            )
        )
        + 2
    )
    smpl_value = dataset_binding["smpl"]["motion_root"]
    return (
        f"checkpoint={_hydra_json(str(checkpoint))}",
        f"++num_envs={batch.num_envs}",
        f"++seed={batch.runtime_rng_seed}",
        "++headless=true",
        "++manager_env.config.terrain_type=plane",
        "++manager_env.config.render_results=false",
        "++manager_env.observations.policy.enable_corruption=false",
        "++manager_env.observations.tokenizer.enable_corruption=false",
        "++use_encoder=g1",
        "++eval_callbacks=[]",
        "++run_eval_loop=true",
        "++run_once=true",
        f"++max_render_steps={max_render_steps}",
        "~manager_env/recorders=empty",
        "+manager_env/recorders=lace_atlas",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file="
        f"{_hydra_json(dataset_binding['robot']['motion_root'])}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        f"{_hydra_json(smpl_value)}",
        f"++manager_env.commands.motion.filter_motion_keys={_hydra_json(motion_keys)}",
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
        f"{_hydra_json(motion_keys)}",
        "++manager_env.commands.motion.atlas_probe_mode=true",
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        "++manager_env.commands.motion.atlas_probe_schedule_sha256=" f"{batch.schedule_sha256}",
        "++manager_env.commands.motion.atlas_probe_assignments=" f"{_hydra_value(assignments)}",
        "++manager_env.recorders.failure_atlas.enabled=true",
        "++manager_env.recorders.failure_atlas.allow_append_existing=false",
        "++manager_env.recorders.failure_atlas.output_path=" f"{_hydra_json(str(rollout_output))}",
    )


def rebuild_eval_hydra_overrides_from_launch_plan(
    plan: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reconstruct the frozen base Hydra overrides from a stored launch plan."""

    assignments = plan.get("atlas_probe_assignments")
    motion_keys = plan.get("motion_keys")
    cell = plan.get("cell")
    dataset_binding = plan.get("dataset_binding")
    _require(
        isinstance(assignments, list) and assignments,
        "launch plan assignments are invalid",
    )
    _require(isinstance(motion_keys, list) and motion_keys, "launch plan motion keys invalid")
    _require(isinstance(cell, Mapping), "launch plan cell is invalid")
    _require(isinstance(dataset_binding, Mapping), "launch plan dataset binding invalid")
    robot = dataset_binding.get("robot")
    smpl = dataset_binding.get("smpl")
    _require(isinstance(robot, Mapping) and isinstance(smpl, Mapping), "dataset roots missing")
    checkpoint = plan.get("checkpoint_path")
    rollout_output = plan.get("rollout_output_path")
    max_render_steps = (
        max(int(row["reference_num_steps"]) - int(row["start_step"]) for row in assignments) + 2
    )
    return (
        f"checkpoint={_hydra_json(checkpoint)}",
        f"++num_envs={len(assignments)}",
        f"++seed={cell['runtime_rng_seed']}",
        "++headless=true",
        "++manager_env.config.terrain_type=plane",
        "++manager_env.config.render_results=false",
        "++manager_env.observations.policy.enable_corruption=false",
        "++manager_env.observations.tokenizer.enable_corruption=false",
        "++use_encoder=g1",
        "++eval_callbacks=[]",
        "++run_eval_loop=true",
        "++run_once=true",
        f"++max_render_steps={max_render_steps}",
        "~manager_env/recorders=empty",
        "+manager_env/recorders=lace_atlas",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file="
        f"{_hydra_json(robot.get('motion_root'))}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        f"{_hydra_json(smpl.get('motion_root'))}",
        f"++manager_env.commands.motion.filter_motion_keys={_hydra_json(motion_keys)}",
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
        f"{_hydra_json(motion_keys)}",
        "++manager_env.commands.motion.atlas_probe_mode=true",
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        "++manager_env.commands.motion.atlas_probe_schedule_sha256="
        f"{plan.get('schedule_sha256')}",
        "++manager_env.commands.motion.atlas_probe_assignments=" f"{_hydra_value(assignments)}",
        "++manager_env.recorders.failure_atlas.enabled=true",
        "++manager_env.recorders.failure_atlas.allow_append_existing=false",
        "++manager_env.recorders.failure_atlas.output_path=" f"{_hydra_json(rollout_output)}",
    )


def build_launch_plan(
    selection: AtlasProbeSelection,
    *,
    checkpoint_path: str | Path,
    rollout_output_path: str | Path,
    robot_motion_root: str | Path,
    smpl_motion_root: str | Path | None,
    use_dummy_smpl: bool,
) -> dict[str, Any]:
    """Serialize a selected batch and the generated eval overrides."""

    overrides = build_eval_hydra_overrides(
        selection,
        checkpoint_path=checkpoint_path,
        rollout_output_path=rollout_output_path,
        robot_motion_root=robot_motion_root,
        smpl_motion_root=smpl_motion_root,
        use_dummy_smpl=use_dummy_smpl,
    )
    batch = selection.batch
    locked = selection.locked_schedule
    dataset_binding = validate_motion_dataset_binding(
        selection,
        robot_motion_root=robot_motion_root,
        smpl_motion_root=smpl_motion_root,
        use_dummy_smpl=use_dummy_smpl,
    )
    max_render_steps = (
        max(
            reference_steps - start_step
            for reference_steps, start_step in zip(
                batch.reference_num_steps,
                batch.start_steps,
                strict=True,
            )
        )
        + 2
    )
    plan: dict[str, Any] = {
        "kind": LAUNCH_PLAN_KIND,
        "schema_version": LAUNCH_PLAN_SCHEMA_VERSION,
        "scientific_use": locked.manifest["scientific_use"],
        "schedule_lock_path": str(locked.lock_path),
        "schedule_path": str(locked.schedule_path),
        "schedule_file_sha256": locked.schedule_file_sha256,
        "schedule_sha256": batch.schedule_sha256,
        "schedule_spec_path": str(locked.schedule_spec_path),
        "cell": {
            "probe_policy_id": batch.probe_policy_id,
            "checkpoint_sha256": batch.checkpoint_sha256,
            "domain_randomization_seed": batch.domain_randomization_seed,
            "runtime_rng_seed": batch.runtime_rng_seed,
            "phase_id": batch.phase_id,
            "target_fraction": batch.target_fraction,
            "repeat_index": batch.repeat_index,
        },
        "num_envs": batch.num_envs,
        "motion_keys": list(batch.motion_keys),
        "reference_num_steps": list(batch.reference_num_steps),
        "rollout_ids": list(batch.rollout_ids),
        "atlas_probe_assignments": [dict(row) for row in selection.assignments],
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "rollout_output_path": str(Path(rollout_output_path).expanduser().resolve()),
        "dataset_binding": dataset_binding,
        "instrumentation_invariants": {
            "headless": True,
            "terrain_type": "plane",
            "render_results": False,
            "policy_enable_corruption": False,
            "tokenizer_enable_corruption": False,
            "use_encoder": "g1",
            "eval_callbacks": [],
            "run_once": True,
            "max_render_steps": max_render_steps,
            "native_adaptive_sampling": False,
        },
        "eval_entrypoint": "gear_sonic/eval_agent_trl.py",
        "hydra_overrides": list(overrides),
    }
    plan[LAUNCH_PLAN_DIGEST_FIELD] = canonical_sha256(
        plan,
        digest_field=LAUNCH_PLAN_DIGEST_FIELD,
    )
    return plan
