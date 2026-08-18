"""Frozen assignment-only probe schedules for non-atlas LACE partitions.

The primary failure atlas is fit exactly once on ``D_atlas``.  Motions in the
training and geometry partitions still need probe rollouts so they can be
assigned in those frozen coordinates, but those rollouts must never refit the
normalizer.  This module gives those measurements distinct, self-hashed
artifact kinds instead of overloading the atlas schedule contract.

Everything here is simulator independent.  Builders read and hash the frozen
motion files, while validators can deterministically rebuild both artifacts on
CPU before any rollout is launched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

from gear_sonic.research.lace.reference_lengths import (
    RESAMPLING_RULE,
    _dataset_provenance_from_split,
    _fraction_record,
    _load_materialized_variants,
    _load_motion_payload,
    _motion_path,
    _sha256_file,
    _source_contract,
    _source_file_set_sha256,
    _validate_materialized_variant,
    sonic_target_frame_count,
)
from gear_sonic.research.lace.schedule import (
    DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
    QUANTIZATION_SPEC,
    RUNTIME_RNG_SEED_DERIVATION_SPEC,
    derive_runtime_rng_seed,
    quantize_start_step,
)
from gear_sonic.research.lace.schema import (
    SPLIT_NAMES,
    canonical_sha256,
    validate_split_manifest,
)

ASSIGNMENT_INVENTORY_KIND = "lace_assignment_reference_length_inventory"
ASSIGNMENT_INVENTORY_SCHEMA_VERSION = 1
ASSIGNMENT_INVENTORY_DIGEST_FIELD = "inventory_sha256"
ASSIGNMENT_SCHEDULE_SPEC_KIND = "lace_assignment_probe_schedule_spec"
ASSIGNMENT_SCHEDULE_SPEC_SCHEMA_VERSION = 1
ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD = "spec_sha256"
ASSIGNMENT_SCHEDULE_KIND = "lace_assignment_probe_schedule"
ASSIGNMENT_SCHEDULE_SCHEMA_VERSION = 1
ASSIGNMENT_SCHEDULE_DIGEST_FIELD = "schedule_sha256"

ASSIGNMENT_POLICY_ORDER = (
    "pi_lite_early",
    "pi_lite_mid",
    "pi_lite_late",
)
ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS = (101, 202)
ASSIGNMENT_PHASE_TARGETS = (
    {"phase_id": "start", "target_fraction": 0.0},
    {"phase_id": "mid", "target_fraction": 0.5},
)
ASSIGNMENT_REPEAT_COUNT = 2
ASSIGNMENT_TARGET_FPS = 50
ASSIGNMENT_SOURCE_FPS = Fraction(30, 1)
ASSIGNMENT_RUNTIME_CONTRACT_BASE = {
    "target_fps": 50,
    "sim_fps": 50,
    "motion_fps_scale": {"numerator": 1, "denominator": 1},
    "max_len": -1,
    "reference_num_steps_equals_target_num_frames": True,
}
ASSIGNMENT_HASH_SEMANTICS = {
    "parent_normalizer_sha256": ("canonical_sha256 of the exact frozen D_atlas normalizer mapping"),
    "analysis_protocol_sha256": (
        "analysis_protocol_sha256 self-digest of the frozen atlas analysis protocol"
    ),
}

_INVENTORY_MOTION_FIELDS = {
    "motion_key",
    "split_robot_path",
    "source_path",
    "source_file_sha256",
    "source_num_frames",
    "frame_axis",
    "source_fps",
    "target_num_frames",
}
_INVENTORY_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "assignment_only",
    "fit_normalizer",
    "split_sha256",
    "split_selection_sha256",
    "partition",
    "final_open",
    "selected_motion_keys",
    "selection_complete_for_partition",
    "motion_count",
    "target_fps",
    "runtime_contract",
    "resampling_rule",
    "dataset_provenance",
    "source_file_set_sha256",
    "motions",
    ASSIGNMENT_INVENTORY_DIGEST_FIELD,
}
_SPEC_FIELDS = {
    "kind",
    "schema_version",
    "partition",
    "final_open",
    "assignment_only",
    "fit_normalizer",
    "reference_length_inventory_sha256",
    "parent_normalizer_sha256",
    "analysis_protocol_sha256",
    "probe_policies",
    "domain_randomization_seeds",
    "phase_targets",
    "repeats",
    "rollout_id_prefix",
    ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
}
_POLICY_FIELDS = {"id", "checkpoint_sha256"}
_RUNTIME_SEED_FIELDS = {
    "domain_randomization_seed",
    "phase_id",
    "repeat_index",
    "runtime_rng_seed",
}
_CARTESIAN_ORDER = [
    "motion_key",
    "probe_policy_id",
    "domain_randomization_seed",
    "phase_id",
    "repeat_index",
]
_ROLLOUT_IDENTITY_FIELDS = {
    "partition",
    "final_open",
    "assignment_only",
    "fit_normalizer",
    "split_sha256",
    "split_selection_sha256",
    "reference_length_inventory_sha256",
    "parent_normalizer_sha256",
    "analysis_protocol_sha256",
    "motion_key",
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "reference_num_steps",
    "start_step",
    "realized_fraction",
    "repeat_index",
}
_ROLLOUT_FIELDS = _ROLLOUT_IDENTITY_FIELDS | {"rollout_id"}
_SCHEDULE_FIELDS = {
    "kind",
    "schema_version",
    "scientific_use",
    "assignment_only",
    "fit_normalizer",
    "split_sha256",
    "split_selection_sha256",
    "partition",
    "final_open",
    "selected_motion_keys",
    "selection_complete_for_partition",
    "motion_count",
    "reference_length_inventory_binding",
    "schedule_spec_sha256",
    "dependency_hash_semantics",
    "parent_normalizer_sha256",
    "analysis_protocol_sha256",
    "reference_num_steps",
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
    "rollout_id_prefix",
    "cartesian_order",
    "rollout_count",
    "rollouts",
    ASSIGNMENT_SCHEDULE_DIGEST_FIELD,
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


def _partition_records(
    split_manifest: Mapping[str, Any],
    partition: str,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    _require(partition in SPLIT_NAMES, f"unknown split partition: {partition!r}")
    records = {
        str(record["motion_key"]): record
        for record in split_manifest["motions"]
        if record["partition"] == partition
    }
    selected = sorted(records)
    _require(bool(selected), f"partition {partition!r} must be non-empty")
    return selected, records


def _validate_final_open(partition: str, final_open: Any) -> bool:
    _require(isinstance(final_open, bool), "final_open must be boolean")
    if partition == "D_test":
        _require(final_open is True, "D_test is forbidden unless final_open=true")
    else:
        _require(final_open is False, "final_open must be false outside D_test")
    return bool(final_open)


def _runtime_contract(*, has_materialized_provenance: bool) -> dict[str, Any]:
    return {
        **deepcopy(ASSIGNMENT_RUNTIME_CONTRACT_BASE),
        "float32_runtime_equivalence_scope": (
            "locked_30_to_50_hz_release_cohort_with_materialized_pair_cross_check"
            if has_materialized_provenance
            else "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
        ),
    }


def _assemble_assignment_reference_length_inventory(
    split_manifest: Mapping[str, Any],
    *,
    partition: str,
    final_open: bool,
    motion_root: str | Path | None,
) -> dict[str, Any]:
    validate_split_manifest(split_manifest, verify_digest=True)
    final_open = _validate_final_open(partition, final_open)
    selected, split_records = _partition_records(split_manifest, partition)
    root = Path(motion_root).expanduser().resolve() if motion_root is not None else None
    if root is not None:
        _require(root.is_dir(), f"motion_root must be a directory: {root}")

    dataset_provenance = _dataset_provenance_from_split(split_manifest)
    materialized_variants = _load_materialized_variants(dataset_provenance)
    motion_records: list[dict[str, Any]] = []
    for motion_key in selected:
        split_robot_path, source_path = _motion_path(
            split_records[motion_key],
            motion_key,
            root,
        )
        payload = _load_motion_payload(source_path, motion_key)
        source_num_frames, source_fps = _source_contract(payload, motion_key)
        _require(
            source_fps == ASSIGNMENT_SOURCE_FPS,
            "scientific assignment inventories require every source motion to be exactly 30 Hz",
        )
        target_num_frames = sonic_target_frame_count(
            source_num_frames,
            source_fps,
            ASSIGNMENT_TARGET_FPS,
        )
        _require(target_num_frames >= 2, f"{motion_key} produces fewer than two target frames")
        source_digest = _sha256_file(source_path)
        if materialized_variants is not None:
            _require(
                motion_key in materialized_variants,
                f"materialized manifest does not contain partition motion {motion_key!r}",
            )
            _validate_materialized_variant(
                materialized_variants[motion_key],
                motion_key=motion_key,
                source_digest=source_digest,
                source_num_frames=source_num_frames,
                source_fps=source_fps,
                target_num_frames=target_num_frames,
                target_fps=ASSIGNMENT_TARGET_FPS,
            )
        motion_records.append(
            {
                "motion_key": motion_key,
                "split_robot_path": split_robot_path,
                "source_path": str(source_path),
                "source_file_sha256": source_digest,
                "source_num_frames": source_num_frames,
                "frame_axis": "root_trans_offset.shape[0]",
                "source_fps": _fraction_record(source_fps),
                "target_num_frames": target_num_frames,
            }
        )

    manifest: dict[str, Any] = {
        "kind": ASSIGNMENT_INVENTORY_KIND,
        "schema_version": ASSIGNMENT_INVENTORY_SCHEMA_VERSION,
        "scientific_use": True,
        "assignment_only": True,
        "fit_normalizer": False,
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "partition": partition,
        "final_open": final_open,
        "selected_motion_keys": selected,
        "selection_complete_for_partition": True,
        "motion_count": len(selected),
        "target_fps": ASSIGNMENT_TARGET_FPS,
        "runtime_contract": _runtime_contract(
            has_materialized_provenance=materialized_variants is not None,
        ),
        "resampling_rule": deepcopy(RESAMPLING_RULE),
        "dataset_provenance": dataset_provenance,
        "source_file_set_sha256": _source_file_set_sha256(motion_records),
        "motions": motion_records,
    }
    manifest[ASSIGNMENT_INVENTORY_DIGEST_FIELD] = canonical_sha256(
        manifest,
        digest_field=ASSIGNMENT_INVENTORY_DIGEST_FIELD,
    )
    return manifest


def build_assignment_reference_length_inventory(
    split_manifest: Mapping[str, Any],
    *,
    partition: str,
    final_open: bool = False,
    motion_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a full-partition, source-hashed 30-to-50 Hz inventory."""

    manifest = _assemble_assignment_reference_length_inventory(
        split_manifest,
        partition=partition,
        final_open=final_open,
        motion_root=motion_root,
    )
    validate_assignment_reference_length_inventory(
        manifest,
        split_manifest=split_manifest,
        verify_source_files=False,
        deterministic_rebuild=False,
    )
    return manifest


def _infer_rebuild_motion_root(
    manifest: Mapping[str, Any],
    split_records: Mapping[str, Mapping[str, Any]],
) -> Path | None:
    paths = [
        Path(str(record["source_path"])).expanduser().resolve() for record in manifest["motions"]
    ]
    split_paths = [
        Path(str(split_records[str(record["motion_key"])]["robot_path"])).expanduser().resolve()
        for record in manifest["motions"]
    ]
    if paths == split_paths:
        return None
    parents = {path.parent for path in paths}
    _require(
        len(parents) == 1,
        "deterministic rebuild requires either split paths or one explicit motion root",
    )
    return next(iter(parents))


def validate_assignment_reference_length_inventory(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    verify_digest: bool = True,
    verify_source_files: bool = False,
    deterministic_rebuild: bool = False,
) -> None:
    """Validate full coverage, resampling, source hashes, and deterministic rebuild."""

    _require(isinstance(manifest, Mapping), "assignment inventory must be a mapping")
    _require(set(manifest) == _INVENTORY_FIELDS, "assignment inventory fields are invalid")
    _require(manifest.get("kind") == ASSIGNMENT_INVENTORY_KIND, "inventory kind mismatch")
    _require(
        manifest.get("schema_version") == ASSIGNMENT_INVENTORY_SCHEMA_VERSION,
        "inventory schema version mismatch",
    )
    _require(manifest.get("scientific_use") is True, "inventory must be scientific")
    _require(manifest.get("assignment_only") is True, "assignment_only must be true")
    _require(manifest.get("fit_normalizer") is False, "fit_normalizer must be false")
    split_sha256 = _sha256(manifest.get("split_sha256"), "split_sha256")
    split_selection_sha256 = _sha256(
        manifest.get("split_selection_sha256"),
        "split_selection_sha256",
    )
    partition = manifest.get("partition")
    _require(partition in SPLIT_NAMES, "inventory partition is invalid")
    final_open = _validate_final_open(str(partition), manifest.get("final_open"))

    selected_raw = manifest.get("selected_motion_keys")
    _require(
        isinstance(selected_raw, Sequence) and not isinstance(selected_raw, (str, bytes)),
        "selected_motion_keys must be a sequence",
    )
    selected = list(selected_raw)
    _require(
        bool(selected)
        and all(isinstance(key, str) and key for key in selected)
        and selected == sorted(set(selected)),
        "selected_motion_keys must be non-empty, unique, and canonically sorted",
    )
    _require(
        manifest.get("selection_complete_for_partition") is True,
        "selection_complete_for_partition must be true",
    )
    _require(manifest.get("motion_count") == len(selected), "motion_count mismatch")
    _require(manifest.get("target_fps") == ASSIGNMENT_TARGET_FPS, "target_fps must be 50")

    dataset_provenance = manifest.get("dataset_provenance")
    _require(isinstance(dataset_provenance, Mapping), "dataset_provenance must be a mapping")
    provenance_fields = {
        "materialized_manifest",
        "materialized_manifest_sha256",
        "paired_dataset_sha256",
    }
    _require(
        set(dataset_provenance) == provenance_fields,
        "dataset_provenance fields are invalid",
    )
    provenance_values = [dataset_provenance[field] for field in sorted(provenance_fields)]
    _require(
        all(value is None for value in provenance_values)
        or all(value is not None for value in provenance_values),
        "dataset_provenance must be entirely bound or entirely unavailable",
    )
    for field in ("materialized_manifest_sha256", "paired_dataset_sha256"):
        if dataset_provenance[field] is not None:
            _sha256(dataset_provenance[field], f"dataset_provenance.{field}")
    has_materialized_provenance = dataset_provenance["materialized_manifest"] is not None
    _require(
        manifest.get("runtime_contract")
        == _runtime_contract(has_materialized_provenance=has_materialized_provenance),
        "runtime_contract must freeze 30-to-50 Hz, scale=1, and no crop",
    )
    _require(manifest.get("resampling_rule") == RESAMPLING_RULE, "resampling_rule mismatch")

    motions = manifest.get("motions")
    _require(isinstance(motions, list), "motions must be a list")
    _require(len(motions) == len(selected), "motions must cover the partition exactly")
    materialized_variants = (
        _load_materialized_variants(dataset_provenance) if verify_source_files else None
    )
    normalized_motions: list[Mapping[str, Any]] = []
    for index, record in enumerate(motions):
        _require(isinstance(record, Mapping), f"motions[{index}] must be a mapping")
        _require(set(record) == _INVENTORY_MOTION_FIELDS, f"motions[{index}] fields are invalid")
        motion_key = record.get("motion_key")
        _require(motion_key == selected[index], "motions must follow selected_motion_keys order")
        _require(
            isinstance(record.get("split_robot_path"), str) and bool(record["split_robot_path"]),
            f"motions[{index}].split_robot_path must be non-empty",
        )
        source_path = record.get("source_path")
        _require(
            isinstance(source_path, str) and Path(source_path).is_absolute(),
            f"motions[{index}].source_path must be absolute",
        )
        _require(
            Path(source_path).name == f"{motion_key}.pkl",
            f"motions[{index}].source_path basename mismatch",
        )
        source_digest = _sha256(
            record.get("source_file_sha256"),
            f"motions[{index}].source_file_sha256",
        )
        source_num_frames = record.get("source_num_frames")
        target_num_frames = record.get("target_num_frames")
        _require(
            isinstance(source_num_frames, int)
            and not isinstance(source_num_frames, bool)
            and source_num_frames >= 2,
            f"motions[{index}].source_num_frames must be an integer >= 2",
        )
        _require(
            isinstance(target_num_frames, int)
            and not isinstance(target_num_frames, bool)
            and target_num_frames >= 2,
            f"motions[{index}].target_num_frames must be an integer >= 2",
        )
        _require(
            record.get("frame_axis") == "root_trans_offset.shape[0]",
            f"motions[{index}].frame_axis mismatch",
        )
        _require(
            record.get("source_fps") == {"numerator": 30, "denominator": 1},
            f"motions[{index}].source_fps must be exactly 30 Hz",
        )
        expected_target_frames = sonic_target_frame_count(
            source_num_frames,
            ASSIGNMENT_SOURCE_FPS,
            ASSIGNMENT_TARGET_FPS,
        )
        _require(
            target_num_frames == expected_target_frames,
            f"motions[{index}].target_num_frames violates exact 30-to-50 Hz resampling",
        )
        if verify_source_files:
            path = Path(source_path)
            _require(path.is_file(), f"motions[{index}] source file is missing: {path}")
            _require(
                _sha256_file(path) == source_digest,
                f"motions[{index}] source file SHA-256 mismatch",
            )
            payload = _load_motion_payload(path, str(motion_key))
            live_frames, live_fps = _source_contract(payload, str(motion_key))
            _require(
                live_frames == source_num_frames and live_fps == ASSIGNMENT_SOURCE_FPS,
                f"motions[{index}] live source timeline drifted",
            )
            if materialized_variants is not None:
                _require(
                    motion_key in materialized_variants,
                    f"materialized manifest does not contain {motion_key!r}",
                )
                _validate_materialized_variant(
                    materialized_variants[str(motion_key)],
                    motion_key=str(motion_key),
                    source_digest=source_digest,
                    source_num_frames=source_num_frames,
                    source_fps=ASSIGNMENT_SOURCE_FPS,
                    target_num_frames=target_num_frames,
                    target_fps=ASSIGNMENT_TARGET_FPS,
                )
        normalized_motions.append(record)
    _require(
        manifest.get("source_file_set_sha256") == _source_file_set_sha256(normalized_motions),
        "source_file_set_sha256 mismatch",
    )

    split_records: dict[str, Mapping[str, Any]] | None = None
    if split_manifest is not None:
        validate_split_manifest(split_manifest, verify_digest=True)
        _require(split_sha256 == split_manifest["split_sha256"], "split_sha256 binding mismatch")
        _require(
            split_selection_sha256 == split_manifest["selection_sha256"],
            "split_selection_sha256 binding mismatch",
        )
        expected_selected, split_records = _partition_records(split_manifest, str(partition))
        _require(
            selected == expected_selected,
            "assignment inventory must exactly cover its full split partition",
        )
        _require(
            dict(dataset_provenance) == _dataset_provenance_from_split(split_manifest),
            "dataset_provenance does not match the frozen split",
        )
        for index, record in enumerate(motions):
            _require(
                record["split_robot_path"]
                == split_records[str(record["motion_key"])]["robot_path"],
                f"motions[{index}].split_robot_path does not match the split",
            )

    if verify_digest:
        expected_digest = _sha256(
            manifest.get(ASSIGNMENT_INVENTORY_DIGEST_FIELD),
            ASSIGNMENT_INVENTORY_DIGEST_FIELD,
        )
        _require(
            expected_digest
            == canonical_sha256(manifest, digest_field=ASSIGNMENT_INVENTORY_DIGEST_FIELD),
            f"{ASSIGNMENT_INVENTORY_DIGEST_FIELD} mismatch",
        )

    if deterministic_rebuild:
        _require(split_manifest is not None, "deterministic rebuild requires split_manifest")
        _require(verify_source_files, "deterministic rebuild requires verify_source_files=true")
        assert split_records is not None
        motion_root = _infer_rebuild_motion_root(manifest, split_records)
        rebuilt = _assemble_assignment_reference_length_inventory(
            split_manifest,
            partition=str(partition),
            final_open=final_open,
            motion_root=motion_root,
        )
        _require(
            dict(manifest) == rebuilt,
            "assignment inventory differs from deterministic source-file rebuild",
        )


def _normalize_probe_policies(raw: Any) -> list[dict[str, str]]:
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
        "probe_policies must be a sequence",
    )
    policies = list(raw)
    _require(len(policies) == len(ASSIGNMENT_POLICY_ORDER), "exactly three probe policies required")
    normalized: list[dict[str, str]] = []
    for index, (record, expected_id) in enumerate(
        zip(policies, ASSIGNMENT_POLICY_ORDER, strict=True)
    ):
        _require(isinstance(record, Mapping), f"probe_policies[{index}] must be a mapping")
        _require(set(record) == _POLICY_FIELDS, f"probe_policies[{index}] fields are invalid")
        _require(
            record.get("id") == expected_id, f"probe policy order must be {ASSIGNMENT_POLICY_ORDER}"
        )
        normalized.append(
            {
                "id": expected_id,
                "checkpoint_sha256": _sha256(
                    record.get("checkpoint_sha256"),
                    f"probe_policies[{index}].checkpoint_sha256",
                ),
            }
        )
    return normalized


def validate_assignment_schedule_spec(
    spec: Mapping[str, Any],
    *,
    verify_digest: bool = True,
) -> None:
    """Validate the pre-outcome assignment grid and dependency hashes."""

    _require(isinstance(spec, Mapping), "assignment schedule spec must be a mapping")
    _require(set(spec) == _SPEC_FIELDS, "assignment schedule spec fields are invalid")
    _require(spec.get("kind") == ASSIGNMENT_SCHEDULE_SPEC_KIND, "schedule spec kind mismatch")
    _require(
        spec.get("schema_version") == ASSIGNMENT_SCHEDULE_SPEC_SCHEMA_VERSION,
        "schedule spec schema version mismatch",
    )
    partition = spec.get("partition")
    _require(partition in SPLIT_NAMES, "schedule spec partition is invalid")
    _validate_final_open(str(partition), spec.get("final_open"))
    _require(spec.get("assignment_only") is True, "assignment_only must be true")
    _require(spec.get("fit_normalizer") is False, "fit_normalizer must be false")
    for field in (
        "reference_length_inventory_sha256",
        "parent_normalizer_sha256",
        "analysis_protocol_sha256",
    ):
        _sha256(spec.get(field), field)
    _normalize_probe_policies(spec.get("probe_policies"))
    _require(
        spec.get("domain_randomization_seeds") == list(ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS),
        f"domain_randomization_seeds must be {ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS}",
    )
    phases = spec.get("phase_targets")
    _require(
        isinstance(phases, list)
        and all(isinstance(phase, Mapping) for phase in phases)
        and [dict(phase) for phase in phases] == list(ASSIGNMENT_PHASE_TARGETS),
        f"phase_targets must be {ASSIGNMENT_PHASE_TARGETS}",
    )
    _require(spec.get("repeats") == ASSIGNMENT_REPEAT_COUNT, "repeats must be exactly 2")
    prefix = spec.get("rollout_id_prefix")
    _require(isinstance(prefix, str) and bool(prefix), "rollout_id_prefix must be non-empty")
    _require(":" not in prefix, "rollout_id_prefix must not contain ':'")
    if verify_digest:
        expected = _sha256(
            spec.get(ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD),
            ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
        )
        _require(
            expected == canonical_sha256(spec, digest_field=ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD),
            f"{ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD} mismatch",
        )


def _runtime_seed_schedule() -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    observed: set[int] = set()
    for seed in ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS:
        for phase in ASSIGNMENT_PHASE_TARGETS:
            for repeat_index in range(ASSIGNMENT_REPEAT_COUNT):
                runtime_seed = derive_runtime_rng_seed(seed, phase["phase_id"], repeat_index)
                _require(runtime_seed not in observed, "runtime RNG seed collision")
                observed.add(runtime_seed)
                records.append(
                    {
                        "domain_randomization_seed": seed,
                        "phase_id": phase["phase_id"],
                        "repeat_index": repeat_index,
                        "runtime_rng_seed": runtime_seed,
                    }
                )
    return records


def _inventory_binding(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": inventory["kind"],
        "schema_version": inventory["schema_version"],
        ASSIGNMENT_INVENTORY_DIGEST_FIELD: inventory[ASSIGNMENT_INVENTORY_DIGEST_FIELD],
        "scientific_use": inventory["scientific_use"],
        "assignment_only": inventory["assignment_only"],
        "fit_normalizer": inventory["fit_normalizer"],
        "split_sha256": inventory["split_sha256"],
        "split_selection_sha256": inventory["split_selection_sha256"],
        "partition": inventory["partition"],
        "final_open": inventory["final_open"],
        "selected_motion_keys": list(inventory["selected_motion_keys"]),
        "selection_complete_for_partition": inventory["selection_complete_for_partition"],
        "source_file_set_sha256": inventory["source_file_set_sha256"],
        "target_fps": inventory["target_fps"],
        "runtime_contract": deepcopy(inventory["runtime_contract"]),
        "dataset_provenance": deepcopy(inventory["dataset_provenance"]),
    }


def _rollout_identity(
    *,
    inventory: Mapping[str, Any],
    spec: Mapping[str, Any],
    motion_key: str,
    policy: Mapping[str, str],
    seed: int,
    runtime_rng_seed: int,
    phase_id: str,
    target_fraction: float,
    reference_num_steps: int,
    start_step: int,
    realized_fraction: float,
    repeat_index: int,
) -> dict[str, Any]:
    return {
        "identity_version": 1,
        "partition": inventory["partition"],
        "final_open": inventory["final_open"],
        "assignment_only": True,
        "fit_normalizer": False,
        "split_sha256": inventory["split_sha256"],
        "split_selection_sha256": inventory["split_selection_sha256"],
        "reference_length_inventory_sha256": inventory[ASSIGNMENT_INVENTORY_DIGEST_FIELD],
        "parent_normalizer_sha256": spec["parent_normalizer_sha256"],
        "analysis_protocol_sha256": spec["analysis_protocol_sha256"],
        "motion_key": motion_key,
        "probe_policy_id": policy["id"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "domain_randomization_seed": seed,
        "runtime_rng_seed": runtime_rng_seed,
        "phase_id": phase_id,
        "target_fraction": target_fraction,
        "reference_num_steps": reference_num_steps,
        "start_step": start_step,
        "realized_fraction": realized_fraction,
        "repeat_index": repeat_index,
    }


def _assemble_assignment_probe_schedule(
    split_manifest: Mapping[str, Any],
    *,
    reference_length_inventory: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    validate_split_manifest(split_manifest, verify_digest=True)
    validate_assignment_reference_length_inventory(
        reference_length_inventory,
        split_manifest=split_manifest,
        verify_digest=True,
        verify_source_files=False,
        deterministic_rebuild=False,
    )
    validate_assignment_schedule_spec(spec, verify_digest=True)
    _require(
        spec["partition"] == reference_length_inventory["partition"]
        and spec["final_open"] == reference_length_inventory["final_open"],
        "schedule spec partition/final_open does not match its inventory",
    )
    _require(
        spec["reference_length_inventory_sha256"]
        == reference_length_inventory[ASSIGNMENT_INVENTORY_DIGEST_FIELD],
        "schedule spec does not bind the supplied reference-length inventory",
    )

    policies = _normalize_probe_policies(spec["probe_policies"])
    selected = list(reference_length_inventory["selected_motion_keys"])
    steps_by_motion = {
        str(record["motion_key"]): int(record["target_num_frames"])
        for record in reference_length_inventory["motions"]
    }
    runtime_seed_schedule = _runtime_seed_schedule()
    runtime_seed_by_condition = {
        (
            int(record["domain_randomization_seed"]),
            str(record["phase_id"]),
            int(record["repeat_index"]),
        ): int(record["runtime_rng_seed"])
        for record in runtime_seed_schedule
    }

    rollouts: list[dict[str, Any]] = []
    for motion_key in selected:
        reference_num_steps = steps_by_motion[motion_key]
        for policy in policies:
            for seed in ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS:
                for phase in ASSIGNMENT_PHASE_TARGETS:
                    phase_id = str(phase["phase_id"])
                    target_fraction = float(phase["target_fraction"])
                    start_step = quantize_start_step(target_fraction, reference_num_steps)
                    realized_fraction = start_step / (reference_num_steps - 1)
                    for repeat_index in range(ASSIGNMENT_REPEAT_COUNT):
                        runtime_seed = runtime_seed_by_condition[(seed, phase_id, repeat_index)]
                        identity = _rollout_identity(
                            inventory=reference_length_inventory,
                            spec=spec,
                            motion_key=motion_key,
                            policy=policy,
                            seed=seed,
                            runtime_rng_seed=runtime_seed,
                            phase_id=phase_id,
                            target_fraction=target_fraction,
                            reference_num_steps=reference_num_steps,
                            start_step=start_step,
                            realized_fraction=realized_fraction,
                            repeat_index=repeat_index,
                        )
                        rollouts.append(
                            {
                                "rollout_id": (
                                    f"{spec['rollout_id_prefix']}:" f"{canonical_sha256(identity)}"
                                ),
                                **{
                                    field: value
                                    for field, value in identity.items()
                                    if field != "identity_version"
                                },
                            }
                        )

    manifest: dict[str, Any] = {
        "kind": ASSIGNMENT_SCHEDULE_KIND,
        "schema_version": ASSIGNMENT_SCHEDULE_SCHEMA_VERSION,
        "scientific_use": True,
        "assignment_only": True,
        "fit_normalizer": False,
        "split_sha256": reference_length_inventory["split_sha256"],
        "split_selection_sha256": reference_length_inventory["split_selection_sha256"],
        "partition": reference_length_inventory["partition"],
        "final_open": reference_length_inventory["final_open"],
        "selected_motion_keys": selected,
        "selection_complete_for_partition": True,
        "motion_count": len(selected),
        "reference_length_inventory_binding": _inventory_binding(reference_length_inventory),
        "schedule_spec_sha256": spec[ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD],
        "dependency_hash_semantics": deepcopy(ASSIGNMENT_HASH_SEMANTICS),
        "parent_normalizer_sha256": spec["parent_normalizer_sha256"],
        "analysis_protocol_sha256": spec["analysis_protocol_sha256"],
        "reference_num_steps": [
            {
                "motion_key": motion_key,
                "reference_num_steps": steps_by_motion[motion_key],
            }
            for motion_key in selected
        ],
        "probe_policies": policies,
        "policy_order": list(ASSIGNMENT_POLICY_ORDER),
        "domain_randomization_seeds": list(ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS),
        "domain_randomization_seed_semantics": DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "phase_targets": [dict(phase) for phase in ASSIGNMENT_PHASE_TARGETS],
        "repeat_count": ASSIGNMENT_REPEAT_COUNT,
        "repeat_indices": list(range(ASSIGNMENT_REPEAT_COUNT)),
        "runtime_rng_seed_derivation": deepcopy(RUNTIME_RNG_SEED_DERIVATION_SPEC),
        "runtime_rng_seed_schedule": runtime_seed_schedule,
        "quantization": deepcopy(QUANTIZATION_SPEC),
        "rollout_id_prefix": spec["rollout_id_prefix"],
        "cartesian_order": list(_CARTESIAN_ORDER),
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    manifest[ASSIGNMENT_SCHEDULE_DIGEST_FIELD] = canonical_sha256(
        manifest,
        digest_field=ASSIGNMENT_SCHEDULE_DIGEST_FIELD,
    )
    return manifest


def build_assignment_probe_schedule(
    split_manifest: Mapping[str, Any],
    *,
    reference_length_inventory: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact full-partition assignment schedule declared by ``spec``."""

    manifest = _assemble_assignment_probe_schedule(
        split_manifest,
        reference_length_inventory=reference_length_inventory,
        spec=spec,
    )
    validate_assignment_probe_schedule(
        manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        spec=spec,
    )
    return manifest


def validate_assignment_probe_schedule(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    spec: Mapping[str, Any],
    verify_digest: bool = True,
) -> None:
    """Deep-validate a schedule by exact deterministic reconstruction."""

    _require(isinstance(manifest, Mapping), "assignment schedule must be a mapping")
    _require(set(manifest) == _SCHEDULE_FIELDS, "assignment schedule fields are invalid")
    _require(manifest.get("kind") == ASSIGNMENT_SCHEDULE_KIND, "schedule kind mismatch")
    _require(
        manifest.get("schema_version") == ASSIGNMENT_SCHEDULE_SCHEMA_VERSION,
        "schedule schema version mismatch",
    )
    _require(manifest.get("scientific_use") is True, "schedule must be scientific")
    _require(manifest.get("assignment_only") is True, "assignment_only must be true")
    _require(manifest.get("fit_normalizer") is False, "fit_normalizer must be false")
    partition = manifest.get("partition")
    _require(partition in SPLIT_NAMES, "schedule partition is invalid")
    _validate_final_open(str(partition), manifest.get("final_open"))
    _sha256(manifest.get("parent_normalizer_sha256"), "parent_normalizer_sha256")
    _sha256(manifest.get("analysis_protocol_sha256"), "analysis_protocol_sha256")
    rollouts = manifest.get("rollouts")
    _require(isinstance(rollouts, list), "rollouts must be a list")
    for index, row in enumerate(rollouts):
        _require(isinstance(row, Mapping), f"rollouts[{index}] must be a mapping")
        _require(set(row) == _ROLLOUT_FIELDS, f"rollouts[{index}] fields are invalid")
    runtime_seeds = manifest.get("runtime_rng_seed_schedule")
    _require(isinstance(runtime_seeds, list), "runtime_rng_seed_schedule must be a list")
    _require(
        all(isinstance(row, Mapping) and set(row) == _RUNTIME_SEED_FIELDS for row in runtime_seeds),
        "runtime_rng_seed_schedule fields are invalid",
    )
    if verify_digest:
        expected_digest = _sha256(
            manifest.get(ASSIGNMENT_SCHEDULE_DIGEST_FIELD),
            ASSIGNMENT_SCHEDULE_DIGEST_FIELD,
        )
        _require(
            expected_digest
            == canonical_sha256(manifest, digest_field=ASSIGNMENT_SCHEDULE_DIGEST_FIELD),
            f"{ASSIGNMENT_SCHEDULE_DIGEST_FIELD} mismatch",
        )
    rebuilt = _assemble_assignment_probe_schedule(
        split_manifest,
        reference_length_inventory=reference_length_inventory,
        spec=spec,
    )
    _require(
        dict(manifest) == rebuilt,
        "assignment schedule differs from deterministic inventory/spec reconstruction",
    )


def deep_validate_assignment_artifacts(
    *,
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    spec: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> None:
    """Rehash source bytes and reconstruct the complete assignment artifact pair."""

    validate_assignment_reference_length_inventory(
        reference_length_inventory,
        split_manifest=split_manifest,
        verify_digest=True,
        verify_source_files=True,
        deterministic_rebuild=True,
    )
    validate_assignment_probe_schedule(
        schedule,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        spec=spec,
        verify_digest=True,
    )
