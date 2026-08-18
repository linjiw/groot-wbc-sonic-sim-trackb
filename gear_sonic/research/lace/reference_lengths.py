"""Frozen SONIC reference-length inventories for LACE rollout schedules.

This module freezes SONIC's G1 reference timeline contract without importing
Torch or Isaac Lab.  Exact rational arithmetic is runtime-equivalent for the
locked 30-to-50 Hz release cohort (also cross-checked against its materialized
50 Hz pairs).  Generic pilot rates remain useful for contract tests, but make
no claim of bit-for-bit equivalence with float32 ``torch.arange`` endpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from gear_sonic.research.lace.schema import canonical_sha256, validate_split_manifest

REFERENCE_LENGTH_KIND = "lace_reference_length_inventory"
REFERENCE_LENGTH_SCHEMA_VERSION = 1
REFERENCE_LENGTH_DIGEST_FIELD = "inventory_sha256"
REFERENCE_LENGTH_MODES = ("scientific", "pilot")
DEFAULT_TARGET_FPS = 50
DEFAULT_SIM_FPS = 50
DEFAULT_MOTION_FPS_SCALE = 1
DEFAULT_MAX_LEN = -1
SCIENTIFIC_SOURCE_FPS = 30
RESAMPLING_RULE_ID = "sonic_exclusive_end_exact_rational_v1"
RESAMPLING_RULE = {
    "id": RESAMPLING_RULE_ID,
    "runtime_source": (
        "gear_sonic.utils.motion_lib.torch_humanoid_batch:" "TorchHumanoidBatch.interploate_pose"
    ),
    "equal_fps": "return the raw source frame count because interpolation is skipped",
    "different_fps": "len(arange(0, (n - 1) / source_fps, 1 / target_fps))",
    "exact_count": "ceil((n - 1) * target_fps / source_fps)",
    "endpoint": "exclusive",
    "arithmetic": "reduced integer rationals; no binary floating-point rounding",
}

_MOTION_RECORD_FIELDS = {
    "motion_key",
    "split_robot_path",
    "source_path",
    "source_file_sha256",
    "source_num_frames",
    "frame_axis",
    "source_fps",
    "target_num_frames",
}
_DATASET_PROVENANCE_FIELDS = {
    "materialized_manifest",
    "materialized_manifest_sha256",
    "paired_dataset_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash motion file {path}: {error}") from error
    return digest.hexdigest()


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}: {error}") from error
    _require(isinstance(payload, Mapping), f"{name} must contain a JSON mapping")
    return payload


def _dataset_provenance_from_split(split_manifest: Mapping[str, Any]) -> dict[str, str | None]:
    dataset = split_manifest.get("dataset")
    _require(isinstance(dataset, Mapping), "split dataset provenance must be a mapping")
    values = {
        "materialized_manifest": dataset.get("materialized_manifest"),
        "materialized_manifest_sha256": dataset.get("materialized_manifest_sha256"),
        "paired_dataset_sha256": dataset.get("paired_dataset_sha256"),
    }
    present = {name: value is not None for name, value in values.items()}
    _require(
        len(set(present.values())) == 1,
        "split materialized-manifest provenance must provide path, file digest, and paired digest together",
    )
    if not any(present.values()):
        return {name: None for name in sorted(_DATASET_PROVENANCE_FIELDS)}
    _require(
        isinstance(values["materialized_manifest"], str) and bool(values["materialized_manifest"]),
        "split dataset.materialized_manifest must be a non-empty path",
    )
    return {
        "materialized_manifest": str(values["materialized_manifest"]),
        "materialized_manifest_sha256": _require_sha256(
            values["materialized_manifest_sha256"],
            "split dataset.materialized_manifest_sha256",
        ),
        "paired_dataset_sha256": _require_sha256(
            values["paired_dataset_sha256"],
            "split dataset.paired_dataset_sha256",
        ),
    }


def _load_materialized_variants(
    provenance: Mapping[str, str | None],
) -> dict[str, Mapping[str, Any]] | None:
    path_value = provenance["materialized_manifest"]
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    _require(path.is_file(), f"materialized manifest does not exist: {path}")
    expected_file_digest = provenance["materialized_manifest_sha256"]
    _require(
        _sha256_file(path) == expected_file_digest,
        "materialized manifest file SHA-256 does not match split provenance",
    )
    manifest = _load_json_mapping(path, "materialized manifest")
    output = manifest.get("output")
    _require(isinstance(output, Mapping), "materialized manifest output must be a mapping")
    _require(
        output.get("paired_dataset_sha256") == provenance["paired_dataset_sha256"],
        "materialized manifest paired dataset digest does not match split provenance",
    )
    variants = output.get("variants")
    _require(isinstance(variants, list), "materialized manifest output.variants must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, variant in enumerate(variants):
        _require(
            isinstance(variant, Mapping),
            f"materialized manifest output.variants[{index}] must be a mapping",
        )
        motion_key = variant.get("motion_key")
        _require(
            isinstance(motion_key, str) and motion_key,
            f"materialized manifest output.variants[{index}].motion_key must be non-empty",
        )
        _require(motion_key not in indexed, f"duplicate materialized motion key: {motion_key}")
        indexed[motion_key] = variant
    return indexed


def _positive_fraction(value: Any, name: str) -> Fraction:
    _require(not isinstance(value, bool), f"{name} must be numeric, not boolean")
    if isinstance(value, Integral):
        result = Fraction(int(value), 1)
    elif isinstance(value, Fraction):
        result = value
    elif isinstance(value, Real):
        numeric = float(value)
        _require(math.isfinite(numeric), f"{name} must be finite")
        result = Fraction(str(numeric))
    else:
        raise ValueError(f"{name} must be a finite positive numeric value")
    _require(result > 0, f"{name} must be positive")
    return result


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_record(value: Any, name: str) -> Fraction:
    _require(isinstance(value, Mapping), f"{name} must be a rational mapping")
    _require(set(value) == {"numerator", "denominator"}, f"{name} fields are invalid")
    numerator = value["numerator"]
    denominator = value["denominator"]
    _require(
        isinstance(numerator, int) and not isinstance(numerator, bool),
        f"{name}.numerator must be an integer",
    )
    _require(
        isinstance(denominator, int) and not isinstance(denominator, bool) and denominator > 0,
        f"{name}.denominator must be a positive integer",
    )
    result = Fraction(numerator, denominator)
    _require(result > 0, f"{name} must be positive")
    _require(
        (result.numerator, result.denominator) == (numerator, denominator),
        f"{name} must be reduced to lowest terms",
    )
    return result


def sonic_target_frame_count(
    source_num_frames: int,
    source_fps: Any,
    target_fps: int = DEFAULT_TARGET_FPS,
) -> int:
    """Return SONIC's exclusive-end target length using exact arithmetic.

    When frame rates differ, ``torch.arange(0, duration, 1 / target_fps)``
    mathematically has ``ceil(duration * target_fps)`` samples.  Integer
    rational arithmetic makes the exclusive endpoint deterministic and avoids
    occasional extra samples caused by float32 endpoint rounding.
    """

    _require(
        isinstance(source_num_frames, int)
        and not isinstance(source_num_frames, bool)
        and source_num_frames >= 1,
        "source_num_frames must be an integer >= 1",
    )
    _require(
        isinstance(target_fps, int) and not isinstance(target_fps, bool) and target_fps > 0,
        "target_fps must be a positive integer",
    )
    source_rate = _positive_fraction(source_fps, "source_fps")
    target_rate = Fraction(target_fps, 1)
    if source_rate == target_rate:
        return source_num_frames
    ratio = Fraction(source_num_frames - 1, 1) * target_rate / source_rate
    return (ratio.numerator + ratio.denominator - 1) // ratio.denominator


def _atlas_selection(
    split_manifest: Mapping[str, Any],
    *,
    artifact_mode: str,
    selected_motion_keys: Sequence[str] | None,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    _require(
        artifact_mode in REFERENCE_LENGTH_MODES,
        f"artifact_mode must be one of {REFERENCE_LENGTH_MODES}",
    )
    atlas_records = {
        str(record["motion_key"]): record
        for record in split_manifest["motions"]
        if record["partition"] == "D_atlas"
    }
    atlas_keys = sorted(atlas_records)
    if selected_motion_keys is None:
        _require(
            artifact_mode == "scientific",
            "pilot mode requires an explicit selected_motion_keys subset",
        )
        selected = atlas_keys
    else:
        _require(
            isinstance(selected_motion_keys, Sequence)
            and not isinstance(selected_motion_keys, (str, bytes)),
            "selected_motion_keys must be a sequence",
        )
        values = list(selected_motion_keys)
        _require(bool(values), "selected_motion_keys must be non-empty")
        _require(
            all(isinstance(key, str) and key for key in values),
            "selected_motion_keys must contain non-empty strings",
        )
        _require(len(values) == len(set(values)), "selected_motion_keys must be unique")
        selected = sorted(values)
    _require(set(selected) <= set(atlas_keys), "selected motions must be drawn only from D_atlas")
    if artifact_mode == "scientific":
        _require(selected == atlas_keys, "scientific inventory must cover the full D_atlas")
    else:
        _require(set(selected) < set(atlas_keys), "pilot selection must be a strict D_atlas subset")
    return selected, atlas_records


def _motion_path(
    record: Mapping[str, Any],
    motion_key: str,
    motion_root: Path | None,
) -> tuple[str, Path]:
    split_path = record.get("robot_path")
    _require(
        isinstance(split_path, str) and split_path,
        f"split record {motion_key!r} must contain a non-empty robot_path",
    )
    if motion_root is None:
        path = Path(split_path).expanduser()
    else:
        path = motion_root / f"{motion_key}.pkl"
    path = path.resolve()
    _require(
        path.name == f"{motion_key}.pkl",
        f"motion file name must exactly match its key: {path.name!r} != {motion_key!r}.pkl",
    )
    _require(path.is_file(), f"motion file does not exist: {path}")
    return split_path, path


def _load_motion_payload(path: Path, motion_key: str) -> Mapping[str, Any]:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - dependency is part of gear_sonic core
        raise ValueError("joblib is required to build a reference-length inventory") from error
    try:
        container = joblib.load(path)
    except Exception as error:
        raise ValueError(f"cannot load motion file {path}: {error}") from error
    _require(isinstance(container, Mapping), f"{path} must contain a keyed mapping")
    _require(
        list(container) == [motion_key],
        f"{path} must contain exactly one key matching {motion_key!r}",
    )
    payload = container[motion_key]
    _require(isinstance(payload, Mapping), f"{path}[{motion_key!r}] must be a mapping")
    return payload


def _shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    _require(shape is not None, f"{name} must expose an array shape")
    try:
        result = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has an invalid shape: {shape!r}") from error
    return result


def _require_finite_array(value: Any, name: str) -> None:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - dependency is part of gear_sonic core
        raise ValueError("numpy is required to validate motion arrays") from error
    try:
        finite = bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    _require(finite, f"{name} must contain only finite values")


def _source_contract(payload: Mapping[str, Any], motion_key: str) -> tuple[int, Fraction]:
    _require("fps" in payload, f"{motion_key}.fps must be explicit")
    source_fps = _positive_fraction(payload["fps"], f"{motion_key}.fps")
    root_shape = _shape(payload.get("root_trans_offset"), f"{motion_key}.root_trans_offset")
    _require(
        len(root_shape) == 2 and root_shape[1] == 3 and root_shape[0] >= 2,
        f"{motion_key}.root_trans_offset must have shape (frames >= 2, 3)",
    )
    pose_shape = _shape(payload.get("pose_aa"), f"{motion_key}.pose_aa")
    _require(
        pose_shape[1:] == (30, 3),
        f"{motion_key}.pose_aa must have shape (frames, 30, 3)",
    )
    _require(
        pose_shape[0] == root_shape[0],
        f"{motion_key}.pose_aa frame axis does not match root_trans_offset",
    )
    for field, expected_tail in (
        ("dof", (29,)),
        ("root_rot", (4,)),
        ("smpl_joints", (24, 3)),
    ):
        field_shape = _shape(payload.get(field), f"{motion_key}.{field}")
        _require(
            field_shape[1:] == expected_tail,
            f"{motion_key}.{field} must have shape (frames, {expected_tail})",
        )
        _require(
            field_shape[0] == root_shape[0],
            f"{motion_key}.{field} frame axis does not match root_trans_offset",
        )
    for field in (
        "root_trans_offset",
        "pose_aa",
        "dof",
        "root_rot",
        "smpl_joints",
    ):
        _require_finite_array(payload[field], f"{motion_key}.{field}")
    if "length" in payload:
        length = payload["length"]
        _require(
            isinstance(length, Integral)
            and not isinstance(length, bool)
            and int(length) == root_shape[0],
            f"{motion_key}.length does not match the validated frame axis",
        )
    return root_shape[0], source_fps


def _validate_materialized_variant(
    variant: Mapping[str, Any],
    *,
    motion_key: str,
    source_digest: str,
    source_num_frames: int,
    source_fps: Fraction,
    target_num_frames: int,
    target_fps: int,
) -> None:
    robot = variant.get("robot")
    smpl = variant.get("smpl")
    _require(isinstance(robot, Mapping), f"materialized {motion_key}.robot must be a mapping")
    _require(isinstance(smpl, Mapping), f"materialized {motion_key}.smpl must be a mapping")
    _require(
        robot.get("sha256") == source_digest,
        f"materialized {motion_key}.robot SHA-256 does not match source bytes",
    )
    _require(
        robot.get("frames") == source_num_frames,
        f"materialized {motion_key}.robot frame count mismatch",
    )
    _require(
        _positive_fraction(robot.get("fps"), f"materialized {motion_key}.robot.fps") == source_fps,
        f"materialized {motion_key}.robot FPS mismatch",
    )
    _require(
        smpl.get("frames") == target_num_frames,
        f"materialized {motion_key}.smpl frame count disagrees with exact resampling",
    )
    _require(
        _positive_fraction(smpl.get("fps"), f"materialized {motion_key}.smpl.fps") == target_fps,
        f"materialized {motion_key}.smpl FPS mismatch",
    )
    _require(
        smpl.get("original_frames") == source_num_frames,
        f"materialized {motion_key}.smpl original frame count mismatch",
    )
    _require(
        _positive_fraction(
            smpl.get("original_fps"),
            f"materialized {motion_key}.smpl.original_fps",
        )
        == source_fps,
        f"materialized {motion_key}.smpl original FPS mismatch",
    )


def _source_file_set_sha256(motions: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "files": [
                {
                    "motion_key": record["motion_key"],
                    "source_file_sha256": record["source_file_sha256"],
                }
                for record in motions
            ]
        }
    )


def build_reference_length_inventory(
    split_manifest: Mapping[str, Any],
    *,
    target_fps: int = DEFAULT_TARGET_FPS,
    sim_fps: int | None = None,
    motion_fps_scale: Any = DEFAULT_MOTION_FPS_SCALE,
    max_len: int = DEFAULT_MAX_LEN,
    artifact_mode: str = "scientific",
    selected_motion_keys: Sequence[str] | None = None,
    motion_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a frozen length inventory from split-bound G1 joblib files."""

    validate_split_manifest(split_manifest, verify_digest=True)
    _require(
        isinstance(target_fps, int) and not isinstance(target_fps, bool) and target_fps > 0,
        "target_fps must be a positive integer",
    )
    if sim_fps is None:
        sim_fps = target_fps
    _require(
        isinstance(sim_fps, int) and not isinstance(sim_fps, bool) and sim_fps > 0,
        "sim_fps must be a positive integer",
    )
    scale = _positive_fraction(motion_fps_scale, "motion_fps_scale")
    _require(scale == 1, "motion_fps_scale must be exactly 1")
    _require(max_len == -1, "max_len must be exactly -1 so no pre-resampling crop occurs")
    _require(
        sim_fps == target_fps,
        "sim_fps must equal target_fps so target frames are reference steps",
    )
    if artifact_mode == "scientific":
        _require(
            target_fps == DEFAULT_TARGET_FPS and sim_fps == DEFAULT_SIM_FPS,
            "scientific inventories require the release target_fps=sim_fps=50 contract",
        )
    selected, atlas_records = _atlas_selection(
        split_manifest,
        artifact_mode=artifact_mode,
        selected_motion_keys=selected_motion_keys,
    )
    root = Path(motion_root).expanduser().resolve() if motion_root is not None else None
    if root is not None:
        _require(root.is_dir(), f"motion_root must be a directory: {root}")
    dataset_provenance = _dataset_provenance_from_split(split_manifest)
    materialized_variants = _load_materialized_variants(dataset_provenance)

    motion_records: list[dict[str, Any]] = []
    for motion_key in selected:
        split_robot_path, source_path = _motion_path(atlas_records[motion_key], motion_key, root)
        payload = _load_motion_payload(source_path, motion_key)
        source_num_frames, source_fps = _source_contract(payload, motion_key)
        if artifact_mode == "scientific":
            _require(
                source_fps == SCIENTIFIC_SOURCE_FPS,
                "scientific inventories require every frozen source motion to be exactly 30 Hz",
            )
        target_num_frames = sonic_target_frame_count(
            source_num_frames,
            source_fps,
            target_fps,
        )
        _require(
            target_num_frames >= 2,
            f"{motion_key} produces fewer than two target frames and cannot be scheduled",
        )
        source_digest = _sha256_file(source_path)
        if materialized_variants is not None:
            _require(
                motion_key in materialized_variants,
                f"materialized manifest does not contain selected motion {motion_key!r}",
            )
            _validate_materialized_variant(
                materialized_variants[motion_key],
                motion_key=motion_key,
                source_digest=source_digest,
                source_num_frames=source_num_frames,
                source_fps=source_fps,
                target_num_frames=target_num_frames,
                target_fps=target_fps,
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
        "kind": REFERENCE_LENGTH_KIND,
        "schema_version": REFERENCE_LENGTH_SCHEMA_VERSION,
        "artifact_mode": artifact_mode,
        "scientific_use": artifact_mode == "scientific",
        "pilot_status": (
            None
            if artifact_mode == "scientific"
            else "non_scientific_subset_for_contract_or_runtime_validation_only"
        ),
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "partition": "D_atlas",
        "selected_motion_keys": selected,
        "selection_complete_for_d_atlas": artifact_mode == "scientific",
        "motion_count": len(selected),
        "target_fps": target_fps,
        "runtime_contract": {
            "target_fps": target_fps,
            "sim_fps": sim_fps,
            "motion_fps_scale": _fraction_record(scale),
            "max_len": max_len,
            "reference_num_steps_equals_target_num_frames": True,
            "float32_runtime_equivalence_scope": (
                "locked_30_to_50_hz_release_cohort_with_materialized_pair_cross_check"
                if artifact_mode == "scientific" and materialized_variants is not None
                else (
                    "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
                    if artifact_mode == "scientific"
                    else "not_claimed_for_generic_pilot_rates"
                )
            ),
        },
        "resampling_rule": dict(RESAMPLING_RULE),
        "dataset_provenance": dataset_provenance,
        "source_file_set_sha256": _source_file_set_sha256(motion_records),
        "motions": motion_records,
    }
    manifest[REFERENCE_LENGTH_DIGEST_FIELD] = canonical_sha256(
        manifest,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    validate_reference_length_inventory(manifest, split_manifest=split_manifest)
    return manifest


def reference_num_steps_by_motion(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Return schedule-ready target lengths after validating the inventory."""

    validate_reference_length_inventory(manifest)
    return {
        str(record["motion_key"]): int(record["target_num_frames"])
        for record in manifest["motions"]
    }


def validate_reference_length_inventory(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    verify_digest: bool = True,
    verify_source_files: bool = False,
) -> None:
    """Validate schema, exact counts, split binding, hashes, and mode labels."""

    _require(manifest.get("kind") == REFERENCE_LENGTH_KIND, "invalid reference inventory kind")
    _require(
        manifest.get("schema_version") == REFERENCE_LENGTH_SCHEMA_VERSION,
        "unsupported reference inventory schema version",
    )
    artifact_mode = manifest.get("artifact_mode")
    _require(
        artifact_mode in REFERENCE_LENGTH_MODES,
        f"artifact_mode must be one of {REFERENCE_LENGTH_MODES}",
    )
    _require(
        manifest.get("scientific_use") is (artifact_mode == "scientific"),
        "scientific_use must agree with artifact_mode",
    )
    expected_pilot_status = (
        None
        if artifact_mode == "scientific"
        else "non_scientific_subset_for_contract_or_runtime_validation_only"
    )
    _require(
        manifest.get("pilot_status") == expected_pilot_status,
        "pilot_status must explicitly mark pilot inventories non-scientific",
    )
    split_sha256 = _require_sha256(manifest.get("split_sha256"), "split_sha256")
    selection_sha256 = _require_sha256(
        manifest.get("split_selection_sha256"), "split_selection_sha256"
    )
    _require(manifest.get("partition") == "D_atlas", "partition must be D_atlas")
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
        manifest.get("selection_complete_for_d_atlas") is (artifact_mode == "scientific"),
        "selection_complete_for_d_atlas must agree with artifact_mode",
    )
    _require(manifest.get("motion_count") == len(selected), "motion_count mismatch")
    target_fps = manifest.get("target_fps")
    _require(
        isinstance(target_fps, int) and not isinstance(target_fps, bool) and target_fps > 0,
        "target_fps must be a positive integer",
    )
    runtime_contract = manifest.get("runtime_contract")
    _require(isinstance(runtime_contract, Mapping), "runtime_contract must be a mapping")
    _require(
        set(runtime_contract)
        == {
            "target_fps",
            "sim_fps",
            "motion_fps_scale",
            "max_len",
            "reference_num_steps_equals_target_num_frames",
            "float32_runtime_equivalence_scope",
        },
        "runtime_contract fields are invalid",
    )
    _require(
        runtime_contract.get("target_fps") == target_fps,
        "runtime_contract.target_fps mismatch",
    )
    _require(
        runtime_contract.get("sim_fps") == target_fps,
        "runtime_contract.sim_fps must equal target_fps",
    )
    _require(
        _fraction_from_record(
            runtime_contract.get("motion_fps_scale"),
            "runtime_contract.motion_fps_scale",
        )
        == 1,
        "runtime_contract.motion_fps_scale must equal 1",
    )
    _require(runtime_contract.get("max_len") == -1, "runtime_contract.max_len must equal -1")
    _require(
        runtime_contract.get("reference_num_steps_equals_target_num_frames") is True,
        "runtime_contract must bind reference steps to target frames",
    )
    expected_runtime_scope = (
        "locked_30_to_50_hz_release_cohort_with_materialized_pair_cross_check"
        if artifact_mode == "scientific"
        and isinstance(manifest.get("dataset_provenance"), Mapping)
        and manifest["dataset_provenance"].get("materialized_manifest") is not None
        else (
            "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
            if artifact_mode == "scientific"
            else "not_claimed_for_generic_pilot_rates"
        )
    )
    _require(
        runtime_contract.get("float32_runtime_equivalence_scope") == expected_runtime_scope,
        "runtime_contract.float32_runtime_equivalence_scope mismatch",
    )
    if artifact_mode == "scientific":
        _require(
            target_fps == DEFAULT_TARGET_FPS,
            "scientific inventory target_fps must be 50",
        )
    _require(manifest.get("resampling_rule") == RESAMPLING_RULE, "resampling_rule mismatch")
    dataset_provenance = manifest.get("dataset_provenance")
    _require(
        isinstance(dataset_provenance, Mapping),
        "dataset_provenance must be a mapping",
    )
    _require(
        set(dataset_provenance) == _DATASET_PROVENANCE_FIELDS,
        "dataset_provenance fields are invalid",
    )
    provenance_values = [dataset_provenance[field] for field in _DATASET_PROVENANCE_FIELDS]
    _require(
        all(value is None for value in provenance_values)
        or all(value is not None for value in provenance_values),
        "dataset_provenance must be entirely bound or entirely unavailable",
    )
    if all(value is not None for value in provenance_values):
        _require(
            isinstance(dataset_provenance["materialized_manifest"], str)
            and bool(dataset_provenance["materialized_manifest"]),
            "dataset_provenance.materialized_manifest must be a non-empty path",
        )
        _require_sha256(
            dataset_provenance["materialized_manifest_sha256"],
            "dataset_provenance.materialized_manifest_sha256",
        )
        _require_sha256(
            dataset_provenance["paired_dataset_sha256"],
            "dataset_provenance.paired_dataset_sha256",
        )
    materialized_variants = (
        _load_materialized_variants(dataset_provenance) if verify_source_files else None
    )

    motions_raw = manifest.get("motions")
    _require(isinstance(motions_raw, list), "motions must be a list")
    _require(len(motions_raw) == len(selected), "motions must cover every selected key exactly")
    normalized_motions: list[Mapping[str, Any]] = []
    for index, record in enumerate(motions_raw):
        _require(isinstance(record, Mapping), f"motions[{index}] must be a mapping")
        _require(set(record) == _MOTION_RECORD_FIELDS, f"motions[{index}] fields are invalid")
        _require(
            record.get("motion_key") == selected[index],
            "motions must follow selected_motion_keys order",
        )
        source_path = record.get("source_path")
        split_robot_path = record.get("split_robot_path")
        _require(
            isinstance(source_path, str) and source_path,
            f"motions[{index}].source_path must be non-empty",
        )
        _require(
            isinstance(split_robot_path, str) and split_robot_path,
            f"motions[{index}].split_robot_path must be non-empty",
        )
        source_digest = _require_sha256(
            record.get("source_file_sha256"), f"motions[{index}].source_file_sha256"
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
        source_fps = _fraction_from_record(record.get("source_fps"), f"motions[{index}].source_fps")
        if artifact_mode == "scientific":
            _require(
                source_fps == SCIENTIFIC_SOURCE_FPS,
                f"motions[{index}].source_fps must be exactly 30 for scientific use",
            )
        expected_frames = sonic_target_frame_count(source_num_frames, source_fps, target_fps)
        _require(
            target_num_frames == expected_frames,
            f"motions[{index}].target_num_frames violates the exact resampling rule",
        )
        if verify_source_files:
            path = Path(source_path)
            _require(path.is_file(), f"motions[{index}] source file is missing: {path}")
            _require(
                _sha256_file(path) == source_digest,
                f"motions[{index}] source file SHA-256 mismatch",
            )
            if materialized_variants is not None:
                motion_key = str(record["motion_key"])
                _require(
                    motion_key in materialized_variants,
                    f"materialized manifest does not contain {motion_key!r}",
                )
                _validate_materialized_variant(
                    materialized_variants[motion_key],
                    motion_key=motion_key,
                    source_digest=source_digest,
                    source_num_frames=source_num_frames,
                    source_fps=source_fps,
                    target_num_frames=target_num_frames,
                    target_fps=target_fps,
                )
        normalized_motions.append(record)
    _require(
        manifest.get("source_file_set_sha256") == _source_file_set_sha256(normalized_motions),
        "source_file_set_sha256 mismatch",
    )

    if split_manifest is not None:
        validate_split_manifest(split_manifest, verify_digest=True)
        _require(split_sha256 == split_manifest["split_sha256"], "split_sha256 binding mismatch")
        _require(
            selection_sha256 == split_manifest["selection_sha256"],
            "split_selection_sha256 binding mismatch",
        )
        _require(
            dict(dataset_provenance) == _dataset_provenance_from_split(split_manifest),
            "dataset_provenance does not match the frozen split",
        )
        expected_selected, atlas_records = _atlas_selection(
            split_manifest,
            artifact_mode=str(artifact_mode),
            selected_motion_keys=selected,
        )
        _require(selected == expected_selected, "selected motion binding mismatch")
        for index, record in enumerate(normalized_motions):
            expected_split_path = atlas_records[str(record["motion_key"])].get("robot_path")
            _require(
                record["split_robot_path"] == expected_split_path,
                f"motions[{index}].split_robot_path does not match the frozen split",
            )

    if verify_digest:
        expected_digest = _require_sha256(
            manifest.get(REFERENCE_LENGTH_DIGEST_FIELD),
            REFERENCE_LENGTH_DIGEST_FIELD,
        )
        actual_digest = canonical_sha256(
            manifest,
            digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
        )
        _require(
            expected_digest == actual_digest,
            f"{REFERENCE_LENGTH_DIGEST_FIELD} mismatch: "
            f"expected {expected_digest}, computed {actual_digest}",
        )
