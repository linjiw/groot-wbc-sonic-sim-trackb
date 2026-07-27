#!/usr/bin/env python3
"""Materialize a hash-locked official BONES-SEED cohort without broad extraction.

The command is deliberately offline.  It verifies the complete G1 archive and
all seven ordered SMPL tar parts against the frozen source lock before creating
the staging directory.  Only members declared by the frozen cohort are copied,
and the final directory is published only after the release-timeline validator
has built ``dataset_manifest.json``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from typing import Any, BinaryIO, Callable, Sequence

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    convert_sequence,
    downsample_sequence,
    load_bones_csv,
)
from scripts.research.build_bones_seed_paired_manifest import (  # noqa: E402
    build_manifest,
)

GENERATOR = "scripts/research/materialize_bones_seed_official_cohort.py"
SOURCE_FPS = 120
TARGET_FPS = 30
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class MaterializationError(RuntimeError):
    """Raised when a frozen input or extracted inventory fails closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_lock_record(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise MaterializationError(f"{label} source-lock record must be an object")
    byte_count = record.get("bytes")
    digest = record.get("sha256")
    if not isinstance(byte_count, int) or byte_count < 0:
        raise MaterializationError(f"{label} source-lock bytes must be a non-negative integer")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise MaterializationError(
            f"{label} source-lock sha256 must be 64 lowercase hex characters"
        )
    return {**record, "bytes": byte_count, "sha256": digest}


def _verify_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise MaterializationError(f"{label} is not a file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != record["bytes"]:
        raise MaterializationError(
            f"{label} byte count mismatch: expected {record['bytes']}, got {actual_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != record["sha256"]:
        raise MaterializationError(
            f"{label} sha256 mismatch: expected {record['sha256']}, got {actual_sha256}"
        )


def _safe_archive_path(value: str, *, label: str, suffix: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise MaterializationError(f"unsafe {label} path: {value!r}")
    if value.startswith("/") or value.endswith("/"):
        raise MaterializationError(f"unsafe {label} path: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise MaterializationError(f"unsafe {label} path: {value!r}")
    path = PurePosixPath(value)
    if not path.name.endswith(suffix):
        raise MaterializationError(f"{label} path must end in {suffix}: {value!r}")
    return path


def _load_member_manifest(path: Path, *, label: str, suffix: str) -> list[str]:
    try:
        members = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MaterializationError(f"cannot read {label} member manifest: {path}") from exc
    if not members or any(not member for member in members):
        raise MaterializationError(f"{label} member manifest must contain non-empty lines")
    for member in members:
        _safe_archive_path(member, label=label, suffix=suffix)
    if len(members) != len(set(members)):
        raise MaterializationError(f"{label} member manifest contains duplicate paths")
    return members


def _validate_motion_key(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise MaterializationError(f"unsafe motion key: {value!r}")
    if "/" in value or "\\" in value or "\x00" in value:
        raise MaterializationError(f"unsafe motion key: {value!r}")
    return value


def _load_frozen_inputs(
    source_lock_path: Path,
    cohort_path: Path,
    g1_members_path: Path,
    smpl_members_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[str],
    list[str],
    dict[str, Any],
    list[dict[str, Any]],
]:
    source_lock = _load_json_object(source_lock_path, label="source lock")
    cohort = _load_json_object(cohort_path, label="cohort manifest")
    try:
        bones_lock = source_lock["bones_seed"]
        smpl_lock = source_lock["smpl"]
        g1_record = _validate_lock_record(bones_lock["files"]["g1.tar.gz"], label="G1 archive")
        raw_smpl_records = smpl_lock["parts"]
    except (KeyError, TypeError) as exc:
        raise MaterializationError("source lock is missing BONES-SEED archive records") from exc
    if not isinstance(raw_smpl_records, list) or len(raw_smpl_records) != 7:
        raise MaterializationError("source lock must declare exactly seven ordered SMPL parts")
    smpl_records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_smpl_records):
        normalized = _validate_lock_record(record, label=f"SMPL part {index}")
        remote_path = normalized.get("path")
        _safe_archive_path(
            remote_path, label=f"SMPL part {index}", suffix=f"part_a{chr(97 + index)}"
        )
        smpl_records.append(normalized)

    try:
        motions = cohort["motions"]
        cohort_source = cohort["source"]
    except (KeyError, TypeError) as exc:
        raise MaterializationError("cohort manifest is missing source or motions") from exc
    if not isinstance(motions, list) or not motions:
        raise MaterializationError("cohort motions must be a non-empty list")
    normalized_motions: list[dict[str, Any]] = []
    for index, motion in enumerate(motions):
        if not isinstance(motion, dict):
            raise MaterializationError(f"cohort motion {index} must be an object")
        key = _validate_motion_key(motion.get("motion_key"))
        source_frames = motion.get("duration_source_frames")
        if not isinstance(source_frames, int) or source_frames <= 0:
            raise MaterializationError(f"{key}: duration_source_frames must be positive")
        g1_member = str(
            _safe_archive_path(motion.get("g1_archive_member"), label=f"{key} G1", suffix=".csv")
        )
        smpl_member = str(
            _safe_archive_path(
                motion.get("smpl_archive_member"), label=f"{key} SMPL", suffix=".pkl"
            )
        )
        if PurePosixPath(g1_member).stem != key:
            raise MaterializationError(f"{key}: G1 archive member stem differs from motion key")
        if smpl_member != f"smpl_filtered/{key}.pkl":
            raise MaterializationError(
                f"{key}: SMPL archive member is not the canonical release path"
            )
        normalized_motions.append(
            {
                **motion,
                "motion_key": key,
                "g1_archive_member": g1_member,
                "smpl_archive_member": smpl_member,
            }
        )
    keys = [motion["motion_key"] for motion in normalized_motions]
    if len(keys) != len(set(keys)):
        raise MaterializationError("cohort contains duplicate motion keys")

    g1_members = _load_member_manifest(g1_members_path, label="G1", suffix=".csv")
    smpl_members = _load_member_manifest(smpl_members_path, label="SMPL", suffix=".pkl")
    expected_g1 = [motion["g1_archive_member"] for motion in normalized_motions]
    expected_smpl = [motion["smpl_archive_member"] for motion in normalized_motions]
    if g1_members != expected_g1:
        raise MaterializationError("G1 member manifest does not exactly match cohort order")
    if smpl_members != expected_smpl:
        raise MaterializationError("SMPL member manifest does not exactly match cohort order")

    try:
        if cohort_source["revision"] != bones_lock["revision"]:
            raise MaterializationError("cohort and source lock BONES-SEED revisions differ")
        if cohort_source.get("repo_id") not in {None, bones_lock["repo_id"]}:
            raise MaterializationError("cohort and source lock BONES-SEED repo IDs differ")
    except (KeyError, TypeError) as exc:
        raise MaterializationError("source lock is missing repository identity fields") from exc
    return (
        source_lock,
        cohort,
        normalized_motions,
        g1_members,
        smpl_members,
        g1_record,
        smpl_records,
    )


class _ConcatenatedReader(io.RawIOBase):
    """Read multiple files as one binary stream without making a joined copy."""

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        self._paths = iter(paths)
        self._current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed concatenated stream")
        view = memoryview(buffer)
        while True:
            if self._current is None:
                try:
                    self._current = next(self._paths).open("rb")
                except StopIteration:
                    return 0
            count = self._current.readinto(view)
            if count:
                return count
            self._current.close()
            self._current = None

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


def _copy_exact(source: BinaryIO, destination: Path, expected_bytes: int) -> None:
    copied = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        while copied < expected_bytes:
            chunk = source.read(min(1024 * 1024, expected_bytes - copied))
            if not chunk:
                raise MaterializationError(
                    f"archive member ended early: {destination} ({copied}/{expected_bytes})"
                )
            output.write(chunk)
            copied += len(chunk)
    if copied != expected_bytes:
        raise MaterializationError(f"archive member size changed while extracting: {destination}")


def _extract_selected(
    archive: tarfile.TarFile,
    *,
    targets: dict[str, Path],
    destination_root: Path,
) -> None:
    root = destination_root.resolve()
    seen: set[str] = set()
    for member in archive:
        destination = targets.get(member.name)
        if destination is None:
            continue
        if member.name in seen:
            raise MaterializationError(f"archive contains duplicate selected member: {member.name}")
        seen.add(member.name)
        _safe_archive_path(member.name, label="selected archive", suffix=destination.suffix)
        if not member.isreg():
            raise MaterializationError(
                f"selected archive member is not a regular file: {member.name}"
            )
        resolved_destination = destination.resolve()
        if not resolved_destination.is_relative_to(root):
            raise MaterializationError(
                f"selected archive member escapes destination: {member.name}"
            )
        source = archive.extractfile(member)
        if source is None:
            raise MaterializationError(f"cannot read selected archive member: {member.name}")
        with source:
            _copy_exact(source, destination, member.size)
    missing = sorted(set(targets) - seen)
    if missing:
        raise MaterializationError(f"archive is missing selected members: {missing[:10]}")


def _convert_robot_inventory(
    raw_root: Path,
    robot_output: Path,
    motions: Sequence[dict[str, Any]],
) -> None:
    robot_output.mkdir()
    for motion in motions:
        key = motion["motion_key"]
        csv_path = raw_root.joinpath(*PurePosixPath(motion["g1_archive_member"]).parts)
        sequence = load_bones_csv(str(csv_path))
        source_frames = int(sequence["joint_pos"].shape[0])
        expected_source_frames = int(motion["duration_source_frames"])
        if source_frames != expected_source_frames:
            raise MaterializationError(
                f"{key}: CSV frames {source_frames} differ from cohort metadata {expected_source_frames}"
            )
        if tuple(sequence["joint_pos"].shape) != (source_frames, 29):
            raise MaterializationError(f"{key}: G1 CSV must contain exactly 29 joint DOFs")
        converted = convert_sequence(sequence, SOURCE_FPS)
        converted = downsample_sequence(converted, SOURCE_FPS, TARGET_FPS)
        expected_output_frames = (source_frames + 3) // 4
        if converted["root_trans_offset"].shape[0] != expected_output_frames:
            raise MaterializationError(f"{key}: official 120->30 conversion changed frame contract")
        joblib.dump({key: converted}, robot_output / f"{key}.pkl", compress=True)


def _assert_flat_inventory(root: Path, keys: Sequence[str], *, label: str) -> None:
    direct = sorted(path.stem for path in root.glob("*.pkl"))
    recursive = sorted(root.rglob("*.pkl"))
    if len(direct) != len(recursive) or direct != sorted(keys):
        raise MaterializationError(f"{label} output inventory is not flat and exact")


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _publish_staging(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output already exists; refusing to replace it: {output_dir}")
    if staging_dir.parent.stat().st_dev != output_dir.parent.stat().st_dev:
        raise MaterializationError("staging and output directories must be on the same filesystem")
    # Linux renameat2 publishes the complete directory atomically while the
    # NOREPLACE flag closes the check/rename race around a concurrently created
    # output.  Failing when the syscall is unavailable is safer than weakening
    # the no-overwrite contract.
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise MaterializationError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(staging_dir),
        -100,
        os.fsencode(output_dir),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(
                f"output appeared during build; refusing to replace it: {output_dir}"
            )
        raise OSError(error, os.strerror(error), str(output_dir))


def materialize_cohort(
    *,
    g1_archive: Path,
    smpl_parts: Sequence[Path],
    source_lock_path: Path,
    cohort_path: Path,
    g1_members_path: Path,
    smpl_members_path: Path,
    staging_dir: Path,
    output_dir: Path,
    robot_converter: Callable[
        [Path, Path, Sequence[dict[str, Any]]], None
    ] = _convert_robot_inventory,
) -> dict[str, Any]:
    """Verify, selectively extract, convert, validate, and publish one cohort."""
    g1_archive = Path(g1_archive).resolve()
    smpl_parts = [Path(path).resolve() for path in smpl_parts]
    source_lock_path = Path(source_lock_path).resolve()
    cohort_path = Path(cohort_path).resolve()
    g1_members_path = Path(g1_members_path).resolve()
    smpl_members_path = Path(smpl_members_path).resolve()
    # Preserve the final path component so a broken output/staging symlink is
    # detected and rejected rather than silently followed by ``resolve()``.
    staging_dir = Path(staging_dir).absolute()
    output_dir = Path(output_dir).absolute()

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output already exists; refusing to replace it: {output_dir}")
    if staging_dir.exists() or staging_dir.is_symlink():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")
    if _paths_overlap(staging_dir, output_dir):
        raise MaterializationError("staging and output directories must not overlap")
    for source_path in (
        g1_archive,
        *smpl_parts,
        source_lock_path,
        cohort_path,
        g1_members_path,
        smpl_members_path,
    ):
        if _paths_overlap(source_path, staging_dir) or _paths_overlap(source_path, output_dir):
            raise MaterializationError(f"input path overlaps staging/output: {source_path}")

    (
        source_lock,
        _cohort,
        motions,
        g1_members,
        smpl_members,
        g1_record,
        smpl_records,
    ) = _load_frozen_inputs(
        source_lock_path,
        cohort_path,
        g1_members_path,
        smpl_members_path,
    )
    if len(smpl_parts) != len(smpl_records):
        raise MaterializationError(
            f"expected {len(smpl_records)} ordered SMPL parts, got {len(smpl_parts)}"
        )
    if len({path for path in smpl_parts}) != len(smpl_parts):
        raise MaterializationError("SMPL part paths must be distinct and ordered")

    _verify_file(g1_archive, g1_record, label="G1 archive")
    for index, (path, record) in enumerate(zip(smpl_parts, smpl_records, strict=True)):
        _verify_file(path, record, label=f"SMPL part {index}")

    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if staging_dir.parent.stat().st_dev != output_dir.parent.stat().st_dev:
        raise MaterializationError("staging and output directories must be on the same filesystem")
    staging_dir.mkdir()
    raw_root = staging_dir / ".raw_g1"
    robot_output = staging_dir / "robot_filtered"
    smpl_output = staging_dir / "smpl_filtered"
    raw_root.mkdir()
    smpl_output.mkdir()
    keys = [motion["motion_key"] for motion in motions]
    try:
        g1_targets = {
            member: raw_root.joinpath(*PurePosixPath(member).parts) for member in g1_members
        }
        with tarfile.open(g1_archive, mode="r|gz") as archive:
            _extract_selected(archive, targets=g1_targets, destination_root=raw_root)

        robot_converter(raw_root, robot_output, motions)
        shutil.rmtree(raw_root)

        smpl_targets = {
            member: smpl_output / f"{motion['motion_key']}.pkl"
            for member, motion in zip(smpl_members, motions, strict=True)
        }
        with _ConcatenatedReader(smpl_parts) as parts_stream:
            with tarfile.open(fileobj=parts_stream, mode="r|") as archive:
                _extract_selected(archive, targets=smpl_targets, destination_root=smpl_output)

        _assert_flat_inventory(robot_output, keys, label="robot")
        _assert_flat_inventory(smpl_output, keys, label="SMPL")
        manifest = build_manifest(cohort_path, robot_output, smpl_output)
        manifest["materialization"] = {
            "schema_version": 1,
            "generator": GENERATOR,
            "source_lock": {
                "path": str(source_lock_path),
                "sha256": _sha256(source_lock_path),
            },
            "archives": {
                "g1": {"path": "g1.tar.gz", **g1_record},
                "smpl_parts": smpl_records,
                "all_full_archives_verified_before_extraction": True,
            },
            "member_manifests": {
                "g1_sha256": _sha256(g1_members_path),
                "smpl_sha256": _sha256(smpl_members_path),
                "exact_cohort_order": True,
            },
            "robot_transform": {
                "implementation": (
                    "gear_sonic.data_process.convert_soma_csv_to_motion_lib:"
                    "load_bones_csv+convert_sequence+downsample_sequence"
                ),
                "source_fps": SOURCE_FPS,
                "target_fps": TARGET_FPS,
                "stride": 4,
            },
            "selective_regular_files_only": True,
            "source_lock_schema_version": source_lock.get("schema_version"),
        }
        manifest_path = staging_dir / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Re-run against serialized paths immediately before publication.
        rebuilt = build_manifest(cohort_path, robot_output, smpl_output)
        if (
            rebuilt["output"]["paired_dataset_sha256"]
            != manifest["output"]["paired_dataset_sha256"]
        ):
            raise MaterializationError("paired dataset hash changed before publication")
        _publish_staging(staging_dir, output_dir)
        return manifest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-archive", type=Path, required=True)
    parser.add_argument(
        "--smpl-parts",
        type=Path,
        nargs="+",
        required=True,
        help="Seven local SMPL tar parts in source-lock order (aa through ag).",
    )
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--g1-members", type=Path, required=True)
    parser.add_argument("--smpl-members", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_cohort(
        g1_archive=args.g1_archive,
        smpl_parts=args.smpl_parts,
        source_lock_path=args.source_lock,
        cohort_path=args.cohort,
        g1_members_path=args.g1_members,
        smpl_members_path=args.smpl_members,
        staging_dir=args.staging_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "motion_count": manifest["output"]["motion_count"],
                "paired_dataset_sha256": manifest["output"]["paired_dataset_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
