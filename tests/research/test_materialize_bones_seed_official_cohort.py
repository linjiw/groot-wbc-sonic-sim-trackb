from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np
import pytest

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import BONES_CSV_JOINT_NAMES
from scripts.research.materialize_bones_seed_official_cohort import (
    MaterializationError,
    materialize_cohort,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_g1_csv(path: Path, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *BONES_CSV_JOINT_NAMES,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for frame in range(frames):
            writer.writerow([frame, frame, 0, 76, 0, 0, 0, *([0] * 29)])


def _smpl_bytes(path: Path) -> bytes:
    smpl = {
        "pose_aa": np.zeros((3, 72), dtype=np.float32),
        "transl": np.zeros((3, 3), dtype=np.float32),
        "smpl_joints": np.zeros((3, 24, 3), dtype=np.float32),
        "fps": 50.0,
        "original_pose_aa": np.zeros((2, 72), dtype=np.float32),
        "original_fps": 30.0,
    }
    joblib.dump(smpl, path)
    return path.read_bytes()


def _add_bytes(archive: tarfile.TarFile, member: str, content: bytes) -> None:
    info = tarfile.TarInfo(member)
    info.size = len(content)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(content))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _frozen_inputs(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source"
    source.mkdir()
    key = "motion_a"
    g1_member = f"g1/csv/session/{key}.csv"
    smpl_member = f"smpl_filtered/{key}.pkl"

    csv_path = source / "motion.csv"
    _write_g1_csv(csv_path)
    g1_archive = source / "g1.tar.gz"
    with tarfile.open(g1_archive, "w:gz") as archive:
        _add_bytes(archive, "unselected/ignored.txt", b"ignored")
        _add_bytes(archive, g1_member, csv_path.read_bytes())

    smpl_pickle = source / "smpl.pkl"
    payload = _smpl_bytes(smpl_pickle)
    joined_tar = source / "smpl.tar"
    with tarfile.open(joined_tar, "w") as archive:
        _add_bytes(archive, smpl_member, payload)
    joined = joined_tar.read_bytes()
    smpl_parts = []
    for index in range(7):
        begin = len(joined) * index // 7
        end = len(joined) * (index + 1) // 7
        part = source / f"smpl.part_a{chr(97 + index)}"
        part.write_bytes(joined[begin:end])
        smpl_parts.append(part)

    revision = "a" * 40
    source_lock = source / "source_lock.json"
    _write_json(
        source_lock,
        {
            "schema_version": 1,
            "bones_seed": {
                "repo_id": "bones-studio/seed",
                "repo_type": "dataset",
                "revision": revision,
                "files": {"g1.tar.gz": _record(g1_archive)},
            },
            "smpl": {
                "repo_id": "nvidia/GEAR-SONIC",
                "repo_type": "model",
                "revision": "b" * 40,
                "parts": [
                    {
                        "path": f"bones_seed_smpl/bones_seed_smpl.tar.part_a{chr(97 + index)}",
                        **_record(part),
                    }
                    for index, part in enumerate(smpl_parts)
                ],
            },
        },
    )
    cohort = source / "cohort.json"
    _write_json(
        cohort,
        {
            "schema_version": 1,
            "source": {
                "repo_id": "bones-studio/seed",
                "revision": revision,
            },
            "selection": {"selection_sha256": "c" * 64},
            "motions": [
                {
                    "motion_key": key,
                    "duration_source_frames": 8,
                    "category": "Baseline",
                    "package": "Locomotion",
                    "duration_bin": 0,
                    "is_mirror": False,
                    "g1_archive_member": g1_member,
                    "smpl_archive_member": smpl_member,
                }
            ],
        },
    )
    g1_members = source / "g1_members.txt"
    smpl_members = source / "smpl_members.txt"
    g1_members.write_text(g1_member + "\n", encoding="utf-8")
    smpl_members.write_text(smpl_member + "\n", encoding="utf-8")
    return {
        "g1_archive": g1_archive,
        "smpl_parts": smpl_parts,
        "source_lock_path": source_lock,
        "cohort_path": cohort,
        "g1_members_path": g1_members,
        "smpl_members_path": smpl_members,
        "staging_dir": tmp_path / "staging",
        "output_dir": tmp_path / "output",
    }


def test_materializes_verified_archives_with_exact_flat_inventory(tmp_path: Path) -> None:
    inputs = _frozen_inputs(tmp_path)

    manifest = materialize_cohort(**inputs)

    output = inputs["output_dir"]
    assert isinstance(output, Path)
    assert manifest["output"]["motion_keys"] == ["motion_a"]
    assert manifest["materialization"]["archives"]["all_full_archives_verified_before_extraction"]
    assert sorted(path.name for path in (output / "robot_filtered").iterdir()) == ["motion_a.pkl"]
    assert sorted(path.name for path in (output / "smpl_filtered").iterdir()) == ["motion_a.pkl"]
    assert not (output / ".raw_g1").exists()
    robot = joblib.load(output / "robot_filtered" / "motion_a.pkl")["motion_a"]
    assert robot["root_trans_offset"].shape == (2, 3)
    assert robot["fps"] == 30
    persisted = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert (
        persisted["output"]["paired_dataset_sha256"] == manifest["output"]["paired_dataset_sha256"]
    )


def test_hash_mismatch_fails_before_creating_staging(tmp_path: Path) -> None:
    inputs = _frozen_inputs(tmp_path)
    archive = inputs["g1_archive"]
    assert isinstance(archive, Path)
    corrupted = bytearray(archive.read_bytes())
    corrupted[0] ^= 1
    archive.write_bytes(corrupted)

    with pytest.raises(MaterializationError, match="sha256 mismatch"):
        materialize_cohort(**inputs)

    assert not inputs["staging_dir"].exists()
    assert not inputs["output_dir"].exists()


def test_traversal_member_is_rejected_before_extraction(tmp_path: Path) -> None:
    inputs = _frozen_inputs(tmp_path)
    cohort_path = inputs["cohort_path"]
    g1_members_path = inputs["g1_members_path"]
    assert isinstance(cohort_path, Path)
    assert isinstance(g1_members_path, Path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["motions"][0]["g1_archive_member"] = "../motion_a.csv"
    _write_json(cohort_path, cohort)
    g1_members_path.write_text("../motion_a.csv\n", encoding="utf-8")

    with pytest.raises(MaterializationError, match="unsafe motion_a G1 path"):
        materialize_cohort(**inputs)

    assert not (tmp_path / "motion_a.csv").exists()
    assert not inputs["staging_dir"].exists()
    assert not inputs["output_dir"].exists()


def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    inputs = _frozen_inputs(tmp_path)
    output = inputs["output_dir"]
    assert isinstance(output, Path)
    output.mkdir()
    marker = output / "user-data.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        materialize_cohort(**inputs)

    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not inputs["staging_dir"].exists()


def test_broken_output_symlink_is_never_followed(tmp_path: Path) -> None:
    inputs = _frozen_inputs(tmp_path)
    output = inputs["output_dir"]
    assert isinstance(output, Path)
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        materialize_cohort(**inputs)

    assert output.is_symlink()
    assert not inputs["staging_dir"].exists()
