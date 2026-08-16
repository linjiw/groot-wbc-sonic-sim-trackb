from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest

from gear_sonic.data_process.convert_kimodo_to_motion_lib import main as converter_main
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS
from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    KIMODO_G1_JOINT_NAMES,
    KimodoQposError,
    load_kimodo_qpos_csv,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
    transform_qpos_to_scene,
    validate_kimodo_qpos,
)
from gear_sonic.dataset_generation.schemas import (
    ArtifactRef,
    ConversionResult,
    GenerationResult,
)

EXPECTED_KIMODO_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _qpos(frame_count: int = 3) -> np.ndarray:
    qpos = np.zeros((frame_count, 36), dtype=np.float64)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.arange(frame_count, dtype=np.float64)
    return qpos


def _write_parent_generation_result(
    source: Path,
    manifest: Path,
    *,
    source_fps: float = 30.0,
) -> GenerationResult:
    result = GenerationResult(
        episode_request_id="sha256:" + "d" * 64,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=source_fps,
        seed=11,
        artifacts=(ArtifactRef.from_path("generated_qpos", source, media_type="text/csv"),),
    )
    result.write_json(manifest)
    return result


def test_loads_headerless_kimodo_g1_csv(tmp_path: Path) -> None:
    path = tmp_path / "motion.csv"
    np.savetxt(path, _qpos(), delimiter=",")

    loaded = load_kimodo_qpos_csv(path)

    np.testing.assert_allclose(loaded, _qpos())


@pytest.mark.parametrize(
    "bad_qpos",
    [
        np.zeros((2, 35)),
        np.full((2, 36), np.nan),
        np.zeros((2, 36)),
    ],
)
def test_rejects_invalid_qpos(bad_qpos: np.ndarray) -> None:
    with pytest.raises(KimodoQposError):
        validate_kimodo_qpos(bad_qpos)


def test_scene_transform_rotates_translation_and_root_orientation() -> None:
    qpos = _qpos()
    transformed = transform_qpos_to_scene(
        qpos,
        scene_start_xyz=(10.0, 20.0, 0.25),
        scene_yaw=math.pi / 2,
    )

    np.testing.assert_allclose(transformed[:, 0], [10.0, 10.0, 10.0], atol=1e-7)
    np.testing.assert_allclose(transformed[:, 1], [20.0, 21.0, 22.0], atol=1e-7)
    np.testing.assert_allclose(transformed[:, 2], [1.05, 1.05, 1.05], atol=1e-7)
    expected_yaw_quat_wxyz = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    np.testing.assert_allclose(
        transformed[:, 3:7],
        np.tile(expected_yaw_quat_wxyz, (3, 1)),
        atol=1e-7,
    )


def test_conversion_preserves_mujoco_joint_order_and_axis_mapping() -> None:
    qpos = _qpos()
    joint_values = np.linspace(-0.29, 0.29, 29)
    qpos[:, 7:] = joint_values

    entry = qpos_to_sonic_motion_entry(qpos, source_fps=30)

    assert KIMODO_G1_JOINT_NAMES == EXPECTED_KIMODO_G1_JOINT_NAMES
    np.testing.assert_allclose(entry["dof"], np.tile(joint_values, (3, 1)), atol=1e-7)
    np.testing.assert_allclose(
        entry["pose_aa"][:, 1:, :],
        np.tile(DOF_AXIS[None, :, :] * joint_values[None, :, None], (3, 1, 1)),
        atol=1e-7,
    )
    assert entry["fps"] == 30
    assert entry["root_rot"].shape == (3, 4)
    np.testing.assert_allclose(entry["root_rot"], np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)))


def test_conversion_writes_loadable_sonic_joblib_mapping(tmp_path: Path) -> None:
    entry = qpos_to_sonic_motion_entry(_qpos(), source_fps=30)
    output = tmp_path / "motion.pkl"

    save_sonic_motion_file(output, motion_key="golden_forward", motion_entry=entry)
    loaded = joblib.load(output)

    assert list(loaded) == ["golden_forward"]
    np.testing.assert_allclose(loaded["golden_forward"]["dof"], entry["dof"])


def test_rejects_non_integral_fps_until_motion_lib_schema_supports_it() -> None:
    with pytest.raises(KimodoQposError, match="integer fps"):
        qpos_to_sonic_motion_entry(_qpos(), source_fps=29.97)


def test_converter_cli_writes_hashed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "kimodo.csv"
    output_path = tmp_path / "sonic.pkl"
    np.savetxt(input_path, _qpos(), delimiter=",")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_kimodo_to_motion_lib.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--motion-key",
            "golden",
        ],
    )

    assert converter_main() == 0

    manifest_path = output_path.with_suffix(".pkl.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["motion_key"] == "golden"
    assert manifest["frame_count"] == 3
    assert manifest["input"]["sha256"].startswith("sha256:")
    assert manifest["output"]["sha256"].startswith("sha256:")
    assert list(joblib.load(output_path)) == ["golden"]


def test_converter_cli_writes_typed_parent_linked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "kimodo.csv"
    output_path = tmp_path / "sonic.pkl"
    parent_path = tmp_path / "generation.json"
    manifest_path = tmp_path / "conversion" / "result.json"
    np.savetxt(input_path, _qpos(), delimiter=",")
    parent = _write_parent_generation_result(input_path, parent_path, source_fps=50.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_kimodo_to_motion_lib.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--motion-key",
            "typed_golden",
            "--generation-result",
            str(parent_path),
            "--manifest",
            str(manifest_path),
            "--scene-start",
            "1.0",
            "2.0",
            "0.25",
            "--scene-yaw",
            "0.5",
        ],
    )

    assert converter_main() == 0

    loaded = ConversionResult.read_json(manifest_path)
    loaded.verify_parent(GenerationResult.read_json(parent_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded.episode_request_id == parent.episode_request_id
    assert loaded.source_generation_result_id == parent.result_id
    assert loaded.source_artifact_name == "generated_qpos"
    assert loaded.source_fps == 50.0
    assert loaded.motion_key == "typed_golden"
    assert loaded.scene_start_xyz == (1.0, 2.0, 0.25)
    assert loaded.scene_yaw == 0.5
    assert manifest["conversion_result_id"] == loaded.result_id
    assert manifest["converter"] == manifest["converter_name"]
    assert not Path(manifest["input"]["path"]).is_absolute()
    assert not Path(manifest["output"]["path"]).is_absolute()
    assert joblib.load(output_path)["typed_golden"]["fps"] == 50


def test_converter_cli_rejects_input_not_in_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_input = tmp_path / "parent.csv"
    other_input = tmp_path / "other.csv"
    output_path = tmp_path / "sonic.pkl"
    parent_path = tmp_path / "generation.json"
    np.savetxt(parent_input, _qpos(), delimiter=",")
    other_qpos = _qpos()
    other_qpos[:, 0] += 0.25
    np.savetxt(other_input, other_qpos, delimiter=",")
    _write_parent_generation_result(parent_input, parent_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_kimodo_to_motion_lib.py",
            "--input",
            str(other_input),
            "--output",
            str(output_path),
            "--motion-key",
            "wrong_source",
            "--generation-result",
            str(parent_path),
        ],
    )

    with pytest.raises(ValueError, match="does not match any parent GenerationResult artifact"):
        converter_main()
    assert not output_path.exists()


def test_converter_cli_rejects_parent_fps_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "kimodo.csv"
    output_path = tmp_path / "sonic.pkl"
    parent_path = tmp_path / "generation.json"
    np.savetxt(input_path, _qpos(), delimiter=",")
    _write_parent_generation_result(input_path, parent_path, source_fps=50.0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_kimodo_to_motion_lib.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--motion-key",
            "wrong_fps",
            "--generation-result",
            str(parent_path),
            "--source-fps",
            "30",
        ],
    )

    with pytest.raises(ValueError, match="does not match parent source_fps"):
        converter_main()
    assert not output_path.exists()


def test_converter_cli_rejects_destructive_path_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "kimodo.csv"
    np.savetxt(input_path, _qpos(), delimiter=",")
    original = input_path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_kimodo_to_motion_lib.py",
            "--input",
            str(input_path),
            "--output",
            str(input_path),
        ],
    )

    with pytest.raises(ValueError, match="paths must be distinct"):
        converter_main()
    assert input_path.read_bytes() == original
