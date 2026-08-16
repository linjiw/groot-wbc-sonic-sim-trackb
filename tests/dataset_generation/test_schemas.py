from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from gear_sonic.dataset_generation.schemas import (
    G1_MUJOCO_JOINT_ORDER,
    KIMODO_ROOT_QUATERNION_ORDER,
    SCENE_TRANSFORM_ORDER,
    SONIC_ROOT_QUATERNION_ORDER,
    ArtifactRef,
    ConversionResult,
    EpisodeRequest,
    GenerationResult,
    content_sha256,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _request(**overrides) -> EpisodeRequest:
    values = {
        "scene_id": "household_room_001",
        "task_family": "navigation",
        "task_prompt": "walk to the kitchen",
        "style_prompt": "careful natural walk",
        "route_xy": ((0.0, 0.0), (1.0, 0.0), (2.0, 0.5)),
        "nominal_speed_mps": 0.8,
        "duration_s": 3.0,
        "kimodo_model": "kimodo-g1-rp",
        "kimodo_seed": 11,
        "simulation_seed": 12,
        "render_seed": 13,
        "candidate_index": 0,
        "scene_hash": SHA_A,
        "controller_hash": SHA_B,
        "physics_hash": SHA_C,
    }
    values.update(overrides)
    return EpisodeRequest(**values)


def _generation_result(source: Path, *, source_fps: float = 30.0) -> GenerationResult:
    return GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=source_fps,
        seed=11,
        artifacts=(ArtifactRef.from_path("generated_qpos", source, media_type="text/csv"),),
    )


def _conversion_result(
    source: Path,
    output: Path,
    parent: GenerationResult,
    **overrides,
) -> ConversionResult:
    values = {
        "episode_request_id": parent.episode_request_id,
        "source_generation_result_id": parent.result_id,
        "source_artifact_name": "generated_qpos",
        "converter_name": "gear_sonic.dataset_generation.kimodo_motion_adapter",
        "converter_version": "1",
        "source_fps": parent.source_fps,
        "motion_key": "golden_forward",
        "frame_count": 3,
        "scene_start_xyz": (1.0, 2.0, 0.25),
        "scene_yaw": 0.5,
        "canonicalize_horizontal_origin": True,
        # Reverse the semantic order deliberately; the contract canonicalizes it.
        "artifacts": (
            ArtifactRef.from_path("sonic_motion_lib", output, media_type="application/x-joblib"),
            ArtifactRef.from_path("kimodo_qpos_csv", source, media_type="text/csv"),
        ),
    }
    values.update(overrides)
    return ConversionResult(**values)


def test_request_id_is_stable_under_json_key_order() -> None:
    request = _request()
    shuffled = dict(reversed(list(request.to_dict().items())))

    loaded = EpisodeRequest.from_dict(shuffled)

    assert loaded == request
    assert loaded.request_id == request.request_id
    assert content_sha256({"b": 2, "a": 1}) == content_sha256({"a": 1, "b": 2})


def test_request_id_changes_with_seed_or_route() -> None:
    request = _request()

    assert _request(kimodo_seed=99).request_id != request.request_id
    assert _request(route_xy=((0.0, 0.0), (1.1, 0.0))).request_id != request.request_id
    assert _request(physics_hash=SHA_A).request_id != request.request_id


def test_request_round_trip_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    request = _request()
    request.write_json(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["duration_s"] = 9.0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        EpisodeRequest.read_json(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"route_xy": ((0.0, 0.0),)}, "at least two"),
        ({"nominal_speed_mps": 0.0}, "positive finite"),
        ({"scene_hash": "not-a-hash"}, "sha256"),
        ({"route_frame": "world"}, "scene_local"),
    ],
)
def test_request_validation(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


def test_generation_result_hashes_artifacts_and_round_trips(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_text("1,2,3\n", encoding="utf-8")
    artifact = ArtifactRef.from_path("qpos", motion, media_type="text/csv")
    result = GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=11,
        artifacts=(artifact,),
    )
    path = tmp_path / "generation.json"
    result.write_json(path)

    loaded = GenerationResult.read_json(path)
    serialized = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == result
    assert loaded.result_id == result.result_id
    assert artifact.size_bytes == motion.stat().st_size
    assert serialized["artifacts"][0]["path"] == "motion.csv"


def test_generation_result_rejects_missing_artifact(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_text("1,2,3\n", encoding="utf-8")
    result = GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=11,
        artifacts=(ArtifactRef.from_path("qpos", motion),),
    )
    manifest = tmp_path / "generation.json"
    result.write_json(manifest)
    motion.unlink()

    with pytest.raises(FileNotFoundError, match="artifact 'qpos'.*missing"):
        GenerationResult.read_json(manifest)


def test_generation_result_rejects_artifact_size_change(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_text("1,2,3\n", encoding="utf-8")
    result = GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=11,
        artifacts=(ArtifactRef.from_path("qpos", motion),),
    )
    manifest = tmp_path / "generation.json"
    result.write_json(manifest)
    motion.write_text("1,2,3,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact 'qpos' size mismatch"):
        GenerationResult.read_json(manifest)


def test_generation_result_rejects_same_size_artifact_mutation(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_bytes(b"original")
    result = GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=11,
        artifacts=(ArtifactRef.from_path("qpos", motion),),
    )
    manifest = tmp_path / "generation.json"
    result.write_json(manifest)
    motion.write_bytes(b"mutated!")

    with pytest.raises(ValueError, match="artifact 'qpos' SHA-256 mismatch"):
        GenerationResult.read_json(manifest)


def test_generation_result_bundle_is_relocatable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    motion = bundle / "motion.csv"
    motion.write_text("1,2,3\n", encoding="utf-8")
    result = GenerationResult(
        episode_request_id=_request().request_id,
        generator_name="kimodo",
        generator_version="901a98b",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=11,
        artifacts=(ArtifactRef.from_path("qpos", motion),),
    )
    manifest = bundle / "generation.json"
    result.write_json(manifest)

    relocated = tmp_path / "relocated"
    bundle.rename(relocated)
    loaded = GenerationResult.read_json(relocated / "generation.json")

    assert loaded.result_id == result.result_id
    assert Path(loaded.artifacts[0].path) == (relocated / "motion.csv").resolve()


def test_generation_result_id_excludes_artifact_locator(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text("1,2,3\n", encoding="utf-8")
    second_path.write_bytes(first_path.read_bytes())
    common = {
        "episode_request_id": _request().request_id,
        "generator_name": "kimodo",
        "generator_version": "901a98b",
        "resolved_model": "kimodo-g1-rp",
        "source_fps": 30.0,
        "seed": 11,
    }

    first = GenerationResult(
        **common,
        artifacts=(ArtifactRef.from_path("qpos", first_path),),
    )
    second = GenerationResult(
        **common,
        artifacts=(ArtifactRef.from_path("qpos", second_path),),
    )

    assert first.result_id == second.result_id


def test_generation_result_rejects_duplicate_artifact_names(tmp_path: Path) -> None:
    motion = tmp_path / "motion.csv"
    motion.write_text("1,2,3\n", encoding="utf-8")
    artifact = ArtifactRef.from_path("qpos", motion)

    with pytest.raises(ValueError, match="unique"):
        GenerationResult(
            episode_request_id=_request().request_id,
            generator_name="kimodo",
            generator_version="901a98b",
            resolved_model="kimodo-g1-rp",
            source_fps=30.0,
            seed=11,
            artifacts=(artifact, artifact),
        )


def test_conversion_result_round_trip_verifies_parent_and_relocates(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "kimodo.csv"
    output = bundle / "sonic.pkl"
    source.write_text("1,2,3\n", encoding="utf-8")
    output.write_bytes(b"joblib-placeholder")
    parent = _generation_result(source)
    parent_manifest = bundle / "generation.json"
    parent.write_json(parent_manifest)
    conversion = _conversion_result(source, output, parent)
    manifest = bundle / "conversion.json"
    conversion.write_json(manifest, include_legacy_aliases=True)

    serialized = json.loads(manifest.read_text(encoding="utf-8"))
    assert serialized["kind"] == "kimodo_sonic_conversion_result"
    assert serialized["artifacts"][0]["name"] == "kimodo_qpos_csv"
    assert serialized["artifacts"][0]["path"] == "kimodo.csv"
    assert serialized["artifacts"][1]["path"] == "sonic.pkl"
    assert serialized["input"] == serialized["artifacts"][0]
    assert serialized["output"] == serialized["artifacts"][1]

    relocated = tmp_path / "relocated"
    bundle.rename(relocated)
    loaded_parent = GenerationResult.read_json(relocated / "generation.json")
    loaded = ConversionResult.read_json(relocated / "conversion.json")
    loaded.verify_parent(loaded_parent)

    assert loaded.result_id == conversion.result_id
    assert loaded.scene_frame == "scene_local"
    assert loaded.scene_transform_order == SCENE_TRANSFORM_ORDER
    assert loaded.source_root_quaternion_order == KIMODO_ROOT_QUATERNION_ORDER
    assert loaded.output_root_quaternion_order == SONIC_ROOT_QUATERNION_ORDER
    assert loaded.source_joint_order == G1_MUJOCO_JOINT_ORDER
    assert loaded.output_joint_order == G1_MUJOCO_JOINT_ORDER


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_generation_result_id", SHA_A),
        ("converter_version", "2"),
        ("source_fps", 50.0),
        ("motion_key", "different"),
        ("frame_count", 4),
        ("scene_start_xyz", (1.5, 2.0, 0.25)),
        ("scene_yaw", 0.75),
        ("canonicalize_horizontal_origin", False),
    ],
)
def test_conversion_result_id_covers_parent_and_parameters(
    tmp_path: Path, field: str, value: object
) -> None:
    source = tmp_path / "kimodo.csv"
    output = tmp_path / "sonic.pkl"
    source.write_text("1,2,3\n", encoding="utf-8")
    output.write_bytes(b"joblib-placeholder")
    conversion = _conversion_result(source, output, _generation_result(source))

    changed = replace(conversion, **{field: value})

    assert changed.result_id != conversion.result_id


def test_conversion_result_rejects_tampered_semantics_and_aliases(tmp_path: Path) -> None:
    source = tmp_path / "kimodo.csv"
    output = tmp_path / "sonic.pkl"
    source.write_text("1,2,3\n", encoding="utf-8")
    output.write_bytes(b"joblib-placeholder")
    conversion = _conversion_result(source, output, _generation_result(source))
    manifest = tmp_path / "conversion.json"
    conversion.write_json(manifest, include_legacy_aliases=True)
    original = json.loads(manifest.read_text(encoding="utf-8"))

    tampered = dict(original)
    tampered["scene_yaw"] = 0.75
    with pytest.raises(ValueError, match="conversion_result_id does not match"):
        ConversionResult.from_dict(tampered)

    missing_id = dict(original)
    missing_id.pop("conversion_result_id")
    with pytest.raises(ValueError, match="conversion_result_id is required"):
        ConversionResult.from_dict(missing_id)

    unknown = dict(original)
    unknown["unhashed_note"] = "not allowed"
    with pytest.raises(ValueError, match="unknown ConversionResult fields"):
        ConversionResult.from_dict(unknown)

    bad_alias = dict(original)
    bad_alias["converter"] = "different-converter"
    with pytest.raises(ValueError, match="legacy converter alias"):
        ConversionResult.from_dict(bad_alias)


def test_conversion_result_rejects_mutated_output_artifact(tmp_path: Path) -> None:
    source = tmp_path / "kimodo.csv"
    output = tmp_path / "sonic.pkl"
    source.write_text("1,2,3\n", encoding="utf-8")
    output.write_bytes(b"original")
    conversion = _conversion_result(source, output, _generation_result(source))
    manifest = tmp_path / "conversion.json"
    conversion.write_json(manifest)
    output.write_bytes(b"mutated!")

    with pytest.raises(ValueError, match="sonic_motion_lib.*SHA-256 mismatch"):
        ConversionResult.read_json(manifest)


def test_conversion_result_rejects_invalid_order_and_parent_artifact(tmp_path: Path) -> None:
    source = tmp_path / "kimodo.csv"
    other = tmp_path / "other.csv"
    output = tmp_path / "sonic.pkl"
    source.write_text("1,2,3\n", encoding="utf-8")
    other.write_text("4,5,6\n", encoding="utf-8")
    output.write_bytes(b"joblib-placeholder")
    parent = _generation_result(source)
    conversion = _conversion_result(source, output, parent)

    with pytest.raises(ValueError, match="source_joint_order"):
        replace(conversion, source_joint_order="isaaclab")

    wrong_source = _conversion_result(other, output, parent)
    with pytest.raises(ValueError, match="does not match the selected parent artifact"):
        wrong_source.verify_parent(parent)
