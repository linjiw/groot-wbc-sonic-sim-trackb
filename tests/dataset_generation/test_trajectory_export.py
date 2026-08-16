from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation import trajectory_export as trajectory_export_module
from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    KIMODO_G1_JOINT_NAMES,
)
from gear_sonic.dataset_generation.latent_parity import (
    LatentParityReport,
    check_latent_parity,
    fsq_level_code_values,
)
from gear_sonic.dataset_generation.schemas import (
    ArtifactRef,
    ConversionResult,
    EpisodeRequest,
    GenerationResult,
)
from gear_sonic.dataset_generation.trajectory_acceptance import G1_CONTACT_BODY_NAMES
from gear_sonic.dataset_generation.trajectory_export import (
    EGO_FRAME_SHAPE,
    VideoInfo,
    align_post_step_trajectory_for_bc,
    build_export_provenance,
    build_synthetic_g1_frame,
    convert_trajectory_joint_order_to_mujoco,
    export_sonic_trajectory,
    iter_rgb_video_frames,
    load_upstream_provenance,
    normalize_frame_range,
    require_accepted_locomotion,
    sha256_file,
    slice_trajectory_frames,
    validate_rgb_video,
    validate_robot_joint_contract,
    validate_runtime_artifact_binding,
    validate_upstream_runtime_binding,
)
from scripts.research import export_sonic_trajectory as export_cli
from tests.dataset_generation.test_trajectory_acceptance import _acceptable_trajectory
from tests.dataset_generation.test_trajectory_validation import _trajectory as _raw_trajectory


class StubRobotModel:
    supplemental_info = SimpleNamespace(body_actuated_joints=KIMODO_G1_JOINT_NAMES)

    def get_configuration_from_actuated_joints(
        self,
        *,
        body_actuated_joint_values: np.ndarray,
        left_hand_actuated_joint_values: np.ndarray,
        right_hand_actuated_joint_values: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            (
                body_actuated_joint_values[:22],
                left_hand_actuated_joint_values,
                body_actuated_joint_values[22:],
                right_hand_actuated_joint_values,
            )
        )


def _trajectory() -> dict:
    return {
        "fps": 50.0,
        "dof_pos": np.arange(58, dtype=np.float32).reshape(2, 29),
        "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (2, 1)).astype(np.float32),
        "action_motion_token": np.ones((2, 64), dtype=np.float32),
        "reference_g1_qpos": np.ones((2, 36), dtype=np.float32),
    }


def _long_acceptable_trajectory(frame_count: int = 80) -> dict:
    trajectory = _raw_trajectory(frame_count=frame_count)
    trajectory["root_pos_w"][:, 2] = 0.8
    trajectory["reference_g1_qpos"][:, 2] = 0.8
    route_x = np.linspace(0.0, 2.0, frame_count, dtype=np.float32)
    trajectory["root_pos_w"][:, 0] = route_x
    trajectory["reference_g1_qpos"][:, 0] = route_x
    trajectory["contact_body_names"] = G1_CONTACT_BODY_NAMES
    trajectory["allowed_foot_contact_body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    trajectory["robot_contact_force_w"] = np.zeros(
        (frame_count, len(G1_CONTACT_BODY_NAMES), 3), dtype=np.float32
    )
    for foot_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        trajectory["robot_contact_force_w"][:, G1_CONTACT_BODY_NAMES.index(foot_name), 2] = 50.0
    trajectory["robot_contact_force_norm_w"] = np.linalg.norm(
        trajectory["robot_contact_force_w"], axis=-1
    )
    trajectory["left_foot_ground_contact_force_w"] = np.tile(
        [0.0, 0.0, 50.0], (frame_count, 1)
    ).astype(np.float32)
    trajectory["right_foot_ground_contact_force_w"] = np.tile(
        [0.0, 0.0, 50.0], (frame_count, 1)
    ).astype(np.float32)
    trajectory["support_floor_prim_path"] = "/World/ground/terrain/Structure/Floor"
    trajectory["max_nonfoot_contact_force_n"] = np.zeros(frame_count, dtype=np.float32)
    trajectory["left_foot_contact_force_n"] = np.full(frame_count, 50.0, dtype=np.float32)
    trajectory["right_foot_contact_force_n"] = np.full(frame_count, 50.0, dtype=np.float32)
    return trajectory


def test_builds_minimal_float32_groot_frame() -> None:
    image = np.zeros(EGO_FRAME_SHAPE, dtype=np.uint8)

    frame = build_synthetic_g1_frame(_trajectory(), 1, image, StubRobotModel())

    assert frame["observation.state"].shape == (43,)
    assert frame["observation.state"].dtype == np.float32
    assert frame["action.motion_token"].shape == (64,)
    assert frame["reference.g1_qpos"].shape == (36,)
    assert frame["teleop.left_hand_joints"].shape == (7,)
    assert frame["teleop.right_hand_joints"].shape == (7,)
    assert "timestamp" not in frame


def test_reorders_legacy_isaaclab_state_and_reference_to_mujoco() -> None:
    trajectory = _trajectory()
    trajectory["dof_pos"][0] = np.arange(29, dtype=np.float32)
    trajectory["reference_g1_qpos"][0, :7] = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    trajectory["reference_g1_qpos"][0, 7:] = np.arange(100, 129, dtype=np.float32)

    normalized, evidence = convert_trajectory_joint_order_to_mujoco(trajectory)
    expected_body = np.arange(29, dtype=np.float32)[list(G1_ISAACLAB_TO_MUJOCO_DOF)]
    expected_reference = np.concatenate(
        (
            np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.arange(100, 129, dtype=np.float32)[list(G1_ISAACLAB_TO_MUJOCO_DOF)],
        )
    )

    assert normalized["dof_pos"][0].tolist() == expected_body.tolist()
    assert normalized["reference_g1_qpos"][0].tolist() == expected_reference.tolist()
    assert normalized["dof_order"] == "mujoco"
    assert normalized["dof_joint_names"] == KIMODO_G1_JOINT_NAMES
    assert evidence["source_dof_order"] == "isaaclab"
    assert evidence["source_dof_order_declared"] is False

    frame = build_synthetic_g1_frame(
        trajectory, 0, np.zeros(EGO_FRAME_SHAPE, np.uint8), StubRobotModel()
    )
    exported_body = np.concatenate(
        (frame["observation.state"][:22], frame["observation.state"][29:36])
    )
    assert exported_body.tolist() == expected_body.tolist()
    assert frame["reference.g1_qpos"].tolist() == expected_reference.tolist()


def test_declared_joint_names_are_authoritative_and_must_cover_g1() -> None:
    trajectory = _trajectory()
    source_names = tuple(reversed(KIMODO_G1_JOINT_NAMES))
    trajectory["dof_order"] = "isaaclab"
    trajectory["dof_joint_names"] = source_names
    trajectory["reference_g1_qpos_dof_order"] = "isaaclab"
    trajectory["reference_g1_qpos_joint_names"] = source_names
    trajectory["dof_pos"][0] = np.arange(29, dtype=np.float32)
    trajectory["reference_g1_qpos"][0, 7:] = np.arange(29, dtype=np.float32)

    normalized, evidence = convert_trajectory_joint_order_to_mujoco(trajectory)
    expected_indices = [source_names.index(name) for name in KIMODO_G1_JOINT_NAMES]

    assert normalized["dof_pos"][0].tolist() == expected_indices
    assert normalized["reference_g1_qpos"][0, 7:].tolist() == expected_indices
    assert evidence["source_dof_joint_names_declared"] is True

    trajectory["dof_joint_names"] = source_names[:-1] + ("unknown_joint",)
    with pytest.raises(ValueError, match="each registered G1 body joint"):
        convert_trajectory_joint_order_to_mujoco(trajectory)


def test_rejects_non_contract_camera_frame() -> None:
    bad_image = np.zeros((480, 640, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="ego RGB frame"):
        build_synthetic_g1_frame(_trajectory(), 0, bad_image, StubRobotModel())


def test_rejects_changed_robot_body_joint_order() -> None:
    robot_model = StubRobotModel()
    robot_model.supplemental_info = SimpleNamespace(
        body_actuated_joints=tuple(reversed(KIMODO_G1_JOINT_NAMES))
    )

    with pytest.raises(ValueError, match="joint order"):
        validate_robot_joint_contract(robot_model)


def test_export_gate_requires_accepted_physics_locomotion() -> None:
    report = require_accepted_locomotion(_acceptable_trajectory())

    assert report.accepted

    rejected = _acceptable_trajectory()
    # Lateral push = collision; a purely upward force would be floor support.
    rejected["robot_contact_force_w"][4, 0, 0] = 20.0
    rejected["robot_contact_force_norm_w"][4, 0] = 20.0
    with pytest.raises(ValueError, match="disallowed_robot_contact"):
        require_accepted_locomotion(rejected)


def test_slices_all_nested_frame_aligned_arrays_before_acceptance() -> None:
    source = _long_acceptable_trajectory()
    source["robot_contact_force_w"][:40, 0, 0] = 20.0
    source["robot_contact_force_norm_w"][:40, 0] = 20.0
    source["nested_evidence"] = {
        "frame_values": np.arange(80, dtype=np.float32),
        "frame_list": list(range(80)),
        "fixed_shape": np.arange(3, dtype=np.float32),
    }
    source["partial_tracking_metrics"] = {"unindexed": np.arange(79)}

    with pytest.raises(ValueError, match="disallowed_robot_contact"):
        require_accepted_locomotion(source)

    selected = slice_trajectory_frames(source, start_frame=40, end_frame=80)
    report = require_accepted_locomotion(selected)

    assert report.accepted
    assert selected["total_frames"] == 40
    assert selected["dof_pos"].shape == (40, 29)
    assert selected["tracking_metrics"]["error_joint_pos"].shape == (40,)
    assert selected["motion_time_step"].tolist() == list(range(40, 80))
    assert selected["nested_evidence"]["frame_values"].tolist() == list(range(40, 80))
    assert selected["nested_evidence"]["frame_list"] == list(range(40, 80))
    assert selected["nested_evidence"]["fixed_shape"].tolist() == [0.0, 1.0, 2.0]
    assert "partial_tracking_metrics" not in selected
    assert selected["dropped_unindexed_partial_tracking_metrics"] == ("unindexed",)
    assert source["total_frames"] == 80


def test_causal_alignment_pairs_post_step_observation_with_next_action() -> None:
    trajectory = _long_acceptable_trajectory(frame_count=42)
    trajectory["recording_phase"] = "post_physics_step"
    trajectory["policy_action_phase"] = "pre_physics_step_policy_output"
    trajectory["dof_pos"][:, 0] = np.arange(42, dtype=np.float32) + 10.0
    trajectory["reference_g1_qpos"][:, 0] = np.arange(42, dtype=np.float32) + 20.0
    trajectory["action_motion_token"][:, 0] = np.arange(42, dtype=np.float32) + 100.0
    trajectory["applied_joint_action"][:, 0] = np.arange(42, dtype=np.float32) + 200.0

    aligned, evidence = align_post_step_trajectory_for_bc(trajectory)

    assert aligned["total_frames"] == 41
    assert aligned["dof_pos"][:, 0].tolist() == list(np.arange(41) + 10.0)
    assert aligned["reference_g1_qpos"][:, 0].tolist() == list(np.arange(41) + 20.0)
    assert aligned["action_motion_token"][:, 0].tolist() == list(np.arange(41) + 101.0)
    assert aligned["applied_joint_action"][:, 0].tolist() == list(np.arange(41) + 201.0)
    assert aligned["motion_time_step"].tolist() == list(range(41))
    assert evidence["observation_indices_within_selected_interval"] == [0, 41]
    assert evidence["action_indices_within_selected_interval"] == [1, 42]
    assert evidence["dropped_boundary_frames"] == 1


def test_causal_alignment_rejects_unknown_phase() -> None:
    trajectory = _long_acceptable_trajectory(frame_count=42)
    trajectory["recording_phase"] = "pre_physics_step"

    with pytest.raises(ValueError, match="recording_phase"):
        align_post_step_trajectory_for_bc(trajectory)


@pytest.mark.parametrize(
    ("start_frame", "end_frame", "message"),
    [
        (-1, 40, "non-negative"),
        (20, 20, "non-empty"),
        (20, 81, "exceeds"),
        (True, 40, "integer"),
    ],
)
def test_rejects_invalid_source_frame_ranges(
    start_frame: int, end_frame: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_frame_range(80, start_frame, end_frame)


def test_video_subclip_decodes_full_source_and_yields_only_selection(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.full(EGO_FRAME_SHAPE, frame_index, dtype=np.uint8) for frame_index in range(5)
            ]
            self.read_count = 0
            self.released = False

        def read(self):
            self.read_count += 1
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda path: capture,
        COLOR_BGR2RGB=1,
        cvtColor=lambda frame, conversion: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        trajectory_export_module,
        "probe_video",
        lambda path: VideoInfo(fps=50.0, frame_count=5, width=640, height=480),
    )

    frames = list(
        iter_rgb_video_frames(
            "source.mp4",
            expected_frames=5,
            expected_fps=50.0,
            start_frame=1,
            end_frame=4,
        )
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == [1, 2, 3]
    assert capture.read_count == 6
    assert capture.released


def test_eager_video_preflight_exhausts_decoder(monkeypatch) -> None:
    captured = {"decoded": 0}

    def frames(*args, **kwargs):
        for _ in range(5):
            captured["decoded"] += 1
            yield np.zeros(EGO_FRAME_SHAPE, dtype=np.uint8)

    monkeypatch.setattr(trajectory_export_module, "iter_rgb_video_frames", frames)
    monkeypatch.setattr(
        trajectory_export_module,
        "probe_video",
        lambda path: VideoInfo(fps=50.0, frame_count=5, width=640, height=480),
    )

    info = validate_rgb_video("source.mp4", expected_frames=5, expected_fps=50.0)

    assert captured["decoded"] == 5
    assert info.frame_count == 5


def test_export_evaluates_selected_clip_and_validates_full_video_count(
    tmp_path: Path, monkeypatch
) -> None:
    source = _long_acceptable_trajectory()
    source["max_nonfoot_contact_force_n"][:40] = 20.0
    trajectory_path = tmp_path / "trajectory.pkl"
    video_path = tmp_path / "ego.mp4"
    trajectory_path.write_bytes(b"trajectory")
    video_path.write_bytes(b"video")
    captured: dict = {"frames": 0}

    class FakeExporter:
        def add_frame(self, frame) -> None:
            captured["frames"] += 1

        def save_episode(self) -> None:
            captured["saved"] = True

        def close(self) -> None:
            captured["closed"] = True

        def cancel_video_writers(self) -> None:
            captured["cancelled"] = True

    class FakeExporterFactory:
        @staticmethod
        def create(**kwargs):
            captured["create_kwargs"] = kwargs
            return FakeExporter()

    exporter_module = ModuleType("gear_sonic.data.exporter")
    exporter_module.Gr00tDataExporter = FakeExporterFactory
    features_module = ModuleType("gear_sonic.data.features_sonic_vla")
    features_module.get_features_synthetic_g1 = lambda robot_model: {}
    features_module.get_modality_config_synthetic_g1 = lambda robot_model: {}
    features_module.get_g1_robot_model = lambda: StubRobotModel()
    monkeypatch.setitem(sys.modules, "gear_sonic.data.exporter", exporter_module)
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.data.features_sonic_vla",
        features_module,
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "load_validated_trajectory",
        lambda path: source,
    )

    def accept_selected(trajectory):
        captured["accepted_trajectory"] = trajectory
        return SimpleNamespace(to_dict=lambda: {"accepted": True})

    def selected_frames(path, **kwargs):
        captured["video_kwargs"] = kwargs
        image = np.zeros(EGO_FRAME_SHAPE, dtype=np.uint8)
        return (image for _ in range(39))

    monkeypatch.setattr(
        trajectory_export_module,
        "require_accepted_locomotion",
        accept_selected,
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "iter_rgb_video_frames",
        selected_frames,
    )
    monkeypatch.setattr(
        trajectory_export_module, "validate_rgb_video", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "validate_runtime_artifact_binding",
        lambda *args, **kwargs: {"capture_episode_id": "000000"},
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "load_upstream_provenance",
        lambda *args, **kwargs: {"typed_lineage": {"conversion_result_id": "conversion"}},
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "validate_upstream_runtime_binding",
        lambda *args, **kwargs: {"conversion_result_id": "conversion"},
    )
    monkeypatch.setattr(
        trajectory_export_module,
        "build_synthetic_g1_frame",
        lambda trajectory, frame_index, rgb_frame, robot_model: {"frame": frame_index},
    )

    exported_frames = export_sonic_trajectory(
        trajectory_path=trajectory_path,
        ego_video_path=video_path,
        output_path=tmp_path / "dataset",
        task="walk through the room",
        camera_provenance="isaac_sim",
        runtime_manifest_path=tmp_path / "success.json",
        scene_id="household_room",
        scene_hash="sha256:scene",
        upstream_provenance_path=tmp_path / "conversion.json",
        start_frame=40,
        end_frame=80,
    )

    accepted = captured["accepted_trajectory"]
    assert exported_frames == 39
    assert accepted["total_frames"] == 39
    assert accepted["motion_time_step"][0] == 40
    assert not np.any(accepted["max_nonfoot_contact_force_n"])
    assert captured["video_kwargs"] == {
        "expected_frames": 80,
        "expected_fps": 50.0,
        "start_frame": 40,
        "end_frame": 79,
    }
    assert captured["frames"] == 39
    assert captured["saved"] is True
    assert captured["closed"] is True


def test_provenance_embeds_selected_range_and_upstream_json(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    video_path = tmp_path / "ego.mp4"
    upstream_path = tmp_path / "episode_request.json"
    trajectory_path.write_bytes(b"trajectory")
    video_path.write_bytes(b"video")
    upstream_path.write_text(
        json.dumps({"episode_request_id": "request-001", "kimodo_seed": 7}) + "\n",
        encoding="utf-8",
    )
    trajectory = _long_acceptable_trajectory()
    selected = slice_trajectory_frames(trajectory, start_frame=20, end_frame=70)
    acceptance = require_accepted_locomotion(selected)
    aligned, alignment_evidence = align_post_step_trajectory_for_bc(selected)
    exported_acceptance = require_accepted_locomotion(aligned)

    provenance = build_export_provenance(
        trajectory_path=trajectory_path,
        ego_video_path=video_path,
        trajectory=selected,
        source_total_frames=80,
        start_frame=20,
        end_frame=70,
        camera_provenance="isaac_sim",
        scene_id="household_room",
        scene_hash="sha256:scene",
        acceptance=acceptance,
        exported_observation_acceptance=exported_acceptance,
        upstream_provenance_path=upstream_path,
        causal_alignment_evidence=alignment_evidence,
    )

    assert provenance["source_total_frames"] == 80
    assert provenance["validated_source_frame_range"] == {
        "start": 20,
        "end_exclusive": 70,
    }
    assert provenance["observation_source_frame_range"] == {
        "start": 20,
        "end_exclusive": 69,
    }
    assert provenance["action_source_frame_range"] == {
        "start": 21,
        "end_exclusive": 70,
    }
    assert provenance["locomotion_acceptance"]["accepted"] is True
    assert provenance["exported_observation_acceptance"]["accepted"] is True
    assert provenance["causal_alignment"]["dropped_boundary_frames"] == 1
    assert provenance["upstream_provenance"] == load_upstream_provenance(upstream_path)
    assert provenance["upstream_provenance"]["hash"] == sha256_file(upstream_path)

    latent = provenance["latent_representation"]
    assert latent["total_dim"] == 64
    assert latent["representation"] == "post_quantization_fsq_codes"
    assert latent["flatten_order"] == "row_major_over_(num_tokens, token_dim)"
    # The recorded representation must be stated, never left implicit.
    assert latent["parity_report"]["verdict"] in {
        LatentParityReport.ENCODER_EQUIVALENT,
        LatentParityReport.RESIDUAL_PERTURBED,
    }
    # It must describe the rows actually exported: the causally shifted actions.
    assert latent["parity_report"]["frame_count"] == len(aligned["action_motion_token"])
    assert latent["parity_report"] == check_latent_parity(aligned["action_motion_token"]).to_dict()


def test_provenance_latent_block_distinguishes_encoder_tokens_from_a_residual(
    tmp_path: Path,
) -> None:
    """The two 64D conventions must not both silently export as the same feature."""
    trajectory_path = tmp_path / "trajectory.pkl"
    video_path = tmp_path / "ego.mp4"
    trajectory_path.write_bytes(b"trajectory")
    video_path.write_bytes(b"video")

    codes = fsq_level_code_values(32)
    rng = np.random.default_rng(11)

    def _provenance_for(motion_token: np.ndarray) -> dict:
        trajectory = _long_acceptable_trajectory()
        trajectory["action_motion_token"] = motion_token
        selected = slice_trajectory_frames(trajectory, start_frame=20, end_frame=70)
        aligned, evidence = align_post_step_trajectory_for_bc(selected)
        return build_export_provenance(
            trajectory_path=trajectory_path,
            ego_video_path=video_path,
            trajectory=selected,
            source_total_frames=80,
            start_frame=20,
            end_frame=70,
            camera_provenance="isaac_sim",
            scene_id="household_room",
            scene_hash="sha256:scene",
            acceptance=require_accepted_locomotion(selected),
            exported_observation_acceptance=require_accepted_locomotion(aligned),
            upstream_provenance_path=None,
            causal_alignment_evidence=evidence,
        )

    encoder_tokens = rng.choice(codes, size=(80, 64)).astype(np.float32)
    encoder_report = _provenance_for(encoder_tokens)["latent_representation"]["parity_report"]
    assert encoder_report["verdict"] == LatentParityReport.ENCODER_EQUIVALENT
    assert encoder_report["residual_rms"] == pytest.approx(0.0, abs=1e-9)

    residual_tokens = (encoder_tokens + rng.normal(scale=0.2, size=(80, 64))).astype(np.float32)
    residual_report = _provenance_for(residual_tokens)["latent_representation"]["parity_report"]
    assert residual_report["verdict"] == LatentParityReport.RESIDUAL_PERTURBED
    assert residual_report["residual_rms"] > 0.0


def test_runtime_manifest_rejects_a_video_from_another_episode(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    video_path = tmp_path / "video.mp4"
    other_video_path = tmp_path / "other.mp4"
    scene_path = tmp_path / "factory_aisle.usda"
    motion_path = tmp_path / "motion.pkl"
    manifest_path = tmp_path / "success.json"
    trajectory_path.write_bytes(b"trajectory")
    video_path.write_bytes(b"video-a")
    other_video_path.write_bytes(b"video-b")
    scene_path.write_bytes(b"scene")
    motion_path.write_bytes(b"motion")
    scene_hash = sha256_file(scene_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "sonic_isaac_eval_success",
                "status": "success",
                "exit_reason": "max_render_steps",
                "physics_steps": 80,
                "policy_iterations": 81,
                "capture_context": {
                    "terrain_type": "scene_usd",
                    "scene_id": "factory_aisle",
                    "scene": {
                        "resolved": str(scene_path),
                        "hash": scene_hash,
                    },
                    "task": "walk through the aisle",
                    "motion": {
                        "resolved": str(motion_path),
                        "hash": sha256_file(motion_path),
                    },
                    "use_encoder": "g1",
                    "camera": {
                        "provenance": "isaac_sim",
                        "name": "ego_camera",
                        "track_root": False,
                        "render_ego": True,
                        "resolution": [480, 640],
                        "render_frame_skip": 1,
                    },
                },
                "artifact_pairs": {
                    "000000": {
                        "trajectory_hash": sha256_file(trajectory_path),
                        "video_hash": sha256_file(video_path),
                        "trajectory_path": str(trajectory_path),
                        "video_path": str(video_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_runtime_artifact_binding(
        manifest_path,
        trajectory_path=trajectory_path,
        ego_video_path=video_path,
        scene_id="factory_aisle",
        scene_hash=scene_hash,
        camera_provenance="isaac_sim",
        task="walk through the aisle",
    )
    assert evidence["capture_episode_id"] == "000000"

    with pytest.raises(ValueError, match="not one uniquely bound pair"):
        validate_runtime_artifact_binding(
            manifest_path,
            trajectory_path=trajectory_path,
            ego_video_path=other_video_path,
            scene_id="factory_aisle",
            scene_hash=scene_hash,
            camera_provenance="isaac_sim",
            task="walk through the aisle",
        )

    with pytest.raises(ValueError, match="scene_id"):
        validate_runtime_artifact_binding(
            manifest_path,
            trajectory_path=trajectory_path,
            ego_video_path=video_path,
            scene_id="household_room",
            scene_hash=scene_hash,
            camera_provenance="isaac_sim",
            task="walk through the aisle",
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["capture_context"]["camera"]["name"] = "eval_camera"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ego dataset contract"):
        validate_runtime_artifact_binding(
            manifest_path,
            trajectory_path=trajectory_path,
            ego_video_path=video_path,
            scene_id="factory_aisle",
            scene_hash=scene_hash,
            camera_provenance="isaac_sim",
            task="walk through the aisle",
        )


def test_rejects_non_object_upstream_provenance(tmp_path: Path) -> None:
    upstream_path = tmp_path / "invalid.json"
    upstream_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level object"):
        load_upstream_provenance(upstream_path)


def test_typed_upstream_bundle_binds_request_and_motion_to_runtime(tmp_path: Path) -> None:
    qpos_path = tmp_path / "motion.csv"
    motion_path = tmp_path / "motion.pkl"
    qpos_path.write_bytes(b"qpos")
    motion_path.write_bytes(b"motion")
    scene_hash = "sha256:" + "a" * 64
    checkpoint_hash = "sha256:" + "b" * 64
    physics_hash = "sha256:" + "c" * 64
    request = EpisodeRequest(
        scene_id="factory_aisle",
        task_family="locomotion",
        task_prompt="walk through the aisle",
        style_prompt="",
        route_xy=((-7.0, 0.0), (-4.0, 0.0)),
        nominal_speed_mps=0.7,
        duration_s=5.0,
        kimodo_model="kimodo-g1-rp",
        kimodo_seed=43,
        simulation_seed=0,
        render_seed=0,
        candidate_index=0,
        scene_hash=scene_hash,
        controller_hash=checkpoint_hash,
        physics_hash=physics_hash,
    )
    request.write_json(tmp_path / "episode_request.json")
    generation = GenerationResult(
        episode_request_id=request.request_id,
        generator_name="Kimodo",
        generator_version="git:abc",
        resolved_model="kimodo-g1-rp",
        source_fps=30.0,
        seed=43,
        artifacts=(ArtifactRef.from_path("kimodo_qpos_csv", qpos_path, media_type="text/csv"),),
    )
    generation.write_json(tmp_path / "generation_result.json")
    conversion = ConversionResult(
        episode_request_id=request.request_id,
        source_generation_result_id=generation.result_id,
        source_artifact_name="kimodo_qpos_csv",
        converter_name="kimodo_motion_adapter",
        converter_version="1",
        source_fps=30.0,
        motion_key="walk",
        frame_count=50,
        scene_start_xyz=(-7.0, 0.0, 0.0),
        scene_yaw=0.0,
        canonicalize_horizontal_origin=True,
        artifacts=(
            ArtifactRef.from_path("kimodo_qpos_csv", qpos_path, media_type="text/csv"),
            ArtifactRef.from_path(
                "sonic_motion_lib", motion_path, media_type="application/x-joblib"
            ),
        ),
    )
    conversion.write_json(tmp_path / "conversion_result.json")

    upstream = load_upstream_provenance(tmp_path / "conversion_result.json")
    runtime = {
        "checkpoint_hash": checkpoint_hash,
        "runtime_config_hash": physics_hash,
        "capture_context": {"motion": {"hash": ArtifactRef.from_path("m", motion_path).sha256}},
    }
    binding = validate_upstream_runtime_binding(
        upstream,
        runtime,
        scene_id="factory_aisle",
        scene_hash=scene_hash,
        task="walk through the aisle",
    )

    assert binding["episode_request_id"] == request.request_id
    assert binding["generation_result_id"] == generation.result_id
    assert binding["conversion_result_id"] == conversion.result_id
    with pytest.raises(ValueError, match="EpisodeRequest"):
        validate_upstream_runtime_binding(
            upstream,
            runtime,
            scene_id="factory_aisle",
            scene_hash=scene_hash,
            task="a different task",
        )


def test_cli_forwards_subclip_and_upstream_provenance(tmp_path: Path, monkeypatch, capsys) -> None:
    captured: dict = {}

    def fake_export(**kwargs) -> int:
        captured.update(kwargs)
        return 40

    trajectory_path = tmp_path / "trajectory.pkl"
    video_path = tmp_path / "ego.mp4"
    output_path = tmp_path / "dataset"
    upstream_path = tmp_path / "upstream.json"
    monkeypatch.setattr(export_cli, "export_sonic_trajectory", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_sonic_trajectory.py",
            str(trajectory_path),
            str(video_path),
            str(output_path),
            "--task",
            "walk through the room",
            "--camera-provenance",
            "isaac_sim",
            "--runtime-success-manifest",
            str(tmp_path / "success.json"),
            "--scene-id",
            "household_room",
            "--scene-hash",
            "sha256:scene",
            "--start-frame",
            "25",
            "--end-frame",
            "65",
            "--upstream-provenance-json",
            str(upstream_path),
        ],
    )

    assert export_cli.main() == 0
    assert captured["start_frame"] == 25
    assert captured["end_frame"] == 65
    assert captured["upstream_provenance_path"] == upstream_path
    assert captured["runtime_manifest_path"] == tmp_path / "success.json"
    assert captured["scene_id"] == "household_room"
    assert captured["scene_hash"] == "sha256:scene"
    assert "frames=40" in capsys.readouterr().out
