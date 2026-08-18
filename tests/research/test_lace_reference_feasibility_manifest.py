from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.research.lace.reference_feasibility import (
    REFERENCE_FEASIBILITY_SCOPE,
)
from gear_sonic.research.lace.reference_feasibility_manifest import (
    G1_REFERENCE_CONTRACT_DIGEST_FIELD,
    LIVE_ROBOT_EVIDENCE_BINDING_KIND,
    REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD,
    _derive_motion_reference_contract,
    _foot_positions_from_mjcf,
    _runtime_reference,
    build_reference_feasibility_manifest,
    derive_g1_reference_contract,
    load_live_robot_evidence,
    validate_g1_reference_contract,
    validate_reference_feasibility_manifest,
)
from gear_sonic.research.lace.reference_lengths import (
    build_reference_length_inventory,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_CONFIG = REPO_ROOT / "gear_sonic/envs/manager_env/robots/g1.py"
G1_URDF = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
MOTION_COMMAND_CONFIG = REPO_ROOT / "gear_sonic/config/manager_env/commands/terms/motion.yaml"
MOTION_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_motion(path: Path, motion_key: str, *, frames: int = 8) -> None:
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 2] = 0.8
    root_quaternion = np.zeros((frames, 4), dtype=np.float32)
    root_quaternion[:, 3] = 1.0
    payload = {
        "root_trans_offset": root,
        "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "root_rot": root_quaternion,
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "fps": 30,
    }
    joblib.dump({motion_key: payload}, path, compress=True)


def _scientific_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    cohort_path = tmp_path / "cohort.json"
    cohort_motions = []
    split_inputs = []
    for index in range(30):
        motion_key = f"motion_{index:03d}__A{index:03d}"
        path = motion_root / f"{motion_key}.pkl"
        cohort_motions.append(
            {
                "motion_key": motion_key,
                "release_filter_key": f"release/{motion_key}.pkl",
            }
        )
        split_inputs.append(
            {
                "motion_key": motion_key,
                "robot_path": str(path),
                "duration_source_frames": 8,
                "stratum": f"kind_{index % 3}",
            }
        )
    cohort = {
        "kind": "bones_seed_official_metadata_cohort",
        "schema_version": 1,
        "eligibility": {
            "rule": "official_release_filename_filter",
            "filter_keywords": ["chair", "handstand"],
        },
        "motions": cohort_motions,
    }
    cohort_path.write_text(json.dumps(cohort, sort_keys=True), encoding="utf-8")
    split = build_source_disjoint_split(
        split_inputs,
        seed=23,
        dataset={
            "cohort_manifest": str(cohort_path),
            "cohort_manifest_sha256": _file_sha256(cohort_path),
        },
    )
    atlas_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    for motion_key in atlas_keys:
        _write_motion(motion_root / f"{motion_key}.pkl", motion_key)

    # An invalid non-selected file proves the builder does not scan or load the directory.
    (motion_root / "unselected_corrupt.pkl").write_bytes(b"not-a-joblib-file")
    inventory = build_reference_length_inventory(split)
    split_path = tmp_path / "split.json"
    inventory_path = tmp_path / "reference_lengths.json"
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    return split_path, inventory_path, cohort_path, split, inventory


def _build_fixture_manifest(tmp_path: Path, *, live: bool = False) -> tuple[dict, dict, dict]:
    split_path, inventory_path, cohort_path, split, inventory = _scientific_fixture(tmp_path)
    live_robot_data = None
    if live:
        cpu_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
        live_robot_data = _recorder_readback(cpu_contract)
    manifest = build_reference_feasibility_manifest(
        split_manifest_path=split_path,
        reference_inventory_path=inventory_path,
        cohort_manifest_path=cohort_path,
        g1_config_path=G1_CONFIG,
        urdf_path=G1_URDF,
        motion_command_config_path=MOTION_COMMAND_CONFIG,
        motion_mjcf_path=MOTION_MJCF,
        live_robot_data=live_robot_data,
    )
    return manifest, split, inventory


def _recorder_readback(robot_contract: dict) -> dict:
    expected = robot_contract["expected_live_robot_data"]
    limits = {}
    for field in (
        "joint_pos_limits",
        "soft_joint_pos_limits",
        "joint_vel_limits",
        "soft_joint_vel_limits",
    ):
        values = np.asarray(expected[field], dtype=np.float64)
        limits[field] = {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "values": values.tolist(),
        }
    readback = {
        "kind": "lace_robot_contract_readback",
        "schema_version": 1,
        "capture_lifecycle": (
            "RecorderTerm.record_post_reset_after_scene_reset_events_action_reset_and_command_reset"
        ),
        "ordered_body_names": robot_contract["isaaclab_body_names_with_root"],
        "ordered_joint_names": expected["joint_names"],
        "source_properties": {
            "joint_names": "robot.joint_names",
            "joint_pos_limits": "robot.data.joint_pos_limits[0]",
            "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
            "joint_vel_limits": "robot.data.joint_vel_limits[0]",
            "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
        },
        "limits": limits,
        "environment_invariance_verified": True,
    }
    return {
        "robot_contract_readback": readback,
        "robot_contract_readback_sha256": canonical_sha256(readback),
    }


def test_g1_contract_derives_all_orders_limits_and_explicitly_defers_live_check() -> None:
    first = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    second = derive_g1_reference_contract(G1_CONFIG, G1_URDF)

    assert first == second
    assert first["dof_count"] == 29
    assert len(first["isaaclab_joint_names"]) == 29
    assert set(first["isaaclab_joint_names"]) == set(first["mujoco_source_joint_names"])
    assert first["g1_config"]["sha256"] == _file_sha256(G1_CONFIG)
    assert first["urdf"]["sha256"] == _file_sha256(G1_URDF)
    assert first["live_robot_data_validation"] == {
        "verified": False,
        "status": "unverified_no_isaac_launch",
        "source_properties": {
            "joint_names": "robot.joint_names",
            "joint_pos_limits": "robot.data.joint_pos_limits[0]",
            "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
            "joint_vel_limits": "robot.data.joint_vel_limits[0]",
            "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
        },
        "required_follow_up": (
            "launch the pinned G1 articulation and compare all ordered robot.data tensors"
        ),
    }
    assert first[G1_REFERENCE_CONTRACT_DIGEST_FIELD] == canonical_sha256(
        first,
        digest_field=G1_REFERENCE_CONTRACT_DIGEST_FIELD,
    )
    validate_g1_reference_contract(first, verify_source_files=True)


def test_supplied_live_readback_must_match_every_ordered_limit_tensor() -> None:
    cpu_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    expected = _recorder_readback(cpu_contract)

    verified = derive_g1_reference_contract(G1_CONFIG, G1_URDF, live_robot_data=expected)
    assert verified["live_robot_data_validation"]["verified"] is True

    expected["robot_contract_readback"]["limits"]["joint_vel_limits"]["values"][0] += 0.1
    expected["robot_contract_readback_sha256"] = canonical_sha256(
        expected["robot_contract_readback"]
    )
    with pytest.raises(ValueError, match="joint_vel_limits"):
        derive_g1_reference_contract(G1_CONFIG, G1_URDF, live_robot_data=expected)


def test_live_robot_evidence_accepts_one_json_object_and_binds_source(tmp_path: Path) -> None:
    cpu_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    row = {
        **_recorder_readback(cpu_contract),
        "scientific_runtime_ready": True,
    }
    evidence_path = tmp_path / "recorder.json"
    evidence_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    evidence = load_live_robot_evidence(evidence_path)

    assert evidence["robot_contract_readback"] == row["robot_contract_readback"]
    assert evidence["robot_contract_readback_sha256"] == row["robot_contract_readback_sha256"]
    assert evidence["recorder_evidence_binding"] == {
        "format": "json_object",
        "kind": LIVE_ROBOT_EVIDENCE_BINDING_KIND,
        "path": str(evidence_path.resolve()),
        "row_count": 1,
        "schema_version": 1,
        "scientific_runtime_ready_all_rows": True,
        "sha256": _file_sha256(evidence_path),
    }
    verified = derive_g1_reference_contract(G1_CONFIG, G1_URDF, live_robot_data=evidence)
    assert (
        verified["live_robot_data_validation"]["recorder_evidence_binding"]
        == evidence["recorder_evidence_binding"]
    )
    validate_g1_reference_contract(verified, verify_source_files=True)

    evidence_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recorder evidence source SHA-256 mismatch"):
        validate_g1_reference_contract(verified, verify_source_files=True)


def test_live_robot_evidence_accepts_homogeneous_nonempty_jsonl(tmp_path: Path) -> None:
    cpu_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    pair = _recorder_readback(cpu_contract)
    rows = [
        {
            "motion_key": "motion_a",
            **pair,
            "scientific_runtime_ready": True,
        },
        {
            "scientific_runtime_ready": True,
            **deepcopy(pair),
            "motion_key": "motion_b",
        },
    ]
    evidence_path = tmp_path / "recorder.jsonl"
    evidence_path.write_text(
        "".join(f"{json.dumps(row, sort_keys=index == 0)}\n" for index, row in enumerate(rows)),
        encoding="utf-8",
    )

    evidence = load_live_robot_evidence(evidence_path)

    assert evidence["robot_contract_readback"] == pair["robot_contract_readback"]
    assert evidence["recorder_evidence_binding"]["format"] == "jsonl"
    assert evidence["recorder_evidence_binding"]["row_count"] == 2
    assert evidence["recorder_evidence_binding"]["scientific_runtime_ready_all_rows"] is True
    assert evidence["recorder_evidence_binding"]["sha256"] == _file_sha256(evidence_path)


def test_live_robot_jsonl_rejects_mixed_malformed_empty_missing_and_bad_digest(
    tmp_path: Path,
) -> None:
    cpu_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    pair = _recorder_readback(cpu_contract)
    valid_row = {**pair, "scientific_runtime_ready": True}

    empty = tmp_path / "empty.jsonl"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        load_live_robot_evidence(empty)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(f"{json.dumps(valid_row)}\n{{not json}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2.*malformed JSON"):
        load_live_robot_evidence(malformed)

    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        f"{json.dumps(valid_row)}\n{json.dumps({'scientific_runtime_ready': True})}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"line 2.*missing required fields"):
        load_live_robot_evidence(missing)

    bad_digest_row = deepcopy(valid_row)
    bad_digest_row["robot_contract_readback_sha256"] = "0" * 64
    bad_digest = tmp_path / "bad-digest.jsonl"
    bad_digest.write_text(
        f"{json.dumps(valid_row)}\n{json.dumps(bad_digest_row)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"line 2.*readback_sha256 mismatch"):
        load_live_robot_evidence(bad_digest)

    mixed_row = deepcopy(valid_row)
    mixed_row["robot_contract_readback"]["ordered_joint_names"] = list(
        reversed(mixed_row["robot_contract_readback"]["ordered_joint_names"])
    )
    mixed_row["robot_contract_readback_sha256"] = canonical_sha256(
        mixed_row["robot_contract_readback"]
    )
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        f"{json.dumps(valid_row)}\n{json.dumps(mixed_row)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixed, non-byte-equivalent"):
        load_live_robot_evidence(mixed)

    blank_row = tmp_path / "blank-row.jsonl"
    blank_row.write_text(f"{json.dumps(valid_row)}\n\n{json.dumps(valid_row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contain no blank rows"):
        load_live_robot_evidence(blank_row)


def test_cpu_reference_math_matches_pinned_sonic_motion_library() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("open3d")
    easydict = pytest.importorskip("easydict")
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    robot_contract = derive_g1_reference_contract(G1_CONFIG, G1_URDF)
    _, body_records = _derive_motion_reference_contract(
        MOTION_COMMAND_CONFIG,
        MOTION_MJCF,
        robot_contract=robot_contract,
    )
    frames = 8
    time = np.arange(frames, dtype=np.float32) / np.float32(30.0)
    pose = np.zeros((frames, 30, 3), dtype=np.float32)
    pose[:, 0, 2] = 0.3 * time
    axes = np.asarray(
        [record["joint_axis"] for record in body_records[1:]],
        dtype=np.float32,
    )
    dof = np.column_stack([0.2 * np.sin((index + 1) * time) for index in range(29)]).astype(
        np.float32
    )
    pose[:, 1:, :] = dof[:, :, None] * axes[None, :, :]
    root = np.column_stack((0.1 * time, -0.05 * time, np.full_like(time, 0.8))).astype(np.float32)
    target_frames = 12
    joints, runtime_root, runtime_quaternion, local_rotations = _runtime_reference(
        pose_rotation_vector=pose,
        root_position=root,
        source_fps=Fraction(30, 1),
        target_fps=50,
        target_num_frames=target_frames,
    )
    feet = _foot_positions_from_mjcf(
        body_records=body_records,
        local_rotation_matrices=local_rotations,
        root_position=runtime_root,
        root_quaternion_xyzw=runtime_quaternion,
    )

    cfg = easydict.EasyDict(
        asset=easydict.EasyDict(
            assetRoot=str(MOTION_MJCF.parent),
            assetFileName=MOTION_MJCF.name,
        ),
        extend_config=[],
    )
    runtime = Humanoid_Batch(cfg, device=torch.device("cpu"))
    expected = runtime.fk_batch(
        torch.from_numpy(pose)[None],
        torch.from_numpy(root)[None],
        return_full=True,
        fps=30,
        target_fps=50,
        interpolate_data=True,
    )
    foot_indices = [
        runtime.body_names.index("left_ankle_roll_link"),
        runtime.body_names.index("right_ankle_roll_link"),
    ]
    np.testing.assert_allclose(joints, expected.dof_pos[0].numpy(), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        runtime_root,
        expected.global_translation[0, :, 0].numpy(),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        runtime_quaternion,
        expected.global_rotation[0, :, 0].numpy(),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        feet,
        expected.global_translation[0, :, foot_indices].numpy(),
        rtol=0.0,
        atol=3e-7,
    )


def test_scientific_builder_is_deterministic_selected_only_and_provisional(tmp_path: Path) -> None:
    first, split, inventory = _build_fixture_manifest(tmp_path)
    split_path = Path(first["split_binding"]["path"])
    inventory_path = Path(first["reference_inventory_binding"]["path"])
    cohort_path = Path(first["cohort_binding"]["path"])
    second = build_reference_feasibility_manifest(
        split_manifest_path=split_path,
        reference_inventory_path=inventory_path,
        cohort_manifest_path=cohort_path,
        g1_config_path=G1_CONFIG,
        urdf_path=G1_URDF,
        motion_command_config_path=MOTION_COMMAND_CONFIG,
        motion_mjcf_path=MOTION_MJCF,
    )

    assert first == second
    assert first["scope"] == REFERENCE_FEASIBILITY_SCOPE
    assert first["dynamic_feasibility_proof"] is False
    assert first["artifact_mode"] == "scientific"
    assert first["scientific_use"] is False
    assert first["scientific_use_status"] == (
        "provisional_pending_live_robot_data_velocity_and_order_validation"
    )
    assert first["selected_motion_keys"] == inventory["selected_motion_keys"]
    assert set(first["selected_motion_keys"]) == {
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    }
    assert first["motion_count"] == len(first["motions"])
    assert first["hard_corruption_exclusions_are_separate_from_continuous_features"] is True
    for record in first["motions"]:
        proxy = record["proxy"]
        assert proxy["hard_corruption_exclusion"] is False
        assert proxy["hard_corruption_checks"]["source_quaternion_norm_violation"] is False
        assert proxy["hard_corruption_checks"]["reference_contact_kinematic_rule_mismatch"] is False
        assert proxy["reference_contact_contract"]["contact_labels_are_kinematically_derived"]
        assert not proxy["reference_contact_contract"][
            "independent_physical_contact_labels_available"
        ]
        assert len(proxy["continuous_feature_names"]) == len(proxy["continuous_feature_vector"])
    assert first[REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD] == canonical_sha256(
        first,
        digest_field=REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD,
    )
    validate_reference_feasibility_manifest(
        first,
        split_manifest=split,
        reference_inventory=inventory,
        verify_source_files=True,
        verify_robot_sources=True,
        verify_motion_sources=True,
        verify_implementation_sources=True,
    )


def test_live_verified_scientific_manifest_becomes_ready(tmp_path: Path) -> None:
    manifest, _, _ = _build_fixture_manifest(tmp_path, live=True)

    assert manifest["scientific_use"] is True
    assert manifest["scientific_use_status"] == "ready"


def test_manifest_and_robot_contract_tampering_fail_closed(tmp_path: Path) -> None:
    manifest, _, _ = _build_fixture_manifest(tmp_path)

    tampered_proxy = deepcopy(manifest)
    tampered_proxy["motions"][0]["proxy"]["continuous_feature_vector"][0] += 1.0
    with pytest.raises(ValueError, match="reference feasibility digest"):
        validate_reference_feasibility_manifest(tampered_proxy)

    tampered_live = deepcopy(manifest["robot_contract"])
    tampered_live["expected_live_robot_data"]["joint_vel_limits"][0] += 1.0
    tampered_live[G1_REFERENCE_CONTRACT_DIGEST_FIELD] = canonical_sha256(
        tampered_live,
        digest_field=G1_REFERENCE_CONTRACT_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="expected live joint_vel_limits"):
        validate_g1_reference_contract(tampered_live)

    source_path = Path(manifest["motions"][0]["source_binding"]["path"])
    source_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source file SHA-256"):
        validate_reference_feasibility_manifest(manifest, verify_source_files=True)

    tampered_implementation = deepcopy(manifest)
    tampered_implementation["implementation_bindings"][0]["sha256"] = "0" * 64
    tampered_implementation[REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD] = canonical_sha256(
        tampered_implementation,
        digest_field=REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="implementation source .* SHA-256"):
        validate_reference_feasibility_manifest(
            tampered_implementation,
            verify_implementation_sources=True,
        )
