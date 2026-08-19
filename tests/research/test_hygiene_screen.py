"""Tests for the hygiene motion I/O and the dynamic-feasibility screen.

Everything here is synthetic on purpose: the 4950-clip Bones-SEED bank lives on a
research volume that CI does not mount, and a screen whose correctness can only
be demonstrated on data nobody else can read is not a screen anybody can trust.
The two physics tests are the ones that matter - a robot standing on the floor
must screen as supportable, and the same robot teleported 0.5 m into the air must
screen as unsupportable by exactly its own body weight.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gear_sonic.research.hygiene.motion_io import (
    DOF_AXIS,
    MOTION_KEYS,
    Motion,
    load_motion,
    motion_from_dof,
    motion_sha256,
    resample_to,
    save_motion,
    validate_motion,
)
from gear_sonic.research.hygiene.screen import (
    DEFAULT_G1_MJCF,
    SCREEN_SCHEMA_VERSION,
    ClipScreen,
    ScreenThresholds,
    load_model,
    screen_motion,
    standing_root_height,
    verify_dof_order,
)

mujoco = pytest.importorskip("mujoco")

requires_robot = pytest.mark.skipif(
    not DEFAULT_G1_MJCF.is_file(), reason=f"G1 MJCF not present at {DEFAULT_G1_MJCF}"
)

NUM_DOF = 29
NUM_BODIES = 30


def _static_clip(key: str, *, frames: int, root_z: float, fps: int = 50) -> Motion:
    """A clip that holds the zero joint pose at a fixed pelvis height."""
    dof = np.zeros((frames, NUM_DOF), dtype=np.float32)
    trans = np.tile(np.array([0.0, 0.0, root_z], dtype=np.float32), (frames, 1))
    return motion_from_dof(key, dof=dof, root_trans_offset=trans, fps=fps)


def _wiggling_clip(key: str, *, frames: int, fps: int) -> Motion:
    """A clip with a moving root and moving joints, so slerp has real work to do."""
    time = np.arange(frames, dtype=np.float32) / float(fps)
    dof = np.zeros((frames, NUM_DOF), dtype=np.float32)
    dof[:, 3] = 0.4 * np.sin(2.0 * np.pi * 0.7 * time)
    dof[:, 9] = -0.3 * np.sin(2.0 * np.pi * 0.5 * time)
    trans = np.stack(
        [0.5 * time, 0.1 * np.sin(2.0 * np.pi * 0.3 * time), 0.79 + 0.02 * np.cos(time)], axis=1
    ).astype(np.float32)
    root_aa = np.zeros((frames, 3), dtype=np.float32)
    root_aa[:, 2] = 0.9 * time  # yaw sweep: slerp, not lerp, has to handle this
    return motion_from_dof(key, dof=dof, root_trans_offset=trans, root_aa=root_aa, fps=fps)


# --------------------------------------------------------------------------- I/O


def test_motion_keys_match_the_on_disk_contract() -> None:
    assert MOTION_KEYS == ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints", "fps")


def test_round_trip_preserves_arrays_and_dtype(tmp_path) -> None:
    motion = _wiggling_clip("round_trip", frames=17, fps=30)
    path = tmp_path / "round_trip.pkl"
    save_motion(motion, path)
    reloaded = load_motion(path)

    assert reloaded.key == motion.key
    assert reloaded.fps == motion.fps
    assert reloaded.num_frames == motion.num_frames
    for name in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
        original = getattr(motion, name)
        restored = getattr(reloaded, name)
        assert restored.dtype == np.float32, name
        assert restored.shape == original.shape, name
        np.testing.assert_array_equal(restored, original, err_msg=name)
    assert motion_sha256(reloaded) == motion_sha256(motion)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path) -> None:
    motion = _static_clip("atomic", frames=4, root_z=0.79)
    path = tmp_path / "atomic.pkl"
    save_motion(motion, path)
    save_motion(motion.replace(key="atomic"), path)  # overwrite must also be clean
    assert [p.name for p in tmp_path.iterdir()] == ["atomic.pkl"]


def test_load_rejects_multi_key_payloads(tmp_path) -> None:
    import joblib

    path = tmp_path / "two.pkl"
    joblib.dump({"a": {}, "b": {}}, path)
    with pytest.raises(ValueError, match="exactly one motion key"):
        load_motion(path)


def test_validate_motion_accepts_a_clean_clip() -> None:
    assert validate_motion(_wiggling_clip("clean", frames=11, fps=30)) == []


def test_validate_motion_catches_broken_pose_aa_dof_redundancy() -> None:
    motion = _wiggling_clip("broken", frames=11, fps=30)
    pose = motion.pose_aa.copy()
    pose[5, 4, 1] += np.float32(1e-3)  # edit pose_aa without touching dof
    problems = validate_motion(motion.replace(pose_aa=pose))
    assert any("pose_aa/dof redundancy broken" in p for p in problems), problems
    assert any("frame 5" in p for p in problems), problems


def test_validate_motion_catches_dof_edited_without_pose_aa() -> None:
    motion = _wiggling_clip("broken_dof", frames=11, fps=30)
    dof = motion.dof.copy()
    dof[2, 3] += np.float32(0.05)  # the mistake a naive repair operator makes
    assert any(
        "pose_aa/dof redundancy broken" in p for p in validate_motion(motion.replace(dof=dof))
    )


def test_validate_motion_catches_dtype_and_norm_damage() -> None:
    motion = _wiggling_clip("damaged", frames=8, fps=30)
    problems = validate_motion(motion.replace(dof=motion.dof.astype(np.float64)))
    assert any("dtype float64" in p for p in problems), problems

    bad_quat = motion.root_rot.copy()
    bad_quat[3] *= np.float32(1.5)
    problems = validate_motion(motion.replace(root_rot=bad_quat))
    assert any("root_rot is not unit norm" in p for p in problems), problems


def test_validate_motion_catches_non_finite_values() -> None:
    motion = _static_clip("nan", frames=5, root_z=0.79)
    trans = motion.root_trans_offset.copy()
    trans[2, 1] = np.float32("nan")
    assert any("non-finite" in p for p in validate_motion(motion.replace(root_trans_offset=trans)))


# ---------------------------------------------------------------------- resample


def test_resample_30_to_50_has_sonic_frame_count_and_unit_quaternions() -> None:
    pytest.importorskip("torch")
    source_frames = 91
    motion = _wiggling_clip("resample", frames=source_frames, fps=30)
    resampled = resample_to(motion, 50)

    # SONIC: len(arange(0, (T-1)/src_fps, 1/target_fps)) -- exclusive endpoint.
    expected = math.ceil((source_frames - 1) * 50 / 30)
    assert resampled.num_frames == expected
    assert resampled.fps == 50
    for name in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
        assert getattr(resampled, name).dtype == np.float32, name
        assert getattr(resampled, name).shape[0] == expected, name

    norms = np.linalg.norm(resampled.root_rot.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert validate_motion(resampled) == []


def test_resample_preserves_endpoints_and_monotone_translation() -> None:
    pytest.importorskip("torch")
    motion = _wiggling_clip("endpoints", frames=61, fps=30)
    resampled = resample_to(motion, 50)
    # Exclusive endpoint: frame 0 is exact, the last sample is strictly inside.
    np.testing.assert_allclose(
        resampled.root_trans_offset[0], motion.root_trans_offset[0], atol=1e-6
    )
    assert resampled.root_trans_offset[-1, 0] < motion.root_trans_offset[-1, 0]
    assert np.all(np.diff(resampled.root_trans_offset[:, 0]) > 0)


def test_resample_is_a_no_op_at_matching_fps() -> None:
    motion = _wiggling_clip("noop", frames=13, fps=50)
    assert resample_to(motion, 50) is motion


def test_resample_rejects_degenerate_inputs() -> None:
    motion = _static_clip("tiny", frames=1, root_z=0.79, fps=30)
    with pytest.raises(ValueError, match="cannot resample"):
        resample_to(motion, 50)
    with pytest.raises(ValueError, match="target_fps must be positive"):
        resample_to(_static_clip("t", frames=5, root_z=0.79, fps=30), 0)


# ------------------------------------------------------------------------ screen


@requires_robot
def test_mjcf_joint_order_matches_the_pkl_dof_order() -> None:
    """The pkl documents `dof` as MuJoCo actuator order.  Prove it, do not assume it."""
    model = load_model()
    assert verify_dof_order(model) == []
    np.testing.assert_allclose(np.asarray(model.jnt_axis[1:]), DOF_AXIS, atol=1e-9)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert names[1] == "left_hip_pitch_joint"
    assert names[NUM_DOF] == "right_wrist_yaw_joint"
    assert model.nq == NUM_DOF + 7 and model.nv == NUM_DOF + 6
    assert model.body_mass.sum() == pytest.approx(33.341, abs=1e-3)


@requires_robot
def test_standing_clip_screens_as_supportable() -> None:
    model = load_model()
    motion = _static_clip("synthetic_stand", frames=30, root_z=standing_root_height(model))
    result = screen_motion(motion, model)

    assert result.motion_key == "synthetic_stand"
    assert result.num_frames == 30
    assert result.fps == 50
    assert result.airborne_frac == pytest.approx(0.0, abs=1e-9)
    assert result.infeasible_frac == pytest.approx(0.0, abs=1e-9)
    assert result.unsupported_force_N_max == pytest.approx(0.0, abs=1.0)
    assert result.torque_infeasible_frac == pytest.approx(0.0, abs=1e-9)
    assert result.max_tau_ratio_p95 < 1.0
    assert result.num_evaluable_frames == result.num_frames


@requires_robot
def test_hovering_clip_screens_as_airborne_and_infeasible() -> None:
    model = load_model()
    lift = 0.5
    motion = _static_clip("synthetic_hover", frames=30, root_z=standing_root_height(model) + lift)
    result = screen_motion(motion, model)

    assert result.airborne_frac == pytest.approx(1.0)
    assert result.infeasible_frac == pytest.approx(1.0)
    assert result.num_contact_free_frames == result.num_frames
    # Nothing is touching, the reference is not accelerating: the whole of the
    # robot's weight is force that no contact can supply.
    assert result.unsupported_force_N_p50 == pytest.approx(result.body_weight_N, rel=0.02)
    assert result.unsupported_impulse_per_weight_s == pytest.approx(result.duration_s, rel=0.02)
    assert result.foot_gap_m_p50 == pytest.approx(lift, abs=1e-3)


@requires_robot
def test_threshold_changes_move_the_verdict() -> None:
    model = load_model()
    motion = _static_clip(
        "synthetic_hover_gap", frames=12, root_z=standing_root_height(model) + 0.03
    )
    strict = screen_motion(motion, model, ScreenThresholds(gap_m=0.01))
    loose = screen_motion(motion, model, ScreenThresholds(gap_m=0.06))
    assert strict.airborne_frac == pytest.approx(1.0)
    assert loose.airborne_frac == pytest.approx(0.0)


@requires_robot
def test_screen_result_is_json_serializable_and_stamped() -> None:
    model = load_model()
    motion = _static_clip("stamped", frames=6, root_z=standing_root_height(model))
    result = screen_motion(motion, model)
    payload = result.to_dict()
    assert payload["schema_version"] == SCREEN_SCHEMA_VERSION
    assert payload["thresholds"] == ScreenThresholds().to_dict()
    assert len(payload["robot_sha256"]) == 64
    assert payload["source_sha256"] == motion_sha256(motion)
    json.dumps(payload)  # must not raise: no numpy scalars, no NaN


@requires_robot
def test_screen_requires_two_frames() -> None:
    model = load_model()
    motion = _static_clip("one_frame", frames=1, root_z=standing_root_height(model))
    with pytest.raises(ValueError, match="at least 2 frames"):
        screen_motion(motion, model)


def test_clip_screen_contract_fields_are_present_and_ordered() -> None:
    """Other agents build against this field list; a rename here breaks them."""
    from dataclasses import fields

    names = [f.name for f in fields(ClipScreen)]
    assert names[:15] == [
        "motion_key",
        "num_frames",
        "fps",
        "airborne_frac",
        "infeasible_frac",
        "unsupported_force_N_p50",
        "unsupported_force_N_p95",
        "unsupported_force_N_max",
        "unsupported_impulse_per_weight_s",
        "max_tau_ratio_p95",
        "torque_infeasible_frac",
        "schema_version",
        "thresholds",
        "robot_sha256",
        "source_sha256",
    ]
