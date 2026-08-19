"""Tests for the contact-projection repair operator and the bank CLI that drives it.

Everything here is synthetic: clips are built from the G1 model's own default pose so the tests
state physical facts ("this clip hovers 8 cm too high") rather than magic numbers, and stay true if
the shipped MJCF changes.  The properties under test are the ones that make the operator safe to
run over a 4950-clip bank unattended:

* it lowers what floats and leaves alone what does not;
* it never rewrites ``dof``/``pose_aa``/``root_rot``/``smpl_joints`` -- checked bit-for-bit, because
  the whole claim of this operator in the SONIC format is that a repair is one column of one array;
* it refuses rather than guesses when a clip is outside its scope (in contact but infeasible) or
  outside its licence (offset over budget, absolute-frame SMPL joints);
* what it writes still loads and still validates.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS  # noqa: E402
from gear_sonic.research.hygiene import repair as repair_module  # noqa: E402
from gear_sonic.research.hygiene.motion_io import (  # noqa: E402
    Motion,
    load_motion,
    save_motion,
    validate_motion,
)
from gear_sonic.research.hygiene.repair import (  # noqa: E402
    REASON_ALREADY_FEASIBLE,
    REASON_OFFSET_OVER_BUDGET,
    REASON_OUT_OF_SCOPE_NOT_AIRBORNE,
    REASON_REPAIRED,
    REASON_SMPL_JOINTS_ABSOLUTE_FRAME,
    REASON_UNSCOREABLE,
    RepairBudget,
    apply_root_offset,
    floor_clearance,
    gaussian_smooth,
    plan_root_offset,
    repair_motion,
    screen_infeasible_frac,
    smpl_joints_are_root_relative,
)
from gear_sonic.research.hygiene.screen import ScreenThresholds, load_model  # noqa: E402

NUM_FRAMES = 8
FPS = 30


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def ground_root_z(model) -> float:
    """Root height at which the default-pose G1 rests with a 3 mm clearance under its lowest geom.

    Measured, not assumed: a plane sits at z = 0 and a pure vertical root shift moves every geom by
    the same amount, so ``clearance(z) = clearance(0) + z`` exactly.
    """

    probe = make_motion(root_z=0.0)
    clearance_at_zero = float(floor_clearance(probe, model)[0])
    return RepairBudget.clearance_m - clearance_at_zero


def make_motion(
    root_z: float,
    *,
    key: str = "synthetic_clip",
    num_frames: int = NUM_FRAMES,
    dof: np.ndarray | None = None,
    smpl_joints: np.ndarray | None = None,
) -> Motion:
    """A structurally valid clip: default pose, identity root rotation, constant root height.

    ``pose_aa`` is derived from ``dof`` through the same ``DOF_AXIS`` identity the motion library
    asserts, so :func:`validate_motion` passes and the redundancy claim under test is real.
    """

    dof_array = (
        np.zeros((num_frames, 29), dtype=np.float32)
        if dof is None
        else np.asarray(dof, dtype=np.float32)
    )
    pose_aa = np.zeros((num_frames, 30, 3), dtype=np.float32)
    pose_aa[:, 1:, :] = (
        DOF_AXIS.astype(np.float64)[None] * dof_array.astype(np.float64)[:, :, None]
    ).astype(np.float32)
    root_trans = np.zeros((num_frames, 3), dtype=np.float32)
    root_trans[:, 2] = root_z
    root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_frames, 1))
    joints = (
        np.zeros((num_frames, 24, 3), dtype=np.float32)
        if smpl_joints is None
        else np.asarray(smpl_joints, dtype=np.float32)
    )
    return Motion(
        key=key,
        root_trans_offset=root_trans,
        pose_aa=pose_aa,
        dof=dof_array,
        root_rot=root_rot,
        smpl_joints=joints,
        fps=FPS,
    )


def assert_non_root_fields_identical(before: Motion, after: Motion) -> None:
    """The operator's core promise: only ``root_trans_offset[:, 2]`` may differ."""

    for field in ("pose_aa", "dof", "root_rot", "smpl_joints"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field), err_msg=field)
    np.testing.assert_array_equal(before.root_trans_offset[:, :2], after.root_trans_offset[:, :2])
    assert before.key == after.key
    assert before.fps == after.fps


# --------------------------------------------------------------------------------------
# offset planning (pure numerics, no physics)
# --------------------------------------------------------------------------------------


def test_gaussian_smooth_preserves_a_constant():
    values = np.full(64, 0.07)
    np.testing.assert_allclose(gaussian_smooth(values, 7.2), values, atol=1e-12)


def test_gaussian_smooth_is_identity_for_zero_sigma():
    values = np.arange(10, dtype=float)
    np.testing.assert_array_equal(gaussian_smooth(values, 0.0), values)


def test_plan_root_offset_ignores_frames_already_in_contact():
    clearance = np.full(20, 0.01)  # everywhere below the 6 cm gap
    offset = plan_root_offset(clearance, FPS, gap_m=0.06, clearance_m=0.003, smooth_s=0.24)
    np.testing.assert_array_equal(offset, np.zeros(20))


def test_plan_root_offset_never_raises_and_never_penetrates():
    """Two invariants, over a profile that flies, then scuffs the floor, then flies again.

    The floor invariant is *do no harm*, not *end up above the clearance*: frames that already
    penetrate (negative clearance) are the retarget's problem, and lifting them is out of scope for
    an operator whose whole licence is "never raise".
    """

    rng = np.random.default_rng(0)
    clearance = np.concatenate([np.full(30, 0.20), rng.uniform(-0.02, 0.02, 30), np.full(30, 0.30)])
    offset = plan_root_offset(clearance, FPS, gap_m=0.06, clearance_m=0.003, smooth_s=0.24)
    after = clearance - offset
    assert offset.min() >= 0.0, "the operator must never lift the root"
    assert np.all(
        after >= np.minimum(clearance, 0.003) - 1e-9
    ), "the blend must not deepen a penetration"
    # The blend does bleed into the contact band -- that is what makes the touchdown smooth rather
    # than a step -- but only ever down to the clearance, never through it.
    assert offset[30:60].max() > 0.0
    assert np.all(after[30:60] >= np.minimum(clearance[30:60], 0.003) - 1e-9)


def test_apply_root_offset_touches_only_the_z_column():
    motion = make_motion(root_z=1.0)
    offset = np.linspace(0.0, 0.1, NUM_FRAMES)
    moved = apply_root_offset(motion, offset)
    assert_non_root_fields_identical(motion, moved)
    np.testing.assert_allclose(moved.root_trans_offset[:, 2], 1.0 - offset, atol=1e-6)
    assert moved.root_trans_offset.dtype == np.float32
    # and the source is not mutated in place
    np.testing.assert_array_equal(
        motion.root_trans_offset[:, 2], np.ones(NUM_FRAMES, dtype=np.float32)
    )


def test_apply_root_offset_rejects_a_mismatched_profile():
    with pytest.raises(ValueError):
        apply_root_offset(make_motion(root_z=1.0), np.zeros(NUM_FRAMES + 1))


# --------------------------------------------------------------------------------------
# clearance geometry
# --------------------------------------------------------------------------------------


def test_floor_clearance_tracks_root_height_one_for_one(model, ground_root_z):
    low = floor_clearance(make_motion(root_z=ground_root_z), model)
    high = floor_clearance(make_motion(root_z=ground_root_z + 0.25), model)
    np.testing.assert_allclose(high - low, 0.25, atol=1e-6)
    np.testing.assert_allclose(low, RepairBudget.clearance_m, atol=1e-6)


# --------------------------------------------------------------------------------------
# the operator end to end
# --------------------------------------------------------------------------------------


def test_hovering_clip_is_lowered_and_stops_being_airborne(model, ground_root_z):
    hover = 0.08  # well past the 6 cm contact gap, well inside the 15 cm budget
    motion = make_motion(root_z=ground_root_z + hover)
    repaired, result = repair_motion(motion, model=model)

    assert result.airborne_frac_before == pytest.approx(1.0)
    assert result.airborne_frac_after == pytest.approx(0.0, abs=1e-9)
    assert result.offset_max_m == pytest.approx(hover, abs=1e-3)
    assert result.reason == REASON_REPAIRED
    assert result.success
    assert_non_root_fields_identical(motion, repaired)
    np.testing.assert_allclose(repaired.root_trans_offset[:, 2], ground_root_z, atol=2e-3)


def test_grounded_clip_is_returned_bit_identical(model, ground_root_z):
    motion = make_motion(root_z=ground_root_z)
    repaired, result = repair_motion(motion, model=model)

    assert result.success
    assert result.reason == REASON_ALREADY_FEASIBLE
    assert result.offset_max_m == 0.0
    assert result.offset_mean_m == 0.0
    assert repaired is motion
    np.testing.assert_array_equal(repaired.root_trans_offset, motion.root_trans_offset)
    assert_non_root_fields_identical(motion, repaired)


def test_clip_beyond_the_offset_budget_is_refused_untouched(model, ground_root_z):
    motion = make_motion(root_z=ground_root_z + 0.60)
    repaired, result = repair_motion(motion, model=model)

    assert not result.success
    assert result.reason == REASON_OFFSET_OVER_BUDGET
    assert result.offset_max_m > RepairBudget.max_offset_m
    assert repaired is motion, "a refused repair must ship the original clip"
    np.testing.assert_array_equal(repaired.root_trans_offset, motion.root_trans_offset)


def test_a_bigger_budget_accepts_the_same_clip(model, ground_root_z):
    """The refusal above is the licence talking, not the physics: widen it and the clip repairs."""

    motion = make_motion(root_z=ground_root_z + 0.60)
    repaired, result = repair_motion(motion, model=model, budget=RepairBudget(max_offset_m=1.0))
    assert result.success
    assert result.reason == REASON_REPAIRED
    assert repaired.root_trans_offset[:, 2].max() < motion.root_trans_offset[:, 2].max()


def test_in_contact_but_infeasible_is_reported_out_of_scope(model, ground_root_z, monkeypatch):
    """Friction-cone / torque failures are a category error for a root projection, not a failure.

    The clip is grounded, so the operator plans a zero offset; without the dedicated reason it would
    be indistinguishable from "the projection was tried and did not help".
    """

    class _FakeScreen:
        airborne_frac = 0.0
        infeasible_frac = 0.9

    monkeypatch.setattr(repair_module, "screen_motion", lambda *a, **k: _FakeScreen())
    motion = make_motion(root_z=ground_root_z)
    repaired, result = repair_motion(motion, model=model)

    assert not result.success
    assert result.reason == REASON_OUT_OF_SCOPE_NOT_AIRBORNE
    assert result.offset_max_m == 0.0
    assert result.infeasible_frac_before == pytest.approx(0.9)
    assert repaired is motion


def test_an_unscoreable_screen_never_passes_the_gate(model, ground_root_z, monkeypatch):
    """``screen_motion`` returns ``infeasible_frac=None`` when every LP failed.

    That is the *most* infeasible state the screen can report, so it must not slip through the
    ``<= max_infeasible_frac_after`` comparison -- which is exactly what ``float(None)`` or a
    ``None or 0.0`` coercion would do.
    """

    class _Unscoreable:
        airborne_frac = 0.0
        infeasible_frac = None

    monkeypatch.setattr(repair_module, "screen_motion", lambda *a, **k: _Unscoreable())
    _, result = repair_motion(make_motion(root_z=ground_root_z), model=model)

    assert not result.success
    assert result.reason == REASON_UNSCOREABLE
    assert np.isnan(result.infeasible_frac_before)
    assert result.to_dict()["infeasible_frac_before"] is None, "null, never a fake 0.0"


def test_screen_infeasible_frac_maps_none_to_nan():
    class _S:
        infeasible_frac = None

    assert np.isnan(screen_infeasible_frac(_S()))
    _S.infeasible_frac = 0.25
    assert screen_infeasible_frac(_S()) == pytest.approx(0.25)


# --------------------------------------------------------------------------------------
# smpl_joints frame decision
# --------------------------------------------------------------------------------------


def test_smpl_joints_are_not_shifted_by_the_repair(model, ground_root_z):
    """Decision under test: SMPL joints are root-relative, so the root shift must NOT move them.

    A non-zero but origin-rooted SMPL stream is the realistic case; it must come out unchanged.
    """

    rng = np.random.default_rng(7)
    joints = rng.normal(scale=0.3, size=(NUM_FRAMES, 24, 3)).astype(np.float32)
    joints[:, 0, :] = 0.0  # root-relative: joint 0 is the pelvis, at the origin
    motion = make_motion(root_z=ground_root_z + 0.08, smpl_joints=joints)
    repaired, result = repair_motion(motion, model=model)

    assert result.success
    assert result.offset_max_m > 0.0
    np.testing.assert_array_equal(repaired.smpl_joints, joints)
    assert repaired.smpl_joints is motion.smpl_joints


def test_absolute_frame_smpl_joints_refuse_rather_than_desynchronise(model, ground_root_z):
    """If the SMPL root is not at the origin the frame assumption is wrong -- refuse, do not shift."""

    joints = np.zeros((NUM_FRAMES, 24, 3), dtype=np.float32)
    joints[:, :, 2] = ground_root_z + 0.08  # world-absolute heights
    motion = make_motion(root_z=ground_root_z + 0.08, smpl_joints=joints)
    repaired, result = repair_motion(motion, model=model)

    assert not result.success
    assert result.reason == REASON_SMPL_JOINTS_ABSOLUTE_FRAME
    assert repaired is motion
    np.testing.assert_array_equal(repaired.smpl_joints, joints)


def test_smpl_joints_frame_probe():
    assert smpl_joints_are_root_relative(make_motion(root_z=1.0))
    absolute = np.zeros((NUM_FRAMES, 24, 3), dtype=np.float32)
    absolute[:, 0, 2] = 0.9
    assert not smpl_joints_are_root_relative(make_motion(root_z=1.0, smpl_joints=absolute))


# --------------------------------------------------------------------------------------
# what gets written
# --------------------------------------------------------------------------------------


def test_repaired_clip_round_trips_and_still_validates(model, ground_root_z, tmp_path):
    motion = make_motion(root_z=ground_root_z + 0.08, key="round_trip_clip")
    repaired, result = repair_motion(motion, model=model)
    assert result.success

    path = tmp_path / "round_trip_clip.pkl"
    save_motion(repaired, path)
    reloaded = load_motion(path)

    assert validate_motion(reloaded) == []
    assert reloaded.key == "round_trip_clip"
    assert reloaded.fps == FPS
    np.testing.assert_allclose(reloaded.root_trans_offset, repaired.root_trans_offset, atol=0.0)
    assert_non_root_fields_identical(repaired, reloaded)


def test_result_to_dict_is_json_shaped(model, ground_root_z):
    _, result = repair_motion(make_motion(root_z=ground_root_z + 0.08), model=model)
    payload = result.to_dict()
    assert set(payload) == {
        "motion_key",
        "success",
        "reason",
        "airborne_frac_before",
        "airborne_frac_after",
        "infeasible_frac_before",
        "infeasible_frac_after",
        "offset_max_m",
        "offset_mean_m",
    }
    assert isinstance(payload["success"], bool)
    assert all(
        isinstance(payload[k], float)
        for k in payload
        if k.endswith(("_frac_before", "_frac_after", "_m"))
    )


# --------------------------------------------------------------------------------------
# the bank CLI
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bank_script():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))
    import hygiene_repair_bank

    return hygiene_repair_bank


def test_needs_repair_defaults_to_yes_when_the_screen_is_silent(bank_script):
    assert bank_script.needs_repair(None, 0.05) is True
    assert bank_script.needs_repair({"infeasible_frac": 0.9}, 0.05) is True
    assert bank_script.needs_repair({"infeasible_frac": 0.01}, 0.05) is False
    assert bank_script.needs_repair({"infeasible_frac": "not a number"}, 0.05) is True
    assert (
        bank_script.needs_repair({"infeasible_frac": None}, 0.05) is True
    ), "null means unscoreable"
    assert bank_script.needs_repair({"infeasible_frac": float("nan")}, 0.05) is True


def test_read_screen_reports_accepts_per_clip_and_aggregate_json(bank_script, tmp_path):
    import json

    (tmp_path / "a.json").write_text(json.dumps({"motion_key": "a", "infeasible_frac": 0.2}))
    (tmp_path / "bundle.json").write_text(
        json.dumps({"clips": [{"motion_key": "b", "infeasible_frac": 0.0}]})
    )
    (tmp_path / "junk.json").write_text("{not json")
    reports = bank_script.read_screen_reports(tmp_path)
    assert set(reports) == {"a", "b"}
    assert reports["b"]["infeasible_frac"] == 0.0


def test_bank_run_is_complete_and_copies_clean_clips_byte_for_byte(
    bank_script, model, ground_root_z, tmp_path
):
    """The output bank must have every clip, real files only, and untouched clips bit-identical."""

    import hashlib
    import json

    bank = tmp_path / "bank"
    out_bank = tmp_path / "out"
    reports = tmp_path / "reports"
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    bank.mkdir()

    save_motion(make_motion(root_z=ground_root_z, key="grounded"), bank / "grounded.pkl")
    save_motion(make_motion(root_z=ground_root_z + 0.08, key="hovering"), bank / "hovering.pkl")
    # the screen says "grounded" is fine, so it must be copied without being screened again
    (screen_dir / "grounded.json").write_text(
        json.dumps({"motion_key": "grounded", "infeasible_frac": 0.0})
    )
    (screen_dir / "hovering.json").write_text(
        json.dumps({"motion_key": "hovering", "infeasible_frac": 1.0})
    )

    digest_before = hashlib.sha256((bank / "grounded.pkl").read_bytes()).hexdigest()
    code = bank_script.main(
        [
            "--bank",
            str(bank),
            "--out-bank",
            str(out_bank),
            "--screen-dir",
            str(screen_dir),
            "--out-reports",
            str(reports),
        ]
    )

    assert code == 0
    assert {p.name for p in out_bank.glob("*.pkl")} == {"grounded.pkl", "hovering.pkl"}
    for path in out_bank.glob("*.pkl"):
        assert not path.is_symlink(), "throughput.verify_partition_subset rejects symlinked banks"
    assert hashlib.sha256((out_bank / "grounded.pkl").read_bytes()).hexdigest() == digest_before
    assert (reports / "COMPLETED").is_file()

    summary = json.loads((reports / "repair_summary.json").read_text())
    assert summary["n_clips"] == 2
    assert summary["n_attempted"] == 1
    assert summary["n_success"] == 1
    assert summary["by_reason"][REASON_REPAIRED] == 1
    assert summary["n_bank_files_rewritten"] == 1

    repaired = load_motion(out_bank / "hovering.pkl")
    assert validate_motion(repaired) == []
    assert repaired.root_trans_offset[:, 2].max() < ground_root_z + 0.08

    census = (reports / "repair_census.csv").read_text().splitlines()
    assert census[0].startswith("motion_key,action,success,reason")
    assert len(census) == 3


def test_bank_run_resumes_without_redoing_work(bank_script, model, ground_root_z, tmp_path):
    import json

    bank = tmp_path / "bank"
    out_bank = tmp_path / "out"
    reports = tmp_path / "reports"
    bank.mkdir()
    save_motion(make_motion(root_z=ground_root_z + 0.08, key="hovering"), bank / "hovering.pkl")

    argv = [
        "--bank",
        str(bank),
        "--out-bank",
        str(out_bank),
        "--out-reports",
        str(reports),
        "--repair-all",
    ]
    assert bank_script.main(argv) == 0
    first = (out_bank / "hovering.pkl").stat().st_mtime_ns

    assert bank_script.main(argv) == 0
    assert (out_bank / "hovering.pkl").stat().st_mtime_ns == first, "a resumed run must not rewrite"
    summary = json.loads((reports / "repair_summary.json").read_text())
    assert summary["n_clips"] == 1
    assert summary["n_success"] == 1


def test_summarise_counts_reasons_and_averages(bank_script):
    rows = [
        {
            "motion_key": "a",
            "action": bank_script.ACTION_WRITTEN,
            "success": True,
            "reason": REASON_REPAIRED,
            "screened_here": True,
            "airborne_frac_before": 1.0,
            "airborne_frac_after": 0.0,
            "infeasible_frac_before": 0.8,
            "infeasible_frac_after": 0.0,
            "offset_max_m": 0.08,
            "offset_mean_m": 0.06,
            "seconds": 1.0,
        },
        {
            "motion_key": "b",
            "action": bank_script.ACTION_COPIED,
            "success": False,
            "reason": REASON_OFFSET_OVER_BUDGET,
            "screened_here": True,
            "airborne_frac_before": 1.0,
            "airborne_frac_after": 1.0,
            "infeasible_frac_before": 1.0,
            "infeasible_frac_after": 1.0,
            "offset_max_m": 0.9,
            "offset_mean_m": 0.9,
            "seconds": 3.0,
        },
        {
            "motion_key": "c",
            "action": bank_script.ACTION_COPIED,
            "success": True,
            "reason": "screen_feasible",
            "screened_here": False,
            "airborne_frac_before": 0.0,
            "airborne_frac_after": 0.0,
            "infeasible_frac_before": 0.0,
            "infeasible_frac_after": 0.0,
            "offset_max_m": 0.0,
            "offset_mean_m": 0.0,
            "seconds": 0.01,
        },
    ]
    summary = bank_script.summarise(rows, budget=RepairBudget(), thresholds=ScreenThresholds())
    assert summary["n_clips"] == 3
    assert summary["n_attempted"] == 2
    assert summary["n_success"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["by_reason"][REASON_REPAIRED] == 1
    assert summary["by_reason"][REASON_OFFSET_OVER_BUDGET] == 1
    assert summary["by_reason"]["screen_feasible"] == 1
    assert summary["offset_max_m_max"] == pytest.approx(0.9)
    assert summary["infeasible_frac_after_mean"] == pytest.approx(0.5)
    assert summary["seconds_per_clip_mean"] == pytest.approx(2.0)
