"""Contact-projection repair for SONIC reference motions (CPU, MuJoCo).

WHY this module exists
----------------------
``gear_sonic.research.hygiene.screen`` answers "could *any* controller supply the forces this
reference demands, given the contacts it offers?".  A large share of the clips it rejects fail for
one banal reason: the retarget *floats*.  The reference asks for support while no collision geom is
anywhere near the floor, so the required base wrench is unsupportable no matter how good the policy
is.  Discarding those clips throws away otherwise fine motion.  This module is the cheap repair
that recovers them: it projects the root trajectory down onto the contact manifold — lowering the
pelvis just enough that the lowest geom regains contact wherever the reference is unsupported,
blended smoothly in time, never raising — and then re-screens the result.

WHY it is cheap *in this format specifically*
---------------------------------------------
The reference operator this ports from (``/data/robotixx/climb/tools/repair_contact_projection.py``)
stores per-body world poses and velocities, so a root shift forces a full forward-kinematics rebuild
of ``body_pos_w``/``body_quat_w``/``body_lin_vel_w``, and every rebuilt array is a fresh chance to
desynchronise the file from itself.  The SONIC format has no such redundancy to rebuild: the root
pose lives in ``root_trans_offset`` alone and the joint configuration lives in ``dof``/``pose_aa``.
A pure vertical root projection therefore touches **exactly one column of one array**,
``root_trans_offset[:, 2]``.  ``dof`` and ``pose_aa`` are not regenerated, not re-derived, not even
read for writing, so the ``pose_aa[:, 1:, :].sum(-1) == dof`` redundancy contract that the LACE
manifest asserts to ~1e-6 is preserved *by construction* rather than by a numerical check.
``root_rot`` is untouched (a translation cannot change an orientation) and ``smpl_joints`` is
untouched (see the frame decision below).

smpl_joints frame decision: NOT shifted
---------------------------------------
``smpl_joints`` is *not* offset by the root projection, and this is deliberate.  In the SONIC
motion library the SMPL stream carries its world translation in a separate ``smpl_transl`` field
(``motion_lib_base.py`` loads ``smpl_pose``/``smpl_joints``/``smpl_transl`` side by side), and every
consumer of the joints — ``mdp.observations.smpl_joints_multi_future``,
``smpl_joints_multi_future_local``, ``smpl_joints_lower_multi_future_local`` — feeds them through
``quat_apply(quat_inv(root_quat), joints)`` with **no translation subtraction at all**.  That is
only well posed if the stored joints are already root-relative.  Adding a metre-scale z offset to a
root-relative quantity would inject a constant bias straight into the SMPL encoder input, which is
precisely the silent desynchronisation we are trying to avoid.  Empirically the shipped 4950-clip
bank stores ``smpl_joints`` as an all-zero placeholder (verified over random samples), so shifting
would also manufacture a fake non-zero signal out of a clean "absent" marker.

Because that reasoning depends on the frame convention, the operator *checks* it instead of
assuming it: :func:`smpl_joints_are_root_relative` tests whether the SMPL root joint sits at the
origin.  If a clip stores absolute-world SMPL joints and a non-zero offset would be applied, the
repair refuses with reason ``smpl_joints_absolute_frame`` rather than silently desynchronising the
SMPL stream from the G1 reference.

Scope honesty
-------------
The operator *triggers* on airborne frames (no collision geom within ``gap_m`` of the floor) but is
*scored* on the screen's full LP-based ``infeasible_frac``.  A clip that is infeasible while in
contact — friction-cone or actuator-torque reasons — offers this operator nothing to push on, gets a
zero offset, and would otherwise be recorded as an operator failure.  That is a category error, so
it gets its own reason string, ``out_of_scope_not_airborne``: the clip needs a stronger operator
(IK, time warp, retarget), not a better root projection.

Two more honesty notes about that word "airborne":

* The screen's ``airborne_frac`` counts frames with no *foot* near the floor; this operator's
  trigger counts frames with no *collision geom of any kind* near the floor.  The operator is
  deliberately the stricter of the two, because a kneeling or prone reference has its feet in the
  air and its knees on the ground, and lowering the root there would drive the knee through the
  floor.  Such a clip reads as airborne in the screen record and lands in
  ``out_of_scope_not_airborne`` here.  That disagreement is the point, not a bug.
* The screen may report ``infeasible_frac=None`` -- it refuses to score a clip whose every frame had
  contacts and whose every torque LP failed.  That travels through here as ``NaN``, fails every
  comparison, and is reported as ``unscoreable_screen`` rather than being coerced to a number the
  screen declined to produce.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import mujoco
import numpy as np

from gear_sonic.research.hygiene.motion_io import Motion
from gear_sonic.research.hygiene.screen import (
    FLOOR_GEOM_NAME,
    ScreenThresholds,
    load_model,
    screen_motion,
)

REPAIR_SCHEMA_VERSION = 1

#: Offsets at or below this are treated as "the operator did nothing" (metres).
ZERO_OFFSET_EPS_M = 1e-6

#: Tolerance for calling the stored SMPL root joint "at the origin" (metres).
SMPL_ROOT_ORIGIN_TOL_M = 1e-3

REASON_ALREADY_FEASIBLE = "already_feasible"
REASON_REPAIRED = "repaired"
REASON_OUT_OF_SCOPE_NOT_AIRBORNE = "out_of_scope_not_airborne"
REASON_OFFSET_OVER_BUDGET = "offset_over_budget"
REASON_RESIDUAL_INFEASIBLE = "residual_infeasible"
REASON_REGRESSED = "regressed"
REASON_SMPL_JOINTS_ABSOLUTE_FRAME = "smpl_joints_absolute_frame"
REASON_UNSCOREABLE = "unscoreable_screen"

REPAIR_REASONS = (
    REASON_ALREADY_FEASIBLE,
    REASON_REPAIRED,
    REASON_OUT_OF_SCOPE_NOT_AIRBORNE,
    REASON_OFFSET_OVER_BUDGET,
    REASON_RESIDUAL_INFEASIBLE,
    REASON_REGRESSED,
    REASON_SMPL_JOINTS_ABSOLUTE_FRAME,
    REASON_UNSCOREABLE,
)


@dataclass(frozen=True)
class RepairBudget:
    """Deviation licence for the contact projection.

    ``max_offset_m`` is how far the pelvis may be moved before the repair stops being a repair and
    starts being a different motion; ``max_infeasible_frac_after`` is the screen bar the repaired
    clip must clear; ``clearance_m`` is the gap left under the lowest geom at touchdown;
    ``smooth_s`` is the width of the temporal blend applied to the offset profile.
    """

    max_offset_m: float = 0.15
    max_infeasible_frac_after: float = 0.05
    clearance_m: float = 0.003
    smooth_s: float = 0.24


@dataclass(frozen=True)
class RepairResult:
    """Outcome of one repair attempt.

    ``success`` is exactly ``infeasible_frac_after <= budget.max_infeasible_frac_after and
    offset_max_m <= budget.max_offset_m``.  The ``*_after`` metrics always describe the *attempted*
    projection, even when it was rejected; :func:`repair_motion` returns the untouched original
    motion whenever ``success`` is False, so the output bank differs from the input bank on exactly
    the clips the operator succeeded on.
    """

    motion_key: str
    success: bool
    reason: str
    airborne_frac_before: float
    airborne_frac_after: float
    infeasible_frac_before: float
    infeasible_frac_after: float
    offset_max_m: float
    offset_mean_m: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict.  Unmeasurable metrics become ``null``, never ``0.0``.

        ``screen_motion`` returns ``infeasible_frac=None`` when every frame had contacts and every
        torque-limited LP failed, i.e. the clip is so far outside the actuator envelope that the
        solver cannot even quantify how far.  That travels through this dataclass as ``NaN`` so the
        gate arithmetic stays float, and leaves it as ``null`` so no reader mistakes "could not be
        scored" for "scored zero".
        """

        payload = asdict(self)
        for name, value in payload.items():
            if isinstance(value, float) and math.isnan(value):
                payload[name] = None
        return payload


def screen_infeasible_frac(screen: Any) -> float:
    """``screen.infeasible_frac`` as a float, with the screen's ``None`` mapped to ``NaN``.

    Every comparison against ``NaN`` is False, so a clip the screen could not score can never pass
    the success gate by accident -- which is the behaviour we want, since ``None`` there means
    "contacts existed and every LP failed", the most infeasible state the screen can report.
    """

    value = getattr(screen, "infeasible_frac", None)
    return float("nan") if value is None else float(value)


def floor_geom_id(model: Any) -> int:
    """Return the geom id of the ground plane.

    Reproduces ``screen._build_layout``'s rule exactly -- the geom named
    :data:`~gear_sonic.research.hygiene.screen.FLOOR_GEOM_NAME`, else the last plane geom -- because
    a repair that measures clearance against one plane while the screen scores contacts against
    another would be scored on physics it never performed.
    """

    named = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    if named >= 0:
        return int(named)
    planes = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    if not planes:
        raise ValueError(
            f"model has no geom named {FLOOR_GEOM_NAME!r} and no plane geom to fall back on"
        )
    return int(planes[-1])


def contact_geom_ids(model: Any) -> list[int]:
    """Collision geoms that may touch the floor: everything collidable except the plane itself.

    Deliberately not restricted to the feet.  A prone or kneeling reference is supported by knees,
    forearms or torso, and a foot-only clearance test would report those frames as airborne and
    invite the operator to drive the whole body through the floor.
    """

    plane = floor_geom_id(model)
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id != plane and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    ]


def motion_qpos(model: Any, motion: Motion) -> np.ndarray:
    """Assemble MuJoCo ``qpos`` from a SONIC motion (root xyz, root wxyz quat, 29 MuJoCo dofs)."""

    num_frames = int(motion.root_trans_offset.shape[0])
    if model.nq != 7 + motion.dof.shape[1]:
        raise ValueError(
            f"model nq={model.nq} does not match a free root plus {motion.dof.shape[1]} dofs"
        )
    qpos = np.zeros((num_frames, model.nq), dtype=np.float64)
    qpos[:, 0:3] = motion.root_trans_offset
    # SONIC stores root_rot as XYZW (scipy convention); MuJoCo free joints want WXYZ.
    qpos[:, 3:7] = motion.root_rot[:, [3, 0, 1, 2]]
    qpos[:, 7:] = motion.dof
    return qpos


def floor_clearance(
    motion: Motion,
    model: Any | None = None,
    *,
    data: Any | None = None,
    distmax: float = 5.0,
) -> np.ndarray:
    """Per-frame distance from the lowest collision geom to the floor plane, in metres.

    Negative where the reference already penetrates the floor.  Saturates at ``distmax`` for
    references that fly far above the ground; such clips blow the offset budget regardless.
    """

    if model is None:
        model = load_model()
    if data is None:
        data = mujoco.MjData(model)
    plane = floor_geom_id(model)
    geoms = contact_geom_ids(model)
    qpos = motion_qpos(model, motion)
    clearance = np.empty(qpos.shape[0], dtype=np.float64)
    fromto = np.zeros(6)
    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        clearance[frame] = min(
            mujoco.mj_geomDistance(model, data, geom_id, plane, distmax, fromto)
            for geom_id in geoms
        )
    return clearance


def gaussian_smooth(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Edge-padded Gaussian blur of a 1-D profile (ports ``smooth1d`` from the reference operator)."""

    values = np.asarray(values, dtype=np.float64)
    if sigma_frames <= 0 or values.size == 0:
        return values.copy()
    width = int(4 * sigma_frames) | 1
    kernel = np.exp(-0.5 * ((np.arange(width) - width // 2) / sigma_frames) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plan_root_offset(
    clearance: np.ndarray,
    fps: int,
    *,
    gap_m: float,
    clearance_m: float,
    smooth_s: float,
) -> np.ndarray:
    """Non-negative per-frame downward root offset that lands the lowest geom where support is due.

    Three rules, in order: (1) only frames whose lowest geom is further than ``gap_m`` from the
    floor are candidates — the reference is asking for support it cannot receive there; (2) the raw
    offset is exactly enough to leave ``clearance_m`` under that geom; (3) the profile is blended
    over ``smooth_s`` so the pelvis does not teleport, then re-clamped so the blend can never push a
    geom through the floor and never *raise* the root.

    Rule (3)'s re-clamp is the operator's known rough edge: it truncates the blend at touchdown, so
    the offset profile can still have a kink there.  Callers that care should diff the root height
    before and after and check the induced vertical speed.
    """

    clearance = np.asarray(clearance, dtype=np.float64)
    headroom = np.maximum(clearance - clearance_m, 0.0)
    raw = np.where(clearance > gap_m, headroom, 0.0)
    offset = gaussian_smooth(raw, smooth_s * float(fps))
    offset = np.minimum(offset, headroom)
    return np.maximum(offset, 0.0)


def smpl_joints_are_root_relative(motion: Motion, tol_m: float = SMPL_ROOT_ORIGIN_TOL_M) -> bool:
    """True when the stored SMPL root joint sits at the origin, i.e. the joints are root-relative.

    An all-zero placeholder (what the shipped bank contains) passes trivially, which is correct: a
    field that carries no information cannot be desynchronised by leaving it alone.
    """

    joints = np.asarray(motion.smpl_joints)
    if joints.size == 0:
        return True
    return float(np.abs(joints[:, 0, :]).max()) <= tol_m


def apply_root_offset(motion: Motion, offset: np.ndarray) -> Motion:
    """Lower the pelvis by ``offset`` metres per frame; touch nothing else.

    This is the whole physical edit.  ``dof`` and ``pose_aa`` are passed through by reference, so
    their mutual redundancy cannot drift; ``root_rot`` and ``smpl_joints`` likewise.
    """

    offset = np.asarray(offset, dtype=np.float64)
    root = np.array(motion.root_trans_offset, dtype=np.float32, copy=True)
    if offset.shape != (root.shape[0],):
        raise ValueError(f"offset shape {offset.shape} does not match {root.shape[0]} frames")
    root[:, 2] = (root[:, 2].astype(np.float64) - offset).astype(np.float32)
    return motion.replace(root_trans_offset=root)


def repair_motion(
    motion: Motion,
    model: Any | None = None,
    budget: RepairBudget = RepairBudget(),
    thresholds: ScreenThresholds = ScreenThresholds(),
) -> tuple[Motion, RepairResult]:
    """Screen, project the root onto the contact manifold if that helps, re-screen, and judge.

    Returns the motion that should be written to the output bank — the repaired one on success, the
    untouched original on every failure path — together with the full before/after record.
    """

    if model is None:
        model = load_model()
    before = screen_motion(motion, model=model, thresholds=thresholds)

    def _result(success: bool, reason: str, after: Any, offset: np.ndarray) -> RepairResult:
        return RepairResult(
            motion_key=motion.key,
            success=success,
            reason=reason,
            airborne_frac_before=float(before.airborne_frac),
            airborne_frac_after=float(after.airborne_frac),
            infeasible_frac_before=screen_infeasible_frac(before),
            infeasible_frac_after=screen_infeasible_frac(after),
            offset_max_m=float(offset.max()) if offset.size else 0.0,
            offset_mean_m=float(offset.mean()) if offset.size else 0.0,
        )

    zero = np.zeros(int(motion.root_trans_offset.shape[0]), dtype=np.float64)
    infeasible_before = screen_infeasible_frac(before)

    # Nothing to fix: leave a passing clip bit-identical rather than spend a second screen on it.
    # NaN (the screen could not score the clip at all) never satisfies this, by design.
    if infeasible_before <= budget.max_infeasible_frac_after:
        return motion, _result(True, REASON_ALREADY_FEASIBLE, before, zero)

    clearance = floor_clearance(motion, model)
    offset = plan_root_offset(
        clearance,
        int(motion.fps),
        gap_m=thresholds.gap_m,
        clearance_m=budget.clearance_m,
        smooth_s=budget.smooth_s,
    )

    # The clip is infeasible but nothing is liftable: friction cone or torque limits, not float.
    # The projection has no lever here, so say so instead of blaming the operator.  Note the
    # asymmetry with the screen's ``airborne_frac``, which is a *foot* statistic: a kneeling clip
    # reads as airborne there while a knee geom is on the floor here, and lands in this branch.
    # That is the right answer -- lowering the root would drive the knee through the ground.
    if float(offset.max()) <= ZERO_OFFSET_EPS_M:
        reason = (
            REASON_UNSCOREABLE
            if math.isnan(infeasible_before)
            else REASON_OUT_OF_SCOPE_NOT_AIRBORNE
        )
        return motion, _result(False, reason, before, zero)

    if not smpl_joints_are_root_relative(motion):
        return motion, _result(False, REASON_SMPL_JOINTS_ABSOLUTE_FRAME, before, zero)

    repaired = apply_root_offset(motion, offset)
    after = screen_motion(repaired, model=model, thresholds=thresholds)

    infeasible_after = screen_infeasible_frac(after)
    offset_ok = float(offset.max()) <= budget.max_offset_m
    feasible_ok = infeasible_after <= budget.max_infeasible_frac_after
    if offset_ok and feasible_ok:
        return repaired, _result(True, REASON_REPAIRED, after, offset)
    if not offset_ok:
        # Over the deviation licence: the after-metrics are still reported so the census can answer
        # "would a larger budget have worked?", but the original clip is what ships.
        return motion, _result(False, REASON_OFFSET_OVER_BUDGET, after, offset)
    if math.isnan(infeasible_after):
        return motion, _result(False, REASON_UNSCOREABLE, after, offset)
    if infeasible_after > infeasible_before:
        return motion, _result(False, REASON_REGRESSED, after, offset)
    return motion, _result(False, REASON_RESIDUAL_INFEASIBLE, after, offset)
