"""Dynamic feasibility screen for SONIC G1 reference clips (CPU, MuJoCo).

WHY this exists, and how it relates to LACE
-------------------------------------------
``gear_sonic/research/lace/reference_feasibility.py`` says of itself that it is
"a kinematic proxy, not a dynamic feasibility proof": it asks whether the
reference respects joint limits and whether a foot is plausibly on the ground.
This module is the *dynamic* counterpart.  It asks a different question:

    given the contacts this reference actually offers on this frame, could
    **any** controller supply the forces the reference demands?

A clip can pass every kinematic check and still be unreachable - a reference
that floats 40 cm off the floor while decelerating needs a force nothing is
touching.  A clip can also violate a soft joint limit and still be perfectly
supportable.  The two screens are complementary; this one does not replace,
duplicate, or re-derive LACE, and it deliberately imports nothing from it.

Method (ported from ``/data/robotixx/climb/tools/n1_knee_id.py`` and
``refeas/refeas/screen.py``; the physics is theirs, the I/O and the robot are
ours)
--------------------------------------------------------------------------
Per frame:

1. Drive ``qpos`` straight from the clip: pelvis position, root quaternion
   converted XYZW -> MuJoCo WXYZ and renormalized, then the 29 joint angles.
   ``qvel``/``qacc`` come from finite differences at the clip's own frame rate
   (``mj_differentiatePos`` for ``qvel`` so the free joint's angular rows land in
   the body frame MuJoCo expects, then a boxcar smooth and ``np.gradient``).
2. Contact-free inverse dynamics (``mj_inverse`` with ``mjDSBL_CONTACT``) gives
   the generalized force the environment must supply: a 6-D root wrench ``W``
   that *only* contact can produce, plus joint torques ``tau_free``.
3. Contact candidates are the collision geoms within ``gap_m`` of the floor
   plane.  Each contributes four pyramidal friction-cone edges at coefficient
   ``mu``; column ``j`` of ``A`` is the root-wrench contribution of a unit force
   along edge ``j``, and column ``j`` of ``J_c`` is its joint-torque contribution.
4. Feasibility is an LP: minimize the L1 wrench residual ``|A f - W|`` subject to
   ``f >= 0`` (inside the cones) and ``|tau_free - J_c f| <= tau_max`` (inside the
   actuator limits).  The residual force magnitude that survives is the
   *unsupported force*: newtons that no admissible contact force can produce.
   Torque rows of the wrench are weighted x2 so newtons and newton-metres are
   comparable at a ~0.5 m lever, exactly as the climb reference does.

Robot constants, read off ``g1_29dof_rev_1_0.xml`` at import time
------------------------------------------------------------------
* total mass **33.341 kg**, so body weight **327.077 N** at g = 9.81 m/s^2.
  ``infeasible_frac`` counts frames whose unsupported force exceeds
  ``unsupported_force_frac`` (default 0.5) of that number, i.e. **163.5 N**.
* 36 collision geoms plus the ``floor`` plane; **8 of them are the foot contact
  spheres** (four per ankle-roll link).
* actuator torque limits are **not** on the ``<motor>`` elements in this MJCF
  (``actuator_forcerange`` is all zeros); they are ``actuatorfrcrange`` on the
  joints, so the limits are read from ``jnt_actfrcrange``.  The climb reference
  read ``actuator_forcerange`` because its mjlab model set it there; copying that
  line verbatim would have given every joint a limit of 0 N.m.

How the joint-order assumption was verified (NOT assumed)
---------------------------------------------------------
The pkl documents ``dof`` as "MuJoCo (MJCF actuator) order".  :func:`verify_dof_order`
checks it three ways and :func:`load_model` raises if any of them fails:

1. the MJCF's 30 joints are one free joint followed by 29 hinges, one per body,
   in tree order, and the 29 ``<motor>`` actuators name those hinges in the same
   order (``actuator i -> joint i+1``);
2. the MJCF joint names, in order, equal ``BONES_CSV_JOINT_NAMES`` from
   ``gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`` - the column
   order the bank was *written* with - after stripping the ``_dof`` suffix;
3. ``model.jnt_axis[1:]`` equals ``DOF_AXIS`` elementwise.  ``DOF_AXIS`` is the
   axis table used to build ``pose_aa[:, i+1] = DOF_AXIS[i] * dof[:, i]``, so
   agreement pins both the *order* and the *sign* of every column: any
   permutation (e.g. the IsaacLab ordering via ``MJ_TO_IL``) breaks it, because
   the pitch/roll/yaw axis pattern differs under that permutation.

Empirically it also shows up in the physics, which is the check that would have
caught a plausible-looking but wrong mapping.  Driving the model with ``dof``
verbatim over the full 4950-clip release bank puts the feet on the floor (median
foot-to-floor clearance -3.8 mm, i.e. the light uniform penetration a
ground-aligned retarget leaves behind) and the median clip is fully supportable
(``infeasible_frac`` median 0.000).  Re-run on the same clips with the columns
permuted by ``MJ_TO_IL`` - the IsaacLab ordering, the one plausible way to get
this wrong - the median ``infeasible_frac`` jumps to 0.354, and to 0.148 under
the inverse permutation.  A wrong DOF order is not a subtle bias here; it makes
a third of every clip physically unsupportable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import functools
import hashlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, nnls

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    BONES_CSV_JOINT_NAMES,
    DOF_AXIS,
    NUM_DOF,
)
from gear_sonic.research.hygiene.motion_io import Motion, motion_sha256

__all__ = [
    "DEFAULT_G1_MJCF",
    "SCREEN_SCHEMA_VERSION",
    "ClipScreen",
    "ScreenThresholds",
    "load_model",
    "model_sha256",
    "screen_motion",
    "standing_root_height",
    "verify_dof_order",
]

SCREEN_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G1_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"

#: Name of the ground plane geom in the shipped G1 MJCF.
FLOOR_GEOM_NAME = "floor"

#: Substrings that mark a collision geom (or its body) as a foot.  The climb
#: screen looked for "foot"; the Unitree G1 MJCF has no geom names at all on its
#: collision geoms and calls the feet ``left/right_ankle_roll_link`` - the same
#: two links LACE pins as ``_EXPECTED_FOOT_LINKS``.  Both spellings are accepted
#: so the rule still works if a future MJCF renames them.
FOOT_NAME_TOKENS = ("foot", "ankle_roll")

#: Wrench-row weights that make newtons and newton-metres comparable at a ~0.5 m
#: lever before the L1 objective mixes them (verbatim from the climb reference).
WRENCH_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

#: Boxcar width (frames) used to smooth qvel and qacc, verbatim from the climb
#: reference.  Unlike the reference we replicate the edge samples instead of
#: zero-padding: ``np.convolve(..., mode="same")`` pulls the first/last two
#: velocity samples toward zero, which manufactures an acceleration spike - and
#: therefore a fake infeasible frame - at both ends of every clip.
SMOOTH_WINDOW = 5

#: Standard gravity used for the body-weight normalization.
GRAVITY = 9.81

#: ``mj_geomDistance`` clamps its answer at ``distmax``.  Contact candidacy only
#: needs ``gap_m``, but the reported foot clearance is a diagnostic the repair
#: operator uses, so foot geoms get a generous probe distance.
_FOOT_PROBE_DISTMAX = 2.0
_CANDIDATE_PROBE_MARGIN = 0.05

#: An LP is skipped (its optimum is provably 0) when the unconstrained NNLS
#: solve already explains the wrench to within this weighted residual norm and
#: stays inside the torque limits.
_TRIVIAL_RESIDUAL_TOL = 1e-6

_MODEL_SHA256: dict[int, str] = {}
_LAYOUT_CACHE: dict[int, "_RobotLayout"] = {}


@dataclass(frozen=True)
class ScreenThresholds:
    """Knobs that define what "supportable" means.  Recorded in every ClipScreen."""

    gap_m: float = 0.06
    mu: float = 0.7
    unsupported_force_frac: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "gap_m": float(self.gap_m),
            "mu": float(self.mu),
            "unsupported_force_frac": float(self.unsupported_force_frac),
        }


@dataclass(frozen=True)
class ClipScreen:
    """Clip-level dynamic-feasibility summary.

    Nullability is deliberate.  ``unsupported_force_N_*``, ``infeasible_frac`` and
    ``unsupported_impulse_per_weight_s`` are ``None`` when *no* frame produced a
    defined unsupported force - which happens only if every frame had contact
    candidates and every LP failed.  A failed solve is never coerced to 0; it is
    counted in ``torque_infeasible_frac`` and excluded from the statistics, and
    ``num_evaluable_frames`` says how many frames the statistics rest on.
    """

    motion_key: str
    num_frames: int
    fps: int
    airborne_frac: float
    infeasible_frac: float | None
    unsupported_force_N_p50: float | None
    unsupported_force_N_p95: float | None
    unsupported_force_N_max: float | None
    unsupported_impulse_per_weight_s: float | None
    max_tau_ratio_p95: float
    torque_infeasible_frac: float
    schema_version: int
    thresholds: dict[str, float]
    robot_sha256: str
    source_sha256: str
    # --- extras (defaulted, appended after the contract fields so positional
    # construction of the contract prefix keeps working) ---
    num_evaluable_frames: int = 0
    num_contact_free_frames: int = 0
    duration_s: float = 0.0
    body_weight_N: float = 0.0
    foot_gap_m_p50: float = 0.0
    foot_gap_m_p05: float = 0.0
    mean_contacts: float = 0.0
    max_tau_ratio_max: float = 0.0
    torque_saturated_frac: float = 0.0
    mjcf_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (all values are plain Python scalars, dicts, or None)."""
        return asdict(self)


@functools.lru_cache(maxsize=4)
def load_model(mjcf_path: str | Path = DEFAULT_G1_MJCF) -> mujoco.MjModel:
    """Compile the G1 MJCF once per process, with contacts disabled.

    Contacts are disabled on the model itself because every use in this module is
    inverse dynamics: we want the force the environment *must* supply, not the
    force MuJoCo's solver would have produced.  The returned model is cached and
    shared - treat it as read-only, and give every worker its own
    :class:`mujoco.MjData`.

    Raises ``ValueError`` if :func:`verify_dof_order` finds the MJCF joint layout
    inconsistent with the pkl's documented DOF order.
    """
    path = Path(mjcf_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    problems = verify_dof_order(model)
    if problems:
        raise ValueError(
            f"{path}: MJCF DOF layout does not match the motion pkl contract: {problems}"
        )
    _MODEL_SHA256[id(model)] = _sha256_file(path)
    return model


def model_sha256(model: mujoco.MjModel) -> str:
    """SHA-256 of the MJCF a model was compiled from, or ``"unknown"``."""
    return _MODEL_SHA256.get(id(model), "unknown")


def verify_dof_order(model: mujoco.MjModel) -> list[str]:
    """Check that MJCF joint order == the pkl ``dof`` column order.  See module docstring."""
    problems: list[str] = []
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)
    ]
    if model.njnt != NUM_DOF + 1:
        return [f"expected {NUM_DOF + 1} joints (1 free + {NUM_DOF} hinge), got {model.njnt}"]
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        problems.append(f"joint 0 ({joint_names[0]!r}) is not a free joint")
    if not np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE):
        problems.append("joints 1.. are not all hinges")
    if model.nu != NUM_DOF:
        problems.append(f"expected {NUM_DOF} actuators, got {model.nu}")
    else:
        driven = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        if driven != list(range(1, NUM_DOF + 1)):
            problems.append(f"actuator i does not drive joint i+1: {driven}")

    expected_names = [name[: -len("_dof")] for name in BONES_CSV_JOINT_NAMES]
    if joint_names[1:] != expected_names:
        mismatch = [
            (i, got, want)
            for i, (got, want) in enumerate(zip(joint_names[1:], expected_names))
            if got != want
        ]
        problems.append(f"MJCF joint names differ from the pkl column order at {mismatch[:5]}")

    if not np.allclose(
        np.asarray(model.jnt_axis[1:], dtype=np.float64), DOF_AXIS.astype(np.float64), atol=1e-9
    ):
        bad = np.where(
            ~np.isclose(np.asarray(model.jnt_axis[1:]), DOF_AXIS, atol=1e-9).all(axis=1)
        )[0]
        problems.append(f"MJCF joint axes differ from DOF_AXIS at joint indices {bad.tolist()}")

    if model.nq != NUM_DOF + 7 or model.nv != NUM_DOF + 6:
        problems.append(f"unexpected nq/nv: {model.nq}/{model.nv}")
    return problems


@dataclass(frozen=True)
class _RobotLayout:
    """Cached per-model index tables and limits."""

    floor_geom: int
    collision_geoms: tuple[int, ...]
    foot_geoms: tuple[int, ...]
    torque_limit: NDArray[np.float64]
    actuated_dof_rows: NDArray[np.int64]
    total_mass_kg: float
    body_weight_N: float
    geom_body: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _build_layout(model: mujoco.MjModel) -> _RobotLayout:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    if floor < 0:
        planes = [
            i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        if not planes:
            raise ValueError(
                f"model has no geom named {FLOOR_GEOM_NAME!r} and no plane geom to fall back on"
            )
        floor = planes[-1]

    def geom_name(index: int) -> str:
        return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").lower()

    def body_name(index: int) -> str:
        return (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])) or ""
        ).lower()

    collision = tuple(
        i
        for i in range(model.ngeom)
        if i != floor and (model.geom_contype[i] or model.geom_conaffinity[i])
    )
    feet = tuple(
        i
        for i in collision
        if any(token in geom_name(i) or token in body_name(i) for token in FOOT_NAME_TOKENS)
    )
    if not feet:
        raise ValueError(
            f"no foot collision geoms matched {FOOT_NAME_TOKENS} - airborne_frac would be meaningless"
        )

    limits = np.empty(model.nu, dtype=np.float64)
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if model.actuator_forcelimited[actuator] and model.actuator_forcerange[actuator, 1] > 0:
            limits[actuator] = float(model.actuator_forcerange[actuator, 1])
        elif model.jnt_actfrclimited[joint] and model.jnt_actfrcrange[joint, 1] > 0:
            limits[actuator] = float(model.jnt_actfrcrange[joint, 1])
        else:
            limits[actuator] = np.inf
    if not np.isfinite(limits).any():
        raise ValueError("no actuator torque limits found on either the actuators or the joints")

    rows = np.array(
        [int(model.jnt_dofadr[int(model.actuator_trnid[i, 0])]) - 6 for i in range(model.nu)]
    )
    mass = float(model.body_mass.sum())
    return _RobotLayout(
        floor_geom=int(floor),
        collision_geoms=collision,
        foot_geoms=feet,
        torque_limit=limits,
        actuated_dof_rows=rows.astype(np.int64),
        total_mass_kg=mass,
        body_weight_N=mass * GRAVITY,
        geom_body=np.asarray(model.geom_bodyid, dtype=np.int64).copy(),
    )


def _robot_layout(model: mujoco.MjModel) -> _RobotLayout:
    """Index tables and limits for a model, cached by object identity.

    ``MjModel`` is not hashable, so ``functools.lru_cache`` is not an option; the
    cache is keyed on ``id`` and the models it holds are the ones
    :func:`load_model` keeps alive for the life of the process.
    """
    layout = _LAYOUT_CACHE.get(id(model))
    if layout is None:
        layout = _build_layout(model)
        _LAYOUT_CACHE[id(model)] = layout
    return layout


def standing_root_height(model: mujoco.MjModel) -> float:
    """Root height at which the lowest foot contact point rests exactly on z = 0.

    Zero joint angles, identity root rotation.  Used by tests (and available to
    the repair operator) to build a physically grounded synthetic pose instead of
    hard-coding a magic number lifted from one MJCF revision.
    """
    layout = _robot_layout(model)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6)
    lowest = min(
        float(
            mujoco.mj_geomDistance(
                model, data, geom, layout.floor_geom, _FOOT_PROBE_DISTMAX, fromto
            )
        )
        for geom in layout.foot_geoms
    )
    return -lowest


def screen_motion(
    motion: Motion,
    model: mujoco.MjModel | None = None,
    thresholds: ScreenThresholds = ScreenThresholds(),
    *,
    source_sha256: str | None = None,
) -> ClipScreen:
    """Screen one clip for dynamic feasibility.  See the module docstring for the method.

    ``motion`` is screened at its own ``fps``; resample it first (SONIC trains at
    50 Hz) if you want the timeline the policy actually sees.  ``source_sha256``
    defaults to a content digest of the arrays, so the result is reproducible
    from the clip alone.
    """
    if model is None:
        model = load_model()
    layout = _robot_layout(model)
    data = mujoco.MjData(model)

    frames = motion.num_frames
    if frames < 2:
        raise ValueError(f"{motion.key}: need at least 2 frames to finite-difference, got {frames}")
    fps = int(motion.fps)
    dt = 1.0 / float(fps)

    qpos, qvel, qacc = _reference_trajectory(model, motion, dt)

    weight = layout.body_weight_N
    unsupported_threshold = float(thresholds.unsupported_force_frac) * weight
    cone = _cone_basis(float(thresholds.mu))
    gap = float(thresholds.gap_m)
    candidate_distmax = gap + _CANDIDATE_PROBE_MARGIN
    torque_limit = layout.torque_limit
    act_rows = layout.actuated_dof_rows

    unsupported = np.full(frames, np.nan)
    lp_failed = np.zeros(frames, dtype=bool)
    contact_free = np.zeros(frames, dtype=bool)
    foot_gap = np.full(frames, np.inf)
    tau_ratio = np.zeros(frames)
    contact_counts = np.zeros(frames, dtype=np.int64)

    fromto = np.zeros(6)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        data.qacc[:] = qacc[frame]
        mujoco.mj_inverse(model, data)
        wrench = np.array(data.qfrc_inverse[:6], dtype=np.float64)
        tau_free = np.array(data.qfrc_inverse[6:], dtype=np.float64)
        mujoco.mj_forward(model, data)

        candidates: list[tuple[int, NDArray[np.float64]]] = []
        lowest_foot = np.inf
        for geom in layout.collision_geoms:
            is_foot = geom in layout.foot_geoms
            distmax = _FOOT_PROBE_DISTMAX if is_foot else candidate_distmax
            distance = float(
                mujoco.mj_geomDistance(model, data, geom, layout.floor_geom, distmax, fromto)
            )
            if is_foot:
                lowest_foot = min(lowest_foot, distance)
            if distance <= gap:
                candidates.append((geom, fromto[:3].copy()))
        foot_gap[frame] = lowest_foot
        contact_counts[frame] = len(candidates)

        if not candidates:
            contact_free[frame] = True
            unsupported[frame] = float(np.linalg.norm(wrench[:3]))
            tau_ratio[frame] = float(np.max(np.abs(tau_free[act_rows]) / torque_limit))
            continue

        wrench_cols: list[NDArray[np.float64]] = []
        torque_cols: list[NDArray[np.float64]] = []
        for geom, point in candidates:
            mujoco.mj_jac(model, data, jacp, jacr, point, int(layout.geom_body[geom]))
            for edge in cone:
                generalized = jacp.T @ edge
                wrench_cols.append(generalized[:6])
                torque_cols.append(generalized[6:])
        contact_map = np.stack(wrench_cols, axis=1)
        torque_map = np.stack(torque_cols, axis=1)

        weighted_map = contact_map * WRENCH_WEIGHTS[:, None]
        weighted_wrench = wrench * WRENCH_WEIGHTS
        nnls_force, _ = nnls(weighted_map, weighted_wrench)
        nnls_residual = weighted_wrench - weighted_map @ nnls_force
        nnls_tau = tau_free - torque_map @ nnls_force
        saturation = np.abs(nnls_tau[act_rows]) / torque_limit
        tau_ratio[frame] = float(np.max(saturation))

        if (
            float(np.linalg.norm(nnls_residual)) <= _TRIVIAL_RESIDUAL_TOL
            and float(saturation.max()) <= 1.0
        ):
            # The NNLS solution is already LP-feasible with objective 0, so the
            # LP optimum is exactly 0.  Skipping it here is an identity, not an
            # approximation, and it removes most of the LP cost on clean clips.
            unsupported[frame] = 0.0
            continue

        solved = _torque_limited_lp(
            weighted_map,
            weighted_wrench,
            torque_map[act_rows],
            tau_free[act_rows],
            torque_limit,
        )
        if solved is None:
            lp_failed[frame] = True
            continue
        residual = (weighted_wrench - weighted_map @ solved) / WRENCH_WEIGHTS
        unsupported[frame] = float(np.linalg.norm(residual[:3]))

    evaluable = ~np.isnan(unsupported)
    n_evaluable = int(evaluable.sum())
    values = unsupported[evaluable]

    return ClipScreen(
        motion_key=motion.key,
        num_frames=frames,
        fps=fps,
        airborne_frac=float(np.mean(foot_gap > gap)),
        infeasible_frac=(float(np.mean(values > unsupported_threshold)) if n_evaluable else None),
        unsupported_force_N_p50=(float(np.percentile(values, 50)) if n_evaluable else None),
        unsupported_force_N_p95=(float(np.percentile(values, 95)) if n_evaluable else None),
        unsupported_force_N_max=(float(values.max()) if n_evaluable else None),
        unsupported_impulse_per_weight_s=(
            float((values / weight).sum() * dt) if n_evaluable else None
        ),
        max_tau_ratio_p95=float(np.percentile(tau_ratio, 95)),
        torque_infeasible_frac=float(np.mean(lp_failed)),
        schema_version=SCREEN_SCHEMA_VERSION,
        thresholds=thresholds.to_dict(),
        robot_sha256=model_sha256(model),
        source_sha256=source_sha256 if source_sha256 is not None else motion_sha256(motion),
        num_evaluable_frames=n_evaluable,
        num_contact_free_frames=int(contact_free.sum()),
        duration_s=float(frames) * dt,
        body_weight_N=float(weight),
        foot_gap_m_p50=float(np.percentile(foot_gap, 50)),
        foot_gap_m_p05=float(np.percentile(foot_gap, 5)),
        mean_contacts=float(contact_counts.mean()),
        max_tau_ratio_max=float(tau_ratio.max()),
        torque_saturated_frac=float(np.mean(tau_ratio > 1.0)),
        mjcf_path=str(DEFAULT_G1_MJCF),
    )


def _reference_trajectory(
    model: mujoco.MjModel, motion: Motion, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build ``qpos``/``qvel``/``qacc`` in MuJoCo layout from a clip.

    ``mj_differentiatePos`` is used for ``qvel`` rather than a naive array
    difference: for the free joint it returns the linear velocity in world axes
    and the angular velocity in the *body* frame, which is exactly MuJoCo's
    ``qvel`` convention for a free joint (and what the climb reference built by
    hand with ``R.T @ omega_world``).
    """
    frames = motion.num_frames
    qpos = np.zeros((frames, model.nq))
    qpos[:, 0:3] = motion.root_trans_offset.astype(np.float64)
    quat_xyzw = motion.root_rot.astype(np.float64)
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    quat_xyzw = quat_xyzw / norms
    qpos[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = motion.dof.astype(np.float64)

    qvel = np.zeros((frames, model.nv))
    scratch = np.zeros(model.nv)
    for frame in range(frames):
        lo = max(frame - 1, 0)
        hi = min(frame + 1, frames - 1)
        mujoco.mj_differentiatePos(model, scratch, (hi - lo) * dt, qpos[lo], qpos[hi])
        qvel[frame] = scratch
    qvel = _smooth(qvel, SMOOTH_WINDOW)
    qacc = _smooth(np.gradient(qvel, dt, axis=0), SMOOTH_WINDOW)
    return qpos, qvel, qacc


def _smooth(signal: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Boxcar smooth along the frame axis with edge replication (see SMOOTH_WINDOW)."""
    if window <= 1 or signal.shape[0] < 2:
        return signal
    window = min(window, signal.shape[0])
    pad = window // 2
    padded = np.pad(signal, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(signal.shape[1])], axis=1
    )
    return smoothed[: signal.shape[0]]


def _cone_basis(mu: float) -> NDArray[np.float64]:
    """Four pyramidal friction-cone edges for a +z contact normal (flat floor)."""
    normal = np.array([0.0, 0.0, 1.0])
    if mu <= 0.0:
        return normal[None, :]
    tangent_x = np.array([1.0, 0.0, 0.0])
    tangent_y = np.array([0.0, 1.0, 0.0])
    return np.stack(
        [
            normal + mu * tangent_x,
            normal - mu * tangent_x,
            normal + mu * tangent_y,
            normal - mu * tangent_y,
        ]
    )


def _torque_limited_lp(
    contact_map: NDArray[np.float64],
    wrench: NDArray[np.float64],
    torque_map: NDArray[np.float64],
    tau_free: NDArray[np.float64],
    torque_limit: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """``min sum|A f - W|`` s.t. ``f >= 0`` and ``|tau_free - J f| <= tau_max``.

    Returns the contact force coefficients, or ``None`` when the LP is infeasible
    or the solver fails.  ``None`` means "no admissible contact force keeps the
    joint torques inside their limits at all" - a strictly worse outcome than a
    large residual, and the caller must not fold it into a 0.
    """
    n_forces = contact_map.shape[1]
    n_rows = contact_map.shape[0]
    finite = np.isfinite(torque_limit)

    objective = np.concatenate([np.zeros(n_forces), np.ones(n_rows)])
    slack = -np.eye(n_rows)
    blocks = [np.hstack([contact_map, slack]), np.hstack([-contact_map, slack])]
    bounds_rhs = [wrench, -wrench]
    if finite.any():
        limited = torque_map[finite]
        pad = np.zeros((limited.shape[0], n_rows))
        blocks.append(np.hstack([-limited, pad]))
        blocks.append(np.hstack([limited, pad]))
        bounds_rhs.append(torque_limit[finite] - tau_free[finite])
        bounds_rhs.append(torque_limit[finite] + tau_free[finite])

    result = linprog(
        objective,
        A_ub=np.vstack(blocks),
        b_ub=np.concatenate(bounds_rhs),
        bounds=[(0.0, None)] * (n_forces + n_rows),
        method="highs",
    )
    if not result.success:
        return None
    return np.asarray(result.x[:n_forces], dtype=np.float64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
