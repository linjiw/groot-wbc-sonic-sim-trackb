"""Detect a reference pose the robot cannot hold, using the robot's own collision model.

The joint-limit prefilter catches references asking for angles outside the G1's range. It
does not catch the failure that killed every squat: Kimodo's human prior commands a waist
and hip fold the G1 cannot make, the joints are *clamped* to their limits -- so every angle
reads in-range -- and the resulting pose puts the thigh through the pelvis. Measured, the
three ``squat_pick`` references scored 0.0078, 0.0000 and 0.0000 on joint saturation, well
under the 0.009 screening threshold, and then failed 0-for-3 in simulation.

That failure is geometric, so it needs a geometric instrument: forward kinematics on the
reference qpos, then the model's own collision geometry.

**Two instruments were tried and one was wrong, which is worth recording.** The first
attempt reused the 29-capsule set built for the swept-volume work. It flagged all 164
episodes, dominated by ``torso_link~left_elbow_link`` at 0.10 m, and correlated +0.12 with
executed self-contact -- useless. The reason is that those capsules are a deliberately
*conservative outer approximation*: over-estimating the envelope is the safe direction for
clearance, and the exact wrong direction for self-collision, where it manufactures overlap
in every ordinary pose. The capsule set is not a self-collision model and cannot be made
into one by tuning an exclusion list.

The model's own collision geometry works because it was authored for this. Measured over
164 episodes, penetration between the pelvis and a hip-roll link in the *reference*:

| Reference depth | Episodes | Accepted | Worst executed self-contact |
|---|---|---|---|
| 0 m            | 95 | 92% |   367 N |
| 0.00–0.05 m    |  9 | 78% |   945 N |
| 0.05–0.10 m    | 57 | 81% |  1809 N |
| **above 0.10 m** | **3** | **0%** | **9686 N** |

Moderate pelvis-to-hip overlap is normal in the reference and must not be screened -- the
0.05–0.10 m band still accepts at 81%. Only the deep band is disqualifying, and it contains
exactly the crouch and squat references that failed.

The threshold rests on three episodes, which is enough to screen on and not enough to call
calibrated. It should be re-derived as the behaviour library widens, and the screen reports
by default rather than skipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Reference penetration between the pelvis and a hip-roll link, above which the pose asks
#: for a fold the robot cannot make. Placed at the observed boundary: every episode above it
#: was rejected, while the band immediately below still accepts at 81%.
DEFAULT_PELVIS_HIP_LIMIT_M = 0.10

#: Only every Nth frame is checked. Self-intersection from a clamped joint persists across
#: many frames -- the failing references intersect on 98-100% of them -- so sampling costs
#: nothing in sensitivity and a quarter of the time.
DEFAULT_FRAME_STRIDE = 4

#: The MJCF whose qpos convention `reference_g1_qpos` follows. Its 36 columns and 14
#: collision-bearing links match the recorded reference exactly.
DEFAULT_G1_MJCF = Path.home() / "kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


class SelfIntersectionError(ValueError):
    """Raised when self-intersection cannot be assessed from the inputs given."""


@dataclass(frozen=True)
class SelfIntersectionReport:
    """Where and how deeply a reference pose intersects itself."""

    frames_checked: int
    #: Deepest penetration per link pair, worst first.
    pair_depths: tuple[tuple[str, str, float], ...]
    #: Deepest penetration between the pelvis and a hip-roll link, the pair that
    #: discriminates an unreachable fold.
    pelvis_hip_depth_m: float
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_depth_m(self) -> float:
        return self.pair_depths[0][2] if self.pair_depths else 0.0


def _load_model(mjcf_path: str | Path):
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise SelfIntersectionError(
            "mujoco is required to evaluate reference self-intersection"
        ) from error
    path = Path(mjcf_path)
    if not path.exists():
        raise SelfIntersectionError(f"MJCF not found: {path}")
    return mujoco, mujoco.MjModel.from_xml_path(str(path))


def check_reference_self_intersection(
    reference_qpos: np.ndarray,
    *,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    pelvis_hip_limit_m: float = DEFAULT_PELVIS_HIP_LIMIT_M,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
) -> SelfIntersectionReport:
    """Run FK on a reference clip and report body-body interpenetration.

    ``reference_qpos`` is the 36-column MuJoCo layout the corpus records: 3 root
    translation, 4 root quaternion, 29 joint DOFs.
    """
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise SelfIntersectionError(f"expected (T, nq) reference qpos, got {qpos.shape}")
    if frame_stride < 1:
        raise SelfIntersectionError(f"frame_stride must be >= 1, got {frame_stride}")

    mujoco, model = _load_model(mjcf_path)
    if qpos.shape[1] != model.nq:
        raise SelfIntersectionError(
            f"reference has {qpos.shape[1]} columns but the model expects {model.nq}"
        )
    data = mujoco.MjData(model)

    depths: dict[tuple[str, str], float] = {}
    checked = 0
    for index in range(0, len(qpos), frame_stride):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        checked += 1
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if contact.dist >= 0.0:
                continue
            first = model.geom_bodyid[contact.geom1]
            second = model.geom_bodyid[contact.geom2]
            # The floor is body 0. A foot resting on it is not self-intersection, and
            # including it made the ground contact dominate every result.
            if first == second or first == 0 or second == 0:
                continue
            key = tuple(
                sorted(
                    (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second),
                    )
                )
            )
            depths[key] = max(depths.get(key, 0.0), float(-contact.dist))

    ordered = tuple(
        (pair[0], pair[1], depth)
        for pair, depth in sorted(depths.items(), key=lambda item: -item[1])
    )
    pelvis_hip = max(
        (
            depth
            for (first, second, depth) in ordered
            if "pelvis" in (first, second) and any("hip_roll" in n for n in (first, second))
        ),
        default=0.0,
    )

    reasons: list[str] = []
    if pelvis_hip > pelvis_hip_limit_m:
        reasons.append(
            f"pelvis_hip_interpenetration {pelvis_hip:.3f} > {pelvis_hip_limit_m:.3f} m"
        )

    return SelfIntersectionReport(
        frames_checked=checked,
        pair_depths=ordered,
        pelvis_hip_depth_m=pelvis_hip,
        passed=not reasons,
        reasons=tuple(reasons),
    )
