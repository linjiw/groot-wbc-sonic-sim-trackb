"""Reject untrackable reference motions before spending a rollout on them.

A generated motion costs milliseconds; simulating it costs minutes. At Phase B scale --
hundreds of prompts, several seeds each -- the difference decides how much of the GPU
calendar goes to motions that were never going to work.

The signal is joint-limit saturation. Kimodo generates human motion which is then mapped
onto the G1's joint ranges, and configurations the robot cannot reach get clamped at their
limits. A clamped configuration is not merely inaccurate: it changes the pose, and a
crouch clamped at the hips puts the thigh through the pelvis. That is what the physics
sees, and the resulting self-contact is real rather than an artifact of a strict gate.

Validated over 11 distinct source motions (78 rollouts): the correlation between
pre-simulation saturation and peak self-contact force is **0.933**, monotone in the tail --
saturation at or below 0.0071 never exceeded 318 N, while 0.0116 reached 712 N, 0.0199
reached 1101 N, and 0.0444 reached 9686 N. Every accepted motion sat at or below 0.0071.
Full numbers in ``docs/motion_prefilter_validation.md``.

**The threshold depends on which representation you screen.** The same motion measures
0.0444 as a raw Kimodo CSV and 0.1391 as the ``reference_g1_qpos`` the tracker was actually
commanded, because SONIC transforms the reference on the way in. Applying one scale's
threshold to the other under-screens badly -- the raw crouch reads below a
recorded-reference threshold despite being the motion that failed. Two constants are
therefore defined here, and the caller must pick the one matching its input.

Two limits on what this screen claims:

* **It predicts self-contact, not acceptance.** Plenty of low-saturation motions are still
  rejected, for path error or other reasons this signal knows nothing about. Screening on
  it removes untrackable motions, not bad ones in general.
* **The evidence is 11 motions from a behaviourally narrow corpus.** The threshold should
  be re-derived as the behaviour library widens, and the screen defaults to *reporting*
  rather than skipping: a rollout wrongly skipped is a behaviour permanently absent from
  the corpus, while a rollout wrongly run only costs GPU time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

#: Threshold for a raw Kimodo qpos CSV (or the adapter's ``dof``, which matches it). This
#: is the screening point that matters: it is reachable before any GPU time is spent.
#: Placed in the measured gap between the highest-saturation source motion that stayed
#: under the 343 N self-contact gate (0.0071, peak 318 N) and the lowest that blew through
#: it (0.0116, peak 712 N).
PRESIM_SATURATION_LIMIT = 0.009

#: Threshold for a recorded ``reference_g1_qpos``, which sits on a different scale because
#: SONIC transforms the reference. Used for post-hoc analysis of finished rollouts, never
#: interchangeably with the pre-simulation limit. Gap: 0.0554 (peak 318 N) to 0.0600
#: (peak 711 N).
RECORDED_SATURATION_LIMIT = 0.058

#: Default for the common case, which is screening before simulating.
DEFAULT_SATURATION_LIMIT = PRESIM_SATURATION_LIMIT

#: How close to a limit counts as clamped, as a fraction of the joint's range.
SATURATION_EPSILON = 0.01

#: Root height below which a reference is a deep crouch rather than a walk. Reported, not
#: screened on: crouching is a behaviour the corpus wants, so it must not be filtered away
#: simply for being low.
CROUCH_ROOT_HEIGHT_M = 0.55


class PrefilterError(ValueError):
    """Raised when a motion cannot be screened with the inputs given."""


@dataclass(frozen=True)
class PrefilterReport:
    """Screening result for one reference motion."""

    saturated_cell_fraction: float
    saturated_frame_fraction: float
    #: Joint names ordered by how often they sat at a limit, worst first.
    worst_joints: tuple[tuple[str, float], ...]
    root_height_min_m: float
    frames: int
    passed: bool
    reasons: tuple[str, ...]

    @property
    def is_crouch(self) -> bool:
        return self.root_height_min_m < CROUCH_ROOT_HEIGHT_M


def load_joint_limits(mjcf_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read per-joint ranges from a MuJoCo XML, resolving class defaults.

    Only 18 of the G1's 29 joints carry an explicit ``range``; the rest inherit one from
    their ``default`` class. Reading only the explicit ones would leave a third of the
    joints unscreened while appearing to work.
    """
    tree = ElementTree.parse(Path(mjcf_path))
    root = tree.getroot()

    class_ranges: dict[str, list[float]] = {}
    for default in root.iter("default"):
        name = default.get("class")
        if not name:
            continue
        for joint in default.findall("joint"):
            if joint.get("range"):
                class_ranges[name] = [float(v) for v in joint.get("range").split()]

    names: list[str] = []
    limits: list[list[float]] = []
    for joint in root.iter("joint"):
        joint_name = joint.get("name")
        if joint_name is None:
            continue
        explicit = joint.get("range")
        if explicit:
            limits.append([float(v) for v in explicit.split()])
        elif joint.get("class") in class_ranges:
            limits.append(list(class_ranges[joint.get("class")]))
        else:
            raise PrefilterError(f"joint {joint_name!r} has no range and no class default")
        names.append(joint_name)

    if not names:
        raise PrefilterError(f"no named joints found in {mjcf_path}")
    return names, np.asarray(limits, dtype=np.float64)


def screen_reference_motion(
    qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    *,
    saturation_limit: float = DEFAULT_SATURATION_LIMIT,
    epsilon: float = SATURATION_EPSILON,
) -> PrefilterReport:
    """Screen a 36-column reference for configurations the G1 cannot reach.

    ``qpos`` is the standard layout: 3 root translation, 4 root quaternion, then the
    joint DOFs.
    """
    data = np.asarray(qpos, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 8:
        raise PrefilterError(f"expected (T, 7+J) qpos, got {data.shape}")

    dofs = data[:, 7:]
    count = min(dofs.shape[1], joint_limits.shape[0])
    if count == 0:
        raise PrefilterError("no joints to screen")

    low, high = joint_limits[:count, 0], joint_limits[:count, 1]
    span = high - low
    if np.any(span <= 0):
        raise PrefilterError("joint limit range must be positive")

    normalised = (dofs[:, :count] - low) / span
    saturated = (normalised < epsilon) | (normalised > 1.0 - epsilon)

    per_joint = saturated.mean(axis=0)
    order = np.argsort(-per_joint)
    worst = tuple(
        (joint_names[i], float(per_joint[i])) for i in order[:5] if per_joint[i] > 0.0
    )

    cell_fraction = float(saturated.mean())
    reasons: list[str] = []
    if cell_fraction > saturation_limit:
        reasons.append(
            f"joint_saturation {cell_fraction:.3f} > {saturation_limit:.3f}"
        )

    return PrefilterReport(
        saturated_cell_fraction=cell_fraction,
        saturated_frame_fraction=float(saturated.any(axis=1).mean()),
        worst_joints=worst,
        root_height_min_m=float(data[:, 2].min()),
        frames=int(data.shape[0]),
        passed=not reasons,
        reasons=tuple(reasons),
    )
