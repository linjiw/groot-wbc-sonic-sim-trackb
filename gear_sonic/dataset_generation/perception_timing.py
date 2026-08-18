"""Was the obstacle visible before the robot started adapting to it?

A counterfactual family shows which prescribed behaviour physics permits. It does not show
that a robot could have *chosen* it, and the difference decides what the data can be used for.
If the adaptation begins before the obstacle enters the camera, the pair supports
map-conditioned selection and nothing more; a model trained on it and evaluated from ego
observations would be learning to guess, and might score well by always adapting.

So the three moments are measured and their order checked:

    t_first_visible  <  t_adaptation_onset  <  t_bottleneck

The first is when the obstacle enters the ego camera's frustum. The second is when the adapted
motion first departs from the nominal. The third is when the robot is closest to the obstacle.

This is the argument for a *local* adaptation window rather than a whole-route one. A clip
crouched from frame zero fails the first inequality by construction: nothing was visible yet.

**Visibility here is geometric, not rendered.** It asks whether the obstacle's box falls inside
the camera frustum with a clear line of sight assumed. A rendered check would also account for
occlusion by the robot's own body and by other furniture, which can only make ``t_first_visible``
later, so this is the optimistic bound -- and an optimistic bound is the right thing to fail
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .self_intersection import DEFAULT_G1_MJCF

#: Ego camera, from `manager_env/base_env.yaml`. Mounted on head_link beneath torso_link.
CAMERA_LINK = "head_link"
CAMERA_OFFSET_M = (0.09, 0.0, 0.43)
FOCAL_LENGTH_MM = 1.88
HORIZONTAL_APERTURE_MM = 2.6035
VERTICAL_APERTURE_MM = 1.9526
CLIPPING_RANGE_M = (0.05, 30.0)


def field_of_view_rad() -> tuple[float, float]:
    """Horizontal and vertical FOV from the recorded intrinsics."""
    return (
        2.0 * np.arctan(0.5 * HORIZONTAL_APERTURE_MM / FOCAL_LENGTH_MM),
        2.0 * np.arctan(0.5 * VERTICAL_APERTURE_MM / FOCAL_LENGTH_MM),
    )


@dataclass(frozen=True)
class PerceptionTiming:
    """When the obstacle appeared, when the robot reacted, and when it mattered."""

    t_first_visible: int | None
    t_adaptation_onset: int | None
    t_bottleneck: int | None
    fps: float

    @property
    def ordering_holds(self) -> bool:
        """Whether the robot could have been reacting to what it saw."""
        moments = (self.t_first_visible, self.t_adaptation_onset, self.t_bottleneck)
        if any(m is None for m in moments):
            return False
        return moments[0] < moments[1] < moments[2]

    @property
    def reaction_window_s(self) -> float:
        """Seconds between the obstacle becoming visible and the adaptation starting."""
        if self.t_first_visible is None or self.t_adaptation_onset is None:
            return float("nan")
        return (self.t_adaptation_onset - self.t_first_visible) / self.fps

    def verdict(self) -> str:
        if self.t_first_visible is None:
            return "the obstacle never enters the camera; ego selection is not supported"
        if self.t_adaptation_onset is None:
            return "the adapted motion never departs from the nominal"
        if self.t_adaptation_onset <= self.t_first_visible:
            return (
                f"adaptation begins at frame {self.t_adaptation_onset}, before the obstacle "
                f"is visible at {self.t_first_visible} -- map-conditioned only"
            )
        if self.t_bottleneck is not None and self.t_bottleneck <= self.t_adaptation_onset:
            return "the robot reaches the obstacle before it starts adapting"
        return (
            f"visible at {self.t_first_visible}, adapting at {self.t_adaptation_onset}, "
            f"closest at {self.t_bottleneck} -- {self.reaction_window_s:.2f} s to react"
        )


def perception_timing(
    nominal_qpos: np.ndarray,
    adapted_qpos: np.ndarray,
    obstacle_box: tuple[float, float, float, float, float, float],
    *,
    fps: float = 30.0,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    onset_tolerance_rad: float = 0.01,
) -> PerceptionTiming:
    """Locate the three moments for one adapted/nominal pair against one obstacle."""
    from .reference_payload import payload_from_reference

    nominal = np.asarray(nominal_qpos, dtype=np.float64)
    adapted = np.asarray(adapted_qpos, dtype=np.float64)
    if nominal.shape != adapted.shape:
        raise ValueError(
            f"clips must be the same shape to compare frame by frame; "
            f"got {nominal.shape} and {adapted.shape}"
        )

    payload = payload_from_reference(nominal, fps=fps, mjcf_path=mjcf_path)
    names = list(payload["body_names"])
    if CAMERA_LINK not in names:
        # The G1 model may carry the camera on the torso when no head link exists.
        link = "torso_link" if "torso_link" in names else names[0]
    else:
        link = CAMERA_LINK
    index = names.index(link)
    positions = np.asarray(payload["body_pos_w"], dtype=np.float64)[:, index, :]
    quats = np.asarray(payload["body_quat_w"], dtype=np.float64)[:, index, :]

    # Camera position and forward axis in world frame.
    w, x, y, z = (quats[:, i] for i in range(4))
    forward = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)
    ], axis=1)
    left = np.stack([
        2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)
    ], axis=1)
    up = np.stack([
        2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
    ], axis=1)
    offset = np.asarray(CAMERA_OFFSET_M)
    camera = positions + forward * offset[0] + left * offset[1] + up * offset[2]

    min_x, min_y, min_z, max_x, max_y, max_z = obstacle_box
    corners = np.array([
        [cx, cy, cz]
        for cx in (min_x, max_x) for cy in (min_y, max_y) for cz in (min_z, max_z)
    ])
    h_fov, v_fov = field_of_view_rad()
    near, far = CLIPPING_RANGE_M

    visible = np.zeros(len(camera), dtype=bool)
    for frame in range(len(camera)):
        rays = corners - camera[frame]
        depth = rays @ forward[frame]
        within = (depth > near) & (depth < far)
        if not within.any():
            continue
        lateral = np.abs(rays @ left[frame])
        vertical = np.abs(rays @ up[frame])
        safe = np.maximum(depth, 1e-6)
        inside = (
            within
            & (np.arctan2(lateral, safe) < h_fov / 2)
            & (np.arctan2(vertical, safe) < v_fov / 2)
        )
        visible[frame] = bool(inside.any())

    first_visible = int(np.argmax(visible)) if visible.any() else None

    departure = np.abs(adapted[:, 7:] - nominal[:, 7:]).max(axis=1)
    moved = departure > onset_tolerance_rad
    onset = int(np.argmax(moved)) if moved.any() else None

    # Bottleneck: the middle of the closest-approach interval, not its first frame.
    # Once the head is inside the obstacle's x-span the clipped distance stops changing, so a
    # bare argmin over a plateau returns whichever frame the robot entered on -- which put the
    # bottleneck at the same frame as the adaptation onset and read as the robot arriving
    # before it reacted.
    centre = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    half = np.array([(max_x - min_x) / 2, (max_y - min_y) / 2, (max_z - min_z) / 2])
    delta = np.abs(positions - centre) - half
    distance = np.linalg.norm(np.clip(delta, 0.0, None), axis=1)
    closest = np.flatnonzero(distance <= distance.min() + 1e-9)
    bottleneck = int(np.median(closest))

    return PerceptionTiming(
        t_first_visible=first_visible,
        t_adaptation_onset=onset,
        t_bottleneck=bottleneck,
        fps=fps,
    )
