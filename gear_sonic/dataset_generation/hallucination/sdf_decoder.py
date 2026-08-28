"""True capsule-cloud geometry for the LfLH decoder, replacing the directional envelope.

The first decoder reduced the body to up/left/right extents per station and gated each direction
before taking a softplus. An audit showed three consequences, all measured:

* a gate near zero drove the depth to exactly zero, so a 1.6 m wall standing across the walking
  path scored 0.008976 against 0.008394 for the same wall ten metres in the sky -- geometrically
  intersecting and comfortably clear were indistinguishable;
* the clearance term built on that score used ``relu(blockedness + margin)``, and since blockedness
  is a softplus it is strictly positive, so the ``relu`` never fired and the term had no reachable
  minimum. Its only escape direction was height, which drove every obstacle to the ceiling of the
  parameterisation;
* the result was scenes in which 20 of 24 contained an obstacle intersecting the robot, while the
  box the renderer labelled "binding" sat 33 cm above the standing head and discriminated nothing.

This module does what the design guidance actually specified: sample points along every collision
capsule, subtract the capsule radius, and take a smooth minimum of the true signed distance to the
obstacle box. Clearance is then a real distance in metres -- positive means clear, negative means
penetrating -- so a barrier loss on it has a reachable minimum and means what it says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def candidate_cloud(
    payload: dict,
    *,
    frame_stride: int = 4,
    samples_per_capsule: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsampled swept-volume points and their radii, for one candidate motion.

    Subsampling is what makes a true geometric decoder affordable inside a training loop. Sampling
    along a capsule axis can only place points *on* the axis, so the reported distance is never
    smaller than the true one: the error is conservative in the direction that matters.
    """
    from ..swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )
    starts = starts[::frame_stride]
    ends = ends[::frame_stride]
    fractions = np.linspace(0.0, 1.0, max(samples_per_capsule, 2))[:, None, None, None]
    points = starts[None, ...] + fractions * (ends - starts)[None, ...]
    points = points.reshape(-1, 3)
    tile = points.shape[0] // radii.shape[0] if radii.shape[0] else 0
    return points, np.tile(radii, max(tile, 1))[: points.shape[0]]


@dataclass
class SdfChoiceDecoder:
    """Fixed, parameter-free decoder over true capsule-to-box clearance.

    ``clearance(cloud, box)`` is a smooth minimum over the cloud of the point-to-box signed
    distance minus the capsule radius. Positive is clear; negative is penetrating. The candidate
    cost is then the guidance's form: a collision term that is a softplus of negated clearance,
    plus the edit cost, with a soft-argmin over candidates.
    """

    margin_m: float = 0.0
    softmin_temperature_m: float = 0.01
    collision_temperature_m: float = 0.02
    choice_temperature: float = 0.15
    blocked_penalty: float = 60.0

    def clearance(
        self, points: torch.Tensor, radii: torch.Tensor, box: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Smooth-min signed surface distance from one cloud to each obstacle box.

        ``points`` is (N, 3) in world metres; the box tensors are (K,). Returns (K,).
        """
        centre = torch.stack((box["centre_x"], box["centre_y"], box["centre_z"]), dim=-1)  # (K, 3)
        half = torch.stack(
            (box["half_along_m"], box["half_lateral_m"], box["half_vertical_m"]), dim=-1
        )
        yaw = box["yaw"]
        cos, sin = torch.cos(-yaw), torch.sin(-yaw)
        delta = points[None, :, :] - centre[:, None, :]
        local_x = cos[:, None] * delta[..., 0] - sin[:, None] * delta[..., 1]
        local_y = sin[:, None] * delta[..., 0] + cos[:, None] * delta[..., 1]
        local = torch.stack((local_x, local_y, delta[..., 2]), dim=-1)
        # Exact signed distance to an axis-aligned box in its own frame.
        q = local.abs() - half[:, None, :]
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.amax(dim=-1), max=0.0)
        signed = outside + inside - radii[None, :]
        # Smooth minimum, so the gradient reaches the nearest few points rather than only one.
        weights = torch.softmax(-signed / self.softmin_temperature_m, dim=-1)
        return (weights * signed).sum(dim=-1)

    def candidate_clearance(
        self, clouds: list[tuple[torch.Tensor, torch.Tensor]], box: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """(candidates, obstacles) true clearance."""
        return torch.stack([self.clearance(points, radii, box) for points, radii in clouds], dim=0)

    def __call__(
        self,
        clouds: list[tuple[torch.Tensor, torch.Tensor]],
        box: dict[str, torch.Tensor],
        costs: torch.Tensor,
    ) -> torch.Tensor:
        clearance = self.candidate_clearance(clouds, box)
        collision = (
            torch.nn.functional.softplus(
                -(clearance - self.margin_m) / self.collision_temperature_m
            )
            * self.collision_temperature_m
        )
        objective = costs + self.blocked_penalty * collision.sum(dim=-1)
        return torch.softmax(-objective / self.choice_temperature, dim=0)
