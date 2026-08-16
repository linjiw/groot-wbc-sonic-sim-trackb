"""Separate robot self-contact from robot-environment contact.

Why this exists
---------------

The Isaac contact sensor records ``net_forces_w`` per body: the *sum* of every
contact acting on that body. A single number therefore cannot distinguish

  * the robot's own wrist swinging into its own hip (a gait artifact), from
  * the robot's hip striking a wall (a genuine navigation failure),

yet the locomotion acceptance gate has to treat those two very differently. Run
on the M0 fixtures, the undecomposed signal was dominated entirely by the first
case: a 1.0 N non-foot threshold rejected every episode while the robot had in
fact touched no scene geometry at all.

How the decomposition works
---------------------------

Newton's third law. When two *tracked* bodies of the same articulation contact
each other, the sensor books equal and opposite net forces on both. When a
tracked body contacts something untracked -- the floor, a wall, a rack -- there
is no counterpart column, because the scene is not a sensor body.

So for each frame we greedily match active bodies into ``f_a ~= -f_b`` pairs.
Matched force is self-contact; whatever remains unmatched is external contact.
This was verified on three M0 rollouts (household, factory, and a bare-plane
control): every non-foot contact matched a partner to within 1e-4 N, and the
unmatched non-foot residual was exactly 0.

Limits, stated plainly
----------------------

* A body touching the scene *and* itself in the same frame reports one summed
  vector that will not pair exactly. The residual is then attributed to external
  contact, which is the conservative direction: it can over-report an external
  contact, never hide one.
* Two bodies independently striking the scene with coincidentally opposite
  forces would be misread as self-contact. Bit-level opposition by coincidence
  is not a realistic failure mode, but the tolerance is exposed so a caller can
  tighten it.
* Feet are excluded from the external-contact total by default because
  foot-to-floor contact is the expected load path; the separate
  ``foot_non_ground_contact`` gate covers feet touching the wrong thing.

Lateral collision vs support load
---------------------------------

"External" still lumps two very different things together. A crouching reference
(``02_multi_text_ee_constraint``, pelvis down to 0.32 m) puts the robot's knee on
the floor: measured force was ``[0, 0, 104.9]`` and ``[0, 0, 549.2]`` N -- purely
vertical and upward on every one of 56 and 58 contact frames, at a root height of
0.41 m. That is kneeling, not a rack strike, and a rack strike would be mostly
horizontal.

So the external force is further split by direction: the horizontal component is
the collision signal, and the upward vertical component is support load. The
limitation to keep in mind is that standing *on top of* a box also produces an
upward force and will read as support, not collision; in the M0 scenes the
obstacles are racks, walls and a pallet, and no accepted episode climbs them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ContactDecomposition",
    "decompose_contact_forces",
]


@dataclass(frozen=True)
class ContactDecomposition:
    """Per-frame split of contact force into self-contact and external contact."""

    #: (T,) max paired self-contact magnitude per frame.
    self_contact_by_frame: np.ndarray
    #: (T,) max unpaired non-foot magnitude per frame -- robot-to-environment.
    external_contact_by_frame: np.ndarray
    #: (T,) max unpaired non-foot *horizontal* magnitude -- lateral collision.
    external_lateral_by_frame: np.ndarray
    #: (T,) max unpaired non-foot *upward* force -- support load on a non-foot body.
    external_support_by_frame: np.ndarray
    #: Body-name pairs observed in self-contact, ordered and de-duplicated.
    self_contact_pairs: tuple[tuple[str, str], ...]
    #: Body names carrying unpaired non-foot force above the pairing tolerance.
    external_contact_bodies: tuple[str, ...]
    #: Number of (frame, pair) self-contact events.
    self_contact_events: int

    @property
    def max_self_contact(self) -> float:
        return float(self.self_contact_by_frame.max()) if self.self_contact_by_frame.size else 0.0

    @property
    def max_external_contact(self) -> float:
        return (
            float(self.external_contact_by_frame.max())
            if self.external_contact_by_frame.size
            else 0.0
        )

    @property
    def max_self_contact_frame(self) -> int:
        if not self.self_contact_by_frame.size:
            return 0
        return int(np.argmax(self.self_contact_by_frame))

    @property
    def max_lateral_contact(self) -> float:
        return (
            float(self.external_lateral_by_frame.max())
            if self.external_lateral_by_frame.size
            else 0.0
        )

    @property
    def max_support_contact(self) -> float:
        return (
            float(self.external_support_by_frame.max())
            if self.external_support_by_frame.size
            else 0.0
        )

    @property
    def max_lateral_contact_frame(self) -> int:
        if not self.external_lateral_by_frame.size:
            return 0
        return int(np.argmax(self.external_lateral_by_frame))

    @property
    def max_external_contact_frame(self) -> int:
        if not self.external_contact_by_frame.size:
            return 0
        return int(np.argmax(self.external_contact_by_frame))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_self_contact_force_n": self.max_self_contact,
            "max_self_contact_frame": self.max_self_contact_frame,
            "max_external_contact_force_n": self.max_external_contact,
            "max_external_contact_frame": self.max_external_contact_frame,
            "max_lateral_contact_force_n": self.max_lateral_contact,
            "max_lateral_contact_frame": self.max_lateral_contact_frame,
            "max_support_contact_force_n": self.max_support_contact,
            "self_contact_pairs": [list(pair) for pair in self.self_contact_pairs],
            "external_contact_bodies": list(self.external_contact_bodies),
            "self_contact_events": self.self_contact_events,
        }


def decompose_contact_forces(
    contact_force_vectors: np.ndarray,
    contact_body_names: Sequence[str],
    *,
    foot_body_names: Sequence[str] = (),
    pair_atol: float = 1e-4,
    active_atol: float = 1e-6,
) -> ContactDecomposition:
    """Split per-body net contact forces into self-contact and external contact.

    Args:
        contact_force_vectors: ``(T, B, 3)`` net contact force per body, world frame.
        contact_body_names: ``B`` body names aligned with axis 1.
        foot_body_names: Bodies whose environment contact is expected and therefore
            excluded from the external-contact total.
        pair_atol: Absolute tolerance, in newtons, for calling two force vectors
            an action-reaction pair.
        active_atol: Magnitude below which a body is treated as not in contact.

    Returns:
        A :class:`ContactDecomposition`.
    """
    forces = np.asarray(contact_force_vectors, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise ValueError(f"contact_force_vectors must be (T, B, 3); got {forces.shape}")
    names = tuple(contact_body_names)
    if len(names) != forces.shape[1]:
        raise ValueError(
            f"contact_body_names has {len(names)} entries but forces have {forces.shape[1]} bodies"
        )
    feet = set(foot_body_names)

    frame_count = forces.shape[0]
    self_by_frame = np.zeros(frame_count, dtype=np.float64)
    external_by_frame = np.zeros(frame_count, dtype=np.float64)
    lateral_by_frame = np.zeros(frame_count, dtype=np.float64)
    support_by_frame = np.zeros(frame_count, dtype=np.float64)
    pairs: set[tuple[str, str]] = set()
    external_bodies: set[str] = set()
    events = 0

    magnitudes = np.linalg.norm(forces, axis=-1)
    for frame in range(frame_count):
        frame_forces = forces[frame]
        active = np.flatnonzero(magnitudes[frame] > active_atol)
        matched: set[int] = set()
        for first in active:
            if first in matched:
                continue
            for second in active:
                if second <= first or second in matched:
                    continue
                if np.allclose(frame_forces[first], -frame_forces[second], atol=pair_atol, rtol=0.0):
                    matched.add(int(first))
                    matched.add(int(second))
                    pairs.add(tuple(sorted((names[first], names[second]))))  # type: ignore[arg-type]
                    events += 1
                    self_by_frame[frame] = max(
                        self_by_frame[frame], float(magnitudes[frame, first])
                    )
                    break

        for body in active:
            if body in matched or names[body] in feet:
                continue
            magnitude = float(magnitudes[frame, body])
            vector = frame_forces[body]
            lateral = float(math.hypot(vector[0], vector[1]))
            external_by_frame[frame] = max(external_by_frame[frame], magnitude)
            lateral_by_frame[frame] = max(lateral_by_frame[frame], lateral)
            support_by_frame[frame] = max(support_by_frame[frame], float(max(vector[2], 0.0)))
            # Only a lateral push marks a body as having struck the scene; a purely
            # upward reaction is the floor holding a knee or hand up.
            if lateral > pair_atol:
                external_bodies.add(names[body])

    return ContactDecomposition(
        self_contact_by_frame=self_by_frame,
        external_contact_by_frame=external_by_frame,
        external_lateral_by_frame=lateral_by_frame,
        external_support_by_frame=support_by_frame,
        self_contact_pairs=tuple(sorted(pairs)),
        external_contact_bodies=tuple(sorted(external_bodies)),
        self_contact_events=events,
    )


def decompose_payload_contacts(
    payload: Mapping[str, Any],
    *,
    pair_atol: float = 1e-4,
) -> ContactDecomposition:
    """Convenience wrapper over a recorder payload."""
    return decompose_contact_forces(
        np.asarray(payload["robot_contact_force_w"], dtype=np.float64),
        tuple(payload["contact_body_names"]),
        foot_body_names=tuple(payload.get("allowed_foot_contact_body_names", ())),
        pair_atol=pair_atol,
    )
