"""Per-frame metres to the binding constraint, on the executed trajectory.

Three functions in this package already compute this quantity and destroy it inside their own
return statement: `scene_route_check.check_route_meets_obstacle` reduces a whole distance series to
a closest approach, `capsule_box_clearance` reduces a whole clearance field to one number and one
frame, and `local_adaptation` reduces route progress to a station. Each reduction is right for the
gate it serves and wrong for a learner, and the difference is the whole of the plan's timing
problem: **a model trained on the scalars learns that a crouch is needed and never when.**

The project's own sentence is "scene variation is not scene-conditioned behaviour". The same
argument applies one level down. Pairing an episode with a video does not produce scene-conditioned
*timing* unless something in the record ties the pixels to how far away the obstacle is, and this is
that something.

Everything here is measured on the **executed** trajectory, not the reference. The reference says
where the robot was asked to be; the constraint binds where the robot actually was, and on the one
family where the two disagree by half a metre it is the executed path that struck the ceiling.

Nothing new is computed. Every series is the un-reduced form of a number the pipeline already
trusts, taken from the same functions, so a series and the gate it explains cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_route_check import capsule_box_clearance_series, check_route_meets_obstacle
from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

#: Frames per second of a SONIC rollout record. Carried explicitly rather than assumed, because the
#: reference clips are 30 fps and the physics record is not.
DEFAULT_FPS = 50.0


@dataclass(frozen=True)
class ConstraintDistance:
    """Where the obstacle is, per frame, from the robot's point of view."""

    fps: float
    #: (T,) cumulative executed arclength normalised to [0, 1]; nondecreasing.
    route_progress: np.ndarray
    #: Total executed planar path length, in metres.
    route_length_m: float
    #: (T,) signed arclength to the obstacle station: **positive before it, negative after**. This
    #: is the field a policy needs to learn onset from, and its sign is what makes "not yet" and
    #: "too late" different states rather than the same distance.
    remaining_to_station_m: np.ndarray
    #: (T,) planar metres from the root to the obstacle footprint, zero while inside it.
    footprint_distance_m: np.ndarray
    #: (T,) smallest gap from any collision capsule to the obstacle box; negative is penetration,
    #: measured *through the capsule radius* -- the distance from a point inside a box is zero, so
    #: a zero-radius capsule reports contact as 0.0 and never as a negative depth.
    body_clearance_m: np.ndarray
    #: (T,) index into the capsule array of whichever capsule binds that frame.
    binding_capsule: np.ndarray
    #: (T,) name of the link owning the binding capsule.
    binding_body: tuple[str, ...]
    #: Frame at which the robot is nearest the station along the route.
    station_frame: int
    #: Frame at which ``body_clearance_m`` is smallest — the moment the obstacle binds hardest.
    bottleneck_frame: int
    #: First frame whose clearance is negative, or None if the body never overlapped the obstacle.
    first_overlap_frame: int | None

    @property
    def frames(self) -> int:
        return len(self.route_progress)

    @property
    def seconds_to_station(self) -> np.ndarray:
        """Remaining arclength converted to time at each frame's own speed, in seconds.

        Undefined where the robot is not moving, and NaN there rather than infinite: a stopped robot
        has no time-to-contact, and a large finite number would be read as a safe one.
        """
        speed = np.gradient(self.route_progress) * self.route_length_m * self.fps
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(speed > 1e-6, self.remaining_to_station_m / speed, np.nan)
        return out

    def as_arrays(self) -> dict[str, np.ndarray]:
        """The series, ready for ``np.savez``. Scalars are included as 0-d arrays."""
        return {
            "fps": np.asarray(self.fps),
            "route_progress": self.route_progress,
            "route_length_m": np.asarray(self.route_length_m),
            "remaining_to_station_m": self.remaining_to_station_m,
            "footprint_distance_m": self.footprint_distance_m,
            "body_clearance_m": self.body_clearance_m,
            "binding_capsule": self.binding_capsule,
            "binding_body": np.asarray(self.binding_body),
            "station_frame": np.asarray(self.station_frame),
            "bottleneck_frame": np.asarray(self.bottleneck_frame),
            "first_overlap_frame": np.asarray(
                -1 if self.first_overlap_frame is None else self.first_overlap_frame
            ),
        }


def route_arclength(root_xy: np.ndarray) -> np.ndarray:
    """Cumulative planar path length in metres, one value per frame."""
    xy = np.asarray(root_xy, dtype=np.float64)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def constraint_distance(
    root_pos_w: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_names: list[str],
    box: tuple[float, float, float, float, float, float],
    *,
    fps: float = DEFAULT_FPS,
    capsules=G1_COLLISION_CAPSULES,
) -> ConstraintDistance:
    """Build the per-frame constraint record for one executed episode against one obstacle.

    ``box`` is the obstacle's world-axis-aligned ``(x0, y0, z0, x1, y1, z1)``, read back out of the
    scene file physics loaded rather than recomputed from the builder's variables — recomputing
    would only confirm the builder agrees with itself, which was true the whole time the frame-
    mismatch bug was live.
    """
    root = np.asarray(root_pos_w, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] < 2:
        raise ValueError(f"expected (T, >=2) executed root positions, got {root.shape}")
    frames = len(root)
    if frames < 2:
        raise ValueError("a constraint distance needs at least two frames of motion")

    x0, y0, z0, x1, y1, z1 = box
    footprint = (x0, y0, x1, y1)

    cumulative = route_arclength(root[:, :2])
    total = float(cumulative[-1])
    progress = cumulative / total if total > 1e-9 else np.linspace(0.0, 1.0, frames)

    check = check_route_meets_obstacle(root[:, :2], footprint)
    footprint_distance = (
        np.asarray(check.distance_m, dtype=np.float64)
        if check.distance_m is not None
        else np.zeros(frames)
    )

    # The station is where the robot passes the obstacle's centre, taken along the route rather
    # than in x: on a route that turns -- and 013 turns sharply left -- an x coordinate is reached
    # twice and an arclength is reached once.
    centre = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    station_frame = int(np.argmin(np.linalg.norm(root[:, :2] - centre, axis=1)))
    remaining = float(cumulative[station_frame]) - cumulative

    starts, ends, radii, owners = body_capsules_world(
        np.asarray(body_pos_w, dtype=np.float64),
        np.asarray(body_quat_w, dtype=np.float64),
        list(body_names),
        capsules=capsules,
    )
    clearance, binding = capsule_box_clearance_series(starts, ends, radii, box)
    overlapping = np.flatnonzero(clearance < 0.0)

    return ConstraintDistance(
        fps=float(fps),
        route_progress=progress,
        route_length_m=total,
        remaining_to_station_m=remaining,
        footprint_distance_m=footprint_distance,
        body_clearance_m=clearance,
        binding_capsule=binding,
        binding_body=tuple(owners[int(i)] for i in binding),
        station_frame=station_frame,
        bottleneck_frame=int(np.argmin(clearance)),
        first_overlap_frame=int(overlapping[0]) if len(overlapping) else None,
    )


def constraint_distance_from_payload(
    payload: dict,
    box: tuple[float, float, float, float, float, float],
    *,
    fps: float = DEFAULT_FPS,
) -> ConstraintDistance:
    """The same, read straight off a validated SONIC trajectory payload.

    Raises rather than filling in defaults when a field is absent: an episode that cannot say where
    its body was is not an episode with a zero constraint distance.
    """
    required = ("root_pos_w", "body_pos_w", "body_quat_w", "body_names")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"trajectory payload is missing {missing}; it cannot locate its own body")
    return constraint_distance(
        np.asarray(payload["root_pos_w"], dtype=np.float64),
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        box,
        fps=fps,
    )
