"""Build rooms that can grade a policy, rather than rooms fitted to one trajectory.

The training scenes are generated around an executed path, with clearance measured against
that motion's own swept volume. That is right for training and disqualifying for
evaluation: the room encodes the tracking error of the policy that produced it, so it
measures whether a policy reproduces that trajectory rather than whether it can navigate.
``eval_scene_gate`` refuses such scenes; this is what to build instead.

Three differences from the training generator, each deliberate:

* **Clearance is budgeted from the reference, not the execution.** The room clears the
  *commanded* path by the robot's own width plus an allowance for how far any reasonable
  policy may drift from it. Nothing about the executed trajectory enters the geometry.
* **The allowance is a measurement.** Over a 40-episode diverse batch, executed paths
  drifted 0.053-0.357 m from their reference (median 0.113). The default covers the
  observed maximum with room to spare, because a policy worse than the one measured is
  exactly what an evaluation is supposed to admit.
* **The room must admit more than one route.** A scene traversable only along the reference
  is policy-specific whatever its clearance says, so routability is verified rather than
  assumed, and a scene that fails is rejected rather than shipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clutter_scene_builder import build_clutter_scene
from .scene_route_sampler import Obstacle, RouteSamplingError, sample_routes

#: Widest measured swept half-width across 48 episodes carrying per-body pose. This is the
#: robot, not the pelvis: the 0.45 m figure used previously was exceeded on 98% of them.
MEASURED_BODY_HALF_WIDTH_M = 0.664

#: Allowance for how far a policy's executed path may sit from the commanded one. Measured
#: max over 40 diverse episodes was 0.357 m; rounded up, because an evaluation must admit
#: policies worse than the one that produced the measurement.
POLICY_DRIFT_ALLOWANCE_M = 0.45

#: Extra clearance an evaluation room carries over a training room, so that a policy which
#: deviates more than any observed episode still has somewhere to be.
EVAL_CLEARANCE_M = MEASURED_BODY_HALF_WIDTH_M + POLICY_DRIFT_ALLOWANCE_M

#: An evaluation scene must admit at least this many routes independent of the reference.
MIN_INDEPENDENT_ROUTES = 2


class EvalSceneError(ValueError):
    """Raised when a scene cannot be certified for evaluation use."""


@dataclass(frozen=True)
class EvalSceneReport:
    """A built evaluation scene and the evidence that it may grade a policy."""

    scene_id: str
    clearance_m: float
    body_half_width_m: float
    drift_allowance_m: float
    independent_routes: int
    min_route_clearance_m: float
    placed_pieces: int
    occupancy: float
    #: The bar this scene was held to. Stored rather than read from the module constant so
    #: a caller that raises the requirement is actually held to the number it asked for.
    required_routes: int = MIN_INDEPENDENT_ROUTES

    @property
    def certified(self) -> bool:
        return self.independent_routes >= self.required_routes


def build_eval_scene(
    reference_path_xy: np.ndarray,
    *,
    scene_id: str,
    seed: int = 0,
    target_pieces: int = 16,
    body_half_width_m: float = MEASURED_BODY_HALF_WIDTH_M,
    drift_allowance_m: float = POLICY_DRIFT_ALLOWANCE_M,
    max_distance_from_path_m: float = 3.0,
    min_independent_routes: int = MIN_INDEPENDENT_ROUTES,
):
    """Build a scene around a *reference* path and certify it admits other routes.

    Returns ``(spec, report)``. Raises when the room cannot be certified, rather than
    returning an uncertified scene that a caller might ship by accident.
    """
    path = np.asarray(reference_path_xy, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < 2:
        raise EvalSceneError("reference path needs at least two points")

    clearance = float(body_half_width_m + drift_allowance_m)
    try:
        spec = build_clutter_scene(
            path,
            scene_id=scene_id,
            seed=seed,
            clearance_m=clearance,
            target_pieces=target_pieces,
            max_distance_from_path_m=max_distance_from_path_m,
        )
    except (ValueError, ZeroDivisionError) as error:
        # A clearance wider than the band furniture may occupy leaves the sampler with an
        # empty interval, which surfaces from numpy as "high - low < 0". That is a
        # statement about the requested geometry, so it belongs in this module's error
        # type rather than escaping as an arithmetic complaint from three layers down.
        raise EvalSceneError(
            f"{scene_id}: no room geometry satisfies a {clearance:.3f} m clearance with "
            f"furniture within {max_distance_from_path_m:.2f} m of the path ({error})"
        ) from error

    obstacles = [Obstacle(rect=piece.rect, z_base=piece.z_base) for piece in spec.pieces]
    try:
        routes = sample_routes(
            obstacles,
            spec.room_size_xy,
            count=min_independent_routes + 1,
            seed=seed,
            body_half_width_m=body_half_width_m,
            margin_m=drift_allowance_m,
        )
    except RouteSamplingError as error:
        raise EvalSceneError(
            f"{scene_id} admits no independent route at {clearance:.3f} m clearance, so it "
            f"is traversable only along the reference it was built around: {error}"
        ) from error

    report = EvalSceneReport(
        scene_id=scene_id,
        clearance_m=clearance,
        body_half_width_m=float(body_half_width_m),
        drift_allowance_m=float(drift_allowance_m),
        independent_routes=len(routes),
        min_route_clearance_m=float(min(r.min_clearance_m for r in routes)),
        placed_pieces=int(spec.metrics.get("placed_pieces", len(spec.pieces))),
        occupancy=float(spec.metrics.get("clutter_occupancy", 0.0)),
        required_routes=int(min_independent_routes),
    )
    if not report.certified:
        raise EvalSceneError(
            f"{scene_id} admits only {report.independent_routes} independent route(s); "
            f"at least {min_independent_routes} are required for an evaluation scene"
        )
    return spec, report
