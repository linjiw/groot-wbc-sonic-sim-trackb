"""Build scene triplets where geometry, not the prompt, decides which motion works.

The corpus so far generates a room *around* a motion, which guarantees the motion fits. That
produces positives efficiently and cannot, on its own, show that a scene determines
behaviour: when the furniture never touches the robot, two layouts give two different images
and a **bit-identical** trajectory. Measured on this corpus, 156 evaluable episodes carry
only 131 distinct executed trajectories, so some of that duplication is already visual
variation masquerading as behavioural variation.

A counterfactual family fixes that by construction. One task, one start, one goal, three
rollouts:

===========================  ==========================================================
``(nominal, easy)``          the ordinary motion succeeds
``(nominal, hard)``          the *same* motion fails, because the geometry moved
``(adapted, hard)``          a different whole-body behaviour succeeds in that geometry
===========================  ==========================================================

The middle row is the one nothing else in the corpus provides: a matched negative in which
the only thing that changed is the scene. Together the triplet is the supervision that makes
"look at the room before choosing how to move" learnable rather than optional.

**Finding the boundary rather than guessing it.** An obstacle is placed along a
one-dimensional family of transforms -- a shelf lowered toward the head, a wall closed toward
the shoulder, a bar raised toward the swing foot -- and the transform is binary-searched
against the robot's *executed* full-body swept volume until the smallest interference is
located. Easy and hard scenes sit a stated margin either side of that boundary. This is why
the family is controlled: the two scenes differ by one scalar, not by a re-roll of the
layout sampler.

The swept volume is the executed one deliberately. For a matched negative, the question is
whether *this* trajectory collides, and the executed trajectory is what the robot did.
Because such scenes encode one policy's path, they are training and analysis material and
must never appear in an evaluation split -- ``eval_scene_gate`` enforces that separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

#: The geometry regimes a family can probe. Each names the body region that a scene in this
#: regime is designed to interfere with, which is also what the adapted motion must avoid.
GEOMETRY_REGIMES = ("overhead", "lateral", "floor", "compound")

#: How far either side of the collision boundary the easy and hard scenes are placed, in
#: metres. Wide enough that neither is marginal, narrow enough that the two scenes are
#: recognisably the same room.
DEFAULT_BOUNDARY_MARGIN_M = 0.08

#: Binary-search resolution. Below a centimetre the distinction stops being meaningful
#: against the capsule model's own fidelity.
DEFAULT_SEARCH_TOLERANCE_M = 0.005


class CounterfactualError(ValueError):
    """Raised when a family cannot be constructed from the inputs given."""


@dataclass(frozen=True)
class ObstacleSpec:
    """An axis-aligned box, positioned by a single scalar the search varies."""

    name: str
    size: tuple[float, float, float]
    #: Position at parameter zero.
    base_center: tuple[float, float, float]
    #: Direction the parameter moves the box along, in metres per unit.
    axis: tuple[float, float, float]
    regime: str

    def at(self, parameter: float) -> tuple[float, float, float]:
        return (
            self.base_center[0] + self.axis[0] * parameter,
            self.base_center[1] + self.axis[1] * parameter,
            self.base_center[2] + self.axis[2] * parameter,
        )

    def box_at(self, parameter: float) -> tuple[float, float, float, float, float, float]:
        cx, cy, cz = self.at(parameter)
        hx, hy, hz = (s / 2.0 for s in self.size)
        return (cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz)


@dataclass(frozen=True)
class BoundaryResult:
    """Where the obstacle first interferes with the swept volume."""

    parameter: float
    #: Signed clearance at the boundary; near zero by construction.
    clearance_m: float
    #: Frame at which the interference is deepest.
    frame: int
    iterations: int


@dataclass(frozen=True)
class CounterfactualFamily:
    """One task, one start, one goal, and the three scenes that make it a family."""

    family_id: str
    regime: str
    obstacle: ObstacleSpec
    boundary: BoundaryResult
    easy_parameter: float
    hard_parameter: float
    margin_m: float
    #: Clearance the nominal motion has in each scene. Easy is positive, hard is negative.
    easy_clearance_m: float
    hard_clearance_m: float

    @property
    def separated(self) -> bool:
        """Whether the two scenes actually straddle the boundary.

        A family whose 'hard' scene does not interfere is not a counterfactual; it is two
        easy scenes with different pixels, which is exactly the failure this module exists
        to avoid.
        """
        return self.easy_clearance_m > 0.0 > self.hard_clearance_m


def swept_clearance_profile(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    box: tuple[float, float, float, float, float, float],
    *,
    capsules=G1_COLLISION_CAPSULES,
) -> np.ndarray:
    """Per-frame clearance between the executed swept volume and an axis-aligned box.

    The minimum over this is what a boundary search needs, but the profile itself answers a
    different question: *when* does the obstacle first interfere, as opposed to when is the
    interference deepest. Those are not the same frame, and comparing a predicted deepest
    frame against an observed first contact makes a prediction look worse than it is.
    """
    starts, ends, radii, _ = body_capsules_world(
        body_pos, body_quat, body_names, capsules=capsules
    )
    min_x, min_y, min_z, max_x, max_y, max_z = box
    lower = np.array([min_x, min_y, min_z])
    upper = np.array([max_x, max_y, max_z])

    # Sample points along each capsule segment, including both ends.
    weights = np.linspace(0.0, 1.0, 5).reshape(1, 1, 5, 1)
    points = starts[:, :, None, :] * (1.0 - weights) + ends[:, :, None, :] * weights

    clamped = np.clip(points, lower, upper)
    distances = np.linalg.norm(points - clamped, axis=-1) - radii[None, :, None]
    return distances.min(axis=(1, 2))


def swept_clearance_to_box(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    box: tuple[float, float, float, float, float, float],
    *,
    capsules=G1_COLLISION_CAPSULES,
) -> tuple[float, int]:
    """Smallest clearance between the executed swept volume and an axis-aligned box.

    Negative means interference. Returns ``(clearance, frame)`` so a caller can point at the
    moment of contact rather than only report that one exists.

    Capsules are sampled along their axis rather than solved analytically: a capsule-box
    distance has no short closed form, and at these sizes a few samples per capsule is both
    accurate to well under the search tolerance and fast enough to sit inside a binary
    search over hundreds of frames.

    **The negative branch saturates.** A capsule wholly inside the box has point-to-box
    distance zero, so the clearance reads exactly ``-radius`` however deeply it is engulfed.
    That is harmless for a boundary search, which only needs the zero crossing, and it makes
    the value useless for ranking two motions at a *fixed* obstacle position: two bodies both
    engulfed return the same number. Compare boundaries instead -- see
    ``build_paired_family``, which exists because this was learned the hard way, with two
    different motions both reading -0.0680 m, the torso capsule's radius.
    """
    per_frame = swept_clearance_profile(
        body_pos, body_quat, body_names, box, capsules=capsules
    )
    frame = int(np.argmin(per_frame))
    return float(per_frame[frame]), frame


def find_collision_boundary(
    clearance_at: Callable[[float], tuple[float, int]],
    low: float,
    high: float,
    *,
    tolerance_m: float = DEFAULT_SEARCH_TOLERANCE_M,
    max_iterations: int = 40,
) -> BoundaryResult:
    """Binary-search the parameter at which clearance crosses zero.

    ``clearance_at(parameter)`` must be positive at ``low`` (obstacle clear of the robot)
    and negative at ``high`` (obstacle interfering). Both are checked, because a search that
    silently assumes a bracket returns a confident answer about nothing.

    Note the sweep is **not monotone**: a shelf lowered far enough passes below the torso
    and the clearance turns positive again, so a range spanning both crossings is a caller
    error. The endpoint checks catch that case rather than converging on whichever crossing
    the bisection happens to walk into.
    """
    low_clearance, _ = clearance_at(low)
    high_clearance, _ = clearance_at(high)
    if low_clearance <= 0.0:
        raise CounterfactualError(
            f"the obstacle already interferes at the clear end (clearance "
            f"{low_clearance:.4f} m at parameter {low}); nothing to bracket"
        )
    if high_clearance >= 0.0:
        raise CounterfactualError(
            f"the obstacle never interferes (clearance {high_clearance:.4f} m at parameter "
            f"{high}); widen the search range"
        )

    iterations = 0
    frame = 0
    clearance = low_clearance
    while high - low > tolerance_m and iterations < max_iterations:
        middle = 0.5 * (low + high)
        clearance, frame = clearance_at(middle)
        if clearance > 0.0:
            low = middle
        else:
            high = middle
        iterations += 1

    boundary = 0.5 * (low + high)
    clearance, frame = clearance_at(boundary)
    return BoundaryResult(
        parameter=boundary, clearance_m=clearance, frame=frame, iterations=iterations
    )


def build_family(
    family_id: str,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    obstacle: ObstacleSpec,
    *,
    search_low: float = 0.0,
    search_high: float = 1.0,
    margin_m: float = DEFAULT_BOUNDARY_MARGIN_M,
    tolerance_m: float = DEFAULT_SEARCH_TOLERANCE_M,
    capsules=G1_COLLISION_CAPSULES,
) -> CounterfactualFamily:
    """Locate the boundary for one obstacle and place the easy and hard scenes around it."""
    if obstacle.regime not in GEOMETRY_REGIMES:
        raise CounterfactualError(
            f"unknown geometry regime {obstacle.regime!r}; known: {list(GEOMETRY_REGIMES)}"
        )

    def clearance_at(parameter: float) -> tuple[float, int]:
        return swept_clearance_to_box(
            body_pos, body_quat, body_names, obstacle.box_at(parameter), capsules=capsules
        )

    boundary = find_collision_boundary(
        clearance_at, search_low, search_high, tolerance_m=tolerance_m
    )

    # The parameter increases toward interference, so easy sits below the boundary.
    easy_parameter = boundary.parameter - margin_m
    hard_parameter = boundary.parameter + margin_m
    easy_clearance, _ = clearance_at(easy_parameter)
    hard_clearance, _ = clearance_at(hard_parameter)

    family = CounterfactualFamily(
        family_id=family_id,
        regime=obstacle.regime,
        obstacle=obstacle,
        boundary=boundary,
        easy_parameter=easy_parameter,
        hard_parameter=hard_parameter,
        margin_m=margin_m,
        easy_clearance_m=easy_clearance,
        hard_clearance_m=hard_clearance,
    )
    if not family.separated:
        raise CounterfactualError(
            f"{family_id}: the two scenes do not straddle the boundary "
            f"(easy {easy_clearance:.4f} m, hard {hard_clearance:.4f} m). Without a matched "
            "negative this is two easy scenes with different pixels, not a counterfactual."
        )
    return family


@dataclass(frozen=True)
class PairedFamily:
    """A family built from two motions' own boundaries, rather than one clearance value."""

    family_id: str
    regime: str
    obstacle: ObstacleSpec
    nominal_boundary: BoundaryResult
    adapted_boundary: BoundaryResult
    easy_parameter: float
    hard_parameter: float

    @property
    def window_m(self) -> float:
        """How much obstacle travel separates the two motions. The family's whole content."""
        return self.adapted_boundary.parameter - self.nominal_boundary.parameter

    @property
    def separated(self) -> bool:
        return self.window_m > 0.0


def build_paired_family(
    family_id: str,
    nominal: tuple[np.ndarray, np.ndarray, Sequence[str]],
    adapted: tuple[np.ndarray, np.ndarray, Sequence[str]],
    obstacle: ObstacleSpec,
    *,
    search_low: float = 0.0,
    search_high: float = 1.5,
    tolerance_m: float = DEFAULT_SEARCH_TOLERANCE_M,
    capsules=G1_COLLISION_CAPSULES,
) -> PairedFamily:
    """Place the hard scene *between* two motions' collision boundaries.

    Each motion is searched for the obstacle position at which it first interferes. The hard
    scene then sits strictly between them, so the nominal motion collides and the adapted one
    does not -- which is the counterfactual, stated as a property of the construction rather
    than hoped for after the fact.

    Comparing clearances at a single obstacle position cannot do this, because the clearance
    saturates at ``-radius`` once a capsule is engulfed and two colliding motions become
    indistinguishable.
    """
    if obstacle.regime not in GEOMETRY_REGIMES:
        raise CounterfactualError(
            f"unknown geometry regime {obstacle.regime!r}; known: {list(GEOMETRY_REGIMES)}"
        )

    def boundary_for(bodies) -> BoundaryResult:
        pos, quat, names = bodies
        return find_collision_boundary(
            lambda parameter: swept_clearance_to_box(
                pos, quat, names, obstacle.box_at(parameter), capsules=capsules
            ),
            search_low, search_high, tolerance_m=tolerance_m,
        )

    nominal_boundary = boundary_for(nominal)
    adapted_boundary = boundary_for(adapted)
    window = adapted_boundary.parameter - nominal_boundary.parameter
    if window <= 0.0:
        raise CounterfactualError(
            f"{family_id}: the adapted motion interferes no later than the nominal one "
            f"(boundaries {adapted_boundary.parameter:.3f} m and "
            f"{nominal_boundary.parameter:.3f} m). This pair does not separate: the adapted "
            "motion is not actually clearing anything the nominal one does not."
        )

    return PairedFamily(
        family_id=family_id,
        regime=obstacle.regime,
        obstacle=obstacle,
        nominal_boundary=nominal_boundary,
        adapted_boundary=adapted_boundary,
        # Easy sits clear of both; hard sits between them.
        easy_parameter=nominal_boundary.parameter - 0.5 * window,
        hard_parameter=0.5 * (nominal_boundary.parameter + adapted_boundary.parameter),
    )
