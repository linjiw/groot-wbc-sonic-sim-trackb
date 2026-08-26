"""q(S | executed pair): a proposal distribution over critical scenes, and its scorecard.

The inverse problem "which scene explains this motion" is non-identifiable from one trajectory:
a crouch is explained by a beam, a lintel, a branch, or by nothing at all. Conditioning on a
*counterfactual pair* -- the executed nominal and the executed adaptation -- makes the part that
matters identifiable in closed form, and leaves a genuinely free remainder to be sampled.

This module separates those two things explicitly:

* **Support** (`CriticalAtom`) is deterministic. Given two executed reaches it yields the exact
  interval of face coordinates for which the nominal strikes and the adaptation clears.
* **The proposal** (`SceneDistribution`) samples only inside that support: where in the window to
  place the face, which archetype realises it, and how much context to dress it with.

Nothing here predicts a physics verdict, and nothing here estimates the natural frequency of
obstacles in the world -- that distribution is not observed by this corpus and cannot be
identified from it.

The scorecard implements the membership test for

    S_{eps,delta}(tau) = { S : Feasible = 1, Regret <= eps, Necessity >= delta, Realism >= r0 }

and its central observation is that on this construction Regret and Necessity are **not
independent**: their sum is fixed by the executed window, so the only way to improve both is a
deeper *delivered* adaptation (see `CriticalAtom.budget_m`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

#: Published-dimension priors used to score realism, in metres. Each entry is the plausible range
#: of the archetype's *non-binding* dimensions -- the binding coordinate is set by the trajectory,
#: so scoring it would only measure the solver.
ARCHETYPE_DIMENSION_PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "door_lintel": {"thickness_m": (0.08, 0.30), "across_m": (0.80, 1.60)},
    "shelf_plank": {"thickness_m": (0.02, 0.08), "across_m": (0.30, 1.20)},
    "ibeam": {"thickness_m": (0.10, 0.40), "across_m": (0.10, 0.40)},
    "hanging_panel": {"thickness_m": (0.20, 0.60), "across_m": (0.40, 2.00)},
    "hvac_duct": {"thickness_m": (0.20, 0.60), "across_m": (0.20, 0.80)},
}


class SupportError(ValueError):
    """The executed pair does not admit a critical face on this axis."""


@dataclass(frozen=True)
class CriticalAtom:
    """The identifiable part: an exact interval of face coordinates, from one executed pair.

    ``reach_nominal_m`` and ``reach_adapted_m`` are executed order statistics at one finite face,
    never reference-side estimates. ``delta_clear_m`` and ``delta_strike_m`` are the engineering
    margins applied to the adapted and nominal sides respectively.
    """

    source_id: str
    station_xy_m: tuple[float, float]
    route_axis: str
    route_progress: float
    face_along_route_m: float
    face_across_route_m: float
    reach_nominal_m: float
    reach_adapted_m: float
    delta_clear_m: float
    delta_strike_m: float
    binding_keypoint: str = "head_torso"

    def __post_init__(self) -> None:
        if self.face_along_route_m <= 0 or self.face_across_route_m <= 0:
            raise ValueError("face extents must be positive")
        if min(self.delta_clear_m, self.delta_strike_m) < 0:
            raise ValueError("engineering margins must be non-negative")
        if not math.isfinite(self.reach_nominal_m) or not math.isfinite(self.reach_adapted_m):
            raise ValueError("executed reaches must be finite")

    @property
    def lower_m(self) -> float:
        """Lowest face the adaptation still clears."""
        return self.reach_adapted_m + self.delta_clear_m

    @property
    def upper_m(self) -> float:
        """Highest face the nominal still strikes."""
        return self.reach_nominal_m - self.delta_strike_m

    @property
    def width_m(self) -> float:
        return self.upper_m - self.lower_m

    @property
    def raw_width_m(self) -> float:
        """Separation before engineering margins -- the physics the operator actually bought."""
        return self.reach_nominal_m - self.reach_adapted_m

    def axis_type_or_default(self) -> str:
        """The constraint axis this atom's window is expressed on."""
        return "overhead"

    def coordinate_at(self, xi: float) -> float:
        """Face coordinate at normalized window position ``xi`` in [0, 1]."""
        if not 0.0 <= xi <= 1.0:
            raise ValueError(f"xi must lie in [0, 1]; got {xi}")
        if self.width_m <= 0:
            raise SupportError(
                f"{self.source_id}: window is {1000 * self.width_m:.3f} mm after margins"
            )
        return self.lower_m + xi * self.width_m

    def regret_m(self, xi: float) -> float:
        """How much deeper the adaptation went than this scene required.

        Closed form, and exactly the adapted motion's clearance under the face. With the
        lexicographic cost J(tau; S) = infinity if infeasible else D(tau, tau_0), and D monotone in
        edit depth, the cheapest feasible edit is the one that just clears the face; anything
        deeper is regret. Placing the face at ``xi`` therefore *chooses* the regret budget.
        """
        return xi * self.width_m

    def necessity_m(self, xi: float) -> float:
        """How far the nominal penetrates the face -- the margin by which the edit is required."""
        return self.delta_strike_m + (1.0 - xi) * self.width_m

    def budget_m(self) -> float:
        """Necessity + regret, which is invariant in ``xi``.

        The structural fact about this construction. Since ``regret = xi * W`` and
        ``necessity = delta_strike + (1 - xi) * W``, their sum is fixed by the executed window, and
        **neither can be improved except by widening that window** -- that is, by an adaptation
        that actually separates further, not by placing the face more cleverly.

        Note what this does *not* say. Both terms are best at ``xi -> 0``: the tightest face is
        simultaneously the most necessary and the least regretful, so there is no
        necessity/regret dilemma to resolve. What ``xi`` really trades is explanatory tightness
        against the adapted motion's **execution margin** -- at ``xi = 0`` the adaptation clears
        by exactly ``delta_clear`` and nothing more, so any execution variability turns a clear
        into a strike. The corpus convention of placing the hard face at the window centre buys
        that robustness and pays half the window in regret for it.
        """
        return self.delta_strike_m + self.width_m


@dataclass(frozen=True)
class ObstacleItem:
    """One entry in an obstacle inventory, described in the route frame of the face it can realise.

    An inventory is a list of real objects of different sizes and shapes. Only some of them can
    realise a given critical point, and which ones is decidable without physics. All extents are
    metres in the route frame: ``along`` follows the executed tangent, ``across`` is lateral,
    ``vertical`` is world up.
    """

    item_id: str
    axis_type: str
    #: Extent of the planar face that can act as the binding surface.
    face_along_m: float
    face_across_m: float
    #: Thickness of the item measured *away* from the binding face, into free space.
    thickness_m: float
    #: How far any non-binding part of the item protrudes back past the binding face, towards the
    #: body. A plain plank protrudes nothing; a lintel's jambs protrude to the floor.
    protrusion_m: float = 0.0
    #: Lateral half-gap between the item's protruding parts, if it has any. A door lintel's jambs
    #: stand this far either side of the route centreline; ``inf`` means nothing protrudes.
    protrusion_half_gap_m: float = math.inf

    def __post_init__(self) -> None:
        if min(self.face_along_m, self.face_across_m, self.thickness_m) <= 0:
            raise ValueError("face extents and thickness must be positive")
        if self.protrusion_m < 0:
            raise ValueError("protrusion must be non-negative")


@dataclass(frozen=True)
class Admissibility:
    """Why an inventory item can or cannot realise a critical point."""

    item_id: str
    admissible: bool
    reasons: tuple[str, ...]


def item_admissibility(
    item: ObstacleItem,
    atom: CriticalAtom,
    *,
    body_half_width_m: float,
    keepout_m: float = 0.050,
    thickness_prior_m: tuple[float, float] = (0.02, 0.60),
) -> Admissibility:
    """Decide whether ``item`` can realise ``atom``'s binding face, without spending physics.

    Four conditions, in the order they matter:

    1. the item offers a face of the right kind for this constraint axis;
    2. that face spans the crossing footprint -- otherwise the body passes beside it and the
       window simply does not apply;
    3. every non-binding part clears both swept volumes, so the item cannot become a second,
       unlabelled cause;
    4. the item's own dimensions are plausible.

    Conditions 1-3 are what LfLH spends reconstruction and collision losses on; here they are
    decidable in closed form because the support already is.
    """
    reasons: list[str] = []
    if item.axis_type != atom.axis_type_or_default():
        reasons.append("wrong_axis")
    if item.face_along_m + 1e-12 < atom.face_along_route_m:
        reasons.append("face_too_short_along_route")
    if item.face_across_m + 1e-12 < 2.0 * (body_half_width_m + keepout_m):
        reasons.append("face_too_narrow_across_route")
    # A protruding part reaches back past the binding face towards the body. It is harmless only
    # if it stands outside the corridor the body sweeps, with the keep-out margin.
    if item.protrusion_m > 0 and item.protrusion_half_gap_m < body_half_width_m + keepout_m:
        reasons.append("protrusion_enters_swept_corridor")
    if not thickness_prior_m[0] <= item.thickness_m <= thickness_prior_m[1]:
        reasons.append("implausible_thickness")
    return Admissibility(item.item_id, not reasons, tuple(reasons))


def admissible_items(
    inventory: Iterable[ObstacleItem],
    atom: CriticalAtom,
    *,
    body_half_width_m: float,
    keepout_m: float = 0.050,
) -> list[Admissibility]:
    """Filter an inventory against one critical atom, keeping the refusals for reporting."""
    return [
        item_admissibility(item, atom, body_half_width_m=body_half_width_m, keepout_m=keepout_m)
        for item in inventory
    ]


@dataclass(frozen=True)
class AtomScore:
    """The eps-delta membership test for one proposed scene, computed from geometry alone."""

    source_id: str
    archetype: str
    xi: float
    coordinate_m: float
    in_support: bool
    regret_mm: float
    necessity_mm: float
    budget_mm: float
    realism: float
    feasible_geometric: bool

    def admits(self, *, epsilon_mm: float, delta_mm: float, realism_floor: float) -> bool:
        return bool(
            self.in_support
            and self.feasible_geometric
            and self.regret_mm <= epsilon_mm
            and self.necessity_mm >= delta_mm
            and self.realism >= realism_floor
        )


def realism_score(archetype: str, thickness_m: float, across_m: float) -> float:
    """Fraction of an archetype's non-binding dimensions inside its published prior.

    Deliberately crude and deliberately not a learned critic: it answers "would a person call this
    a door lintel" with a dimension check, which is the part of realism this corpus can defend.
    """
    prior = ARCHETYPE_DIMENSION_PRIORS.get(archetype)
    if prior is None:
        return 0.0
    measured = {"thickness_m": thickness_m, "across_m": across_m}
    inside = [1.0 if low <= measured[name] <= high else 0.0 for name, (low, high) in prior.items()]
    return float(np.mean(inside))


def score_atom(
    atom: CriticalAtom,
    *,
    archetype: str,
    xi: float,
    thickness_m: float,
    across_m: float,
) -> AtomScore:
    """Score one proposal without spending physics.

    ``feasible_geometric`` is a *necessary* condition read off the swept volumes, never a verdict:
    it says the adaptation clears this face by the engineering margin. Physics still decides
    whether the controller delivers that clearance in the authored scene.
    """
    in_support = atom.width_m > 0 and 0.0 <= xi <= 1.0
    coordinate = atom.coordinate_at(xi) if in_support else float("nan")
    return AtomScore(
        source_id=atom.source_id,
        archetype=archetype,
        xi=xi,
        coordinate_m=coordinate,
        in_support=in_support,
        regret_mm=1000 * atom.regret_m(xi) if in_support else float("nan"),
        necessity_mm=1000 * atom.necessity_m(xi) if in_support else float("nan"),
        budget_mm=1000 * atom.budget_m(),
        realism=realism_score(archetype, thickness_m, across_m),
        feasible_geometric=bool(
            in_support and coordinate - atom.delta_clear_m >= atom.reach_adapted_m - 1e-12
        ),
    )


@dataclass(frozen=True)
class SceneSample:
    source_id: str
    archetype: str
    xi: float
    coordinate_m: float
    station_xy_m: tuple[float, float]
    face_along_route_m: float
    face_across_route_m: float
    thickness_m: float
    context_count: int


class SceneDistribution:
    """Source-balanced proposal over critical scenes, conditioned on executed pairs.

    Archetype choice is **not** conditioned on trajectory features. A 2026-08-26 audit found that
    such conditioning scored worse than feature-blind counting on the four-source corpus, because
    the executed pair constrains *where* a face must be and not *what it looks like*. The
    archetype component is therefore an explicit source-balanced categorical with an exploration
    floor, and it is labelled as a design choice rather than a fit.

    ``regret_budget_mm`` is the distribution's one substantive knob. Sampling ``xi`` low produces
    tight scenes that barely admit the observed adaptation -- scenes that *explain* the motion --
    at the cost of the nominal's strike margin. See `CriticalAtom.budget_m`.
    """

    def __init__(
        self,
        atoms: list[CriticalAtom],
        archetype_counts: dict[str, int],
        *,
        exploration: float = 0.10,
        regret_budget_mm: float | None = None,
        context_range: tuple[int, int] = (0, 4),
    ) -> None:
        if not atoms:
            raise ValueError("at least one critical atom is required")
        if not 0.0 <= exploration < 1.0:
            raise ValueError("exploration must lie in [0, 1)")
        usable = [atom for atom in atoms if atom.width_m > 0]
        if not usable:
            raise SupportError("no atom retains a positive window after engineering margins")
        self.atoms = usable
        self.refused = len(atoms) - len(usable)
        self.exploration = exploration
        self.regret_budget_mm = regret_budget_mm
        self.context_range = context_range
        vocabulary = sorted(set(ARCHETYPE_DIMENSION_PRIORS) | set(archetype_counts))
        total = sum(archetype_counts.values())
        self.archetype_names = vocabulary
        if total <= 0:
            learned = np.full(len(vocabulary), 1.0 / len(vocabulary))
        else:
            learned = np.asarray(
                [archetype_counts.get(name, 0) / total for name in vocabulary], dtype=np.float64
            )
        self.archetype_probabilities = (1.0 - exploration) * learned + exploration / len(vocabulary)

    def source_weights(self) -> dict[str, float]:
        """Equal mass per independent source, so a dense window cannot dominate the corpus."""
        sources = sorted({atom.source_id for atom in self.atoms})
        return {source: 1.0 / len(sources) for source in sources}

    def _xi_ceiling(self, atom: CriticalAtom) -> float:
        if self.regret_budget_mm is None:
            return 1.0
        if atom.width_m <= 0:
            return 0.0
        return float(min(1.0, self.regret_budget_mm / (1000 * atom.width_m)))

    def sample(self, count: int, *, seed: int) -> list[SceneSample]:
        rng = np.random.default_rng(seed)
        weights = self.source_weights()
        by_source: dict[str, list[CriticalAtom]] = {}
        for atom in self.atoms:
            by_source.setdefault(atom.source_id, []).append(atom)
        sources = sorted(by_source)
        source_p = np.asarray([weights[source] for source in sources])
        source_p = source_p / source_p.sum()

        samples: list[SceneSample] = []
        for _ in range(count):
            source = sources[int(rng.choice(len(sources), p=source_p))]
            atom = by_source[source][int(rng.integers(len(by_source[source])))]
            ceiling = self._xi_ceiling(atom)
            if ceiling <= 0.0:
                continue
            xi = float(rng.uniform(0.0, ceiling))
            archetype = self.archetype_names[
                int(rng.choice(len(self.archetype_names), p=self.archetype_probabilities))
            ]
            prior = ARCHETYPE_DIMENSION_PRIORS[archetype]
            thickness = float(rng.uniform(*prior["thickness_m"]))
            across = float(rng.uniform(*prior["across_m"]))
            samples.append(
                SceneSample(
                    source_id=source,
                    archetype=archetype,
                    xi=xi,
                    coordinate_m=atom.coordinate_at(xi),
                    station_xy_m=atom.station_xy_m,
                    face_along_route_m=atom.face_along_route_m,
                    face_across_route_m=across,
                    thickness_m=thickness,
                    context_count=int(
                        rng.integers(self.context_range[0], self.context_range[1] + 1)
                    ),
                )
            )
        return samples
