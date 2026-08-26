"""Trajectory-conditioned distributions over LFH critical-scene proposals.

Exact executed geometry defines the support.  This module learns only how to distribute proposal
mass *within* that support from physics-verified source/archetype atoms.  It never predicts a
physics verdict and never turns an unverified archetype into accepted evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

from .archetypes import ARCHETYPES
from .critical_support import CriticalSupport


class CriticalDistributionError(ValueError):
    """The requested conditional proposal distribution is not evidence-supported."""


@dataclass(frozen=True)
class VerifiedSceneAtom:
    """One trajectory support atom with its physics-verified visual realizations."""

    support: CriticalSupport
    verified_archetypes: tuple[str, ...]
    hard_quantile: float

    def __post_init__(self) -> None:
        if not self.verified_archetypes:
            raise CriticalDistributionError("a verified atom needs at least one archetype")
        if len(set(self.verified_archetypes)) != len(self.verified_archetypes):
            raise CriticalDistributionError("verified archetypes must be unique")
        if not math.isfinite(self.hard_quantile) or not 0.0 <= self.hard_quantile <= 1.0:
            raise CriticalDistributionError("hard_quantile must lie in [0, 1]")
        compatible = {
            archetype_id
            for archetype_id, archetype in ARCHETYPES.items()
            if archetype.axis_type == self.support.axis_type
        }
        unknown = set(self.verified_archetypes) - compatible
        if unknown:
            raise CriticalDistributionError(
                f"archetypes {sorted(unknown)} are incompatible with {self.support.axis_type}"
            )

    @property
    def source_pair_id(self) -> str:
        return self.support.source_pair_id


@dataclass(frozen=True)
class CriticalSceneProposal:
    """One sampled scene proposal, still requiring keep-out and physics validation."""

    archetype_id: str
    hard_quantile: float
    hard_coordinate_m: float
    geometry_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "archetype_id": self.archetype_id,
            "hard_quantile": self.hard_quantile,
            "hard_coordinate_m": self.hard_coordinate_m,
            "geometry_seed": self.geometry_seed,
            "visual_seed": self.geometry_seed,  # deprecated alias
            "evidence_status": "proposal_only_requires_keepout_and_physics",
        }


@dataclass(frozen=True)
class CriticalSceneDistribution:
    """An evidence-weighted conditional proposal distribution for one query trajectory."""

    query: CriticalSupport
    archetype_probabilities: Mapping[str, float]
    empirical_quantile_mean: float
    empirical_quantile_std: float
    quantile_exploration_std: float
    evidence_record_weights: Mapping[str, float]
    evidence_source_weights: Mapping[str, float]
    exploration_mass: float

    def __post_init__(self) -> None:
        if not self.archetype_probabilities:
            raise CriticalDistributionError("the archetype distribution is empty")
        total = sum(self.archetype_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise CriticalDistributionError(f"archetype probabilities sum to {total}, not one")
        if any(value <= 0.0 for value in self.archetype_probabilities.values()):
            raise CriticalDistributionError("every retained archetype needs positive mass")

    @property
    def effective_evidence_records(self) -> float:
        return 1.0 / sum(weight * weight for weight in self.evidence_record_weights.values())

    def sample(self, seed: int) -> CriticalSceneProposal:
        """Sample deterministically from the fitted categorical and critical-window density."""

        rng = random.Random(seed)
        draw = rng.random()
        cumulative = 0.0
        selected = sorted(self.archetype_probabilities)[-1]
        for archetype_id in sorted(self.archetype_probabilities):
            cumulative += self.archetype_probabilities[archetype_id]
            if draw <= cumulative:
                selected = archetype_id
                break

        std = math.hypot(self.empirical_quantile_std, self.quantile_exploration_std)
        quantile = self.empirical_quantile_mean
        if std > 0:
            for _ in range(64):
                candidate = rng.gauss(self.empirical_quantile_mean, std)
                if 0.0 <= candidate <= 1.0:
                    quantile = candidate
                    break
            else:
                quantile = min(1.0, max(0.0, quantile))
        return CriticalSceneProposal(
            archetype_id=selected,
            hard_quantile=quantile,
            hard_coordinate_m=self.query.coordinate_at(quantile),
            geometry_seed=seed,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "q_lfh_conditional_v1",
            "query": self.query.to_dict(quantile=self.empirical_quantile_mean),
            "archetype_probabilities": dict(sorted(self.archetype_probabilities.items())),
            "critical_quantile": {
                "empirical_mean": self.empirical_quantile_mean,
                "empirical_std": self.empirical_quantile_std,
                "explicit_exploration_std": self.quantile_exploration_std,
            },
            "evidence_record_weights": dict(sorted(self.evidence_record_weights.items())),
            "evidence_source_weights": dict(sorted(self.evidence_source_weights.items())),
            "effective_evidence_records": self.effective_evidence_records,
            "exploration_mass": self.exploration_mass,
            "physics_verdict_predicted": False,
        }


def compatible_archetypes(axis_type: str) -> tuple[str, ...]:
    """Return deterministic visual families supported by one constraint axis."""

    return tuple(
        sorted(
            archetype_id
            for archetype_id, archetype in ARCHETYPES.items()
            if archetype.axis_type == axis_type
        )
    )


def _feature_distance_sq(query: CriticalSupport, evidence: CriticalSupport) -> float:
    """Distance over trajectory-local variables, with fixed interpretable bandwidths."""

    route = (query.route_progress - evidence.route_progress) / 0.15
    exposure = math.log(query.face_along_route_m / evidence.face_along_route_m) / math.log(3.0)
    width = math.log(query.engineering_width_m / evidence.engineering_width_m) / math.log(3.0)
    across = math.log(query.face_across_route_m / evidence.face_across_route_m) / math.log(3.0)
    operator = (
        0.0
        if query.operator == evidence.operator or "unknown" in {query.operator, evidence.operator}
        else 1.0
    )
    return route * route + exposure * exposure + width * width + across * across + operator


def infer_critical_scene_distribution(
    query: CriticalSupport,
    atoms: Iterable[VerifiedSceneAtom],
    *,
    excluded_source_pair_id: str | None = None,
    excluded_archetypes: Iterable[str] = (),
    exploration_mass: float = 0.10,
    quantile_exploration_std: float = 0.05,
) -> CriticalSceneDistribution:
    """Infer ``q(G | trajectory, critical condition)`` from verified neighboring atoms.

    Source balancing is applied before the trajectory kernel so multiplying variants from one
    motion cannot dominate the conditional density.  ``excluded_source_pair_id`` enables honest
    leave-one-source-out evaluation and novelty proposals without self-copying.
    """

    if query.engineering_width_m <= 0:
        raise CriticalDistributionError("query trajectory has empty critical support")
    if not 0.0 < exploration_mass < 1.0:
        raise ValueError("exploration_mass must lie strictly between zero and one")
    if not math.isfinite(quantile_exploration_std) or quantile_exploration_std < 0:
        raise ValueError("quantile_exploration_std must be finite and non-negative")

    compatible = [
        atom
        for atom in atoms
        if atom.support.axis_type == query.axis_type
        and atom.support.binding_keypoint == query.binding_keypoint
        and atom.source_pair_id != excluded_source_pair_id
        and atom.support.engineering_width_m > 0
    ]
    if not compatible:
        raise CriticalDistributionError("no independent compatible evidence atoms")

    by_source: dict[str, list[VerifiedSceneAtom]] = {}
    for atom in compatible:
        by_source.setdefault(atom.source_pair_id, []).append(atom)
    unnormalized: dict[str, float] = {}
    for atom in compatible:
        source_mass = 1.0 / len(by_source)
        within_source_mass = source_mass / len(by_source[atom.source_pair_id])
        kernel = math.exp(-0.5 * _feature_distance_sq(query, atom.support))
        unnormalized[atom.support.record_id] = within_source_mass * kernel
    normalizer = sum(unnormalized.values())
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise CriticalDistributionError("trajectory kernel has zero evidence mass")
    atom_weights = {record_id: value / normalizer for record_id, value in unnormalized.items()}

    candidates = set(compatible_archetypes(query.axis_type)) - set(excluded_archetypes)
    if not candidates:
        raise CriticalDistributionError("all compatible archetypes were excluded")
    learned = {archetype_id: 0.0 for archetype_id in candidates}
    evidence_by_record = {atom.support.record_id: atom for atom in compatible}
    for record_id, atom_weight in atom_weights.items():
        atom = evidence_by_record[record_id]
        retained = candidates.intersection(atom.verified_archetypes)
        if not retained:
            continue
        for archetype_id in retained:
            learned[archetype_id] += atom_weight / len(retained)
    learned_total = sum(learned.values())
    if learned_total > 0:
        learned = {key: value / learned_total for key, value in learned.items()}
    else:
        learned = {key: 1.0 / len(candidates) for key in candidates}
    probabilities = {
        key: (1.0 - exploration_mass) * learned[key] + exploration_mass / len(candidates)
        for key in candidates
    }

    quantile_mean = sum(
        atom_weights[atom.support.record_id] * atom.hard_quantile for atom in compatible
    )
    quantile_variance = sum(
        atom_weights[atom.support.record_id] * (atom.hard_quantile - quantile_mean) ** 2
        for atom in compatible
    )
    source_weights = {
        source: sum(atom_weights[atom.support.record_id] for atom in source_atoms)
        for source, source_atoms in by_source.items()
    }
    return CriticalSceneDistribution(
        query=query,
        archetype_probabilities=probabilities,
        empirical_quantile_mean=quantile_mean,
        empirical_quantile_std=math.sqrt(max(0.0, quantile_variance)),
        quantile_exploration_std=quantile_exploration_std,
        evidence_record_weights=atom_weights,
        evidence_source_weights=source_weights,
        exploration_mass=exploration_mass,
    )


def load_q_lfh_atoms(path: Path) -> tuple[VerifiedSceneAtom, ...]:
    """Load the evidence-backed ``q_lfh_v1`` artifact as conditional training atoms."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "q_lfh_v1":
        raise CriticalDistributionError("expected q_lfh_v1 evidence")
    margin = float(payload["engineering_margin"]["value_mm_each_side"]) / 1000.0
    result: list[VerifiedSceneAtom] = []
    for raw in payload.get("atoms", []):
        lower = float(raw["engineering_support_lower_m"])
        upper = float(raw["engineering_support_upper_m"])
        support = CriticalSupport(
            record_id=f"{raw['source_pair_id']}::verified_atom",
            source_pair_id=str(raw["source_pair_id"]),
            operator=str(raw.get("operator") or "unknown"),
            axis_type=str(raw["constraint_axis"]),
            binding_keypoint=str(raw["binding_keypoint"]),
            route_progress=float(raw["route_progress"]),
            face_along_route_m=float(raw["face_along_route_m"]),
            face_across_route_m=float(raw["face_across_route_m"]),
            nominal_reach_m=upper + margin,
            adapted_reach_m=lower - margin,
            clear_margin_m=margin,
            strike_margin_m=margin,
            context_status="physics_verified_source_archetype_atom",
        )
        hard_quantile = support.normalized_position(float(raw["hard_coordinate_m"]))
        result.append(
            VerifiedSceneAtom(
                support=support,
                verified_archetypes=tuple(
                    sorted(item["archetype_id"] for item in raw["verified_archetypes"])
                ),
                hard_quantile=hard_quantile,
            )
        )
    if not result:
        raise CriticalDistributionError("q_lfh_v1 contains no atoms")
    return tuple(result)
