"""Tests for trajectory-conditioned LFH critical-scene inference."""

from __future__ import annotations

import json
import math

import pytest

from gear_sonic.dataset_generation.hallucination.critical_distribution import (
    CriticalDistributionError,
    VerifiedSceneAtom,
    infer_critical_scene_distribution,
    load_q_lfh_atoms,
)
from gear_sonic.dataset_generation.hallucination.critical_support import CriticalSupport


def support(record: str, source: str, route: float = 0.5, exposure: float = 0.1):
    return CriticalSupport(
        record_id=record,
        source_pair_id=source,
        operator="local_crouch",
        axis_type="overhead",
        binding_keypoint="head_torso",
        route_progress=route,
        face_along_route_m=exposure,
        face_across_route_m=3.0,
        nominal_reach_m=1.30,
        adapted_reach_m=1.23,
        clear_margin_m=0.018,
        strike_margin_m=0.018,
        context_status="verified",
    )


def atom(record: str, source: str, archetypes: tuple[str, ...], **kwargs):
    return VerifiedSceneAtom(support(record, source, **kwargs), archetypes, 0.5)


def test_inference_excludes_query_source_and_preserves_exact_support() -> None:
    query = support("query", "source_q", route=0.51)
    atoms = (
        atom("self", "source_q", ("hvac_duct",), route=0.51),
        atom("a", "source_a", ("shelf_plank", "ibeam"), route=0.49),
        atom("b", "source_b", ("hanging_panel",), route=0.7),
    )
    distribution = infer_critical_scene_distribution(
        query,
        atoms,
        excluded_source_pair_id="source_q",
    )

    assert "self" not in distribution.evidence_record_weights
    assert math.isclose(sum(distribution.archetype_probabilities.values()), 1.0)
    assert (
        distribution.archetype_probabilities["shelf_plank"]
        > distribution.archetype_probabilities["hvac_duct"]
    )
    for seed in range(50):
        proposal = distribution.sample(seed)
        assert query.lower_m <= proposal.hard_coordinate_m <= query.upper_m
        assert 0.0 <= proposal.hard_quantile <= 1.0


def test_source_balancing_prevents_duplicated_source_from_dominating() -> None:
    query = support("query", "query")
    atoms = (
        atom("a1", "source_a", ("shelf_plank",)),
        atom("a2", "source_a", ("shelf_plank",)),
        atom("b1", "source_b", ("hanging_panel",)),
    )
    distribution = infer_critical_scene_distribution(query, atoms)

    assert distribution.evidence_source_weights["source_a"] == pytest.approx(0.5)
    assert distribution.evidence_source_weights["source_b"] == pytest.approx(0.5)


def test_incomplete_cross_infers_verified_hanging_panel_as_novel_candidate() -> None:
    query = support("query", "source_089")
    atoms = (
        atom("a", "source_cf", ("door_lintel", "ibeam", "shelf_plank")),
        atom("b", "source_086", ("hanging_panel", "ibeam", "shelf_plank")),
        atom("c", "source_090", ("door_lintel", "ibeam", "shelf_plank")),
    )
    distribution = infer_critical_scene_distribution(
        query,
        atoms,
        excluded_archetypes=("door_lintel", "ibeam", "shelf_plank"),
    )

    top = max(distribution.archetype_probabilities, key=distribution.archetype_probabilities.get)
    assert top == "hanging_panel"
    assert distribution.archetype_probabilities["hanging_panel"] > 0.9


def test_empty_or_incompatible_evidence_refuses() -> None:
    query = support("query", "source_q")
    lateral = CriticalSupport(
        record_id="lateral",
        source_pair_id="source_l",
        operator="local_arm_tuck",
        axis_type="lateral_gap",
        binding_keypoint="wrist_right",
        route_progress=0.5,
        face_along_route_m=0.1,
        face_across_route_m=1.0,
        nominal_reach_m=0.9,
        adapted_reach_m=0.8,
        clear_margin_m=0.01,
        strike_margin_m=0.01,
        context_status="verified",
    )
    with pytest.raises(CriticalDistributionError, match="no independent compatible"):
        infer_critical_scene_distribution(
            query,
            (VerifiedSceneAtom(lateral, ("pinch_panels",), 0.5),),
        )


def test_loads_q_lfh_v1_without_treating_easy_xi_as_critical_support(tmp_path) -> None:
    payload = {
        "schema_version": "q_lfh_v1",
        "engineering_margin": {"value_mm_each_side": 18.0},
        "atoms": [
            {
                "source_pair_id": "source_a",
                "operator": "local_crouch",
                "constraint_axis": "overhead",
                "binding_keypoint": "head_torso",
                "route_progress": 0.5,
                "face_along_route_m": 0.1,
                "face_across_route_m": 3.0,
                "engineering_support_lower_m": 1.24,
                "engineering_support_upper_m": 1.28,
                "hard_coordinate_m": 1.26,
                "easy_xi": 3.0,
                "verified_archetypes": [{"archetype_id": "shelf_plank"}],
            }
        ],
    }
    path = tmp_path / "q.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_q_lfh_atoms(path)

    assert loaded[0].hard_quantile == pytest.approx(0.5)
    assert loaded[0].support.lower_m == pytest.approx(1.24)
    assert loaded[0].support.upper_m == pytest.approx(1.28)
