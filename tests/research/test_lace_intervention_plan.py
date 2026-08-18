from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gear_sonic.research.lace.intervention_plan import (
    INTERVENTION_PLAN_DIGEST_FIELD,
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    build_intervention_plan,
    intervention_probabilities_by_panel,
    validate_intervention_plan,
    validate_intervention_protocol,
)
from gear_sonic.research.lace.panels import build_representation_blind_panels
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split


def _split() -> dict:
    motions = []
    for group_index in range(40):
        for variant in range(1 + group_index % 3):
            motions.append(
                {
                    "motion_key": f"motion_{group_index:02d}_{variant}__A{group_index:03d}",
                    "release_filter_key": f"motion_{group_index:02d}_{variant}.pkl",
                    "source_group_id": f"actor_{group_index:03d}",
                    "duration_source_frames": 120 + group_index * 5 + variant,
                    "stratum": f"duration_q{group_index % 4}",
                }
            )
    return build_source_disjoint_split(motions, seed=81)


def _inputs() -> tuple[dict, dict, dict]:
    split = _split()
    panel_count = 4
    panel_seed = 27
    panels = build_representation_blind_panels(
        split,
        panel_count=panel_count,
        seed=panel_seed,
    )
    motion_count = sum(record["partition"] == "D_curriculum" for record in split["motions"])
    protocol = {
        "kind": "lace_rq1_intervention_protocol",
        "schema_version": 1,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_curriculum",
        "expected_motion_count": motion_count,
        "expected_panel_count": panel_count,
        "panel_seed": panel_seed,
        "base_distribution_method": "uniform_over_canonically_ordered_motions_v1",
        "sequence_length_agnostic": True,
        "target_kl_nats": 0.03,
        "max_probability_ratio": 3.0,
        "kl_tolerance": 1e-12,
        "probability_tolerance": 1e-12,
        "bisection_iterations": 100,
        "maximum_added_exposure_range": 0.02,
        "interpretation": "unit-test pre-outcome dose",
    }
    protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )
    return split, panels, protocol


def test_plan_is_deterministic_fixed_budget_and_exactly_covers_motion_order() -> None:
    split, panels, protocol = _inputs()

    first = build_intervention_plan(split, panels, protocol)
    second = build_intervention_plan(split, panels, protocol)

    assert first == second
    assert first[INTERVENTION_PLAN_DIGEST_FIELD] == canonical_sha256(
        first,
        digest_field=INTERVENTION_PLAN_DIGEST_FIELD,
    )
    expected_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_curriculum"
    )
    assert first["motion_keys"] == expected_keys
    assert first["motion_count"] == len(expected_keys)
    assert first["panel_count"] == 4
    assert first["dose_balance"]["passes"] is True
    assert first["dose_balance"]["added_panel_exposure_range"] <= 0.02
    for row in first["interventions"]:
        intervention = row["intervention"]
        assert intervention["realized_kl"] == pytest.approx(0.03, abs=1e-12)
        assert sum(intervention["p_plus"]) == pytest.approx(1.0, abs=1e-15)
        assert sum(intervention["delta_p"]) == pytest.approx(0.0, abs=1e-15)
        assert intervention["ratio_diagnostics"]["maximum_probability_ratio"] <= 3.0


def test_whole_plan_reconstruction_rejects_fully_rehashed_probability_tamper() -> None:
    split, panels, protocol = _inputs()
    plan = build_intervention_plan(split, panels, protocol)
    tampered = deepcopy(plan)
    probabilities = tampered["interventions"][0]["intervention"]["p_plus"]
    low = min(range(len(probabilities)), key=probabilities.__getitem__)
    high = max(range(len(probabilities)), key=probabilities.__getitem__)
    assert probabilities[low] != probabilities[high]
    probabilities[low], probabilities[high] = probabilities[high], probabilities[low]
    nested = tampered["interventions"][0]["intervention"]
    nested["intervention_probabilities"] = list(probabilities)
    nested["intervention_sha256"] = canonical_sha256(
        nested,
        digest_field="intervention_sha256",
    )
    tampered[INTERVENTION_PLAN_DIGEST_FIELD] = canonical_sha256(
        tampered,
        digest_field=INTERVENTION_PLAN_DIGEST_FIELD,
    )

    with pytest.raises(ValueError, match="deterministic reconstruction"):
        validate_intervention_plan(
            tampered,
            split_manifest=split,
            panel_manifest=panels,
            protocol=protocol,
        )


def test_deep_panel_reconstruction_rejects_rehashed_seed_claim() -> None:
    split, panels, protocol = _inputs()
    forged = deepcopy(panels)
    forged["seed"] += 1
    forged["panel_sha256"] = canonical_sha256(forged, digest_field="panel_sha256")
    forged_protocol = deepcopy(protocol)
    forged_protocol["panel_seed"] = forged["seed"]
    forged_protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        forged_protocol,
        digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )

    with pytest.raises(ValueError, match="deterministic representation-blind reconstruction"):
        build_intervention_plan(split, forged, forged_protocol)


def test_protocol_digest_and_preoutcome_guards_fail_closed() -> None:
    _, _, protocol = _inputs()
    validate_intervention_protocol(protocol)

    stale = deepcopy(protocol)
    stale["target_kl_nats"] = 0.04
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        validate_intervention_protocol(stale)

    post_outcome = deepcopy(protocol)
    post_outcome["declared_before_transfer_outcomes"] = False
    post_outcome[INTERVENTION_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        post_outcome,
        digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="before transfer outcomes"):
        validate_intervention_protocol(post_outcome)


def test_probability_accessor_preserves_frozen_motion_order() -> None:
    split, panels, protocol = _inputs()
    plan = build_intervention_plan(split, panels, protocol)
    probabilities = intervention_probabilities_by_panel(plan)

    assert list(probabilities) == [row["panel_id"] for row in plan["interventions"]]
    assert all(len(values) == plan["motion_count"] for values in probabilities.values())


def test_scale512_protocol_is_valid_and_self_hashed() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/research/lace/rq1_intervention_protocol_scale512_v1.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    validate_intervention_protocol(protocol)
    assert protocol["expected_motion_count"] == 233
    assert protocol["expected_panel_count"] == 8
    assert protocol["target_kl_nats"] == 0.05
    assert protocol["maximum_added_exposure_range"] == 0.002
