from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS
from gear_sonic.research.lace.transport import (
    TRANSPORT_PROTOCOL_DIGEST_FIELD,
    build_cross_policy_transport_report,
    validate_transport_protocol,
)

REFERENCE = "release"
TRAINEE = "early"


def _protocol() -> dict:
    protocol = {
        "kind": "lace_cross_policy_transport_protocol",
        "schema_version": 1,
        "frozen": True,
        "scientific_use": True,
        "report_only": True,
        "causal_capacity_claim_allowed": False,
        "declared_before_primary_atlas_outcomes": True,
        "split_selection_sha256": "1" * 64,
        "partition": "D_atlas",
        "mechanism_names": list(DEFAULT_MECHANISMS),
        "reference_policy_id": REFERENCE,
        "trainee_policy_ids": [TRAINEE, "mid", "late"],
        "comparison_rule": ("reference_vs_each_trainee_stage_separately_no_policy_axis_averaging"),
        "normalizer_rule": ("apply_one_primary_lite_d_atlas_normalizer_to_both_policies_no_refit"),
        "missingness_rule": "complete_pair_support_no_imputation",
        "rollouts_per_motion_policy": 8,
        "minimum_resolved_failures": 3,
        "neighbor_k": 3,
        "bootstrap_unit": "source_group",
        "bootstrap_replicates": 100,
        "bootstrap_seed": 37,
        "confidence_level": 0.95,
        "minimum_common_support_motion_count": 8,
        "minimum_common_support_fraction": 0.5,
        "minimum_common_support_source_group_count": 4,
        "low_transport_action": (
            "continue_trainee_geometry_and_narrow_external_validity_not_an_h1_kill"
        ),
    }
    protocol[TRANSPORT_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=TRANSPORT_PROTOCOL_DIGEST_FIELD,
    )
    return protocol


def _records(policy_id: str, *, permutation: tuple[int, ...] | None = None) -> list[dict]:
    records = []
    for motion_index in range(12):
        raw = [
            1.0 + motion_index,
            2.0 + (motion_index * 2) % 7,
            3.0 + (motion_index * 3) % 5,
            4.0 + (motion_index * 5) % 11,
            5.0 + (motion_index * 7) % 13,
            6.0 + (motion_index * 11) % 17,
        ]
        total = sum(raw)
        q = [value / total for value in raw]
        if permutation is not None:
            q = [q[index] for index in permutation]
        records.append(
            {
                "motion_key": f"motion_{motion_index:02d}",
                "probe_policy_id": policy_id,
                "num_rollouts": 8,
                "num_resolved_failures": 5,
                "q": q,
            }
        )
    return records


def _groups() -> dict[str, str]:
    return {f"motion_{index:02d}": f"source_{index // 2:02d}" for index in range(12)}


def _bindings() -> dict[str, str]:
    return {
        "reference_artifact_sha256": "2" * 64,
        "trainee_artifact_sha256": "3" * 64,
        "parent_normalizer_sha256": "4" * 64,
        "measurement_protocol_sha256": "5" * 64,
        "condition_grid_sha256": "6" * 64,
    }


def _report(*, trainee: list[dict] | None = None) -> dict:
    return build_cross_policy_transport_report(
        _records(REFERENCE),
        _records(TRAINEE) if trainee is None else trainee,
        trainee_policy_id=TRAINEE,
        source_group_by_motion=_groups(),
        artifact_bindings=_bindings(),
        protocol=_protocol(),
    )


def test_identical_policy_coordinates_have_perfect_transport_and_are_deterministic() -> None:
    first = _report()
    second = _report()

    assert first == second
    assert first["causal_capacity_claim_allowed"] is False
    assert first["coverage"]["passes"] is True
    assert first["mean_motion_js_divergence_bits"] == pytest.approx(0.0)
    assert first["distance_matrix_spearman"] == pytest.approx(1.0)
    assert first["mean_neighbor_overlap_fraction"] == pytest.approx(1.0)
    assert all(
        row["point_spearman"] == pytest.approx(1.0)
        for row in first["per_channel_spearman"].values()
    )
    assert first["transport_report_sha256"] == canonical_sha256(
        first,
        digest_field="transport_report_sha256",
    )


def test_channel_permutation_changes_semantics_but_preserves_pairwise_geometry() -> None:
    trainee = _records(TRAINEE, permutation=(1, 0, 2, 3, 4, 5))
    report = _report(trainee=trainee)

    assert report["mean_motion_js_divergence_bits"] > 0.0
    assert report["distance_matrix_spearman"] == pytest.approx(1.0)


def test_missing_q_is_excluded_without_imputation_and_can_fail_coverage() -> None:
    trainee = _records(TRAINEE)
    for record in trainee[:5]:
        record["q"] = None
        record["num_resolved_failures"] = 0
    report = _report(trainee=trainee)

    assert report["num_motion_keys_common_support"] == 7
    assert report["coverage"]["passes"] is False
    assert len(report["excluded_motion_keys"]) == 5
    assert all(row["trainee_q_missing"] for row in report["excluded_motion_keys"])


@pytest.mark.parametrize(
    ("q", "resolved", "message"),
    [
        (None, 3, "missing q despite sufficient"),
        ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, "without sufficient"),
    ],
)
def test_q_presence_must_match_the_frozen_evidence_floor(q, resolved: int, message: str) -> None:
    trainee = _records(TRAINEE)
    trainee[0]["q"] = q
    trainee[0]["num_resolved_failures"] = resolved

    with pytest.raises(ValueError, match=message):
        _report(trainee=trainee)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "motion universes"),
        (lambda rows: rows[0].update(num_rollouts=7), "rollout count drifted"),
        (lambda rows: rows[0].update(q=[0.5, 0.5]), "length mismatch"),
        (lambda rows: rows[0].update(q=[0.5] * 6), "sum to one"),
    ],
)
def test_signature_drift_fails_closed(mutation, message: str) -> None:
    trainee = _records(TRAINEE)
    mutation(trainee)

    with pytest.raises(ValueError, match=message):
        _report(trainee=trainee)


def test_artifact_bindings_and_selected_stage_are_strict() -> None:
    bindings = _bindings()
    bindings["reference_artifact_sha256"] = bindings["trainee_artifact_sha256"]
    with pytest.raises(ValueError, match="must be distinct"):
        build_cross_policy_transport_report(
            _records(REFERENCE),
            _records(TRAINEE),
            trainee_policy_id=TRAINEE,
            source_group_by_motion=_groups(),
            artifact_bindings=bindings,
            protocol=_protocol(),
        )

    with pytest.raises(ValueError, match="not declared"):
        build_cross_policy_transport_report(
            _records(REFERENCE),
            _records("unknown"),
            trainee_policy_id="unknown",
            source_group_by_motion=_groups(),
            artifact_bindings=_bindings(),
            protocol=_protocol(),
        )


def test_protocol_tamper_and_capacity_relabel_are_rejected() -> None:
    protocol = _protocol()
    stale = deepcopy(protocol)
    stale["bootstrap_seed"] += 1
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        validate_transport_protocol(stale)

    relabeled = deepcopy(protocol)
    relabeled["causal_capacity_claim_allowed"] = True
    relabeled[TRANSPORT_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        relabeled,
        digest_field=TRANSPORT_PROTOCOL_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="flags are invalid"):
        validate_transport_protocol(relabeled)


def test_frozen_scale512_protocol_is_valid() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/research/lace/cross_policy_transport_scale512_v1.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))

    validate_transport_protocol(protocol)
    assert protocol["reference_policy_id"] == "sonic_release_external_audit"
    assert protocol["trainee_policy_ids"] == [
        "pi_lite_early",
        "pi_lite_mid",
        "pi_lite_late",
    ]
    assert protocol["report_only"] is True
