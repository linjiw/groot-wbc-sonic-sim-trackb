import copy
import json
from pathlib import Path

import pytest

from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.stability import (
    bootstrap_split_half_channel_spearman,
    compare_split_half_signatures,
    cross_validated_difficulty_reconstruction,
    evaluate_stability_gate,
    summarize_mechanism_incidence,
)

MECHANISMS = ("contact", "slip", "drift")
POLICIES = ("early", "late")


def _records(*, swap: bool = False):
    values = {
        "m0": ([0.8, 0.1, 0.1], [0.7, 0.2, 0.1]),
        "m1": ([0.1, 0.8, 0.1], [0.2, 0.7, 0.1]),
        "m2": ([0.1, 0.1, 0.8], [0.1, 0.2, 0.7]),
        "m3": ([0.6, 0.3, 0.1], [0.5, 0.4, 0.1]),
    }
    records = []
    for motion_key, (early, late) in values.items():
        for policy_id, q in zip(POLICIES, (early, late), strict=True):
            if swap:
                q = [q[1], q[0], q[2]]
            records.append(
                {
                    "motion_key": motion_key,
                    "probe_policy_id": policy_id,
                    "num_rollouts": 4,
                    "num_resolved_failures": 3,
                    "q": q,
                }
            )
    return records


def test_identical_halves_are_perfectly_stable_and_deterministic():
    records = _records()
    first = compare_split_half_signatures(
        records,
        copy.deepcopy(records),
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        neighbor_k=2,
    )
    second = compare_split_half_signatures(
        records,
        copy.deepcopy(records),
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        neighbor_k=2,
    )

    assert first == second
    assert first["mean_motion_policy_js_divergence_bits"] == pytest.approx(0.0)
    assert first["distance_matrix_spearman"] == pytest.approx(1.0)
    assert first["mean_neighbor_overlap_fraction"] == pytest.approx(1.0)
    assert first["common_support_fraction"] == pytest.approx(1.0)
    assert first["per_policy_channel_spearman"]["early"]["contact"] == pytest.approx(1.0)


def test_channel_permutation_is_detected_even_when_geometry_is_preserved():
    result = compare_split_half_signatures(
        _records(),
        _records(swap=True),
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        neighbor_k=2,
    )

    assert result["mean_motion_policy_js_divergence_bits"] > 0.0
    # A global channel permutation preserves pairwise geometry, which is why
    # direct per-motion divergence and geometry stability are both reported.
    assert result["distance_matrix_spearman"] == pytest.approx(1.0)


def test_primary_support_requires_every_policy_in_both_halves():
    right = _records()
    right[0]["q"] = None
    right[0]["num_resolved_failures"] = 0

    result = compare_split_half_signatures(
        _records(),
        right,
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        neighbor_k=9,
    )

    assert result["num_motion_keys_common_support"] == 3
    assert result["neighbor_k_effective"] == 2
    assert result["excluded_motion_keys"] == [
        {
            "motion_key": "m0",
            "unresolved_policy_ids_a": [],
            "unresolved_policy_ids_b": ["early"],
        }
    ]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda records: records.pop(), "same motion-policy"),
        (lambda records: records[0].update(q=[0.5, 0.5]), "length"),
        (lambda records: records[0].update(q=[0.5, 0.5, 0.5]), "sum to one"),
        (lambda records: records[0].update(num_rollouts=0), "positive integer"),
    ],
)
def test_invalid_or_mismatched_inputs_fail_closed(mutator, match):
    right = _records()
    mutator(right)
    with pytest.raises(ValueError, match=match):
        compare_split_half_signatures(
            _records(),
            right,
            mechanism_names=MECHANISMS,
            policy_order=POLICIES,
            neighbor_k=2,
        )


def test_fewer_than_three_complete_motions_is_rejected():
    left = _records()
    right = _records()
    for record in right:
        if record["motion_key"] in {"m0", "m1"}:
            record["q"] = None
            record["num_resolved_failures"] = 0

    with pytest.raises(ValueError, match="at least three"):
        compare_split_half_signatures(
            left,
            right,
            mechanism_names=MECHANISMS,
            policy_order=POLICIES,
            neighbor_k=1,
        )


def _reconstruction_records():
    records = []
    source_groups = {}
    folds = {}
    for motion_index in range(18):
        motion_key = f"m{motion_index:02d}"
        group = f"g{motion_index // 2:02d}"
        source_groups[motion_key] = group
        folds[group] = (motion_index // 2) % 3
        early_difficulty = 0.05 + 0.045 * motion_index
        late_difficulty = 0.9 - 0.04 * motion_index
        for policy_id, difficulty in (
            ("early", early_difficulty),
            ("late", late_difficulty),
        ):
            contact = 0.15 + 0.7 * difficulty
            records.append(
                {
                    "motion_key": motion_key,
                    "probe_policy_id": policy_id,
                    "num_rollouts": 8,
                    "num_resolved_failures": 5,
                    "difficulty": difficulty,
                    "q": [contact, 1.0 - contact],
                }
            )
    return records, source_groups, folds


def test_difficulty_reconstruction_is_source_group_cross_validated():
    records, source_groups, folds = _reconstruction_records()
    result = cross_validated_difficulty_reconstruction(
        records,
        mechanism_names=("contact", "other"),
        policy_order=POLICIES,
        source_group_by_motion=source_groups,
        fold_by_source_group=folds,
        ridge_alpha=0.0,
    )

    assert result["cross_validated_reconstruction_r2"] == pytest.approx(1.0)
    assert result["common_support_fraction"] == pytest.approx(1.0)
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert set(fold["train_source_groups"]).isdisjoint(fold["test_source_groups"])


def test_difficulty_reconstruction_excludes_whole_motion_on_missing_policy_q():
    records, source_groups, folds = _reconstruction_records()
    records[0]["q"] = None
    records[0]["num_resolved_failures"] = 0
    result = cross_validated_difficulty_reconstruction(
        records,
        mechanism_names=("contact", "other"),
        policy_order=POLICIES,
        source_group_by_motion=source_groups,
        fold_by_source_group=folds,
        ridge_alpha=0.1,
    )

    assert result["num_motion_keys_common_support"] == 17
    assert result["excluded_motion_keys"] == [
        {"motion_key": "m00", "unresolved_policy_ids": ["early"]}
    ]


def test_difficulty_reconstruction_rejects_source_group_leakage_contract_drift():
    records, source_groups, folds = _reconstruction_records()
    source_groups.pop("m00")
    with pytest.raises(ValueError, match="exactly match observed motion"):
        cross_validated_difficulty_reconstruction(
            records,
            mechanism_names=("contact", "other"),
            policy_order=POLICIES,
            source_group_by_motion=source_groups,
            fold_by_source_group=folds,
            ridge_alpha=0.1,
        )


def test_source_group_bootstrap_is_deterministic_and_perfect_for_identical_halves():
    records = _records()
    source_groups = {f"m{index}": f"g{index}" for index in range(4)}
    first = bootstrap_split_half_channel_spearman(
        records,
        copy.deepcopy(records),
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        source_group_by_motion=source_groups,
        bootstrap_replicates=100,
        bootstrap_seed=17,
    )
    second = bootstrap_split_half_channel_spearman(
        records,
        copy.deepcopy(records),
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        source_group_by_motion=source_groups,
        bootstrap_replicates=100,
        bootstrap_seed=17,
    )

    assert first == second
    assert first["num_source_groups_common_support"] == 4
    row = first["channels"]["early"]["contact"]
    assert row["point_spearman"] == pytest.approx(1.0)
    assert row["valid_bootstrap_fraction"] > 0.9
    assert row["bootstrap_ci_lower_median_upper"] == pytest.approx([1.0, 1.0, 1.0])


def test_mechanism_incidence_uses_resolved_failure_mass_and_groups():
    records = _records()
    for record in records:
        record["attributed_failure_rate"] = 0.75
    result = summarize_mechanism_incidence(
        records,
        mechanism_names=MECHANISMS,
        policy_order=POLICIES,
        source_group_by_motion={f"m{index}": f"g{index}" for index in range(4)},
    )

    assert result["num_supported_signatures"] == 8
    assert result["total_attributed_mechanism_mass"] == pytest.approx(6.0)
    assert sum(row["global_mass_fraction"] for row in result["mechanisms"]) == pytest.approx(1.0)
    assert all(row["positive_source_group_count"] >= 3 for row in result["mechanisms"])


def _gate_config():
    config = {
        "schema_version": 1,
        "kind": "lace_signature_stability_gate_config",
        "frozen": True,
        "mechanism_names": list(MECHANISMS),
        "policy_order": list(POLICIES),
        "analysis": {
            "neighbor_k": 2,
            "bootstrap_replicates": 100,
            "bootstrap_seed": 17,
            "bootstrap_confidence_level": 0.95,
            "difficulty_ridge_alpha": 0.1,
        },
        "thresholds": {
            "minimum_common_support_motion_count": 3,
            "minimum_common_support_fraction": 0.5,
            "minimum_common_support_source_group_count": 3,
            "maximum_median_js_divergence_bits": 0.1,
            "minimum_distance_matrix_spearman": 0.5,
            "minimum_mean_neighbor_overlap_fraction": 0.4,
            "minimum_channel_point_spearman": 0.4,
            "minimum_channel_bootstrap_ci_lower": 0.0,
            "minimum_channel_valid_bootstrap_fraction": 0.9,
            "minimum_global_mechanism_mass_fraction": 0.01,
            "maximum_global_mechanism_mass_fraction": 0.6,
            "minimum_positive_source_groups_per_mechanism": 3,
            "maximum_single_source_group_share": 0.5,
            "maximum_difficulty_reconstruction_r2": 0.8,
        },
        "failure_action": {"failed": "stop"},
    }
    config["gate_config_sha256"] = canonical_sha256(
        config,
        digest_field="gate_config_sha256",
    )
    return config


def _passing_gate_reports():
    stability = {
        "kind": "lace_split_half_signature_stability",
        "mechanism_names": list(MECHANISMS),
        "policy_order": list(POLICIES),
        "neighbor_k_requested": 2,
        "num_motion_keys_common_support": 4,
        "common_support_fraction": 1.0,
        "median_motion_policy_js_divergence_bits": 0.01,
        "distance_matrix_spearman": 0.9,
        "mean_neighbor_overlap_fraction": 0.8,
    }
    bootstrap = {
        "kind": "lace_split_half_channel_spearman_bootstrap",
        "mechanism_names": list(MECHANISMS),
        "policy_order": list(POLICIES),
        "bootstrap_replicates": 100,
        "bootstrap_seed": 17,
        "confidence_level": 0.95,
        "num_source_groups_common_support": 4,
        "channels": {
            policy: {
                mechanism: {
                    "point_spearman": 0.8,
                    "valid_bootstrap_fraction": 1.0,
                    "bootstrap_ci_lower_median_upper": [0.2, 0.8, 0.95],
                }
                for mechanism in MECHANISMS
            }
            for policy in POLICIES
        },
    }
    incidence = {
        "kind": "lace_mechanism_incidence",
        "mechanism_names": list(MECHANISMS),
        "policy_order": list(POLICIES),
        "mechanisms": [
            {
                "mechanism": mechanism,
                "global_mass_fraction": 1.0 / len(MECHANISMS),
                "positive_source_group_count": 4,
                "maximum_single_source_group_share": 0.3,
            }
            for mechanism in MECHANISMS
        ],
    }
    reconstruction = {
        "kind": "lace_difficulty_reconstruction",
        "mechanism_names": list(MECHANISMS),
        "policy_order": list(POLICIES),
        "ridge_alpha": 0.1,
        "cross_validated_reconstruction_r2": 0.4,
    }
    return stability, bootstrap, incidence, reconstruction


def test_frozen_stability_gate_passes_and_fails_without_tuning():
    reports = _passing_gate_reports()
    passed = evaluate_stability_gate(*reports, gate_config=_gate_config())
    assert passed["passed"] is True
    assert passed["failed_checks"] == []

    reports[0]["distance_matrix_spearman"] = 0.2
    reports[2]["mechanisms"][0]["global_mass_fraction"] = 0.8
    failed = evaluate_stability_gate(*reports, gate_config=_gate_config())
    assert failed["passed"] is False
    assert "distance_matrix_spearman" in failed["failed_checks"]
    assert "incidence.contact.maximum_mass_fraction" in failed["failed_checks"]


def test_stability_gate_rejects_config_digest_or_analysis_drift():
    config = _gate_config()
    config["thresholds"]["maximum_median_js_divergence_bits"] = 0.2
    with pytest.raises(ValueError, match="gate_config_sha256 mismatch"):
        evaluate_stability_gate(*_passing_gate_reports(), gate_config=config)

    config = _gate_config()
    reports = _passing_gate_reports()
    reports[1]["bootstrap_seed"] = 18
    with pytest.raises(ValueError, match="seed does not match"):
        evaluate_stability_gate(*reports, gate_config=config)


def test_tracked_scale512_gate_config_has_a_valid_self_digest():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/research/lace/signature_stability_gate_scale512_v1.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["gate_config_sha256"] == canonical_sha256(
        config,
        digest_field="gate_config_sha256",
    )
