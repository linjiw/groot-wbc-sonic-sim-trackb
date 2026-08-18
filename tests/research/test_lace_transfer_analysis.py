from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.transfer_analysis import (
    TRANSFER_ANALYSIS_KIND,
    analyze_transfer_predictiveness,
    bootstrap_clustered_loss_improvement,
    permute_whole_source_failure_rows,
)

_BASELINE_NAMES = ["kinematic_distance", "difficulty"]
_FAILURE_NAMES = ["failure_distance"]
_BASELINE_HASH = "a" * 64
_FAILURE_HASH = "b" * 64
_GAIN_HASH = "c" * 64
_HEADROOM_HASH = "d" * 64


def _records(
    *,
    source_count: int = 4,
    group_motion_counts: tuple[int, ...] = (1, 2, 1, 2, 1),
    seed_count: int = 2,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    motion_index = 0
    for group_index, motion_count in enumerate(group_motion_counts):
        for local_motion_index in range(motion_count):
            target_motion_id = f"motion_{motion_index:02d}"
            for source_index in range(source_count):
                baseline = [
                    source_index + 0.2 * motion_index,
                    group_index + 0.1 * source_index + 0.05 * local_motion_index,
                ]
                failure = [(source_index - 1.5) * (motion_index - 3.0) + 0.2 * group_index]
                gain = 1.0 + 0.2 * baseline[0] - 0.1 * baseline[1] + 1.3 * failure[0]
                for paired_seed in range(seed_count):
                    headroom = 0.5 + 0.1 * motion_index + 0.01 * paired_seed
                    records.append(
                        {
                            "source_panel_id": f"panel_{source_index}",
                            "target_motion_id": target_motion_id,
                            "target_source_group_id": f"group_{group_index}",
                            "paired_seed": paired_seed,
                            "gain": gain,
                            "independent_headroom": headroom,
                            "baseline_features": list(baseline),
                            "failure_features": list(failure),
                        }
                    )
            motion_index += 1
    return records


def _analysis_kwargs(**overrides) -> dict:
    kwargs = {
        "baseline_feature_names": _BASELINE_NAMES,
        "failure_feature_names": _FAILURE_NAMES,
        "baseline_feature_provenance_sha256": _BASELINE_HASH,
        "failure_feature_provenance_sha256": _FAILURE_HASH,
        "gain_evaluation_artifact_sha256": _GAIN_HASH,
        "headroom_evaluation_artifact_sha256": _HEADROOM_HASH,
        "gain_rollout_seeds": [100, 101],
        "headroom_rollout_seeds": [200, 201],
        "h_min": 0.2,
        "alpha_grid": [1e-6, 1e-3, 1.0],
    }
    kwargs.update(overrides)
    return kwargs


def _analyze(
    records: list[dict[str, object]] | None = None,
    **overrides,
) -> dict:
    return analyze_transfer_predictiveness(
        _records() if records is None else records,
        **_analysis_kwargs(**overrides),
    )


def test_v2_analysis_is_deterministic_order_invariant_and_self_digesting() -> None:
    records = _records()
    original = deepcopy(records)

    first = _analyze(records)
    second = _analyze(list(reversed(records)))

    assert first == second
    assert records == original
    assert first["kind"] == TRANSFER_ANALYSIS_KIND
    assert first["schema_version"] == 2
    assert first["analysis_sha256"] == canonical_sha256(
        first,
        digest_field="analysis_sha256",
    )
    assert first["feature_schemas"]["baseline"]["names"] == _BASELINE_NAMES
    assert first["feature_schemas"]["failure"]["provenance_sha256"] == _FAILURE_HASH
    assert first["evaluation_provenance"]["rollout_seed_sets_disjoint"] is True
    assert first["primary_raw"]["status"] == "estimated"
    assert first["normalized_headroom_sensitivity"]["status"] == "estimated"


def test_primary_is_raw_gain_and_normalized_outcome_is_explicit_sensitivity() -> None:
    result = _analyze()
    primary = result["primary_raw"]["predictions"][0]
    sensitivity_by_key = {
        (
            prediction["source_panel_id"],
            prediction["target_motion_id"],
            prediction["paired_seed"],
        ): prediction
        for prediction in result["normalized_headroom_sensitivity"]["predictions"]
    }
    key = (
        primary["source_panel_id"],
        primary["target_motion_id"],
        primary["paired_seed"],
    )

    assert result["primary_raw"]["outcome_name"] == "primary_raw_gain"
    assert primary["outcome"] == primary["gain"]
    assert sensitivity_by_key[key]["outcome"] == pytest.approx(
        primary["gain"] / primary["independent_headroom"]
    )
    assert result["normalized_headroom_sensitivity"]["denominator_clipping"] is None
    assert result["normalized_headroom_sensitivity"]["h_min"] == 0.2


def test_motion_level_predictions_are_blocked_by_target_source_group() -> None:
    result = _analyze()
    primary = result["primary_raw"]
    assert primary["record_count"] == 56
    assert len(primary["predictions"]) == 56
    assert len({item["target_motion_id"] for item in primary["predictions"]}) == 7

    two_motion_fold = next(
        fold
        for fold in primary["outer_folds"]
        if fold["held_source_panel_id"] == "panel_0"
        and fold["held_target_source_group_id"] == "group_1"
    )
    assert two_motion_fold["held_target_motion_count"] == 2
    assert two_motion_fold["test_record_count"] == 4
    assert two_motion_fold["training_record_count"] == 30
    assert two_motion_fold["embargo_record_count"] == 22


def test_h_min_excludes_whole_motion_but_preserves_raw_primary() -> None:
    records = _records()
    for record in records:
        if record["target_motion_id"] == "motion_00" and record["paired_seed"] == 0:
            record["independent_headroom"] = 0.1
    result = _analyze(records, h_min=0.2)

    assert any(
        prediction["target_motion_id"] == "motion_00"
        for prediction in result["primary_raw"]["predictions"]
    )
    sensitivity = result["normalized_headroom_sensitivity"]
    assert sensitivity["status"] == "estimated"
    assert "motion_00" not in sensitivity["supported_target_motion_ids"]
    assert sensitivity["unsupported_targets"] == [
        {
            "target_motion_id": "motion_00",
            "target_source_group_id": "group_0",
            "minimum_independent_headroom": 0.1,
            "reason": "minimum_headroom_below_h_min",
        }
    ]


def test_normalized_sensitivity_reports_not_estimable_after_group_loss() -> None:
    records = _records()
    unsupported_groups = {"group_0", "group_1"}
    for record in records:
        if record["target_source_group_id"] in unsupported_groups:
            record["independent_headroom"] = 0.1
    result = _analyze(records, h_min=0.2)

    sensitivity = result["normalized_headroom_sensitivity"]
    assert sensitivity["status"] == "not_estimable"
    assert sensitivity["supported_target_source_group_count"] == 3
    assert "fewer_than_four" in sensitivity["reason"]


def test_training_only_standardization_cannot_see_held_group_intersection() -> None:
    records = _records()
    for record in records:
        if record["source_panel_id"] == "panel_0" and record["target_source_group_id"] == "group_0":
            record["baseline_features"] = [10_000.0, 20_000.0]
    result = _analyze(records)
    fold = next(
        item
        for item in result["primary_raw"]["outer_folds"]
        if item["held_source_panel_id"] == "panel_0"
        and item["held_target_source_group_id"] == "group_0"
    )
    expected = np.asarray(
        [
            record["baseline_features"]
            for record in records
            if record["source_panel_id"] != "panel_0"
            and record["target_source_group_id"] != "group_0"
        ],
        dtype=np.float64,
    )

    assert fold["baseline"]["outer_refit"]["feature_mean"] == pytest.approx(expected.mean(axis=0))
    assert max(fold["baseline"]["outer_refit"]["feature_mean"]) < 10.0


def test_losses_preserve_panel_seed_target_clusters_before_inference() -> None:
    result = _analyze()
    primary = result["primary_raw"]
    summaries = primary["cluster_loss_improvements"]

    assert len(summaries) == 4 * 5 * 2
    assert {summary["target_motion_count"] for summary in summaries} == {1, 2}
    assert len(primary["panel_seed_loss_improvements"]) == 4 * 2
    cluster_mean = math.fsum(
        summary["mean_paired_loss_improvement"] for summary in summaries
    ) / len(summaries)
    assert cluster_mean == pytest.approx(primary["metrics"]["paired_loss_improvement"])
    assert primary["metrics"]["individual_motion_cell_inference_permitted"] is False
    assert primary["metrics"]["paired_loss_improvement"] > 1.0


def test_duplicate_missing_or_cross_group_motion_support_fails_closed() -> None:
    records = _records()
    with pytest.raises(ValueError, match="duplicate transfer cell"):
        _analyze(records + [deepcopy(records[0])])
    with pytest.raises(ValueError, match="exactly cover"):
        _analyze(records[:-1])

    changed_group = deepcopy(records)
    changed_group[0]["target_source_group_id"] = "group_4"
    with pytest.raises(ValueError, match="maps to multiple source groups"):
        _analyze(changed_group)


def test_headroom_and_features_are_invariant_across_panel_or_seed() -> None:
    varying_headroom = _records()
    record = next(
        item
        for item in varying_headroom
        if item["source_panel_id"] == "panel_1"
        and item["target_motion_id"] == "motion_00"
        and item["paired_seed"] == 0
    )
    record["independent_headroom"] = 0.6
    with pytest.raises(ValueError, match="invariant across source panels"):
        _analyze(varying_headroom)

    drifting_features = _records()
    record = next(
        item
        for item in drifting_features
        if item["source_panel_id"] == "panel_0"
        and item["target_motion_id"] == "motion_00"
        and item["paired_seed"] == 1
    )
    record["failure_features"] = [999.0]
    with pytest.raises(ValueError, match="frozen across paired seeds"):
        _analyze(drifting_features)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baseline_feature_names": ["only_one"]}, "dimension"),
        ({"failure_feature_names": ["x", "x"]}, "unique"),
        ({"baseline_feature_provenance_sha256": "bad"}, "SHA-256"),
        ({"headroom_evaluation_artifact_sha256": _GAIN_HASH}, "must be distinct"),
        ({"headroom_rollout_seeds": [101, 200]}, "must be disjoint"),
        ({"gain_rollout_seeds": [101, 100]}, "canonical order"),
        ({"h_min": 0.0}, "finite and positive"),
    ],
)
def test_named_schema_and_independent_evaluation_contracts_fail_closed(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _analyze(**overrides)


def test_hierarchical_bootstrap_resamples_all_three_cluster_axes() -> None:
    analysis = _analyze()
    first = bootstrap_clustered_loss_improvement(
        analysis,
        seed=17,
        resamples=500,
    )
    second = bootstrap_clustered_loss_improvement(
        analysis,
        seed=17,
        resamples=500,
    )
    changed = bootstrap_clustered_loss_improvement(
        analysis,
        seed=18,
        resamples=500,
    )

    assert first == second
    assert first["bootstrap_distribution_sha256"] != changed["bootstrap_distribution_sha256"]
    assert first["source_panel_resampling"] is True
    assert first["global_paired_seed_resampling"] is True
    assert first["target_source_group_resampling"] is True
    assert first["motion_cell_resampling"] is False
    assert first["source_panel_count"] == 4
    assert first["paired_seed_count"] == 2
    assert first["target_source_group_count"] == 5
    assert first["bootstrap_sha256"] == canonical_sha256(
        first,
        digest_field="bootstrap_sha256",
    )


def test_bootstrap_can_target_supported_normalized_sensitivity() -> None:
    analysis = _analyze()
    result = bootstrap_clustered_loss_improvement(
        analysis,
        seed=5,
        resamples=20,
        outcome="normalized_headroom_sensitivity",
    )
    assert result["outcome"] == "normalized_headroom_sensitivity"


def test_exact_whole_source_row_permutation_refits_nested_augmented_model() -> None:
    analysis = _analyze()
    result = permute_whole_source_failure_rows(
        analysis,
        method="exact",
    )

    assert result["method"] == "exact"
    assert result["evaluated_permutation_count"] == math.factorial(4)
    assert result["identity_included"] is True
    assert result["identity_position"] == 0
    assert result["monte_carlo_sampling_frame"] is None
    assert result["target_motion_blocks_preserved"] is True
    assert result["paired_seed_blocks_preserved"] is True
    assert result["identity_statistic"] == pytest.approx(
        analysis["primary_raw"]["metrics"]["paired_loss_improvement"],
        abs=1e-10,
    )
    assert result["p_value_one_sided"] >= 1.0 / math.factorial(4)
    assert result["permutation_sha256"] == canonical_sha256(
        result,
        digest_field="permutation_sha256",
    )


def test_monte_carlo_permutation_is_seeded_includes_identity_and_uses_plus_one() -> None:
    analysis = _analyze()
    first = permute_whole_source_failure_rows(
        analysis,
        method="monte_carlo",
        seed=9,
        monte_carlo_permutations=7,
    )
    second = permute_whole_source_failure_rows(
        analysis,
        method="monte_carlo",
        seed=9,
        monte_carlo_permutations=7,
    )

    assert first == second
    assert first["evaluated_permutation_count"] == 8
    assert first["monte_carlo_random_draw_count"] == 7
    assert (
        first["monte_carlo_sampling_frame"]
        == "uniform_with_replacement_over_nonidentity_permutations"
    )
    assert first["p_value_rule"] == "plus_one_one_sided_monte_carlo"
    assert first["minimum_achievable_p"] == 1.0 / 8.0
    assert first["p_value_one_sided"] * 8 == pytest.approx(round(first["p_value_one_sided"] * 8))


def test_inference_primitives_reject_tampered_or_nonestimable_artifacts() -> None:
    analysis = _analyze()
    tampered = deepcopy(analysis)
    tampered["primary_raw"]["metrics"]["paired_loss_improvement"] += 1.0
    with pytest.raises(ValueError, match="analysis_sha256 mismatch"):
        bootstrap_clustered_loss_improvement(tampered, seed=1, resamples=10)
    with pytest.raises(ValueError, match="analysis_sha256 mismatch"):
        permute_whole_source_failure_rows(tampered, method="exact")

    records = _records()
    for record in records:
        if record["target_source_group_id"] in {"group_0", "group_1"}:
            record["independent_headroom"] = 0.1
    nonestimable = _analyze(records)
    with pytest.raises(ValueError, match="not estimable"):
        bootstrap_clustered_loss_improvement(
            nonestimable,
            seed=1,
            resamples=10,
            outcome="normalized_headroom_sensitivity",
        )


def test_degenerate_features_and_alpha_grid_fail_closed() -> None:
    records = _records()
    for record in records:
        record["failure_features"] = [1.0]
    with pytest.raises(ValueError, match="degenerate.*feature columns"):
        _analyze(records)

    with pytest.raises(ValueError, match="alpha_grid"):
        _analyze(alpha_grid=[1.0, 0.1])
