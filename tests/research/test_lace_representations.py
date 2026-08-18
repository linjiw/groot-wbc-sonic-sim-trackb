from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.research.lace.representations import (
    ASSIGNMENT_KIND,
    CLUSTERING_METHOD,
    PRIMARY_K,
    RANDOM_ASSIGNMENT_KIND,
    REPRESENTATION_KIND,
    SCALE512_CANDIDATE_REPRESENTATIONS,
    SCALE512_SPLIT_SELECTION_SHA256,
    SCALER_METHOD,
    assign_frozen_representation,
    fit_frozen_representation,
    random_matched_size_assignment,
    resolve_random_control_seeds,
    resolve_representation_fit_kwargs,
    validate_frozen_representation,
    validate_representation_protocol,
)
from gear_sonic.research.lace.schema import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs/research/lace/representation_protocol_scale512_v1.json"
PROTOCOL_SHA256 = "8c42d6a26dfdf7751b58c0cad0e008ff4b9921185ec8d0eb6e96fd5a137f2adb"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def _atlas_rows(k: int = 4) -> dict:
    matrix: list[list[float]] = []
    motions: list[str] = []
    groups: list[str] = []
    for cluster in range(k):
        angle = 2.0 * np.pi * cluster / k
        center = np.asarray([10.0 * np.cos(angle), 10.0 * np.sin(angle)])
        for offset in range(4):
            matrix.append([*(center + [0.02 * offset, -0.01 * offset]), 7.0])
            motions.append(f"motion_{cluster:02d}_{offset:02d}")
            groups.append(f"source_{cluster:02d}_{offset // 2:02d}")
    return {
        "feature_matrix": np.asarray(matrix),
        "feature_names": ["velocity", "contact_schedule", "constant_duration"],
        "motion_keys": motions,
        "source_group_ids": groups,
        "partitions": ["D_atlas"] * len(motions),
    }


def _fit(k: int = 4, **overrides) -> dict:
    kwargs = {
        **_atlas_rows(k),
        "representation_id": f"kinematics_k{k}",
        "k": k,
        "seed": 1701,
        "minimum_cell_size": 3,
        "minimum_source_groups_per_cell": 2,
        "provenance": {"split_sha256": "a" * 64, "extractor": "reference_only_v1"},
        "n_init": 12,
        "max_iterations": 100,
    }
    kwargs.update(overrides)
    return fit_frozen_representation(**kwargs)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_fit_freezes_identical_method_at_supported_k_with_exact_hashes(k: int) -> None:
    artifact = _fit(k)

    assert artifact["kind"] == REPRESENTATION_KIND
    assert artifact["fit_partition"] == "D_atlas"
    assert artifact["outcome_blind"] is True
    assert artifact["k"] == k
    assert artifact["primary_k"] == PRIMARY_K == 6
    assert artifact["is_primary_k"] is (k == 6)
    assert artifact["scaler"]["method"] == SCALER_METHOD
    assert artifact["scaler"]["constant_feature_indices"] == [2]
    assert artifact["scaler"]["scale"][2] == 1.0
    assert artifact["clustering"]["method"] == CLUSTERING_METHOD
    assert len(artifact["centroids_standardized"]) == k
    assert all(cell["motion_count"] >= 3 for cell in artifact["fit_support"])
    assert all(cell["source_group_count"] >= 2 for cell in artifact["fit_support"])
    assert artifact["provenance_sha256"] == canonical_sha256(artifact["provenance"])
    assert artifact["representation_sha256"] == canonical_sha256(
        artifact,
        digest_field="representation_sha256",
    )
    validate_frozen_representation(artifact)


def test_fit_is_deterministic_and_canonical_under_input_row_permutation() -> None:
    rows = _atlas_rows()
    baseline = _fit()
    order = np.asarray([7, 12, 0, 15, 3, 6, 10, 14, 1, 11, 4, 13, 8, 2, 9, 5])
    permuted = _fit(
        feature_matrix=rows["feature_matrix"][order],
        motion_keys=[rows["motion_keys"][index] for index in order],
        source_group_ids=[rows["source_group_ids"][index] for index in order],
        partitions=[rows["partitions"][index] for index in order],
    )

    assert permuted == baseline
    assert [row["motion_key"] for row in baseline["fit_assignments"]] == sorted(rows["motion_keys"])


def test_candidate_name_does_not_change_the_common_clustering_pipeline() -> None:
    failure = _fit(representation_id="failure")
    semantics = _fit(representation_id="semantics")

    assert failure["scaler"] == semantics["scaler"]
    assert failure["clustering"] == semantics["clustering"]
    assert failure["centroids_standardized"] == semantics["centroids_standardized"]
    assert failure["fit_assignments"] == semantics["fit_assignments"]


def test_assignment_applies_frozen_parameters_without_mutation_or_refit() -> None:
    artifact = _fit()
    before = deepcopy(artifact)
    geometry = np.asarray(
        [
            [10.01, -0.01, 7.0],
            [0.01, 10.01, 7.0],
            [-10.01, 0.01, 7.0],
            [-0.01, -10.01, 7.0],
        ]
    )
    result = assign_frozen_representation(
        artifact,
        geometry,
        feature_names=["velocity", "contact_schedule", "constant_duration"],
        motion_keys=["geometry_d", "geometry_c", "geometry_b", "geometry_a"],
        source_group_ids=["held_d", "held_c", "held_b", "held_a"],
        partitions=["D_geometry"] * 4,
        provenance={"feature_manifest_sha256": "b" * 64},
    )

    assert artifact == before
    assert result["kind"] == ASSIGNMENT_KIND
    assert result["fit_or_refit_performed"] is False
    assert result["representation_sha256"] == artifact["representation_sha256"]
    assert [row["motion_key"] for row in result["assignments"]] == [
        "geometry_a",
        "geometry_b",
        "geometry_c",
        "geometry_d",
    ]
    assert all(sum(row["membership"]) == 1.0 for row in result["assignments"])
    assert result["assignment_sha256"] == canonical_sha256(
        result,
        digest_field="assignment_sha256",
    )


def test_assignment_uses_lowest_canonical_index_on_an_exact_distance_tie() -> None:
    artifact = _fit()
    artifact["centroids_standardized"] = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ]
    artifact["representation_sha256"] = canonical_sha256(
        artifact,
        digest_field="representation_sha256",
    )
    centroids = np.asarray(artifact["centroids_standardized"])
    standardized = (centroids[0] + centroids[1]) / 2.0
    raw = standardized * np.asarray(artifact["scaler"]["scale"]) + np.asarray(
        artifact["scaler"]["mean"]
    )
    expected = int(np.argmin(np.square(centroids - standardized).sum(axis=1)))

    result = assign_frozen_representation(
        artifact,
        [raw],
        feature_names=artifact["feature_names"],
        motion_keys=["tie_motion"],
        source_group_ids=["tie_source"],
        partitions=["D_geometry"],
        provenance={"purpose": "tie_test"},
    )
    assert expected == 0
    assert result["assignments"][0]["cluster_index"] == expected


def test_assignment_input_hash_binds_motion_and_source_group_identity() -> None:
    artifact = _fit()

    def assign(source_group_id: str) -> dict:
        return assign_frozen_representation(
            artifact,
            [[10.0, 0.0, 7.0]],
            feature_names=artifact["feature_names"],
            motion_keys=["same_motion"],
            source_group_ids=[source_group_id],
            partitions=["D_geometry"],
            provenance={"extractor": "same"},
        )

    left = assign("source_left")
    right = assign("source_right")
    assert left["input_sha256"] != right["input_sha256"]
    assert left["assignment_sha256"] != right["assignment_sha256"]


@pytest.mark.parametrize("partition", ["D_curriculum", "D_controller", "D_atlas"])
def test_assignment_allows_declared_non_test_partitions(partition: str) -> None:
    artifact = _fit()
    result = assign_frozen_representation(
        artifact,
        [[0.0, 10.0, 7.0]],
        feature_names=artifact["feature_names"],
        motion_keys=[f"new_{partition}"],
        source_group_ids=["new_source"],
        partitions=[partition],
        provenance={"partition": partition},
    )
    assert result["assignments"][0]["partition"] == partition


def test_fit_rejects_non_atlas_rows_instead_of_silently_selecting_them() -> None:
    rows = _atlas_rows()
    rows["partitions"][-1] = "D_geometry"
    with pytest.raises(ValueError, match="partitions must be drawn from.*D_atlas"):
        _fit(**rows)


def test_assignment_rejects_test_partition_and_ordered_schema_drift() -> None:
    artifact = _fit()
    kwargs = {
        "artifact": artifact,
        "feature_matrix": [[1.0, 2.0, 7.0]],
        "feature_names": artifact["feature_names"],
        "motion_keys": ["held_out"],
        "source_group_ids": ["held_group"],
        "partitions": ["D_test"],
        "provenance": {"extractor": "v1"},
    }
    with pytest.raises(ValueError, match="partitions must be drawn from"):
        assign_frozen_representation(**kwargs)

    kwargs["partitions"] = ["D_geometry"]
    kwargs["feature_names"] = [
        "contact_schedule",
        "velocity",
        "constant_duration",
    ]
    with pytest.raises(ValueError, match="exactly match the frozen ordered feature schema"):
        assign_frozen_representation(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows["feature_matrix"].__setitem__((0, 0), np.nan), "finite"),
        (lambda rows: rows["motion_keys"].__setitem__(1, rows["motion_keys"][0]), "unique"),
        (lambda rows: rows["source_group_ids"].__setitem__(1, ""), "non-empty"),
        (lambda rows: rows["feature_names"].reverse(), None),
    ],
)
def test_numeric_identity_and_order_are_bound(mutation, message: str | None) -> None:
    rows = _atlas_rows()
    baseline = _fit()
    mutation(rows)
    if message is not None:
        with pytest.raises(ValueError, match=message):
            _fit(**rows)
    else:
        changed = _fit(**rows)
        assert changed["input_sha256"] != baseline["input_sha256"]
        assert changed["feature_names"] != baseline["feature_names"]


def test_support_constraints_are_explicit_and_fail_closed() -> None:
    with pytest.raises(ValueError, match="too few rows"):
        _fit(minimum_cell_size=5)

    rows = _atlas_rows()
    rows["source_group_ids"] = ["single_source"] * len(rows["motion_keys"])
    with pytest.raises(ValueError, match="too few source groups"):
        _fit(**rows)

    with pytest.raises(ValueError, match="converged and satisfied"):
        _fit(max_iterations=1)


def test_tampered_artifact_is_rejected_before_assignment() -> None:
    artifact = _fit()
    artifact["centroids_standardized"][0][0] += 1.0
    with pytest.raises(ValueError, match="representation_sha256 mismatch"):
        validate_frozen_representation(artifact)

    artifact = _fit()
    artifact["clustering"]["method"] = "outcome_selected_method"
    artifact["representation_sha256"] = canonical_sha256(
        artifact,
        digest_field="representation_sha256",
    )
    with pytest.raises(ValueError, match="clustering method mismatch"):
        validate_frozen_representation(artifact)


def test_random_control_preserves_exact_observed_cell_sizes_and_is_hashed() -> None:
    observed = {
        "m7": 3,
        "m6": 3,
        "m5": 2,
        "m0": 0,
        "m3": 1,
        "m2": 0,
        "m4": 1,
        "m1": 0,
    }
    groups = {motion: f"source_{motion}" for motion in observed}
    result = random_matched_size_assignment(
        observed,
        source_group_ids=groups,
        seed=81,
        control_id="random_control_81",
    )

    assert result["kind"] == RANDOM_ASSIGNMENT_KIND
    assert result["unit"] == "motion"
    assert result["source_groups_are_indivisible"] is False
    assert result["observed_cell_sizes"] == [3, 2, 1, 2]
    assert result["randomized_cell_sizes"] == [3, 2, 1, 2]
    assert result == random_matched_size_assignment(
        observed,
        source_group_ids=groups,
        seed=81,
        control_id="random_control_81",
    )
    assert result["random_assignment_sha256"] == canonical_sha256(
        result,
        digest_field="random_assignment_sha256",
    )
    assert all(sum(record["membership"]) == 1.0 for record in result["assignments"])


def test_random_control_rejects_noncontiguous_labels_or_missing_identity() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        random_matched_size_assignment(
            {"a": 0, "b": 2},
            source_group_ids={"a": "g0", "b": "g1"},
            seed=1,
            control_id="bad",
        )
    with pytest.raises(ValueError, match="keys must exactly match"):
        random_matched_size_assignment(
            {"a": 0, "b": 1},
            source_group_ids={"a": "g0"},
            seed=1,
            control_id="bad",
        )
    with pytest.raises(ValueError, match="must use K"):
        random_matched_size_assignment(
            {"a": 0, "b": 1, "c": 2},
            source_group_ids={"a": "g0", "b": "g1", "c": "g2"},
            seed=1,
            control_id="unsupported_k",
        )


def test_tracked_scale512_protocol_is_self_hashed_and_exact() -> None:
    protocol = _protocol()
    validate_representation_protocol(protocol)

    assert protocol["protocol_sha256"] == PROTOCOL_SHA256
    assert protocol["protocol_sha256"] == canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    assert protocol["split"] == {
        "selection_sha256": SCALE512_SPLIT_SELECTION_SHA256,
        "fit_partition": "D_atlas",
        "motion_count": 107,
        "source_group_count": 55,
    }
    assert protocol["candidate_representations"] == list(SCALE512_CANDIDATE_REPRESENTATIONS)
    assert protocol["clustering"]["supported_k"] == [4, 6, 8]
    assert protocol["clustering"]["primary_k"] == 6
    assert protocol["clustering"]["shared_seed"] == 8132026
    assert protocol["clustering"]["n_init"] == 32
    assert protocol["clustering"]["max_iterations"] == 300
    assert protocol["clustering"]["convergence_tolerance"] == 1e-10
    assert protocol["clustering"]["minimum_support_by_k"] == {
        "4": {"minimum_cell_size": 12, "minimum_source_groups_per_cell": 8},
        "6": {"minimum_cell_size": 8, "minimum_source_groups_per_cell": 5},
        "8": {"minimum_cell_size": 5, "minimum_source_groups_per_cell": 4},
    }
    controls = protocol["random_matched_size_controls"]
    assert controls["unit"] == "motion"
    assert controls["control_count"] == 100
    assert controls["seed_base"] == 8133000
    assert controls["preserve_observed_cell_sizes_exactly"] is True
    assert controls["retain_source_group_ids_for_blocked_inference"] is True
    assert controls["selection_from_transfer_outcomes_permitted"] is False
    assert (
        controls["whole_source_group_sensitivity"]["role"]
        == "separate_sensitivity_not_primary_matched_size_null"
    )
    assert (
        controls["whole_source_group_sensitivity"]["exact_cell_size_preservation_guaranteed"]
        is False
    )


@pytest.mark.parametrize(
    ("k", "minimum_cell_size", "minimum_source_groups"),
    [(4, 12, 8), (6, 8, 5), (8, 5, 4)],
)
def test_protocol_resolver_produces_outcome_blind_fit_kwargs(
    k: int,
    minimum_cell_size: int,
    minimum_source_groups: int,
) -> None:
    kwargs = resolve_representation_fit_kwargs(
        _protocol(),
        k=k,
        observed_split_selection_sha256=SCALE512_SPLIT_SELECTION_SHA256,
        observed_d_atlas_motion_count=107,
        observed_d_atlas_source_group_count=55,
    )
    assert kwargs == {
        "k": k,
        "seed": 8132026,
        "minimum_cell_size": minimum_cell_size,
        "minimum_source_groups_per_cell": minimum_source_groups,
        "n_init": 32,
        "max_iterations": 300,
        "convergence_tolerance": 1e-10,
    }
    assert set(kwargs).issubset(inspect.signature(fit_frozen_representation).parameters)
    assert all(
        "outcome" not in name and "transfer" not in name
        for name in inspect.signature(resolve_representation_fit_kwargs).parameters
    )


def test_protocol_resolves_all_one_hundred_random_control_seeds() -> None:
    seeds = resolve_random_control_seeds(_protocol())
    assert len(seeds) == 100
    assert seeds == tuple(range(8133000, 8133100))
    assert len(set(seeds)) == 100


def _redigest_protocol(protocol: dict) -> None:
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda protocol: protocol["split"].update(selection_sha256="b" * 64),
            "selection_sha256 drift",
        ),
        (
            lambda protocol: protocol["split"].update(motion_count=108),
            "motion_count drift",
        ),
        (
            lambda protocol: protocol["split"].update(source_group_count=54),
            "source_group_count drift",
        ),
        (
            lambda protocol: protocol["clustering"].update(supported_k=[4, 5, 8]),
            "supported_k drift",
        ),
        (
            lambda protocol: protocol["clustering"].update(method="sklearn_default"),
            "clustering method drift",
        ),
        (
            lambda protocol: protocol["clustering"]["minimum_support_by_k"]["6"].update(
                minimum_cell_size=7
            ),
            "minimum cell size drift for K=6",
        ),
        (
            lambda protocol: protocol["random_matched_size_controls"].update(control_count=99),
            "control_count drift",
        ),
        (
            lambda protocol: protocol["random_matched_size_controls"].update(unit="source_group"),
            "unit drift",
        ),
    ],
)
def test_protocol_rejects_semantic_drift_even_if_attacker_recomputes_digest(
    mutate, message: str
) -> None:
    protocol = _protocol()
    mutate(protocol)
    _redigest_protocol(protocol)
    with pytest.raises(ValueError, match=message):
        validate_representation_protocol(protocol)


def test_protocol_rejects_digest_tampering() -> None:
    protocol = _protocol()
    protocol["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol_sha256 mismatch"):
        validate_representation_protocol(protocol)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"observed_split_selection_sha256": "b" * 64},
            "observed split selection",
        ),
        ({"observed_d_atlas_motion_count": 106}, "observed D_atlas motion count"),
        (
            {"observed_d_atlas_source_group_count": 54},
            "observed D_atlas source-group count",
        ),
        ({"k": 5}, "k must be one of"),
    ],
)
def test_fit_resolver_rejects_observed_split_or_k_drift(overrides, message: str) -> None:
    kwargs = {
        "protocol": _protocol(),
        "k": 6,
        "observed_split_selection_sha256": SCALE512_SPLIT_SELECTION_SHA256,
        "observed_d_atlas_motion_count": 107,
        "observed_d_atlas_source_group_count": 55,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        resolve_representation_fit_kwargs(**kwargs)
