"""Frozen, outcome-blind representation baselines for LACE RQ1.

This module accepts numeric feature matrices; it has no API for policy-transfer
outcomes and never chooses ``K`` or support thresholds from downstream results.
Every candidate representation therefore uses the same deterministic pipeline:

1. sort and identity-bind the explicitly supplied ``D_atlas`` rows;
2. fit a population-standardization transform on those rows only;
3. run deterministic multi-start Lloyd k-means; and
4. freeze the scaler and centroids in a self-hashed artifact.

Centroid assignment uses ``numpy.argmin``, so an exact distance tie goes to the
lowest canonical cluster index.  Cluster indices are canonicalized by centroid
coordinates and then by their lexicographically first member.  A fit selects the
lowest-inertia *support-feasible* start, with canonical centroids and assignments
as deterministic secondary keys.  Assignment only applies the stored transform
and centroids: it never repairs, refits, or mutates them.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from gear_sonic.research.lace.schema import canonical_sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

REPRESENTATION_KIND = "lace_frozen_representation"
REPRESENTATION_SCHEMA_VERSION = 1
ASSIGNMENT_KIND = "lace_frozen_representation_assignment"
ASSIGNMENT_SCHEMA_VERSION = 1
RANDOM_ASSIGNMENT_KIND = "lace_random_matched_size_assignment"
RANDOM_ASSIGNMENT_SCHEMA_VERSION = 1
REPRESENTATION_PROTOCOL_KIND = "lace_representation_protocol_scale512"
REPRESENTATION_PROTOCOL_SCHEMA_VERSION = 1
SUPPORTED_K = (4, 6, 8)
PRIMARY_K = 6
FIT_PARTITION = "D_atlas"
ALLOWED_ASSIGNMENT_PARTITIONS = (
    "D_atlas",
    "D_curriculum",
    "D_geometry",
    "D_controller",
)
CLUSTERING_METHOD = "standardized_deterministic_multistart_lloyd_kmeans_v1"
SCALER_METHOD = "d_atlas_population_mean_std_zero_variance_scale_one_v1"
SCALE512_SPLIT_SELECTION_SHA256 = "4a9d530513b15557a556a8a790e5dab49c11b796d84aa0384c3db0a283e555e7"
SCALE512_D_ATLAS_MOTION_COUNT = 107
SCALE512_D_ATLAS_SOURCE_GROUP_COUNT = 55
SCALE512_SHARED_SEED = 8132026
SCALE512_SUPPORT_BY_K = {4: (12, 8), 6: (8, 5), 8: (5, 4)}
SCALE512_RANDOM_CONTROL_COUNT = 100
SCALE512_RANDOM_SEED_BASE = 8133000
SCALE512_CANDIDATE_REPRESENTATIONS = (
    "failure_mechanism",
    "scalar_difficulty",
    "reference_kinematics",
    "motion_semantics",
    "kinematics_plus_difficulty",
    "reference_feasibility",
    "feasibility_plus_kinematics_plus_difficulty",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identifier(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    _require(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
        f"{name} must be an integer",
    )
    result = int(value)
    _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def _real(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
        f"{name} must be a real scalar",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if positive:
        _require(result > 0.0, f"{name} must be positive")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _identifier(value, name)
    _require(len(result) == 64 and result == result.lower(), f"{name} must be a SHA-256")
    try:
        int(result, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256") from error
    return result


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    _require(set(mapping) == expected, f"{name} keys mismatch")


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{name} must be a sequence",
    )
    result = tuple(values)
    _require(bool(result), f"{name} must be non-empty")
    _require(
        all(isinstance(value, str) and bool(value) for value in result),
        f"{name} must contain non-empty strings",
    )
    _require(len(result) == len(set(result)), f"{name} must be unique")
    return result


def _matrix(values: Any, *, rows: int, columns: int, name: str) -> FloatArray:
    raw = np.asarray(values)
    _require(
        raw.ndim == 2 and raw.shape == (rows, columns),
        f"{name} must have shape [{rows}, {columns}]",
    )
    _require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must contain real numeric values",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(np.all(np.isfinite(result)), f"{name} must contain only finite values")
    return result


def _validate_rows(
    feature_matrix: Any,
    *,
    feature_names: Sequence[str],
    motion_keys: Sequence[str],
    source_group_ids: Sequence[str],
    partitions: Sequence[str],
    allowed_partitions: Sequence[str],
) -> tuple[FloatArray, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    features = _names(feature_names, "feature_names")
    motions = _names(motion_keys, "motion_keys")
    _require(
        isinstance(source_group_ids, Sequence) and not isinstance(source_group_ids, (str, bytes)),
        "source_group_ids must be a sequence",
    )
    groups = tuple(source_group_ids)
    _require(len(groups) == len(motions), "source_group_ids length must match motion_keys")
    _require(
        all(isinstance(value, str) and bool(value) for value in groups),
        "source_group_ids must contain non-empty strings",
    )
    _require(
        isinstance(partitions, Sequence) and not isinstance(partitions, (str, bytes)),
        "partitions must be a sequence",
    )
    resolved_partitions = tuple(partitions)
    _require(len(resolved_partitions) == len(motions), "partitions length must match motion_keys")
    allowed = set(allowed_partitions)
    _require(
        all(value in allowed for value in resolved_partitions),
        f"partitions must be drawn from {tuple(allowed_partitions)!r}",
    )
    matrix = _matrix(
        feature_matrix,
        rows=len(motions),
        columns=len(features),
        name="feature_matrix",
    )

    order = np.asarray(
        sorted(range(len(motions)), key=lambda index: motions[index]), dtype=np.int64
    )
    return (
        matrix[order],
        features,
        tuple(motions[index] for index in order),
        tuple(groups[index] for index in order),
        tuple(resolved_partitions[index] for index in order),
    )


def _input_payload(
    matrix: FloatArray,
    *,
    feature_names: tuple[str, ...],
    motion_keys: tuple[str, ...],
    source_group_ids: tuple[str, ...],
    partitions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "feature_names": list(feature_names),
        "rows": [
            {
                "motion_key": motion_key,
                "source_group_id": source_group_id,
                "partition": partition,
                "features": [float(value) for value in row],
            }
            for motion_key, source_group_id, partition, row in zip(
                motion_keys,
                source_group_ids,
                partitions,
                matrix,
                strict=True,
            )
        ],
    }


def _standardize_fit(matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, list[int]]:
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, ddof=0, dtype=np.float64)
    constant_columns = np.flatnonzero(scale == 0.0).astype(int).tolist()
    scale = scale.copy()
    scale[scale == 0.0] = 1.0
    return (matrix - mean) / scale, mean, scale, constant_columns


def _initial_centroids(
    matrix: FloatArray,
    *,
    motion_keys: tuple[str, ...],
    k: int,
    seed: int,
    start: int,
) -> FloatArray:
    # A seeded canonical hash chooses the first point. Subsequent points use
    # farthest-first traversal; exact ties resolve by motion key.
    first = min(
        range(matrix.shape[0]),
        key=lambda index: canonical_sha256(
            {"seed": seed, "start": start, "motion_key": motion_keys[index]}
        ),
    )
    chosen = [first]
    minimum_distances = np.square(matrix - matrix[first]).sum(axis=1)
    for _ in range(1, k):
        candidates = [index for index in range(matrix.shape[0]) if index not in chosen]
        next_index = min(
            candidates,
            key=lambda index: (-float(minimum_distances[index]), motion_keys[index]),
        )
        chosen.append(next_index)
        distances = np.square(matrix - matrix[next_index]).sum(axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    return matrix[np.asarray(chosen, dtype=np.int64)].copy()


def _repair_empty_clusters(
    labels: IntArray,
    distances: FloatArray,
    *,
    motion_keys: tuple[str, ...],
    k: int,
) -> IntArray:
    result = labels.copy()
    counts = np.bincount(result, minlength=k)
    for empty_cluster in np.flatnonzero(counts == 0):
        candidates = [index for index, label in enumerate(result) if counts[label] > 1]
        _require(bool(candidates), "cannot repair an empty cluster without a donor point")
        donor = min(
            candidates,
            key=lambda index: (-float(distances[index, result[index]]), motion_keys[index]),
        )
        old_cluster = int(result[donor])
        result[donor] = int(empty_cluster)
        counts[old_cluster] -= 1
        counts[empty_cluster] += 1
    return result


def _canonicalize(
    centroids: FloatArray,
    labels: IntArray,
    motion_keys: tuple[str, ...],
) -> tuple[FloatArray, IntArray]:
    members = [
        tuple(motion_keys[index] for index in np.flatnonzero(labels == cluster))
        for cluster in range(len(centroids))
    ]
    order = sorted(
        range(len(centroids)),
        key=lambda cluster: (tuple(float(value) for value in centroids[cluster]), members[cluster]),
    )
    old_to_new = np.empty(len(order), dtype=np.int64)
    for new, old in enumerate(order):
        old_to_new[old] = new
    return centroids[np.asarray(order, dtype=np.int64)], old_to_new[labels]


def _lloyd(
    matrix: FloatArray,
    *,
    motion_keys: tuple[str, ...],
    initial_centroids: FloatArray,
    max_iterations: int,
    tolerance: float,
) -> tuple[FloatArray, IntArray, float, int, bool]:
    k = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    prior_labels: IntArray | None = None
    did_converge = False
    for iteration in range(1, max_iterations + 1):
        distances = np.square(matrix[:, None, :] - centroids[None, :, :]).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        labels = _repair_empty_clusters(labels, distances, motion_keys=motion_keys, k=k)
        updated = np.stack([matrix[labels == cluster].mean(axis=0) for cluster in range(k)])
        shift = float(np.max(np.linalg.norm(updated - centroids, axis=1)))
        converged = prior_labels is not None and np.array_equal(labels, prior_labels)
        centroids = updated
        prior_labels = labels
        if converged or shift <= tolerance:
            did_converge = True
            break
    else:
        iteration = max_iterations

    distances = np.square(matrix[:, None, :] - centroids[None, :, :]).sum(axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    labels = _repair_empty_clusters(labels, distances, motion_keys=motion_keys, k=k)
    centroids = np.stack([matrix[labels == cluster].mean(axis=0) for cluster in range(k)])
    centroids, labels = _canonicalize(centroids, labels, motion_keys)
    inertia = float(np.square(matrix - centroids[labels]).sum(dtype=np.float64))
    return centroids, labels, inertia, iteration, did_converge


def _support(
    labels: IntArray,
    source_group_ids: tuple[str, ...],
    k: int,
) -> list[dict[str, int]]:
    return [
        {
            "cluster_index": cluster,
            "motion_count": int(np.count_nonzero(labels == cluster)),
            "source_group_count": len(
                {source_group_ids[index] for index in np.flatnonzero(labels == cluster)}
            ),
        }
        for cluster in range(k)
    ]


def _support_is_feasible(
    support: Sequence[Mapping[str, int]],
    *,
    minimum_cell_size: int,
    minimum_source_groups_per_cell: int,
) -> bool:
    return all(
        cell["motion_count"] >= minimum_cell_size
        and cell["source_group_count"] >= minimum_source_groups_per_cell
        for cell in support
    )


def fit_frozen_representation(
    feature_matrix: Any,
    *,
    feature_names: Sequence[str],
    motion_keys: Sequence[str],
    source_group_ids: Sequence[str],
    partitions: Sequence[str],
    representation_id: str,
    k: int,
    seed: int,
    minimum_cell_size: int,
    minimum_source_groups_per_cell: int,
    provenance: Mapping[str, Any],
    n_init: int = 32,
    max_iterations: int = 300,
    convergence_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit and freeze a matched-``K`` representation using only ``D_atlas``.

    The support minima are explicit inputs because their scientific values are
    a preregistration decision, not something this implementation may infer.
    Starts that violate them are recorded and excluded before the deterministic
    minimum-inertia comparison.
    """

    resolved_id = _identifier(representation_id, "representation_id")
    resolved_k = _integer(k, "k", minimum=2)
    _require(resolved_k in SUPPORTED_K, f"k must be one of {SUPPORTED_K}")
    resolved_seed = _integer(seed, "seed", minimum=0)
    min_cell = _integer(minimum_cell_size, "minimum_cell_size", minimum=1)
    min_groups = _integer(
        minimum_source_groups_per_cell,
        "minimum_source_groups_per_cell",
        minimum=1,
    )
    starts = _integer(n_init, "n_init", minimum=1)
    iterations = _integer(max_iterations, "max_iterations", minimum=1)
    tolerance = _real(convergence_tolerance, "convergence_tolerance", positive=True)
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    provenance_payload = dict(provenance)
    provenance_digest = canonical_sha256(provenance_payload)

    matrix, features, motions, groups, resolved_partitions = _validate_rows(
        feature_matrix,
        feature_names=feature_names,
        motion_keys=motion_keys,
        source_group_ids=source_group_ids,
        partitions=partitions,
        allowed_partitions=(FIT_PARTITION,),
    )
    _require(
        matrix.shape[0] >= resolved_k * min_cell,
        "D_atlas has too few rows for k and minimum_cell_size",
    )
    _require(
        len(set(groups)) >= min_groups,
        "D_atlas has too few source groups for minimum_source_groups_per_cell",
    )
    standardized, mean, scale, constant_columns = _standardize_fit(matrix)

    candidates: list[
        tuple[tuple[Any, ...], FloatArray, IntArray, float, int, list[dict[str, int]], int]
    ] = []
    rejected_starts: list[dict[str, Any]] = []
    for start in range(starts):
        initial = _initial_centroids(
            standardized,
            motion_keys=motions,
            k=resolved_k,
            seed=resolved_seed,
            start=start,
        )
        centroids, labels, inertia, iteration_count, did_converge = _lloyd(
            standardized,
            motion_keys=motions,
            initial_centroids=initial,
            max_iterations=iterations,
            tolerance=tolerance,
        )
        if not did_converge:
            rejected_starts.append(
                {
                    "start_index": start,
                    "reason": "did_not_converge",
                    "iterations": iteration_count,
                }
            )
            continue
        support = _support(labels, groups, resolved_k)
        if not _support_is_feasible(
            support,
            minimum_cell_size=min_cell,
            minimum_source_groups_per_cell=min_groups,
        ):
            rejected_starts.append(
                {"start_index": start, "reason": "minimum_support", "support": support}
            )
            continue
        key = (
            inertia,
            tuple(float(value) for value in centroids.ravel()),
            tuple(int(value) for value in labels),
            start,
        )
        candidates.append((key, centroids, labels, inertia, iteration_count, support, start))
    _require(
        bool(candidates),
        "no deterministic k-means start converged and satisfied the frozen minimum support constraints",
    )
    _, centroids, labels, inertia, iteration_count, support, selected_start = min(
        candidates, key=lambda candidate: candidate[0]
    )

    input_payload = _input_payload(
        matrix,
        feature_names=features,
        motion_keys=motions,
        source_group_ids=groups,
        partitions=resolved_partitions,
    )
    artifact: dict[str, Any] = {
        "kind": REPRESENTATION_KIND,
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "representation_id": resolved_id,
        "outcome_blind": True,
        "fit_partition": FIT_PARTITION,
        "allowed_assignment_partitions": list(ALLOWED_ASSIGNMENT_PARTITIONS),
        "k": resolved_k,
        "primary_k": PRIMARY_K,
        "is_primary_k": resolved_k == PRIMARY_K,
        "feature_names": list(features),
        "input_sha256": canonical_sha256(input_payload),
        "provenance": provenance_payload,
        "provenance_sha256": provenance_digest,
        "scaler": {
            "method": SCALER_METHOD,
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
            "constant_feature_indices": constant_columns,
        },
        "clustering": {
            "method": CLUSTERING_METHOD,
            "seed": resolved_seed,
            "n_init": starts,
            "max_iterations": iterations,
            "convergence_tolerance": tolerance,
            "selected_start_index": selected_start,
            "selected_iterations": iteration_count,
            "feasible_start_count": len(candidates),
            "rejected_starts": rejected_starts,
            "inertia": inertia,
            "tie_breaking": "argmin_lowest_canonical_cluster_then_lexicographic_fit_key",
        },
        "minimum_support": {
            "minimum_cell_size": min_cell,
            "minimum_source_groups_per_cell": min_groups,
        },
        "centroids_standardized": [[float(value) for value in centroid] for centroid in centroids],
        "fit_support": support,
        "fit_assignments": [
            {
                "motion_key": motion,
                "source_group_id": group,
                "cluster_index": int(label),
            }
            for motion, group, label in zip(motions, groups, labels, strict=True)
        ],
    }
    artifact["representation_sha256"] = canonical_sha256(
        artifact, digest_field="representation_sha256"
    )
    return artifact


def validate_frozen_representation(artifact: Mapping[str, Any]) -> None:
    """Fail closed on a malformed or tampered frozen representation artifact."""

    _require(isinstance(artifact, Mapping), "artifact must be a mapping")
    _require(artifact.get("kind") == REPRESENTATION_KIND, "representation kind mismatch")
    _require(
        artifact.get("schema_version") == REPRESENTATION_SCHEMA_VERSION,
        "representation schema version mismatch",
    )
    _identifier(artifact.get("representation_id"), "representation_id")
    _require(artifact.get("outcome_blind") is True, "outcome_blind must be true")
    _require(artifact.get("fit_partition") == FIT_PARTITION, "fit_partition must be D_atlas")
    _require(
        artifact.get("allowed_assignment_partitions") == list(ALLOWED_ASSIGNMENT_PARTITIONS),
        "allowed_assignment_partitions mismatch",
    )
    k = _integer(artifact.get("k"), "k", minimum=2)
    _require(k in SUPPORTED_K, f"k must be one of {SUPPORTED_K}")
    _require(artifact.get("primary_k") == PRIMARY_K, "primary_k mismatch")
    _require(artifact.get("is_primary_k") is (k == PRIMARY_K), "is_primary_k mismatch")
    features = _names(artifact.get("feature_names"), "feature_names")
    _sha256(artifact.get("input_sha256"), "input_sha256")
    provenance = artifact.get("provenance")
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    provenance_digest = _sha256(artifact.get("provenance_sha256"), "provenance_sha256")
    _require(provenance_digest == canonical_sha256(provenance), "provenance_sha256 mismatch")
    scaler = artifact.get("scaler")
    _require(
        isinstance(scaler, Mapping) and scaler.get("method") == SCALER_METHOD,
        "scaler method mismatch",
    )
    mean = _matrix([scaler.get("mean")], rows=1, columns=len(features), name="scaler.mean")[0]
    scale = _matrix([scaler.get("scale")], rows=1, columns=len(features), name="scaler.scale")[0]
    _require(np.all(scale > 0.0), "scaler.scale must be positive")
    constants = scaler.get("constant_feature_indices")
    _require(isinstance(constants, list), "constant_feature_indices must be a list")
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(features)
            for value in constants
        ),
        "constant_feature_indices are invalid",
    )
    _require(
        constants == sorted(set(constants)), "constant_feature_indices must be sorted and unique"
    )
    clustering = artifact.get("clustering")
    _require(
        isinstance(clustering, Mapping) and clustering.get("method") == CLUSTERING_METHOD,
        "clustering method mismatch",
    )
    _integer(clustering.get("seed"), "clustering.seed", minimum=0)
    n_init = _integer(clustering.get("n_init"), "clustering.n_init", minimum=1)
    _integer(clustering.get("max_iterations"), "clustering.max_iterations", minimum=1)
    _real(
        clustering.get("convergence_tolerance"),
        "clustering.convergence_tolerance",
        positive=True,
    )
    selected_start = _integer(
        clustering.get("selected_start_index"),
        "clustering.selected_start_index",
        minimum=0,
    )
    _require(selected_start < n_init, "clustering.selected_start_index must be below n_init")
    _integer(clustering.get("selected_iterations"), "clustering.selected_iterations", minimum=1)
    feasible_starts = _integer(
        clustering.get("feasible_start_count"),
        "clustering.feasible_start_count",
        minimum=1,
    )
    _require(feasible_starts <= n_init, "clustering.feasible_start_count must not exceed n_init")
    _real(clustering.get("inertia"), "clustering.inertia")
    _require(
        clustering.get("tie_breaking")
        == "argmin_lowest_canonical_cluster_then_lexicographic_fit_key",
        "clustering tie_breaking mismatch",
    )
    rejected_starts = clustering.get("rejected_starts")
    _require(isinstance(rejected_starts, list), "clustering.rejected_starts must be a list")
    _require(
        feasible_starts + len(rejected_starts) == n_init,
        "feasible and rejected start counts must sum to n_init",
    )
    centroids = _matrix(
        artifact.get("centroids_standardized"),
        rows=k,
        columns=len(features),
        name="centroids_standardized",
    )
    _require(
        centroids.shape == (k, len(features)) and mean.shape == scale.shape,
        "representation dimensions mismatch",
    )
    support = artifact.get("fit_support")
    _require(
        isinstance(support, list)
        and len(support) == k
        and all(isinstance(cell, Mapping) for cell in support),
        "fit_support must contain k cell mappings",
    )
    minimum = artifact.get("minimum_support")
    _require(isinstance(minimum, Mapping), "minimum_support must be a mapping")
    min_cell = _integer(minimum.get("minimum_cell_size"), "minimum_cell_size", minimum=1)
    min_groups = _integer(
        minimum.get("minimum_source_groups_per_cell"),
        "minimum_source_groups_per_cell",
        minimum=1,
    )
    _require(
        [cell.get("cluster_index") for cell in support] == list(range(k)),
        "fit_support cluster indices must be canonical and complete",
    )
    for cluster, cell in enumerate(support):
        _integer(cell.get("motion_count"), f"fit_support[{cluster}].motion_count", minimum=1)
        _integer(
            cell.get("source_group_count"),
            f"fit_support[{cluster}].source_group_count",
            minimum=1,
        )
    _require(
        _support_is_feasible(
            support,
            minimum_cell_size=min_cell,
            minimum_source_groups_per_cell=min_groups,
        ),
        "fit_support violates frozen minimum support",
    )
    assignments = artifact.get("fit_assignments")
    _require(
        isinstance(assignments, list) and bool(assignments), "fit_assignments must be non-empty"
    )
    _require(
        [record.get("motion_key") for record in assignments]
        == sorted(record.get("motion_key") for record in assignments),
        "fit_assignments must be ordered by motion_key",
    )
    _require(
        len({record.get("motion_key") for record in assignments}) == len(assignments),
        "fit_assignments motion keys must be unique",
    )
    _require(
        all(
            isinstance(record, Mapping)
            and isinstance(record.get("source_group_id"), str)
            and bool(record.get("source_group_id"))
            and isinstance(record.get("cluster_index"), int)
            and not isinstance(record.get("cluster_index"), bool)
            and 0 <= record.get("cluster_index") < k
            for record in assignments
        ),
        "fit_assignments contain invalid records",
    )
    recomputed_support = [
        {
            "cluster_index": cluster,
            "motion_count": sum(record["cluster_index"] == cluster for record in assignments),
            "source_group_count": len(
                {
                    record["source_group_id"]
                    for record in assignments
                    if record["cluster_index"] == cluster
                }
            ),
        }
        for cluster in range(k)
    ]
    _require(recomputed_support == support, "fit_support does not match fit_assignments")
    digest = _sha256(artifact.get("representation_sha256"), "representation_sha256")
    _require(
        digest == canonical_sha256(artifact, digest_field="representation_sha256"),
        "representation_sha256 mismatch",
    )


def assign_frozen_representation(
    artifact: Mapping[str, Any],
    feature_matrix: Any,
    *,
    feature_names: Sequence[str],
    motion_keys: Sequence[str],
    source_group_ids: Sequence[str],
    partitions: Sequence[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign rows using frozen atlas parameters without any fit or fallback."""

    validate_frozen_representation(artifact)
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    matrix, features, motions, groups, resolved_partitions = _validate_rows(
        feature_matrix,
        feature_names=feature_names,
        motion_keys=motion_keys,
        source_group_ids=source_group_ids,
        partitions=partitions,
        allowed_partitions=ALLOWED_ASSIGNMENT_PARTITIONS,
    )
    _require(
        list(features) == artifact["feature_names"],
        "feature_names must exactly match the frozen ordered feature schema",
    )
    mean = np.asarray(artifact["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler"]["scale"], dtype=np.float64)
    centroids = np.asarray(artifact["centroids_standardized"], dtype=np.float64)
    standardized = (matrix - mean) / scale
    distances = np.square(standardized[:, None, :] - centroids[None, :, :]).sum(axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    k = int(artifact["k"])
    input_payload = _input_payload(
        matrix,
        feature_names=features,
        motion_keys=motions,
        source_group_ids=groups,
        partitions=resolved_partitions,
    )
    result: dict[str, Any] = {
        "kind": ASSIGNMENT_KIND,
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "representation_id": artifact["representation_id"],
        "representation_sha256": artifact["representation_sha256"],
        "k": k,
        "feature_names": list(features),
        "input_sha256": canonical_sha256(input_payload),
        "provenance": dict(provenance),
        "provenance_sha256": canonical_sha256(provenance),
        "fit_or_refit_performed": False,
        "tie_breaking": "numpy_argmin_lowest_canonical_cluster_index",
        "assignments": [
            {
                "motion_key": motion,
                "source_group_id": group,
                "partition": partition,
                "cluster_index": int(label),
                "membership": [1.0 if cluster == label else 0.0 for cluster in range(k)],
                "squared_distance_to_centroid": float(distances[index, label]),
            }
            for index, (motion, group, partition, label) in enumerate(
                zip(motions, groups, resolved_partitions, labels, strict=True)
            )
        ],
    }
    result["assignment_sha256"] = canonical_sha256(result, digest_field="assignment_sha256")
    return result


def random_matched_size_assignment(
    observed_assignments: Mapping[str, int],
    *,
    source_group_ids: Mapping[str, str],
    seed: int,
    control_id: str,
) -> dict[str, Any]:
    """Permute observed labels while preserving exact cell sizes.

    This is a motion-level random control.  It binds source-group identities for
    blocked downstream inference, but it intentionally does not preserve entire
    source groups as indivisible clusters.  Callers must use multiple explicit
    seeds rather than select a favorable random control from transfer outcomes.
    """

    _require(
        isinstance(observed_assignments, Mapping) and bool(observed_assignments),
        "observed_assignments must be a non-empty mapping",
    )
    _require(isinstance(source_group_ids, Mapping), "source_group_ids must be a mapping")
    motions = tuple(sorted(observed_assignments))
    _require(
        all(isinstance(key, str) and bool(key) for key in motions),
        "observed_assignments keys must be non-empty strings",
    )
    _require(
        set(source_group_ids) == set(motions),
        "source_group_ids keys must exactly match observed_assignments",
    )
    groups = tuple(
        _identifier(source_group_ids[motion], f"source_group_ids[{motion!r}]") for motion in motions
    )
    labels = tuple(observed_assignments[motion] for motion in motions)
    _require(
        all(
            isinstance(label, Integral)
            and not isinstance(label, (bool, np.bool_))
            and int(label) >= 0
            for label in labels
        ),
        "observed assignments must be nonnegative integer labels",
    )
    integer_labels = tuple(int(label) for label in labels)
    unique_labels = sorted(set(integer_labels))
    _require(
        unique_labels == list(range(len(unique_labels))),
        "observed labels must be contiguous from zero",
    )
    _require(
        len(unique_labels) in SUPPORTED_K,
        f"observed assignments must use K in {SUPPORTED_K}",
    )
    resolved_seed = _integer(seed, "seed", minimum=0)
    resolved_control_id = _identifier(control_id, "control_id")
    rng = np.random.default_rng(resolved_seed)
    permutation = rng.permutation(len(motions))
    randomized = [integer_labels[int(index)] for index in permutation]
    observed_counts = np.bincount(integer_labels, minlength=len(unique_labels)).astype(int)
    randomized_counts = np.bincount(randomized, minlength=len(unique_labels)).astype(int)
    _require(
        np.array_equal(observed_counts, randomized_counts),
        "internal error: random assignment changed cell sizes",
    )
    input_payload = {
        "observed_assignments": [
            {
                "motion_key": motion,
                "source_group_id": group,
                "cluster_index": label,
            }
            for motion, group, label in zip(motions, groups, integer_labels, strict=True)
        ]
    }
    result: dict[str, Any] = {
        "kind": RANDOM_ASSIGNMENT_KIND,
        "schema_version": RANDOM_ASSIGNMENT_SCHEMA_VERSION,
        "control_id": resolved_control_id,
        "seed": resolved_seed,
        "unit": "motion",
        "source_groups_are_indivisible": False,
        "k": len(unique_labels),
        "input_sha256": canonical_sha256(input_payload),
        "observed_cell_sizes": observed_counts.tolist(),
        "randomized_cell_sizes": randomized_counts.tolist(),
        "assignments": [
            {
                "motion_key": motion,
                "source_group_id": group,
                "cluster_index": label,
                "membership": [
                    1.0 if cluster == label else 0.0 for cluster in range(len(unique_labels))
                ],
            }
            for motion, group, label in zip(motions, groups, randomized, strict=True)
        ],
    }
    result["random_assignment_sha256"] = canonical_sha256(
        result, digest_field="random_assignment_sha256"
    )
    return result


def validate_representation_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the preregistered scale-512 representation protocol.

    The constants are intentionally exact rather than configurable.  A new
    split, support rule, clustering implementation, or random-null definition
    requires a new protocol schema/config rather than a permissive override.
    """

    _require(isinstance(protocol, Mapping), "protocol must be a mapping")
    _exact_keys(
        protocol,
        {
            "schema_version",
            "kind",
            "frozen",
            "scientific_use",
            "declared_before_transfer_outcomes",
            "outcome_access_permitted",
            "split",
            "candidate_representations",
            "clustering",
            "random_matched_size_controls",
            "protocol_sha256",
        },
        "protocol",
    )
    _require(protocol.get("kind") == REPRESENTATION_PROTOCOL_KIND, "protocol kind mismatch")
    _require(
        protocol.get("schema_version") == REPRESENTATION_PROTOCOL_SCHEMA_VERSION,
        "protocol schema version mismatch",
    )
    _require(protocol.get("frozen") is True, "protocol must be frozen")
    _require(protocol.get("scientific_use") is True, "scientific_use must be true")
    _require(
        protocol.get("declared_before_transfer_outcomes") is True,
        "protocol must be declared before transfer outcomes",
    )
    _require(
        protocol.get("outcome_access_permitted") is False,
        "protocol must prohibit transfer-outcome access",
    )

    split = protocol.get("split")
    _require(isinstance(split, Mapping), "split must be a mapping")
    _exact_keys(
        split,
        {"selection_sha256", "fit_partition", "motion_count", "source_group_count"},
        "split",
    )
    _require(
        _sha256(split.get("selection_sha256"), "split.selection_sha256")
        == SCALE512_SPLIT_SELECTION_SHA256,
        "split selection_sha256 drift",
    )
    _require(split.get("fit_partition") == FIT_PARTITION, "split fit_partition drift")
    _require(
        _integer(split.get("motion_count"), "split.motion_count", minimum=1)
        == SCALE512_D_ATLAS_MOTION_COUNT,
        "split motion_count drift",
    )
    _require(
        _integer(split.get("source_group_count"), "split.source_group_count", minimum=1)
        == SCALE512_D_ATLAS_SOURCE_GROUP_COUNT,
        "split source_group_count drift",
    )
    _require(
        protocol.get("candidate_representations") == list(SCALE512_CANDIDATE_REPRESENTATIONS),
        "candidate_representations drift",
    )

    clustering = protocol.get("clustering")
    _require(isinstance(clustering, Mapping), "clustering must be a mapping")
    _exact_keys(
        clustering,
        {
            "method",
            "scaler_method",
            "supported_k",
            "primary_k",
            "shared_seed",
            "n_init",
            "max_iterations",
            "convergence_tolerance",
            "tie_breaking",
            "minimum_support_by_k",
        },
        "clustering",
    )
    _require(clustering.get("method") == CLUSTERING_METHOD, "clustering method drift")
    _require(clustering.get("scaler_method") == SCALER_METHOD, "scaler method drift")
    _require(clustering.get("supported_k") == list(SUPPORTED_K), "supported_k drift")
    _require(clustering.get("primary_k") == PRIMARY_K, "primary_k drift")
    _require(
        _integer(clustering.get("shared_seed"), "clustering.shared_seed", minimum=0)
        == SCALE512_SHARED_SEED,
        "shared_seed drift",
    )
    _require(
        _integer(clustering.get("n_init"), "clustering.n_init", minimum=1) == 32,
        "n_init drift",
    )
    _require(
        _integer(
            clustering.get("max_iterations"),
            "clustering.max_iterations",
            minimum=1,
        )
        == 300,
        "max_iterations drift",
    )
    _require(
        _real(
            clustering.get("convergence_tolerance"),
            "clustering.convergence_tolerance",
            positive=True,
        )
        == 1e-10,
        "convergence_tolerance drift",
    )
    _require(
        clustering.get("tie_breaking")
        == "argmin_lowest_canonical_cluster_then_lexicographic_fit_key",
        "tie_breaking drift",
    )
    support = clustering.get("minimum_support_by_k")
    _require(isinstance(support, Mapping), "minimum_support_by_k must be a mapping")
    _exact_keys(support, {str(k) for k in SUPPORTED_K}, "minimum_support_by_k")
    for k, (minimum_cell_size, minimum_source_groups) in SCALE512_SUPPORT_BY_K.items():
        cell = support.get(str(k))
        _require(isinstance(cell, Mapping), f"minimum_support_by_k.{k} must be a mapping")
        _exact_keys(
            cell,
            {"minimum_cell_size", "minimum_source_groups_per_cell"},
            f"minimum_support_by_k.{k}",
        )
        _require(
            _integer(
                cell.get("minimum_cell_size"),
                f"minimum_support_by_k.{k}.minimum_cell_size",
                minimum=1,
            )
            == minimum_cell_size,
            f"minimum cell size drift for K={k}",
        )
        _require(
            _integer(
                cell.get("minimum_source_groups_per_cell"),
                f"minimum_support_by_k.{k}.minimum_source_groups_per_cell",
                minimum=1,
            )
            == minimum_source_groups,
            f"minimum source groups drift for K={k}",
        )

    random_controls = protocol.get("random_matched_size_controls")
    _require(isinstance(random_controls, Mapping), "random controls must be a mapping")
    _exact_keys(
        random_controls,
        {
            "unit",
            "control_count",
            "seed_base",
            "seed_rule",
            "preserve_observed_cell_sizes_exactly",
            "retain_source_group_ids_for_blocked_inference",
            "selection_from_transfer_outcomes_permitted",
            "whole_source_group_sensitivity",
        },
        "random_matched_size_controls",
    )
    _require(random_controls.get("unit") == "motion", "random control unit drift")
    _require(
        _integer(random_controls.get("control_count"), "random control_count", minimum=1)
        == SCALE512_RANDOM_CONTROL_COUNT,
        "random control_count drift",
    )
    _require(
        _integer(random_controls.get("seed_base"), "random seed_base", minimum=0)
        == SCALE512_RANDOM_SEED_BASE,
        "random seed_base drift",
    )
    _require(
        random_controls.get("seed_rule") == "seed_base_plus_zero_based_control_index",
        "random seed rule drift",
    )
    _require(
        random_controls.get("preserve_observed_cell_sizes_exactly") is True,
        "random controls must preserve observed cell sizes exactly",
    )
    _require(
        random_controls.get("retain_source_group_ids_for_blocked_inference") is True,
        "random controls must retain source IDs",
    )
    _require(
        random_controls.get("selection_from_transfer_outcomes_permitted") is False,
        "random control selection must prohibit transfer outcomes",
    )
    sensitivity = random_controls.get("whole_source_group_sensitivity")
    _require(isinstance(sensitivity, Mapping), "whole_source_group_sensitivity must be a mapping")
    _exact_keys(
        sensitivity,
        {"role", "exact_cell_size_preservation_guaranteed", "reason"},
        "whole_source_group_sensitivity",
    )
    _require(
        sensitivity.get("role") == "separate_sensitivity_not_primary_matched_size_null",
        "whole-source-group sensitivity role drift",
    )
    _require(
        sensitivity.get("exact_cell_size_preservation_guaranteed") is False,
        "whole-source-group sensitivity cannot guarantee exact cell sizes",
    )
    reason = sensitivity.get("reason")
    _require(
        isinstance(reason, str)
        and "heterogeneous motion counts" in reason
        and "cannot generally preserve" in reason,
        "whole-source-group sensitivity reason must explain the cell-size conflict",
    )

    digest = _sha256(protocol.get("protocol_sha256"), "protocol_sha256")
    _require(
        digest == canonical_sha256(protocol, digest_field="protocol_sha256"),
        "protocol_sha256 mismatch",
    )


def resolve_representation_fit_kwargs(
    protocol: Mapping[str, Any],
    *,
    k: int,
    observed_split_selection_sha256: str,
    observed_d_atlas_motion_count: int,
    observed_d_atlas_source_group_count: int,
) -> dict[str, Any]:
    """Resolve fit hyperparameters only after binding the observed split.

    No transfer metric or outcome is accepted by this API.  The result can be
    expanded directly into :func:`fit_frozen_representation` alongside the
    representation's matrix, identities, ID, and provenance.
    """

    validate_representation_protocol(protocol)
    resolved_k = _integer(k, "k", minimum=2)
    _require(resolved_k in SUPPORTED_K, f"k must be one of {SUPPORTED_K}")
    _require(
        _sha256(observed_split_selection_sha256, "observed_split_selection_sha256")
        == SCALE512_SPLIT_SELECTION_SHA256,
        "observed split selection does not match the frozen protocol",
    )
    _require(
        _integer(
            observed_d_atlas_motion_count,
            "observed_d_atlas_motion_count",
            minimum=1,
        )
        == SCALE512_D_ATLAS_MOTION_COUNT,
        "observed D_atlas motion count does not match the frozen protocol",
    )
    _require(
        _integer(
            observed_d_atlas_source_group_count,
            "observed_d_atlas_source_group_count",
            minimum=1,
        )
        == SCALE512_D_ATLAS_SOURCE_GROUP_COUNT,
        "observed D_atlas source-group count does not match the frozen protocol",
    )
    minimum_cell_size, minimum_source_groups = SCALE512_SUPPORT_BY_K[resolved_k]
    clustering = protocol["clustering"]
    return {
        "k": resolved_k,
        "seed": clustering["shared_seed"],
        "minimum_cell_size": minimum_cell_size,
        "minimum_source_groups_per_cell": minimum_source_groups,
        "n_init": clustering["n_init"],
        "max_iterations": clustering["max_iterations"],
        "convergence_tolerance": clustering["convergence_tolerance"],
    }


def resolve_random_control_seeds(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    """Return all preregistered matched-size null seeds in frozen order."""

    validate_representation_protocol(protocol)
    controls = protocol["random_matched_size_controls"]
    base = int(controls["seed_base"])
    count = int(controls["control_count"])
    return tuple(base + index for index in range(count))
