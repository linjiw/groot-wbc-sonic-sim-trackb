"""Frozen motion-level intervention plans for the LACE RQ1 experiment.

The low-level solver in :mod:`interventions` constructs one fixed-budget
distribution.  This module binds that solver to the source-disjoint split and
the representation-blind panel artifact, reconstructs both deterministically,
and produces all source-panel rows before transfer outcomes can be observed.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.research.lace.interventions import solve_fixed_budget_intervention
from gear_sonic.research.lace.panels import (
    build_representation_blind_panels,
    validate_panel_manifest,
)
from gear_sonic.research.lace.schema import canonical_sha256, validate_split_manifest

INTERVENTION_PLAN_KIND = "lace_rq1_motion_intervention_plan"
INTERVENTION_PLAN_SCHEMA_VERSION = 1
INTERVENTION_PROTOCOL_KIND = "lace_rq1_intervention_protocol"
INTERVENTION_PROTOCOL_SCHEMA_VERSION = 1
INTERVENTION_PLAN_DIGEST_FIELD = "intervention_plan_sha256"
INTERVENTION_PROTOCOL_DIGEST_FIELD = "protocol_sha256"
BASE_DISTRIBUTION_METHOD = "uniform_over_canonically_ordered_motions_v1"

_PROTOCOL_FIELDS = {
    "kind",
    "schema_version",
    "frozen",
    "scientific_use",
    "declared_before_transfer_outcomes",
    "split_selection_sha256",
    "partition",
    "expected_motion_count",
    "expected_panel_count",
    "panel_seed",
    "base_distribution_method",
    "sequence_length_agnostic",
    "target_kl_nats",
    "max_probability_ratio",
    "kl_tolerance",
    "probability_tolerance",
    "bisection_iterations",
    "maximum_added_exposure_range",
    "interpretation",
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    return value


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    _require(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return int(value)


def _finite_real(value: Any, name: str, *, minimum: float, strict: bool = False) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
        f"{name} must be a real scalar",
    )
    result = float(value)
    comparison = result > minimum if strict else result >= minimum
    _require(math.isfinite(result) and comparison, f"{name} is outside its valid range")
    return result


def validate_intervention_protocol(
    protocol: Mapping[str, Any],
    *,
    verify_digest: bool = True,
) -> None:
    """Validate the pre-outcome RQ1 dose and solver contract."""

    _require(isinstance(protocol, Mapping), "intervention protocol must be a mapping")
    _require(set(protocol) == _PROTOCOL_FIELDS, "intervention protocol fields are invalid")
    _require(
        protocol.get("kind") == INTERVENTION_PROTOCOL_KIND,
        "intervention protocol kind mismatch",
    )
    _require(
        protocol.get("schema_version") == INTERVENTION_PROTOCOL_SCHEMA_VERSION,
        "intervention protocol schema version mismatch",
    )
    _require(protocol.get("frozen") is True, "intervention protocol must be frozen")
    _require(protocol.get("scientific_use") is True, "intervention protocol must be scientific")
    _require(
        protocol.get("declared_before_transfer_outcomes") is True,
        "intervention protocol must be declared before transfer outcomes",
    )
    _sha256(protocol.get("split_selection_sha256"), "split_selection_sha256")
    _require(protocol.get("partition") == "D_curriculum", "partition must be D_curriculum")
    _positive_integer(protocol.get("expected_motion_count"), "expected_motion_count", minimum=2)
    _positive_integer(protocol.get("expected_panel_count"), "expected_panel_count", minimum=2)
    _require(
        isinstance(protocol.get("panel_seed"), int)
        and not isinstance(protocol.get("panel_seed"), bool),
        "panel_seed must be an integer",
    )
    _require(
        protocol.get("base_distribution_method") == BASE_DISTRIBUTION_METHOD,
        "base distribution method mismatch",
    )
    _require(
        protocol.get("sequence_length_agnostic") is True,
        "base distribution must be sequence-length agnostic",
    )
    _finite_real(protocol.get("target_kl_nats"), "target_kl_nats", minimum=0.0, strict=True)
    _finite_real(
        protocol.get("max_probability_ratio"),
        "max_probability_ratio",
        minimum=1.0,
    )
    _finite_real(protocol.get("kl_tolerance"), "kl_tolerance", minimum=0.0, strict=True)
    _finite_real(
        protocol.get("probability_tolerance"),
        "probability_tolerance",
        minimum=0.0,
        strict=True,
    )
    _positive_integer(protocol.get("bisection_iterations"), "bisection_iterations")
    _finite_real(
        protocol.get("maximum_added_exposure_range"),
        "maximum_added_exposure_range",
        minimum=0.0,
    )
    _require(
        isinstance(protocol.get("interpretation"), str) and protocol["interpretation"],
        "interpretation must be non-empty",
    )
    expected_digest = _sha256(
        protocol.get(INTERVENTION_PROTOCOL_DIGEST_FIELD),
        INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )
    if verify_digest:
        actual_digest = canonical_sha256(
            protocol,
            digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
        )
        _require(expected_digest == actual_digest, "intervention protocol digest mismatch")


def _uniform_probabilities(count: int) -> list[float]:
    raw = np.full(count, 1.0 / count, dtype=np.float64)
    # The low-level solver uses an accurate sum and freezes its correction.
    no_op = solve_fixed_budget_intervention(
        raw,
        np.ones(count, dtype=bool),
        target_kl=0.0,
        max_probability_ratio=1.0,
    )
    return list(no_op["base_probabilities"])


def _deep_validate_panels(
    panel_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_panel_manifest(panel_manifest, split_manifest=split_manifest, verify_digest=True)
    _require(
        panel_manifest.get("partition") == protocol["partition"],
        "panel partition does not match protocol",
    )
    _require(
        panel_manifest.get("panel_count") == protocol["expected_panel_count"],
        "panel count does not match protocol",
    )
    _require(
        panel_manifest.get("seed") == protocol["panel_seed"],
        "panel seed does not match protocol",
    )
    expected = build_representation_blind_panels(
        split_manifest,
        partition=str(protocol["partition"]),
        panel_count=int(protocol["expected_panel_count"]),
        seed=int(protocol["panel_seed"]),
    )
    _require(
        dict(panel_manifest) == expected,
        "panel artifact differs from deterministic representation-blind reconstruction",
    )
    return expected


def _build_intervention_plan(
    split_manifest: Mapping[str, Any],
    panel_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_split_manifest(split_manifest, verify_digest=True)
    validate_intervention_protocol(protocol, verify_digest=True)
    _require(
        protocol["split_selection_sha256"] == split_manifest["selection_sha256"],
        "protocol split selection does not match split manifest",
    )
    panels = _deep_validate_panels(panel_manifest, split_manifest, protocol)
    partition = str(protocol["partition"])
    motion_keys = sorted(
        str(record["motion_key"])
        for record in split_manifest["motions"]
        if record["partition"] == partition
    )
    _require(
        len(motion_keys) == protocol["expected_motion_count"],
        "partition motion count does not match protocol",
    )
    motion_index = {motion_key: index for index, motion_key in enumerate(motion_keys)}
    base_probabilities = _uniform_probabilities(len(motion_keys))

    rows: list[dict[str, Any]] = []
    for panel in panels["panels"]:
        support = np.zeros(len(motion_keys), dtype=bool)
        for motion_key in panel["motion_keys"]:
            _require(motion_key in motion_index, f"panel contains unknown motion {motion_key!r}")
            support[motion_index[motion_key]] = True
        intervention = solve_fixed_budget_intervention(
            base_probabilities,
            support,
            target_kl=float(protocol["target_kl_nats"]),
            max_probability_ratio=float(protocol["max_probability_ratio"]),
            kl_tolerance=float(protocol["kl_tolerance"]),
            probability_tolerance=float(protocol["probability_tolerance"]),
            bisection_iterations=int(protocol["bisection_iterations"]),
        )
        rows.append(
            {
                "panel_id": panel["panel_id"],
                "panel_motion_keys": list(panel["motion_keys"]),
                "panel_motion_count": panel["motion_count"],
                "panel_source_group_count": panel["source_group_count"],
                "intervention": intervention,
            }
        )

    added_exposures = [float(row["intervention"]["added_panel_exposure"]) for row in rows]
    exposure_range = max(added_exposures) - min(added_exposures)
    maximum_range = float(protocol["maximum_added_exposure_range"])
    _require(
        exposure_range <= maximum_range,
        "panel added-exposure range exceeds the frozen protocol",
    )
    manifest: dict[str, Any] = {
        "kind": INTERVENTION_PLAN_KIND,
        "schema_version": INTERVENTION_PLAN_SCHEMA_VERSION,
        "frozen": True,
        "scientific_use": True,
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "partition": partition,
        "panel_sha256": panels["panel_sha256"],
        "protocol": deepcopy(dict(protocol)),
        "protocol_sha256": protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD],
        "motion_keys": motion_keys,
        "motion_count": len(motion_keys),
        "motion_order_sha256": canonical_sha256({"motion_keys": motion_keys}),
        "base_distribution_method": BASE_DISTRIBUTION_METHOD,
        "base_probabilities": base_probabilities,
        "base_distribution_sha256": canonical_sha256(
            {"motion_keys": motion_keys, "probabilities": base_probabilities}
        ),
        "panel_count": len(rows),
        "interventions": rows,
        "dose_balance": {
            "minimum_added_panel_exposure": min(added_exposures),
            "maximum_added_panel_exposure": max(added_exposures),
            "added_panel_exposure_range": exposure_range,
            "maximum_allowed_range": maximum_range,
            "passes": True,
        },
    }
    manifest[INTERVENTION_PLAN_DIGEST_FIELD] = canonical_sha256(
        manifest,
        digest_field=INTERVENTION_PLAN_DIGEST_FIELD,
    )
    return manifest


def build_intervention_plan(
    split_manifest: Mapping[str, Any],
    panel_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and immediately reconstruct-validate the full RQ1 treatment plan."""

    manifest = _build_intervention_plan(split_manifest, panel_manifest, protocol)
    validate_intervention_plan(
        manifest,
        split_manifest=split_manifest,
        panel_manifest=panel_manifest,
        protocol=protocol,
    )
    return manifest


def validate_intervention_plan(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    panel_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Deep-verify a plan by deterministic reconstruction from frozen inputs."""

    _require(isinstance(manifest, Mapping), "intervention plan must be a mapping")
    _require(
        manifest.get("kind") == INTERVENTION_PLAN_KIND,
        "intervention plan kind mismatch",
    )
    _require(
        manifest.get("schema_version") == INTERVENTION_PLAN_SCHEMA_VERSION,
        "intervention plan schema version mismatch",
    )
    _sha256(manifest.get(INTERVENTION_PLAN_DIGEST_FIELD), INTERVENTION_PLAN_DIGEST_FIELD)
    expected = _build_intervention_plan(split_manifest, panel_manifest, protocol)
    _require(
        dict(manifest) == expected,
        "intervention plan differs from deterministic reconstruction",
    )


def intervention_probabilities_by_panel(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[float, ...]]:
    """Return immutable motion-order-aligned distributions from a validated plan payload."""

    rows = manifest.get("interventions")
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)), "rows missing")
    result: dict[str, tuple[float, ...]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "intervention row must be a mapping")
        panel_id = row.get("panel_id")
        _require(isinstance(panel_id, str) and panel_id, "panel_id missing")
        _require(panel_id not in result, f"duplicate panel_id {panel_id!r}")
        intervention = row.get("intervention")
        _require(isinstance(intervention, Mapping), "intervention payload missing")
        probabilities = intervention.get("p_plus")
        _require(isinstance(probabilities, list), "p_plus must be a list")
        result[panel_id] = tuple(float(value) for value in probabilities)
    return result
