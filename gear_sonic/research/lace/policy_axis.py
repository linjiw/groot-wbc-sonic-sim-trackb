"""Outcome-blind policy-axis construction for LACE failure representations.

Episode aggregation deliberately emits one signature per ``(motion, policy)``.
The primary representation in the design is instead one motion row formed by
concatenating early, middle, and late Lite signatures in a frozen order.  This
module makes that transformation explicit and reusable for both D_atlas fitting
and assignment-only partitions; it never inspects transfer outcomes.  Other
ordered policy sets are emitted only as sensitivity axes and cannot claim the
predeclared primary representation.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

from gear_sonic.research.lace.analysis_protocol import ANALYSIS_PROTOCOL_DIGEST_FIELD
from gear_sonic.research.lace.schedule import (
    SCHEDULE_DIGEST_FIELD,
    SCHEDULE_KIND,
)
from gear_sonic.research.lace.schema import (
    ATLAS_KIND,
    canonical_sha256,
    validate_split_manifest,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

POLICY_AXIS_KIND = "lace_policy_axis_failure_features"
POLICY_AXIS_SCHEMA_VERSION = 2
POLICY_AXIS_DIGEST_FIELD = "policy_axis_sha256"
PRIMARY_AXIS_METHOD = "ordered_policy_q_concatenation_v1"
SENSITIVITY_AXIS_METHOD = "ordered_policy_q_concatenation_sensitivity_v1"
AVERAGE_SENSITIVITY_METHOD = "equal_policy_q_mean_common_support_v1"
ALLOWED_PARTITIONS = ("D_atlas", "D_curriculum", "D_geometry", "D_controller")
PRIMARY_POLICY_IDS = ("pi_lite_early", "pi_lite_mid", "pi_lite_late")

_POLICY_FIELDS = {"id", "checkpoint_sha256"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _verified_self_digest(
    payload: Mapping[str, Any],
    *,
    digest_field: str,
    expected_sha256: str,
    name: str,
) -> str:
    """Verify both an artifact's self-digest and an independent frozen pin."""

    _require(isinstance(payload, Mapping), f"{name} must be a mapping")
    embedded = _sha256(payload.get(digest_field), f"{name}.{digest_field}")
    computed = canonical_sha256(payload, digest_field=digest_field)
    _require(embedded == computed, f"{name} self-digest mismatch")
    expected = _sha256(expected_sha256, f"expected_{name}_sha256")
    _require(
        computed == expected,
        f"{name} does not match independently expected digest",
    )
    return computed


def _ordered_policies(raw: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
        "policy_checkpoints must be a sequence",
    )
    policies: list[dict[str, str]] = []
    for index, policy in enumerate(raw):
        _require(isinstance(policy, Mapping), f"policy_checkpoints[{index}] must be a mapping")
        _require(set(policy) == _POLICY_FIELDS, f"policy_checkpoints[{index}] fields invalid")
        policy_id = policy.get("id")
        _require(
            isinstance(policy_id, str) and policy_id, f"policy_checkpoints[{index}].id invalid"
        )
        policies.append(
            {
                "id": policy_id,
                "checkpoint_sha256": _sha256(
                    policy.get("checkpoint_sha256"),
                    f"policy_checkpoints[{index}].checkpoint_sha256",
                ),
            }
        )
    _require(len(policies) >= 2, "policy axis requires at least two ordered policies")
    ids = [policy["id"] for policy in policies]
    _require(len(ids) == len(set(ids)), "policy IDs must be unique")
    return tuple(policies)


def _verify_parent_bindings(
    signatures: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    policy_checkpoints: Sequence[Mapping[str, Any]],
    signature_artifact: Mapping[str, Any],
    parent_normalizer: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    policy_parent_manifest: Mapping[str, Any],
    mechanism_names: Sequence[str],
    minimum_resolved_failures: int,
    expected_signature_artifact_sha256: str,
    expected_parent_normalizer_sha256: str,
    expected_analysis_protocol_sha256: str,
    expected_policy_parent_sha256: str,
) -> dict[str, Any]:
    """Resolve provenance from actual parents and independent pre-outcome pins.

    The expected digests must come from the frozen experiment configuration,
    not from the policy-axis artifact being checked.  Requiring both sides is
    what makes a fully rehashed parent substitution detectable.
    """

    _require(
        signature_artifact.get("kind") == ATLAS_KIND,
        "signature_artifact must be a LACE failure atlas",
    )
    signature_sha256 = _verified_self_digest(
        signature_artifact,
        digest_field="atlas_sha256",
        expected_sha256=expected_signature_artifact_sha256,
        name="signature_artifact",
    )
    _require(
        signature_artifact.get("split_sha256") == split_manifest.get("split_sha256"),
        "signature_artifact split_sha256 mismatch",
    )
    _require(
        signature_artifact.get("split_selection_sha256") == split_manifest.get("selection_sha256"),
        "signature_artifact split selection mismatch",
    )
    _require(
        signature_artifact.get("signatures") == list(signatures),
        "signature rows do not exactly match signature_artifact",
    )
    names = _mechanisms(mechanism_names)
    _require(
        signature_artifact.get("mechanism_names") == list(names),
        "mechanism_names do not exactly match signature_artifact",
    )
    signature_config = signature_artifact.get("signature_config")
    _require(
        isinstance(signature_config, Mapping)
        and signature_config.get("minimum_resolved_failures") == minimum_resolved_failures,
        "minimum_resolved_failures does not match signature_artifact",
    )

    _require(isinstance(parent_normalizer, Mapping), "parent_normalizer must be a mapping")
    normalizer_sha256 = canonical_sha256(parent_normalizer)
    _require(
        normalizer_sha256
        == _sha256(
            expected_parent_normalizer_sha256,
            "expected_parent_normalizer_sha256",
        ),
        "parent_normalizer does not match independently expected digest",
    )
    _require(
        signature_artifact.get("normalizer") == dict(parent_normalizer),
        "parent_normalizer does not exactly match signature_artifact.normalizer",
    )

    protocol_sha256 = _verified_self_digest(
        analysis_protocol,
        digest_field=ANALYSIS_PROTOCOL_DIGEST_FIELD,
        expected_sha256=expected_analysis_protocol_sha256,
        name="analysis_protocol",
    )
    _require(
        signature_artifact.get("analysis_protocol") == dict(analysis_protocol),
        "analysis_protocol does not exactly match signature_artifact.analysis_protocol",
    )
    _require(
        signature_artifact.get(ANALYSIS_PROTOCOL_DIGEST_FIELD) == protocol_sha256,
        "signature_artifact analysis protocol digest mismatch",
    )

    policies = _ordered_policies(policy_checkpoints)
    atlas_policies = _ordered_policies(signature_artifact.get("probe_policies"))
    _require(
        atlas_policies == policies,
        "policy checkpoints do not exactly match signature_artifact probe_policies",
    )
    parent_kind = policy_parent_manifest.get("kind")
    if parent_kind == SCHEDULE_KIND:
        parent_digest_field = SCHEDULE_DIGEST_FIELD
        parent_label = "policy_parent_schedule"
    elif parent_kind == ATLAS_KIND:
        parent_digest_field = "atlas_sha256"
        parent_label = "policy_parent_atlas"
    else:
        raise ValueError("policy_parent_manifest must be a rollout schedule or failure atlas")
    policy_parent_sha256 = _verified_self_digest(
        policy_parent_manifest,
        digest_field=parent_digest_field,
        expected_sha256=expected_policy_parent_sha256,
        name=parent_label,
    )
    _require(
        policy_parent_manifest.get("split_sha256") == split_manifest.get("split_sha256"),
        "policy parent split_sha256 mismatch",
    )
    parent_policies = _ordered_policies(policy_parent_manifest.get("probe_policies"))
    _require(
        parent_policies == policies,
        "policy checkpoints do not exactly match policy parent probe_policies",
    )
    if parent_kind == SCHEDULE_KIND:
        _require(
            signature_artifact.get("rollout_schedule_sha256") == policy_parent_sha256,
            "signature_artifact is not bound to the policy parent schedule",
        )
    else:
        _require(
            policy_parent_manifest.get("probe_policies")
            == signature_artifact.get("probe_policies"),
            "signature and policy-parent atlas policy axes differ",
        )

    return {
        "signature_artifact_sha256": signature_sha256,
        "parent_normalizer_sha256": normalizer_sha256,
        "analysis_protocol_sha256": protocol_sha256,
        "policy_parent_kind": parent_kind,
        "policy_parent_sha256": policy_parent_sha256,
        "policies": policies,
    }


def _mechanisms(raw: Sequence[str]) -> tuple[str, ...]:
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)),
        "mechanism_names must be a sequence",
    )
    names = tuple(raw)
    _require(
        len(names) >= 2 and all(isinstance(name, str) and name for name in names),
        "mechanism_names are invalid",
    )
    _require(len(names) == len(set(names)), "mechanism_names must be unique")
    return names


def _bounded_real(value: Any, name: str) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{name} must be a real scalar",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{name} must lie in [0, 1]")
    return result


def _count(value: Any, name: str) -> int:
    _require(
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0,
        f"{name} must be a nonnegative integer",
    )
    return int(value)


def _q_vector(
    raw: Any,
    names: tuple[str, ...],
    name: str,
    *,
    require_unit_sum: bool = True,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    _require(isinstance(raw, list) and len(raw) == len(names), f"{name} shape mismatch")
    values = tuple(_bounded_real(value, f"{name}[{index}]") for index, value in enumerate(raw))
    if require_unit_sum:
        _require(abs(math.fsum(values) - 1.0) <= 1e-12, f"{name} must sum to one")
    return values


def _canonical_signatures(
    signatures: Iterable[Mapping[str, Any]],
    *,
    expected_motion_keys: tuple[str, ...],
    policy_ids: tuple[str, ...],
    mechanism_names: tuple[str, ...],
    minimum_resolved_failures: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    expected_motions = set(expected_motion_keys)
    expected_policies = set(policy_ids)
    for index, raw in enumerate(signatures):
        _require(isinstance(raw, Mapping), f"signatures[{index}] must be a mapping")
        row = dict(raw)
        motion_key = row.get("motion_key")
        policy_id = row.get("probe_policy_id")
        _require(motion_key in expected_motions, f"signatures[{index}] motion is outside partition")
        _require(policy_id in expected_policies, f"signatures[{index}] policy is undeclared")
        key = (str(motion_key), str(policy_id))
        _require(key not in rows, f"duplicate signature row {key!r}")
        rollouts = _count(row.get("num_rollouts"), f"signatures[{index}].num_rollouts")
        failures = _count(row.get("num_failures"), f"signatures[{index}].num_failures")
        resolved = _count(
            row.get("num_resolved_failures"),
            f"signatures[{index}].num_resolved_failures",
        )
        _require(rollouts > 0 and resolved <= failures <= rollouts, "signature counts invalid")
        _require(
            row.get("minimum_resolved_failures") == minimum_resolved_failures,
            "signature minimum-resolved-failures drifted",
        )
        difficulty = _bounded_real(row.get("difficulty"), f"signatures[{index}].difficulty")
        attributed = _bounded_real(
            row.get("attributed_failure_rate"),
            f"signatures[{index}].attributed_failure_rate",
        )
        unresolved = _bounded_real(
            row.get("unresolved_failure_probability"),
            f"signatures[{index}].unresolved_failure_probability",
        )
        _require(
            abs(difficulty - failures / rollouts) <= 1e-12
            and abs(attributed - resolved / rollouts) <= 1e-12
            and abs(unresolved - (failures - resolved) / rollouts) <= 1e-12,
            "signature rates disagree with counts",
        )
        q = _q_vector(row.get("q"), mechanism_names, f"signatures[{index}].q")
        _require(
            (q is not None) == (resolved >= minimum_resolved_failures),
            "signature q support disagrees with resolved-failure threshold",
        )
        f = _q_vector(
            row.get("f"),
            mechanism_names,
            f"signatures[{index}].f",
            require_unit_sum=False,
        )
        if q is None:
            _require(f is None, "unsupported q must have missing f")
        else:
            expected_f = tuple(attributed * value for value in q)
            _require(
                f is not None
                and all(
                    abs(actual - expected) <= 1e-12
                    for actual, expected in zip(f, expected_f, strict=True)
                ),
                "signature f is inconsistent with attributed_failure_rate * q",
            )
        rows[key] = row
    expected = {(motion, policy) for motion in expected_motion_keys for policy in policy_ids}
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    _require(not missing and not extra, f"signature Cartesian coverage mismatch; missing={missing}")
    return rows


def _assemble_policy_axis_features(
    signatures: Iterable[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    partition: str,
    policy_checkpoints: Sequence[Mapping[str, Any]],
    signature_artifact_sha256: str,
    parent_normalizer_sha256: str,
    analysis_protocol_sha256: str,
    policy_parent_kind: str,
    policy_parent_sha256: str,
    mechanism_names: Sequence[str],
    minimum_resolved_failures: int,
) -> dict[str, Any]:
    validate_split_manifest(split_manifest, verify_digest=True)
    _require(partition in ALLOWED_PARTITIONS, "policy-axis partition is not allowed")
    _require(
        isinstance(minimum_resolved_failures, int)
        and not isinstance(minimum_resolved_failures, bool)
        and minimum_resolved_failures >= 1,
        "minimum_resolved_failures must be a positive integer",
    )
    policies = _ordered_policies(policy_checkpoints)
    policy_ids = tuple(policy["id"] for policy in policies)
    is_primary_axis = policy_ids == PRIMARY_POLICY_IDS
    axis_method = PRIMARY_AXIS_METHOD if is_primary_axis else SENSITIVITY_AXIS_METHOD
    names = _mechanisms(mechanism_names)
    records = {
        str(row["motion_key"]): row
        for row in split_manifest["motions"]
        if row["partition"] == partition
    }
    motion_keys = tuple(sorted(records))
    _require(motion_keys, f"split partition {partition} is empty")
    signature_rows = _canonical_signatures(
        signatures,
        expected_motion_keys=motion_keys,
        policy_ids=policy_ids,
        mechanism_names=names,
        minimum_resolved_failures=minimum_resolved_failures,
    )
    canonical_input_rows = [
        signature_rows[(motion, policy)] for motion in motion_keys for policy in policy_ids
    ]
    rows: list[dict[str, Any]] = []
    support_by_policy = {policy_id: 0 for policy_id in policy_ids}
    common_support = 0
    for motion_key in motion_keys:
        policy_rows = [signature_rows[(motion_key, policy_id)] for policy_id in policy_ids]
        q_by_policy = [row["q"] for row in policy_rows]
        missing_policy_ids = [
            policy_id for policy_id, q in zip(policy_ids, q_by_policy, strict=True) if q is None
        ]
        for policy_id, q in zip(policy_ids, q_by_policy, strict=True):
            if q is not None:
                support_by_policy[policy_id] += 1
        supported = not missing_policy_ids
        if supported:
            common_support += 1
            concat = [float(value) for q in q_by_policy for value in q]
            average = [
                math.fsum(float(q[index]) for q in q_by_policy) / len(q_by_policy)
                for index in range(len(names))
            ]
        else:
            concat = None
            average = None
        rows.append(
            {
                "motion_key": motion_key,
                "source_group_id": records[motion_key]["source_group_id"],
                "partition": partition,
                "common_q_support": supported,
                "missing_policy_ids": missing_policy_ids,
                "failure_q_concat": concat,
                "failure_q_average_sensitivity": average,
                "scalar_difficulty_axis": [float(row["difficulty"]) for row in policy_rows],
                "unresolved_failure_axis": [
                    float(row["unresolved_failure_probability"]) for row in policy_rows
                ],
            }
        )
    artifact: dict[str, Any] = {
        "kind": POLICY_AXIS_KIND,
        "schema_version": POLICY_AXIS_SCHEMA_VERSION,
        "frozen": True,
        "scientific_use": True,
        "outcome_blind": True,
        "partition": partition,
        "representation_role": "fit" if partition == "D_atlas" else "assignment",
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "signature_artifact_sha256": _sha256(
            signature_artifact_sha256, "signature_artifact_sha256"
        ),
        "signature_rows_sha256": canonical_sha256({"rows": canonical_input_rows}),
        "parent_normalizer_sha256": _sha256(parent_normalizer_sha256, "parent_normalizer_sha256"),
        "analysis_protocol_sha256": _sha256(analysis_protocol_sha256, "analysis_protocol_sha256"),
        "policy_parent_kind": policy_parent_kind,
        "policy_parent_sha256": _sha256(policy_parent_sha256, "policy_parent_sha256"),
        "is_primary_axis": is_primary_axis,
        "axis_method": axis_method,
        "primary_axis_method": PRIMARY_AXIS_METHOD if is_primary_axis else None,
        "average_sensitivity_method": AVERAGE_SENSITIVITY_METHOD,
        "policy_checkpoints": [dict(policy) for policy in policies],
        "policy_order_sha256": canonical_sha256({"policy_checkpoints": policies}),
        "mechanism_names": list(names),
        "minimum_resolved_failures": minimum_resolved_failures,
        "feature_names": [
            f"{policy_id}::q::{mechanism}" for policy_id in policy_ids for mechanism in names
        ],
        "average_sensitivity_feature_names": [f"mean_policy_q::{name}" for name in names],
        "difficulty_feature_names": [f"{policy_id}::difficulty" for policy_id in policy_ids],
        "unresolved_nuisance_feature_names": [
            f"{policy_id}::unresolved_failure_probability" for policy_id in policy_ids
        ],
        "motion_count": len(motion_keys),
        "common_support_motion_count": common_support,
        "common_support_fraction": common_support / len(motion_keys),
        "support_motion_count_by_policy": support_by_policy,
        "rows": rows,
    }
    artifact[POLICY_AXIS_DIGEST_FIELD] = canonical_sha256(
        artifact,
        digest_field=POLICY_AXIS_DIGEST_FIELD,
    )
    return artifact


def build_policy_axis_features(
    signatures: Iterable[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    partition: str,
    policy_checkpoints: Sequence[Mapping[str, Any]],
    signature_artifact: Mapping[str, Any],
    parent_normalizer: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    policy_parent_manifest: Mapping[str, Any],
    expected_signature_artifact_sha256: str,
    expected_parent_normalizer_sha256: str,
    expected_analysis_protocol_sha256: str,
    expected_policy_parent_sha256: str,
    mechanism_names: Sequence[str] = DEFAULT_MECHANISMS,
    minimum_resolved_failures: int = 3,
) -> dict[str, Any]:
    """Build a provenance-bound ordered concat plus policy-mean sensitivity."""

    signature_rows = list(signatures)
    parents = _verify_parent_bindings(
        signature_rows,
        split_manifest,
        policy_checkpoints=policy_checkpoints,
        signature_artifact=signature_artifact,
        parent_normalizer=parent_normalizer,
        analysis_protocol=analysis_protocol,
        policy_parent_manifest=policy_parent_manifest,
        mechanism_names=mechanism_names,
        minimum_resolved_failures=minimum_resolved_failures,
        expected_signature_artifact_sha256=expected_signature_artifact_sha256,
        expected_parent_normalizer_sha256=expected_parent_normalizer_sha256,
        expected_analysis_protocol_sha256=expected_analysis_protocol_sha256,
        expected_policy_parent_sha256=expected_policy_parent_sha256,
    )
    return _assemble_policy_axis_features(
        signature_rows,
        split_manifest,
        partition=partition,
        policy_checkpoints=parents["policies"],
        signature_artifact_sha256=parents["signature_artifact_sha256"],
        parent_normalizer_sha256=parents["parent_normalizer_sha256"],
        analysis_protocol_sha256=parents["analysis_protocol_sha256"],
        policy_parent_kind=parents["policy_parent_kind"],
        policy_parent_sha256=parents["policy_parent_sha256"],
        mechanism_names=mechanism_names,
        minimum_resolved_failures=minimum_resolved_failures,
    )


def validate_policy_axis_features(
    artifact: Mapping[str, Any],
    signatures: Iterable[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    signature_artifact: Mapping[str, Any],
    parent_normalizer: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    policy_parent_manifest: Mapping[str, Any],
    expected_signature_artifact_sha256: str,
    expected_parent_normalizer_sha256: str,
    expected_analysis_protocol_sha256: str,
    expected_policy_parent_sha256: str,
) -> None:
    """Deep-rebuild against actual parents and independently frozen digests."""

    _require(artifact.get("kind") == POLICY_AXIS_KIND, "policy-axis kind mismatch")
    _require(
        artifact.get("schema_version") == POLICY_AXIS_SCHEMA_VERSION,
        "policy-axis schema version mismatch",
    )
    signature_rows = list(signatures)
    parents = _verify_parent_bindings(
        signature_rows,
        split_manifest,
        policy_checkpoints=artifact.get("policy_checkpoints"),
        signature_artifact=signature_artifact,
        parent_normalizer=parent_normalizer,
        analysis_protocol=analysis_protocol,
        policy_parent_manifest=policy_parent_manifest,
        mechanism_names=artifact.get("mechanism_names"),
        minimum_resolved_failures=artifact.get("minimum_resolved_failures"),
        expected_signature_artifact_sha256=expected_signature_artifact_sha256,
        expected_parent_normalizer_sha256=expected_parent_normalizer_sha256,
        expected_analysis_protocol_sha256=expected_analysis_protocol_sha256,
        expected_policy_parent_sha256=expected_policy_parent_sha256,
    )
    expected = _assemble_policy_axis_features(
        signature_rows,
        split_manifest,
        partition=artifact.get("partition"),
        policy_checkpoints=parents["policies"],
        signature_artifact_sha256=parents["signature_artifact_sha256"],
        parent_normalizer_sha256=parents["parent_normalizer_sha256"],
        analysis_protocol_sha256=parents["analysis_protocol_sha256"],
        policy_parent_kind=parents["policy_parent_kind"],
        policy_parent_sha256=parents["policy_parent_sha256"],
        mechanism_names=artifact.get("mechanism_names"),
        minimum_resolved_failures=artifact.get("minimum_resolved_failures"),
    )
    _require(dict(artifact) == expected, "policy-axis artifact differs from deep reconstruction")


def common_support_feature_rows(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Return aligned primary/sensitivity/baseline rows on exact common q support."""

    rows = artifact.get("rows")
    _require(isinstance(rows, list), "policy-axis rows missing")
    supported = [row for row in rows if row.get("common_q_support") is True]
    return {
        "motion_keys": [row["motion_key"] for row in supported],
        "source_group_ids": [row["source_group_id"] for row in supported],
        "partitions": [row["partition"] for row in supported],
        "failure_q_concat": [row["failure_q_concat"] for row in supported],
        "failure_q_average_sensitivity": [
            row["failure_q_average_sensitivity"] for row in supported
        ],
        "scalar_difficulty_axis": [row["scalar_difficulty_axis"] for row in supported],
        "unresolved_failure_axis": [row["unresolved_failure_axis"] for row in supported],
    }
