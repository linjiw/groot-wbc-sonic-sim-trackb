from __future__ import annotations

from copy import deepcopy

import pytest

from gear_sonic.research.lace.analysis_protocol import ANALYSIS_PROTOCOL_DIGEST_FIELD
from gear_sonic.research.lace.policy_axis import (
    POLICY_AXIS_DIGEST_FIELD,
    PRIMARY_AXIS_METHOD,
    SENSITIVITY_AXIS_METHOD,
    build_policy_axis_features,
    common_support_feature_rows,
    validate_policy_axis_features,
)
from gear_sonic.research.lace.schedule import SCHEDULE_DIGEST_FIELD, SCHEDULE_KIND
from gear_sonic.research.lace.schema import ATLAS_KIND, canonical_sha256
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS
from gear_sonic.research.lace.split import build_source_disjoint_split


def _split() -> dict:
    motions = []
    for actor in range(25):
        for variant in range(2):
            motions.append(
                {
                    "motion_key": f"motion_{actor:02d}_{variant}__A{actor:03d}",
                    "source_group_id": f"actor-{actor:03d}",
                    "duration_source_frames": 120 + actor + variant,
                    "stratum": f"duration-{actor % 4}",
                }
            )
    return build_source_disjoint_split(motions, seed=813)


def _policies() -> list[dict[str, str]]:
    return [
        {"id": "pi_lite_early", "checkpoint_sha256": "1" * 64},
        {"id": "pi_lite_mid", "checkpoint_sha256": "2" * 64},
        {"id": "pi_lite_late", "checkpoint_sha256": "3" * 64},
    ]


def _q(motion_index: int, policy_index: int) -> list[float]:
    raw = [float(1 + motion_index + policy_index + channel) for channel in range(6)]
    total = sum(raw)
    return [value / total for value in raw]


def _signatures(split: dict, policies: list[dict[str, str]] | None = None) -> list[dict]:
    policies = _policies() if policies is None else policies
    motion_keys = sorted(
        row["motion_key"] for row in split["motions"] if row["partition"] == "D_atlas"
    )
    rows = []
    for motion_index, motion_key in enumerate(motion_keys):
        for policy_index, policy in enumerate(policies):
            q = _q(motion_index, policy_index)
            rows.append(
                {
                    "motion_key": motion_key,
                    "probe_policy_id": policy["id"],
                    "num_rollouts": 8,
                    "num_failures": 4,
                    "num_resolved_failures": 4,
                    "difficulty": 0.5,
                    "attributed_failure_rate": 0.5,
                    "unresolved_failure_probability": 0.0,
                    "minimum_resolved_failures": 3,
                    "q": q,
                    "f": [0.5 * value for value in q],
                }
            )
    return rows


def _parents(
    split: dict,
    signatures: list[dict],
    policies: list[dict[str, str]] | None = None,
) -> dict:
    policies = deepcopy(_policies() if policies is None else policies)
    normalizer = {
        "frozen": True,
        "fit_partition": "D_atlas",
        "mechanism_names": list(DEFAULT_MECHANISMS),
        "mechanism_scales": {
            name: float(index + 1) for index, name in enumerate(DEFAULT_MECHANISMS)
        },
    }
    protocol = {
        "kind": "test_frozen_analysis_protocol",
        "frozen": True,
        "mechanism_names": list(DEFAULT_MECHANISMS),
    }
    protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=ANALYSIS_PROTOCOL_DIGEST_FIELD,
    )
    schedule = {
        "kind": SCHEDULE_KIND,
        "split_sha256": split["split_sha256"],
        "probe_policies": deepcopy(policies),
        "fixture": "exact_policy_parent_v1",
    }
    schedule[SCHEDULE_DIGEST_FIELD] = canonical_sha256(
        schedule,
        digest_field=SCHEDULE_DIGEST_FIELD,
    )
    atlas = {
        "kind": ATLAS_KIND,
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "probe_policies": deepcopy(policies),
        "mechanism_names": list(DEFAULT_MECHANISMS),
        "signature_config": {"minimum_resolved_failures": 3},
        "normalizer": deepcopy(normalizer),
        "analysis_protocol": deepcopy(protocol),
        ANALYSIS_PROTOCOL_DIGEST_FIELD: protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "rollout_schedule_sha256": schedule[SCHEDULE_DIGEST_FIELD],
        "signatures": deepcopy(signatures),
    }
    atlas["atlas_sha256"] = canonical_sha256(atlas, digest_field="atlas_sha256")
    return {
        "signature_artifact": atlas,
        "parent_normalizer": normalizer,
        "analysis_protocol": protocol,
        "policy_parent_manifest": schedule,
        "expected_signature_artifact_sha256": atlas["atlas_sha256"],
        "expected_parent_normalizer_sha256": canonical_sha256(normalizer),
        "expected_analysis_protocol_sha256": protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "expected_policy_parent_sha256": schedule[SCHEDULE_DIGEST_FIELD],
    }


def _build(
    split: dict,
    signatures: list[dict],
    policies: list[dict[str, str]] | None = None,
    parents: dict | None = None,
) -> dict:
    policies = _policies() if policies is None else policies
    parents = _parents(split, signatures, policies) if parents is None else parents
    return build_policy_axis_features(
        signatures,
        split,
        partition="D_atlas",
        policy_checkpoints=policies,
        **parents,
    )


def _validate(
    artifact: dict,
    split: dict,
    signatures: list[dict],
    parents: dict | None = None,
) -> None:
    parents = (
        _parents(split, signatures, artifact["policy_checkpoints"]) if parents is None else parents
    )
    validate_policy_axis_features(artifact, signatures, split, **parents)


def _rehash_atlas(parents: dict) -> None:
    atlas = parents["signature_artifact"]
    atlas["atlas_sha256"] = canonical_sha256(atlas, digest_field="atlas_sha256")
    parents["expected_signature_artifact_sha256"] = atlas["atlas_sha256"]


def _rehash_protocol(parents: dict) -> None:
    protocol = parents["analysis_protocol"]
    protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=ANALYSIS_PROTOCOL_DIGEST_FIELD,
    )
    parents["expected_analysis_protocol_sha256"] = protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
    atlas = parents["signature_artifact"]
    atlas["analysis_protocol"] = deepcopy(protocol)
    atlas[ANALYSIS_PROTOCOL_DIGEST_FIELD] = protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
    _rehash_atlas(parents)


def _rehash_schedule(parents: dict) -> None:
    schedule = parents["policy_parent_manifest"]
    schedule[SCHEDULE_DIGEST_FIELD] = canonical_sha256(
        schedule,
        digest_field=SCHEDULE_DIGEST_FIELD,
    )
    parents["expected_policy_parent_sha256"] = schedule[SCHEDULE_DIGEST_FIELD]
    parents["signature_artifact"]["rollout_schedule_sha256"] = schedule[SCHEDULE_DIGEST_FIELD]
    _rehash_atlas(parents)


def test_primary_axis_is_frozen_early_mid_late_concatenation() -> None:
    split = _split()
    signatures = _signatures(split)

    artifact = _build(split, signatures)

    assert artifact["is_primary_axis"] is True
    assert artifact["axis_method"] == PRIMARY_AXIS_METHOD
    assert artifact["primary_axis_method"] == PRIMARY_AXIS_METHOD
    assert artifact["representation_role"] == "fit"
    assert artifact["common_support_motion_count"] == artifact["motion_count"]
    first_motion = artifact["rows"][0]["motion_key"]
    expected = []
    for policy in _policies():
        expected.extend(
            next(
                row["q"]
                for row in signatures
                if row["motion_key"] == first_motion and row["probe_policy_id"] == policy["id"]
            )
        )
    assert artifact["rows"][0]["failure_q_concat"] == expected
    assert len(artifact["feature_names"]) == 3 * len(DEFAULT_MECHANISMS)
    assert artifact[POLICY_AXIS_DIGEST_FIELD] == canonical_sha256(
        artifact,
        digest_field=POLICY_AXIS_DIGEST_FIELD,
    )
    _validate(artifact, split, signatures)


def test_missing_one_policy_is_missing_on_primary_common_support() -> None:
    split = _split()
    signatures = _signatures(split)
    target = signatures[1]
    target["num_failures"] = 3
    target["num_resolved_failures"] = 2
    target["difficulty"] = 3 / 8
    target["attributed_failure_rate"] = 2 / 8
    target["unresolved_failure_probability"] = 1 / 8
    target["q"] = None
    target["f"] = None

    artifact = _build(split, signatures)
    row = next(item for item in artifact["rows"] if item["motion_key"] == target["motion_key"])

    assert row["common_q_support"] is False
    assert row["missing_policy_ids"] == [target["probe_policy_id"]]
    assert row["failure_q_concat"] is None
    assert row["failure_q_average_sensitivity"] is None
    common = common_support_feature_rows(artifact)
    assert target["motion_key"] not in common["motion_keys"]


def test_policy_order_is_explicit_and_changes_concat_semantics() -> None:
    split = _split()
    signatures = _signatures(split)
    original = _build(split, signatures)
    reversed_axis = _build(split, signatures, list(reversed(_policies())))

    assert reversed_axis["is_primary_axis"] is False
    assert reversed_axis["axis_method"] == SENSITIVITY_AXIS_METHOD
    assert reversed_axis["primary_axis_method"] is None
    assert original["policy_order_sha256"] != reversed_axis["policy_order_sha256"]
    assert original["rows"][0]["failure_q_concat"] != reversed_axis["rows"][0]["failure_q_concat"]
    assert (
        original["rows"][0]["failure_q_average_sensitivity"]
        == reversed_axis["rows"][0]["failure_q_average_sensitivity"]
    )


@pytest.mark.parametrize(
    "policies",
    [
        _policies()[:2],
        [*_policies(), {"id": "pi_lite_extra", "checkpoint_sha256": "4" * 64}],
        list(reversed(_policies())),
    ],
    ids=("two-policy", "four-policy", "reordered-primary-ids"),
)
def test_generic_policy_axes_are_sensitivity_only(policies: list[dict[str, str]]) -> None:
    split = _split()
    signatures = _signatures(split, policies)

    artifact = _build(split, signatures, policies)

    assert artifact["is_primary_axis"] is False
    assert artifact["axis_method"] == SENSITIVITY_AXIS_METHOD
    assert artifact["primary_axis_method"] is None
    assert len(artifact["feature_names"]) == len(policies) * len(DEFAULT_MECHANISMS)


def test_rehashed_generic_axis_cannot_assert_primary_status() -> None:
    split = _split()
    policies = list(reversed(_policies()))
    signatures = _signatures(split, policies)
    artifact = _build(split, signatures, policies)
    forged = deepcopy(artifact)
    forged["is_primary_axis"] = True
    forged["axis_method"] = PRIMARY_AXIS_METHOD
    forged["primary_axis_method"] = PRIMARY_AXIS_METHOD
    forged[POLICY_AXIS_DIGEST_FIELD] = canonical_sha256(
        forged,
        digest_field=POLICY_AXIS_DIGEST_FIELD,
    )

    with pytest.raises(ValueError, match="deep reconstruction"):
        _validate(forged, split, signatures)


def test_policy_checkpoint_hashes_must_exactly_match_parent_axis() -> None:
    split = _split()
    signatures = _signatures(split)
    parents = _parents(split, signatures)
    drifted = deepcopy(_policies())
    drifted[1]["checkpoint_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="signature_artifact probe_policies"):
        _build(split, signatures, drifted, parents)


def test_policy_axis_can_bind_checkpoints_to_parent_atlas() -> None:
    split = _split()
    signatures = _signatures(split)
    parents = _parents(split, signatures)
    parents["policy_parent_manifest"] = parents["signature_artifact"]
    parents["expected_policy_parent_sha256"] = parents["expected_signature_artifact_sha256"]

    artifact = _build(split, signatures, parents=parents)

    assert artifact["policy_parent_kind"] == ATLAS_KIND
    assert artifact["policy_parent_sha256"] == artifact["signature_artifact_sha256"]


def test_exact_motion_policy_cartesian_coverage_is_required() -> None:
    split = _split()
    signatures = _signatures(split)

    with pytest.raises(ValueError, match="Cartesian coverage"):
        _build(split, signatures[:-1])

    duplicated = signatures + [deepcopy(signatures[0])]
    with pytest.raises(ValueError, match="duplicate signature"):
        _build(split, duplicated)


def test_deep_rebuild_rejects_rehashed_feature_substitution() -> None:
    split = _split()
    signatures = _signatures(split)
    artifact = _build(split, signatures)
    forged = deepcopy(artifact)
    forged["rows"][0]["failure_q_concat"][0] += 0.1
    forged[POLICY_AXIS_DIGEST_FIELD] = canonical_sha256(
        forged,
        digest_field=POLICY_AXIS_DIGEST_FIELD,
    )

    with pytest.raises(ValueError, match="deep reconstruction"):
        _validate(forged, split, signatures)


def test_independent_pin_rejects_fully_rehashed_signature_parent() -> None:
    split = _split()
    signatures = _signatures(split)
    genuine = _parents(split, signatures)
    forged_parents = deepcopy(genuine)
    forged_signatures = deepcopy(signatures)
    forged_signatures[0]["q"][0], forged_signatures[0]["q"][1] = (
        forged_signatures[0]["q"][1],
        forged_signatures[0]["q"][0],
    )
    forged_signatures[0]["f"] = [0.5 * value for value in forged_signatures[0]["q"]]
    forged_parents["signature_artifact"]["signatures"] = deepcopy(forged_signatures)
    _rehash_atlas(forged_parents)
    forged_artifact = _build(split, forged_signatures, parents=forged_parents)
    forged_parents["expected_signature_artifact_sha256"] = genuine[
        "expected_signature_artifact_sha256"
    ]

    with pytest.raises(ValueError, match="signature_artifact.*independently expected"):
        _validate(forged_artifact, split, forged_signatures, forged_parents)


def test_independent_pin_rejects_fully_rehashed_normalizer_parent() -> None:
    split = _split()
    signatures = _signatures(split)
    genuine = _parents(split, signatures)
    forged_parents = deepcopy(genuine)
    first_mechanism = DEFAULT_MECHANISMS[0]
    forged_parents["parent_normalizer"]["mechanism_scales"][first_mechanism] += 0.25
    forged_parents["expected_parent_normalizer_sha256"] = canonical_sha256(
        forged_parents["parent_normalizer"]
    )
    forged_parents["signature_artifact"]["normalizer"] = deepcopy(
        forged_parents["parent_normalizer"]
    )
    _rehash_atlas(forged_parents)
    forged_artifact = _build(split, signatures, parents=forged_parents)
    forged_parents["expected_parent_normalizer_sha256"] = genuine[
        "expected_parent_normalizer_sha256"
    ]

    with pytest.raises(ValueError, match="parent_normalizer.*independently expected"):
        _validate(forged_artifact, split, signatures, forged_parents)


def test_independent_pin_rejects_fully_rehashed_analysis_protocol_parent() -> None:
    split = _split()
    signatures = _signatures(split)
    genuine = _parents(split, signatures)
    forged_parents = deepcopy(genuine)
    forged_parents["analysis_protocol"]["fixture_revision"] = 2
    _rehash_protocol(forged_parents)
    forged_artifact = _build(split, signatures, parents=forged_parents)
    forged_parents["expected_analysis_protocol_sha256"] = genuine[
        "expected_analysis_protocol_sha256"
    ]

    with pytest.raises(ValueError, match="analysis_protocol.*independently expected"):
        _validate(forged_artifact, split, signatures, forged_parents)


def test_independent_pin_rejects_fully_rehashed_policy_parent_checkpoint() -> None:
    split = _split()
    signatures = _signatures(split)
    genuine = _parents(split, signatures)
    forged_parents = deepcopy(genuine)
    drifted_policies = deepcopy(_policies())
    drifted_policies[1]["checkpoint_sha256"] = "9" * 64
    forged_parents["policy_parent_manifest"]["probe_policies"] = deepcopy(drifted_policies)
    forged_parents["signature_artifact"]["probe_policies"] = deepcopy(drifted_policies)
    _rehash_schedule(forged_parents)
    forged_artifact = _build(
        split,
        signatures,
        policies=drifted_policies,
        parents=forged_parents,
    )
    forged_parents["expected_policy_parent_sha256"] = genuine["expected_policy_parent_sha256"]

    with pytest.raises(ValueError, match="policy_parent_schedule.*independently expected"):
        _validate(forged_artifact, split, signatures, forged_parents)


def test_signature_f_and_q_consistency_is_required() -> None:
    split = _split()
    signatures = _signatures(split)
    signatures[0]["f"][0] += 0.01

    with pytest.raises(ValueError, match="attributed_failure_rate"):
        _build(split, signatures)


def test_d_test_cannot_be_constructed_as_an_assignment_axis() -> None:
    split = _split()
    parents = _parents(split, [], _policies())
    with pytest.raises(ValueError, match="partition is not allowed"):
        build_policy_axis_features(
            [],
            split,
            partition="D_test",
            policy_checkpoints=_policies(),
            **parents,
        )
