from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.research.lace.interventions import solve_fixed_budget_intervention
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.transfer_features import (
    TRANSFER_FEATURE_KIND,
    build_signed_exposure_transfer_features,
)


def _intervention() -> dict:
    return solve_fixed_budget_intervention(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([True, True, False, False]),
        target_kl=0.05,
        max_probability_ratio=3.0,
    )


def _build(intervention: dict | None = None) -> dict:
    return build_signed_exposure_transfer_features(
        _intervention() if intervention is None else intervention,
        panel_id="source_panel_00",
        motion_keys=["m0", "m1", "m2", "m3"],
        source_memberships={
            "m0": [1.0, 0.0],
            "m1": [0.75, 0.25],
            "m2": [0.25, 0.75],
            "m3": [0.0, 1.0],
        },
        target_memberships={
            "target_b": [0.1, 0.9],
            "target_a": [0.9, 0.1],
        },
        target_source_groups={
            "target_a": "actor_a",
            "target_b": "actor_b",
        },
        representation_id="failure_k2",
        representation_artifact_sha256="a" * 64,
        component_names=["contact", "actuation"],
    )


def test_signed_features_include_removed_exposure_and_are_self_bound() -> None:
    intervention = _intervention()
    result = _build(intervention)
    source = np.asarray(
        [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
    )
    delta = np.asarray(intervention["delta_p"])
    expected_shift = delta @ source

    assert result["kind"] == TRANSFER_FEATURE_KIND
    assert result["removed_exposure_is_included"] is True
    assert result["delta_membership_centroid"] == pytest.approx(expected_shift)
    assert sum(result["delta_membership_centroid"]) == pytest.approx(0.0, abs=1e-12)
    assert result["target_motion_order"] == ["target_a", "target_b"]
    assert result["transfer_feature_sha256"] == canonical_sha256(
        result,
        digest_field="transfer_feature_sha256",
    )

    # The treatment is not the upweighted panel centroid: the signed shift also
    # accounts for the mass removed from m2 and m3.
    assert result["delta_membership_centroid"] != pytest.approx(result["panel_membership_centroid"])


def test_interactions_and_distance_change_match_the_frozen_formula() -> None:
    result = _build()
    base = np.asarray(result["base_membership_centroid"])
    plus = np.asarray(result["intervention_membership_centroid"])
    shift = np.asarray(result["delta_membership_centroid"])
    target = result["targets"][0]
    target_membership = np.asarray(target["target_membership"])

    expected_interactions = shift * (target_membership - base)
    assert target["feature_vector"] == pytest.approx(expected_interactions)
    assert target["signed_exposure_alignment"] == pytest.approx(expected_interactions.sum())
    assert target["signed_squared_l2_change"] == pytest.approx(
        np.square(plus - target_membership).sum() - np.square(base - target_membership).sum()
    )
    assert len(target["feature_vector"]) == result["representation_dimension"] == 2


def test_same_k_contract_accepts_soft_or_one_hot_memberships() -> None:
    result = _build()

    assert result["matched_k_required_across_candidate_representations"] is True
    assert result["component_names"] == ["contact", "actuation"]
    assert result["feature_names"] == [
        "failure_k2:signed_exposure_interaction:contact",
        "failure_k2:signed_exposure_interaction:actuation",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda kwargs: kwargs["source_memberships"].pop("m3"),
            "keys must exactly match",
        ),
        (
            lambda kwargs: kwargs["source_memberships"].update(m0=[0.2, 0.2]),
            "sum to one",
        ),
        (
            lambda kwargs: kwargs["target_memberships"].update(target_a=[1.0, 0.0, 0.0]),
            r"shape \[2\]",
        ),
        (
            lambda kwargs: kwargs["target_source_groups"].pop("target_b"),
            "keys must exactly match",
        ),
        (
            lambda kwargs: kwargs.update(component_names=["only_one"]),
            "must contain 2",
        ),
    ],
)
def test_assignment_contracts_fail_closed(mutation, message: str) -> None:
    kwargs = {
        "panel_id": "source_panel_00",
        "motion_keys": ["m0", "m1", "m2", "m3"],
        "source_memberships": {
            "m0": [1.0, 0.0],
            "m1": [0.75, 0.25],
            "m2": [0.25, 0.75],
            "m3": [0.0, 1.0],
        },
        "target_memberships": {
            "target_a": [0.9, 0.1],
            "target_b": [0.1, 0.9],
        },
        "target_source_groups": {
            "target_a": "actor_a",
            "target_b": "actor_b",
        },
        "representation_id": "failure_k2",
        "representation_artifact_sha256": "a" * 64,
        "component_names": ["contact", "actuation"],
    }
    mutation(kwargs)

    with pytest.raises(ValueError, match=message):
        build_signed_exposure_transfer_features(_intervention(), **kwargs)


def test_tampered_intervention_or_delta_fails_closed() -> None:
    tampered = deepcopy(_intervention())
    tampered["delta_p"][0] += 0.01
    with pytest.raises(ValueError, match="intervention_sha256 mismatch"):
        _build(tampered)

    inconsistent = deepcopy(_intervention())
    inconsistent["delta_p"] = [0.0, 0.0, 0.0, 0.0]
    inconsistent["intervention_sha256"] = canonical_sha256(
        inconsistent,
        digest_field="intervention_sha256",
    )
    with pytest.raises(ValueError, match="does not equal"):
        _build(inconsistent)
