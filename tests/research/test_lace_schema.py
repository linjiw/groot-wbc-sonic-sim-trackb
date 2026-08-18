from __future__ import annotations

from copy import deepcopy

import pytest

from gear_sonic.research.lace.schema import (
    ATLAS_KIND,
    canonical_sha256,
    validate_atlas_manifest,
)


def _atlas() -> dict[str, object]:
    atlas: dict[str, object] = {
        "kind": ATLAS_KIND,
        # Historical contract-smoke evidence remains read-only schema v4;
        # schema v5 is reserved for externally protocol-bound scientific data.
        "schema_version": 4,
        "artifact_mode": "contract_smoke",
        "scientific_use": False,
        "split_sha256": "a" * 64,
        "split_selection_sha256": "d" * 64,
        "mechanism_names": ["contact", "slip"],
        "selected_motion_keys": ["jump__A001"],
        "motion_count": 1,
        "rollout_count": 4,
        "selection_complete_for_d_atlas": False,
        "rollout_schedule": [
            {"domain_randomization_seed": 11, "initial_phase": 0.0, "repeat_index": 0},
            {"domain_randomization_seed": 12, "initial_phase": 0.0, "repeat_index": 0},
            {"domain_randomization_seed": 11, "initial_phase": 0.5, "repeat_index": 1},
            {"domain_randomization_seed": 12, "initial_phase": 0.5, "repeat_index": 1},
        ],
        "signature_config": {
            "minimum_resolved_failures": 1,
            "mechanism_dirichlet_prior": 0.5,
            "credible_interval_level": 0.95,
        },
        "normalizer": {
            "kind": "identity_contract_smoke",
            "frozen": True,
            "fit_partition": "D_atlas",
            "mechanism_scales": {"contact": 1.0, "slip": 1.0},
        },
        "probe_policies": [{"id": "lite_early", "checkpoint_sha256": "b" * 64}],
        "domain_randomization_seeds": [11, 12],
        "signatures": [
            {
                "motion_key": "jump__A001",
                "probe_policy_id": "lite_early",
                "num_rollouts": 4,
                "num_failures": 2,
                "num_resolved_failures": 2,
                "difficulty": 0.5,
                "attributed_failure_rate": 0.5,
                "unresolved_failure_probability": 0.0,
                "q": [0.25, 0.75],
                "f": [0.125, 0.375],
                "mechanism_posterior_alpha": [1.0, 2.0],
                "q_posterior_mean": [1.0 / 3.0, 2.0 / 3.0],
                "q_credible_interval": [[0.0, 1.0], [0.0, 1.0]],
                "f_posterior_mean": [1.0 / 6.0, 1.0 / 3.0],
                "f_all_failure_mar_sensitivity": [0.125, 0.375],
            }
        ],
    }
    atlas["atlas_sha256"] = canonical_sha256(atlas)
    return atlas


def test_atlas_schema_accepts_factorized_signature_and_self_digest() -> None:
    validate_atlas_manifest(_atlas())


def test_atlas_schema_rejects_stale_digest() -> None:
    atlas = _atlas()
    atlas["domain_randomization_seeds"] = [99]

    with pytest.raises(ValueError, match="atlas_sha256 mismatch"):
        validate_atlas_manifest(atlas)


def test_atlas_schema_rejects_difficulty_leaking_into_q_normalization() -> None:
    atlas = deepcopy(_atlas())
    atlas["signatures"][0]["q"] = [0.125, 0.375]
    atlas["atlas_sha256"] = canonical_sha256(atlas, digest_field="atlas_sha256")

    with pytest.raises(ValueError, match="q must sum to one"):
        validate_atlas_manifest(atlas)
