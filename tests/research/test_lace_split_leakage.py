from __future__ import annotations

from copy import deepcopy

import pytest

from gear_sonic.research.lace.schema import (
    canonical_sha256,
    split_selection_sha256,
    validate_split_manifest,
)
from gear_sonic.research.lace.split import build_source_disjoint_split, derive_source_group_id


def _cohort(group_count: int = 25) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for actor_index in range(group_count):
        actor = f"A{actor_index:03d}"
        for mirror in (False, True):
            suffix = "_M" if mirror else ""
            records.append(
                {
                    "motion_key": f"walk_{actor_index:03d}__{actor}{suffix}",
                    "release_filter_key": f"240101/walk_{actor_index:03d}__{actor}{suffix}.pkl",
                    "category": "locomotion" if actor_index % 2 else "gesture",
                    "stratum": f"category_{actor_index % 4}",
                    "duration_source_frames": 100 + actor_index,
                    "is_mirror": mirror,
                }
            )
    return records


def test_source_group_uses_actor_provenance_and_groups_mirrors() -> None:
    original = {"motion_key": "walk__A057", "release_filter_key": "211117/walk__A057.pkl"}
    mirrored = {
        "motion_key": "walk__A057_M",
        "release_filter_key": "211117/walk__A057_M.pkl",
    }

    assert derive_source_group_id(original) == "bones_actor:A057"
    assert derive_source_group_id(mirrored) == "bones_actor:A057"


def test_split_is_deterministic_source_disjoint_and_valid() -> None:
    first = build_source_disjoint_split(_cohort(), seed=17, dataset={"name": "fixture"})
    second = build_source_disjoint_split(_cohort(), seed=17, dataset={"name": "fixture"})

    assert first == second
    assert first["selection_sha256"] == split_selection_sha256(first)
    assert first["split_sha256"] == canonical_sha256(first, digest_field="split_sha256")
    validate_split_manifest(first)
    assert all(summary["motion_count"] > 0 for summary in first["partition_summary"].values())


def test_validator_rejects_source_group_leakage_even_with_fresh_digest() -> None:
    manifest = build_source_disjoint_split(_cohort(), seed=17)
    leaked = deepcopy(manifest)
    first_group = leaked["motions"][0]["source_group_id"]
    same_group = [
        record for record in leaked["motions"] if record["source_group_id"] == first_group
    ]
    assert len(same_group) == 2
    original_partition = same_group[0]["partition"]
    replacement = next(name for name in leaked["partition_summary"] if name != original_partition)
    same_group[0]["partition"] = replacement
    leaked["split_sha256"] = canonical_sha256(leaked, digest_field="split_sha256")

    with pytest.raises(ValueError, match="leaks across"):
        validate_split_manifest(leaked)


def test_source_group_derivation_refuses_semantic_fallback() -> None:
    with pytest.raises(ValueError, match="cannot derive"):
        derive_source_group_id({"motion_key": "walk_forward"})
