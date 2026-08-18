from __future__ import annotations

from copy import deepcopy

import pytest

from gear_sonic.research.lace.panels import (
    build_representation_blind_panels,
    validate_panel_manifest,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split


def _split() -> dict:
    motions = []
    for group_index in range(30):
        for variant in range(1 + group_index % 2):
            motions.append(
                {
                    "motion_key": f"motion_{group_index:02d}_{variant}__A{group_index:03d}",
                    "release_filter_key": f"motion_{group_index:02d}_{variant}.pkl",
                    "source_group_id": f"actor_{group_index:03d}",
                    "duration_source_frames": 100 + group_index * 7 + variant,
                    "stratum": f"duration_q{group_index % 3}",
                }
            )
    return build_source_disjoint_split(motions, seed=41)


def test_panels_are_deterministic_representation_blind_and_cover_partition() -> None:
    split = _split()

    first = build_representation_blind_panels(split, panel_count=4, seed=7)
    second = build_representation_blind_panels(split, panel_count=4, seed=7)

    assert first == second
    assert first["representation_blind"] is True
    assert "mechanism_scores" not in first["assignment_inputs"]
    validate_panel_manifest(first, split_manifest=split)
    expected = {
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_curriculum"
    }
    actual = {motion for panel in first["panels"] for motion in panel["motion_keys"]}
    assert actual == expected
    durations = [panel["duration_source_frames"] for panel in first["panels"]]
    motion_counts = [panel["motion_count"] for panel in first["panels"]]
    assert max(durations) / min(durations) < 1.15
    assert max(motion_counts) - min(motion_counts) <= 1


def test_source_groups_are_never_split_between_panels() -> None:
    split = _split()
    manifest = build_representation_blind_panels(split, panel_count=4, seed=7)
    group_to_panel = {}
    for panel in manifest["panels"]:
        for source_group in panel["source_group_ids"]:
            assert source_group not in group_to_panel
            group_to_panel[source_group] = panel["panel_id"]


def test_panel_validator_rejects_duplicate_motion_and_digest_tampering() -> None:
    split = _split()
    manifest = build_representation_blind_panels(split, panel_count=4, seed=7)
    duplicate = deepcopy(manifest)
    duplicate["panels"][1]["motion_keys"].append(duplicate["panels"][0]["motion_keys"][0])
    duplicate["panels"][1]["motion_count"] += 1
    with pytest.raises(ValueError, match="multiple panels"):
        validate_panel_manifest(duplicate, split_manifest=split, verify_digest=False)

    tampered = deepcopy(manifest)
    tampered["seed"] += 1
    with pytest.raises(ValueError, match="panel_sha256 mismatch"):
        validate_panel_manifest(tampered, split_manifest=split)


def test_panel_builder_rejects_missing_duration() -> None:
    split = _split()
    target = next(record for record in split["motions"] if record["partition"] == "D_curriculum")
    del target["duration_source_frames"]
    split["split_sha256"] = canonical_sha256(split, digest_field="split_sha256")

    with pytest.raises(ValueError, match="duration_source_frames"):
        build_representation_blind_panels(split, panel_count=4, seed=7)
