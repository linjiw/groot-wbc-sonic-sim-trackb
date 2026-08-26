"""Tests for the additive LFH identity and keypoint-DCS contract."""

from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.dataset_generation.sweepcf_coverage import (
    UNKNOWN,
    binding_station_offset_mm,
    constraint_axis,
    coordinate_bucket,
    family_identity,
    independent_verified_sources,
    margin_bucket,
    motion_gate_fields,
    semantic_keypoint,
    target_bins,
    target_occupancy_episodes,
    verified_variant_keys,
)
from gear_sonic.dataset_generation.swept_volume import G1_COLLISION_CAPSULES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((REPO_ROOT / "configs/research/sweepcf_dcs_v1.json").read_text())


def test_every_collision_bearing_link_has_one_semantic_group():
    groups = {body: semantic_keypoint(body) for body in G1_COLLISION_CAPSULES}
    assert len(groups) == 14
    assert UNKNOWN not in groups.values()
    assert groups["torso_link"] == "head_torso"
    assert groups["left_ankle_roll_link"] == "foot_left"


def test_two_scene_variants_share_one_manifest_backed_causal_source():
    first = family_identity({"family_id": "cf_005_056"}, "duck_002")
    second = family_identity({"family_id": "cf_005_056"}, "duck_003")
    assert first.source_family_id == second.source_family_id == "cf_005_056"
    assert first.variant_id == "duck_002"
    assert second.variant_id == "duck_003"


def test_a_directory_name_is_not_promoted_to_causal_identity_without_evidence():
    identity = family_identity({}, "n_013_wall_chest_left")
    assert identity.source_family_id == UNKNOWN
    assert identity.variant_id == "n_013_wall_chest_left"
    assert identity.source_identity_status == UNKNOWN


def test_the_independent_family_count_deduplicates_verified_variants():
    rows = [
        {"status": "verified", "source_family_id": "cf_005_056"},
        {"status": "verified", "source_family_id": "cf_005_056"},
        {"status": "verified", "source_family_id": "mf_005_c08"},
        {
            "status": "verified",
            "evidence_valid": "0",
            "source_family_id": "displaced_variant",
        },
        {"status": "not_verified", "source_family_id": "candidate"},
        {"status": "verified", "source_family_id": UNKNOWN},
    ]
    assert independent_verified_sources(rows) == {"cf_005_056", "mf_005_c08"}


def test_constraint_axis_uses_manifest_evidence_and_keeps_other_cases_unknown():
    assert constraint_axis({"regime": "overhead"}) == "overhead"
    assert constraint_axis({"obstacle": "wall"}) == "lateral_gap"
    assert constraint_axis({}, 1.25) == "overhead"
    assert constraint_axis({}) == UNKNOWN


def test_a_loaded_shelf_in_the_wrong_frame_is_measured_not_re_admitted():
    meta = {"shelf_center_xy_m": [1.54, -0.15]}
    box = (-0.71, -1.65, 1.2, -0.21, 1.35, 1.3)
    assert binding_station_offset_mm(meta, box) == 2000.0
    assert binding_station_offset_mm({}, box) is None


def test_signed_margin_buckets_keep_contact_at_zero():
    assert margin_bucket(-0.01, CONFIG) == "strike"
    assert margin_bucket(0.0, CONFIG) == "strike"
    assert margin_bucket(10.0, CONFIG) == "clear_0_10"
    assert margin_bucket(10.01, CONFIG) == "clear_10_25"
    assert margin_bucket("", CONFIG) == UNKNOWN


def test_coordinate_buckets_have_deterministic_half_open_edges():
    assert coordinate_bucket("overhead", 1.1999, CONFIG) == "1.1_1.2"
    assert coordinate_bucket("overhead", 1.2, CONFIG) == "1.2_1.3"
    assert coordinate_bucket("lateral_gap", "", CONFIG) == UNKNOWN


def test_only_empty_room_evidence_populates_controller_trackability():
    screen = {"csv": "005_a_walk_s0.csv"}
    gate = {
        5: {
            "reference_semantic_valid": True,
            "self_collision_free": True,
            "worth_a_rollout": True,
        }
    }
    nonempty = {
        "005_a_walk": {
            "tracked": True,
            "empty_room_trackable": None,
            "empty_room_rollouts": 0,
        }
    }
    fields = motion_gate_fields(screen, gate, nonempty)
    assert fields["reference_semantic_valid"] == 1
    assert fields["controller_trackable"] == ""
    assert fields["controller_trackability_evidence"] == "accepted_nonempty_only"


def test_target_space_contains_only_reviewed_v1_operator_axis_pairs():
    rows = target_bins(CONFIG)
    pairs = {(row["operator"], row["constraint_axis"], row["edit_behaviour_class"]) for row in rows}
    assert pairs == {
        ("local_crouch", "overhead", "crouch"),
        ("local_arm_tuck", "lateral_gap", "arms"),
    }
    assert all(row["constraint_axis"] != "ground_support" for row in rows)


def test_verified_variant_keys_do_not_promote_refused_siblings():
    families = [
        {
            "status": "verified",
            "evidence_valid": "1",
            "source_family_id": "source_a",
            "variant_id": "variant_good",
        },
        {
            "status": "not_verified",
            "evidence_valid": "1",
            "source_family_id": "source_a",
            "variant_id": "variant_bad",
        },
    ]
    assert verified_variant_keys(families) == {("source_a", "variant_good")}


def test_v2_target_occupancy_requires_verified_canonical_cells():
    families = [
        {
            "status": "verified",
            "evidence_valid": "1",
            "source_family_id": "source_a",
            "variant_id": "variant_good",
        }
    ]
    episodes = [
        {
            "source_family_id": "source_a",
            "variant_id": "variant_good",
            "cell_role": "adapted_hard",
        },
        {
            "source_family_id": "source_a",
            "variant_id": "variant_good",
            "cell_role": "probe",
        },
        {
            "source_family_id": "source_a",
            "variant_id": "variant_bad",
            "cell_role": "adapted_hard",
        },
    ]
    selected = target_occupancy_episodes(
        episodes,
        families,
        {"target_occupancy_policy": "verified_canonical_family_cells"},
    )
    assert selected == episodes[:1]


def test_v2_feasibility_mask_excludes_crouch_shoulder_targets():
    config = json.loads((REPO_ROOT / "configs/research/sweepcf_dcs_v2.json").read_text())
    rows = target_bins(config)
    assert len(rows) == 120
    assert {row["binding_keypoint"] for row in rows if row["operator"] == "local_crouch"} == {
        "head_torso"
    }
