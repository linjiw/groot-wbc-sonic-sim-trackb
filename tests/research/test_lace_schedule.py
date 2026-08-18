from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    REFERENCE_LENGTH_KIND,
    REFERENCE_LENGTH_SCHEMA_VERSION,
    RESAMPLING_RULE,
    reference_num_steps_by_motion,
    sonic_target_frame_count,
    validate_reference_length_inventory,
)
import gear_sonic.research.lace.schedule as schedule_module
from gear_sonic.research.lace.schedule import (
    LEGACY_SCHEDULE_SCHEMA_VERSION,
    QUANTIZATION_RULE_ID,
    RUNTIME_RNG_SEED_DERIVATION_ID,
    SCHEDULE_SCHEMA_VERSION,
    build_rollout_schedule,
    derive_runtime_rng_seed,
    quantize_start_step,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split

_BUILD_SCHEDULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research" / "build_lace_schedule.py"
)
_BUILD_SCHEDULE_SPEC = importlib.util.spec_from_file_location(
    "lace_build_schedule_cli_for_test",
    _BUILD_SCHEDULE_PATH,
)
assert _BUILD_SCHEDULE_SPEC is not None and _BUILD_SCHEDULE_SPEC.loader is not None
_BUILD_SCHEDULE_MODULE = importlib.util.module_from_spec(_BUILD_SCHEDULE_SPEC)
_BUILD_SCHEDULE_SPEC.loader.exec_module(_BUILD_SCHEDULE_MODULE)
build_schedule_main = _BUILD_SCHEDULE_MODULE.main


def _split() -> dict[str, object]:
    motions = [
        {
            "motion_key": f"motion_{index:02d}__A{index:03d}",
            "release_filter_key": f"motion_{index:02d}__A{index:03d}.pkl",
            "source_group_id": f"actor_{index:03d}",
            "robot_path": f"/fixture/motion_{index:02d}__A{index:03d}.pkl",
            "duration_source_frames": 80 + index,
            "stratum": f"duration_{index % 3}",
        }
        for index in range(25)
    ]
    return build_source_disjoint_split(motions, seed=71)


def _atlas_keys(split: dict[str, object]) -> list[str]:
    return sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )


def _policies() -> list[dict[str, str]]:
    # Deliberately not alphabetical: the caller's scientific policy order is
    # part of the frozen artifact and the rollout order must preserve it.
    return [
        {"id": "strong", "checkpoint_sha256": "c" * 64},
        {"id": "weak", "checkpoint_sha256": "a" * 64},
    ]


def _inventory(
    split: dict[str, object],
    *,
    artifact_mode: str = "scientific",
    selected_motion_keys: list[str] | None = None,
) -> dict[str, object]:
    selected = _atlas_keys(split) if selected_motion_keys is None else sorted(selected_motion_keys)
    split_records = {record["motion_key"]: record for record in split["motions"]}
    motions = []
    for index, motion_key in enumerate(selected):
        source_frames = 4 + 2 * index
        source_digest = canonical_sha256({"fixture_motion": motion_key})
        motions.append(
            {
                "motion_key": motion_key,
                "split_robot_path": split_records[motion_key]["robot_path"],
                "source_path": split_records[motion_key]["robot_path"],
                "source_file_sha256": source_digest,
                "source_num_frames": source_frames,
                "frame_axis": "root_trans_offset.shape[0]",
                "source_fps": {"numerator": 30, "denominator": 1},
                "target_num_frames": sonic_target_frame_count(source_frames, 30, 50),
            }
        )
    inventory = {
        "kind": REFERENCE_LENGTH_KIND,
        "schema_version": REFERENCE_LENGTH_SCHEMA_VERSION,
        "artifact_mode": artifact_mode,
        "scientific_use": artifact_mode == "scientific",
        "pilot_status": (
            None
            if artifact_mode == "scientific"
            else "non_scientific_subset_for_contract_or_runtime_validation_only"
        ),
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_atlas",
        "selected_motion_keys": selected,
        "selection_complete_for_d_atlas": artifact_mode == "scientific",
        "motion_count": len(selected),
        "target_fps": 50,
        "runtime_contract": {
            "target_fps": 50,
            "sim_fps": 50,
            "motion_fps_scale": {"numerator": 1, "denominator": 1},
            "max_len": -1,
            "reference_num_steps_equals_target_num_frames": True,
            "float32_runtime_equivalence_scope": (
                "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
                if artifact_mode == "scientific"
                else "not_claimed_for_generic_pilot_rates"
            ),
        },
        "resampling_rule": dict(RESAMPLING_RULE),
        "dataset_provenance": {
            "materialized_manifest": None,
            "materialized_manifest_sha256": None,
            "paired_dataset_sha256": None,
        },
        "source_file_set_sha256": canonical_sha256(
            {
                "files": [
                    {
                        "motion_key": record["motion_key"],
                        "source_file_sha256": record["source_file_sha256"],
                    }
                    for record in motions
                ]
            }
        ),
        "motions": motions,
    }
    inventory[REFERENCE_LENGTH_DIGEST_FIELD] = canonical_sha256(
        inventory,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    validate_reference_length_inventory(inventory, split_manifest=split)
    return inventory


def _build(
    split: dict[str, object],
    *,
    artifact_mode: str = "scientific",
    selected_motion_keys: list[str] | None = None,
) -> dict[str, object]:
    inventory = _inventory(
        split,
        artifact_mode=artifact_mode,
        selected_motion_keys=selected_motion_keys,
    )
    return build_rollout_schedule(
        split,
        reference_length_inventory=inventory,
        probe_policies=_policies(),
        domain_randomization_seeds=[101, 303],
        phase_targets=[
            {"phase_id": "start", "target_fraction": 0.0},
            {"phase_id": "middle", "target_fraction": 0.5},
            {"phase_id": "end", "target_fraction": 1.0},
        ],
        repeats=2,
    )


def _validate_v2(
    manifest: dict[str, object],
    split: dict[str, object],
    *,
    verify_digest: bool = True,
) -> None:
    artifact_mode = str(manifest["artifact_mode"])
    inventory = _inventory(
        split,
        artifact_mode=artifact_mode,
        selected_motion_keys=(
            list(manifest["selected_motion_keys"]) if artifact_mode == "pilot" else None
        ),
    )
    validate_rollout_schedule(
        manifest,
        split_manifest=split,
        reference_length_inventory=inventory,
        verify_digest=verify_digest,
    )


def test_quantization_is_nearest_integer_with_exact_ties_to_lower() -> None:
    # Four frames have valid indices 0..3. The midpoint is exactly 1.5 and
    # therefore resolves to the lower integer under the declared rule.
    assert quantize_start_step(0.0, 4) == 0
    assert quantize_start_step(0.5, 4) == 1
    assert quantize_start_step(0.5001, 4) == 2
    assert quantize_start_step(1.0, 4) == 3

    with pytest.raises(ValueError, match=">= 2"):
        quantize_start_step(0.5, 1)


def test_runtime_rng_seed_derivation_is_frozen_repeat_specific_and_uint32() -> None:
    assert derive_runtime_rng_seed(101, "start", 0) == 1_197_556_082
    assert derive_runtime_rng_seed(101, "start", 1) == 3_244_532_705
    assert derive_runtime_rng_seed(101, "middle", 0) == 2_220_313_824
    assert derive_runtime_rng_seed(101, "start", 0) != derive_runtime_rng_seed(101, "start", 1)
    assert 0 <= derive_runtime_rng_seed(303, "start", 0) <= 0xFFFFFFFF


def test_scientific_schedule_is_deterministic_complete_and_self_bound() -> None:
    split = _split()

    first = _build(split)
    second = _build(split)

    assert first == second
    assert first["artifact_mode"] == "scientific"
    assert first["schema_version"] == SCHEDULE_SCHEMA_VERSION
    assert first["scientific_use"] is True
    assert first["pilot_status"] is None
    assert first["split_sha256"] == split["split_sha256"]
    assert first["split_selection_sha256"] == split["selection_sha256"]
    assert first["selected_motion_keys"] == _atlas_keys(split)
    assert first["selection_complete_for_d_atlas"] is True
    assert first["policy_order"] == ["strong", "weak"]
    inventory = _inventory(split)
    assert first["reference_length_inventory_binding"]["kind"] == REFERENCE_LENGTH_KIND
    assert first["reference_length_inventory_binding"][REFERENCE_LENGTH_DIGEST_FIELD] == (
        inventory[REFERENCE_LENGTH_DIGEST_FIELD]
    )
    assert reference_num_steps_by_motion(inventory) == {
        row["motion_key"]: row["reference_num_steps"] for row in first["reference_num_steps"]
    }
    assert first["runtime_rng_seed_derivation"]["id"] == RUNTIME_RNG_SEED_DERIVATION_ID
    runtime_seeds = [row["runtime_rng_seed"] for row in first["runtime_rng_seed_schedule"]]
    assert len(runtime_seeds) == len(set(runtime_seeds))
    expected_count = len(_atlas_keys(split)) * 2 * 2 * 3 * 2
    assert first["rollout_count"] == expected_count
    assert len({row["rollout_id"] for row in first["rollouts"]}) == expected_count
    assert first["schedule_sha256"] == canonical_sha256(
        first,
        digest_field="schedule_sha256",
    )

    round_tripped = json.loads(json.dumps(first, allow_nan=False))
    validate_rollout_schedule(
        round_tripped,
        split_manifest=split,
        reference_length_inventory=inventory,
    )
    with pytest.raises(ValueError, match="requires split_manifest and reference_length_inventory"):
        validate_rollout_schedule(round_tripped, split_manifest=split)

    tampered_binding = deepcopy(round_tripped)
    tampered_binding["reference_length_inventory_binding"]["target_fps"] = 49
    with pytest.raises(ValueError, match="frozen runtime configuration"):
        validate_rollout_schedule(
            tampered_binding,
            split_manifest=split,
            reference_length_inventory=inventory,
            verify_digest=False,
        )


def test_schedule_materializes_realized_integer_phase_per_motion_and_pairs_policies() -> None:
    split = _split()
    manifest = _build(split)
    midpoint_rows = [row for row in manifest["rollouts"] if row["phase_id"] == "middle"]

    for motion_key in manifest["selected_motion_keys"]:
        rows = [row for row in midpoint_rows if row["motion_key"] == motion_key]
        starts = {(row["start_step"], row["realized_fraction"]) for row in rows}
        assert len(starts) == 1
        start_step, realized_fraction = starts.pop()
        num_steps = rows[0]["reference_num_steps"]
        assert start_step == quantize_start_step(0.5, num_steps)
        assert realized_fraction == pytest.approx(start_step / (num_steps - 1))

    first_motion_rows = [
        row
        for row in manifest["rollouts"]
        if row["motion_key"] == manifest["selected_motion_keys"][0]
    ]
    assert [first_motion_rows[0]["probe_policy_id"], first_motion_rows[12]["probe_policy_id"]] == [
        "strong",
        "weak",
    ]
    assert manifest["quantization"]["id"] == QUANTIZATION_RULE_ID

    runtime_by_condition: dict[tuple[int, str, int], set[int]] = {}
    for row in manifest["rollouts"]:
        condition = (
            row["domain_randomization_seed"],
            row["phase_id"],
            row["repeat_index"],
        )
        runtime_by_condition.setdefault(condition, set()).add(row["runtime_rng_seed"])
    assert all(len(values) == 1 for values in runtime_by_condition.values())
    assert runtime_by_condition[(101, "start", 0)] != runtime_by_condition[(101, "start", 1)]


def test_builder_fails_closed_on_runtime_seed_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = _split()
    monkeypatch.setattr(schedule_module, "derive_runtime_rng_seed", lambda *args: 7)

    with pytest.raises(ValueError, match="runtime RNG seed collision"):
        _build(split)


def test_pilot_requires_explicit_strict_subset_and_is_marked_non_scientific() -> None:
    split = _split()
    atlas_keys = _atlas_keys(split)

    pilot = _build(
        split,
        artifact_mode="pilot",
        selected_motion_keys=atlas_keys[:2],
    )

    assert pilot["scientific_use"] is False
    assert pilot["selection_complete_for_d_atlas"] is False
    assert "non_scientific" in pilot["pilot_status"]
    _validate_v2(pilot, split)

    with pytest.raises(ValueError, match="strict D_atlas subset"):
        _build(split, artifact_mode="pilot")
    with pytest.raises(ValueError, match="strict D_atlas subset"):
        _build(split, artifact_mode="pilot", selected_motion_keys=atlas_keys)
    with pytest.raises(ValueError, match="full D_atlas"):
        _build(split, artifact_mode="scientific", selected_motion_keys=atlas_keys[:2])


def test_builder_rejects_manual_lengths_and_tampered_inventory() -> None:
    split = _split()
    inventory = _inventory(split)
    tampered = deepcopy(inventory)
    tampered["motions"][0]["target_num_frames"] += 1

    with pytest.raises(TypeError, match="reference_num_steps"):
        build_rollout_schedule(
            split,
            reference_num_steps={key: 10 for key in _atlas_keys(split)},  # type: ignore[call-arg]
            probe_policies=_policies(),
            domain_randomization_seeds=[1],
            phase_targets=[{"phase_id": "start", "target_fraction": 0.0}],
            repeats=1,
        )
    with pytest.raises(ValueError, match="exact resampling rule"):
        build_rollout_schedule(
            split,
            reference_length_inventory=tampered,
            probe_policies=_policies(),
            domain_randomization_seeds=[1],
            phase_targets=[{"phase_id": "start", "target_fraction": 0.0}],
            repeats=1,
        )


def test_validator_rejects_missing_cartesian_cell_and_policy_specific_start() -> None:
    split = _split()
    manifest = _build(split)
    missing = deepcopy(manifest)
    missing["rollouts"].pop()
    with pytest.raises(ValueError, match="full Cartesian coverage"):
        _validate_v2(missing, split, verify_digest=False)

    unpaired = deepcopy(manifest)
    first = unpaired["rollouts"][0]
    counterpart = next(
        row
        for row in unpaired["rollouts"]
        if row["motion_key"] == first["motion_key"]
        and row["probe_policy_id"] != first["probe_policy_id"]
        and row["domain_randomization_seed"] == first["domain_randomization_seed"]
        and row["phase_id"] == first["phase_id"]
        and row["repeat_index"] == first["repeat_index"]
    )
    counterpart["start_step"] += 1
    counterpart["realized_fraction"] = counterpart["start_step"] / (
        counterpart["reference_num_steps"] - 1
    )
    with pytest.raises(ValueError, match="common integer starts"):
        _validate_v2(unpaired, split, verify_digest=False)


def test_validator_rejects_checkpoint_binding_and_stale_self_digest() -> None:
    split = _split()
    manifest = _build(split)
    wrong_checkpoint = deepcopy(manifest)
    wrong_checkpoint["rollouts"][0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint does not match"):
        _validate_v2(wrong_checkpoint, split, verify_digest=False)

    wrong_split_binding = deepcopy(manifest)
    wrong_split_binding["rollouts"][0]["split_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="split_sha256 mismatch"):
        _validate_v2(wrong_split_binding, split, verify_digest=False)

    extra_tuple_field = deepcopy(manifest)
    extra_tuple_field["rollouts"][0]["unfrozen_note"] = "not allowed"
    with pytest.raises(ValueError, match="exact tuple schema"):
        _validate_v2(extra_tuple_field, split, verify_digest=False)

    duplicate_runtime_seed = deepcopy(manifest)
    duplicate_runtime_seed["runtime_rng_seed_schedule"][1]["runtime_rng_seed"] = (
        duplicate_runtime_seed["runtime_rng_seed_schedule"][0]["runtime_rng_seed"]
    )
    with pytest.raises(ValueError, match="runtime RNG seed collision"):
        _validate_v2(duplicate_runtime_seed, split, verify_digest=False)

    wrong_runtime_seed = deepcopy(manifest)
    wrong_runtime_seed["rollouts"][0]["runtime_rng_seed"] ^= 1
    with pytest.raises(ValueError, match="runtime_rng_seed violates"):
        _validate_v2(wrong_runtime_seed, split, verify_digest=False)

    stale_digest = deepcopy(manifest)
    stale_digest["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schedule_sha256 mismatch"):
        _validate_v2(stale_digest, split)


def test_validator_retains_read_only_schema_v1_compatibility() -> None:
    split = _split()
    inventory = _inventory(split)
    legacy = deepcopy(_build(split))
    legacy["schema_version"] = LEGACY_SCHEDULE_SCHEMA_VERSION
    legacy.pop("reference_length_inventory_binding")
    legacy["schedule_sha256"] = canonical_sha256(
        legacy,
        digest_field="schedule_sha256",
    )

    validate_rollout_schedule(legacy, split_manifest=split)
    with pytest.raises(ValueError, match="schema-v1 schedules cannot claim"):
        validate_rollout_schedule(
            legacy,
            split_manifest=split,
            reference_length_inventory=inventory,
        )


def test_cpu_cli_writes_canonical_json_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    split = _split()
    inventory = _inventory(split)
    split_path = tmp_path / "split.json"
    inventory_path = tmp_path / "reference-lengths.json"
    spec_path = tmp_path / "schedule-spec.json"
    output_path = tmp_path / "artifacts" / "schedule.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    spec = {
        "schema_version": 2,
        "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "probe_policies": _policies(),
        "domain_randomization_seeds": [101, 303],
        "phase_targets": [
            {"phase_id": "start", "target_fraction": 0.0},
            {"phase_id": "middle", "target_fraction": 0.5},
        ],
        "repeats": 2,
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    arguments = [
        "--split",
        str(split_path),
        "--reference-length-inventory",
        str(inventory_path),
        "--spec",
        str(spec_path),
        "--output",
        str(output_path),
    ]

    assert build_schedule_main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    validate_rollout_schedule(
        artifact,
        split_manifest=split,
        reference_length_inventory=inventory,
    )
    assert summary["schedule_sha256"] == artifact["schedule_sha256"]
    assert output_path.read_text(encoding="utf-8") == (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_schedule_main(arguments)
    assert build_schedule_main([*arguments, "--force"]) == 0


def test_cpu_cli_rejects_unknown_spec_fields(tmp_path: Path) -> None:
    split = _split()
    inventory = _inventory(split)
    split_path = tmp_path / "split.json"
    inventory_path = tmp_path / "reference-lengths.json"
    spec_path = tmp_path / "schedule-spec.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
                "probe_policies": _policies(),
                "domain_randomization_seeds": [101],
                "phase_targets": [{"phase_id": "start", "target_fraction": 0.0}],
                "repeats": 1,
                "typo_field": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        build_schedule_main(
            [
                "--split",
                str(split_path),
                "--reference-length-inventory",
                str(inventory_path),
                "--spec",
                str(spec_path),
                "--output",
                str(tmp_path / "schedule.json"),
            ]
        )
