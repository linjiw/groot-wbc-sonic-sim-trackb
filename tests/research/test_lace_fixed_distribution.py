from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.research.lace.fixed_distribution import (
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    FIXED_DISTRIBUTION_STATE_KEY,
    apply_fixed_distribution,
    assert_fixed_distribution_integrity,
    fixed_distribution_draw_report,
    resolve_fixed_distribution,
    validate_fixed_distribution_command_modes,
)
from gear_sonic.research.lace.intervention_plan import (
    INTERVENTION_PLAN_DIGEST_FIELD,
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    build_intervention_plan,
)
from gear_sonic.research.lace.panels import build_representation_blind_panels
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split
from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _research_inputs() -> tuple[dict, dict, dict, dict]:
    motions = []
    for group_index in range(40):
        for variant in range(1 + group_index % 3):
            motions.append(
                {
                    "motion_key": f"motion_{group_index:02d}_{variant}__A{group_index:03d}",
                    "release_filter_key": f"motion_{group_index:02d}_{variant}.pkl",
                    "source_group_id": f"actor_{group_index:03d}",
                    "duration_source_frames": 120 + group_index * 5 + variant,
                    "stratum": f"duration_q{group_index % 4}",
                }
            )
    split = build_source_disjoint_split(motions, seed=81)
    panel_count = 4
    panel_seed = 27
    panels = build_representation_blind_panels(
        split,
        panel_count=panel_count,
        seed=panel_seed,
    )
    motion_count = sum(record["partition"] == "D_curriculum" for record in split["motions"])
    protocol = {
        "kind": "lace_rq1_intervention_protocol",
        "schema_version": 1,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_curriculum",
        "expected_motion_count": motion_count,
        "expected_panel_count": panel_count,
        "panel_seed": panel_seed,
        "base_distribution_method": "uniform_over_canonically_ordered_motions_v1",
        "sequence_length_agnostic": True,
        "target_kl_nats": 0.03,
        "max_probability_ratio": 3.0,
        "kl_tolerance": 1e-12,
        "probability_tolerance": 1e-12,
        "bisection_iterations": 100,
        "maximum_added_exposure_range": 0.02,
        "interpretation": "fixed-sampler unit-test pre-outcome dose",
    }
    protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )
    plan = build_intervention_plan(split, panels, protocol)
    return split, panels, protocol, plan


def _write_lock(tmp_path: Path) -> tuple[Path, dict]:
    split, panels, protocol, plan = _research_inputs()
    split_path = tmp_path / "inputs/split.json"
    panel_path = tmp_path / "inputs/panels.json"
    protocol_path = tmp_path / "inputs/protocol.json"
    plan_path = tmp_path / "artifacts/intervention-plan.json"
    _write_json(split_path, split)
    _write_json(panel_path, panels)
    _write_json(protocol_path, protocol)
    _write_json(plan_path, plan)
    lock = {
        "kind": "lace_rq1_intervention_plan_artifact_lock",
        "schema_version": 1,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "artifact": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            INTERVENTION_PLAN_DIGEST_FIELD: plan[INTERVENTION_PLAN_DIGEST_FIELD],
        },
        "inputs": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": _file_sha256(split_path),
            "split_sha256": split["split_sha256"],
            "split_selection_sha256": split["selection_sha256"],
            "panel_manifest": str(panel_path),
            "panel_manifest_sha256": _file_sha256(panel_path),
            "panel_sha256": panels["panel_sha256"],
            "intervention_protocol": str(protocol_path),
            "intervention_protocol_sha256": _file_sha256(protocol_path),
            INTERVENTION_PROTOCOL_DIGEST_FIELD: protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD],
        },
    }
    lock_path = tmp_path / "locks/intervention-plan-lock.json"
    _write_json(lock_path, lock)
    return lock_path, plan


def _config(lock_path: Path, distribution_id: str = "base") -> dict:
    return {
        "enable": True,
        "plan_lock_path": str(lock_path),
        "plan_lock_file_sha256": _file_sha256(lock_path),
        "distribution_id": distribution_id,
    }


def _motion_lib(plan: dict) -> SimpleNamespace:
    count = len(plan["motion_keys"])
    return SimpleNamespace(
        use_adaptive_sampling=False,
        all_motions_loaded=True,
        curr_motion_keys=list(plan["motion_keys"]),
        _num_unique_motions=count,
        _num_motions=count,
        _curr_motion_ids=torch.arange(count, dtype=torch.long),
        motion_ids=torch.arange(count, dtype=torch.long),
        _sampling_prob=torch.full((count,), 1.0 / count),
        _sampling_batch_prob=torch.full((count,), 1.0 / count),
        _device="cpu",
        lace_fixed_distribution_binding=None,
    )


def test_apply_base_distribution_binds_deep_lock_and_exact_resident_order(
    tmp_path: Path,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)

    binding = apply_fixed_distribution(motion_lib, _config(lock_path))

    assert motion_lib._sampling_prob.dtype == torch.float64
    assert motion_lib._sampling_prob.tolist() == plan["base_probabilities"]
    assert motion_lib._sampling_batch_prob.tolist() == plan["base_probabilities"]
    assert binding["distribution_id"] == "base"
    assert binding["arm_id"] == "base"
    assert binding["plan_lock_path"] == str(lock_path)
    assert binding["plan_lock_file_sha256"] == _file_sha256(lock_path)
    assert binding[INTERVENTION_PLAN_DIGEST_FIELD] == plan[INTERVENTION_PLAN_DIGEST_FIELD]
    assert binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD] == canonical_sha256(
        binding,
        digest_field=FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    )
    assert assert_fixed_distribution_integrity(motion_lib) is True


def test_disabled_runtime_keeps_legacy_sampler_and_checkpoint_shape() -> None:
    _, _, _, plan = _research_inputs()
    motion_lib = _motion_lib(plan)

    sampled = MotionLibBase.sample_motions(motion_lib, 5)

    assert sampled.shape == (5,)
    assert MotionLibBase.get_state_dict(motion_lib) == {}
    assert not hasattr(motion_lib, "_lace_fixed_distribution_draw_counts")


def test_apply_panel_distribution_uses_frozen_p_plus(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    panel_id = plan["interventions"][0]["panel_id"]

    binding = apply_fixed_distribution(motion_lib, _config(lock_path, panel_id))

    expected = plan["interventions"][0]["intervention"]["p_plus"]
    assert motion_lib._sampling_batch_prob.tolist() == expected
    assert binding["distribution_kind"] == "panel_upweight"
    assert binding["arm_id"] == panel_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plan_lock_file_sha256", "0" * 64, "lock bytes drifted"),
        ("distribution_id", "missing-panel", "not a unique panel"),
    ],
)
def test_resolver_rejects_lock_or_arm_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    config = _config(lock_path)
    config[field] = value

    with pytest.raises(ValueError, match=message):
        resolve_fixed_distribution(config, loaded_motion_keys=plan["motion_keys"])


def test_resolver_rejects_direct_plan_bypass_config(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    config = {
        "enable": True,
        "plan_path": str(tmp_path / "artifacts/intervention-plan.json"),
        "plan_file_sha256": "0" * 64,
        INTERVENTION_PLAN_DIGEST_FIELD: plan[INTERVENTION_PLAN_DIGEST_FIELD],
        "distribution_id": "base",
    }

    with pytest.raises(ValueError, match="config fields"):
        resolve_fixed_distribution(config, loaded_motion_keys=plan["motion_keys"])


def test_resolver_rejects_resident_motion_reorder(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    reordered = list(reversed(plan["motion_keys"]))

    with pytest.raises(ValueError, match="resident motion order"):
        resolve_fixed_distribution(_config(lock_path), loaded_motion_keys=reordered)


@pytest.mark.parametrize(
    ("attribute", "value_factory", "message"),
    [
        ("_num_unique_motions", lambda count: count - 1, "resident motion count"),
        ("_num_motions", lambda count: count - 1, "loaded motion count"),
        ("_curr_motion_ids", lambda count: torch.arange(count).float(), "torch.int64"),
        ("motion_ids", lambda count: torch.roll(torch.arange(count), 1), "identity mapping"),
    ],
)
def test_install_rejects_noncanonical_resident_identity(
    tmp_path: Path,
    attribute: str,
    value_factory,
    message: str,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    setattr(motion_lib, attribute, value_factory(plan["motion_count"]))

    with pytest.raises(ValueError, match=message):
        apply_fixed_distribution(motion_lib, _config(lock_path))


def test_lock_deep_reconstruction_rejects_fully_rehashed_plan_tamper(
    tmp_path: Path,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    plan_path = tmp_path / "artifacts/intervention-plan.json"
    tampered = deepcopy(plan)
    nested = tampered["interventions"][0]["intervention"]
    nested["p_plus"].reverse()
    nested["intervention_probabilities"] = list(nested["p_plus"])
    nested["intervention_sha256"] = canonical_sha256(
        nested,
        digest_field="intervention_sha256",
    )
    tampered[INTERVENTION_PLAN_DIGEST_FIELD] = canonical_sha256(
        tampered,
        digest_field=INTERVENTION_PLAN_DIGEST_FIELD,
    )
    _write_json(plan_path, tampered)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifact"]["file_sha256"] = _file_sha256(plan_path)
    lock["artifact"][INTERVENTION_PLAN_DIGEST_FIELD] = tampered[INTERVENTION_PLAN_DIGEST_FIELD]
    _write_json(lock_path, lock)

    with pytest.raises(ValueError, match="deterministic reconstruction"):
        resolve_fixed_distribution(
            _config(lock_path),
            loaded_motion_keys=plan["motion_keys"],
        )


def test_single_buffer_parser_rejects_duplicate_nested_json_keys(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    plan_path = tmp_path / "artifacts/intervention-plan.json"
    raw = plan_path.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '"kind": "lace_rq1_motion_intervention_plan",',
        '"kind": "lace_rq1_motion_intervention_plan",\n  "kind": "forged",',
        1,
    )
    plan_path.write_text(duplicate, encoding="utf-8")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifact"]["file_sha256"] = _file_sha256(plan_path)
    _write_json(lock_path, lock)

    with pytest.raises(ValueError, match="duplicate key"):
        resolve_fixed_distribution(
            _config(lock_path),
            loaded_motion_keys=plan["motion_keys"],
        )


@pytest.mark.parametrize(
    "active_mode",
    [
        "use_paired_motions",
        "sample_unique_motions",
        "is_evaluating",
        "atlas_probe_mode",
        "multi_object_mode",
    ],
)
def test_command_contract_rejects_all_sampler_bypasses(active_mode: str) -> None:
    modes = {
        "use_paired_motions": False,
        "sample_unique_motions": False,
        "is_evaluating": False,
        "atlas_probe_mode": False,
        "multi_object_mode": False,
    }
    modes[active_mode] = True

    with pytest.raises(ValueError, match="sampler bypass"):
        validate_fixed_distribution_command_modes(**modes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda lib: lib.curr_motion_keys.reverse(), "keys/order"),
        (
            lambda lib: setattr(lib, "_num_unique_motions", lib._num_unique_motions - 1),
            "unique-motion count",
        ),
        (lambda lib: setattr(lib, "_num_motions", lib._num_motions - 1), "loaded-motion count"),
        (lambda lib: setattr(lib, "_curr_motion_ids", lib._curr_motion_ids.float()), "torch.int64"),
        (lambda lib: setattr(lib, "motion_ids", torch.roll(lib.motion_ids, 1)), "identity mapping"),
        (lambda lib: setattr(lib, "_sampling_prob", lib._sampling_prob.float()), "torch.float64"),
        (
            lambda lib: setattr(lib, "_sampling_batch_prob", lib._sampling_batch_prob.float()),
            "torch.float64",
        ),
    ],
)
def test_every_draw_revalidates_resident_identity_and_float64_tensors(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    apply_fixed_distribution(motion_lib, _config(lock_path))
    mutation(motion_lib)

    with pytest.raises(ValueError, match=message):
        MotionLibBase.sample_motions(motion_lib, 2)


def test_binding_is_active_sentinel_even_without_legacy_probability_attribute(
    tmp_path: Path,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    apply_fixed_distribution(motion_lib, _config(lock_path))
    assert not hasattr(motion_lib, "_lace_fixed_distribution_probabilities")
    motion_lib._sampling_batch_prob[0] += 0.01

    with pytest.raises(ValueError, match="sampling_batch_prob.*drifted"):
        MotionLibBase.sample_motions(motion_lib, 1)


def test_missing_binding_cannot_disable_installed_sampler_guard(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    apply_fixed_distribution(motion_lib, _config(lock_path))
    motion_lib.lace_fixed_distribution_binding = None

    with pytest.raises(ValueError, match="configured.*binding is missing"):
        MotionLibBase.sample_motions(motion_lib, 1)


def test_draw_counts_and_report_have_exact_sum_and_identity(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    binding = apply_fixed_distribution(motion_lib, _config(lock_path))

    first = MotionLibBase.sample_motions(motion_lib, 17)
    second = MotionLibBase.sample_motions(motion_lib, 23)
    report = fixed_distribution_draw_report(motion_lib)

    expected_counts = torch.bincount(
        torch.cat((first, second)),
        minlength=plan["motion_count"],
    ).tolist()
    assert report["draw_counts"] == expected_counts
    assert report["total_draw_count"] == 40
    assert report["counter_sum"] == 40
    assert report["motion_keys"] == plan["motion_keys"]
    assert (
        report[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD]
        == binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD]
    )
    assert report[FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD] == canonical_sha256(
        report,
        digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    )
    assert "occupancy_and_optimizer_updates_runner_required" in report["accounting_scope"]


def test_checkpoint_round_trip_requires_exact_arm_plan_lock_config_and_counts(
    tmp_path: Path,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    source = _motion_lib(plan)
    apply_fixed_distribution(source, _config(lock_path))
    MotionLibBase.sample_motions(source, 31)
    state = MotionLibBase.get_state_dict(source)

    restored = _motion_lib(plan)
    apply_fixed_distribution(restored, _config(lock_path))
    MotionLibBase.load_state_dict(restored, state)

    assert fixed_distribution_draw_report(restored) == fixed_distribution_draw_report(source)
    assert state[FIXED_DISTRIBUTION_STATE_KEY]["arm_id"] == "base"
    assert (
        state[FIXED_DISTRIBUTION_STATE_KEY][INTERVENTION_PLAN_DIGEST_FIELD]
        == plan[INTERVENTION_PLAN_DIGEST_FIELD]
    )
    assert state[FIXED_DISTRIBUTION_STATE_KEY]["plan_lock_file_sha256"] == _file_sha256(lock_path)


def test_checkpoint_rejects_different_arm(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    source = _motion_lib(plan)
    apply_fixed_distribution(source, _config(lock_path))
    state = MotionLibBase.get_state_dict(source)
    different = _motion_lib(plan)
    panel_id = plan["interventions"][0]["panel_id"]
    apply_fixed_distribution(different, _config(lock_path, panel_id))

    with pytest.raises(ValueError, match="runtime binding mismatch"):
        MotionLibBase.load_state_dict(different, state)


def test_full_resume_requires_sampler_state_but_model_only_start_is_fresh(
    tmp_path: Path,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    fresh_model_only = _motion_lib(plan)
    apply_fixed_distribution(fresh_model_only, _config(lock_path))
    # Explicit trainer resume=False loads model weights without calling the
    # environment state loader; the new arm therefore starts with zero draws.
    assert fixed_distribution_draw_report(fresh_model_only)["total_draw_count"] == 0

    with pytest.raises(ValueError, match="full RQ1 resume requires"):
        MotionLibBase.load_state_dict(fresh_model_only, {})


def test_checkpoint_counter_tamper_is_rejected(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    source = _motion_lib(plan)
    apply_fixed_distribution(source, _config(lock_path))
    MotionLibBase.sample_motions(source, 9)
    state = MotionLibBase.get_state_dict(source)
    state[FIXED_DISTRIBUTION_STATE_KEY]["draw_counts"][0] += 1
    restored = _motion_lib(plan)
    apply_fixed_distribution(restored, _config(lock_path))

    with pytest.raises(ValueError, match="draw-count sum"):
        MotionLibBase.load_state_dict(restored, state)


def test_full_resume_rejects_extra_environment_sampler_state(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    source = _motion_lib(plan)
    apply_fixed_distribution(source, _config(lock_path))
    state = MotionLibBase.get_state_dict(source)
    state["foreign_sampler_state"] = {"accepted": True}
    restored = _motion_lib(plan)
    apply_fixed_distribution(restored, _config(lock_path))

    with pytest.raises(ValueError, match="must contain exactly"):
        MotionLibBase.load_state_dict(restored, state)


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("load_all_motions", (), {}),
        ("load_motions_for_training", (), {"max_num_seqs": 4}),
        ("load_motions_for_evaluation", (), {"start_idx": 0}),
        ("load_motions", (), {"random_sample": False, "num_motions_to_load": 4}),
    ],
)
def test_reload_apis_are_guarded_after_installation(
    tmp_path: Path,
    method_name: str,
    args: tuple,
    kwargs: dict,
) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    apply_fixed_distribution(motion_lib, _config(lock_path))

    with pytest.raises(RuntimeError, match="forbidden after an RQ1 fixed distribution"):
        getattr(MotionLibBase, method_name)(motion_lib, *args, **kwargs)


def test_adaptive_bin_draw_api_is_guarded_after_installation(tmp_path: Path) -> None:
    lock_path, plan = _write_lock(tmp_path)
    motion_lib = _motion_lib(plan)
    apply_fixed_distribution(motion_lib, _config(lock_path))

    with pytest.raises(RuntimeError, match="adaptive bin sampling is forbidden"):
        MotionLibBase.sample_motion_ids_and_time_steps(motion_lib, 2)


def test_scale512_production_lock_deep_resolves_when_data_is_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock_path = (
        repo_root
        / "configs/research/lace/bones_seed_official_scale512_rq1_intervention_plan_lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    plan_path = Path(lock["artifact"]["path"])
    if not plan_path.is_file():
        pytest.skip("materialized scale-512 intervention plan is not available")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    probabilities, binding = resolve_fixed_distribution(
        _config(lock_path),
        loaded_motion_keys=plan["motion_keys"],
    )

    assert len(probabilities) == 233
    assert binding["motion_count"] == 233
    assert binding[INTERVENTION_PLAN_DIGEST_FIELD] == (
        "e726a7f202aa9d9c418ee11950c8a80f767fe4621ebba281c10f31fb225b07e8"
    )


def test_fixed_distribution_hydra_contract_is_inert_by_default(tmp_path: Path) -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[2]
    lock_path, plan = _write_lock(tmp_path)
    panel_id = plan["interventions"][0]["panel_id"]
    overrides = [
        "+exp=manager/universal_token/g1_only/lace_lite_s",
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution.enable=true",
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution.plan_lock_path="
        + str(lock_path),
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution."
        "plan_lock_file_sha256=" + _file_sha256(lock_path),
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution."
        "distribution_id=" + panel_id,
    ]
    with initialize_config_dir(
        config_dir=str(repo_root / "gear_sonic/config"),
        version_base="1.1",
    ):
        enabled = compose(config_name="base", overrides=overrides)
    block = enabled.manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution
    assert block.enable is True
    assert block.plan_lock_path == str(lock_path)
    assert block.distribution_id == panel_id

    with initialize_config_dir(
        config_dir=str(repo_root / "gear_sonic/config"),
        version_base="1.1",
    ):
        release_default = compose(
            config_name="base",
            overrides=["+exp=manager/universal_token/g1_only/lace_lite_s"],
        )
    default_block = (
        release_default.manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution
    )
    assert default_block.enable is False
    assert default_block.plan_lock_path is None
    assert default_block.distribution_id is None
