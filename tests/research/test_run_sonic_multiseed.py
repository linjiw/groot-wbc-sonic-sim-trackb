from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.research.run_sonic_multiseed import (
    _verify_flat_paired_inventory,
    _verify_training_runtime_inputs,
    finalize_multiseed_activation,
    preflight_multiseed_experiment,
    render_spec_for_seed,
    run_multiseed_experiment,
    validate_m5_multiseed_contract,
)
from scripts.research.run_sonic_paired_experiment import (
    _verify_activation_launch_runtime,
)
from scripts.research.summarize_sampler_telemetry import summarize_telemetry_logs


def _summary(path: Path, *, reward: float, mpjpe: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "train": {
                    "kind": "sonic_training_log",
                    "ok": True,
                    "learning_iteration": 50,
                    "mean_rewards": reward,
                    "total_timesteps": 9600,
                    "traceback_count": 0,
                    "log_path": "train.log",
                },
                "eval": {
                    "kind": "sonic_eval_log",
                    "ok": True,
                    "all": {"mpjpe_g": mpjpe, "mpjpe_l": 18.0, "mpjpe_pa": 12.0},
                    "terminated_final": 0,
                    "success_rate_final": 1.0,
                    "traceback_count": 0,
                    "log_path": "eval.log",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _template(tmp_path: Path, seeds: list[int]) -> dict:
    variants = []
    for name, mpjpe_offset in [("uniform_sampling_micro", 0.0), ("adaptive_sampling_micro", 0.5)]:
        for seed in seeds:
            _summary(
                tmp_path / f"summaries/{name}_seed{seed}.json",
                reward=0.9,
                mpjpe=30.0 + seed + mpjpe_offset,
            )
        variants.append(
            {
                "name": name,
                "summary_json": str(tmp_path / "summaries" / f"{name}_seed{{seed}}.json"),
                "train_command": f"python gear_sonic/train_agent_trl.py seed={{seed}} exp_var={name}_seed{{seed}}",
                "eval_command": f"python gear_sonic/eval_agent_trl.py seed={{seed}} exp_var={name}_seed{{seed}}",
                "interpretation": f"multiseed_fixture_{name}",
            }
        )
    return {
        "experiment_group": "multiseed_fixture",
        "hypothesis": "fixture hypothesis",
        "seed": 0,
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "git_commit": "abc1234",
        "variants": variants,
    }


def test_render_spec_substitutes_seed_everywhere() -> None:
    template = {
        "seed": 0,
        "variants": [{"train_command": "train seed={seed} exp_var=v_seed{seed}"}],
        "nested": {"list": ["a_{seed}", 3]},
    }

    spec = render_spec_for_seed(template, 2)

    assert spec["seed"] == 2
    assert spec["variants"][0]["train_command"] == "train seed=2 exp_var=v_seed2"
    assert spec["nested"]["list"] == ["a_2", 3]
    # Template itself is untouched.
    assert "{seed}" in template["variants"][0]["train_command"]


def test_multiseed_dry_run_materializes_seeds_and_aggregate(tmp_path: Path) -> None:
    seeds = [0, 1, 2]
    template = _template(tmp_path, seeds)
    output_dir = tmp_path / "run"

    run = run_multiseed_experiment(
        template,
        seeds=seeds,
        output_dir=output_dir,
        dry_run=True,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert run["dry_run"] is True
    assert run["seeds"] == seeds
    for seed in seeds:
        assert (output_dir / f"seed{seed}" / "spec.json").exists()
        assert (output_dir / f"seed{seed}" / "comparison.json").exists()
        spec = json.loads((output_dir / f"seed{seed}" / "spec.json").read_text(encoding="utf-8"))
        assert spec["seed"] == seed
        assert f"seed{seed}" in spec["variants"][0]["train_command"]
    aggregate = json.loads((output_dir / "aggregate_comparison.json").read_text(encoding="utf-8"))
    assert aggregate["comparison_count"] == 3
    assert aggregate["seeds"] == seeds
    # Fixture puts adaptive 0.5 above uniform on every seed.
    effect = aggregate["effect_summary"]
    assert effect["mean_delta_a_minus_b"] == pytest.approx(0.5)
    assert effect["passes_preregistered_effect_gate"] is False
    assert (output_dir / "aggregate_table.md").exists()
    assert (output_dir / "multiseed_run.json").exists()


def test_multiseed_rejects_unsupported_effect_metric_before_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="unsupported effect metric"):
        run_multiseed_experiment(
            _template(tmp_path, [0]),
            seeds=[0],
            output_dir=output_dir,
            dry_run=True,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric="eval.all.not_a_metric",
            a_minus_b_threshold=0.0,
            min_improved_seeds=1,
        )

    assert not output_dir.exists()


def test_multiseed_rejects_template_without_seed_placeholder(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])
    for variant in template["variants"]:
        for key in ("summary_json", "train_command", "eval_command"):
            variant[key] = variant[key].replace("{seed}", "0")

    with pytest.raises(ValueError, match="placeholder"):
        run_multiseed_experiment(
            template,
            seeds=[0, 1],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )


def test_multiseed_rejects_variant_names_absent_from_template(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])

    with pytest.raises(ValueError, match="not found in spec template"):
        run_multiseed_experiment(
            template,
            seeds=[0],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="error_ema_micro",  # template has adaptive/uniform only
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )


def test_multiseed_invalidates_run_when_a_seed_yields_no_comparison(tmp_path: Path) -> None:
    seeds = [0, 1]
    template = _template(tmp_path, seeds)
    # Break seed 1: point its summaries at nonexistent files so no manifests emerge.
    for name in ("uniform_sampling_micro", "adaptive_sampling_micro"):
        (tmp_path / "summaries" / f"{name}_seed1.json").unlink()

    run = run_multiseed_experiment(
        template,
        seeds=seeds,
        output_dir=tmp_path / "run",
        dry_run=True,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert run["missing_comparison_seeds"] == [1]
    assert run["ok_for_causal_comparison"] is False


def test_multiseed_rejects_duplicate_seeds(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])

    with pytest.raises(ValueError, match="duplicate"):
        run_multiseed_experiment(
            template,
            seeds=[0, 0],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )


def _make_activation_gated_template(tmp_path: Path, seeds: list[int]) -> dict:
    template = _template(tmp_path, seeds)
    for seed in seeds:
        _summary(
            tmp_path / f"summaries/adaptive_sampling_micro_seed{seed}.json",
            reward=1.0,
            mpjpe=29.0 + seed,
        )
    template.update(
        {
            "requires_activation_gate": True,
            "activation_arm": "m5_l",
            "variant_a": "adaptive_sampling_micro",
            "variant_b": "uniform_sampling_micro",
            "dataset_manifest_json": "DATASET_MANIFEST_REPLACE_ME.json",
            "dataset_manifest_sha256": "DATASET_MANIFEST_SHA256_REPLACE_ME",
            "difficulty_ranking_json": "DIFFICULTY_RANKING_REPLACE_ME.json",
            "difficulty_ranking_sha256": "DIFFICULTY_RANKING_SHA256_REPLACE_ME",
            "sim_d1_classification_json": "SIM_D1_CLASSIFICATION_REPLACE_ME.json",
            "sim_d1_classification_sha256": "SIM_D1_CLASSIFICATION_SHA256_REPLACE_ME",
            "sim_d1_source_checkpoint_sha256": ("SIM_D1_SOURCE_CHECKPOINT_SHA256_REPLACE_ME"),
            "training_python_executable": "/resolved/training/python",
            "training_initialization_checkpoint": "release/last.pt",
            "expected_training_iterations": 50,
            "training_num_envs": 2,
        }
    )
    nominal_eval = (
        "/resolved/training/python gear_sonic/eval_agent_trl.py ++seed=0 ++num_envs=1 "
        "++callbacks.im_eval.max_eval_steps=null "
        "+eval_events=nominal_d1 +manager_env/terminations=tracking/eval "
        "++manager_env.config.terrain_type=plane "
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true "
        "++manager_env.observations.policy.enable_corruption=false "
        "++manager_env.observations.tokenizer.enable_corruption=false +use_encoder=g1"
    )
    for variant in template["variants"]:
        variant["checkpoint"] = f"outputs/{variant['name']}_seed{{seed}}/last.pt"
        variant["checkpoint_source"] = "trained_variant_checkpoint"
        variant["train_command"] = variant["train_command"].replace(
            "python gear_sonic/train_agent_trl.py",
            "/resolved/training/python gear_sonic/train_agent_trl.py",
        )
        variant["train_command"] += (
            " +checkpoint=release/last.pt +resume=false num_envs=2 ++algo.config.num_learning_iterations=50"
        )
        variant["eval_command"] = nominal_eval + f" +checkpoint=outputs/{variant['name']}_seed{{seed}}/last.pt"
        variant["metrics_eval_json"] = f"metrics/{variant['name']}_seed{{seed}}.json"
    return template


def test_m5_result_gate_requires_matching_activated_evidence(tmp_path: Path) -> None:
    seeds = [0, 1, 2]
    template = _make_activation_gated_template(tmp_path, seeds)
    common = {
        "template": template,
        "seeds": seeds,
        "dry_run": True,
        "repo_root": tmp_path,
        "variant_a": "adaptive_sampling_micro",
        "variant_b": "uniform_sampling_micro",
        "effect_metric": "eval.all.mpjpe_g",
        "a_minus_b_threshold": -0.5,
        "min_improved_seeds": 2,
    }

    missing = run_multiseed_experiment(output_dir=tmp_path / "missing_activation", **common)
    assert missing["activation_summary"]["passes_required_activation_gate"] is False
    assert missing["passes_all_preregistered_result_gates"] is False
    missing_aggregate = json.loads(
        (tmp_path / "missing_activation/aggregate_comparison.json").read_text(encoding="utf-8")
    )
    assert missing_aggregate["effect_summary"]["passes_preregistered_effect_gate"] is True
    assert missing_aggregate["passes_all_preregistered_result_gates"] is False

    activation_path = tmp_path / "activation.json"
    activation_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "m5_activation_classification",
                "arm": "m5_l",
                "arm_verdict": "activated",
                "seed_verdicts": {str(seed): "activated" for seed in seeds},
                "seeds": {str(seed): {"verdict": "activated", "evidence": {}} for seed in seeds},
                "missing_seeds": [],
                "unexpected_seeds": [],
                "seeds_activated": 3,
                "seeds_total": 3,
            }
        ),
        encoding="utf-8",
    )
    activated = run_multiseed_experiment(
        output_dir=tmp_path / "activated",
        activation_json=activation_path,
        **common,
    )
    assert activated["activation_summary"]["passes_required_activation_gate"] is True
    assert activated["activation_summary"]["sha256"]
    # Dry-run evidence can validate internally but can never become result-ready.
    assert activated["passes_all_preregistered_result_gates"] is False
    assert activated["passes_non_activation_preregistered_result_gates"] is True
    assert activated["result_gate_status"] == "dry_run_not_result_ready"

    evidence = json.loads(activation_path.read_text(encoding="utf-8"))
    evidence["arm"] = "m5_t"
    activation_path.write_text(json.dumps(evidence), encoding="utf-8")
    wrong_arm = run_multiseed_experiment(
        output_dir=tmp_path / "wrong_arm",
        activation_json=activation_path,
        **common,
    )
    assert wrong_arm["activation_summary"]["passes_required_activation_gate"] is False
    assert wrong_arm["passes_all_preregistered_result_gates"] is False


def test_activation_gate_recomputes_detailed_seed_verdicts(tmp_path: Path) -> None:
    seeds = [0, 1, 2]
    template = _make_activation_gated_template(tmp_path, seeds)
    activation_path = tmp_path / "forged_activation.json"
    activation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "m5_activation_classification",
                "arm": "m5_l",
                "arm_verdict": "activated",
                "seed_verdicts": {str(seed): "activated" for seed in seeds},
                "seeds": {
                    "0": {"verdict": "activated", "evidence": {}},
                    "1": {"verdict": "inactive", "evidence": {}},
                    "2": {"verdict": "incomplete", "evidence": {}},
                },
                "missing_seeds": [],
                "unexpected_seeds": [],
                "seeds_activated": 3,
                "seeds_total": 3,
            }
        ),
        encoding="utf-8",
    )

    run = run_multiseed_experiment(
        template,
        seeds=seeds,
        output_dir=tmp_path / "forged",
        dry_run=True,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
        activation_json=activation_path,
    )

    summary = run["activation_summary"]
    assert summary["passes_required_activation_gate"] is False
    assert summary["recomputed_seeds_activated"] == 1
    assert summary["disqualifying_seed_verdicts"] == {"2": "incomplete"}
    assert any("do not match" in error for error in summary["errors"])


def test_m5_contract_rejects_bare_or_mismatched_interpreter_and_initialization(
    tmp_path: Path,
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    assert validate_m5_multiseed_contract(template) == []

    bare = json.loads(json.dumps(template))
    bare["training_python_executable"] = "python"
    for variant in bare["variants"]:
        variant["train_command"] = variant["train_command"].replace("/resolved/training/python", "python")
        variant["eval_command"] = variant["eval_command"].replace("/resolved/training/python", "python")
    errors = validate_m5_multiseed_contract(bare)
    assert any("not a bare interpreter" in error for error in errors)

    wrong_eval = json.loads(json.dumps(template))
    wrong_eval["variants"][0]["eval_command"] = wrong_eval["variants"][0]["eval_command"].replace(
        "/resolved/training/python", "/other/python"
    )
    errors = validate_m5_multiseed_contract(wrong_eval)
    assert any("eval_command must invoke exactly" in error for error in errors)

    chained = json.loads(json.dumps(template))
    chained["variants"][0]["eval_command"] += " && python fallback.py"
    errors = validate_m5_multiseed_contract(chained)
    assert any("forbidden shell syntax" in error for error in errors)

    wrong_init = json.loads(json.dumps(template))
    wrong_init["variants"][0]["train_command"] = (
        wrong_init["variants"][0]["train_command"]
        .replace("+checkpoint=release/last.pt", "+checkpoint=other.pt")
        .replace("+resume=false", "+resume=true")
    )
    errors = validate_m5_multiseed_contract(wrong_init)
    assert any("+checkpoint=release/last.pt" in error for error in errors)
    assert any("exactly +resume=false" in error for error in errors)


def test_flat_inventory_rejects_nested_loader_invisible_smpl(tmp_path: Path) -> None:
    robot = tmp_path / "robot"
    smpl = tmp_path / "smpl"
    robot.mkdir()
    smpl.mkdir()
    (robot / "motion.pkl").write_bytes(b"robot")
    (smpl / "motion.pkl").write_bytes(b"smpl")
    spec = {"dataset_robot": str(robot), "dataset_smpl": str(smpl)}

    verified = _verify_flat_paired_inventory(spec, repo_root=tmp_path, motion_keys=["motion"])
    assert verified["flat_direct_children"] is True
    assert verified["robot_file_count"] == verified["smpl_file_count"] == 1

    nested = smpl / "nested"
    nested.mkdir()
    (nested / "invisible.pkl").write_bytes(b"nested")
    with pytest.raises(ValueError, match="must be flat"):
        _verify_flat_paired_inventory(spec, repo_root=tmp_path, motion_keys=["motion"])


def _resolve_m5_fixture(template: dict, tmp_path: Path) -> tuple[dict, Path]:
    robot = tmp_path / "sample_data/robot_filtered"
    smpl = tmp_path / "sample_data/smpl_filtered"
    robot.mkdir(parents=True)
    smpl.mkdir(parents=True)
    (robot / "motion.pkl").write_bytes(b"robot")
    (smpl / "motion.pkl").write_bytes(b"smpl")

    initialization = tmp_path / "release/last.pt"
    initialization.parent.mkdir()
    initialization.write_bytes(b"release initialization")
    ranking = tmp_path / "ranking.json"
    ranking.write_text('["motion"]\n', encoding="utf-8")
    classification = tmp_path / "classification.json"
    classification.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    resolved = json.loads(json.dumps(template))
    resolved.update(
        {
            "training_python_executable": sys.executable,
            "training_initialization_checkpoint": str(initialization),
            "dataset_robot": str(robot),
            "dataset_smpl": str(smpl),
            "dataset_manifest_json": str(manifest),
            "dataset_manifest_sha256": "1" * 64,
            "difficulty_ranking_json": str(ranking),
            "difficulty_ranking_sha256": "2" * 64,
            "sim_d1_classification_json": str(classification),
            "sim_d1_classification_sha256": "3" * 64,
            "sim_d1_source_checkpoint_sha256": _file_sha256(initialization),
        }
    )
    for variant in resolved["variants"]:
        for command_field in ("train_command", "eval_command"):
            variant[command_field] = variant[command_field].replace("/resolved/training/python", sys.executable)
        variant["train_command"] = variant["train_command"].replace(
            "+checkpoint=release/last.pt", f"+checkpoint={initialization}"
        )
    return resolved, initialization


def test_runtime_preflight_hash_binds_release_initialization_and_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, initialization = _resolve_m5_fixture(render_spec_for_seed(template, 0), tmp_path)

    def fake_data_preflight(spec: dict, *, repo_root: Path) -> dict:
        return {
            **_verify_activation_launch_runtime(spec, repo_root=repo_root),
            "dataset_manifest": {"motion_keys": ["motion"]},
            "dataset_inventory": {"flat_direct_children": True},
        }

    monkeypatch.setattr(
        "scripts.research.run_sonic_multiseed._verify_activation_preflight",
        fake_data_preflight,
    )

    provenance = _verify_training_runtime_inputs(resolved, repo_root=tmp_path)

    initialization_provenance = provenance["training_initialization_checkpoint"]
    assert initialization_provenance["sha256"] == _file_sha256(initialization)
    assert initialization_provenance["hash_matches_sim_d1_source_checkpoint"] is True
    assert initialization_provenance["resume"] is False
    assert provenance["training_python_executable"]["all_train_and_eval_commands_exactly_bound"] is True
    assert provenance["dataset_inventory"]["flat_direct_children"] is True

    resolved["sim_d1_source_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        _verify_training_runtime_inputs(resolved, repo_root=tmp_path)


def test_preflight_mode_never_materializes_or_marks_results_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    calls: list[int] = []

    def fake_runtime_preflight(spec: dict, *, repo_root: Path) -> dict:
        calls.append(spec["seed"])
        return {"verified_seed": spec["seed"]}

    monkeypatch.setattr(
        "scripts.research.run_sonic_multiseed._verify_training_runtime_inputs",
        fake_runtime_preflight,
    )
    report = preflight_multiseed_experiment(
        resolved,
        seeds=[0, 1, 2],
        output_dir=tmp_path / "preflight",
        repo_root=tmp_path,
    )

    assert calls == [0, 1, 2]
    assert report["input_provenance_by_seed"] == {
        "0": {"verified_seed": 0},
        "1": {"verified_seed": 1},
        "2": {"verified_seed": 2},
    }
    assert report["passes_preflight"] is True
    assert report["execution_started"] is False
    assert report["result_ready"] is False
    assert (tmp_path / "preflight/preflight.json").is_file()
    assert not (tmp_path / "preflight/multiseed_run.json").exists()
    assert not list((tmp_path / "preflight").glob("seed*/run_plan.json"))


def test_preflight_rejects_seed_varying_scientific_command_settings(
    tmp_path: Path,
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    resolved["variants"][0]["train_command"] += " ++science.knob={seed}"

    with pytest.raises(ValueError, match="vary in scientific/sampler/schedule"):
        preflight_multiseed_experiment(
            resolved,
            seeds=[0, 1, 2],
            output_dir=tmp_path / "varying",
            repo_root=tmp_path,
        )


def test_preflight_rejects_checkpoint_reuse_across_rendered_seeds(
    tmp_path: Path,
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    for variant in resolved["variants"]:
        checkpoint = variant["checkpoint"]
        fixed_checkpoint = checkpoint.replace("{seed}", "fixed")
        variant["checkpoint"] = fixed_checkpoint
        variant["eval_command"] = variant["eval_command"].replace(
            f"+checkpoint={checkpoint}", f"+checkpoint={fixed_checkpoint}"
        )

    with pytest.raises(ValueError, match="unique across all arms and seeds"):
        preflight_multiseed_experiment(
            resolved,
            seeds=[0, 1, 2],
            output_dir=tmp_path / "reused",
            repo_root=tmp_path,
        )


def test_preflight_refuses_report_overwrite_and_input_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    monkeypatch.setattr(
        "scripts.research.run_sonic_multiseed._verify_training_runtime_inputs",
        lambda spec, repo_root: {"verified_seed": spec["seed"]},
    )
    output_dir = tmp_path / "preflight_safe"
    preflight_multiseed_experiment(resolved, seeds=[0, 1, 2], output_dir=output_dir, repo_root=tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        preflight_multiseed_experiment(
            resolved,
            seeds=[0, 1, 2],
            output_dir=output_dir,
            repo_root=tmp_path,
        )

    collision_dir = tmp_path / "input_collision"
    collision_dir.mkdir()
    collision_path = collision_dir / "preflight.json"
    collision_path.write_text("{}\n", encoding="utf-8")
    colliding = json.loads(json.dumps(resolved))
    colliding["dataset_manifest_json"] = str(collision_path)
    with pytest.raises(ValueError, match="collides with a verified input"):
        preflight_multiseed_experiment(
            colliding,
            seeds=[0, 1, 2],
            output_dir=collision_dir,
            repo_root=tmp_path,
        )


def test_execute_reuses_identical_explicit_preflight_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_sonic_multiseed as multiseed_module

    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    monkeypatch.setattr(
        multiseed_module,
        "_verify_training_runtime_inputs",
        lambda spec, repo_root: {"verified_seed": spec["seed"]},
    )
    output_dir = tmp_path / "preflight_then_execute"
    report_path = output_dir / "preflight.json"
    original_write_json = multiseed_module._write_json
    preflight_writes: list[Path] = []

    def tracked_write_json(path: Path, data: dict) -> None:
        if path.resolve() == report_path.resolve():
            preflight_writes.append(path)
        original_write_json(path, data)

    monkeypatch.setattr(multiseed_module, "_write_json", tracked_write_json)
    preflight_multiseed_experiment(
        resolved,
        seeds=[0, 1, 2],
        output_dir=output_dir,
        repo_root=tmp_path,
    )
    assert len(preflight_writes) == 1
    report_bytes = report_path.read_bytes()
    preflight_writes.clear()
    materialized_seeds: list[int] = []

    def fake_materialize(spec: dict, *, output_dir: Path, dry_run: bool, repo_root: Path) -> dict:
        assert dry_run is False
        materialized_seeds.append(spec["seed"])
        return {
            "input_provenance": {"verified_seed": spec["seed"]},
            "execution_ok": True,
            "comparison_json": None,
            "ok_for_causal_comparison": False,
        }

    monkeypatch.setattr(multiseed_module, "materialize_paired_experiment", fake_materialize)
    run = run_multiseed_experiment(
        resolved,
        seeds=[0, 1, 2],
        output_dir=output_dir,
        dry_run=False,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric=None,
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert materialized_seeds == [0, 1, 2]
    assert preflight_writes == []
    assert report_path.read_bytes() == report_bytes
    assert run["preflight_json"] == str(report_path)


def test_execute_rejects_stale_explicit_preflight_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import run_sonic_multiseed as multiseed_module

    template = _make_activation_gated_template(tmp_path, [0, 1, 2])
    resolved, _ = _resolve_m5_fixture(template, tmp_path)
    monkeypatch.setattr(
        multiseed_module,
        "_verify_training_runtime_inputs",
        lambda spec, repo_root: {"verified_seed": spec["seed"]},
    )
    output_dir = tmp_path / "stale_preflight"
    preflight_multiseed_experiment(
        resolved,
        seeds=[0, 1, 2],
        output_dir=output_dir,
        repo_root=tmp_path,
    )
    report_path = output_dir / "preflight.json"
    stale_report = json.loads(report_path.read_text(encoding="utf-8"))
    stale_report["input_provenance_by_seed"]["0"] = {"verified_seed": 99}
    report_path.write_text(json.dumps(stale_report), encoding="utf-8")
    stale_bytes = report_path.read_bytes()
    materialized_seeds: list[int] = []
    monkeypatch.setattr(
        multiseed_module,
        "materialize_paired_experiment",
        lambda spec, **kwargs: materialized_seeds.append(spec["seed"]),
    )

    with pytest.raises(ValueError, match="stale or does not match"):
        run_multiseed_experiment(
            resolved,
            seeds=[0, 1, 2],
            output_dir=output_dir,
            dry_run=False,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )

    assert materialized_seeds == []
    assert report_path.read_bytes() == stale_bytes


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executed_finalization_fixture(
    tmp_path: Path, *, activation_arm: str = "m5_l"
) -> dict[str, object]:
    output_dir = tmp_path / "executed"
    output_dir.mkdir()
    manifest_sha = "1" * 64
    ranking_sha = "2" * 64
    paired_sha = "3" * 64
    sampler_schema = {
        "kind": "zpd_adaptive_sampler_state",
        "version": 1,
        "sampling_mass_semantics": "selected_target_bin_pre_window_shift",
    }
    sampler_config = {
        "signal": "learnability",
        "optimism_k": 0.0,
        "evidence_half_life": 8.0,
        "advmass_n": 16,
        "uniform_sampling_rate": 0.1,
        "tripwire_max_prob_over_uniform": 20.0,
        "bin_size": 50,
        "paired_dataset_sha256": paired_sha,
    }
    if activation_arm == "m5_l":
        prefix = "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
        train_command = " ".join(
            [
                "python train.py",
                f"{prefix}enable=true",
                f"{prefix}signal=learnability",
                f"{prefix}optimism_k=0.0",
                f"{prefix}evidence_half_life=8.0",
                f"{prefix}advmass_n=16",
                f"{prefix}uniform_sampling_rate=0.1",
                f"{prefix}tripwire_max_prob_over_uniform=20.0",
                f"{prefix}bin_size=50",
                f"{prefix}paired_dataset_sha256={paired_sha}",
            ]
        )
    else:
        assert activation_arm == "m5_t"
        train_command = "python train.py ++manager_env.config.log_m5t_height_termination=true"

    dump_paths: list[Path] = []
    telemetry_logs: list[Path] = []
    evidence_seeds: dict[str, dict] = {}
    for seed in (0, 1, 2):
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir()
        checkpoint = seed_dir / "treatment.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_sha = _file_sha256(checkpoint)
        train_log = seed_dir / "treatment_train.log"
        if activation_arm == "m5_l":
            telemetry_text = (
                "Learning iteration 1\n"
                "Env/adp_samp/tripwire_max_prob_binding: 0.0000\n"
                "Learning iteration 2\n"
                "Env/adp_samp/tripwire_max_prob_binding: 0.0000\n"
            )
        else:
            telemetry_text = "".join(
                f"Learning iteration {iteration}\n"
                "Env/adp_samp/m5t_anchor_pos_termination_density: 0.0400\n"
                "Env/adp_samp/m5t_ee_body_pos_termination_density: 0.0200\n"
                "Env/adp_samp/m5t_height_termination_density: 0.0600\n"
                "Env/adp_samp/m5t_episode_end_density: 0.2000\n"
                for iteration in (1, 2)
            )
        train_log.write_text(
            telemetry_text,
            encoding="utf-8",
        )
        telemetry_logs.append(train_log)
        dump_path = seed_dir / "checkpoint_dump.json"
        dump_record = {
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha,
            "global_step": 2,
        }
        if activation_arm == "m5_l":
            dump_record.update(
                {
                    "sampler_schema": sampler_schema,
                    "sampler_config": sampler_config,
                }
            )
        dump_path.write_text(
            json.dumps(dump_record),
            encoding="utf-8",
        )
        dump_paths.append(dump_path)
        dump_sha = _file_sha256(dump_path)
        spec = {
            "requires_activation_gate": True,
            "activation_arm": activation_arm,
            "expected_training_iterations": 2,
        }
        (seed_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        plan = {
            "schema_version": 1,
            "kind": "sonic_paired_experiment_run",
            "dry_run": False,
            "seed": seed,
            "execution_ok": True,
            "training_completeness_ok": True,
            "input_provenance": {
                "dataset_manifest": {
                    "sha256": manifest_sha,
                    "sha256_kind": "file_bytes",
                    "paired_dataset_sha256": paired_sha,
                },
                "difficulty_ranking": {"sha256": ranking_sha},
            },
            "variants": [
                {
                    "name": "treatment",
                    "execution_ok": True,
                    "checkpoint": str(checkpoint),
                    "checkpoint_source": "trained_variant_checkpoint",
                    "checkpoint_provenance": {"sha256": checkpoint_sha},
                    "metrics_coverage": {"exact_motion_key_coverage": True},
                    "training_completion": {
                        "complete": True,
                        "expected_learning_iteration": 2,
                        "observed_learning_iteration": 2,
                    },
                    "command_results": [{"kind": "train", "returncode": 0, "log": str(train_log)}],
                    "commands": {"train": train_command},
                }
            ],
        }
        (seed_dir / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        seed_provenance = {
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_global_step": 2,
            "checkpoint_dump_sha256": dump_sha,
            "difficulty_ranking_sha256": ranking_sha,
            "telemetry_summary_sha256": None,
            "dataset_manifest_sha256": manifest_sha,
            "dataset_manifest_sha256_kind": "file_bytes",
            "paired_dataset_sha256": paired_sha,
        }
        if activation_arm == "m5_l":
            seed_provenance.update(
                {
                    "sampler_schema": sampler_schema,
                    "sampler_config": sampler_config,
                }
            )
        evidence_seeds[str(seed)] = {
            "verdict": "activated",
            "evidence": {},
            "provenance": seed_provenance,
        }

    telemetry_path = output_dir / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(summarize_telemetry_logs(telemetry_logs), sort_keys=True),
        encoding="utf-8",
    )
    telemetry_sha = _file_sha256(telemetry_path)
    for record in evidence_seeds.values():
        record["provenance"]["telemetry_summary_sha256"] = telemetry_sha
    activation = {
        "schema_version": 2,
        "kind": "m5_activation_classification",
        "arm": activation_arm,
        "expected_iterations": 2,
        "arm_verdict": "activated",
        "seed_verdicts": {str(seed): "activated" for seed in (0, 1, 2)},
        "seeds": evidence_seeds,
        "missing_seeds": [],
        "unexpected_seeds": [],
        "seeds_activated": 3,
        "seeds_total": 3,
        "provenance": {
            "dataset_manifest_sha256": manifest_sha,
            "dataset_manifest_sha256_kind": "file_bytes",
            "paired_dataset_sha256": paired_sha,
            "difficulty_ranking_sha256": ranking_sha,
            "telemetry_summary_sha256": telemetry_sha,
            "checkpoint_dump_sha256_by_seed": {str(seed): _file_sha256(dump_paths[seed]) for seed in (0, 1, 2)},
        },
    }
    activation_path = output_dir / "activation.json"
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    run = {
        "schema_version": 1,
        "kind": "sonic_multiseed_run",
        "dry_run": False,
        "seeds": [0, 1, 2],
        "variant_a": "treatment",
        "variant_b": "control",
        "execution_ok": True,
        "ok_for_causal_comparison": True,
        "missing_comparison_seeds": [],
        "failed_execution_seeds": [],
        "passes_non_activation_preregistered_result_gates": True,
        "passes_all_preregistered_result_gates": False,
    }
    aggregate = {
        "passes_non_activation_preregistered_result_gates": True,
        "passes_all_preregistered_result_gates": False,
        "rows": [],
    }
    (output_dir / "multiseed_run.json").write_text(json.dumps(run), encoding="utf-8")
    (output_dir / "aggregate_comparison.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return {
        "output_dir": output_dir,
        "activation": activation_path,
        "telemetry": telemetry_path,
        "dumps": dump_paths,
    }


def test_finalize_activation_binds_executed_artifacts_without_retraining(
    tmp_path: Path,
) -> None:
    fixture = _executed_finalization_fixture(tmp_path)
    output_dir = fixture["output_dir"]
    plan_hashes_before = {seed: _file_sha256(output_dir / f"seed{seed}/run_plan.json") for seed in (0, 1, 2)}

    run = finalize_multiseed_activation(
        output_dir=output_dir,
        activation_json=fixture["activation"],
        telemetry_summary=fixture["telemetry"],
        checkpoint_dumps=fixture["dumps"],
        repo_root=tmp_path,
    )

    assert run["passes_all_preregistered_result_gates"] is True
    assert run["activation_summary"]["finalized"] is True
    assert run["activation_summary"]["binding_kind"] == ("executed_multiseed_checkpoint_and_log_v1")
    assert plan_hashes_before == {
        seed: _file_sha256(output_dir / f"seed{seed}/run_plan.json") for seed in (0, 1, 2)
    }


def test_finalize_m5t_allows_failure_rate_dump_without_zpd_schema(
    tmp_path: Path,
) -> None:
    fixture = _executed_finalization_fixture(tmp_path, activation_arm="m5_t")

    run = finalize_multiseed_activation(
        output_dir=fixture["output_dir"],
        activation_json=fixture["activation"],
        telemetry_summary=fixture["telemetry"],
        checkpoint_dumps=fixture["dumps"],
        repo_root=tmp_path,
    )

    assert run["passes_all_preregistered_result_gates"] is True
    assert run["activation_summary"]["expected_arm"] == "m5_t"


def test_finalize_m5l_keeps_zpd_sampler_provenance_strict(tmp_path: Path) -> None:
    fixture = _executed_finalization_fixture(tmp_path)
    activation_path = fixture["activation"]
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    del activation["seeds"]["0"]["provenance"]["sampler_schema"]
    del activation["seeds"]["0"]["provenance"]["sampler_config"]
    activation_path.write_text(json.dumps(activation), encoding="utf-8")

    with pytest.raises(ValueError, match="m5_l schema-v2 fields"):
        finalize_multiseed_activation(
            output_dir=fixture["output_dir"],
            activation_json=activation_path,
            telemetry_summary=fixture["telemetry"],
            checkpoint_dumps=fixture["dumps"],
            repo_root=tmp_path,
        )


def test_finalize_activation_rejects_dry_run_without_mutating_seed_plans(
    tmp_path: Path,
) -> None:
    fixture = _executed_finalization_fixture(tmp_path)
    output_dir = fixture["output_dir"]
    run_path = output_dir / "multiseed_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["dry_run"] = True
    run_path.write_text(json.dumps(run), encoding="utf-8")
    plan_hash = _file_sha256(output_dir / "seed0/run_plan.json")

    with pytest.raises(ValueError, match="dry-run artifacts"):
        finalize_multiseed_activation(
            output_dir=output_dir,
            activation_json=fixture["activation"],
            telemetry_summary=fixture["telemetry"],
            checkpoint_dumps=fixture["dumps"],
            repo_root=tmp_path,
        )

    assert _file_sha256(output_dir / "seed0/run_plan.json") == plan_hash
