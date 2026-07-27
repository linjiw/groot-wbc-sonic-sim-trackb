from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

import scripts.research.run_sonic_paired_experiment as paired_runner
from scripts.research.run_sonic_paired_experiment import (
    materialize_paired_experiment,
    validate_spec,
)


def _summary(path: Path, *, reward: float, mpjpe: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {
                    "kind": "sonic_training_log",
                    "ok": True,
                    "learning_iteration": 100,
                    "mean_rewards": reward,
                    "total_timesteps": 38400,
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


def _spec(tmp_path: Path) -> dict:
    baseline_summary = _summary(tmp_path / "baseline_summary.json", reward=0.8, mpjpe=130.0)
    curriculum_summary = _summary(tmp_path / "curriculum_summary.json", reward=0.9, mpjpe=120.0)
    return {
        "experiment_group": "sample_pair_fixture",
        "hypothesis": "curriculum improves tracking at fixed controls",
        "seed": 0,
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "git_commit": "abc1234",
        "variants": [
            {
                "name": "baseline",
                "summary_json": str(baseline_summary),
                "train_command": "python gear_sonic/train_agent_trl.py baseline",
                "eval_command": "python gear_sonic/eval_agent_trl.py baseline",
                "interpretation": "fixture_baseline",
            },
            {
                "name": "curriculum",
                "summary_json": str(curriculum_summary),
                "train_command": "python gear_sonic/train_agent_trl.py curriculum",
                "eval_command": "python gear_sonic/eval_agent_trl.py curriculum",
                "interpretation": "fixture_curriculum",
            },
        ],
    }


def test_validate_spec_requires_top_level_and_variant_fields() -> None:
    errors = validate_spec({"variants": [{"name": "baseline"}]})

    assert "missing experiment_group" in errors
    assert "missing checkpoint" in errors
    assert "variants[0] missing eval_command" in errors
    assert "variants[0] missing interpretation" in errors


def test_materialize_dry_run_builds_manifests_and_comparison(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    output_dir = tmp_path / "paired_run"

    plan = materialize_paired_experiment(spec, output_dir=output_dir, dry_run=True, repo_root=tmp_path)

    assert plan["dry_run"] is True
    assert plan["ok_for_causal_comparison"] is True
    assert (output_dir / "run_plan.json").exists()
    assert (output_dir / "run_plan.md").exists()
    assert (output_dir / "baseline" / "manifest.json").exists()
    assert (output_dir / "curriculum" / "manifest.json").exists()
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["manifest_count"] == 2
    assert comparison["control_mismatches"] == []
    assert comparison["rows"][1]["metrics"]["eval.all.mpjpe_g"] == 120.0
    assert plan["command_environment"]["orchestrator_python"]


def test_bare_python_uses_orchestrator_interpreter_dir_when_path_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "orchestrator" / "python"
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nprintf 'orchestrator-python\\n'\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setattr(paired_runner.sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", "")

    log_path = tmp_path / "command.log"
    returncode = paired_runner._run_command("python ignored", log_path, cwd=tmp_path)

    assert returncode == 0
    assert log_path.read_text(encoding="utf-8") == "orchestrator-python\n"


def test_structured_command_parser_separates_allowlisted_environment() -> None:
    environment, argv = paired_runner._parse_command(
        "WANDB_MODE=disabled HYDRA_FULL_ERROR=1 /opt/python train.py seed=0"
    )

    assert environment == {"WANDB_MODE": "disabled", "HYDRA_FULL_ERROR": "1"}
    assert argv == ("/opt/python", "train.py", "seed=0")
    with pytest.raises(ValueError, match="not allowlisted"):
        paired_runner._parse_command("resume=false /opt/python train.py")


@pytest.mark.parametrize(
    "suffix",
    (
        " # +checkpoint=ignored",
        " & python fallback.py",
        "\npython fallback.py",
        " > stolen.log",
        " $(python fallback.py)",
        " `python fallback.py`",
    ),
)
def test_activation_contract_rejects_shell_syntax(tmp_path: Path, suffix: str) -> None:
    spec = _activation_preflight_spec(tmp_path)
    spec["variants"][0]["train_command"] += suffix

    assert any("forbidden shell syntax" in error for error in validate_spec(spec))


def test_materialize_uses_variant_checkpoint_and_provenance(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", reward=0.8, mpjpe=10.0)
    spec = {
        "experiment_group": "posttrain",
        "hypothesis": "h",
        "seed": 0,
        "dataset_robot": "robot",
        "dataset_smpl": "smpl",
        "checkpoint": "sonic_release/last.pt",
        "variants": [
            {
                "name": "uniform",
                "summary_json": str(summary),
                "checkpoint": "runs/uniform/last.pt",
                "checkpoint_source": "trained_variant_checkpoint",
                "checkpoint_provenance": {"sha256": "abc", "is_release_checkpoint": False},
                "eval_command": "python eval.py +checkpoint=runs/uniform/last.pt",
                "interpretation": "posttrain",
            }
        ],
    }

    materialize_paired_experiment(spec, output_dir=tmp_path / "out", dry_run=True, repo_root=tmp_path)

    manifest = json.loads((tmp_path / "out" / "uniform" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"] == "runs/uniform/last.pt"
    assert manifest["checkpoint_source"] == "trained_variant_checkpoint"
    assert manifest["checkpoint_provenance"]["sha256"] == "abc"


def test_materialize_dry_run_without_summaries_creates_plan_only(tmp_path: Path) -> None:
    spec = {
        "experiment_group": "plan_only",
        "hypothesis": "h",
        "seed": 0,
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "variants": [
            {
                "name": "baseline",
                "train_command": "python train.py",
                "eval_command": "python eval.py",
                "interpretation": "not_run_yet",
            }
        ],
    }

    plan = materialize_paired_experiment(spec, output_dir=tmp_path / "plan", dry_run=True, repo_root=tmp_path)

    assert plan["ok_for_causal_comparison"] is False
    assert plan["comparison_json"] is None
    assert plan["variants"][0]["manifest_json"] is None


def test_materialize_rebuilds_stale_summary_when_logs_are_newer(tmp_path: Path) -> None:
    output_dir = tmp_path / "paired_run"
    variant_dir = output_dir / "baseline"
    variant_dir.mkdir(parents=True)
    summary_path = variant_dir / "summary.json"
    train_log = variant_dir / "train.log"
    eval_log = variant_dir / "eval.log"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {"ok": True, "mean_rewards": -1},
                "eval": {"ok": False, "all": {}},
            }
        ),
        encoding="utf-8",
    )
    train_log.write_text(
        "Learning iteration 1\nMean rewards: 2.5\nTotal timesteps: 24\nTotal time: 1.0s\n",
        encoding="utf-8",
    )
    eval_log.write_text(
        "All:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n"
        "Succ:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n"
        "Terminated: 0 | Succ rate: 1.000\n",
        encoding="utf-8",
    )
    newer = summary_path.stat().st_mtime + 10
    os.utime(train_log, (newer, newer))
    os.utime(eval_log, (newer, newer))

    spec = {
        "experiment_group": "stale_summary",
        "hypothesis": "h",
        "seed": 0,
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "variants": [
            {
                "name": "baseline",
                "train_command": "python train.py",
                "eval_command": "python eval.py",
                "interpretation": "rebuilt",
            }
        ],
    }

    materialize_paired_experiment(spec, output_dir=output_dir, dry_run=True, repo_root=tmp_path)

    rebuilt = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rebuilt["train"]["mean_rewards"] == 2.5
    assert rebuilt["eval"]["all"]["mpjpe_g"] == 1.0


def _execution_spec(checkpoint: str = "outputs/baseline/last.pt") -> dict:
    return {
        "experiment_group": "execute_fixture",
        "hypothesis": "h",
        "seed": 0,
        "dataset_robot": "robot",
        "dataset_smpl": "smpl",
        "checkpoint": "VARIANT_SPECIFIC_CHECKPOINTS",
        "checkpoint_source": "trained_variant_checkpoint",
        "variants": [
            {
                "name": "baseline",
                "checkpoint": checkpoint,
                "checkpoint_source": "trained_variant_checkpoint",
                "train_command": "train baseline",
                "eval_command": f"eval baseline +checkpoint={checkpoint}",
                "interpretation": "execution_fixture",
            }
        ],
    }


def _prepare_execution_inputs(tmp_path: Path) -> None:
    (tmp_path / "robot").mkdir(exist_ok=True)
    (tmp_path / "smpl").mkdir(exist_ok=True)


def test_execute_skips_eval_after_failed_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_execution_inputs(tmp_path)
    calls: list[str] = []

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("train failed\n", encoding="utf-8")
        return 7

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        _execution_spec(), output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path
    )

    assert calls == ["train baseline"]
    assert plan["execution_ok"] is False
    assert plan["comparison_json"] is None
    assert "return code 7" in plan["variants"][0]["execution_errors"][0]


def test_execute_skips_eval_when_train_does_not_create_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_execution_inputs(tmp_path)
    calls: list[str] = []

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Learning iteration 1\nMean rewards: 1.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        _execution_spec(), output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path
    )

    assert calls == ["train baseline"]
    assert plan["execution_ok"] is False
    assert "was not created" in plan["variants"][0]["execution_errors"][0]


def test_execute_rejects_stale_preexisting_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_execution_inputs(tmp_path)
    checkpoint = tmp_path / "outputs" / "baseline" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stale")

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del command, cwd
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Learning iteration 1\nMean rewards: 1.0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        _execution_spec(), output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path
    )

    assert plan["execution_ok"] is False
    assert "was not updated" in plan["variants"][0]["execution_errors"][0]


def test_execute_verifies_fresh_checkpoint_before_eval_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_execution_inputs(tmp_path)
    checkpoint = tmp_path / "outputs" / "baseline" / "last.pt"
    calls: list[str] = []

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if command.startswith("train"):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"fresh checkpoint")
            log_path.write_text(
                "Learning iteration 1\nMean rewards: 1.0\nTotal timesteps: 24\n",
                encoding="utf-8",
            )
        else:
            log_path.write_text(
                "All:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n"
                "Succ:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n"
                "Terminated: 0 | Succ rate: 1.000\n",
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        _execution_spec(), output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path
    )

    assert calls == [
        "train baseline",
        "eval baseline +checkpoint=outputs/baseline/last.pt",
    ]
    assert plan["execution_ok"] is True
    manifest = json.loads((tmp_path / "out" / "baseline" / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["checkpoint_provenance"]
    assert provenance["size_bytes"] == len(b"fresh checkpoint")
    assert len(provenance["sha256"]) == 64
    assert provenance["is_release_checkpoint"] is False


def test_execute_rejects_stale_preexisting_per_motion_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_execution_inputs(tmp_path)
    spec = _execution_spec()
    spec["difficulty_ranking_json"] = "ranking.json"
    spec["variants"][0]["metrics_eval_json"] = "outputs/baseline/metrics_eval.json"
    (tmp_path / "ranking.json").write_text('["easy"]\n', encoding="utf-8")
    metrics_path = tmp_path / "outputs" / "baseline" / "metrics_eval.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "eval/all_metrics_dict": {
                    "motion_keys": ["easy"],
                    "mpjpe_g": [1.0],
                    "terminated": [0.0],
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "outputs" / "baseline" / "last.pt"

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if command.startswith("train"):
            checkpoint.write_bytes(b"fresh checkpoint")
            log_path.write_text("Learning iteration 1\nMean rewards: 1.0\n", encoding="utf-8")
        else:
            log_path.write_text(
                "All:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n",
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(spec, output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path)

    assert plan["execution_ok"] is False
    assert "did not update required per-motion metrics" in plan["variants"][0]["execution_errors"][0]


def test_execute_preflight_checks_variant_level_ranking(tmp_path: Path) -> None:
    _prepare_execution_inputs(tmp_path)
    spec = _execution_spec()
    spec["variants"][0]["difficulty_ranking_json"] = "missing_ranking.json"
    spec["variants"][0]["metrics_eval_json"] = "outputs/baseline/metrics_eval.json"

    with pytest.raises(ValueError, match="variants\\[0\\].difficulty_ranking_json does not exist"):
        materialize_paired_experiment(spec, output_dir=tmp_path / "out", dry_run=False, repo_root=tmp_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_sha256(records: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(records):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _activation_preflight_spec(tmp_path: Path) -> dict:
    robot_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    robot_dir.mkdir()
    smpl_dir.mkdir()
    robot_motion = robot_dir / "motion.pkl"
    smpl_motion = smpl_dir / "motion.pkl"
    robot_motion.write_bytes(b"robot")
    smpl_motion.write_bytes(b"smpl")
    robot_record = ("robot/motion.pkl", _sha256(robot_motion))
    smpl_record = ("smpl/motion.pkl", _sha256(smpl_motion))
    manifest = {
        "schema_version": 1,
        "kind": "sonic_paired_dataset_inventory",
        "generator": "test",
        "output": {
            "robot_dir": "robot",
            "smpl_dir": "smpl",
            "motion_count": 1,
            "motion_keys": ["motion"],
            "robot_dataset_sha256": _inventory_sha256([robot_record]),
            "smpl_dataset_sha256": _inventory_sha256([smpl_record]),
            "paired_dataset_sha256": _inventory_sha256([robot_record, smpl_record]),
            "variants": [
                {
                    "motion_key": "motion",
                    "robot": {"path": robot_record[0], "sha256": robot_record[1]},
                    "smpl": {"path": smpl_record[0], "sha256": smpl_record[1]},
                }
            ],
        },
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(["motion"]), encoding="utf-8")
    initialization_checkpoint = tmp_path / "release" / "last.pt"
    initialization_checkpoint.parent.mkdir()
    initialization_checkpoint.write_bytes(b"release initialization")
    source_checkpoint_sha256 = _sha256(initialization_checkpoint)
    classification = {
        "schema_version": 1,
        "kind": "sim_d1_headroom_classification",
        "verdict": "PASS",
        "pass": True,
        "eligible_for_effect_experiment": True,
        "coverage": {
            "all_motion_coverage_independently_verified": True,
            "exact_key_set_verified": True,
            "evaluated_motion_count": 1,
            "expected_motion_count": 1,
        },
        "ranking": ["motion"],
        "source": {
            "checkpoint_sha256": source_checkpoint_sha256,
            "dataset_robot": str(robot_dir.resolve()),
            "dataset_smpl": str(smpl_dir.resolve()),
            "expected_motion_count": 1,
            "dataset_manifest": {
                "content_hashes_verified": True,
                "paired_dataset_sha256": manifest["output"]["paired_dataset_sha256"],
                "sha256": _sha256(manifest_path),
                "sha256_kind": "file_bytes",
                "motion_count": 1,
                "motion_keys": ["motion"],
                "verified_file_count": 2,
            },
        },
    }
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(json.dumps(classification), encoding="utf-8")
    return {
        "experiment_group": "activation_preflight",
        "hypothesis": "fixture",
        "seed": 0,
        "dataset_robot": "robot",
        "dataset_smpl": "smpl",
        "checkpoint": "VARIANT_SPECIFIC_CHECKPOINTS",
        "checkpoint_source": "trained_variant_checkpoint",
        "requires_activation_gate": True,
        "activation_arm": "m5_l",
        "variant_a": "learnability",
        "dataset_manifest_json": "dataset_manifest.json",
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_manifest_sha256_kind": "file_bytes",
        "sim_d1_classification_json": "classification.json",
        "sim_d1_classification_sha256": _sha256(classification_path),
        "sim_d1_source_checkpoint_sha256": source_checkpoint_sha256,
        "training_python_executable": sys.executable,
        "training_initialization_checkpoint": str(initialization_checkpoint),
        "difficulty_ranking_json": "ranking.json",
        "difficulty_ranking_sha256": _sha256(ranking_path),
        "expected_training_iterations": 1,
        "training_num_envs": 1,
        "variants": [
            {
                "name": "learnability",
                "checkpoint": "outputs/learnability/last.pt",
                "checkpoint_source": "trained_variant_checkpoint",
                "metrics_eval_json": "outputs/learnability/eval_metrics/metrics_eval.json",
                "train_command": (
                    f"{sys.executable} gear_sonic/train_agent_trl.py "
                    f"+checkpoint={initialization_checkpoint} +resume=false seed=0 num_envs=1 "
                    "++algo.config.num_learning_iterations=1 "
                    "++manager_env.commands.motion.motion_lib_cfg.motion_file=robot "
                    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=smpl "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=true "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.signal=learnability "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.optimism_k=0.0 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.evidence_half_life=4.0 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.advmass_n=16 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate=0.1 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
                    "tripwire_max_prob_over_uniform=20.0 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.bin_size=50 "
                    "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
                    "paired_dataset_sha256="
                    f"{manifest['output']['paired_dataset_sha256']}"
                ),
                "eval_command": (
                    f"{sys.executable} gear_sonic/eval_agent_trl.py "
                    "+checkpoint=outputs/learnability/last.pt "
                    "++seed=0 ++num_envs=1 "
                    "++callbacks.im_eval.max_eval_steps=null "
                    "+eval_events=nominal_d1 "
                    "+manager_env/terminations=tracking/eval "
                    "++manager_env.config.terrain_type=plane "
                    "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true "
                    "++manager_env.observations.policy.enable_corruption=false "
                    "++manager_env.observations.tokenizer.enable_corruption=false "
                    "+use_encoder=g1 "
                    "++manager_env.commands.motion.motion_lib_cfg.motion_file=robot "
                    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=smpl"
                ),
                "interpretation": "fixture",
            }
        ],
    }


def _official_sampler_pair_spec(tmp_path: Path) -> dict:
    fixture = _activation_preflight_spec(tmp_path)
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    paired_sha256 = manifest["output"]["paired_dataset_sha256"]
    initialization = str(fixture["training_initialization_checkpoint"])

    shared_sampler = (
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=true "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.bin_size=50 "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.sequence_length_agnostic=true "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.init_num_failures=1 "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.uniform_sampling_rate=0.1 "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.pre_failure_sample_window=200 "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.use_failure_rate_decay=false "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.decay_gamma=0.8 "
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
        "adp_samp_failure_rate_max_over_mean=200"
    )

    def variant(name: str, *, signal: str) -> dict:
        checkpoint = f"outputs/{name}/last.pt"
        zpd = ""
        if signal == "learnability":
            zpd = (
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.optimism_k=0.0"
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.evidence_half_life=null"
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.advmass_n=16"
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
                "tripwire_max_prob_over_uniform=20.0"
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.family_kernel=null"
                " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.paired_dataset_sha256="
                f"{paired_sha256}"
            )
        return {
            "name": name,
            "checkpoint": checkpoint,
            "checkpoint_source": "trained_variant_checkpoint",
            "metrics_eval_json": f"outputs/{name}/eval_metrics/metrics_eval.json",
            "train_command": (
                f"{sys.executable} gear_sonic/train_agent_trl.py "
                f"+checkpoint={initialization} +resume=false "
                "+exp=manager/universal_token/all_modes/sonic_release "
                f"num_envs=1 headless=True use_wandb=false seed=0 exp_var={name} "
                f"experiment_dir=outputs/{name} ++algo.config.num_learning_iterations=1 "
                "++manager_env.commands.motion.motion_lib_cfg.motion_file=robot "
                "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=smpl "
                "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true "
                f"{shared_sampler} "
                "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.signal="
                f"{signal}{zpd}"
            ),
            "eval_command": (
                f"{sys.executable} gear_sonic/eval_agent_trl.py +checkpoint={checkpoint} "
                "+headless=True ++seed=0 ++eval_callbacks=im_eval ++run_eval_loop=False "
                "++num_envs=1 ++callbacks.im_eval.max_eval_steps=null "
                f"++eval_output_dir=outputs/{name}/eval_metrics "
                "+manager_env/terminations=tracking/eval "
                "++manager_env.observations.policy.enable_corruption=False "
                "++manager_env.observations.tokenizer.enable_corruption=False "
                "++manager_env.commands.motion.motion_lib_cfg.motion_file=robot "
                "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=smpl "
                "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true"
            ),
            "interpretation": signal,
        }

    return {
        "experiment_group": "bones_official_sampler_pair",
        "variant_a": "learnability",
        "variant_b": "failure_rate",
        "hypothesis": "learnability improves official tracking metrics at matched compute",
        "seed": 0,
        "dataset_robot": "robot",
        "dataset_smpl": "smpl",
        "checkpoint": "VARIANT_SPECIFIC_CHECKPOINTS",
        "checkpoint_source": "trained_variant_checkpoint",
        "sampler_pair_contract": "official_failure_rate_vs_zpd_learnability_v1",
        "dataset_manifest_json": "dataset_manifest.json",
        "dataset_manifest_sha256": _sha256(tmp_path / "dataset_manifest.json"),
        "dataset_manifest_sha256_kind": "file_bytes",
        "paired_dataset_sha256": paired_sha256,
        "training_python_executable": sys.executable,
        "training_initialization_checkpoint": initialization,
        "training_initialization_checkpoint_sha256": _sha256(Path(initialization)),
        "expected_training_iterations": 1,
        "training_num_envs": 1,
        "variants": [
            variant("failure_rate", signal="failure_rate"),
            variant("learnability", signal="learnability"),
        ],
    }


def test_activation_contract_requires_exact_entrypoint_seed_and_checkpoints(
    tmp_path: Path,
) -> None:
    base = _activation_preflight_spec(tmp_path)

    wrong_entrypoint = json.loads(json.dumps(base))
    wrong_entrypoint["variants"][0]["train_command"] = wrong_entrypoint["variants"][0]["train_command"].replace(
        "gear_sonic/train_agent_trl.py", "-c"
    )
    assert any("exact entrypoint" in error for error in validate_spec(wrong_entrypoint))

    wrong_seed = json.loads(json.dumps(base))
    wrong_seed["seed"] = 1
    assert any("seed=1" in error for error in validate_spec(wrong_seed))

    wrong_eval_checkpoint = json.loads(json.dumps(base))
    wrong_eval_checkpoint["variants"][0]["eval_command"] = wrong_eval_checkpoint["variants"][0][
        "eval_command"
    ].replace(
        "+checkpoint=outputs/learnability/last.pt",
        "+checkpoint=outputs/learnability/last.pt.bak",
    )
    assert any("eval_command must contain exactly" in error for error in validate_spec(wrong_eval_checkpoint))

    resume_as_environment = json.loads(json.dumps(base))
    command = resume_as_environment["variants"][0]["train_command"]
    resume_as_environment["variants"][0]["train_command"] = "resume=false " + command.replace(
        " +resume=false", ""
    )
    assert any("not allowlisted" in error for error in validate_spec(resume_as_environment))

    for invalid_resume in ("resume=false", "+resume=true", "++resume=false"):
        invalid = json.loads(json.dumps(base))
        invalid["variants"][0]["train_command"] = invalid["variants"][0]["train_command"].replace(
            "+resume=false", invalid_resume
        )
        assert any("exactly +resume=false" in error for error in validate_spec(invalid))


def test_official_sampler_pair_contract_needs_no_sim_d1_headroom(tmp_path: Path) -> None:
    spec = _official_sampler_pair_spec(tmp_path)

    assert "requires_activation_gate" not in spec
    assert "sim_d1_classification_json" not in spec
    assert "difficulty_ranking_json" not in spec
    assert validate_spec(spec) == []

    plan = materialize_paired_experiment(
        spec,
        output_dir=tmp_path / "dry_run",
        dry_run=True,
        repo_root=tmp_path,
    )
    assert plan["sampler_pair_contract"] == "official_failure_rate_vs_zpd_learnability_v1"
    assert plan["expected_training_iterations"] == 1
    assert plan["training_completeness_ok"] is None


def test_official_sampler_pair_preflight_hash_binds_data_checkpoint_and_axis(
    tmp_path: Path,
) -> None:
    spec = _official_sampler_pair_spec(tmp_path)

    provenance = paired_runner._verify_official_zpd_pair_preflight(spec, repo_root=tmp_path)

    contract = provenance["sampler_pair_contract"]
    assert contract["baseline_signal"] == "failure_rate"
    assert contract["treatment_signal"] == "learnability"
    assert contract["shared_motion_lib_values"] == {"sort_motion_keys": "true"}
    assert contract["sim_d1_headroom_required"] is False
    assert contract["train_commands_equal_outside_declared_axis"] is True
    assert contract["eval_commands_equal_outside_output_routing"] is True
    assert provenance["dataset_manifest"]["motion_keys"] == ["motion"]
    assert provenance["dataset_inventory"]["exact_direct_child_pkl_inventory"] is True
    assert len(provenance["command_dataset_bindings"]) == 8
    assert provenance["training_num_envs"] == 1


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (
            lambda spec: spec["variants"][0].__setitem__(
                "train_command",
                spec["variants"][0]["train_command"].replace(
                    "signal=failure_rate", "signal=learnability"
                ),
            ),
            "signal=failure_rate",
        ),
        (
            lambda spec: spec["variants"][1].__setitem__(
                "train_command",
                spec["variants"][1]["train_command"]
                + " ++algo.config.num_steps_per_env=12",
            ),
            "differ outside output routing and the declared ZPD sampler axis",
        ),
        (
            lambda spec: spec["variants"][1].__setitem__(
                "eval_command",
                spec["variants"][1]["eval_command"].replace(
                    "++num_envs=1", "++num_envs=2"
                ),
            ),
            "paired eval commands differ outside checkpoint and eval_output_dir",
        ),
        (
            lambda spec: spec["variants"][0].__setitem__(
                "train_command",
                spec["variants"][0]["train_command"]
                + " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.optimism_k=0.0",
            ),
            "must not override ZPD-only adaptive_sampling.optimism_k",
        ),
        (
            lambda spec: spec["variants"][1].__setitem__(
                "train_command",
                spec["variants"][1]["train_command"].replace(
                    "evidence_half_life=null", "evidence_half_life=4.0"
                ),
            ),
            "evidence_half_life=null",
        ),
        (
            lambda spec: spec["variants"][1].__setitem__(
                "train_command",
                spec["variants"][1]["train_command"].replace(
                    " ++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
                    "",
                ),
            ),
            "shared motion_lib_cfg.sort_motion_keys=true",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "eval_command",
                    variant["eval_command"].replace(
                        "callbacks.im_eval.max_eval_steps=null",
                        "callbacks.im_eval.max_eval_steps=64",
                    ),
                )
                for variant in spec["variants"]
            ],
            "callbacks.im_eval.max_eval_steps=null",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "eval_command",
                    variant["eval_command"]
                    + " ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1",
                )
                for variant in spec["variants"]
            ],
            "full-sequence limiter",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "eval_command",
                    variant["eval_command"].replace(
                        "observations.policy.enable_corruption=False",
                        "observations.policy.enable_corruption=True",
                    ),
                )
                for variant in spec["variants"]
            ],
            "observations.policy.enable_corruption=False",
        ),
    ],
)
def test_official_sampler_pair_rejects_any_extra_causal_difference(
    tmp_path: Path,
    mutator,
    expected_error: str,
) -> None:
    spec = _official_sampler_pair_spec(tmp_path)
    mutator(spec)

    assert any(expected_error in error for error in validate_spec(spec))


def test_activation_contract_rejects_output_initialization_collision(
    tmp_path: Path,
) -> None:
    spec = _activation_preflight_spec(tmp_path)
    initialization = spec["training_initialization_checkpoint"]
    checkpoint = spec["variants"][0]["checkpoint"]
    spec["variants"][0]["checkpoint"] = initialization
    spec["variants"][0]["eval_command"] = spec["variants"][0]["eval_command"].replace(
        f"+checkpoint={checkpoint}", f"+checkpoint={initialization}"
    )

    assert any("must not collide" in error for error in validate_spec(spec))


def test_activation_preflight_hash_binds_inputs_and_active_batch(
    tmp_path: Path,
) -> None:
    spec = _activation_preflight_spec(tmp_path)

    provenance = paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)

    assert provenance["all_candidate_motions_active"] is True
    assert provenance["training_num_envs"] == 1
    assert provenance["dataset_manifest"]["verified_file_count"] == 2
    assert provenance["dataset_inventory"]["exact_direct_child_pkl_inventory"] is True
    assert provenance["dataset_inventory"]["flat_direct_children"] is True
    assert len(provenance["command_dataset_bindings"]) == 4
    assert provenance["sim_d1_classification"]["verdict"] == "PASS"
    assert provenance["zpd_treatment_sampler_config"] == {
        "variant": "learnability",
        "enable": True,
        "signal": "learnability",
        "optimism_k": 0.0,
        "evidence_half_life": 4.0,
        "advmass_n": 16,
        "uniform_sampling_rate": 0.1,
        "tripwire_max_prob_over_uniform": 20.0,
        "bin_size": 50,
        "paired_dataset_sha256": provenance["dataset_manifest"]["paired_dataset_sha256"],
        "all_dump_bound_fields_explicit_once": True,
    }
    assert provenance["difficulty_ranking"]["exact_dataset_coverage"] is True
    assert provenance["training_initialization_checkpoint"]["hash_matches_sim_d1_source_checkpoint"] is True
    assert provenance["training_python_executable"]["entrypoints_exactly_bound"] is True

    spec["training_num_envs"] = 2
    with pytest.raises(ValueError, match="must equal the hash-bound dataset motion count"):
        paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)
    spec["training_num_envs"] = 1
    spec["difficulty_ranking_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="difficulty ranking SHA-256 mismatch"):
        paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)


def test_initialization_recheck_detects_post_preflight_mutation(tmp_path: Path) -> None:
    spec = _activation_preflight_spec(tmp_path)
    provenance = paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)
    Path(spec["training_initialization_checkpoint"]).write_bytes(b"mutated")

    with pytest.raises(ValueError, match="changed after preflight"):
        paired_runner._verify_initialization_checkpoint_unchanged(
            spec, repo_root=tmp_path, input_provenance=provenance
        )


def test_direct_paired_preflight_rejects_nested_smpl_inventory(tmp_path: Path) -> None:
    spec = _activation_preflight_spec(tmp_path)
    nested = tmp_path / "smpl" / "nested"
    nested.mkdir()
    (nested / "loader_invisible.pkl").write_bytes(b"invisible")

    with pytest.raises(ValueError, match="must be flat"):
        paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)


def test_activation_preflight_rejects_current_sim_d1_fail_artifact(
    tmp_path: Path,
) -> None:
    spec = _activation_preflight_spec(tmp_path)
    fail_artifact = (
        Path(__file__).resolve().parents[2]
        / "docs/artifacts/sim_d1/d_b_synthetic_release_retro/classification.json"
    )
    spec["sim_d1_classification_json"] = str(fail_artifact)
    spec["sim_d1_classification_sha256"] = _sha256(fail_artifact)
    with pytest.raises(ValueError, match="must have verdict PASS"):
        paired_runner._verify_activation_preflight(spec, repo_root=tmp_path)


def test_activation_preflight_rejects_duplicate_or_wrong_command_dataset_binding(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate = _activation_preflight_spec(duplicate_root)
    duplicate["variants"][0]["train_command"] += " ++manager_env.commands.motion.motion_lib_cfg.motion_file=robot"
    with pytest.raises(ValueError, match="must contain exactly one.*motion_file"):
        paired_runner._verify_activation_preflight(duplicate, repo_root=duplicate_root)

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    wrong = _activation_preflight_spec(wrong_root)
    wrong["variants"][0]["eval_command"] = wrong["variants"][0]["eval_command"].replace(
        "motion_file=robot", "motion_file=other_robot"
    )
    with pytest.raises(ValueError, match="expected verified path"):
        paired_runner._verify_activation_preflight(wrong, repo_root=wrong_root)


def test_zpd_preflight_binds_every_treatment_sampler_field_once(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate_sampler"
    duplicate_root.mkdir()
    duplicate = _activation_preflight_spec(duplicate_root)
    duplicate["variants"][0]["train_command"] += (
        " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.optimism_k=1.0"
    )
    with pytest.raises(ValueError, match="optimism_k exactly once"):
        paired_runner._verify_activation_preflight(duplicate, repo_root=duplicate_root)

    missing_root = tmp_path / "missing_sampler"
    missing_root.mkdir()
    missing = _activation_preflight_spec(missing_root)
    missing["variants"][0]["train_command"] = missing["variants"][0]["train_command"].replace(
        " ++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.bin_size=50",
        "",
    )
    with pytest.raises(ValueError, match="bin_size exactly once"):
        paired_runner._verify_activation_preflight(missing, repo_root=missing_root)

    digest_root = tmp_path / "wrong_digest"
    digest_root.mkdir()
    wrong_digest = _activation_preflight_spec(digest_root)
    verified_digest = json.loads((digest_root / "dataset_manifest.json").read_text(encoding="utf-8"))["output"][
        "paired_dataset_sha256"
    ]
    wrong_digest["variants"][0]["train_command"] = wrong_digest["variants"][0]["train_command"].replace(
        verified_digest, "0" * 64
    )
    with pytest.raises(ValueError, match="does not match the verified dataset manifest"):
        paired_runner._verify_activation_preflight(wrong_digest, repo_root=digest_root)

    signal_root = tmp_path / "wrong_signal"
    signal_root.mkdir()
    wrong_signal = _activation_preflight_spec(signal_root)
    wrong_signal["variants"][0]["train_command"] = wrong_signal["variants"][0]["train_command"].replace(
        "signal=learnability", "signal=advantage_mass"
    )
    with pytest.raises(ValueError, match="signal=learnability"):
        paired_runner._verify_activation_preflight(wrong_signal, repo_root=signal_root)


def test_exhaustive_inventory_rejects_extra_missing_duplicate_and_stem_mismatch(
    tmp_path: Path,
) -> None:
    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra_spec = _activation_preflight_spec(extra_root)
    (extra_root / "robot/extra.pkl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory is not exhaustive"):
        paired_runner._verify_activation_preflight(extra_spec, repo_root=extra_root)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_spec = _activation_preflight_spec(missing_root)
    (missing_root / "smpl/motion.pkl").unlink()
    with pytest.raises(ValueError, match="inventory is not exhaustive"):
        paired_runner._verify_activation_preflight(missing_spec, repo_root=missing_root)

    duplicate_root = tmp_path / "duplicate_stem"
    duplicate_root.mkdir()
    duplicate_spec = _activation_preflight_spec(duplicate_root)
    nested = duplicate_root / "robot/nested"
    nested.mkdir()
    (nested / "motion.pkl").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="must be flat"):
        paired_runner._verify_activation_preflight(duplicate_spec, repo_root=duplicate_root)

    mismatch_root = tmp_path / "stem_mismatch"
    mismatch_root.mkdir()
    _activation_preflight_spec(mismatch_root)
    manifest = json.loads((mismatch_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["output"]["variants"][0]["motion_key"] = "not_the_file_stem"
    with pytest.raises(ValueError, match="motion_key/file-stem mismatch"):
        paired_runner._verify_exhaustive_dataset_inventory(
            manifest,
            dataset_robot=mismatch_root / "robot",
            dataset_smpl=mismatch_root / "smpl",
        )


def test_metrics_motion_coverage_requires_exact_manifest_key_set(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"eval/all_metrics_dict": {"motion_keys": ["a", "b"]}}),
        encoding="utf-8",
    )
    coverage = paired_runner._verify_metrics_motion_coverage(metrics_path, ["b", "a"])
    assert coverage["exact_motion_key_coverage"] is True

    metrics_path.write_text(
        json.dumps({"eval/all_metrics_dict": {"motion_keys": ["a", "extra"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"missing=\['b'\], extras=\['extra'\]"):
        paired_runner._verify_metrics_motion_coverage(metrics_path, ["a", "b"])


def test_activation_execute_blocks_comparison_on_inexact_eval_metric_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _activation_preflight_spec(tmp_path)
    checkpoint = tmp_path / "outputs/learnability/last.pt"
    metrics_path = tmp_path / "outputs/learnability/eval_metrics/metrics_eval.json"

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "train_agent_trl.py" in command:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"fresh checkpoint")
            log_path.write_text(
                "Learning iteration 1\nMean rewards: 1.0\nTotal timesteps: 1\n",
                encoding="utf-8",
            )
        else:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "eval/all_metrics_dict": {
                            "motion_keys": ["wrong_motion"],
                            "mpjpe_g": [1.0],
                            "terminated": [0.0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            log_path.write_text(
                "All:  mpjpe_g: 1.0 mpjpe_l: 2.0 mpjpe_pa: 3.0\n",
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        spec,
        output_dir=tmp_path / "run",
        dry_run=False,
        repo_root=tmp_path,
    )

    assert plan["execution_ok"] is False
    assert plan["comparison_json"] is None
    assert plan["training_completeness_ok"] is True
    assert plan["variants"][0]["training_completion"] == {
        "required": True,
        "expected_learning_iteration": 1,
        "observed_learning_iteration": 1,
        "complete": True,
        "parser_semantics": "one_based_terminal_learning_iteration",
    }
    assert "do not exactly cover" in plan["variants"][0]["execution_errors"][0]


def test_activation_execute_blocks_eval_when_training_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _activation_preflight_spec(tmp_path)
    calls: list[str] = []

    def fake_run(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
        del cwd
        calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Learning iteration 0\nMean rewards: 1.0\nTotal timesteps: 1\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(paired_runner, "_run_command", fake_run)
    plan = materialize_paired_experiment(
        spec,
        output_dir=tmp_path / "run",
        dry_run=False,
        repo_root=tmp_path,
    )

    assert len(calls) == 1
    assert "train_agent_trl.py" in calls[0]
    assert plan["execution_ok"] is False
    assert plan["training_completeness_ok"] is False
    completion = plan["variants"][0]["training_completion"]
    assert completion["expected_learning_iteration"] == 1
    assert completion["observed_learning_iteration"] == 0
    assert completion["complete"] is False
    assert "exact preregistered terminal" in plan["variants"][0]["execution_errors"][0]
