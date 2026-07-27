from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.research.run_sim_d1_all_motion_eval import (
    _canonical_dataset_manifest_sha256,
    build_eval_command,
    run_sim_d1_all_motion_eval,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_hash(records: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(records):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _passing_metrics(keys: list[str]) -> dict:
    count = len(keys)
    assert count == 10
    return {
        "eval/all_metrics_dict": {
            "motion_keys": keys,
            "mpjpe_g": [float(i * 10) for i in range(count)],
            "terminated": [True, True] + [False] * 8,
            "progress": [0.5, 0.8] + [1.0] * 8,
        }
    }


def _fixture(tmp_path: Path, *, keys: list[str] | None = None):
    repo = tmp_path / "repo"
    (repo / "gear_sonic").mkdir(parents=True)
    (repo / "gear_sonic/eval_agent_trl.py").write_text("", encoding="utf-8")
    checkpoint = repo / "sonic_release/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"release checkpoint fixture")
    robot = repo / "data/robot"
    smpl = repo / "data/smpl"
    robot.mkdir(parents=True)
    smpl.mkdir(parents=True)
    keys = keys or [f"m{i}" for i in range(10)]
    coverage_path = repo / "configs/expected.json"
    coverage_path.parent.mkdir()
    coverage_path.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "expected_motion_count": len(keys),
                "motion_keys": keys,
                "provenance": {
                    "source": "frozen test inventory",
                    "independent_of_metrics_eval": True,
                },
            }
        ),
        encoding="utf-8",
    )
    output = repo / "outputs/sim_d1"
    spec = {
        "name": "fixture",
        "dataset": "fixture",
        "dataset_kind": "synthetic",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_robot": str(robot),
        "dataset_smpl": str(smpl),
        "coverage_manifest": str(coverage_path),
        "output_dir": str(output),
        "num_envs": 2,
        "seed": 0,
        "timeout_seconds": 30,
        "extra_overrides": [],
    }
    return repo, spec, output, keys


def _bind_dataset_manifest(repo: Path, spec: dict, keys: list[str]) -> Path:
    robot_root = Path(spec["dataset_robot"])
    smpl_root = Path(spec["dataset_smpl"])
    robot_records: list[tuple[str, str]] = []
    smpl_records: list[tuple[str, str]] = []
    variants = []
    for key in keys:
        robot_path = robot_root / f"{key}.pkl"
        smpl_path = smpl_root / f"{key}.pkl"
        robot_path.write_bytes(f"robot:{key}".encode())
        smpl_path.write_bytes(f"smpl:{key}".encode())
        robot_relative = f"robot/{key}.pkl"
        smpl_relative = f"smpl/{key}.pkl"
        robot_hash = _sha256(robot_path)
        smpl_hash = _sha256(smpl_path)
        robot_records.append((robot_relative, robot_hash))
        smpl_records.append((smpl_relative, smpl_hash))
        variants.append(
            {
                "motion_key": key,
                "robot": {"path": robot_relative, "sha256": robot_hash},
                "smpl": {"path": smpl_relative, "sha256": smpl_hash},
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "test_paired_dataset_inventory",
        "generator": "tests/research/test_run_sim_d1_all_motion_eval.py",
        "source": {"robot_dir": "/host/a", "smpl_dir": "/host/b"},
        "output": {
            "robot_dir": "robot",
            "smpl_dir": "smpl",
            "motion_count": len(keys),
            "motion_keys": keys,
            "robot_dataset_sha256": _dataset_hash(robot_records),
            "smpl_dataset_sha256": _dataset_hash(smpl_records),
            "paired_dataset_sha256": _dataset_hash(robot_records + smpl_records),
            "variants": variants,
        },
    }
    manifest_path = repo / "configs/dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binding = {
        "path": str(manifest_path),
        "sha256": _canonical_dataset_manifest_sha256(manifest),
        "sha256_kind": "canonical_json_without_source_locations_v1",
        "paired_dataset_sha256": manifest["output"]["paired_dataset_sha256"],
    }
    spec["dataset_manifest"] = binding
    coverage_path = Path(spec["coverage_manifest"])
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["provenance"]["dataset_manifest"] = binding
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    return manifest_path


def test_command_has_all_motion_full_sequence_contract(tmp_path: Path) -> None:
    command = build_eval_command(
        python_executable="python",
        repo_root=tmp_path,
        checkpoint=tmp_path / "sonic_release/last.pt",
        dataset_robot=tmp_path / "robot",
        dataset_smpl=tmp_path / "smpl",
        eval_output_dir=tmp_path / "out/eval_metrics",
        num_envs=8,
        seed=0,
    )

    assert not any("max_unique_motions" in value for value in command)
    step_overrides = [value for value in command if "max_eval_steps" in value]
    assert step_overrides == ["++callbacks.im_eval.max_eval_steps=null"]
    assert any("eval_output_dir=" in value for value in command)
    assert "++seed=0" in command
    assert command[0] == "python"


@pytest.mark.parametrize(
    "override",
    [
        "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2",
        "++callbacks.im_eval.max_eval_steps=200",
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[x]",
    ],
)
def test_forbidden_selection_or_truncation_override_fails(tmp_path: Path, override: str) -> None:
    repo, spec, _, _ = _fixture(tmp_path)
    spec["extra_overrides"] = [override]

    with pytest.raises(ValueError, match="all-motion contract"):
        run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)


def _write_motion_key_files(spec: dict, keys: list[str]) -> None:
    for directory_field in ("dataset_robot", "dataset_smpl"):
        directory = Path(spec[directory_field])
        for key in keys:
            (directory / f"{key}.pkl").write_bytes(key.encode())


def test_coverage_manifest_order_is_exact_order_not_filtering(tmp_path: Path) -> None:
    keys = ["z_motion", "a_motion", "m_motion"]
    repo, spec, _, _ = _fixture(tmp_path, keys=keys)
    spec["num_envs"] = 1
    spec["motion_order"] = "coverage_manifest"
    _write_motion_key_files(spec, keys)

    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)

    assert result["ready_to_execute"] is True
    expected_list = json.dumps(keys, separators=(",", ":"))
    assert (
        "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
        + expected_list
    ) in result["command"]
    assert (
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=false"
        in result["command"]
    )
    assert result["motion_order"]["motion_keys"] == keys
    assert result["motion_order"]["exact_list_filter_is_order_only"] is True
    assert result["safeguards"]["coverage_order_exact_filter_verified"] is True


def test_coverage_manifest_order_fails_on_dataset_key_mismatch(tmp_path: Path) -> None:
    keys = ["m0", "m1", "m2"]
    repo, spec, _, _ = _fixture(tmp_path, keys=keys)
    spec["num_envs"] = 1
    spec["motion_order"] = "coverage_manifest"
    _write_motion_key_files(spec, keys)
    (Path(spec["dataset_smpl"]) / "m2.pkl").unlink()
    (Path(spec["dataset_smpl"]) / "unexpected.pkl").write_bytes(b"unexpected")

    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)

    assert result["ready_to_execute"] is False
    assert any(
        "motion-order validation failed" in error
        and "missing=['m2']" in error
        and "unexpected=['unexpected']" in error
        for error in result["preflight_errors"]
    )


def test_coverage_manifest_order_rejects_multiple_envs_or_sort_override(
    tmp_path: Path,
) -> None:
    repo, spec, _, _ = _fixture(tmp_path)
    spec["motion_order"] = "coverage_manifest"

    with pytest.raises(ValueError, match="requires num_envs=1"):
        run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)

    spec["num_envs"] = 1
    spec["extra_overrides"] = [
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true"
    ]
    with pytest.raises(ValueError, match="owns sort_motion_keys"):
        run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)


def test_dry_run_is_parameterized_and_does_not_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, spec, output, _ = _fixture(tmp_path)
    output.mkdir(parents=True)
    canonical_contents = {
        output / "classification.json": "old classification\n",
        output / "result.json": "old result\n",
        output / "eval.log": "old eval log\n",
        output / "run_plan.json": "old execute plan\n",
    }
    metrics_path = output / "eval_metrics/metrics_eval.json"
    metrics_path.parent.mkdir()
    canonical_contents[metrics_path] = "old metrics\n"
    for path, content in canonical_contents.items():
        path.write_text(content, encoding="utf-8")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry-run launched evaluation")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_sim_d1_all_motion_eval(
        spec,
        repo_root=repo,
        python_executable="/custom/python",
        dry_run=True,
    )

    assert result["ready_to_execute"] is True
    assert result["command"][0] == "/custom/python"
    assert result["outputs"]["metrics_eval_json"] == str(output / "eval_metrics/metrics_eval.json")
    assert result["coverage_manifest"]["expected_motion_count"] == 10
    assert result["safeguards"]["paired_dataset_content_is_hashed"] is False
    assert (output / "dry_run_plan.json").is_file()
    assert not (output / "archive").exists()
    for path, content in canonical_contents.items():
        assert path.read_text(encoding="utf-8") == content


def test_execute_classifies_only_fresh_complete_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, spec, output, keys = _fixture(tmp_path)
    metrics_path = output / "eval_metrics/metrics_eval.json"
    output.mkdir(parents=True)
    prior_classification = output / "classification.json"
    prior_classification.write_text('{"verdict": "OLD"}\n', encoding="utf-8")

    def successful_run(command, **kwargs):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(_passing_metrics(keys)), encoding="utf-8")
        assert kwargs["cwd"] == repo
        assert kwargs["env"]["WANDB_MODE"] == "disabled"
        assert "PYTHONPATH" not in kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", successful_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["ok"] is True
    assert result["verdict"] == "PASS"
    classification = json.loads((output / "classification.json").read_text(encoding="utf-8"))
    assert classification["eligible_for_effect_experiment"] is True
    assert classification["coverage"]["exact_key_set_verified"] is True
    assert classification["source"]["expected_motion_count"] == 10
    assert len(classification["source"]["coverage_manifest_sha256"]) == 64
    archived = result["archived_prior_outputs"]
    archived_classification = next(record for record in archived if record["label"] == "classification")
    assert Path(archived_classification["archived_path"]).read_text(encoding="utf-8") == '{"verdict": "OLD"}\n'
    assert Path(archived_classification["provenance_path"]).is_file()
    assert classification["source"]["execute_attempt_id"] == result["attempt_id"]


def test_nonzero_eval_never_classifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, spec, output, keys = _fixture(tmp_path)
    metrics_path = output / "eval_metrics/metrics_eval.json"
    output.mkdir(parents=True)
    prior_classification = output / "classification.json"
    prior_classification.write_text('{"verdict": "OLD"}\n', encoding="utf-8")

    def failed_run(*args, **kwargs):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(_passing_metrics(keys)), encoding="utf-8")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", failed_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "evaluation_failed"
    assert result["exit_code"] == 1
    assert not (output / "classification.json").exists()
    assert not metrics_path.exists()
    archived_classification = next(
        record for record in result["archived_prior_outputs"] if record["label"] == "classification"
    )
    assert Path(archived_classification["archived_path"]).read_text(encoding="utf-8") == '{"verdict": "OLD"}\n'
    assert Path(archived_classification["provenance_path"]).is_file()
    quarantined_metrics = result["quarantined_attempt_outputs"][0]
    assert quarantined_metrics["reason"] == "evaluation_returned_nonzero_or_timed_out"
    assert Path(quarantined_metrics["archived_path"]).is_file()
    assert (output / "eval.log").is_file()
    assert (output / "result.json").is_file()


def test_preflight_failure_preserves_prior_canonical_outputs_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, spec, output, _ = _fixture(tmp_path)
    output.mkdir(parents=True)
    classification_path = output / "classification.json"
    classification_path.write_text('{"verdict": "OLD"}\n', encoding="utf-8")
    result_path = output / "result.json"
    result_path.write_text('{"status": "OLD"}\n', encoding="utf-8")
    Path(spec["checkpoint"]).unlink()

    def unexpected_run(*args, **kwargs):
        raise AssertionError("preflight failure launched evaluation")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "preflight_failed"
    assert result["exit_code"] == 2
    assert classification_path.read_text(encoding="utf-8") == '{"verdict": "OLD"}\n'
    assert result_path.read_text(encoding="utf-8") == '{"status": "OLD"}\n'
    assert result["archived_prior_outputs"] == []
    assert result["canonical_outputs_preserved"] is True
    assert not (output / "archive").exists()


def test_stale_metrics_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, spec, output, keys = _fixture(tmp_path)
    metrics_path = output / "eval_metrics/metrics_eval.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps(_passing_metrics(keys)), encoding="utf-8")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "metrics_missing_or_stale"
    assert result["exit_code"] == 1
    assert not (output / "classification.json").exists()
    assert not metrics_path.exists()
    archived_metrics = next(
        record for record in result["archived_prior_outputs"] if record["label"] == "metrics_eval"
    )
    assert Path(archived_metrics["archived_path"]).is_file()


def test_incomplete_coverage_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, spec, output, keys = _fixture(tmp_path)
    metrics_path = output / "eval_metrics/metrics_eval.json"

    def incomplete_run(*args, **kwargs):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                _passing_metrics(keys)
                | {
                    "eval/all_metrics_dict": {
                        key: values[:-1] for key, values in _passing_metrics(keys)["eval/all_metrics_dict"].items()
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", incomplete_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "classification_failed"
    assert "coverage mismatch" in result["error"]
    assert not (output / "classification.json").exists()
    assert not metrics_path.exists()
    assert Path(result["quarantined_attempt_outputs"][0]["archived_path"]).is_file()


def test_coverage_manifest_must_be_external_to_dataset(tmp_path: Path) -> None:
    repo, spec, _, _ = _fixture(tmp_path)
    robot_dir = Path(spec["dataset_robot"])
    internal = robot_dir / "expected.json"
    internal.write_text(
        Path(spec["coverage_manifest"]).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    spec["coverage_manifest"] = str(internal)

    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)

    assert result["ready_to_execute"] is False
    assert any(
        "outside both dataset directories" in error
        for error in result["preflight_errors"]
    )


def test_optional_dataset_manifest_verifies_all_files(tmp_path: Path) -> None:
    repo, spec, _, keys = _fixture(tmp_path)
    _bind_dataset_manifest(repo, spec, keys)

    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=True)

    assert result["ready_to_execute"] is True
    verified = result["dataset_manifest"]
    assert verified["content_hashes_verified"] is True
    assert verified["verified_file_count"] == 2 * len(keys)
    assert verified["paired_dataset_sha256"] == spec["dataset_manifest"][
        "paired_dataset_sha256"
    ]
    assert result["safeguards"]["paired_dataset_content_is_hashed"] is True


def test_dataset_file_hash_mismatch_preserves_prior_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, spec, output, keys = _fixture(tmp_path)
    _bind_dataset_manifest(repo, spec, keys)
    output.mkdir(parents=True)
    prior_result = output / "result.json"
    prior_result.write_text('{"status": "CLASSIFIED"}\n', encoding="utf-8")
    (Path(spec["dataset_robot"]) / f"{keys[0]}.pkl").write_bytes(b"tampered")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("invalid dataset launched evaluation")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "preflight_failed"
    assert any("dataset file SHA-256 mismatch" in error for error in result["errors"])
    assert prior_result.read_text(encoding="utf-8") == '{"status": "CLASSIFIED"}\n'
    assert result["archived_prior_outputs"] == []
    assert not (output / "archive").exists()


def test_malformed_coverage_preserves_prior_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, spec, output, _ = _fixture(tmp_path)
    output.mkdir(parents=True)
    prior_result = output / "result.json"
    prior_result.write_text('{"status": "CLASSIFIED"}\n', encoding="utf-8")
    Path(spec["coverage_manifest"]).write_text("{not-json", encoding="utf-8")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("malformed coverage launched evaluation")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "preflight_failed"
    assert any("coverage manifest validation failed" in error for error in result["errors"])
    assert prior_result.read_text(encoding="utf-8") == '{"status": "CLASSIFIED"}\n'
    assert not (output / "archive").exists()


def test_checkpoint_hash_mismatch_preserves_prior_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, spec, output, _ = _fixture(tmp_path)
    output.mkdir(parents=True)
    prior_result = output / "result.json"
    prior_result.write_text('{"status": "CLASSIFIED"}\n', encoding="utf-8")
    spec["checkpoint_sha256"] = "0" * 64

    def unexpected_run(*args, **kwargs):
        raise AssertionError("checkpoint hash mismatch launched evaluation")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    result = run_sim_d1_all_motion_eval(spec, repo_root=repo, dry_run=False)

    assert result["status"] == "preflight_failed"
    assert any("checkpoint SHA-256 mismatch" in error for error in result["errors"])
    assert prior_result.read_text(encoding="utf-8") == '{"status": "CLASSIFIED"}\n'
    assert not (output / "archive").exists()


def test_canonical_manifest_hash_ignores_only_source_locations() -> None:
    manifest = {
        "schema_version": 1,
        "kind": "fixture",
        "generator": "fixture",
        "source": {"robot_dir": "/machine/a", "smpl_dir": "/machine/b"},
        "output": {"paired_dataset_sha256": "a" * 64},
    }
    relocated = json.loads(json.dumps(manifest))
    relocated["source"]["robot_dir"] = "/other/robot"
    relocated["source"]["smpl_dir"] = "/other/smpl"
    changed = json.loads(json.dumps(manifest))
    changed["output"]["paired_dataset_sha256"] = "b" * 64

    assert _canonical_dataset_manifest_sha256(manifest) == _canonical_dataset_manifest_sha256(
        relocated
    )
    assert _canonical_dataset_manifest_sha256(manifest) != _canonical_dataset_manifest_sha256(
        changed
    )


def test_nominal_d1_specs_are_opt_in_and_remove_context_randomness() -> None:
    repo = Path(__file__).parents[2]
    required_overrides = {
        "+eval_events=nominal_d1",
        "++manager_env.config.terrain_type=plane",
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
        "++manager_env.observations.policy.enable_corruption=false",
        "++manager_env.observations.tokenizer.enable_corruption=false",
        "+use_encoder=g1",
    }
    for filename in (
        "sim_d1_sample_data_nominal_eval.json",
        "sim_d1_synthetic_nominal_eval.json",
    ):
        spec = json.loads((repo / "configs/research" / filename).read_text(encoding="utf-8"))
        assert spec["num_envs"] == 1
        assert required_overrides <= set(spec["extra_overrides"])
        assert "nominal_v1" in spec["output_dir"]
        assert spec["dataset_manifest"]["sha256_kind"] == (
            "canonical_json_without_source_locations_v1"
        )
