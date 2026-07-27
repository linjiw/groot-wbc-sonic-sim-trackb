from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest
import torch

from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback
from gear_sonic.utils.root_xy_diagnostic import (
    root_xy_distance,
    root_xy_guard_exceeded,
    summarize_root_xy_error_batch,
)
from scripts.research.run_sim_d1_all_motion_eval import (
    _canonical_dataset_manifest_sha256,
)
from scripts.research.run_sim_d2_root_xy_diagnostic import (
    ROOT_XY_CALLBACK_OVERRIDE,
    ROOT_XY_TERM_OVERRIDE,
    _validate_root_xy_metrics,
    run_sim_d2_root_xy_diagnostic,
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


def _fixture(tmp_path: Path) -> tuple[Path, dict, Path, Path, list[str]]:
    repo = tmp_path / "repo"
    (repo / "gear_sonic").mkdir(parents=True)
    (repo / "gear_sonic/eval_agent_trl.py").write_text("", encoding="utf-8")
    checkpoint = repo / "sonic_release/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"release checkpoint fixture")
    robot_root = repo / "data/robot"
    smpl_root = repo / "data/smpl"
    robot_root.mkdir(parents=True)
    smpl_root.mkdir(parents=True)
    keys = ["walk_a", "walk_b", "turn_c"]

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
        "generator": "test_run_sim_d2_root_xy_diagnostic.py",
        "source": {"robot_dir": "/host/robot", "smpl_dir": "/host/smpl"},
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
    configs = repo / "configs"
    configs.mkdir()
    manifest_path = configs / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binding = {
        "path": str(manifest_path),
        "sha256": _canonical_dataset_manifest_sha256(manifest),
        "sha256_kind": "canonical_json_without_source_locations_v1",
        "paired_dataset_sha256": manifest["output"]["paired_dataset_sha256"],
    }
    coverage_path = configs / "expected.json"
    coverage_path.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "expected_motion_count": len(keys),
                "motion_keys": keys,
                "provenance": {
                    "source": "frozen test inventory",
                    "independent_of_metrics_eval": True,
                    "dataset_manifest": binding,
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_d1 = repo / "outputs/research/sim_d1/fixture"
    d2_output = repo / "outputs/research/sim_d2_root_xy/fixture"
    spec = {
        "name": "fixture_d1",
        "dataset": "fixture",
        "dataset_kind": "real",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_robot": str(robot_root),
        "dataset_smpl": str(smpl_root),
        "coverage_manifest": str(coverage_path),
        "dataset_manifest": binding,
        "output_dir": str(canonical_d1),
        "num_envs": 1,
        "seed": 0,
        "timeout_seconds": 30,
        "motion_order": "coverage_manifest",
        "extra_overrides": [],
    }
    return repo, spec, canonical_d1, d2_output, keys


def _valid_metrics(keys: list[str]) -> dict:
    return {
        "eval/root_xy_diagnostic_threshold_m": 0.25,
        "eval/root_xy_diagnostic_exact_guard_samples": True,
        "eval/all_metrics_dict": {
            "motion_keys": list(keys),
            "root_xy_error_mean_m": [0.10, 0.20, 0.26],
            "root_xy_error_p95_m": [0.18, 0.24, 0.29],
            "root_xy_error_max_m": [0.20, 0.25, 0.30],
            "root_xy_guard_hit": [False, False, True],
            "root_xy_first_crossing_frame": [-1, -1, 4],
            "root_xy_first_crossing_progress": [-1.0, -1.0, 0.5],
        },
    }


def _write_metrics_from_command(command: list[str], metrics: dict) -> None:
    value = next(item for item in command if item.startswith("++eval_output_dir="))
    output_dir = Path(value.split("=", 1)[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_eval.json").write_text(json.dumps(metrics), encoding="utf-8")


class TestRootXYTerminationSemantics:
    def test_xy_only_distance_and_strict_threshold(self) -> None:
        reference = torch.tensor(
            [[0.0, 0.0, 99.0], [0.0, 0.0, -99.0], [1.0, 2.0, 3.0]]
        )
        robot = torch.tensor(
            [[0.15, 0.20, -99.0], [0.0, 0.251, 99.0], [1.0, 2.0, -30.0]]
        )

        error = root_xy_distance(reference, robot)
        exceeded = root_xy_guard_exceeded(error, threshold_m=0.25)

        assert error.tolist() == pytest.approx([0.25, 0.251, 0.0])
        assert exceeded.tolist() == [False, True, False]


class TestRootXYConfigComposition:
    def test_d2_adds_xy_term_while_preserving_z_height_term(self) -> None:
        config_dir = Path(__file__).parents[2] / "gear_sonic/config"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(
                config_name="base_eval",
                overrides=[
                    "+manager_env/terminations=tracking/eval",
                    ROOT_XY_TERM_OVERRIDE,
                    ROOT_XY_CALLBACK_OVERRIDE,
                ],
            )

        assert config.manager_env.terminations.anchor_pos.func.endswith(
            ":exceeded_anchor_height"
        )
        assert config.manager_env.terminations.anchor_pos.params.threshold == 0.25
        assert config.manager_env.terminations.anchor_pos_xy.func.endswith(
            ":exceeded_anchor_pos_xy"
        )
        assert config.manager_env.terminations.anchor_pos_xy.params.threshold == 0.25
        assert config.callbacks.im_eval.root_xy_diagnostic_threshold_m == 0.25


class TestRootXYTelemetryAggregation:
    def test_exact_aggregates_and_first_strict_crossing(self) -> None:
        errors = torch.tensor(
            [
                [0.10, 0.00],
                [0.25, 0.25],
                [0.30, 0.20],
                [0.50, 0.90],
            ],
            dtype=torch.float64,
        )

        summary = summarize_root_xy_error_batch(
            errors, [5, 4], threshold_m=0.25
        )

        assert summary["root_xy_error_mean_m"].tolist() == pytest.approx(
            [0.2875, 0.15]
        )
        assert summary["root_xy_error_p95_m"].tolist() == pytest.approx(
            [0.47, 0.245]
        )
        assert summary["root_xy_error_max_m"].tolist() == [0.5, 0.25]
        assert summary["root_xy_guard_hit"].tolist() == [True, False]
        assert summary["root_xy_first_crossing_frame"].tolist() == [2, -1]
        assert summary["root_xy_first_crossing_progress"].tolist() == [0.75, -1.0]

    def test_default_off_callback_does_not_change_metrics_schema(self) -> None:
        callback = ImEvalCallback(eval_frequency=1)
        metrics = callback._post_evaluate_policy(
            {
                "metrics_success": {},
                "metrics_all": {},
                "all_metrics_dict": {"motion_keys": []},
                "failed_metrics_dict": {"motion_keys": []},
            }
        )

        assert not any("root_xy" in key for key in metrics)
        assert callback.root_xy_diagnostic_threshold_m is None

    def test_callback_consumes_the_exact_guard_tensor(self) -> None:
        callback = ImEvalCallback(
            eval_frequency=1, root_xy_diagnostic_threshold_m=0.25
        )
        exact_error = torch.tensor([0.125, 0.375])
        command = SimpleNamespace(
            _root_xy_diagnostic_last_error=exact_error,
            anchor_pos_w=torch.full((2, 3), 100.0),
            robot_anchor_pos_w=torch.zeros(2, 3),
        )
        callback.env = SimpleNamespace(motion_command=command)
        callback.root_xy_error = []
        callback.root_xy_diagnostic_exact_guard_samples = True

        callback._collect_root_xy_diagnostic_error()

        assert torch.equal(callback.root_xy_error[0], exact_error)
        assert callback.root_xy_diagnostic_exact_guard_samples is True
        assert not hasattr(command, "_root_xy_diagnostic_last_error")


class TestD2DryRunIsolation:
    def test_dry_run_isolated_and_never_launches(self, tmp_path: Path, monkeypatch) -> None:
        repo, spec, canonical_d1, output, _ = _fixture(tmp_path)
        canonical_d1.mkdir(parents=True)
        canonical_result = canonical_d1 / "result.json"
        canonical_result.write_text('{"status":"classified"}\n', encoding="utf-8")

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("D2 dry-run launched evaluation")
            ),
        )
        result = run_sim_d2_root_xy_diagnostic(
            spec, output_dir=output, repo_root=repo, dry_run=True
        )

        assert result["ready_to_execute"] is True
        assert result["eligible_for_effect_experiment"] is False
        assert result["safeguards"]["z_height_guard_preserved"] is True
        assert result["safeguards"]["root_xy_guard_added"] is True
        assert canonical_result.read_text(encoding="utf-8") == (
            '{"status":"classified"}\n'
        )
        assert (output / "dry_run_plan.json").is_file()
        assert not (output / "classification.json").exists()
        assert not (output / "root_xy_diagnostic.json").exists()

    def test_canonical_d1_output_path_is_refused(self, tmp_path: Path) -> None:
        repo, spec, canonical_d1, _, _ = _fixture(tmp_path)
        with pytest.raises(ValueError, match="canonical SIM-D1 path"):
            run_sim_d2_root_xy_diagnostic(
                spec,
                output_dir=canonical_d1,
                repo_root=repo,
                dry_run=True,
            )


class TestD2CanonicalImmutability:
    def test_execute_never_classifies_or_mutates_d1(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        repo, spec, canonical_d1, output, keys = _fixture(tmp_path)
        canonical_d1.mkdir(parents=True)
        canonical_files = {
            canonical_d1 / "classification.json": '{"verdict":"PASS"}\n',
            canonical_d1 / "result.json": '{"status":"classified"}\n',
            canonical_d1 / "eval_metrics/metrics_eval.json": '{"old":true}\n',
        }
        for path, content in canonical_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        import scripts.research.run_sim_d1_all_motion_eval as d1_runner

        monkeypatch.setattr(
            d1_runner,
            "classify_sim_d1",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("D2 invoked the D1 classifier")
            ),
        )

        def successful_run(command, **kwargs):
            assert kwargs["cwd"] == repo
            assert kwargs["env"]["WANDB_MODE"] == "disabled"
            _write_metrics_from_command(command, _valid_metrics(keys))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", successful_run)
        result = run_sim_d2_root_xy_diagnostic(
            spec, output_dir=output, repo_root=repo, dry_run=False
        )

        assert result["ok"] is True
        assert result["status"] == "diagnostic_complete"
        assert result["eligible_for_effect_experiment"] is False
        assert not (output / "classification.json").exists()
        for path, content in canonical_files.items():
            assert path.read_text(encoding="utf-8") == content


class TestD2ArtifactSchema:
    @pytest.mark.parametrize(
        "mutation,match",
        [
            (
                lambda value: value["eval/all_metrics_dict"].pop(
                    "root_xy_error_p95_m"
                ),
                "root_xy_error_p95_m",
            ),
            (
                lambda value: value["eval/all_metrics_dict"][
                    "root_xy_error_max_m"
                ].__setitem__(0, float("nan")),
                "must be finite",
            ),
            (
                lambda value: value["eval/all_metrics_dict"]["motion_keys"].reverse(),
                "frozen coverage order",
            ),
            (
                lambda value: value["eval/all_metrics_dict"][
                    "root_xy_guard_hit"
                ].__setitem__(2, False),
                "no-crossing sentinel",
            ),
        ],
    )
    def test_malformed_telemetry_fails_closed(self, mutation, match: str) -> None:
        keys = ["walk_a", "walk_b", "turn_c"]
        metrics = _valid_metrics(keys)
        mutation(metrics)

        with pytest.raises(ValueError, match=match):
            _validate_root_xy_metrics(metrics, expected_motion_keys=keys)

    def test_artifact_is_explicitly_diagnostic_only(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        repo, spec, _, output, keys = _fixture(tmp_path)

        def successful_run(command, **_kwargs):
            _write_metrics_from_command(command, _valid_metrics(keys))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess, "run", successful_run)
        result = run_sim_d2_root_xy_diagnostic(
            spec, output_dir=output, repo_root=repo, dry_run=False
        )
        artifact = json.loads(
            (output / "root_xy_diagnostic.json").read_text(encoding="utf-8")
        )

        assert result["eligible_for_effect_experiment"] is False
        assert artifact["eligible_for_effect_experiment"] is False
        assert artifact["scientific_status"] == (
            "diagnostic_only_not_an_effect_experiment"
        )
        assert artifact["summary"] == {
            "motion_count": 3,
            "guard_hit_count": 1,
            "guard_hit_fraction": pytest.approx(1 / 3),
        }
        assert [row["motion_key"] for row in artifact["per_motion"]] == keys
        assert artifact["safeguards"]["d1_classifier_invoked"] is False
        assert artifact["canonical_d1_immutability"]["unchanged"] is True
