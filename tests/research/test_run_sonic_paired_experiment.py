from __future__ import annotations

import json
from pathlib import Path

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
