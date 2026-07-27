from __future__ import annotations

import json
from pathlib import Path

from scripts.research.compare_sonic_manifests import (
    build_comparison,
    write_comparison_markdown,
)


def _manifest(
    path: Path,
    *,
    experiment_id: str,
    variant: str,
    seed: int = 0,
    checkpoint: str = "sonic_release/last.pt",
) -> Path:
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "hypothesis": "curriculum improves motion tracking reliability",
        "variant": variant,
        "seed": seed,
        "status": "needs_review",
        "interpretation": "paper_harness_fixture",
        "git_commit": "abc1234",
        "controlled_variables": {
            "simulator": "IsaacLab / Isaac Sim headless",
            "env": "env_isaaclab",
            "robot": "Unitree G1 29-DoF dex model",
            "sonic_config": "manager/universal_token/all_modes/sonic_release",
            "action_interface": "64D SONIC motion token + 7D left hand + 7D right hand",
        },
        "datasets": {
            "robot_motion": "sample_data/robot_filtered",
            "smpl_motion": "sample_data/smpl_filtered",
        },
        "checkpoint": checkpoint,
        "commands": {
            "train": "python train.py",
            "eval": "python eval.py",
        },
        "artifacts": {
            "summary_json": "outputs/research/summary.json",
            "train_log": "outputs/research/train.log",
            "eval_log": "outputs/research/eval.log",
        },
        "metrics": {
            "train": {
                "ok": True,
                "learning_iteration": 100,
                "mean_rewards": 0.8 if variant == "baseline" else 0.9,
                "total_timesteps": 38400,
                "traceback_count": 0,
            },
            "eval": {
                "ok": True,
                "all": {"mpjpe_g": 130.0 if variant == "baseline" else 120.0, "mpjpe_l": 18.0, "mpjpe_pa": 12.0},
                "terminated_final": 1 if variant == "baseline" else 0,
                "success_rate_final": 0.0 if variant == "baseline" else 1.0,
                "traceback_count": 0,
            },
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_comparison_extracts_rows_and_metrics(tmp_path: Path) -> None:
    baseline = _manifest(tmp_path / "baseline.json", experiment_id="baseline_seed0", variant="baseline")
    curriculum = _manifest(tmp_path / "curriculum.json", experiment_id="curriculum_seed0", variant="curriculum")

    comparison = build_comparison([baseline, curriculum])

    assert comparison["manifest_count"] == 2
    assert comparison["variants"] == ["baseline", "curriculum"]
    assert comparison["seeds"] == [0]
    assert comparison["control_mismatches"] == []
    assert comparison["validation_errors"] == []
    assert comparison["ok_for_causal_comparison"] is True
    assert comparison["rows"][0]["metrics"]["eval.all.mpjpe_g"] == 130.0
    assert comparison["rows"][1]["metrics"]["eval.terminated_final"] == 0


def test_build_comparison_allows_distinct_trained_variant_checkpoints(tmp_path: Path) -> None:
    baseline = _manifest(
        tmp_path / "baseline.json",
        experiment_id="baseline_seed0",
        variant="baseline",
        checkpoint="runs/baseline/last.pt",
    )
    treatment = _manifest(
        tmp_path / "treatment.json",
        experiment_id="treatment_seed0",
        variant="treatment",
        checkpoint="runs/treatment/last.pt",
    )
    for manifest_path in (baseline, treatment):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_source"] = "trained_variant_checkpoint"
        manifest["checkpoint_provenance"] = {
            "is_release_checkpoint": False,
            "sha256": f"sha-{manifest['variant']}",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    comparison = build_comparison([baseline, treatment])

    assert comparison["control_mismatches"] == []
    assert comparison["checkpoint_warnings"] == []
    assert comparison["ok_for_causal_comparison"] is True


def test_build_comparison_flags_release_checkpoint_when_trained_variant_required(tmp_path: Path) -> None:
    baseline = _manifest(tmp_path / "baseline.json", experiment_id="baseline_seed0", variant="baseline")
    treatment = _manifest(tmp_path / "treatment.json", experiment_id="treatment_seed0", variant="treatment")
    for manifest_path in (baseline, treatment):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_source"] = "trained_variant_checkpoint"
        manifest["checkpoint_provenance"] = {"is_release_checkpoint": True}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    comparison = build_comparison([baseline, treatment])

    assert comparison["ok_for_causal_comparison"] is False
    problems = {warning["problem"] for warning in comparison["checkpoint_warnings"]}
    assert "release_checkpoint_used" in problems
    assert "release_checkpoint_path" in problems


def test_build_comparison_flags_control_mismatch(tmp_path: Path) -> None:
    baseline = _manifest(tmp_path / "baseline.json", experiment_id="baseline_seed0", variant="baseline")
    mismatched = _manifest(
        tmp_path / "mismatched.json",
        experiment_id="curriculum_seed0",
        variant="curriculum",
        checkpoint="different.pt",
    )

    comparison = build_comparison([baseline, mismatched])

    assert comparison["ok_for_causal_comparison"] is False
    assert any(item["field"] == "checkpoint" for item in comparison["control_mismatches"])


def test_build_comparison_flags_missing_primary_metrics(tmp_path: Path) -> None:
    baseline = _manifest(tmp_path / "baseline.json", experiment_id="baseline_seed0", variant="baseline")
    missing_eval = _manifest(
        tmp_path / "missing_eval.json",
        experiment_id="curriculum_seed0",
        variant="curriculum",
    )
    data = json.loads(missing_eval.read_text(encoding="utf-8"))
    data["metrics"]["eval"]["ok"] = False
    data["metrics"]["eval"]["all"] = {}
    missing_eval.write_text(json.dumps(data), encoding="utf-8")

    comparison = build_comparison([baseline, missing_eval])

    assert comparison["ok_for_causal_comparison"] is False
    assert {item["field"] for item in comparison["metric_warnings"]} == {"eval.ok", "eval.all.mpjpe_g"}


def test_build_comparison_reports_validation_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": 1, "experiment_id": "broken"}), encoding="utf-8")

    comparison = build_comparison([invalid])

    assert comparison["validation_errors"]
    assert comparison["ok_for_causal_comparison"] is False


def test_write_comparison_markdown_includes_controls_and_table(tmp_path: Path) -> None:
    baseline = _manifest(tmp_path / "baseline.json", experiment_id="baseline_seed0", variant="baseline")
    curriculum = _manifest(tmp_path / "curriculum.json", experiment_id="curriculum_seed0", variant="curriculum")
    comparison = build_comparison([baseline, curriculum])
    output = tmp_path / "comparison.md"

    write_comparison_markdown(output, comparison)

    text = output.read_text(encoding="utf-8")
    assert "# SONIC Manifest Comparison" in text
    assert "ok_for_causal_comparison" in text
    assert "baseline_seed0" in text
    assert "eval.all.mpjpe_g" in text
