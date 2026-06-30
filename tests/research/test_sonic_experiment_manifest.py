from __future__ import annotations

import json
from pathlib import Path

from scripts.research.sonic_experiment_manifest import (
    build_manifest,
    validate_manifest,
    write_manifest_markdown,
)


def _summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {"ok": True, "mean_rewards": 0.85156, "total_timesteps": 38400},
                "eval": {"ok": True, "all": {"mpjpe_g": 130.802}, "terminated_final": 1},
            }
        ),
        encoding="utf-8",
    )


def test_build_manifest_records_controlled_variables_and_artifacts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _summary(summary_path)

    manifest = build_manifest(
        experiment_id="sample_release_eval_seed0",
        hypothesis="released checkpoint eval harness runs on sample data",
        variant="released_checkpoint_sample_eval",
        seed=0,
        dataset_robot="sample_data/robot_filtered",
        dataset_smpl="sample_data/smpl_filtered",
        checkpoint="sonic_release/last.pt",
        summary_json=summary_path,
        train_command="python gear_sonic/train_agent_trl.py ...",
        eval_command="python gear_sonic/eval_agent_trl.py ...",
        interpretation="harness_ok_not_convergence",
        git_commit="abc1234",
    )

    assert manifest["schema_version"] == 1
    assert manifest["experiment_id"] == "sample_release_eval_seed0"
    assert manifest["controlled_variables"]["simulator"] == "IsaacLab / Isaac Sim headless"
    assert manifest["controlled_variables"]["action_interface"] == "64D SONIC motion token + 7D left hand + 7D right hand"
    assert manifest["datasets"]["robot_motion"] == "sample_data/robot_filtered"
    assert manifest["artifacts"]["summary_json"] == str(summary_path)
    assert manifest["metrics"]["eval"]["all"]["mpjpe_g"] == 130.802
    assert manifest["status"] == "needs_review"


def test_validate_manifest_rejects_missing_required_fields(tmp_path: Path) -> None:
    manifest = {"schema_version": 1, "experiment_id": "x"}

    errors = validate_manifest(manifest)

    assert "missing hypothesis" in errors
    assert "missing datasets.robot_motion" in errors
    assert "missing artifacts.summary_json" in errors


def test_validate_manifest_accepts_builder_output(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _summary(summary_path)
    manifest = build_manifest(
        experiment_id="sample_release_eval_seed0",
        hypothesis="h",
        variant="v",
        seed=0,
        dataset_robot="sample_data/robot_filtered",
        dataset_smpl="sample_data/smpl_filtered",
        checkpoint="sonic_release/last.pt",
        summary_json=summary_path,
        train_command="train",
        eval_command="eval",
        interpretation="pending",
        git_commit="abc1234",
    )

    assert validate_manifest(manifest) == []


def test_write_manifest_markdown_includes_reproducibility_fields(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _summary(summary_path)
    manifest = build_manifest(
        experiment_id="sample_release_eval_seed0",
        hypothesis="h",
        variant="v",
        seed=0,
        dataset_robot="sample_data/robot_filtered",
        dataset_smpl="sample_data/smpl_filtered",
        checkpoint="sonic_release/last.pt",
        summary_json=summary_path,
        train_command="train",
        eval_command="eval",
        interpretation="pending",
        git_commit="abc1234",
    )
    output = tmp_path / "manifest.md"

    write_manifest_markdown(output, manifest)

    text = output.read_text(encoding="utf-8")
    assert "# SONIC Experiment Manifest" in text
    assert "sample_release_eval_seed0" in text
    assert "IsaacLab / Isaac Sim headless" in text
    assert "mpjpe_g" in text
