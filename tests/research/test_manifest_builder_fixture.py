from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_tiny_fixture_validates_and_builds_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "sonic_vla_tiny"
    manifest = tmp_path / "manifest.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    run_cmd(
        [
            "scripts/research/create_tiny_sonic_vla_fixture.py",
            "--output",
            str(dataset),
            "--episodes",
            "2",
            "--frames",
            "8",
        ]
    )

    run_cmd(["scripts/research/check_sonic_vla_dataset.py", "--dataset-path", str(dataset)])
    run_cmd(
        [
            "scripts/research/build_curriculum_manifest.py",
            "--dataset-path",
            str(dataset),
            "--config",
            "configs/research/curriculum_graph_fetch_place.yaml",
            "--output",
            str(manifest),
            "--default-stage",
            "S3",
        ]
    )
    run_cmd(
        [
            "scripts/research/summarize_curriculum_eval.py",
            "--manifest",
            str(manifest),
            "--config",
            "configs/research/curriculum_graph_fetch_place.yaml",
            "--output-json",
            str(summary_json),
            "--output-md",
            str(summary_md),
        ]
    )

    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 2
    assert {record["stage"] for record in records} == {"S3"}
    assert all(record["schema_valid"] for record in records)
    assert all(record["prompt"] == "pick up the red cup and place it on the tray" for record in records)

    summary = json.loads(summary_json.read_text())
    assert summary["stage_metrics"]["S3"]["num_episodes"] == 2
    assert summary["stage_metrics"]["S3"]["success_rate"] == 1.0
    assert "S3" in summary_md.read_text()
