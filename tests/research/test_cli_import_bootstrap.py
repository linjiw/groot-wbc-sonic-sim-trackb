from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script",
    [
        "aggregate_sonic_comparisons.py",
        "classify_m5_activation.py",
        "compare_sonic_manifests.py",
        "run_sonic_eval_metric_smoke.py",
        "run_sonic_multiseed.py",
        "run_sonic_paired_experiment.py",
        "verify_schedule_path.py",
    ],
)
def test_direct_cli_prefers_repo_scripts_package(tmp_path: Path, script: str) -> None:
    """ROS installs a top-level ``scripts`` package; direct CLIs must not import it."""
    hostile_package = tmp_path / "scripts"
    hostile_package.mkdir()
    (hostile_package / "__init__.py").write_text(
        "raise RuntimeError('hostile scripts package imported')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    # Put a conflicting package before a repo root that is already on sys.path.
    # The CLI bootstrap must move the existing repo entry to index zero.
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(REPO_ROOT)))
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, f"scripts/research/{script}", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "hostile scripts package imported" not in result.stderr
