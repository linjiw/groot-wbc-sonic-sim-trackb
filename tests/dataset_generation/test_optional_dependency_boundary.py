from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lightweight_validator_import_does_not_require_joblib() -> None:
    script = """
import sys
sys.modules['joblib'] = None
from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory
assert callable(validate_sonic_trajectory)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
