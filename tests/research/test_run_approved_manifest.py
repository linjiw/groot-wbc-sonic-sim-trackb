from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/research/hallucination/run_approved_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_approved_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_python_uses_manifest_environment(tmp_path: Path) -> None:
    module = _load_module()
    interpreter = tmp_path / "python"
    interpreter.touch()

    resolved = module.resolve_python(
        {"implementation": {"python": str(interpreter)}},
        None,
    )

    assert resolved == interpreter


def test_resolve_python_prefers_explicit_override(tmp_path: Path) -> None:
    module = _load_module()
    pinned = tmp_path / "pinned-python"
    override = tmp_path / "override-python"
    pinned.touch()
    override.touch()

    resolved = module.resolve_python(
        {"implementation": {"python": str(pinned)}},
        override,
    )

    assert resolved == override


def test_resolve_python_fails_closed_without_environment() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="implementation.python is missing"):
        module.resolve_python({"implementation": {}}, None)


def test_capture_mode_arguments_defaults_to_rendering() -> None:
    module = _load_module()

    assert module.capture_mode_arguments({"runtime": {}}) == []


def test_capture_mode_arguments_disables_rendering_when_registered() -> None:
    module = _load_module()

    assert module.capture_mode_arguments({"runtime": {"render_ego": False}}) == [
        "--trajectory-only"
    ]


def test_trajectory_recorder_profile_has_no_camera_term() -> None:
    profile = yaml.safe_load(
        (REPO_ROOT / "gear_sonic/config/manager_env/recorders/trajectory.yaml").read_text()
    )

    assert "trajectory" in profile
    assert "render_envs" not in profile
