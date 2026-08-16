from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.envs.manager_env.scene_usd import resolve_scene_usd_path


def test_resolves_local_scene_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_path = tmp_path / "room.usd"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_scene_usd_path("room.usd", num_envs=1) == str(scene_path)


def test_preserves_scene_uri() -> None:
    uri = "omniverse://localhost/NVIDIA/Assets/factory.usd"

    assert resolve_scene_usd_path(uri, num_envs=1) == uri


def test_rejects_vectorized_global_scene_import() -> None:
    with pytest.raises(ValueError, match="requires num_envs=1"):
        resolve_scene_usd_path("factory.usd", num_envs=2)


@pytest.mark.parametrize("path", [None, "", "   "])
def test_rejects_missing_scene_path(path: object) -> None:
    with pytest.raises(ValueError, match="non-empty scene_usd_path"):
        resolve_scene_usd_path(path, num_envs=1)


def test_rejects_nonexistent_local_scene(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_scene_usd_path(str(tmp_path / "missing.usd"), num_envs=1)
