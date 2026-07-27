from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.verify_sonic_runtime_source_lock import (
    verify_runtime_source_lock,
)

HEAD = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_lock(tmp_path: Path, runtime_files: dict[str, str]) -> Path:
    path = tmp_path / "runtime_lock.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sonic_runtime_source_lock",
                "git_head": HEAD,
                "runtime_files": runtime_files,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_verifies_git_head_and_exact_runtime_bytes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.py"
    runtime.write_text("value = 1\n", encoding="utf-8")
    lock = _write_lock(tmp_path, {"runtime.py": _sha256(runtime)})

    result = verify_runtime_source_lock(
        lock,
        repo_root=tmp_path,
        observed_git_head=HEAD,
    )

    assert result["verified_file_count"] == 1
    assert result["files"][0]["sha256"] == _sha256(runtime)


def test_rejects_runtime_file_drift(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.py"
    runtime.write_text("before\n", encoding="utf-8")
    lock = _write_lock(tmp_path, {"runtime.py": _sha256(runtime)})
    runtime.write_text("after\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime file drifted"):
        verify_runtime_source_lock(lock, repo_root=tmp_path, observed_git_head=HEAD)


def test_rejects_git_head_drift(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.py"
    runtime.write_text("value = 1\n", encoding="utf-8")
    lock = _write_lock(tmp_path, {"runtime.py": _sha256(runtime)})

    with pytest.raises(ValueError, match="git HEAD drifted"):
        verify_runtime_source_lock(
            lock,
            repo_root=tmp_path,
            observed_git_head="b" * 40,
        )


def test_rejects_runtime_path_escape(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, {"../outside.py": "0" * 64})

    with pytest.raises(ValueError, match="escapes repo root"):
        verify_runtime_source_lock(lock, repo_root=tmp_path, observed_git_head=HEAD)
