#!/usr/bin/env python3
"""Fail closed when a frozen SONIC runtime source file has drifted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read runtime source lock {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "sonic_runtime_source_lock":
        raise ValueError("runtime source lock has the wrong kind")
    return value


def verify_runtime_source_lock(
    lock_path: Path,
    *,
    repo_root: Path,
    observed_git_head: str | None = None,
) -> dict[str, Any]:
    """Verify the locked Git revision and every byte-locked runtime file."""
    root = repo_root.resolve()
    lock = _load_lock(lock_path.resolve())
    expected_head = lock.get("git_head")
    if not isinstance(expected_head, str) or len(expected_head) != 40:
        raise ValueError("runtime source lock must contain a full git_head")
    if observed_git_head is None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise ValueError(f"cannot resolve git HEAD: {completed.stderr.strip()}")
        observed_git_head = completed.stdout.strip()
    if observed_git_head != expected_head:
        raise ValueError(
            f"git HEAD drifted: expected {expected_head}, observed {observed_git_head}"
        )

    runtime_files = lock.get("runtime_files")
    if not isinstance(runtime_files, dict) or not runtime_files:
        raise ValueError("runtime source lock must contain runtime_files")
    verified: list[dict[str, Any]] = []
    for path_text, expected_sha256 in sorted(runtime_files.items()):
        if not isinstance(path_text, str) or not isinstance(expected_sha256, str):
            raise ValueError("runtime_files must map path strings to SHA-256 strings")
        relative = Path(path_text)
        if relative.is_absolute():
            raise ValueError(f"runtime file path must be relative: {path_text}")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"runtime file escapes repo root: {path_text}")
        if not target.is_file():
            raise ValueError(f"locked runtime file is missing: {path_text}")
        observed_sha256 = _sha256(target)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"runtime file drifted: {path_text}; expected {expected_sha256}, "
                f"observed {observed_sha256}"
            )
        verified.append(
            {
                "path": path_text,
                "sha256": observed_sha256,
                "size_bytes": target.stat().st_size,
            }
        )
    return {
        "kind": "sonic_runtime_source_lock_verification",
        "git_head": observed_git_head,
        "verified_file_count": len(verified),
        "files": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        result = verify_runtime_source_lock(args.lock, repo_root=args.repo_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
