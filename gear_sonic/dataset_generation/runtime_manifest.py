"""Machine-readable success evidence for Isaac dataset rollouts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _describe_local_file(path: str | Path | None) -> dict[str, Any] | None:
    """Return a content-addressed description when ``path`` is a local file."""
    if path is None:
        return None
    raw_path = str(path)
    if not raw_path.strip() or "://" in raw_path:
        return {"path": raw_path, "resolved": None, "hash": None, "bytes": None}
    resolved = Path(raw_path).resolve()
    return {
        "path": raw_path,
        "resolved": str(resolved),
        "hash": _sha256_file(resolved) if resolved.is_file() else None,
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def build_runtime_capture_context(
    *,
    terrain_type: str,
    scene_id: str | None,
    scene_usd_path: str | Path | None,
    task: str | None,
    motion_file: str | Path | None,
    camera_provenance: str | None,
    camera_name: str | None,
    camera_track_root: bool | None,
    render_ego: bool,
    camera_attached_link: str | None,
    camera_resolution: list[int] | tuple[int, ...] | None,
    render_frame_skip: int | None,
    use_encoder: str | None,
) -> dict[str, Any]:
    """Build immutable scene, motion, task, and camera evidence for a rollout."""
    scene = _describe_local_file(scene_usd_path)
    if scene_id is None and scene is not None and scene.get("resolved"):
        scene_id = Path(scene["resolved"]).stem
    motion = _describe_local_file(motion_file)
    resolution = None if camera_resolution is None else [int(v) for v in camera_resolution]
    return {
        "terrain_type": str(terrain_type),
        "scene_id": scene_id,
        "scene": scene,
        "task": task.strip() if isinstance(task, str) and task.strip() else None,
        "motion": motion,
        "use_encoder": use_encoder,
        "camera": {
            "provenance": camera_provenance,
            "name": camera_name,
            "track_root": camera_track_root,
            "render_ego": bool(render_ego),
            "attached_link": camera_attached_link,
            "resolution": resolution,
            "render_frame_skip": render_frame_skip,
        },
    }


def _build_artifact_pairs(artifacts: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    """Bind each same-index trajectory/video pair by content hash."""
    trajectory_dir_raw = artifacts.get("trajectory_dir")
    rendering_dir_raw = artifacts.get("rendering_dir")
    if not trajectory_dir_raw or not rendering_dir_raw:
        return {}
    trajectory_dir = Path(trajectory_dir_raw).resolve()
    rendering_dir = Path(rendering_dir_raw).resolve()
    if not trajectory_dir.is_dir() or not rendering_dir.is_dir():
        return {}

    pairs: dict[str, dict[str, Any]] = {}
    for trajectory_path in sorted(trajectory_dir.glob("*.trajectory.pkl")):
        episode_id = trajectory_path.name.removesuffix(".trajectory.pkl")
        video_path = rendering_dir / f"{episode_id}.mp4"
        if not video_path.is_file():
            continue
        pairs[episode_id] = {
            "trajectory_path": str(trajectory_path.resolve()),
            "trajectory_hash": _sha256_file(trajectory_path),
            "trajectory_bytes": trajectory_path.stat().st_size,
            "video_path": str(video_path.resolve()),
            "video_hash": _sha256_file(video_path),
            "video_bytes": video_path.stat().st_size,
        }
    return pairs


def write_runtime_success_manifest(
    path: str | Path,
    *,
    checkpoint: str,
    exit_reason: str,
    num_envs: int,
    policy_iterations: int,
    physics_steps: int,
    artifacts: dict[str, str | None],
    runtime_config_hash: str | None = None,
    capture_context: dict[str, Any] | None = None,
) -> Path:
    """Atomically write the marker that distinguishes a real successful run.

    Isaac Kit/Hydra processes have been observed returning status zero after a
    traceback. Dataset automation therefore consumes this manifest plus the
    declared artifacts instead of trusting process status alone.
    """
    output = Path(path).resolve()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if policy_iterations < 0:
        raise ValueError("policy_iterations must be non-negative")
    if physics_steps < 1:
        raise ValueError("physics_steps must be positive")
    if physics_steps > policy_iterations:
        raise ValueError("physics_steps cannot exceed policy_iterations")

    checkpoint_path = Path(checkpoint).resolve()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "sonic_isaac_eval_success",
        "status": "success",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "checkpoint_resolved": str(checkpoint_path),
        "checkpoint_hash": _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "exit_reason": exit_reason,
        "num_envs": num_envs,
        "policy_iterations": policy_iterations,
        "physics_steps": physics_steps,
        "artifacts": artifacts,
        "artifact_pairs": _build_artifact_pairs(artifacts),
        "runtime_config_hash": runtime_config_hash,
        "capture_context": capture_context,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def invalidate_runtime_success_manifest(path: str | Path) -> None:
    """Remove a prior success marker before a new run can fail or abort."""
    manifest_path = Path(path).resolve()
    try:
        manifest_path.unlink()
    except FileNotFoundError:
        return


def read_runtime_success_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a rollout success marker."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") not in (1, 2):
        raise ValueError("runtime manifest schema_version must be 1 or 2")
    if payload.get("kind") != "sonic_isaac_eval_success":
        raise ValueError("runtime manifest kind is not sonic_isaac_eval_success")
    if payload.get("status") != "success":
        raise ValueError("runtime manifest does not record success")
    if not isinstance(payload.get("physics_steps"), int) or payload["physics_steps"] < 1:
        raise ValueError("runtime manifest must record at least one physics step")
    return payload
