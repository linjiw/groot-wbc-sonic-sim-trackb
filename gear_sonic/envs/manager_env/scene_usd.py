"""Validation helpers for loading one full-scene USD in Isaac Lab."""

from __future__ import annotations

import os


def resolve_scene_usd_path(scene_usd_path: object, *, num_envs: int) -> str:
    """Validate MVP full-scene constraints and return a resolved USD path or URI.

    Isaac Lab's ``TerrainImporter`` places a USD at one global prim. Until the
    dataset pipeline has an environment-namespaced scene cloner, this mode is
    intentionally restricted to one environment.
    """
    if num_envs != 1:
        raise ValueError(
            "terrain_type=scene_usd currently requires num_envs=1 because Isaac Lab's "
            "USD TerrainImporter imports one global scene; got "
            f"num_envs={num_envs}"
        )
    if not isinstance(scene_usd_path, str) or not scene_usd_path.strip():
        raise ValueError("terrain_type=scene_usd requires a non-empty scene_usd_path")

    scene_usd_path = scene_usd_path.strip()
    if "://" in scene_usd_path:
        return scene_usd_path

    resolved_path = os.path.abspath(scene_usd_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"scene_usd_path does not exist: {resolved_path}")
    return resolved_path
