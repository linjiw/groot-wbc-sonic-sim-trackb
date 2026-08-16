"""Synthetic G1 dataset generation contracts and adapters.

The package-level adapter exports are loaded lazily.  This keeps lightweight
schema, scene, and rollout validators usable in Isaac environments that do not
install the offline ``joblib`` conversion dependency.
"""

from typing import Any

from gear_sonic.dataset_generation.schemas import (
    ArtifactRef,
    ConversionResult,
    EpisodeRequest,
    GenerationResult,
    content_sha256,
    file_sha256,
)

_ADAPTER_EXPORTS = {
    "KIMODO_G1_QPOS_DIM",
    "load_kimodo_qpos_csv",
    "qpos_to_sonic_motion_entry",
    "transform_qpos_to_scene",
}


def __getattr__(name: str) -> Any:
    """Resolve optional Kimodo adapter exports only when requested."""
    if name not in _ADAPTER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from gear_sonic.dataset_generation import kimodo_motion_adapter

    value = getattr(kimodo_motion_adapter, name)
    globals()[name] = value
    return value


__all__ = [
    "ArtifactRef",
    "ConversionResult",
    "EpisodeRequest",
    "GenerationResult",
    "KIMODO_G1_QPOS_DIM",
    "content_sha256",
    "file_sha256",
    "load_kimodo_qpos_csv",
    "qpos_to_sonic_motion_entry",
    "transform_qpos_to_scene",
]
