"""Versioned, content-addressed contracts for synthetic G1 dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

EPISODE_REQUEST_SCHEMA_VERSION = 1
GENERATION_RESULT_SCHEMA_VERSION = 1
CONVERSION_RESULT_SCHEMA_VERSION = 1
CONVERSION_RESULT_KIND = "kimodo_sonic_conversion_result"
KIMODO_QPOS_ARTIFACT_NAME = "kimodo_qpos_csv"
SONIC_MOTION_ARTIFACT_NAME = "sonic_motion_lib"
KIMODO_ROOT_QUATERNION_ORDER = "wxyz"
SONIC_ROOT_QUATERNION_ORDER = "xyzw"
G1_MUJOCO_JOINT_ORDER = "g1_mujoco_29"
SCENE_TRANSFORM_ORDER = (
    "canonicalize_horizontal_origin",
    "rotate_z_by_scene_yaw",
    "translate_by_scene_start_xyz",
)
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data deterministically for content hashing."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    """Return a prefixed SHA-256 digest of canonical JSON data."""
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a prefixed SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must have form 'sha256:<64 lowercase hex characters>'")


def _require_nonnegative_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _require_positive_finite(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field} must be a positive finite number")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json_object(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to one content-addressed artifact."""

    name: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "name")
        _require_nonempty(self.path, "path")
        _require_sha256(self.sha256, "sha256")
        _require_nonnegative_int(self.size_bytes, "size_bytes")
        _require_nonempty(self.media_type, "media_type")

    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        artifact_path = Path(path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        return cls(
            name=name,
            path=str(artifact_path),
            sha256=file_sha256(artifact_path),
            size_bytes=artifact_path.stat().st_size,
            media_type=media_type,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        return cls(
            name=value["name"],
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
            media_type=value.get("media_type", "application/octet-stream"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_dict(self) -> dict[str, Any]:
        """Return content identity fields, excluding the relocatable path."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    def resolved_path(self, *, base_dir: str | Path | None = None) -> Path:
        """Resolve this artifact locator against an optional manifest directory."""
        artifact_path = Path(self.path)
        if not artifact_path.is_absolute() and base_dir is not None:
            artifact_path = Path(base_dir) / artifact_path
        return artifact_path.resolve()

    def verify(self, *, base_dir: str | Path | None = None) -> Path:
        """Verify that the referenced file exists and matches its size and digest."""
        artifact_path = self.resolved_path(base_dir=base_dir)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"artifact {self.name!r} is missing or is not a file: {artifact_path}"
            )

        actual_size = artifact_path.stat().st_size
        if actual_size != self.size_bytes:
            raise ValueError(
                f"artifact {self.name!r} size mismatch for {artifact_path}: "
                f"declared {self.size_bytes}, actual {actual_size}"
            )

        actual_sha256 = file_sha256(artifact_path)
        if actual_sha256 != self.sha256:
            raise ValueError(
                f"artifact {self.name!r} SHA-256 mismatch for {artifact_path}: "
                f"declared {self.sha256}, actual {actual_sha256}"
            )
        return artifact_path


def _serialized_artifacts(
    artifacts: tuple[ArtifactRef, ...],
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Verify artifacts and serialize paths relative to a manifest bundle."""
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_path = artifact.verify()
        record = artifact.to_dict()
        record["path"] = os.path.relpath(artifact_path, start=manifest_path.parent)
        records.append(record)
    return records


def _verified_artifacts(
    artifacts: tuple[ArtifactRef, ...],
    *,
    base_dir: str | Path | None,
) -> tuple[ArtifactRef, ...]:
    """Verify artifact bytes and normalize their locators to absolute paths."""
    return tuple(
        replace(artifact, path=str(artifact.verify(base_dir=base_dir))) for artifact in artifacts
    )


@dataclass(frozen=True)
class EpisodeRequest:
    """Immutable input to one Kimodo generation and SONIC rollout candidate."""

    scene_id: str
    task_family: str
    task_prompt: str
    style_prompt: str
    route_xy: tuple[tuple[float, float], ...]
    nominal_speed_mps: float
    duration_s: float
    kimodo_model: str
    kimodo_seed: int
    simulation_seed: int
    render_seed: int
    candidate_index: int
    scene_hash: str
    controller_hash: str
    physics_hash: str
    route_frame: str = "scene_local"
    schema_version: int = EPISODE_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EPISODE_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EPISODE_REQUEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        for field in ["scene_id", "task_family", "task_prompt", "kimodo_model"]:
            _require_nonempty(getattr(self, field), field)
        if not isinstance(self.style_prompt, str):
            raise ValueError("style_prompt must be a string")
        if self.route_frame != "scene_local":
            raise ValueError("route_frame must be 'scene_local' in schema version 1")

        route: list[tuple[float, float]] = []
        for index, point in enumerate(self.route_xy):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"route_xy[{index}] must contain exactly two coordinates")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"route_xy[{index}] must contain finite coordinates")
            route.append((x, y))
        if len(route) < 2:
            raise ValueError("route_xy must contain at least two points")
        object.__setattr__(self, "route_xy", tuple(route))

        _require_positive_finite(self.nominal_speed_mps, "nominal_speed_mps")
        _require_positive_finite(self.duration_s, "duration_s")
        for field in ["kimodo_seed", "simulation_seed", "render_seed", "candidate_index"]:
            _require_nonnegative_int(getattr(self, field), field)
        _require_sha256(self.scene_hash, "scene_hash")
        _require_sha256(self.controller_hash, "controller_hash")
        _require_sha256(self.physics_hash, "physics_hash")

    def content_dict(self) -> dict[str, Any]:
        """Return the content-addressed payload, excluding its derived request ID."""
        value = asdict(self)
        value["route_xy"] = [list(point) for point in self.route_xy]
        return value

    @property
    def request_id(self) -> str:
        return content_sha256(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.content_dict()
        value["episode_request_id"] = self.request_id
        return value

    def write_json(self, path: str | Path) -> None:
        _write_json_object(path, self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeRequest:
        payload = dict(value)
        declared_id = payload.pop("episode_request_id", None)
        request = cls(
            scene_id=payload["scene_id"],
            task_family=payload["task_family"],
            task_prompt=payload["task_prompt"],
            style_prompt=payload.get("style_prompt", ""),
            route_xy=tuple(tuple(point) for point in payload["route_xy"]),
            nominal_speed_mps=payload["nominal_speed_mps"],
            duration_s=payload["duration_s"],
            kimodo_model=payload["kimodo_model"],
            kimodo_seed=payload["kimodo_seed"],
            simulation_seed=payload["simulation_seed"],
            render_seed=payload["render_seed"],
            candidate_index=payload["candidate_index"],
            scene_hash=payload["scene_hash"],
            controller_hash=payload["controller_hash"],
            physics_hash=payload["physics_hash"],
            route_frame=payload.get("route_frame", "scene_local"),
            schema_version=payload.get("schema_version", EPISODE_REQUEST_SCHEMA_VERSION),
        )
        if declared_id is not None and declared_id != request.request_id:
            raise ValueError(
                "episode_request_id does not match the canonical request payload: "
                f"declared {declared_id}, computed {request.request_id}"
            )
        return request

    @classmethod
    def read_json(cls, path: str | Path) -> EpisodeRequest:
        return cls.from_dict(_load_json_object(path))


@dataclass(frozen=True)
class GenerationResult:
    """Content-addressed result of a Kimodo generation request."""

    episode_request_id: str
    generator_name: str
    generator_version: str
    resolved_model: str
    source_fps: float
    seed: int
    artifacts: tuple[ArtifactRef, ...]
    schema_version: int = GENERATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GENERATION_RESULT_SCHEMA_VERSION}, got {self.schema_version}"
            )
        _require_sha256(self.episode_request_id, "episode_request_id")
        for field in ["generator_name", "generator_version", "resolved_model"]:
            _require_nonempty(getattr(self, field), field)
        _require_positive_finite(self.source_fps, "source_fps")
        _require_nonnegative_int(self.seed, "seed")
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("artifacts must contain at least one artifact")
        names = [artifact.name for artifact in artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        object.__setattr__(self, "artifacts", artifacts)

    def content_dict(self) -> dict[str, Any]:
        """Return semantic result content, excluding relocatable artifact paths."""
        return {
            "schema_version": self.schema_version,
            "episode_request_id": self.episode_request_id,
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "resolved_model": self.resolved_model,
            "source_fps": float(self.source_fps),
            "seed": self.seed,
            "artifacts": [artifact.content_dict() for artifact in self.artifacts],
        }

    @property
    def result_id(self) -> str:
        return content_sha256(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.content_dict()
        value["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        value["generation_result_id"] = self.result_id
        return value

    def write_json(self, path: str | Path) -> None:
        output = Path(path).resolve()
        value = self.to_dict()
        value["artifacts"] = _serialized_artifacts(self.artifacts, manifest_path=output)
        _write_json_object(output, value)

    def verify_artifacts(self, *, base_dir: str | Path | None = None) -> GenerationResult:
        """Verify all artifact bytes and return refs with absolute resolved locators."""
        return replace(
            self,
            artifacts=_verified_artifacts(self.artifacts, base_dir=base_dir),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationResult:
        payload = dict(value)
        declared_id = payload.pop("generation_result_id", None)
        result = cls(
            episode_request_id=payload["episode_request_id"],
            generator_name=payload["generator_name"],
            generator_version=payload["generator_version"],
            resolved_model=payload["resolved_model"],
            source_fps=payload["source_fps"],
            seed=payload["seed"],
            artifacts=tuple(ArtifactRef.from_dict(item) for item in payload["artifacts"]),
            schema_version=payload.get("schema_version", GENERATION_RESULT_SCHEMA_VERSION),
        )
        if declared_id is not None and declared_id != result.result_id:
            raise ValueError(
                "generation_result_id does not match the canonical result payload: "
                f"declared {declared_id}, computed {result.result_id}"
            )
        return result

    @classmethod
    def read_json(cls, path: str | Path) -> GenerationResult:
        manifest_path = Path(path).resolve()
        result = cls.from_dict(_load_json_object(manifest_path))
        return result.verify_artifacts(base_dir=manifest_path.parent)


@dataclass(frozen=True)
class ConversionResult:
    """Content-addressed Kimodo-qpos to SONIC-motion conversion result."""

    episode_request_id: str
    source_generation_result_id: str
    source_artifact_name: str
    converter_name: str
    converter_version: str
    source_fps: float
    motion_key: str
    frame_count: int
    scene_start_xyz: tuple[float, float, float]
    scene_yaw: float
    canonicalize_horizontal_origin: bool
    artifacts: tuple[ArtifactRef, ...]
    scene_frame: str = "scene_local"
    source_root_quaternion_order: str = KIMODO_ROOT_QUATERNION_ORDER
    source_joint_order: str = G1_MUJOCO_JOINT_ORDER
    output_root_quaternion_order: str = SONIC_ROOT_QUATERNION_ORDER
    output_joint_order: str = G1_MUJOCO_JOINT_ORDER
    scene_transform_order: tuple[str, ...] = SCENE_TRANSFORM_ORDER
    kind: str = CONVERSION_RESULT_KIND
    schema_version: int = CONVERSION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONVERSION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CONVERSION_RESULT_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if self.kind != CONVERSION_RESULT_KIND:
            raise ValueError(f"kind must be {CONVERSION_RESULT_KIND!r}")
        _require_sha256(self.episode_request_id, "episode_request_id")
        _require_sha256(self.source_generation_result_id, "source_generation_result_id")
        for field in (
            "source_artifact_name",
            "converter_name",
            "converter_version",
            "motion_key",
        ):
            _require_nonempty(getattr(self, field), field)
        _require_positive_finite(self.source_fps, "source_fps")
        if not float(self.source_fps).is_integer():
            raise ValueError("source_fps must be integral for the SONIC motion-library schema")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise ValueError("frame_count must be a positive integer")
        if self.frame_count < 2:
            raise ValueError("frame_count must be an integer of at least 2")

        try:
            scene_start_xyz = tuple(float(value) for value in self.scene_start_xyz)
        except (TypeError, ValueError) as exc:
            raise ValueError("scene_start_xyz must contain three finite values") from exc
        if len(scene_start_xyz) != 3 or not all(math.isfinite(value) for value in scene_start_xyz):
            raise ValueError("scene_start_xyz must contain three finite values")
        object.__setattr__(self, "scene_start_xyz", scene_start_xyz)

        if isinstance(self.scene_yaw, bool) or not isinstance(self.scene_yaw, int | float):
            raise ValueError("scene_yaw must be finite")
        if not math.isfinite(float(self.scene_yaw)):
            raise ValueError("scene_yaw must be finite")
        object.__setattr__(self, "scene_yaw", float(self.scene_yaw))
        if not isinstance(self.canonicalize_horizontal_origin, bool):
            raise ValueError("canonicalize_horizontal_origin must be a boolean")
        if self.scene_frame != "scene_local":
            raise ValueError("scene_frame must be 'scene_local'")

        expected_orders = {
            "source_root_quaternion_order": KIMODO_ROOT_QUATERNION_ORDER,
            "source_joint_order": G1_MUJOCO_JOINT_ORDER,
            "output_root_quaternion_order": SONIC_ROOT_QUATERNION_ORDER,
            "output_joint_order": G1_MUJOCO_JOINT_ORDER,
        }
        for field, expected in expected_orders.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must be {expected!r}")
        scene_transform_order = tuple(self.scene_transform_order)
        if scene_transform_order != SCENE_TRANSFORM_ORDER:
            raise ValueError(f"scene_transform_order must be {SCENE_TRANSFORM_ORDER!r}")
        object.__setattr__(self, "scene_transform_order", scene_transform_order)

        artifacts = tuple(sorted(self.artifacts, key=lambda artifact: artifact.name))
        names = [artifact.name for artifact in artifacts]
        expected_names = {KIMODO_QPOS_ARTIFACT_NAME, SONIC_MOTION_ARTIFACT_NAME}
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError(
                "artifacts must contain exactly one kimodo_qpos_csv and one sonic_motion_lib"
            )
        object.__setattr__(self, "artifacts", artifacts)

    def content_dict(self) -> dict[str, Any]:
        """Return semantic conversion content, excluding relocatable artifact paths."""
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "episode_request_id": self.episode_request_id,
            "source_generation_result_id": self.source_generation_result_id,
            "source_artifact_name": self.source_artifact_name,
            "converter_name": self.converter_name,
            "converter_version": self.converter_version,
            "source_fps": float(self.source_fps),
            "motion_key": self.motion_key,
            "frame_count": self.frame_count,
            "scene_start_xyz": list(self.scene_start_xyz),
            "scene_yaw": float(self.scene_yaw),
            "canonicalize_horizontal_origin": self.canonicalize_horizontal_origin,
            "scene_frame": self.scene_frame,
            "source_root_quaternion_order": self.source_root_quaternion_order,
            "source_joint_order": self.source_joint_order,
            "output_root_quaternion_order": self.output_root_quaternion_order,
            "output_joint_order": self.output_joint_order,
            "scene_transform_order": list(self.scene_transform_order),
            "artifacts": [artifact.content_dict() for artifact in self.artifacts],
        }

    @property
    def result_id(self) -> str:
        return content_sha256(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.content_dict()
        value["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        value["conversion_result_id"] = self.result_id
        return value

    def verify_parent(self, parent: GenerationResult) -> None:
        """Verify the parent identity, request, FPS, and selected source artifact."""
        if parent.result_id != self.source_generation_result_id:
            raise ValueError("source_generation_result_id does not match the parent result")
        if parent.episode_request_id != self.episode_request_id:
            raise ValueError("episode_request_id does not match the parent result")
        if not math.isclose(float(parent.source_fps), float(self.source_fps), abs_tol=1e-9):
            raise ValueError("source_fps does not match the parent result")

        parent_by_name = {artifact.name: artifact for artifact in parent.artifacts}
        if self.source_artifact_name not in parent_by_name:
            raise ValueError(
                f"parent result has no source artifact named {self.source_artifact_name!r}"
            )
        conversion_by_name = {artifact.name: artifact for artifact in self.artifacts}
        parent_source = parent_by_name[self.source_artifact_name]
        conversion_source = conversion_by_name[KIMODO_QPOS_ARTIFACT_NAME]
        if (
            parent_source.sha256 != conversion_source.sha256
            or parent_source.size_bytes != conversion_source.size_bytes
            or parent_source.media_type != conversion_source.media_type
        ):
            raise ValueError("Kimodo qpos artifact does not match the selected parent artifact")

    def write_json(self, path: str | Path, *, include_legacy_aliases: bool = False) -> None:
        output = Path(path).resolve()
        value = self.to_dict()
        value["artifacts"] = _serialized_artifacts(self.artifacts, manifest_path=output)
        if include_legacy_aliases:
            artifacts_by_name = {artifact["name"]: artifact for artifact in value["artifacts"]}
            value["converter"] = self.converter_name
            value["input"] = dict(artifacts_by_name[KIMODO_QPOS_ARTIFACT_NAME])
            value["output"] = dict(artifacts_by_name[SONIC_MOTION_ARTIFACT_NAME])
        _write_json_object(output, value)

    def verify_artifacts(self, *, base_dir: str | Path | None = None) -> ConversionResult:
        """Verify all artifact bytes and return refs with absolute resolved locators."""
        return replace(
            self,
            artifacts=_verified_artifacts(self.artifacts, base_dir=base_dir),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConversionResult:
        payload = dict(value)
        allowed_fields = {
            "schema_version",
            "kind",
            "conversion_result_id",
            "episode_request_id",
            "source_generation_result_id",
            "source_artifact_name",
            "converter_name",
            "converter_version",
            "source_fps",
            "motion_key",
            "frame_count",
            "scene_start_xyz",
            "scene_yaw",
            "canonicalize_horizontal_origin",
            "scene_frame",
            "source_root_quaternion_order",
            "source_joint_order",
            "output_root_quaternion_order",
            "output_joint_order",
            "scene_transform_order",
            "artifacts",
            # Compatibility aliases emitted for legacy converter-manifest readers.
            "converter",
            "input",
            "output",
        }
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise ValueError(f"unknown ConversionResult fields: {unknown_fields}")
        declared_id = payload.get("conversion_result_id")
        if declared_id is None:
            raise ValueError("conversion_result_id is required")
        result = cls(
            episode_request_id=payload["episode_request_id"],
            source_generation_result_id=payload["source_generation_result_id"],
            source_artifact_name=payload["source_artifact_name"],
            converter_name=payload["converter_name"],
            converter_version=payload["converter_version"],
            source_fps=payload["source_fps"],
            motion_key=payload["motion_key"],
            frame_count=payload["frame_count"],
            scene_start_xyz=tuple(payload["scene_start_xyz"]),
            scene_yaw=payload["scene_yaw"],
            canonicalize_horizontal_origin=payload["canonicalize_horizontal_origin"],
            artifacts=tuple(ArtifactRef.from_dict(item) for item in payload["artifacts"]),
            scene_frame=payload["scene_frame"],
            source_root_quaternion_order=payload["source_root_quaternion_order"],
            source_joint_order=payload["source_joint_order"],
            output_root_quaternion_order=payload["output_root_quaternion_order"],
            output_joint_order=payload["output_joint_order"],
            scene_transform_order=tuple(payload["scene_transform_order"]),
            kind=payload.get("kind", ""),
            schema_version=payload.get("schema_version", CONVERSION_RESULT_SCHEMA_VERSION),
        )
        if declared_id != result.result_id:
            raise ValueError(
                "conversion_result_id does not match the canonical result payload: "
                f"declared {declared_id}, computed {result.result_id}"
            )
        if "converter" in payload and payload["converter"] != result.converter_name:
            raise ValueError("legacy converter alias does not match converter_name")
        artifacts_by_name = {artifact.name: artifact for artifact in result.artifacts}
        for alias, artifact_name in (
            ("input", KIMODO_QPOS_ARTIFACT_NAME),
            ("output", SONIC_MOTION_ARTIFACT_NAME),
        ):
            if (
                alias in payload
                and ArtifactRef.from_dict(payload[alias]) != artifacts_by_name[artifact_name]
            ):
                raise ValueError(f"legacy {alias} alias does not match artifacts")
        return result

    @classmethod
    def read_json(cls, path: str | Path) -> ConversionResult:
        manifest_path = Path(path).resolve()
        result = cls.from_dict(_load_json_object(manifest_path))
        return result.verify_artifacts(base_dir=manifest_path.parent)
