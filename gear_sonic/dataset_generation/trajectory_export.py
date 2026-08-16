"""Convert validated SONIC physics trajectories into synthetic G1 frames."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    G1_ISAACLAB_TO_MUJOCO_DOF,
    KIMODO_G1_JOINT_NAMES,
)
from gear_sonic.dataset_generation.latent_parity import (
    check_latent_parity,
    latent_parity_provenance,
)
from gear_sonic.dataset_generation.trajectory_acceptance import (
    LocomotionAcceptanceReport,
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory

EGO_FRAME_SHAPE = (480, 640, 3)
HAND_ACTION_DIM = 7
G1_DOF_DIM = len(KIMODO_G1_JOINT_NAMES)
VALID_G1_DOF_ORDERS = ("isaaclab", "mujoco")
SONIC_DATASET_FPS = 50.0
POST_STEP_RECORDING_PHASE = "post_physics_step"
PRE_STEP_POLICY_ACTION_PHASE = "pre_physics_step_policy_output"
ALIGNED_RECORDING_PHASE = "policy_observation"
ALIGNED_POLICY_ACTION_PHASE = "same_observation_policy_output"
_ACTION_FRAME_FIELDS = (
    "action_motion_token",
    "policy_meta_action",
    "applied_joint_action",
)


@dataclass(frozen=True)
class VideoInfo:
    """Decoded video properties used for frame-alignment validation."""

    fps: float
    frame_count: int
    width: int
    height: int


def sha256_file(path: str | Path) -> str:
    """Return a content hash suitable for export provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def normalize_frame_range(
    total_frames: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[int, int]:
    """Validate and normalize a half-open source-frame selection."""
    if not isinstance(total_frames, int) or isinstance(total_frames, bool) or total_frames < 1:
        raise ValueError("total_frames must be a positive integer")
    if not isinstance(start_frame, int) or isinstance(start_frame, bool):
        raise ValueError("start_frame must be an integer")
    if end_frame is None:
        end_frame = total_frames
    if not isinstance(end_frame, int) or isinstance(end_frame, bool):
        raise ValueError("end_frame must be an integer")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame > total_frames:
        raise ValueError(f"end_frame={end_frame} exceeds source trajectory length {total_frames}")
    if start_frame >= end_frame:
        raise ValueError(
            f"frame range must be non-empty and increasing; got [{start_frame}, {end_frame})"
        )
    return start_frame, end_frame


def _slice_frame_aligned_value(
    value: Any,
    *,
    source_total_frames: int,
    start_frame: int,
    end_frame: int,
) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == source_total_frames:
            return value[start_frame:end_frame].copy()
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_frame_aligned_value(
                nested,
                source_total_frames=source_total_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        if len(value) == source_total_frames:
            return [
                _slice_frame_aligned_value(
                    nested,
                    source_total_frames=source_total_frames,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                for nested in value[start_frame:end_frame]
            ]
        return [
            _slice_frame_aligned_value(
                nested,
                source_total_frames=source_total_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for nested in value
        ]
    if isinstance(value, tuple):
        if len(value) == source_total_frames:
            return tuple(
                _slice_frame_aligned_value(
                    nested,
                    source_total_frames=source_total_frames,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                for nested in value[start_frame:end_frame]
            )
        return tuple(
            _slice_frame_aligned_value(
                nested,
                source_total_frames=source_total_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for nested in value
        )
    return value


def slice_trajectory_frames(
    trajectory: Mapping[str, Any],
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, Any]:
    """Return a validated half-open clip with all frame-aligned arrays sliced."""
    source_total_frames = trajectory.get("total_frames")
    if not isinstance(source_total_frames, int) or isinstance(source_total_frames, bool):
        raise ValueError("trajectory total_frames must be an integer")
    start_frame, end_frame = normalize_frame_range(
        source_total_frames,
        start_frame,
        end_frame,
    )
    selected = _slice_frame_aligned_value(
        trajectory,
        source_total_frames=source_total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if not isinstance(selected, dict):
        selected = dict(selected)
    selected["total_frames"] = end_frame - start_frame

    # Legacy recorder metrics with fewer than ``total_frames`` entries have no
    # frame-index evidence. Retaining them after a subclip would silently attach
    # measurements to the wrong frames, so they are deliberately excluded.
    partial_metrics = selected.pop("partial_tracking_metrics", None)
    selected.pop("partial_tracking_metric_frame_indices", None)
    if isinstance(partial_metrics, Mapping) and partial_metrics:
        selected["dropped_unindexed_partial_tracking_metrics"] = tuple(
            sorted(map(str, partial_metrics))
        )

    report = validate_sonic_trajectory(selected)
    if not report.ok:
        details = "\n".join(f"- {error}" for error in report.errors)
        raise ValueError(f"selected trajectory validation failed:\n{details}")
    return selected


def align_post_step_trajectory_for_bc(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pair each post-step observation with the policy action chosen from it.

    Isaac Lab recorder terms run after physics. Raw frame ``i`` therefore
    contains observation ``S[i+1]`` and the policy output ``A[i]`` that caused
    it. A behavior-cloning row must instead contain ``S[i+1], A[i+1]``. The
    last raw observation has no in-window next action, so an ``N``-frame raw
    interval produces ``N-1`` causal rows.
    """
    source_total_frames = trajectory.get("total_frames")
    if not isinstance(source_total_frames, int) or isinstance(source_total_frames, bool):
        raise ValueError("trajectory total_frames must be an integer")
    if source_total_frames < 2:
        raise ValueError("causal alignment requires at least two raw frames")

    recording_phase_declared = "recording_phase" in trajectory
    action_phase_declared = "policy_action_phase" in trajectory
    recording_phase = trajectory.get("recording_phase", POST_STEP_RECORDING_PHASE)
    action_phase = trajectory.get("policy_action_phase", PRE_STEP_POLICY_ACTION_PHASE)
    if recording_phase != POST_STEP_RECORDING_PHASE:
        raise ValueError(
            "recording_phase must be post_physics_step for causal export; "
            f"got {recording_phase!r}"
        )
    if action_phase != PRE_STEP_POLICY_ACTION_PHASE:
        raise ValueError(
            "policy_action_phase must be pre_physics_step_policy_output for causal export; "
            f"got {action_phase!r}"
        )

    aligned = slice_trajectory_frames(
        trajectory,
        start_frame=0,
        end_frame=source_total_frames - 1,
    )
    shifted_fields: list[str] = []
    for key in _ACTION_FRAME_FIELDS:
        if key not in trajectory:
            continue
        values = np.asarray(trajectory[key])
        if values.ndim < 1 or values.shape[0] != source_total_frames:
            raise ValueError(
                f"{key} must have {source_total_frames} source frames for causal alignment"
            )
        aligned[key] = np.ascontiguousarray(values[1:source_total_frames])
        shifted_fields.append(key)

    if "action_motion_token" not in shifted_fields:
        raise ValueError("causal alignment requires action_motion_token")
    aligned["recording_phase"] = ALIGNED_RECORDING_PHASE
    aligned["policy_action_phase"] = ALIGNED_POLICY_ACTION_PHASE

    report = validate_sonic_trajectory(aligned)
    if not report.ok:
        details = "\n".join(f"- {error}" for error in report.errors)
        raise ValueError(f"causally aligned trajectory validation failed:\n{details}")

    evidence = {
        "contract": "post_step_observation_with_next_policy_action",
        "source_recording_phase": recording_phase,
        "source_recording_phase_declared": recording_phase_declared,
        "source_policy_action_phase": action_phase,
        "source_policy_action_phase_declared": action_phase_declared,
        "observation_indices_within_selected_interval": [
            0,
            source_total_frames - 1,
        ],
        "action_indices_within_selected_interval": [1, source_total_frames],
        "shifted_fields": shifted_fields,
        "dropped_boundary_frames": 1,
        "export_frames": source_total_frames - 1,
    }
    return aligned, evidence


def load_upstream_provenance(path: str | Path) -> dict[str, Any]:
    """Load, hash, and verify an optional typed upstream provenance bundle."""
    provenance_path = Path(path).resolve()
    if not provenance_path.is_file():
        raise FileNotFoundError(f"upstream provenance JSON does not exist: {provenance_path}")
    raw = provenance_path.read_bytes()
    try:
        content = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid upstream provenance JSON in {provenance_path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError("upstream provenance JSON must contain a top-level object")
    evidence: dict[str, Any] = {
        "path": str(provenance_path),
        "hash": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "content": content,
    }
    from gear_sonic.dataset_generation.schemas import (
        CONVERSION_RESULT_KIND,
        ConversionResult,
        EpisodeRequest,
        GenerationResult,
    )

    if content.get("kind") != CONVERSION_RESULT_KIND:
        return evidence

    conversion = ConversionResult.read_json(provenance_path)
    sibling_payloads: list[tuple[Path, dict[str, Any]]] = []
    for sibling in sorted(provenance_path.parent.glob("*.json")):
        if sibling == provenance_path:
            continue
        try:
            sibling_content = json.loads(sibling.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(sibling_content, dict):
            sibling_payloads.append((sibling, sibling_content))

    generation_candidates = [
        (candidate_path, candidate)
        for candidate_path, candidate in sibling_payloads
        if "generation_result_id" in candidate
    ]
    request_candidates = [
        (candidate_path, candidate)
        for candidate_path, candidate in sibling_payloads
        if "episode_request_id" in candidate
        and "task_prompt" in candidate
        and "generation_result_id" not in candidate
    ]
    if len(generation_candidates) != 1 or len(request_candidates) != 1:
        raise ValueError(
            "typed ConversionResult bundle must contain exactly one sibling "
            "GenerationResult and EpisodeRequest JSON"
        )

    generation_path, generation_content = generation_candidates[0]
    request_path, request_content = request_candidates[0]
    generation = GenerationResult.read_json(generation_path)
    request = EpisodeRequest.read_json(request_path)
    conversion.verify_parent(generation)
    if request.request_id != generation.episode_request_id:
        raise ValueError("EpisodeRequest identity does not match the GenerationResult parent")
    if generation.resolved_model != request.kimodo_model or generation.seed != request.kimodo_seed:
        raise ValueError("GenerationResult model/seed does not match the EpisodeRequest")

    evidence["typed_lineage"] = {
        "episode_request": {
            "path": str(request_path),
            "hash": sha256_file(request_path),
            "content": request_content,
        },
        "generation_result": {
            "path": str(generation_path),
            "hash": sha256_file(generation_path),
            "content": generation_content,
        },
        "episode_request_id": request.request_id,
        "generation_result_id": generation.result_id,
        "conversion_result_id": conversion.result_id,
    }
    return evidence


def validate_upstream_runtime_binding(
    upstream_evidence: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    *,
    scene_id: str,
    scene_hash: str,
    task: str,
) -> dict[str, Any]:
    """Bind the typed request/generation/conversion chain to the Isaac capture."""
    lineage = upstream_evidence.get("typed_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("production export requires a typed ConversionResult provenance bundle")
    request_record = lineage.get("episode_request")
    generation_record = lineage.get("generation_result")
    conversion = upstream_evidence.get("content")
    if not all(
        isinstance(value, Mapping) for value in (request_record, generation_record, conversion)
    ):
        raise ValueError("typed upstream provenance lineage is incomplete")
    request = request_record["content"]
    generation = generation_record["content"]
    assert isinstance(request, Mapping)
    assert isinstance(generation, Mapping)
    assert isinstance(conversion, Mapping)

    expected_request_fields = {
        "scene_id": scene_id,
        "scene_hash": scene_hash,
        "task_prompt": task.strip(),
        "controller_hash": runtime_evidence.get("checkpoint_hash"),
        "physics_hash": runtime_evidence.get("runtime_config_hash"),
    }
    mismatched_request_fields = [
        name for name, expected in expected_request_fields.items() if request.get(name) != expected
    ]
    if mismatched_request_fields:
        raise ValueError(
            "EpisodeRequest does not match the Isaac runtime capture: "
            + ", ".join(mismatched_request_fields)
        )

    capture_context = runtime_evidence.get("capture_context")
    if not isinstance(capture_context, Mapping):
        raise ValueError("runtime capture context is unavailable for upstream binding")
    runtime_motion = capture_context.get("motion")
    if not isinstance(runtime_motion, Mapping) or not isinstance(runtime_motion.get("hash"), str):
        raise ValueError("runtime capture motion hash is unavailable for upstream binding")
    conversion_artifacts = conversion.get("artifacts")
    if not isinstance(conversion_artifacts, list):
        raise ValueError("ConversionResult artifacts are missing")
    motion_outputs = [
        artifact
        for artifact in conversion_artifacts
        if isinstance(artifact, Mapping) and artifact.get("name") == "sonic_motion_lib"
    ]
    if len(motion_outputs) != 1 or motion_outputs[0].get("sha256") != runtime_motion.get("hash"):
        raise ValueError("ConversionResult SONIC motion does not match the runtime motion input")

    if generation.get("resolved_model") != request.get("kimodo_model"):
        raise ValueError("GenerationResult model does not match the EpisodeRequest")
    if generation.get("seed") != request.get("kimodo_seed"):
        raise ValueError("GenerationResult seed does not match the EpisodeRequest")
    scene_start = conversion.get("scene_start_xyz")
    route = request.get("route_xy")
    if (
        not isinstance(scene_start, list)
        or len(scene_start) < 2
        or not isinstance(route, list)
        or not route
        or list(route[0]) != scene_start[:2]
    ):
        raise ValueError("ConversionResult scene start does not match the EpisodeRequest route")

    return {
        "episode_request_id": lineage.get("episode_request_id"),
        "generation_result_id": lineage.get("generation_result_id"),
        "conversion_result_id": lineage.get("conversion_result_id"),
        "runtime_motion_hash": runtime_motion.get("hash"),
        "scene_id": scene_id,
        "scene_hash": scene_hash,
        "task": task.strip(),
    }


def validate_runtime_artifact_binding(
    manifest_path: str | Path,
    *,
    trajectory_path: Path,
    ego_video_path: Path,
    scene_id: str,
    scene_hash: str,
    camera_provenance: str,
    task: str,
) -> dict[str, Any]:
    """Verify that one successful Isaac run bound the inputs and capture semantics."""
    from gear_sonic.dataset_generation.runtime_manifest import read_runtime_success_manifest

    resolved_manifest = Path(manifest_path).resolve()
    payload = read_runtime_success_manifest(resolved_manifest)
    if payload.get("schema_version") != 2:
        raise ValueError(
            "runtime success manifest must use schema_version 2 with artifact hash bindings"
        )
    pairs = payload.get("artifact_pairs")
    if not isinstance(pairs, Mapping) or not pairs:
        raise ValueError("runtime success manifest contains no trajectory/video artifact pairs")

    trajectory_hash = sha256_file(trajectory_path)
    video_hash = sha256_file(ego_video_path)
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for episode_id, raw_pair in pairs.items():
        if not isinstance(raw_pair, Mapping):
            continue
        if (
            raw_pair.get("trajectory_hash") == trajectory_hash
            and raw_pair.get("video_hash") == video_hash
        ):
            matches.append((str(episode_id), raw_pair))
    if len(matches) != 1:
        raise ValueError(
            "trajectory/video hashes are not one uniquely bound pair in the runtime "
            f"success manifest (matches={len(matches)})"
        )

    capture_context = payload.get("capture_context")
    if not isinstance(capture_context, Mapping):
        raise ValueError("runtime success manifest contains no capture_context")
    if capture_context.get("terrain_type") != "scene_usd":
        raise ValueError("runtime capture terrain_type must be scene_usd")
    if capture_context.get("scene_id") != scene_id:
        raise ValueError("claimed scene_id does not match the runtime capture")
    scene = capture_context.get("scene")
    if not isinstance(scene, Mapping) or scene.get("hash") != scene_hash:
        raise ValueError("claimed scene_hash does not match the runtime capture")
    scene_path = scene.get("resolved")
    if not isinstance(scene_path, str) or not Path(scene_path).is_file():
        raise ValueError("runtime capture scene file is missing")
    if sha256_file(Path(scene_path)) != scene_hash:
        raise ValueError("runtime capture scene file no longer matches its recorded hash")
    if capture_context.get("task") != task.strip():
        raise ValueError("task does not match the runtime capture")
    if capture_context.get("use_encoder") != "g1":
        raise ValueError("runtime capture must use the g1 motion-token encoder")

    motion = capture_context.get("motion")
    if not isinstance(motion, Mapping) or not isinstance(motion.get("hash"), str):
        raise ValueError("runtime capture contains no content-addressed motion input")
    motion_path = motion.get("resolved")
    if not isinstance(motion_path, str) or not Path(motion_path).is_file():
        raise ValueError("runtime capture motion file is missing")
    if sha256_file(Path(motion_path)) != motion.get("hash"):
        raise ValueError("runtime capture motion file no longer matches its recorded hash")

    camera = capture_context.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("runtime success manifest contains no camera capture context")
    expected_camera = {
        "provenance": camera_provenance,
        "name": "ego_camera",
        "track_root": False,
        "render_ego": True,
        "resolution": [480, 640],
        "render_frame_skip": 1,
    }
    mismatched_camera_fields = [
        name for name, expected in expected_camera.items() if camera.get(name) != expected
    ]
    if mismatched_camera_fields:
        raise ValueError(
            "runtime camera context does not satisfy the ego dataset contract: "
            + ", ".join(mismatched_camera_fields)
        )

    episode_id, pair = matches[0]
    return {
        "runtime_success_manifest": str(resolved_manifest),
        "runtime_success_manifest_hash": sha256_file(resolved_manifest),
        "runtime_manifest_schema_version": payload["schema_version"],
        "runtime_exit_reason": payload["exit_reason"],
        "runtime_physics_steps": payload["physics_steps"],
        "runtime_policy_iterations": payload["policy_iterations"],
        "runtime_config_hash": payload.get("runtime_config_hash"),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_resolved": payload.get("checkpoint_resolved"),
        "checkpoint_hash": payload.get("checkpoint_hash"),
        "capture_episode_id": episode_id,
        "trajectory_hash": trajectory_hash,
        "video_hash": video_hash,
        "recorded_trajectory_path": pair.get("trajectory_path"),
        "recorded_video_path": pair.get("video_path"),
        "capture_context": capture_context,
    }


def _source_to_mujoco_indices(
    trajectory: Mapping[str, Any],
    *,
    order_key: str,
    names_key: str,
    fallback_order: str,
    fallback_names: Any = None,
) -> tuple[str, tuple[int, ...], bool, tuple[str, ...] | None]:
    """Resolve a declared or legacy G1 source ordering into MuJoCo selection indices."""
    order_declared = order_key in trajectory
    order = trajectory.get(order_key, fallback_order)
    if order not in VALID_G1_DOF_ORDERS:
        raise ValueError(f"{order_key} must be one of {VALID_G1_DOF_ORDERS}; got {order!r}")

    raw_names = trajectory.get(names_key, fallback_names)
    names: tuple[str, ...] | None = None
    if raw_names is not None:
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError(f"{names_key} must be a list or tuple")
        names = tuple(raw_names)
        if (
            len(names) != G1_DOF_DIM
            or len(set(names)) != G1_DOF_DIM
            or set(names) != set(KIMODO_G1_JOINT_NAMES)
        ):
            raise ValueError(f"{names_key} must contain each registered G1 body joint exactly once")
        if order == "mujoco" and names != KIMODO_G1_JOINT_NAMES:
            raise ValueError(f"{names_key} conflicts with declared MuJoCo order")
        indices = tuple(names.index(name) for name in KIMODO_G1_JOINT_NAMES)
    elif order == "mujoco":
        indices = tuple(range(G1_DOF_DIM))
    else:
        # Schema-v2 recorder artifacts created before explicit order metadata
        # are known to use Isaac Articulation order. Preserve compatibility but
        # surface the fallback in provenance.
        indices = G1_ISAACLAB_TO_MUJOCO_DOF
    return order, indices, order_declared, names


def convert_trajectory_joint_order_to_mujoco(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize physics state and reference qpos to the GR00T MuJoCo joint contract."""
    state_order, state_indices, state_declared, state_names = _source_to_mujoco_indices(
        trajectory,
        order_key="dof_order",
        names_key="dof_joint_names",
        fallback_order="isaaclab",
    )
    reference_order, reference_indices, reference_declared, reference_names = (
        _source_to_mujoco_indices(
            trajectory,
            order_key="reference_g1_qpos_dof_order",
            names_key="reference_g1_qpos_joint_names",
            fallback_order=state_order,
            fallback_names=state_names,
        )
    )

    normalized = dict(trajectory)
    for key in ("dof_pos", "dof_vel", "applied_joint_action"):
        if key not in normalized:
            continue
        values = np.asarray(normalized[key])
        if values.ndim < 1 or values.shape[-1] != G1_DOF_DIM:
            raise ValueError(f"{key} must end with {G1_DOF_DIM} G1 joints")
        normalized[key] = np.ascontiguousarray(values[..., state_indices])

    reference_qpos = np.asarray(normalized["reference_g1_qpos"])
    if reference_qpos.ndim < 1 or reference_qpos.shape[-1] != 7 + G1_DOF_DIM:
        raise ValueError("reference_g1_qpos must end with 36 values")
    normalized["reference_g1_qpos"] = np.ascontiguousarray(
        np.concatenate(
            (reference_qpos[..., :7], reference_qpos[..., 7:][..., reference_indices]),
            axis=-1,
        )
    )
    normalized["dof_order"] = "mujoco"
    normalized["dof_joint_names"] = KIMODO_G1_JOINT_NAMES
    normalized["reference_g1_qpos_dof_order"] = "mujoco"
    normalized["reference_g1_qpos_joint_names"] = KIMODO_G1_JOINT_NAMES

    evidence = {
        "source_dof_order": state_order,
        "source_dof_order_declared": state_declared,
        "source_dof_joint_names_declared": state_names is not None,
        "source_reference_dof_order": reference_order,
        "source_reference_dof_order_declared": reference_declared,
        "source_reference_joint_names_declared": reference_names is not None,
        "source_to_mujoco_dof_indices": list(state_indices),
        "source_reference_to_mujoco_dof_indices": list(reference_indices),
        "export_dof_order": "mujoco",
        "export_dof_joint_names": list(KIMODO_G1_JOINT_NAMES),
    }
    return normalized, evidence


def build_export_provenance(
    *,
    trajectory_path: Path,
    ego_video_path: Path,
    trajectory: Mapping[str, Any],
    source_total_frames: int,
    start_frame: int,
    end_frame: int,
    camera_provenance: str,
    scene_id: str | None,
    scene_hash: str | None,
    acceptance: LocomotionAcceptanceReport,
    exported_observation_acceptance: LocomotionAcceptanceReport,
    upstream_provenance_path: str | Path | None,
    joint_order_evidence: Mapping[str, Any] | None = None,
    causal_alignment_evidence: Mapping[str, Any] | None = None,
    runtime_binding_evidence: Mapping[str, Any] | None = None,
    upstream_provenance_evidence: Mapping[str, Any] | None = None,
    upstream_runtime_binding_evidence: Mapping[str, Any] | None = None,
    latent_parity_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable source and acceptance record embedded in info.json."""
    provenance: dict[str, Any] = {
        "profile": "synthetic_g1",
        "source_kind": trajectory["kind"],
        "source_schema_version": trajectory["schema_version"],
        "source_total_frames": source_total_frames,
        "validated_source_frame_range": {
            "start": start_frame,
            "end_exclusive": end_frame,
        },
        "observation_source_frame_range": {
            "start": start_frame,
            "end_exclusive": end_frame - 1,
        },
        "action_source_frame_range": {
            "start": start_frame + 1,
            "end_exclusive": end_frame,
        },
        "source_trajectory": str(trajectory_path),
        "source_trajectory_hash": sha256_file(trajectory_path),
        "ego_video": str(ego_video_path),
        "ego_video_hash": sha256_file(ego_video_path),
        "camera_provenance": camera_provenance,
        "scene_id": scene_id,
        "scene_hash": scene_hash,
        "hand_action_source": "neutral_zero",
        "locomotion_acceptance": acceptance.to_dict(),
        "exported_observation_acceptance": exported_observation_acceptance.to_dict(),
        "causal_alignment": dict(causal_alignment_evidence or {}),
        "runtime_artifact_binding": dict(runtime_binding_evidence or {}),
        "upstream_runtime_binding": dict(upstream_runtime_binding_evidence or {}),
        "joint_order_conversion": dict(
            joint_order_evidence
            if joint_order_evidence is not None
            else convert_trajectory_joint_order_to_mujoco(trajectory)[1]
        ),
        # Which of the two possible 64D latent conventions this dataset carries.
        # `trajectory` is the already-selected clip, and causal alignment writes
        # `action_motion_token[1:]`, so `[1:]` here is exactly the set of values
        # that lands in `action.motion_token`.
        "latent_representation": dict(
            latent_parity_evidence
            if latent_parity_evidence is not None
            else latent_parity_provenance(
                check_latent_parity(np.asarray(trajectory["action_motion_token"])[1:])
            )
        ),
    }
    if upstream_provenance_path is not None:
        provenance["upstream_provenance"] = dict(
            upstream_provenance_evidence
            if upstream_provenance_evidence is not None
            else load_upstream_provenance(upstream_provenance_path)
        )
    return provenance


def load_validated_trajectory(path: str | Path) -> Mapping[str, Any]:
    """Load a recorder pickle and enforce the synthetic-dataset raw gate."""
    trajectory_path = Path(path)
    with trajectory_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - recorder artifacts are trusted local inputs.
    report = validate_sonic_trajectory(payload)
    if not report.ok:
        details = "\n".join(f"- {error}" for error in report.errors)
        raise ValueError(f"trajectory validation failed for {trajectory_path}:\n{details}")
    return payload


def require_accepted_locomotion(
    trajectory: Mapping[str, Any],
) -> LocomotionAcceptanceReport:
    """Reject raw trajectories that did not pass the physics acceptance gate."""
    report = evaluate_locomotion_trajectory(trajectory)
    if report.accepted:
        return report

    details = [*report.errors, *report.rejection_reasons]
    raise ValueError("locomotion acceptance failed: " + ", ".join(details))


def validate_robot_joint_contract(robot_model: Any) -> None:
    """Ensure 29D physics state and RobotModel use the audited G1 order."""
    supplemental_info = getattr(robot_model, "supplemental_info", None)
    actual = tuple(getattr(supplemental_info, "body_actuated_joints", ()))
    if actual != KIMODO_G1_JOINT_NAMES:
        raise ValueError(
            "RobotModel body joint order does not match the Kimodo/SONIC 29-joint contract"
        )


def build_synthetic_g1_frame(
    trajectory: Mapping[str, Any],
    frame_index: int,
    rgb_frame: np.ndarray,
    robot_model: Any,
) -> dict[str, np.ndarray | float]:
    """Build one minimal UNITREE_G1_SONIC LeRobot frame."""
    if rgb_frame.shape != EGO_FRAME_SHAPE:
        raise ValueError(f"ego RGB frame must have shape {EGO_FRAME_SHAPE}; got {rgb_frame.shape}")
    if rgb_frame.dtype != np.uint8:
        raise ValueError(f"ego RGB frame must use uint8; got {rgb_frame.dtype}")

    if (
        trajectory.get("dof_order") != "mujoco"
        or trajectory.get("reference_g1_qpos_dof_order") != "mujoco"
        or tuple(trajectory.get("dof_joint_names", ())) != KIMODO_G1_JOINT_NAMES
        or tuple(trajectory.get("reference_g1_qpos_joint_names", ())) != KIMODO_G1_JOINT_NAMES
    ):
        trajectory, _ = convert_trajectory_joint_order_to_mujoco(trajectory)

    body_q = np.asarray(trajectory["dof_pos"][frame_index], dtype=np.float32)
    neutral_hand = np.zeros(HAND_ACTION_DIM, dtype=np.float32)
    whole_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=neutral_hand,
        right_hand_actuated_joint_values=neutral_hand,
    )
    whole_q = np.asarray(whole_q, dtype=np.float32)
    if whole_q.shape != (43,):
        raise ValueError(f"G1 state assembly must produce 43 joints; got {whole_q.shape}")

    return {
        "observation.images.ego_view": np.ascontiguousarray(rgb_frame),
        "observation.state": whole_q,
        "observation.projected_gravity": np.asarray(
            trajectory["projected_gravity_b"][frame_index], dtype=np.float32
        ),
        "action.motion_token": np.asarray(
            trajectory["action_motion_token"][frame_index], dtype=np.float32
        ),
        "teleop.left_hand_joints": neutral_hand.copy(),
        "teleop.right_hand_joints": neutral_hand.copy(),
        "reference.g1_qpos": np.asarray(
            trajectory["reference_g1_qpos"][frame_index], dtype=np.float32
        ),
    }


def probe_video(path: str | Path) -> VideoInfo:
    """Decode video metadata with OpenCV and reject unreadable inputs."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open ego video: {path}")
        return VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def iter_rgb_video_frames(
    path: str | Path,
    *,
    expected_frames: int,
    expected_fps: float,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[np.ndarray]:
    """Validate the full source video while yielding one aligned frame range."""
    import cv2

    start_frame, end_frame = normalize_frame_range(
        expected_frames,
        start_frame,
        end_frame,
    )
    info = probe_video(path)
    if (info.height, info.width, 3) != EGO_FRAME_SHAPE:
        raise ValueError(
            f"ego video must be {EGO_FRAME_SHAPE[1]}x{EGO_FRAME_SHAPE[0]}; "
            f"got {info.width}x{info.height}"
        )
    if info.frame_count != expected_frames:
        raise ValueError(
            f"video/trajectory frame mismatch: video={info.frame_count}, "
            f"trajectory={expected_frames}"
        )
    if not np.isclose(info.fps, expected_fps, atol=0.05, rtol=0.0):
        raise ValueError(f"video fps={info.fps:g} does not match trajectory fps={expected_fps:g}")

    capture = cv2.VideoCapture(str(path))
    decoded = 0
    yielded = 0
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            source_frame_index = decoded
            decoded += 1
            if start_frame <= source_frame_index < end_frame:
                rgb_frame = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                if rgb_frame.shape != EGO_FRAME_SHAPE or rgb_frame.dtype != np.uint8:
                    raise ValueError(
                        "decoded ego frame must have shape "
                        f"{EGO_FRAME_SHAPE} and dtype uint8; got "
                        f"{rgb_frame.shape} and {rgb_frame.dtype}"
                    )
                yielded += 1
                yield rgb_frame
    finally:
        capture.release()
    if decoded != expected_frames:
        raise ValueError(f"video decoder returned {decoded} frames; expected {expected_frames}")
    expected_yielded = end_frame - start_frame
    if yielded != expected_yielded:
        raise ValueError(
            f"video decoder yielded {yielded} selected frames; expected {expected_yielded}"
        )


def validate_rgb_video(
    path: str | Path,
    *,
    expected_frames: int,
    expected_fps: float,
) -> VideoInfo:
    """Eagerly decode a complete source video before any output is mutated."""
    decoded = 0
    for _ in iter_rgb_video_frames(
        path,
        expected_frames=expected_frames,
        expected_fps=expected_fps,
    ):
        decoded += 1
    if decoded != expected_frames:
        raise ValueError(f"decoded {decoded} video frames; expected {expected_frames}")
    return probe_video(path)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def export_sonic_trajectory(
    *,
    trajectory_path: str | Path,
    ego_video_path: str | Path,
    output_path: str | Path,
    task: str,
    camera_provenance: str,
    runtime_manifest_path: str | Path,
    scene_id: str | None = None,
    scene_hash: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    upstream_provenance_path: str | Path | None = None,
    overwrite_existing: bool = False,
) -> int:
    """Export one physics trajectory plus its aligned Isaac RGB video."""
    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_synthetic_g1,
        get_g1_robot_model,
        get_modality_config_synthetic_g1,
    )

    trajectory_path = Path(trajectory_path).resolve()
    ego_video_path = Path(ego_video_path).resolve()
    output_path = Path(output_path).resolve()
    if not ego_video_path.is_file():
        raise FileNotFoundError(f"ego video does not exist: {ego_video_path}")
    if _is_within(trajectory_path, output_path) or _is_within(ego_video_path, output_path):
        raise ValueError("trajectory and ego video sources must be outside the output directory")
    if output_path.exists() and not overwrite_existing:
        raise FileExistsError(
            f"output dataset already exists: {output_path}; pass overwrite_existing=True "
            "to replace it"
        )
    if not task.strip():
        raise ValueError("task must be non-empty")
    if camera_provenance != "isaac_sim":
        raise ValueError(
            "camera_provenance must be 'isaac_sim'; diagnostic/generated imagery is not "
            "accepted by the production trajectory exporter"
        )
    if not scene_id or not scene_id.strip():
        raise ValueError("scene_id is required for production trajectory export")
    if not scene_hash or not scene_hash.startswith("sha256:"):
        raise ValueError("scene_hash must be a required sha256: content hash")
    if upstream_provenance_path is None:
        raise ValueError("a typed upstream ConversionResult provenance bundle is required")

    runtime_binding_evidence = validate_runtime_artifact_binding(
        runtime_manifest_path,
        trajectory_path=trajectory_path,
        ego_video_path=ego_video_path,
        scene_id=scene_id,
        scene_hash=scene_hash,
        camera_provenance=camera_provenance,
        task=task,
    )
    upstream_provenance_evidence = load_upstream_provenance(upstream_provenance_path)
    upstream_runtime_binding_evidence = validate_upstream_runtime_binding(
        upstream_provenance_evidence,
        runtime_binding_evidence,
        scene_id=scene_id,
        scene_hash=scene_hash,
        task=task,
    )

    source_trajectory = load_validated_trajectory(trajectory_path)
    source_total_frames = int(source_trajectory["total_frames"])
    start_frame, end_frame = normalize_frame_range(
        source_total_frames,
        start_frame,
        end_frame,
    )
    trajectory = slice_trajectory_frames(
        source_trajectory,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    acceptance = require_accepted_locomotion(trajectory)
    causally_aligned_trajectory, causal_alignment_evidence = align_post_step_trajectory_for_bc(
        trajectory
    )
    exported_observation_acceptance = require_accepted_locomotion(causally_aligned_trajectory)
    export_trajectory, joint_order_evidence = convert_trajectory_joint_order_to_mujoco(
        causally_aligned_trajectory
    )
    frame_count = int(causally_aligned_trajectory["total_frames"])
    fps_float = float(causally_aligned_trajectory["fps"])
    if not np.isclose(fps_float, SONIC_DATASET_FPS, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"UNITREE_G1_SONIC export requires {SONIC_DATASET_FPS:g} Hz; " f"got {fps_float:g} Hz"
        )
    fps = int(round(fps_float))
    if not np.isclose(fps_float, fps, atol=1e-6, rtol=0.0):
        raise ValueError(f"LeRobot export requires an integer FPS; got {fps_float:g}")

    robot_model = get_g1_robot_model()
    validate_robot_joint_contract(robot_model)
    validate_rgb_video(
        ego_video_path,
        expected_frames=source_total_frames,
        expected_fps=fps_float,
    )
    frames = iter_rgb_video_frames(
        ego_video_path,
        expected_frames=source_total_frames,
        expected_fps=fps_float,
        start_frame=start_frame,
        end_frame=end_frame - 1,
    )
    provenance = build_export_provenance(
        trajectory_path=trajectory_path,
        ego_video_path=ego_video_path,
        trajectory=causally_aligned_trajectory,
        source_total_frames=source_total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        camera_provenance=camera_provenance,
        scene_id=scene_id,
        scene_hash=scene_hash,
        acceptance=acceptance,
        exported_observation_acceptance=exported_observation_acceptance,
        upstream_provenance_path=upstream_provenance_path,
        joint_order_evidence=joint_order_evidence,
        causal_alignment_evidence=causal_alignment_evidence,
        runtime_binding_evidence=runtime_binding_evidence,
        upstream_provenance_evidence=upstream_provenance_evidence,
        upstream_runtime_binding_evidence=upstream_runtime_binding_evidence,
    )
    exporter = Gr00tDataExporter.create(
        save_root=output_path,
        fps=fps,
        features=get_features_synthetic_g1(robot_model),
        modality_config=get_modality_config_synthetic_g1(robot_model),
        task=task,
        script_config={"trajectory_export": provenance},
        robot_type="unitree_g1_sonic",
        overwrite_existing=overwrite_existing,
    )
    committed = False
    try:
        for frame_index, rgb_frame in enumerate(frames):
            exporter.add_frame(
                build_synthetic_g1_frame(
                    export_trajectory,
                    frame_index,
                    rgb_frame,
                    robot_model,
                )
            )
        exporter.save_episode()
        committed = True
    finally:
        if committed:
            exporter.close()
        else:
            exporter.cancel_video_writers()
    return frame_count
