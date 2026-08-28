#!/usr/bin/env python3
"""Cross-render a recorded SONIC/Isaac LFH trajectory in MuJoCo.

The robot state and simulator verdict come from the validated recorder pickle.  MuJoCo receives
those states kinematically with ``mj_forward``; no integration step is taken, so this artifact is
visual cross-check evidence and never a second physics verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.mujoco_replay import (  # noqa: E402
    build_mujoco_scene_xml,
    mapping_error_m,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    KIMODO_G1_JOINT_NAMES,
)
from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_export import (  # noqa: E402
    convert_trajectory_joint_order_to_mujoco,
    load_validated_trajectory,
    sha256_file,
)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(1, model.njnt)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("gear_sonic_deploy/g1/g1_29dof_old.xml"),
    )
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--azimuth", type=float, default=140.0)
    parser.add_argument("--elevation", type=float, default=-17.0)
    parser.add_argument("--distance", type=float, default=4.7)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.frame_stride <= 0:
        parser.error("width, height, and frame-stride must be positive")
    trajectory_path = args.trajectory.resolve()
    stage_path = args.stage.resolve()
    robot_xml = args.robot_xml.resolve()
    trajectory = load_validated_trajectory(trajectory_path)
    if trajectory.get("quat_format") != "wxyz":
        raise ValueError("MuJoCo replay requires an explicit wxyz root quaternion contract")
    normalized, joint_evidence = convert_trajectory_joint_order_to_mujoco(trajectory)
    acceptance = evaluate_locomotion_trajectory(trajectory)
    stage = read_stage_geometry(stage_path)
    xml, mappings = build_mujoco_scene_xml(robot_xml, stage)

    with tempfile.TemporaryDirectory(prefix="lfh_mujoco_replay_") as temp_dir:
        temp_xml = Path(temp_dir) / "scene.xml"
        temp_xml.write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        if model.nq != 36:
            raise ValueError(f"expected 36 MuJoCo qpos values, got {model.nq}")
        actual_joint_names = _model_joint_names(model)
        if actual_joint_names != KIMODO_G1_JOINT_NAMES:
            raise ValueError("MuJoCo model joint order differs from the audited trajectory order")
        compiled_mappings = []
        geometry_errors = []
        for mapping in mappings:
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                mapping.mujoco_geom_name,
            )
            if geom_id < 0:
                raise ValueError(f"compiled MuJoCo model is missing {mapping.mujoco_geom_name}")
            compiled_center = tuple(float(value) for value in model.geom_pos[geom_id])
            compiled_size = tuple(float(2.0 * value) for value in model.geom_size[geom_id])
            center_error = max(
                abs(source - compiled)
                for source, compiled in zip(mapping.center_m, compiled_center)
            )
            size_error = max(
                abs(source - compiled) for source, compiled in zip(mapping.size_m, compiled_size)
            )
            geometry_errors.append((center_error, size_error))
            compiled_mappings.append(
                {
                    **mapping.to_dict(),
                    "compiled_mujoco_center_m": list(compiled_center),
                    "compiled_mujoco_full_size_m": list(compiled_size),
                    "compiled_center_error_m": center_error,
                    "compiled_full_size_error_m": size_error,
                }
            )
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        root_positions = np.asarray(normalized["root_pos_w"])
        camera.lookat[:] = (
            float((root_positions[:, 0].min() + root_positions[:, 0].max()) / 2.0),
            float((root_positions[:, 1].min() + root_positions[:, 1].max()) / 2.0),
            1.0,
        )
        camera.azimuth = args.azimuth
        camera.elevation = args.elevation
        camera.distance = args.distance

        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            args.out_video,
            fps=float(trajectory["fps"]) / args.frame_stride,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        desired_poster_frame = int(np.argmin(np.abs(root_positions[:, 0] - 1.75)))
        rendered_indices = range(0, int(trajectory["total_frames"]), args.frame_stride)
        poster_frame = min(rendered_indices, key=lambda index: abs(index - desired_poster_frame))
        poster_rgb = None
        rendered_frames = 0
        try:
            for frame_index in range(0, int(trajectory["total_frames"]), args.frame_stride):
                data.qpos[:3] = normalized["root_pos_w"][frame_index]
                data.qpos[3:7] = normalized["root_quat_w"][frame_index]
                data.qpos[7:] = normalized["dof_pos"][frame_index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                writer.append_data(rgb)
                rendered_frames += 1
                if frame_index == poster_frame:
                    poster_rgb = rgb
        finally:
            writer.close()
            renderer.close()

    if args.poster is not None:
        if poster_rgb is None:
            raise RuntimeError("poster frame was not rendered")
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.poster, poster_rgb)

    construction_errors = [mapping_error_m(mapping) for mapping in mappings]
    if any(max(error) != 0.0 for error in construction_errors):
        raise RuntimeError("constructed MuJoCo half sizes do not reconstruct the source cubes")
    max_center_error = max(error[0] for error in geometry_errors)
    max_size_error = max(error[1] for error in geometry_errors)
    max_nonfoot_contact = float(np.max(trajectory["max_nonfoot_contact_force_n"]))
    report = {
        "schema_version": "lfh_mujoco_cross_render_v1",
        "semantics": "kinematic_cross_render_only_not_a_mujoco_physics_rollout",
        "physics_verdict_source": "validated_sonic_isaac_recorder_trajectory",
        "source_trajectory": str(trajectory_path),
        "source_trajectory_sha256": sha256_file(trajectory_path),
        "source_stage": str(stage_path),
        "source_stage_sha256": sha256_file(stage_path),
        "robot_xml": str(robot_xml),
        "robot_xml_sha256": sha256_file(robot_xml),
        "generated_mjcf_sha256": _sha256_bytes(xml.encode("utf-8")),
        "source_quat_format": trajectory["quat_format"],
        "joint_order_conversion": joint_evidence,
        "isaac_physics_verdict": {
            "accepted": acceptance.accepted,
            "rejection_reasons": list(acceptance.rejection_reasons),
            "max_nonfoot_contact_force_n": max_nonfoot_contact,
        },
        "render": {
            "video": str(args.out_video.resolve()),
            "video_sha256": sha256_file(args.out_video),
            "poster": str(args.poster.resolve()) if args.poster is not None else None,
            "width": args.width,
            "height": args.height,
            "source_fps": float(trajectory["fps"]),
            "output_fps": float(trajectory["fps"]) / args.frame_stride,
            "rendered_frames": rendered_frames,
            "camera": {
                "lookat": camera.lookat.tolist(),
                "azimuth": camera.azimuth,
                "elevation": camera.elevation,
                "distance": camera.distance,
            },
        },
        "geometry_bridge": {
            "rendered_roles": ["binding_constraint", "constraint_context"],
            "intentionally_hidden_roles": ["room_shell"],
            "mappings": compiled_mappings,
            "max_center_error_m": max_center_error,
            "max_full_size_error_m": max_size_error,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "video": report["render"]["video"],
                "rendered_frames": rendered_frames,
                "isaac_accepted": acceptance.accepted,
                "isaac_rejection_reasons": list(acceptance.rejection_reasons),
                "max_geometry_error_m": max(max_center_error, max_size_error),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
