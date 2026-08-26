#!/usr/bin/env python3
"""Render the critical frame of a counterfactual pair, side by side, in one figure.

A full replay shows the scene; only the *critical frame* shows why the scene explains the motion.
This renders each motion at the frame the window solver identified as binding, from a fixed camera,
and labels each panel with the executed reach and the face coordinate so the millimetres that
decide the counterfactual are legible rather than implied.

Kinematic ``mj_forward`` replay of recorded Isaac states. MuJoCo supplies pixels, never a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

os_environ_default = {"MUJOCO_GL": "egl"}
import os  # noqa: E402

for key, value in os_environ_default.items():
    os.environ.setdefault(key, value)

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.mujoco_replay import (  # noqa: E402
    build_mujoco_scene_xml,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.trajectory_export import (  # noqa: E402
    convert_trajectory_joint_order_to_mujoco,
    load_validated_trajectory,
)

DEFAULT_ROBOT_XML = REPO_ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"


def render_frame(
    trajectory_path: Path,
    stage_path: Path,
    frame_index: int,
    *,
    robot_xml: Path,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    distance: float,
    lookat_z: float,
) -> np.ndarray:
    trajectory = load_validated_trajectory(trajectory_path.resolve())
    if trajectory.get("quat_format") != "wxyz":
        raise ValueError("replay requires an explicit wxyz root quaternion contract")
    normalized, _ = convert_trajectory_joint_order_to_mujoco(trajectory)
    stage = read_stage_geometry(stage_path.resolve())
    xml, _ = build_mujoco_scene_xml(robot_xml.resolve(), stage)

    with tempfile.TemporaryDirectory(prefix="lfh_critical_frame_") as temp_dir:
        temp_xml = Path(temp_dir) / "scene.xml"
        temp_xml.write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        root = np.asarray(normalized["root_pos_w"])
        total = int(trajectory["total_frames"])
        index = int(np.clip(frame_index, 0, total - 1))
        # Centre the camera on the robot at the critical frame, not on the whole route, so the
        # binding geometry fills the panel.
        camera.lookat[:] = (float(root[index, 0]), float(root[index, 1]), lookat_z)
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        try:
            data.qpos[:3] = root[index]
            data.qpos[3:7] = np.asarray(normalized["root_quat_w"])[index]
            data.qpos[7:] = np.asarray(normalized["dof_pos"])[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()


def _label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn a small caption block into the top-left of a panel, without extra dependencies."""
    from PIL import Image, ImageDraw

    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    pad, line_height = 6, 14
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle([(0, 0), (picture.width, box_height)], fill=(12, 12, 16))
    for row, text in enumerate(lines):
        draw.text((pad, pad + row * line_height), text, fill=(235, 235, 240))
    return np.asarray(picture)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-report", type=Path, default=REPO_ROOT / "docs/hallucination/e17_ladder_scenes.json"
    )
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--cell", choices=("hard", "easy"), default="hard")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--azimuth", type=float, default=90.0)
    parser.add_argument("--elevation", type=float, default=-8.0)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--lookat-z", type=float, default=1.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.scene_report.read_text())
    scenes = report["scenes"]
    entry = (
        next(scene for scene in scenes if scene["pair_id"] == args.pair_id)
        if args.pair_id
        else scenes[0]
    )
    stage = REPO_ROOT / entry["scenes"][args.cell]
    coordinate = entry[f"{args.cell}_coordinate_m"]
    spec = json.loads((REPO_ROOT / entry["spec"]).read_text())
    reach_nominal = spec["binding"]["reach_orig_m"]
    reach_adapted = spec["binding"]["reach_edit_m"]

    panels = []
    for role, trajectory, frame, reach in (
        (
            "nominal",
            Path(entry["reference_motions"]["nominal"]),
            entry["critical_frames"]["nominal"],
            reach_nominal,
        ),
        (
            "adapted",
            Path(entry["reference_motions"]["adapted"]),
            entry["critical_frames"]["adapted"],
            reach_adapted,
        ),
    ):
        image = render_frame(
            trajectory,
            stage,
            frame,
            robot_xml=args.robot_xml,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat_z=args.lookat_z,
        )
        gap_mm = 1000 * (coordinate - reach)
        verdict = "STRIKES" if gap_mm < 0 else "CLEARS"
        panels.append(
            _label(
                image,
                [
                    f"{role.upper()}  frame {frame}  ({verdict} by {abs(gap_mm):.1f} mm)",
                    f"executed reach {reach:.4f} m   face underside {coordinate:.4f} m",
                ],
            )
        )

    figure = np.concatenate(panels, axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, figure)
    print(f"wrote {args.out}")
    print(
        f"  {entry['pair_id']} ({entry['body_mode']}), {args.cell} cell, "
        f"{entry['archetype']} at {coordinate:.4f} m"
    )
    print(
        f"  nominal reach {reach_nominal:.4f} m -> "
        f"{1000 * (coordinate - reach_nominal):+.1f} mm;  adapted reach {reach_adapted:.4f} m -> "
        f"{1000 * (coordinate - reach_adapted):+.1f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
