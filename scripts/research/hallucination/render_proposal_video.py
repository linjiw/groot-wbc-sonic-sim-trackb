#!/usr/bin/env python3
"""Render a proposed critical scene against the motion that produced it.

For a clip in the case study this recomputes the proposal -- binding station, route-frame yaw,
face coordinate at the window centre -- places the obstacle **oriented along the executed route**,
and replays the nominal and adapted references side by side through it.

This is a *proposal* visualisation, not a verified family. The motions are reference clips replayed
kinematically, not executed rollouts, and the obstacle coordinate comes from D_phi's predicted
reaches rather than from measured ones. Geometry is authored from accepted executed reaches only;
nothing here is authored, and no verdict is implied. What it shows is where the method puts an
obstacle for a given motion, which is the question worth looking at across many motions at once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import (  # noqa: E402
    extract_keypoints,
)
from gear_sonic.dataset_generation.hallucination.reach import (  # noqa: E402
    overhead_face_reach,
    route_frame_yaw,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_payload import (  # noqa: E402
    payload_from_reference,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    box_clearance_to_cloud,
    swept_point_cloud,
)
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402
from scripts.research.hallucination.screen_crouch_ladder import (  # noqa: E402
    STATION_FRACTION,
    WINDOW_FRACTION,
)

ROBOT_XML = REPO_ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 2.0
FACE_THICKNESS_M = 0.06

#: A small inventory of context shapes, in the route frame: (along, across, vertical) full extents
#: and whether the item stands on the floor or hangs. Context carries no causal label -- it is
#: scene dressing that must stay clear of both swept volumes -- but its shape, size, position and
#: yaw are all sampled, which is what makes two scenes at the same critical point look different.
CONTEXT_INVENTORY = (
    ("pillar", (0.28, 0.28, 2.40), "floor"),
    ("crate", (0.55, 0.45, 0.55), "floor"),
    ("low_barrier", (0.18, 1.30, 0.75), "floor"),
    ("wall_segment", (0.20, 2.00, 2.30), "floor"),
    ("duct", (0.90, 0.35, 0.35), "hang"),
    ("shelf_stack", (0.60, 0.90, 1.60), "floor"),
    ("beam_offcut", (1.40, 0.22, 0.22), "hang"),
)


def _numbers(values) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _obb_overlap(
    centre_a, size_a, yaw_a, centre_b, size_b, yaw_b, *, margin_m: float = 0.0
) -> bool:
    """Separating-axis test for two yaw-only oriented boxes, inflated by ``margin_m``."""
    if abs(centre_a[2] - centre_b[2]) > (size_a[2] + size_b[2]) / 2.0 + margin_m:
        return False
    centres = np.asarray(centre_b[:2]) - np.asarray(centre_a[:2])
    axes = []
    for yaw in (yaw_a, yaw_b):
        axes.append(np.asarray((math.cos(yaw), math.sin(yaw))))
        axes.append(np.asarray((-math.sin(yaw), math.cos(yaw))))
    for axis in axes:
        reach_a = sum(
            abs(float(np.dot(axis, direction))) * extent / 2.0
            for direction, extent in (
                (np.asarray((math.cos(yaw_a), math.sin(yaw_a))), size_a[0]),
                (np.asarray((-math.sin(yaw_a), math.cos(yaw_a))), size_a[1]),
            )
        )
        reach_b = sum(
            abs(float(np.dot(axis, direction))) * extent / 2.0
            for direction, extent in (
                (np.asarray((math.cos(yaw_b), math.sin(yaw_b))), size_b[0]),
                (np.asarray((-math.sin(yaw_b), math.cos(yaw_b))), size_b[1]),
            )
        )
        if abs(float(np.dot(axis, centres))) > reach_a + reach_b + margin_m:
            return False
    return True


def sample_context(
    station: tuple[float, float],
    yaw: float,
    *,
    count: int,
    keepout_m: float,
    rng: np.random.Generator,
    cloud_points: np.ndarray,
    cloud_radii: np.ndarray,
    binding: dict,
) -> list[dict]:
    """Sample keep-out-certified context obstacles in the route frame.

    Shape, size, along/lateral position and yaw are all drawn; a draw is kept only if it clears
    both swept bodies by ``keepout_m`` **and** does not intersect the binding obstacle. Rejected
    draws are refused rather than moved.

    Clearance uses the repository's own swept-volume primitive, which densifies along each capsule
    axis and subtracts capsule radii. An earlier version of this function measured against capsule
    *endpoints only* with a circumscribing-radius bound, discarding radii and capsule interiors;
    that under-measured the body by up to 165 mm and certified placements as clear at 131.8 mm
    that were actually 84.0 mm away. The box is yaw-rotated, so the cloud is transformed into the
    box frame and the query box becomes axis-aligned there.
    """
    forward = np.asarray((math.cos(yaw), math.sin(yaw)))
    lateral = np.asarray((-math.sin(yaw), math.cos(yaw)))
    placed: list[dict] = []
    for _ in range(count * 40):
        if len(placed) >= count:
            break
        name, (along, across, vertical), mount = CONTEXT_INVENTORY[
            int(rng.integers(len(CONTEXT_INVENTORY)))
        ]
        scale = float(rng.uniform(0.75, 1.35))
        size = (along * scale, across * scale, vertical * scale)
        offset_along = float(rng.uniform(-2.2, 2.2))
        side = 1.0 if rng.random() < 0.5 else -1.0
        offset_lateral = side * float(rng.uniform(0.75, 2.4))
        centre_xy = np.asarray(station) + forward * offset_along + lateral * offset_lateral
        base_z = 0.0 if mount == "floor" else float(rng.uniform(1.55, 2.10))
        centre = (float(centre_xy[0]), float(centre_xy[1]), base_z + size[2] / 2.0)
        item_yaw = yaw + float(rng.normal(0.0, 0.35))

        # Context must not fuse into the binding face: an orange box welded to the blue plank is
        # exactly the second, unlabelled cause the keep-out discipline exists to prevent.
        if _obb_overlap(
            centre,
            size,
            item_yaw,
            binding["center_m"],
            binding["full_size_m"],
            binding["yaw_rad"],
            margin_m=keepout_m,
        ):
            continue

        cos, sin = math.cos(-item_yaw), math.sin(-item_yaw)
        rotation = np.asarray(((cos, -sin), (sin, cos)))
        local = cloud_points - np.asarray(centre)[None, :]
        local = np.concatenate((local[:, :2] @ rotation.T, local[:, 2:3]), axis=1)
        half = np.asarray(size) / 2.0
        clearance = box_clearance_to_cloud(
            local, cloud_radii, (-half[0], -half[1], -half[2], half[0], half[1], half[2])
        )
        if clearance < keepout_m:
            continue
        placed.append(
            {
                "name": f"{name}_{len(placed)}",
                "center_m": centre,
                "full_size_m": size,
                "yaw_rad": item_yaw,
                "min_clearance_m": float(clearance),
            }
        )
    return placed


def build_scene(robot_xml: Path, obstacle: dict | None, context: list[dict] | None = None) -> str:
    """Robot model plus floor, lights, one yaw-oriented binding box, and sampled context."""
    root = ET.parse(robot_xml.resolve()).getroot()
    # An ElementTree element with no children is falsy, so `find(...) or SubElement(...)` quietly
    # creates a second element and discards the original's attributes -- here, meshdir.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = Path(compiler.get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = robot_xml.resolve().parent / meshdir
    compiler.set("meshdir", str(meshdir.resolve()))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "prop_grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.20 0.23",
            "rgb2": "0.28 0.31 0.35",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "prop_floor", "texture": "prop_grid", "texrepeat": "10 10", "reflectance": "0.06"},
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"robot model has no worldbody: {robot_xml}")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "prop_floor_plane",
            "type": "plane",
            "pos": "0 0 -0.002",
            "size": "8 8 0.05",
            "material": "prop_floor",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for name, pos, direction, diffuse in (
        ("prop_key", "1.8 -2.5 5.0", "0.0 0.25 -1.0", "0.85 0.85 0.85"),
        ("prop_fill", "2.0 3.0 3.0", "0.0 -0.4 -1.0", "0.45 0.48 0.55"),
    ):
        ET.SubElement(
            worldbody,
            "light",
            {"name": name, "pos": pos, "dir": direction, "directional": "true", "diffuse": diffuse},
        )
    for item in context or []:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"prop_ctx_{item['name']}",
                "type": "box",
                "pos": _numbers(item["center_m"]),
                "euler": _numbers((0.0, 0.0, item["yaw_rad"])),
                "size": _numbers([value / 2.0 for value in item["full_size_m"]]),
                "rgba": "0.82 0.45 0.16 0.85",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    if obstacle is not None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "prop_binding_face",
                "type": "box",
                "pos": _numbers(obstacle["center_m"]),
                "euler": _numbers((0.0, 0.0, obstacle["yaw_rad"])),
                "size": _numbers([value / 2.0 for value in obstacle["full_size_m"]]),
                "rgba": "0.18 0.42 0.85 0.92",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    return ET.tostring(root, encoding="unicode")


def proposal_for(qpos: np.ndarray, predict, target_drop_m: float) -> dict:
    root_xy = qpos[:, :2]
    progress = route_progress(np.asarray(root_xy, dtype=np.float64))
    index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    station = (float(root_xy[index, 0]), float(root_xy[index, 1]))
    span = np.ptp(root_xy, axis=0)
    axis = "x" if span[0] >= span[1] else "y"

    adapted, report = local_crouch(
        qpos, STATION_FRACTION, target_drop_m=target_drop_m, window=WINDOW_FRACTION
    )
    nominal_tracks = extract_keypoints(payload_from_reference(qpos))
    adapted_tracks = extract_keypoints(payload_from_reference(adapted))
    yaw = route_frame_yaw(nominal_tracks, station)

    def reach(tracks) -> float:
        face = overhead_face_reach(
            tracks,
            station,
            axis,
            FACE_ALONG_M,
            FACE_ACROSS_M,
            require_all_groups=False,
            route_yaw_rad=yaw,
        )
        return float(face.reach_m)

    nominal_reach, adapted_reach = predict(reach(nominal_tracks)), predict(reach(adapted_tracks))
    coordinate = 0.5 * (nominal_reach + adapted_reach)
    return {
        "station_xy_m": station,
        "route_axis": axis,
        "yaw_rad": yaw,
        "adapted_qpos": adapted,
        "predicted_nominal_reach_m": nominal_reach,
        "predicted_adapted_reach_m": adapted_reach,
        "predicted_window_mm": 1000 * (nominal_reach - adapted_reach),
        "coordinate_m": coordinate,
        "reference_drop_mm": 1000 * report.silhouette_drop_m,
        "obstacle": {
            "center_m": (
                station[0],
                station[1],
                coordinate + FACE_THICKNESS_M / 2.0,
            ),
            "full_size_m": (FACE_ALONG_M, FACE_ACROSS_M, FACE_THICKNESS_M),
            "yaw_rad": yaw,
        },
    }


def _caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    pad, line_height = 5, 13
    draw.rectangle([(0, 0), (picture.width, pad * 2 + line_height * len(lines))], fill=(10, 10, 14))
    for row, text in enumerate(lines):
        draw.text((pad, pad + row * line_height), text, fill=(235, 235, 240))
    return np.asarray(picture)


def render_pair(
    qpos_by_role: dict,
    obstacle: dict,
    out_path: Path,
    *,
    width,
    height,
    fps,
    captions: dict | None = None,
    context: list[dict] | None = None,
) -> int:
    xml = build_scene(ROBOT_XML, obstacle, context)
    with tempfile.TemporaryDirectory(prefix="lfh_proposal_") as temp_dir:
        path = Path(temp_dir) / "scene.xml"
        path.write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        station = obstacle["center_m"]
        camera.lookat[:] = (station[0], station[1], 0.85)
        camera.azimuth = math.degrees(obstacle["yaw_rad"]) + 125.0
        camera.elevation = -16.0
        camera.distance = 5.6
        frames = min(len(value) for value in qpos_by_role.values())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=None)
        try:
            for frame in range(frames):
                panels = []
                for role, qpos in sorted(qpos_by_role.items()):
                    data.qpos[:] = qpos[frame]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render().copy()
                    if captions:
                        image = _caption(image, captions.get(role, []))
                    panels.append(image)
                writer.append_data(np.concatenate(panels, axis=1))
        finally:
            writer.close()
            renderer.close()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-study", type=Path, default=REPO_ROOT / "docs/hallucination/case_study.json"
    )
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "docs/hallucination/reach_delivery_model.json"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument("--index", type=int, action="append")
    parser.add_argument("--all-proposed", action="store_true")
    parser.add_argument("--all", action="store_true", help="every clip in the case study")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=440)
    parser.add_argument("--height", type=int, default=330)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--context", type=int, default=5, help="sampled context obstacles")
    parser.add_argument("--keepout-mm", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    fit = json.loads(args.model.read_text())["fitted_linear"]

    def predict(value: float) -> float:
        return fit["slope"] * value + fit["intercept_m"]

    cases = {c["motion_index"]: c for c in json.loads(args.case_study.read_text())["cases"]}
    if args.all:
        wanted = sorted(cases)
    elif args.all_proposed:
        wanted = [i for i, c in sorted(cases.items()) if c["decision"] == "propose"]
    else:
        wanted = [i for i in (args.index or []) if i in cases]
    if args.limit:
        wanted = wanted[: args.limit]
    if not wanted:
        raise SystemExit("no cases selected")

    records = []
    for position, index in enumerate(wanted, start=1):
        case = cases[index]
        source = sorted(args.source_dir.glob(f"{index:03d}_*.csv"))[0]
        qpos = np.loadtxt(source, delimiter=",")
        usable = [
            rung
            for rung in case["rungs"]
            if rung.get("refusal") is None or rung.get("refusal") == "window_below_min"
        ]
        usable = [rung for rung in usable if rung.get("reference_gate")] or case["rungs"]
        rung = max(usable, key=lambda r: r.get("predicted_window_mm", 0.0))
        proposal = proposal_for(qpos, predict, rung["target_drop_mm"] / 1000.0)
        # Both swept bodies as a densified point cloud with radii, so clearance is conservative.
        clouds = []
        for clip in (qpos, proposal["adapted_qpos"]):
            payload = payload_from_reference(clip)
            points, radii = swept_point_cloud(
                np.asarray(payload["body_pos_w"], dtype=np.float64),
                np.asarray(payload["body_quat_w"], dtype=np.float64),
                list(payload["body_names"]),
            )
            clouds.append((points.reshape(-1, 3), radii.reshape(-1)))
        cloud_points = np.concatenate([item[0] for item in clouds], axis=0)
        cloud_radii = np.concatenate([item[1] for item in clouds], axis=0)
        context = sample_context(
            proposal["station_xy_m"],
            proposal["yaw_rad"],
            count=args.context,
            keepout_m=args.keepout_mm / 1000.0,
            rng=np.random.default_rng(args.seed + index),
            cloud_points=cloud_points,
            cloud_radii=cloud_radii,
            binding=proposal["obstacle"],
        )
        out_path = args.out_dir / f"case_{index:03d}_{case['body_mode']}.mp4"
        # The caption must say whether the case study admitted this rung, so a viewer cannot
        # mistake a refused proposal for a family. 55 of the 94 clips are refused.
        verdict = (
            "PROPOSED"
            if case["decision"] == "propose"
            else "REFUSED (" + ", ".join(case["refusal_reasons"]) + ")"
        )
        shared = [
            f"{index:03d} {case['body_mode']}  {case['route']['turn_sign']}  {verdict}",
            f"straightness {case['route']['straightness']:.2f}  "
            f"straightness {case['route']['straightness']:.2f}",
            f"route yaw {math.degrees(proposal['yaw_rad']):+.0f} deg  "
            f"context {len(context)}  "
            f"misalign {case['route']['face_misalignment_deg']:.0f} deg  "
            f"plank {proposal['coordinate_m']:.3f} m",
        ]
        frames = render_pair(
            {"a_nominal": qpos, "b_adapted": proposal["adapted_qpos"]},
            proposal["obstacle"],
            out_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            context=context,
            captions={
                "a_nominal": [
                    f"NOMINAL   reach {proposal['predicted_nominal_reach_m']:.3f} m (predicted)"
                ]
                + shared,
                "b_adapted": [
                    f"ADAPTED {rung['target_drop_mm']:.0f} mm  reach "
                    f"{proposal['predicted_adapted_reach_m']:.3f} m   "
                    f"window {proposal['predicted_window_mm']:.0f} mm"
                ]
                + shared,
            },
        )
        record = {
            "motion_index": index,
            "body_mode": case["body_mode"],
            "prompt": case["prompt"],
            "turn_sign": case["route"]["turn_sign"],
            "straightness": case["route"]["straightness"],
            "face_misalignment_deg": case["route"]["face_misalignment_deg"],
            "decision": case["decision"],
            "commanded_drop_mm": rung["target_drop_mm"],
            "route_yaw_deg": math.degrees(proposal["yaw_rad"]),
            "predicted_window_mm": proposal["predicted_window_mm"],
            "obstacle_underside_m": proposal["coordinate_m"],
            "video": str(out_path),
            "frames": frames,
            "context_obstacles": [
                {
                    "name": item["name"],
                    "full_size_m": item["full_size_m"],
                    "yaw_deg": math.degrees(item["yaw_rad"]),
                    "min_clearance_mm": 1000 * item["min_clearance_m"],
                }
                for item in context
            ],
            "status": "proposal_only_not_a_verified_family",
        }
        records.append(record)
        print(
            f"[{position}/{len(wanted)}] {index:03d} {case['body_mode']:<14} "
            f"yaw {record['route_yaw_deg']:7.1f} deg  window {record['predicted_window_mm']:6.1f} mm "
            f"-> {out_path.name}",
            flush=True,
        )

    if args.manifest:
        args.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "lfh_proposal_videos_v1",
                    "contract": (
                        "proposal visualisations from reference clips and predicted reaches; "
                        "no scene is authored and no physics verdict is implied"
                    ),
                    "videos": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(f"\n{len(records)} videos -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
