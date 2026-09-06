#!/usr/bin/env python3
"""Render measured Isaac states with MuJoCo meshes; never integrate dynamics."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_motion2scene_report import archived

from gear_sonic.dataset_generation.hallucination.mujoco_replay import build_mujoco_scene_xml
from gear_sonic.dataset_generation.hallucination.stage_geometry import StageCube, StageGeometry
from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES
from gear_sonic.dataset_generation.trajectory_export import load_validated_trajectory


def model_for(beam):
    robot = ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"
    stage = StageGeometry(
        Path("recorded_beam"),
        1.0,
        "Z",
        (StageCube("temporary", "binding_constraint", (0, 0, -10), (0.1, 0.1, 0.1), None, None),),
        (12.0, 9.0),
    )
    xml, _ = build_mujoco_scene_xml(robot, stage)
    tree = ET.fromstring(xml)
    world = tree.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("lfh_cube_"):
            world.remove(geom)
    h = beam["underside_m"]
    thick = beam.get("thickness_m", 0.1)
    a = beam["yaw_rad"] / 2
    ET.SubElement(
        tree.find("worldbody"),
        "geom",
        name="recorded_beam",
        type="box",
        pos=" ".join(map(str, [*beam["center_xy_m"], h + thick / 2])),
        size=f"{beam['length_m']/2} {beam['width_m']/2} {thick/2}",
        quat=f"{np.cos(a)} 0 0 {np.sin(a)}",
        rgba="0.65 0.37 0.15 1",
        contype="0",
        conaffinity="0",
    )
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)
    )
    if names != KIMODO_G1_JOINT_NAMES:
        raise ValueError("visual joint order mismatch")
    return model, hashlib.sha256(xml.encode()).hexdigest()


def render(result_path, out, seed, variation=False, preview=False):
    result = json.loads(result_path.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    cases = (
        [("nominal", "blind"), ("nominal", "reactive"), ("delay_500ms", "reactive")]
        if variation
        else [("present", "blind"), ("present", "reactive"), ("present", "late_oracle")]
    )
    titles = [
        "Blind walk",
        "Reactive crouch",
        "500 ms observation delay" if variation else "Late request refused",
    ]
    traces = []
    models = []
    sources = []
    for condition, mode in cases:
        row = next(
            r
            for r in result["rows"]
            if r["condition"] == condition and r["mode"] == mode and r["seed"] == seed
        )
        for key in ("trajectory", "sensor", "physics_contacts"):
            ref = row[key]
            checked(Path(ref["path"]), ref["sha256"])
            sources.append(ref)
        qpos, _, fps, source = archived(
            {
                "scientific": {
                    "artifacts": {
                        "trajectory": row["trajectory"]["path"],
                        "trajectory_sha256": row["trajectory"]["sha256"],
                    },
                    "outcome": row["tracker_outcome"],
                }
            }
        )
        sensor = json.loads(Path(row["sensor"]["path"]).read_text())
        recorded = load_validated_trajectory(Path(row["trajectory"]["path"]))
        command_times = np.array([o["time_s"] for o in sensor["observations"]])
        if not np.allclose(
            command_times[:-1], np.asarray(recorded["motion_time_s"]).reshape(-1)[1:], atol=1e-6
        ):
            raise ValueError("command/trajectory alignment changed")
        with np.load(row["physics_contacts"]["path"]) as a:
            force = np.linalg.norm(a["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
        if len(qpos) != 199 or len(force) != 199 or fps != 50:
            raise ValueError("incomplete demonstration capture")
        beam = row.get("beam", {**manifest["beam"], "underside_m": manifest["beam_underside_m"]})
        model, xml_hash = model_for(beam)
        models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 384, 512)))
        traces.append((qpos, sensor, force, row))
        sources.append({"source": source, "visual_xml_sha256": xml_hash, "beam": beam})
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 130
    camera.elevation = -12
    camera.distance = 3.8
    camera.lookat[:] = [*beam["center_xy_m"], 0.9]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 16)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        None
        if preview
        else imageio.get_writer(
            out,
            fps=25,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-movflags", "+faststart"],
        )
    )
    try:
        for i in ([55] if preview else range(100)):
            k = min(198, 2 * i)
            camera.lookat[:2] = (traces[0][0][k, :2] + np.array(beam["center_xy_m"])) / 2
            canvas = Image.new("RGB", (1536, 512), "#101b24")
            draw = ImageDraw.Draw(canvas)
            for j, ((model, data, renderer), (qpos, sensor, force, row), title) in enumerate(
                zip(models, traces, titles)
            ):
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (j * 512, 80))
                state = sensor["observations"][k - 1] if k else {"active": 0, "occupied": False}
                draw.text((j * 512 + 16, 10), title, font=font, fill="#eaf3f5")
                draw.text(
                    (j * 512 + 16, 40),
                    f"{'d040' if state['active'] else 'walk'} | obstacle force: {force[k]:.1f} N",
                    font=small,
                    fill="#ec9c76" if force[k] > 1 else "#9ed7d2",
                )
                draw.text(
                    (j * 512 + 16, 62),
                    f"Full-trial contact-free passage: {'PASS' if row['pass'] else 'FAIL'}",
                    font=small,
                    fill="#c3d9dc",
                )
            draw.text(
                (15, 474),
                f"Recorded Isaac states | MuJoCo visual replay only | seed {seed} | t={k/50:.2f} s",
                font=small,
                fill="#eaf3f5",
            )
            draw.text(
                (15, 494),
                "No mj_step; beam pose from manifest. "
                "Visual robot mesh differs from Isaac collision geometry.",
                font=small,
                fill="#9ed7d2",
            )
            if writer:
                writer.append_data(np.asarray(canvas))
            if i == 55:
                canvas.save(out.with_suffix(".jpg"), quality=94)
    finally:
        for _, _, renderer in models:
            renderer.close()
        if writer:
            writer.close()
    if not preview:
        receipt = {
            "result": artifact(result_path),
            "sources": sources,
            "video": artifact(out),
            "poster": artifact(out.with_suffix(".jpg")),
            "frames": 100,
            "fps": 25,
            "source_fps": 50,
            "selection": "Fixed seed and methods; failures retained",
            "scope": (
                "Measured Isaac qpos, MuJoCo mj_forward visual replay, nearest source frames; "
                "no physics integration. Room omitted; exact instrumented beam shown."
            ),
            "renderer": artifact(Path(__file__)),
        }
        out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variation", action="store_true")
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    render(a.result, a.output, a.seed, a.variation, a.preview)
