#!/usr/bin/env python3
"""Replay all four physically labeled transition-construction runs."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from motion2scene_timing_diagnostic import artifact, checked
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_motion2scene_execution_demo import model_for

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.trajectory_export import convert_trajectory_joint_order_to_mujoco


def render(out, output):
    admission = json.loads((out / "admission.json").read_text())
    assert admission["admitted"]
    result = json.loads(
        checked(Path(admission["result"]["path"]), admission["result"]["sha256"]).read_text()
    )
    rows = sorted(result["rows"], key=lambda r: (r["physics_seed"], r["action"]))
    assert len(rows) == 4
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    sources = []
    models = []
    traces = []
    for r in rows:
        p = load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        p, order = convert_trajectory_joint_order_to_mujoco(p)
        q = np.concatenate([p[k] for k in ("root_pos_w", "root_quat_w", "dof_pos")], 1)
        with np.load(checked(Path(r["physics"]["path"]), r["physics"]["sha256"])) as f:
            force = np.linalg.norm(f["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
        model, xml = model_for(r["beam"])
        models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 320, 512)))
        traces.append((q, force))
        sources.append(
            {
                "cell_id": r["cell_id"],
                "trajectory": r["trajectory"],
                "physics": r["physics"],
                "decision": r["decision"],
                "joint_order": order,
                "xml_sha256": xml,
            }
        )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 130
    camera.elevation = -12
    camera.distance = 3.8
    try:
        with imageio.get_writer(
            output,
            fps=25,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-movflags", "+faststart"],
        ) as writer:
            for i in range(100):
                k = min(198, 2 * i)
                canvas = Image.new("RGB", (1024, 848), "#101b24")
                draw = ImageDraw.Draw(canvas)
                for j, ((model, data, renderer), (q, force), r) in enumerate(
                    zip(models, traces, rows)
                ):
                    x, y = (j % 2) * 512, (j // 2) * 380
                    data.qpos[:] = q[k]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [*r["beam"]["center_xy_m"], 0.9]
                    camera.lookat[:2] = (
                        traces[0][0][k, :2] + np.array(r["beam"]["center_xy_m"])
                    ) / 2
                    renderer.update_scene(data, camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()), (x, y + 60))
                    draw.text(
                        (x + 12, y + 5),
                        f"41001 / {r['physics_seed']} | {'request d040' if r['action'] else 'walk commit'}",
                        font=font,
                        fill="#eaf3f5",
                    )
                    draw.text(
                        (x + 12, y + 33),
                        f"{'PASS' if r['pass'] else 'FAIL'} | "
                        f"scored {r['maximum_beam_normal_force_n_through_passage']:.1f} N | now {force[k]:.1f} N",
                        font=small,
                        fill="#9ed7d2",
                    )
                draw.text(
                    (12, 765),
                    f"Analytic achieved-transition scene | station 0.68 | underside 1.2755 m | {k/50:.2f} s",
                    font=small,
                    fill="#eaf3f5",
                )
                draw.text(
                    (12, 788),
                    "All 4 obstacle-present runs; 11/12 assigned scene slots refused. No learned selector here.",
                    font=small,
                    fill="#eaf3f5",
                )
                draw.text(
                    (12, 811),
                    "Isaac dynamics and contact | MuJoCo recorded-state visuals only | "
                    "endpoint tracking rejection retained",
                    font=small,
                    fill="#9ed7d2",
                )
                writer.append_data(np.asarray(canvas))
                if i == 64:
                    canvas.save(output.with_suffix(".jpg"), quality=94)
    finally:
        for _, _, renderer in models:
            renderer.close()
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "result": admission["result"],
                "sources": sources,
                "renderer": artifact(Path(__file__)),
                "video": artifact(output),
                "poster": artifact(output.with_suffix(".jpg")),
                "frames": 100,
                "fps": 25,
                "scope": (
                    "All four physical runs; nominal task success differs from full-reference "
                    "endpoint tracking. No mj_step; room omitted, visual meshes differ from collision assets."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.out, a.output)
