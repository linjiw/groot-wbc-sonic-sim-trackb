#!/usr/bin/env python3
"""Show all four fixed learners on the observed integration encounter, not a test ranking."""

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
from gear_sonic.dataset_generation.trajectory_export import (
    convert_trajectory_joint_order_to_mujoco,
)


def render(result_path, output):
    result = json.loads(result_path.read_text())
    manifest = json.loads(
        checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"]).read_text()
    )
    arms = ("uniform", "analytic", "no_contrast", "motion2scene")
    traces, models, refs = [], [], []
    for arm in arms:
        row = next(
            r for r in result["rows"] if r["group_id"] == "analytic_01" and r["policy_arm"] == arm
        )
        cell = next(c for c in manifest["cells"] if c["cell_id"] == row["cell_id"])
        ref = row["trajectory"]
        p = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
        normalized, order = convert_trajectory_joint_order_to_mujoco(p)
        qpos = np.concatenate(
            [normalized[k] for k in ("root_pos_w", "root_quat_w", "dof_pos")], axis=1
        )
        folder = Path(cell["output"]) / "trajectories"
        with np.load(folder / "physics_beam_contacts.npz") as raw:
            forces = np.linalg.norm(raw["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
        assert len(qpos) == len(forces) == 199 and float(p["fps"]) == 50
        model, xml_hash = model_for(cell["beam"])
        models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 320, 512)))
        traces.append((qpos, forces, row, cell))
        refs.append(
            {
                "trajectory": ref,
                "decision": row["decision"],
                "joint_order": order,
                "visual_xml_sha256": xml_hash,
                "physics_contacts": artifact(folder / "physics_beam_contacts.npz"),
            }
        )
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 20), ImageFont.truetype(font_path, 16)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 130, -12, 3.8
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output,
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    try:
        for i in range(100):
            k = min(198, 2 * i)
            canvas = Image.new("RGB", (1024, 832), "#101b24")
            draw = ImageDraw.Draw(canvas)
            for j, ((model, data, renderer), (qpos, force, row, cell)) in enumerate(
                zip(models, traces)
            ):
                x, y = (j % 2) * 512, (j // 2) * 380
                camera.lookat[:] = [*cell["beam"]["center_xy_m"], 0.9]
                camera.lookat[:2] = (
                    traces[0][0][k, :2] + np.array(cell["beam"]["center_xy_m"])
                ) / 2
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (x, y + 60))
                decision = row["readout"]
                label = (
                    "refusal; robot walks"
                    if decision["refusal"]
                    else ("d040 request" if decision["requested_action"] else "walk commitment")
                )
                draw.text((x + 12, y + 5), f"{arms[j]} data | {label}", font=font, fill="#eaf3f5")
                draw.text(
                    (x + 12, y + 33),
                    f"Force {force[k]:.1f} N | encounter {'PASS' if row['pass'] else 'FAIL'} "
                    f"| resets {row['reset_count']}",
                    font=small,
                    fill="#9ed7d2",
                )
            draw.text(
                (12, 765),
                "Observed integration scene analytic_01; analytic arm saw it during training.",
                font=small,
                fill="#eaf3f5",
            )
            draw.text(
                (12, 787),
                "All four seed-8501 learners; physics seed 8602. Not an independent comparison.",
                font=small,
                fill="#eaf3f5",
            )
            draw.text(
                (12, 809),
                f"Recorded Isaac states | MuJoCo visuals only, no mj_step | elapsed {k/50:.2f} s",
                font=small,
                fill="#9ed7d2",
            )
            writer.append_data(np.asarray(canvas))
            if i == 55:
                canvas.save(output.with_suffix(".jpg"), quality=94)
    finally:
        writer.close()
        for _, _, renderer in models:
            renderer.close()
    receipt = {
        "result": artifact(result_path),
        "sources": refs,
        "video": artifact(output),
        "poster": artifact(output.with_suffix(".jpg")),
        "renderer": artifact(Path(__file__)),
        "frames": 100,
        "fps": 25,
        "scope": (
            "all fixed learners on observed analytic_01; full capture interval with resets retained; "
            "nearest-frame MuJoCo visual replay only; robot visual meshes differ from "
            "Isaac colliders; room omitted"
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.result, args.output)
