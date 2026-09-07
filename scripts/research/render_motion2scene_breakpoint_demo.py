#!/usr/bin/env python3
"""Replay all 24 shared-linear-control executions, including failed conditions."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from motion2scene_comparative_acquisition import ARMS
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


def render(out, output):
    admission = json.loads((out / "admission.json").read_text())
    assert admission["admitted"]
    ref = admission["result"]
    result = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 20), ImageFont.truetype(font_path, 16)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    with imageio.get_writer(
        output,
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for block_index in range(6):
            block_rows = [r for r in result["rows"] if r["original_block"] == block_index]
            assert len(block_rows) == 4
            physics_seed = block_rows[0]["physics_seed"]
            beam = block_rows[0]["beam"]
            traces, models = [], []
            for arm in ARMS:
                row = next(r for r in block_rows if r["arm"] == arm)
                ref = row["trajectory"]
                p = load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))
                p, order = convert_trajectory_joint_order_to_mujoco(p)
                qpos = np.concatenate(
                    [p[k] for k in ("root_pos_w", "root_quat_w", "dof_pos")], axis=1
                )
                ref = row["physics"]
                with np.load(checked(Path(ref["path"]), ref["sha256"])) as raw:
                    forces = np.linalg.norm(raw["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
                assert len(qpos) == len(forces) == 199 and float(p["fps"]) == 50
                model, xml_hash = model_for(beam)
                models.append((model, mujoco.MjData(model), mujoco.Renderer(model, 320, 512)))
                traces.append((qpos, forces, row))
                sources.append(
                    {
                        "cell_id": row["cell_id"],
                        "trajectory": row["trajectory"],
                        "physics": row["physics"],
                        "decision": row["decision"],
                        "joint_order": order,
                        "visual_xml_sha256": xml_hash,
                    }
                )
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = 130, -12, 3.8
            try:
                for i in range(100):
                    k = min(198, 2 * i)
                    canvas = Image.new("RGB", (1024, 848), "#101b24")
                    draw = ImageDraw.Draw(canvas)
                    for j, ((model, data, renderer), (qpos, force, row)) in enumerate(
                        zip(models, traces)
                    ):
                        x, y = (j % 2) * 512, (j // 2) * 380
                        camera.lookat[:] = [*beam["center_xy_m"], 0.9]
                        camera.lookat[:2] = (
                            traces[0][0][k, :2] + np.array(beam["center_xy_m"])
                        ) / 2
                        data.qpos[:] = qpos[k]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera)
                        canvas.paste(Image.fromarray(renderer.render()), (x, y + 60))
                        decision = row["readout"]
                        label = (
                            "refusal; walk"
                            if decision["refusal"]
                            else ("d040 request" if decision["requested_action"] else "walk")
                        )
                        draw.text(
                            (x + 12, y + 5), f"{ARMS[j]} | {label}", font=font, fill="#eaf3f5"
                        )
                        peak = row["maximum_beam_normal_force_n_through_passage"]
                        draw.text(
                            (x + 12, y + 33),
                            f"Scored peak {peak:.1f} N | now {force[k]:.1f} N | "
                            f"{'PASS' if row['pass'] else 'FAIL'}",
                            font=small,
                            fill="#9ed7d2",
                        )
                    draw.text(
                        (12, 765),
                        f"Development station 0.35 | beam underside {beam['underside_m']:.2f} m "
                        f"| elapsed {k/50:.2f} s",
                        font=small,
                        fill="#eaf3f5",
                    )
                    draw.text(
                        (12, 788),
                        f"Four common linear controls; physics seed {physics_seed}; source 41002.",
                        font=small,
                        fill="#eaf3f5",
                    )
                    draw.text(
                        (12, 811),
                        "Recorded Isaac states | MuJoCo visuals only, no mj_step | all three heights shown",
                        font=small,
                        fill="#9ed7d2",
                    )
                    writer.append_data(np.asarray(canvas))
                    if block_index == 3 and i == 33:
                        canvas.save(output.with_suffix(".jpg"), quality=94)
            finally:
                for _, _, renderer in models:
                    renderer.close()
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "result": artifact(out / "result.json"),
                "admission": artifact(out / "admission.json"),
                "sources": sources,
                "video": artifact(output),
                "poster": artifact(output.with_suffix(".jpg")),
                "renderer": artifact(Path(__file__)),
                "frames": 600,
                "fps": 25,
                "selection": (
                    "four deterministic linear controls and physics seeds 8511/8512, "
                    "all three first-station heights and all four arms"
                ),
                "scope": (
                    "all 24 development linear-control executions; full capture intervals retained, "
                    "every other recorded frame; MuJoCo mj_forward visuals only; "
                    "room omitted, visual meshes differ from Isaac colliders"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.out, args.output)
