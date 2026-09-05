#!/usr/bin/env python3
# ruff: noqa: E402
"""Build traceable Motion2Scene report media from local archived experiments.

Only mj_forward is used: videos are visual replays, never new physics evidence.
Inputs are trusted local research artifacts, not arbitrary downloaded pickles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gear_sonic.dataset_generation.hallucination.mujoco_replay import build_mujoco_scene_xml
from gear_sonic.dataset_generation.hallucination.stage_geometry import (
    StageCube,
    StageGeometry,
    read_stage_geometry,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES
from gear_sonic.dataset_generation.trajectory_export import (
    convert_trajectory_joint_order_to_mujoco,
    load_validated_trajectory,
)


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def portable(value):
    """Keep artifact identity while removing machine-specific absolute prefixes."""
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/home/"):
        for marker in ("research-data/groot-wbc/", "motion2scene/", "groot-wbc-sonic-sim-trackb/"):
            if marker in value:
                return value.split(marker, 1)[1]
        return Path(value).name
    return value


def archived(cell):
    artifacts = cell["scientific"]["artifacts"]
    path = Path(artifacts["trajectory"])
    assert digest(path) == artifacts["trajectory_sha256"]
    payload = load_validated_trajectory(path)
    assert payload["quat_format"] == "wxyz"
    normalized, order = convert_trajectory_joint_order_to_mujoco(payload)
    achieved = np.concatenate(
        [normalized["root_pos_w"], normalized["root_quat_w"], normalized["dof_pos"]], axis=1
    )
    return (
        achieved,
        normalized["reference_g1_qpos"],
        float(payload["fps"]),
        {
            "artifact": portable(str(path)),
            "sha256": digest(path),
            "joint_order": order,
            "scientific_outcome": cell["scientific"]["outcome"],
        },
    )


def scene_model(beam=None):
    robot = ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"
    if beam is None:
        stage_path = ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
        stage = read_stage_geometry(stage_path)
        xml, _ = build_mujoco_scene_xml(robot, stage, rendered_roles=("wall",))
    else:
        x, y, h = beam
        cubes = (
            StageCube(
                "illustration/beam",
                "binding_constraint",
                (x, y, h + 0.06),
                (0.12, 1.8, 0.12),
                None,
                None,
            ),
            StageCube(
                "illustration/left_post",
                "constraint_context",
                (x, y - 0.92, (h + 0.12) / 2),
                (0.12, 0.08, h + 0.12),
                None,
                None,
            ),
            StageCube(
                "illustration/right_post",
                "constraint_context",
                (x, y + 0.92, (h + 0.12) / 2),
                (0.12, 0.08, h + 0.12),
                None,
                None,
            ),
        )
        stage = StageGeometry(Path("illustrative_beam"), 1.0, "Z", cubes, (12.0, 9.0))
        xml, _ = build_mujoco_scene_xml(robot, stage)
    model = mujoco.MjModel.from_xml_string(xml)
    assert (
        tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt))
        == KIMODO_G1_JOINT_NAMES
    )
    return model, hashlib.sha256(xml.encode()).hexdigest()


def render(out, name, arrays, fps, titles, footer, sources, beam=None, preview=False):
    model, xml_hash = scene_model(beam)
    data = mujoco.MjData(model)
    panel_w, panel_h = (640 if len(arrays) == 2 else 512), 448
    renderer = mujoco.Renderer(model, height=panel_h, width=panel_w)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    all_pos = np.concatenate([a[:, :3] for a in arrays])
    camera.lookat[:] = [*np.mean([all_pos[:, :2].min(0), all_pos[:, :2].max(0)], axis=0), 0.77]
    camera.azimuth, camera.elevation, camera.distance = 125.0, -12.0, 3.3
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    duration = min((len(a) - 1) / fps for a in arrays)
    frame_count = int(duration * 30) + 1
    video = out / f"{name}.mp4"
    writer = (
        None
        if preview
        else imageio.get_writer(
            video,
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-movflags", "+faststart"],
        )
    )
    try:
        indices = [frame_count // 2] if preview else range(frame_count)
        for i in indices:
            follow_index = min(len(arrays[0]) - 1, int(round(i / 30 * fps)))
            camera.lookat[:2] = arrays[0][follow_index, :2]
            canvas = Image.new("RGB", (panel_w * len(arrays), panel_h + 96), "#101b24")
            draw = ImageDraw.Draw(canvas)
            for j, (qpos, title) in enumerate(zip(arrays, titles)):
                k = min(len(qpos) - 1, int(round(i / 30 * fps)))
                data.qpos[:] = qpos[k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (j * panel_w, 48))
                draw.text((j * panel_w + 20, 13), title, font=font, fill="#eaf3f5")
                if j:
                    draw.line((j * panel_w, 0, j * panel_w, panel_h + 48), fill="#415664", width=2)
            draw.text((20, panel_h + 63), footer, font=small, fill="#9ed7d2")
            draw.text(
                (panel_w * len(arrays) - 95, panel_h + 63),
                f"{i / 30:.2f} s",
                font=small,
                fill="white",
            )
            if writer:
                writer.append_data(np.asarray(canvas))
            if i == frame_count // 2:
                canvas.save(out / f"{name}.jpg", quality=93)
    finally:
        renderer.close()
        if writer:
            writer.close()
    return {
        "name": name,
        "sources": sources,
        "source_fps": fps,
        "output_fps": 30,
        "sampling": "nearest recorded frame at common elapsed time; no interpolation or physics integration",
        "frames": frame_count,
        "camera": {
            "azimuth": 125,
            "elevation": -12,
            "distance": 3.3,
            "tracking": "same first-panel root XY camera for every panel; fixed height 0.77 m",
        },
        "generated_mjcf_sha256": xml_hash,
        "beam_xyz_underside_m": beam,
        "evidence": (
            "reference-only illustrative scene"
            if beam
            else "archived Isaac states, kinematic MuJoCo replay"
        ),
    }


def figures(out, corpus, shared, heldout):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f5f7f5",
            "axes.facecolor": "#f5f7f5",
        }
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    modes = ("duck_under", "arm_tuck")
    bottom = np.zeros(2)
    for status, label, color in [
        ("absent", "S0 absent", "#9ba4ac"),
        ("elicited", "S1 elicited", "#e1b770"),
        ("localized", "S2 localized", "#4e8baf"),
        ("route_aligned", "S3 route-aligned", "#197d72"),
    ]:
        values = [corpus["summary"]["semantics_by_mode"][m].get(status, 0) for m in modes]
        ax[0].barh(["Duck", "Arm tuck"], values, left=bottom, label=label, color=color)
        for y, (left, v) in enumerate(zip(bottom, values)):
            if v:
                ax[0].text(left + v / 2, y, str(v), ha="center", va="center", fontsize=10)
        bottom += values
    ax[0].set(
        xlim=(0, 24),
        xlabel="References per family (8 seeds × 3 routes)",
        title="Prompt compliance is not motion qualification",
    )
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, -0.23), ncol=2, fontsize=9, frameon=False)
    bars = ax[1].bar(
        ["Neutral", "40 mm", "55 mm"], [8, 3, 1], color=["#197d72", "#4e8baf", "#e1b770"]
    )
    ax[1].bar_label(bars, labels=["8/8", "3/8", "1/8"], padding=4)
    ax[1].set(
        ylim=(0, 9.3),
        ylabel="Accepted obstacle-absent runs",
        title="Controlled crouching challenges the tracker",
    )
    fig.savefig(out / "qualification.svg", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for level, color in zip(shared["levels"], ["#ac523d", "#287da1", "#197d72"]):
        heights = level["whole_body_top_m"]
        ax.plot(np.linspace(0, 1, len(heights)), heights, label=level["label"], color=color, lw=2)
    ax.axvspan(0.35, 0.75, color="#197d72", alpha=0.07, label="Central event region (illustration)")
    ax.set(
        xlabel="Normalized route progress",
        ylabel="Whole-body top envelope (m)",
        title="Shared-clock ladder · carrier 41002 · reference geometry only",
    )
    ax.legend(ncol=4, fontsize=9, loc="lower left")
    fig.savefig(out / "height-profiles.svg", bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(heldout / "q3/route_comparison.svg", out / "route-comparison.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene/assets")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    corpus_root = args.data_root / "cg-wbc-v2-shared-seed-confirmatory"
    heldout_root = args.data_root / "cg-wbc-route-retention-heldout-v1"
    original_root = corpus_root / "e1_controlled_duck_ladders_v1"
    paths = {
        "corpus": corpus_root / "e0_shared_seed_reference_eval_v2.json",
        "original_candidates": original_root / "candidates.json",
        "original_runs": original_root / "q3_v1/run_record.json",
        "original_retention": original_root / "q3_v1/retention_v1.json",
        "heldout": heldout_root / "q3/relative_retention_result_v2.json",
        "heldout_runs": heldout_root / "q3/run_record.json",
        "shared_clock": corpus_root / "e1_shared_clock_duck_v1/candidates.json",
    }
    evidence = {k: read_json(p) for k, p in paths.items()}
    assert evidence["heldout"]["analysis_complete"] and evidence["heldout"]["completed"] == 8
    best, ref, fps, source = archived(
        evidence["heldout_runs"]["cells"]["m2s_route_heldout_seed_42001"]
    )
    media = [
        render(
            out,
            "best-walk",
            [ref, best],
            fps,
            ["42001 / reference", "42001 / recorded execution"],
            "Archived open-space run | MuJoCo visual replay, not a new physics test",
            [source],
            preview=args.preview,
        )
    ]
    arrays, sources = [], []
    for label in ("neutral", "d055"):
        achieved, _, fps, source = archived(
            evidence["original_runs"]["cells"][f"m2s_e1q3_duck_seed_41002__{label}"]
        )
        arrays.append(achieved)
        sources.append(source)
    media.append(
        render(
            out,
            "tracked-crouch",
            arrays,
            fps,
            ["41002 / recorded neutral", "41002 / recorded d055 crouch"],
            "Both tracker-accepted | Localized crouch, not S4-qualified | No beam in the run",
            sources,
            preview=args.preview,
        )
    )
    group = next(g for g in evidence["shared_clock"]["ladders"] if g["generation_seed"] == 41002)
    arrays, sources = [], []
    for level in group["levels"]:
        assert digest(level["csv"]) == level["csv_sha256"]
        q = np.loadtxt(level["csv"], delimiter=",")
        assert q.shape[1] == 36 and np.isfinite(q).all()
        arrays.append(q)
        sources.append({"artifact": portable(level["csv"]), "sha256": level["csv_sha256"]})
    mid = len(arrays[0]) // 2
    beam = [float(arrays[0][mid, 0]), float(arrays[0][mid, 1]), 1.275]
    media.append(
        render(
            out,
            "beam-ladder",
            arrays,
            30,
            ["Reference / neutral", "Reference / d055", "Reference / d085"],
            "Illustrative beam at 1.275 m | Reference replay only | No obstacle-present execution",
            sources,
            beam=beam,
            preview=args.preview,
        )
    )
    if args.preview:
        return
    figures(out, evidence["corpus"], group, heldout_root)
    evidence_dir = out.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for key in ("heldout", "original_retention", "shared_clock"):
        write_json(evidence_dir / f"{key}.json", portable(evidence[key]))
    write_json(
        evidence_dir / "corpus.json",
        {k: portable(v) for k, v in evidence["corpus"].items() if k != "rows"},
    )
    # Versioned input snapshots retain complete failure denominators; rendered examples are selected.
    prompt = args.research_repo / "configs/prompts/kimodo_shared_seed_factorial_v2.txt"
    shutil.copyfile(prompt, evidence_dir / "prompts-v2.txt")
    protocols = {
        "E1_ROUTE_RETENTION_HELDOUT_V1.json": "configs/e1_route_retention_heldout_v1.json",
        "E1_SHARED_CLOCK_DUCK_V1.json": "experiments/registrations/E1_SHARED_CLOCK_DUCK_V1.json",
        "E0_SHARED_SEED_V2.json": "configs/e0_kimodo_shared_seed_factorial_v2.json",
    }
    for destination, relative_source in protocols.items():
        source_path = args.research_repo / relative_source
        write_json(evidence_dir / destination, portable(read_json(source_path)))
    write_json(
        out / "manifest.json",
        {
            "schema_version": "motion2scene_public_report_v1",
            "date": "2026-09-05",
            "renderer_sha256": digest(__file__),
            "robot_xml_sha256": digest(ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml"),
            "sources": {
                k: {"artifact": portable(str(p)), "sha256": digest(p)} for k, p in paths.items()
            },
            "selection": {
                "best_walk": "minimum cross-track RMSE among held-out tracker-and-relative-route passes",
                "tracked_crouch": "only tracker-accepted d055 example; localized effect but paired S4 fails",
                "beam_ladder": (
                    "same 41002 carrier for visual continuity; beam manually set for illustration, "
                    "not admitted by a verifier"
                ),
            },
            "media": media,
            "outputs": {
                p.name: digest(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
            "public_evidence": {
                p.name: digest(p) for p in sorted(evidence_dir.iterdir()) if p.is_file()
            },
            "evidence_limit": (
                "No new physics, Q4 qualification, obstacle-present preference reversal, "
                "or learned hallucinator result."
            ),
        },
    )


if __name__ == "__main__":
    main()
