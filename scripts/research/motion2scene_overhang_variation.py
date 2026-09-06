#!/usr/bin/env python3
"""Register and audit a complete pose/delay/blocked-underpass Isaac panel."""

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

import joblib
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reference_bank import audit_root_route

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-overhang-eval-bank-v2"
PREFIX = "gear_sonic.dataset_generation.hallucination.motion2scene_delayed_overhang_execution"
POSES = ("nominal", "height_minus10mm", "height_plus10mm", "station_minus15cm", "station_plus15cm")


def prepare(out):
    prior = json.loads((PRIOR / "manifest.json").read_text())
    for ref in prior["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    assert all(json.loads((PRIOR / "result.json").read_text())["predictions"].values())
    assert json.loads((PRIOR / "bank_audit.json").read_text())["p5"]
    out.mkdir(parents=True, exist_ok=False)
    specs = []
    h = prior["beam_underside_m"]
    normal = np.array([np.cos(prior["beam"]["yaw_rad"]), np.sin(prior["beam"]["yaw_rad"])])
    for name in (*POSES, "absent", "raised", "blocked"):
        beam = {k: prior["beam"][k] for k in ("center_xy_m", "yaw_rad", "length_m", "width_m")}
        beam.update(underside_m=h, thickness_m=0.1)
        if name.startswith("height"):
            beam["underside_m"] += -0.01 if "minus" in name else 0.01
        if name.startswith("station"):
            delta = -0.15 if "minus" in name else 0.15
            beam["center_xy_m"] = (np.array(beam["center_xy_m"]) + delta * normal).tolist()
        if name == "raised":
            beam["underside_m"] = 2.0
        if name == "blocked":
            beam.update(underside_m=0.0, thickness_m=h + 0.1)
        path = out / f"{name}.usda"
        scene_file(path, beam)
        text = path.read_text()
        prefix, cube = text.split('    def Cube "CounterfactualBeam"', 1)
        if name == "absent":
            cube = cube.replace(
                "physics:collisionEnabled = true", "physics:collisionEnabled = false"
            )
        if name == "blocked":
            cube = re.sub(
                r"double3 xformOp:translate = [^\n]+",
                f"double3 xformOp:translate = ({beam['center_xy_m'][0]}, "
                f"{beam['center_xy_m'][1]}, {beam['thickness_m']/2})",
                cube,
            )
            cube = cube.replace(
                "xformOp:scale = (0.1, 1.2, 0.1)",
                f"xformOp:scale = (0.1, 1.2, {beam['thickness_m']})",
            )
        path.write_text(prefix + '    def Cube "CounterfactualBeam"' + cube)
        specs.append(
            {
                "condition": name,
                "beam": beam,
                "delay_s": 0.0,
                "scene": {**artifact(path), "scene_id": name},
            }
        )
    nominal = specs[0]
    specs += [
        {**copy.deepcopy(nominal), "condition": f"delay_{ms}ms", "delay_s": ms / 1000}
        for ms in (100, 250, 500)
    ]
    base = next(
        c for c in prior["cells"] if c["condition"] == "present" and c["mode"] == "reactive"
    )
    cells = []
    for seed in (8041, 8042):
        for spec in specs:
            for mode in (
                ("reactive", "blind", "oracle") if spec["condition"] in POSES else ("reactive",)
            ):
                c = {**copy.deepcopy(base), **copy.deepcopy(spec)}
                c.update(
                    mode=mode,
                    runtime_seed=seed,
                    cell_id=f"variation_{seed}_{spec['condition']}_{mode}",
                )
                c["output"] = str(out / "rollouts" / c["cell_id"])
                c["hydra_overrides"] = [
                    f"++seed={seed}",
                    f"++manager_env._target_={PREFIX}.DelayedOverhangEnvCfg",
                    f"++manager_env.recorders.trajectory._target_={PREFIX}.DelayedOverhangRecorderCfg",
                    next(s for s in base["hydra_overrides"] if "beam_contact_body_names" in s),
                    f"++manager_env.config.reactive_alternate_path={prior['alternate_motion']['path']}",
                    f"++manager_env.config.reactive_mode={mode}",
                    f"++manager_env.config.observation_delay_s={c['delay_s']}",
                ]
                cells.append(c)
    assert len(cells) == 42
    m = copy.deepcopy(prior)
    m.update(
        experiment="M2S-overhang-variation-v1",
        cells=cells,
        purpose="Finite pose changes, causal delay and solid blocked-underpass control",
        registered_predictions=artifact(ROOT / "docs/motion2scene/OVERHANG_VARIATION_V1.md"),
    )
    m["new_dependencies"] += [
        artifact(Path(__file__)),
        artifact(ROOT / "docs/motion2scene/OVERHANG_VARIATION_V1.md"),
        *[
            artifact(PRIOR / f"{name}.json")
            for name in ("manifest", "result", "bank_audit", "run_record")
        ],
        *[
            artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{name}.py")
            for name in (
                "motion2scene_observation_delay",
                "motion2scene_delayed_overhang_execution",
            )
        ],
    ]
    m["execution_policy"]["cost_ceiling"].update(rollouts=42, gpu_hours_contended=42 * 375 / 3600)
    m["execution_policy"]["timing_override"] = (
        "User requests next registered study and publication. "
        "Conservative prior M2S records plus supplement below 2 GPU h; ceiling 4.375 fits daily 8 h."
    )
    m["comparison_contract"] = (
        "Five poses x three modes x two seeds; six additional delayed reactive cells "
        "share the zero-delay nominal blind/oracle controls; six negative-control cells. "
        "42 executions, not 66 independent repeats."
    )
    write_new(out / "manifest.json", m)


def analyze(out):
    path = out / "manifest.json"
    m = json.loads(path.read_text())
    record = json.loads((out / "run_record.json").read_text())
    if record["status"] != "completed" or record["manifest_sha256"] != artifact(path)["sha256"]:
        raise ValueError("requires completed hash-bound full panel")
    rows = []
    first = None
    for c in m["cells"]:
        payload, sampled, row = read_cell(
            {**c, "condition": "absent" if c["condition"] == "absent" else "present"}, record
        )
        beam = c["beam"]
        matrix = np.asarray(row["imported_beam"]["local_to_world_at_capture_start"])
        if not np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
        ):
            raise ValueError("scene center differs from registration")
        a = beam["yaw_rad"]
        scales = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])
        rotation = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        if not np.allclose(matrix[:3, :3], scales[:, None] * rotation, atol=1e-6):
            raise ValueError("scene scale or orientation differs")
        folder = Path(c["output"]) / "trajectories"
        sensor_path = folder / "reactive_interface.json"
        physics_path = folder / "physics_beam_contacts.npz"
        sensor = json.loads(sensor_path.read_text())
        obs = sensor["observations"]
        switches = sensor["switches"]
        with np.load(physics_path) as physics:
            blocks, aggregate, error = physics_windows(physics, sampled)
        if blocks.shape != (199, 4, 30, 3) or len(obs) != 199:
            raise ValueError("incomplete capture")
        delay_frames = int(np.ceil(c["delay_s"] * 50 - 1e-9))
        for i, o in enumerate(obs):
            d = o["delivered"]
            j = i - delay_frames
            if o["capture_frame"] != i:
                raise ValueError("wrong packet index")
            expected = (
                None if j < 0 else {k: obs[j][k] for k in ("time_s", "occupied", "origin", "rays")}
            )
            if expected is not None:
                expected.update(capture_frame=j, capture_elapsed_s=j / 50)
            if d != expected or o["delivered_occupied"] != bool(d and d["occupied"]):
                raise ValueError("noncausal or altered delivered packet")
        score = score_passage(payload, aggregate, beam)
        horizon = score["passage_finish_frame_exclusive"] or score["first_episode_frames"]
        norms = np.linalg.norm(blocks[:horizon].reshape(-1, 30, 3), axis=-1)
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(all(o["transition"][k] for k in flags) for o in obs) and all(
            all(s[k] for k in flags)
            and s["joint_reference_jump_rad"] <= 0.05
            and s["root_reference_jump_m"] <= 0.01
            and (0.2 <= s["time_s"] <= 0.4 if s["to"] else 3.3 <= s["time_s"] <= 3.5)
            for s in switches
        )
        with np.load(folder / "loaded_reference_bank.npz") as bank:
            roots = bank["root_xyz"]
            dofs = bank["joint_pos"]
            if roots.shape != (2, 199, 3) or dofs.shape != (2, 199, 29) or float(bank["fps"]) != 50:
                raise ValueError("invalid bank")
            errors = []
            for i, ref in enumerate((c["motion"], m["alternate_motion"])):
                checked(Path(ref["path"]), ref["sha256"])
                entry = next(iter(joblib.load(ref["path"]).values()))
                errors.append(
                    audit_root_route(roots[i], entry["root_trans_offset"], float(entry["fps"]))
                )
            if first is None:
                first = (roots.copy(), dofs.copy())
            difference = max(
                float(np.max(np.abs(roots - first[0]))), float(np.max(np.abs(dofs - first[1])))
            )
        meta = json.loads((folder / "loaded_reference_bank.json").read_text())
        if (
            meta["source_route_max_error_m"] != errors
            or meta["alternate_loader"] != "load_motions_for_evaluation"
        ):
            raise ValueError("bank metadata mismatch")
        denied = [o for o in obs if o["transition"]["attempted"] and not o["transition"]["allowed"]]
        row.update(
            score,
            condition=c["condition"],
            mode=c["mode"],
            seed=c["runtime_seed"],
            beam=beam,
            requested_delay_s=c["delay_s"],
            realized_delay_s=delay_frames / 50,
            sensor=artifact(sensor_path),
            physics_contacts=artifact(physics_path),
            switches=switches,
            raw_overhang_frames=sum(o["occupied"] for o in obs),
            raw_upper_frames=sum(o["upper_occupied"] for o in obs),
            delivered_positive_frames=sum(o["delivered_occupied"] for o in obs),
            first_delivered_positive_s=next(
                (o["time_s"] for o in obs if o["delivered_occupied"]), None
            ),
            denied_transition_frames=len(denied),
            denied_reasons=sorted(
                {reason for o in denied for reason in o["transition"]["reasons"]}
            ),
            legal_switch_contract=legal,
            causal_packet_audit=True,
            contact_sync_max_error_n=error,
            physics_contact_duration_s=float((norms.max(1) > 1).sum() * 0.005),
            sum_body_normal_magnitude_impulse_ns=float(norms.sum() * 0.005),
            bank=artifact(folder / "loaded_reference_bank.npz"),
            bank_metadata=artifact(folder / "loaded_reference_bank.json"),
            source_route_errors_m=errors,
            max_difference_from_first_bank=difference,
        )
        rows.append(row)
    assert len(rows) == 42
    nominal = [r for r in rows if r["condition"] == "nominal"]
    poses = [r for r in rows if r["condition"] in POSES]
    negatives = [r for r in rows if r["condition"] in ("absent", "raised", "blocked")]
    late = [r for r in rows if r["condition"] == "delay_500ms"]
    predictions = {
        "p1_nominal": all(
            r["pass"] if r["mode"] != "blind" else r["observed_beam_contact"] for r in nominal
        ),
        "p2_pose_separation": all(
            r["pass"] if r["mode"] != "blind" else r["observed_beam_contact"] for r in poses
        ),
        "p3_negative_specificity": all(
            r["raw_overhang_frames"] == 0 and not r["switches"] for r in negatives
        ),
        "p4_delay_boundary": all(
            not r["switches"] and r["denied_transition_frames"] > 0 and not r["pass"] for r in late
        ),
        "p5_measurement_contract": all(
            r["legal_switch_contract"]
            and r["causal_packet_audit"]
            and r["contact_sync_max_error_n"] <= 1e-5
            and r["max_difference_from_first_bank"] == 0
            for r in rows
        ),
    }
    write_new(
        out / "result.json",
        {
            "manifest": artifact(path),
            "run_record": artifact(out / "run_record.json"),
            "rows": rows,
            "predictions": predictions,
            "new_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "scope": (
                "One development source, five beam poses, three packet delays, three negatives, "
                "two physics seeds. Scripted binary interface; no learned policy."
            ),
        },
    )
    print(json.dumps(predictions))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out)
        return
    m = json.loads((a.out / "manifest.json").read_text())
    for r in m["new_dependencies"]:
        checked(Path(r["path"]), r["sha256"])
    if a.command == "analyze":
        analyze(a.out)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(a.out / "manifest.json"),
        "--run-record",
        str(a.out / "run_record.json"),
    ]
    if a.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if a.command == "run":
        analyze(a.out)


if __name__ == "__main__":
    main()
