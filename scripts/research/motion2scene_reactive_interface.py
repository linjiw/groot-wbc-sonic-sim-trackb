#!/usr/bin/env python3
"""Register, run and reconcile the phase-aligned Isaac interface pilot."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_source_execution import read_cell, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-beam-intervention-resource-v3"
PREFIX = "gear_sonic.dataset_generation.hallucination.motion2scene_reactive_execution"
CONDITIONS = (
    ("absent", "reactive"),
    ("present", "reactive"),
    ("raised", "reactive"),
    ("present", "blind"),
    ("absent", "oracle"),
    ("present", "oracle"),
)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    prior_path = PRIOR / "paired_manifest.json"
    middle_path = DATA / "m2s-d040-beam-execution-v1/manifest.json"
    prior = json.loads(prior_path.read_text())
    middle = json.loads(middle_path.read_text())["cells"][0]
    alternate = middle["motion"]
    base = next(c for c in prior["cells"] if c["label"] == "neutral")
    # Verify equal timing and shared route before simulator spend.
    import joblib

    a = next(iter(joblib.load(base["motion"]["path"]).values()))
    b = next(iter(joblib.load(alternate["path"]).values()))
    if a["fps"] != b["fps"] or len(a["dof"]) != len(b["dof"]):
        raise ValueError("unequal reference clocks")
    if not np.allclose(a["root_trans_offset"][:, :2], b["root_trans_offset"][:, :2], atol=1e-7):
        raise ValueError("references do not share a route")
    high = scene_file(out / "beam_raised.usda", {**prior["beam"], "underside_m": 2.0})
    cells = []
    for seed in (8021, 8022):
        for condition, mode in CONDITIONS:
            cell = copy.deepcopy(base)
            cell.update(
                cell_id=f"interface_{seed}_{condition}_{mode}",
                runtime_seed=seed,
                condition=condition,
                mode=mode,
            )
            cell["output"] = str(out / "rollouts" / cell["cell_id"])
            cell["scene"] = (
                high
                if condition == "raised"
                else {
                    **artifact(PRIOR / f"beam_{condition}.usda"),
                    "scene_id": f"beam_{condition}",
                }
            )
            filters = next(s for s in base["hydra_overrides"] if "beam_contact_body_names" in s)
            cell["hydra_overrides"] = [
                f"++seed={seed}",
                f"++manager_env._target_={PREFIX}.ReactiveExecutionEnvCfg",
                f"++manager_env.recorders.trajectory._target_={PREFIX}.ReactiveRecorderCfg",
                filters,
                f"++manager_env.config.reactive_alternate_path={alternate['path']}",
                f"++manager_env.config.reactive_mode={mode}",
            ]
            cells.append(cell)
    manifest = copy.deepcopy(prior)
    manifest.update(
        experiment="M2S-reactive-interface-v1",
        purpose="Beam-visible PhysX rays and phase-aligned online skill switching",
        cells=cells,
        alternate_motion=alternate,
        registered_predictions=artifact(ROOT / "docs/motion2scene/REACTIVE_INTERFACE_V1.md"),
        new_dependencies=prior["new_dependencies"]
        + [
            artifact(Path(__file__)),
            artifact(prior_path),
            artifact(middle_path),
            artifact(Path(alternate["path"])),
            artifact(Path(alternate["conversion_provenance"])),
            *[
                artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{name}.py")
                for name in ("motion2scene_reactive_execution", "motion2scene_reactive_rule")
            ],
            *[
                artifact(
                    Path("/home/linjiw/isaaclab-install/IsaacLab/source/isaaclab/isaaclab") / name
                )
                for name in (
                    "envs/manager_based_rl_env.py",
                    "sensors/contact_sensor/contact_sensor.py",
                )
            ],
            artifact(ROOT / "gear_sonic/envs/manager_env/mdp/commands.py"),
        ],
    )
    manifest["execution_policy"]["runtime"]["free_gpu_mib_required"] = 9000
    manifest["execution_policy"]["cost_ceiling"].update(rollouts=12, gpu_hours_contended=1.25)
    manifest["execution_policy"]["timing_override"] = "User requests next-stage Isaac simulations."
    manifest["stop_conditions"][0] = "refuse hash mismatches; yield below 9000 MiB free GPU"
    write_new(out / "manifest.json", manifest)


def physics_windows(physics, sampled):
    steps = physics["physics_steps"]
    controls = physics["control_steps"]
    force = physics["force_w"]
    if len(steps) != len(force) or len(controls) != len(sampled) or not np.all(np.diff(steps) == 1):
        raise ValueError("nonconsecutive or mismatched force capture")
    indices = np.searchsorted(steps, controls[:, None] - np.arange(3, -1, -1))
    expected = controls[:, None] - np.arange(3, -1, -1)
    if np.any(indices >= len(steps)) or not np.array_equal(steps[indices], expected):
        raise ValueError("missing substeps")
    if not np.all(np.diff(controls) == 4) or float(physics["physics_dt"]) != 0.005:
        raise ValueError("wrong physics cadence")
    blocks = force[indices]
    error = float(np.max(np.abs(blocks[:, -1] - sampled)))
    if error > 1e-5:
        raise ValueError(f"cached/direct contact mismatch: {error}")
    norms = np.linalg.norm(blocks, axis=-1)
    maxima = np.argmax(norms, axis=1)
    aggregate = np.take_along_axis(blocks, maxima[:, None, :, None], axis=1)[:, 0]
    return blocks, aggregate, error


def analyze(out):
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    record = json.loads((out / "run_record.json").read_text())
    if record["status"] != "completed" or record["manifest_sha256"] != artifact(path)["sha256"]:
        raise ValueError("complete hash-bound batch required")
    rows = []
    for cell in manifest["cells"]:
        # The older reader names every enabled beam "present". Validate a raised
        # beam under that collision flag, then restore this pilot's condition.
        reader_cell = {
            **cell,
            "condition": "present" if cell["condition"] == "raised" else cell["condition"],
        }
        payload, sampled, row = read_cell(reader_cell, record)
        row["condition"] = cell["condition"]
        folder = Path(cell["output"]) / "trajectories"
        sensor_path = folder / "reactive_interface.json"
        physics_path = folder / "physics_beam_contacts.npz"
        sensor = json.loads(sensor_path.read_text())
        with np.load(physics_path) as physics:
            blocks, aggregate, error = physics_windows(physics, sampled)
        score = score_passage(payload, aggregate, manifest["beam"])
        horizon = score["passage_finish_frame_exclusive"] or score["first_episode_frames"]
        norms = np.linalg.norm(blocks[:horizon].reshape(-1, 30, 3), axis=-1)
        switches = sensor["switches"]
        invariant = all(
            all(s[k] for k in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged"))
            for s in switches
        )
        expected_switch = cell["mode"] == "oracle" or (
            cell["mode"] == "reactive" and cell["condition"] == "present"
        )
        switch_pass = invariant and (
            len(switches) == 0
            if not expected_switch
            else (
                len(switches) == 2
                and switches[0]["from"] == 0
                and switches[0]["to"] == 1
                and 0.2 <= switches[0]["time_s"] < 1.0
                and switches[1]["from"] == 1
                and switches[1]["to"] == 0
                and switches[1]["time_s"] == 3.3
            )
        )
        row.update(
            score,
            mode=cell["mode"],
            seed=cell["runtime_seed"],
            switches=switches,
            switch_contract_pass=switch_pass,
            sensor=artifact(sensor_path),
            physics_contacts=artifact(physics_path),
            contact_sync_max_error_n=error,
            physics_contact_duration_s=float((norms.max(1) > 1).sum() * 0.005),
            sum_body_normal_magnitude_impulse_ns=float(norms.sum() * 0.005),
            occupied_observations=sum(o["occupied"] for o in sensor["observations"]),
        )
        rows.append(row)
    write_new(
        out / "result.json",
        {
            "manifest": artifact(path),
            "run_record": artifact(out / "run_record.json"),
            "rows": rows,
            "predictions": {
                "p1": all(r["switch_contract_pass"] for r in rows if r["mode"] == "reactive"),
                "p2": all(
                    (
                        r["observed_beam_contact"]
                        if r["mode"] == "blind"
                        else r["pass"] and not r["fall_observed"] and r["reset_count"] == 0
                    )
                    for r in rows
                ),
                "p3": all(r["contact_sync_max_error_n"] <= 1e-5 for r in rows)
                and all(r["observed_beam_contact"] for r in rows if r["mode"] == "blind")
                and all(not r["observed_beam_contact"] for r in rows if r["condition"] == "absent"),
            },
            "new_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "scope": (
                "single development carrier; scripted sparse collision lidar; "
                "phase-aligned switching; no policy training"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
        return
    manifest = json.loads((args.out / "manifest.json").read_text())
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    if args.command == "analyze":
        analyze(args.out)
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(args.out / "run_record.json"),
    ]
    if args.command == "preflight":
        command.append("--dry-run")
    subprocess.run(command, check=True)
    if args.command == "run":
        analyze(args.out)


if __name__ == "__main__":
    main()
