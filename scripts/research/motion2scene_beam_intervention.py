#!/usr/bin/env python3
"""Prepare, preflight and run the registered paired beam intervention when GPU permits."""

import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage

DATA = ROOT.parent / "research-data/groot-wbc"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    template_path = DATA / "m2s-repeatability-v1/manifest.json"
    template = json.loads(template_path.read_text())
    evidence_path = DATA / "m2s-beam-teacher-v1/result.json"
    evidence = json.loads(evidence_path.read_text())
    beam = evidence["selected_candidate"]
    proposal = next(p for p in evidence["proposals"] if p["quantile"] == 0.5)
    if beam["candidate_id"] != "beam_021" or not proposal["jitter_audit"]["all_passed"]:
        raise ValueError("registered beam evidence changed")
    result_path = DATA / "m2s-repeatability-v1/result.json"
    previous = json.loads(result_path.read_text())
    trajectory = previous["rows"][0]["trajectory"]
    with checked(Path(trajectory["path"]), trajectory["sha256"]).open("rb") as handle:
        names = list(pickle.load(handle)["body_names"])
    base_path = ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
    base_scene = base_path.read_text().rstrip()
    scenes = {}
    for condition, height, enabled in (
        ("clear", 2.0, True),
        ("intersecting", 1.1, True),
        ("present", proposal["beam_underside_m"], True),
        ("absent", proposal["beam_underside_m"], False),
    ):
        path = out / f"beam_{condition}.usda"
        text = f"""
    def Cube "CounterfactualBeam" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI",
                              "PhysxCollisionAPI", "PhysxContactReportAPI"]
    ) {{
        double size = 1
        bool physics:collisionEnabled = {str(enabled).lower()}
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = true
        float physxCollision:contactOffset = 0.002
        float physxCollision:restOffset = 0
        float physxContactReport:threshold = 0
        token visibility = "{"inherited" if enabled else "invisible"}"
        double3 xformOp:translate = ({beam["center_xy_m"][0]}, {beam["center_xy_m"][1]}, {height + 0.05})
        double xformOp:rotateZ = {math.degrees(beam["yaw_rad"])}
        double3 xformOp:scale = (0.1, 1.2, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.65, 0.37, 0.15)]
    }}
}}
"""
        path.write_text(base_scene[:-1] + text)
        scenes[condition] = {
            "path": str(path),
            "scene_id": path.stem,
            "sha256": artifact(path)["sha256"],
        }
    templates = {c["label"]: c for c in template["cells"] if c["runtime_seed"] == 7901}
    dependencies = [
        Path(__file__),
        ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_beam_execution.py",
        ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_passage.py",
        ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_collision_inventory.py",
        ROOT / "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
        ROOT / "gear_sonic/envs/manager_env/mdp/recorders.py",
    ]
    for batch, design in (
        ("controls", [(7910, "neutral", c) for c in ("clear", "intersecting")]),
        (
            "paired",
            [
                (seed, label, condition)
                for seed in (7911, 7912, 7913)
                for condition in ("absent", "present")
                for label in ("neutral", "d055")
            ],
        ),
    ):
        manifest = copy.deepcopy(template)
        cells = []
        for seed, label, condition in design:
            cell = copy.deepcopy(templates[label])
            cell.pop("depends_on_acceptance_of", None)
            cell_id = f"beam_{seed}_{condition}_{label}"
            cell.update(
                cell_id=cell_id,
                runtime_seed=seed,
                condition=condition,
                scene=scenes[condition],
                output=str(out / "rollouts" / cell_id),
                hydra_overrides=[
                    f"++seed={seed}",
                    "++manager_env._target_=gear_sonic.dataset_generation.hallucination.motion2scene_beam_execution.BeamExecutionEnvCfg",
                    "++manager_env.recorders.trajectory._target_=gear_sonic.dataset_generation.hallucination.motion2scene_beam_execution.BeamTrajectoryRecorderCfg",
                    "++manager_env.config.beam_contact_body_names=[" + ",".join(names) + "]",
                ],
            )
            cells.append(cell)
        manifest.update(
            experiment=f"M2S-counterfactual-beam-v1-{batch}",
            cells=cells,
            analysis_role="selected_pair_analytic_beam_development_intervention",
            registered_predictions=artifact(ROOT / "docs/motion2scene/COUNTERFACTUAL_STAGE_V1.md"),
            new_dependencies=[artifact(p) for p in dependencies],
            beam_evidence=artifact(evidence_path),
            empty_scene=artifact(base_path),
            source_template=artifact(template_path),
            beam=beam,
            beam_underside_m=proposal["beam_underside_m"],
        )
        manifest["execution_policy"]["cost_ceiling"].update(
            rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
        )
        manifest["execution_policy"][
            "timing_override"
        ] = "User-authorized counterfactual stage; serial development only."
        manifest["authorization"][
            "note"
        ] = "User explicitly requested next-stage research and work toward the supplied beam intervention."
        write_new(out / f"{batch}_manifest.json", manifest)
    print(f"Prepared two controls and twelve paired cells at {out}")


def analyze(out, batch):
    manifest_path = out / f"{batch}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record_path = out / f"{batch}_run_record.json"
    record = json.loads(record_path.read_text())
    if (
        record["status"] != "completed"
        or record["manifest_sha256"] != artifact(manifest_path)["sha256"]
    ):
        raise ValueError("complete matching run required")
    rows = []
    for cell in manifest["cells"]:
        scientific = record["cells"][cell["cell_id"]]["scientific"]
        paths = scientific["artifacts"]
        with checked(Path(paths["trajectory"]), paths["trajectory_sha256"]).open("rb") as handle:
            payload = pickle.load(handle)
        contact_path = Path(cell["output"]) / "trajectories/beam_contacts.npz"
        inventory_path = contact_path.with_name("native_collision_inventory.json")
        contact = np.load(contact_path, allow_pickle=False)
        if float(contact["fps"]) != float(payload["fps"]):
            raise ValueError("contact/trajectory cadence mismatch")
        inventory = json.loads(inventory_path.read_text())
        if list(contact["filter_paths"]) != inventory["filter_paths"]:
            raise ValueError("force filters do not match native inventory")
        scores = score_passage(payload, contact["force_w"], manifest["beam"])
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "seed": cell["runtime_seed"],
                "condition": cell["condition"],
                "label": cell["label"],
                "tracker_outcome": scientific["outcome"],
                **scores,
                "contacts": artifact(contact_path),
                "inventory": artifact(inventory_path),
            }
        )
    result = {
        "manifest": artifact(manifest_path),
        "run_record": artifact(record_path),
        "rows": rows,
    }
    if batch == "controls":
        by_condition = {r["condition"]: r for r in rows}
        result["controls_pass"] = (
            not by_condition["clear"]["observed_beam_contact"]
            and by_condition["intersecting"]["observed_beam_contact"]
        )
    else:
        rates = {
            f"{label}_{condition}": sum(
                r["pass"] for r in rows if r["label"] == label and r["condition"] == condition
            )
            / 3
            for label in ("neutral", "d055")
            for condition in ("absent", "present")
        }
        result["rates"] = rates
        result["counterfactual_interaction"] = (
            rates["d055_present"]
            - rates["neutral_present"]
            - rates["d055_absent"]
            + rates["neutral_absent"]
        )
        result["strict_development_separation"] = rates == {
            "neutral_absent": 1,
            "neutral_present": 0,
            "d055_absent": 1,
            "d055_present": 1,
        } and all(
            r["observed_beam_contact"]
            for r in rows
            if r["label"] == "neutral" and r["condition"] == "present"
        )
    write_new(out / f"{batch}_result.json", result)
    return result


def run(out, dry_run):
    for batch in ("controls", "paired"):
        manifest_path = out / f"{batch}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for ref in [
            *manifest["new_dependencies"],
            manifest["beam_evidence"],
            manifest["empty_scene"],
            manifest["source_template"],
        ]:
            checked(Path(ref["path"]), ref["sha256"])
        if batch == "paired" and not dry_run:
            controls = json.loads((out / "controls_result.json").read_text())
            if not controls["controls_pass"]:
                raise RuntimeError("contact controls failed; paired intervention not launched")
        command = [
            sys.executable,
            str(DRIVER),
            "--manifest",
            str(manifest_path),
            "--run-record",
            str(out / f"{batch}_run_record.json"),
        ]
        if dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
        if not dry_run and not (out / f"{batch}_result.json").exists():
            analyze(out, batch)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    else:
        raise SystemExit(run(args.out, args.command == "preflight"))


if __name__ == "__main__":
    main()
