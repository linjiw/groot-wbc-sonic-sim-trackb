#!/usr/bin/env python3
"""Freeze and run the overhang observer / guarded-switch development panel."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-reactive-interface-typed-v3"
PREFIX = "gear_sonic.dataset_generation.hallucination.motion2scene_overhang_execution"


def prepare(out):
    prior = json.loads((PRIOR / "manifest.json").read_text())
    result = json.loads((PRIOR / "result.json").read_text())
    if result["predictions"] != {"p1": False, "p2": True, "p3": True}:
        raise ValueError("unexpected prior result; revisit scientific design")
    manifest = copy.deepcopy(prior)
    cells = []
    for seed in (8031, 8032):
        templates = [c for c in prior["cells"] if c["runtime_seed"] == 8021]
        late = copy.deepcopy(
            next(c for c in templates if c["condition"] == "present" and c["mode"] == "oracle")
        )
        late["mode"] = "late_oracle"
        for base in [*templates, late]:
            cell = copy.deepcopy(base)
            cell.update(
                runtime_seed=seed, cell_id=f"overhang_{seed}_{cell['condition']}_{cell['mode']}"
            )
            cell["output"] = str(out / "rollouts" / cell["cell_id"])
            filters = next(s for s in base["hydra_overrides"] if "beam_contact_body_names" in s)
            cell["hydra_overrides"] = [
                f"++seed={seed}",
                f"++manager_env._target_={PREFIX}.OverhangExecutionEnvCfg",
                f"++manager_env.recorders.trajectory._target_={PREFIX}.OverhangRecorderCfg",
                filters,
                f"++manager_env.config.reactive_alternate_path={prior['alternate_motion']['path']}",
                f"++manager_env.config.reactive_mode={cell['mode']}",
            ]
            cells.append(cell)
    manifest.update(
        experiment="M2S-overhang-guard-interface-v1",
        cells=cells,
        purpose="Separate overhangs from walls and refuse illegal reference switches",
        registered_predictions=artifact(ROOT / "docs/motion2scene/OVERHANG_INTERFACE_V1.md"),
    )
    manifest["new_dependencies"] += [
        artifact(Path(__file__)),
        artifact(PRIOR / "result.json"),
        artifact(PRIOR / "manifest.json"),
        artifact(PRIOR / "run_record.json"),
        artifact(PRIOR / "false_positive_audit.json"),
        artifact(DATA / "m2s-reactive-orphan-cleanup-v1.json"),
        *[
            artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{name}.py")
            for name in ("motion2scene_overhang_observer", "motion2scene_overhang_execution")
        ],
    ]
    manifest["execution_policy"]["cost_ceiling"].update(rollouts=14, gpu_hours_contended=1.45834)
    manifest["execution_policy"][
        "timing_override"
    ] = "User requests next research stage after failed wall specificity."
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", manifest)


def analyze(out):
    path = out / "manifest.json"
    manifest, record = json.loads(path.read_text()), json.loads(
        (out / "run_record.json").read_text()
    )
    if record["status"] != "completed" or record["manifest_sha256"] != artifact(path)["sha256"]:
        raise ValueError("requires complete hash-bound panel")
    rows = []
    for cell in manifest["cells"]:
        reader = {
            **cell,
            "condition": "present" if cell["condition"] == "raised" else cell["condition"],
        }
        payload, sampled, row = read_cell(reader, record)
        folder = Path(cell["output"]) / "trajectories"
        sensor_path, physics_path = (
            folder / "reactive_interface.json",
            folder / "physics_beam_contacts.npz",
        )
        sensor = json.loads(sensor_path.read_text())
        with np.load(physics_path) as physics:
            blocks, aggregate, error = physics_windows(physics, sampled)
        if blocks.shape != (199, 4, 30, 3):
            raise ValueError("incomplete capture")
        verdict = score_passage(payload, aggregate, manifest["beam"])
        obs, switches = sensor["observations"], sensor["switches"]
        if len(obs) != 199:
            raise ValueError("incomplete observation log")
        invariants = all(
            all(s[k] for k in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged"))
            for s in switches
        )
        invariants = invariants and all(
            all(
                o["transition"][k]
                for k in ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
            )
            for o in obs
        )
        legal = invariants and all(
            s["joint_reference_jump_rad"] <= 0.05
            and s["root_reference_jump_m"] <= 0.01
            and (0.2 <= s["time_s"] <= 0.4 if s["to"] == 1 else 3.3 <= s["time_s"] <= 3.5)
            for s in switches
        )
        early_pair = (
            len(switches) == 2
            and switches[0]["from"] == 0
            and switches[0]["to"] == 1
            and 0.2 <= switches[0]["time_s"] <= 0.4
            and switches[1]["from"] == 1
            and switches[1]["to"] == 0
            and switches[1]["time_s"] == 3.3
        )
        occupied = [o for o in obs if o["occupied"]]
        denied = [o for o in obs if o["transition"]["attempted"] and not o["transition"]["allowed"]]
        lower_blocked = sum(
            r["upper_candidate"] and not r["overhang"] for o in obs for r in o["rays"]
        )
        horizon = verdict["passage_finish_frame_exclusive"] or verdict["first_episode_frames"]
        norms = np.linalg.norm(blocks[:horizon].reshape(-1, 30, 3), axis=-1)
        row.update(
            verdict,
            condition=cell["condition"],
            mode=cell["mode"],
            seed=cell["runtime_seed"],
            sensor=artifact(sensor_path),
            physics_contacts=artifact(physics_path),
            switches=switches,
            raw_overhang_frames=len(occupied),
            raw_upper_frames=sum(o["upper_occupied"] for o in obs),
            first_overhang_time_s=occupied[0]["time_s"] if occupied else None,
            blocked_upper_candidates=int(lower_blocked),
            denied_transition_frames=len(denied),
            denied_reasons=sorted(
                {reason for o in denied for reason in o["transition"]["reasons"]}
            ),
            legal_switch_contract=legal,
            expected_early_pair=early_pair,
            contact_sync_max_error_n=error,
            physics_contact_duration_s=float((norms.max(1) > 1).sum() * 0.005),
            sum_body_normal_magnitude_impulse_ns=float(norms.sum() * 0.005),
        )
        rows.append(row)
    negatives = [r for r in rows if r["mode"] == "reactive" and r["condition"] != "present"]
    reactive = [r for r in rows if r["mode"] == "reactive" and r["condition"] == "present"]
    positives = [r for r in rows if r["mode"] in ("reactive", "oracle")]
    blind = [r for r in rows if r["mode"] == "blind"]
    late = [r for r in rows if r["mode"] == "late_oracle"]
    if list(map(len, (rows, negatives, reactive, positives, blind, late))) != [14, 4, 2, 10, 2, 2]:
        raise ValueError("wrong full experimental denominator")
    write_new(
        out / "result.json",
        {
            "manifest": artifact(path),
            "run_record": artifact(out / "run_record.json"),
            "rows": rows,
            "predictions": {
                "p1": all(
                    r["raw_overhang_frames"] == 0
                    and not r["switches"]
                    and r["blocked_upper_candidates"] > 0
                    for r in negatives
                )
                and all(
                    r["first_overhang_time_s"] is not None
                    and r["first_overhang_time_s"] < 0.4
                    and r["expected_early_pair"]
                    for r in reactive
                ),
                "p2": all(
                    r["pass"] and not r["fall_observed"] and r["reset_count"] == 0
                    for r in positives
                )
                and all(r["observed_beam_contact"] for r in blind),
                "p3": all(r["legal_switch_contract"] for r in rows)
                and all(r["denied_transition_frames"] > 0 and not r["switches"] for r in late),
                "p4": all(r["contact_sync_max_error_n"] <= 1e-5 for r in rows)
                and all(r["observed_beam_contact"] for r in blind)
                and all(not r["observed_beam_contact"] for r in rows if r["condition"] == "absent"),
            },
            "new_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "prior_orphan_reservation_supplement": artifact(
                DATA / "m2s-reactive-orphan-cleanup-v1.json"
            ),
            "scope": (
                "one observed carrier; new physics seeds; ideal sparse lidar and guarded "
                "reference selector, no training"
            ),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.out)
        return
    manifest = json.loads((args.out / "manifest.json").read_text())
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    if args.command == "analyze":
        analyze(args.out)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(args.out / "run_record.json"),
    ]
    if args.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if args.command == "run":
        analyze(args.out)


if __name__ == "__main__":
    main()
