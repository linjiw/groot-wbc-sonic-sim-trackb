#!/usr/bin/env python3
"""Freeze and execute independent layout blocks without changing fitted policies."""

import argparse
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys

from bundle_motion2scene_sources import closure
from motion2scene_check_learned_commands import CORPUS, DATA, RUNTIME
from motion2scene_comparative_acquisition import ARMS, PRIOR, reference_pair
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import beam_at, read_cell, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

PROTOCOL = ROOT / "docs/motion2scene/INDEPENDENT_LAYOUT_EXECUTION_V1.md"
INTEGRATION = DATA / "m2s-learned-command-check-v1"
AUDITOR = ROOT / "scripts/research/audit_motion2scene_comparison_capture.py"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"


def prepare(out):
    fits = json.loads((CORPUS / "fits/result.json").read_text())
    integration = json.loads((INTEGRATION / "result.json").read_text())
    assert fits["fits"] == 20 and all(integration["predictions"].values())
    contract_path = DATA / "m2s-learning-contract-v2/registration.json"
    contract = json.loads(contract_path.read_text())
    assert contract["development_bank"]["decision_time_s"] == 0.3
    parent = json.loads((INTEGRATION / "manifest.json").read_text())
    case, _, binding = reference_pair(json.loads((PRIOR / "manifest.json").read_text()))
    old = json.loads((PRIOR / "result.json").read_text())
    bank = old["rows"][0]["bank"]
    refs = [
        artifact(PROTOCOL),
        artifact(contract_path),
        artifact(CORPUS / "fits/result.json"),
        artifact(INTEGRATION / "result.json"),
        artifact(INTEGRATION / "manifest.json"),
        bank,
    ]
    refs += [r["checkpoint"] for r in fits["rows"]]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME, AUDITOR]))]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    out.mkdir(exist_ok=False)
    (out / "scenes").mkdir()
    specs = []
    for layout in contract["independent_common_bank_layouts"]:
        beam = {**beam_at(case, layout["station"], layout["underside_m"]), "thickness_m": 0.1}
        assert 0.1 <= beam["route_progress"] <= 0.9 and 1.1 <= beam["underside_m"] <= 1.45
        specs.append(
            {
                **layout,
                "condition": "generated",
                "beam": beam,
                "scene": scene_file(out / "scenes" / f"{layout['id']}.usda", beam),
            }
        )
    corpus_manifest = json.loads((CORPUS / "analysis_union/manifest.json").read_text())
    for name in ("absent", "raised", "blocked"):
        c = next(c for c in corpus_manifest["cells"] if c["group_id"] == f"shared_{name}")
        specs.append(
            {
                "id": f"control_{name}",
                "suite": name,
                "condition": name,
                "beam": c["beam"],
                "scene": c["scene"],
            }
        )
    ordered_fits = sorted(fits["rows"], key=lambda r: (ARMS.index(r["arm"]), r["seed"]))
    blocks = []
    for spec in specs:
        for seed in contract["layout_physics_seeds"]:
            blocks.append(
                {
                    "index": len(blocks),
                    "spec": spec,
                    "physics_seed": seed,
                    "directory": str(out.with_name(out.name + f"-b{len(blocks):02d}")),
                }
            )
    assert len(blocks) == 30 and len(ordered_fits) == 20
    write_new(
        out / "master.json",
        {
            "role": "nonexecutable assignment, not an outcome record",
            "execution_policy": {"not_authorized": True},
            "references": refs,
            "reference_binding": binding,
            "blocks": blocks,
            "fits": ordered_fits,
            "assigned_cells": 600,
            "traversal_cells": 480,
            "control_cells": 120,
            "initial_wave_blocks": list(range(6)),
            "initial_wave_cells": 120,
        },
    )
    prepared = []
    for block in blocks:
        folder = Path(block["directory"])
        folder.mkdir(exist_ok=False)
        cells = []
        for fit in ordered_fits:
            c = copy.deepcopy(parent["cells"][0])
            for k in ("expected_readout", "paired_outcome"):
                c.pop(k)
            spec = block["spec"]
            cell_id = f"{spec['id']}_p{block['physics_seed']}_{fit['arm']}_s{fit['seed']}"
            c.update(
                cell_id=cell_id,
                group_id=spec["id"],
                condition=spec["condition"],
                suite=spec["suite"],
                beam=spec["beam"],
                scene=spec["scene"],
                policy=fit["checkpoint"],
                policy_arm=fit["arm"],
                optimizer_seed=fit["seed"],
                runtime_seed=block["physics_seed"],
                output=str(folder / "rollouts" / cell_id),
                role="independent_layout_evaluation",
            )
            c["hydra_overrides"] = [
                s
                for s in c["hydra_overrides"]
                if not any(
                    k in s for k in ("++seed=", "learned_policy_path=", "learned_policy_sha256=")
                )
            ]
            c["hydra_overrides"] += [
                f"++seed={block['physics_seed']}",
                f"++manager_env.config.learned_policy_path={fit['checkpoint']['path']}",
                f"++manager_env.config.learned_policy_sha256={fit['checkpoint']['sha256']}",
            ]
            cells.append(c)
        m = copy.deepcopy(parent)
        m.update(
            experiment=f"M2S-independent-layout-v1-b{block['index']:02d}",
            cells=cells,
            purpose="Independent layout test of fixed development learners",
            registered_predictions=artifact(PROTOCOL),
        )
        m["new_dependencies"] += refs + [artifact(out / "master.json")]
        m["execution_policy"]["cost_ceiling"].update(
            rollouts=20, gpu_hours_contended=20 * 375 / 3600
        )
        m["execution_policy"][
            "timing_override"
        ] = "Registered independent-layout block; cumulative budget gate."
        write_new(folder / "manifest.json", m)
        write_new(
            folder / "capture_audit_registration.json",
            {
                "manifest": artifact(folder / "manifest.json"),
                "bank": bank,
                "implementation": artifact(AUDITOR),
                "scope": "first 0.30 s callback on independent layouts",
                "predictions": ["features exact", "origin 1e-6", "state 1e-7", "bank exact"],
            },
        )
        prepared.append(
            {
                "manifest": artifact(folder / "manifest.json"),
                "audit_registration": artifact(folder / "capture_audit_registration.json"),
            }
        )
    write_new(out / "prepared.json", {"master": artifact(out / "master.json"), "blocks": prepared})
    print(json.dumps({"blocks": 30, "assigned_cells": 600, "initial_wave_cells": 120}))


def budget_gate(folder, remaining):
    now = datetime.datetime.now(datetime.timezone.utc)
    rows = []
    for p in DATA.glob("m2s-*/*run_record*.json"):
        r = json.loads(p.read_text())
        hours = float(r.get("budget", {}).get("actual_contended_gpu_hours", 0))
        if hours:
            assert not r.get("analysis_only", False)
            age = (now - datetime.datetime.fromisoformat(r["started_at"])).total_seconds() / 86400
            rows.append({"path": str(p), "hours": hours, "age_days": age})
    day = sum(r["hours"] for r in rows if r["age_days"] <= 1)
    week = sum(r["hours"] for r in rows if r["age_days"] <= 7)
    reserve = remaining * 375 / 3600
    gate = {
        "records": rows,
        "daily_hours": day,
        "weekly_hours": week,
        "reserved_hours": reserve,
        "admitted": day + reserve <= 8 and week + reserve <= 24,
        "checked_at": now.isoformat(),
    }
    path = folder / f'launch_gate_{len(list(folder.glob("launch_gate_*.json"))):03d}.json'
    write_new(path, gate)
    return gate["admitted"]


def analyze(folder):
    m = json.loads((folder / "manifest.json").read_text())
    record = json.loads((folder / "run_record.json").read_text())
    assert record["status"] == "completed"
    assert record["manifest_sha256"] == artifact(folder / "manifest.json")["sha256"]
    rows, baseline, features = [], None, None
    for c in m["cells"]:
        _, sampled, r = read_cell(
            {**c, "condition": "absent" if c["condition"] == "absent" else "present"}, record
        )
        p = load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        path = Path(c["output"]) / "trajectories"
        d = json.loads((path / "decision_capture.json").read_text())
        sensors = json.loads((path / "reactive_interface.json").read_text())
        with np.load(path / "physics_beam_contacts.npz") as f:
            _, forces, sync = physics_windows(f, sampled)
        if baseline is None:
            baseline, features = p, d["features"]
        prefix = paired_prefix(baseline, p, 0.3)
        prediction = readout(
            load_readout(c["policy"]["path"], c["policy"]["sha256"]), d["features"]
        )
        error = float(
            np.max(
                abs(np.array(prediction["probabilities"]) - d["learned_decision"]["probabilities"])
            )
        )
        model_match = (
            error <= 1e-6
            and d["policy_sha256"] == c["policy"]["sha256"]
            and all(
                prediction[k] == d["learned_decision"][k]
                for k in ("selected_action", "requested_action", "refusal")
            )
        )
        beam = c["beam"]
        matrix = np.array(r["imported_beam"]["local_to_world_at_capture_start"])
        yaw = beam["yaw_rad"]
        rotation = np.array(
            [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        position_ok = np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
            rtol=0,
        )
        scale_ok = np.allclose(
            matrix[:3, :3],
            np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])[:, None] * rotation,
            atol=1e-6,
            rtol=0,
        )
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(
            all(o["transition"][k] for k in flags) for o in sensors["observations"]
        ) and all(
            all(s[k] for k in flags)
            and s["joint_reference_jump_rad"] <= 0.05
            and s["root_reference_jump_m"] <= 0.01
            and (0.2 <= s["time_s"] <= 0.4 if s["to"] else 3.3 <= s["time_s"] <= 3.5)
            for s in sensors["switches"]
        )
        rows.append(
            {
                **r,
                **score_passage(p, forces, beam),
                "group_id": c["group_id"],
                "suite": c["suite"],
                "arm": c["policy_arm"],
                "optimizer_seed": c["optimizer_seed"],
                "physics_seed": c["runtime_seed"],
                "beam": beam,
                "readout": d["learned_decision"],
                "prefix": prefix,
                "probability_error": error,
                "contact_sync_error_n": sync,
                "entry_logged": any(
                    s["to"] == 1 and s["time_s"] == 0.3 for s in sensors["switches"]
                ),
                "return_logged": any(s["to"] == 0 for s in sensors["switches"]),
                "denied_requests": sum(
                    o["transition"]["attempted"] and not o["transition"]["allowed"]
                    for o in sensors["observations"]
                ),
                "predictions": {
                    "matched_inputs": prefix["exact_match"] and d["features"] == features,
                    "model_request_matches": model_match
                    and d["requested_action"] == prediction["requested_action"],
                    "legal_switches": legal,
                    "scene_matches": bool(position_ok and scale_ok),
                },
                "decision": artifact(path / "decision_capture.json"),
                "sensor": artifact(path / "reactive_interface.json"),
                "physics": artifact(path / "physics_beam_contacts.npz"),
                "bank": artifact(path / "loaded_reference_bank.npz"),
            }
        )
    predictions = {k: all(r["predictions"][k] for r in rows) for k in rows[0]["predictions"]}
    write_new(
        folder / "result.json",
        {
            "manifest": artifact(folder / "manifest.json"),
            "run_record": artifact(folder / "run_record.json"),
            "rows": rows,
            "predictions": predictions,
            "scope": "one block of assigned independent layouts, one observed source",
            "actual_contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
        },
    )
    subprocess.run([sys.executable, str(AUDITOR), "analyze", "--out", str(folder)], check=True)
    audit = json.loads((folder / "capture_audit.json").read_text())
    admitted = all(predictions.values()) and all(audit["predictions"].values())
    write_new(
        folder / "admission.json",
        {
            "result": artifact(folder / "result.json"),
            "capture_audit": artifact(folder / "capture_audit.json"),
            "admitted": admitted,
        },
    )
    print(
        json.dumps(
            {
                "block": folder.name,
                "admitted": admitted,
                "passes_by_arm": {
                    arm: sum(r["pass"] for r in rows if r["arm"] == arm) for arm in ARMS
                },
            }
        ),
        flush=True,
    )
    return admitted


def run(out, count, dry=False):
    prepared = json.loads((out / "prepared.json").read_text())
    master = json.loads(
        checked(Path(prepared["master"]["path"]), prepared["master"]["sha256"]).read_text()
    )
    started = 0
    for block, refs in zip(master["blocks"], prepared["blocks"]):
        folder = Path(block["directory"])
        for ref in refs.values():
            checked(Path(ref["path"]), ref["sha256"])
        if (folder / "admission.json").exists():
            assert json.loads((folder / "admission.json").read_text())["admitted"]
            continue
        if started >= count:
            break
        record_path = folder / "run_record.json"
        old = json.loads(record_path.read_text()) if record_path.exists() else {"cells": {}}
        if any(r["status"] not in ("completed", "not_started") for r in old["cells"].values()):
            raise RuntimeError(
                "previous started/failed cells require reconciliation; no automatic retry"
            )
        remaining = 20 - sum(r["status"] == "completed" for r in old["cells"].values())
        if not dry and not budget_gate(folder, remaining):
            print("BUDGET YIELD: retain the remaining frozen assignment", flush=True)
            return
        cmd = [
            sys.executable,
            str(DRIVER),
            "--manifest",
            str(folder / "manifest.json"),
            "--run-record",
            str(record_path),
        ]
        if dry:
            cmd.append("--dry-run")
        completed = subprocess.run(cmd)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        if not dry:
            record = json.loads(record_path.read_text())
            if record["status"] != "completed":
                print(f"RESOURCE YIELD: {record['status']}", flush=True)
                return
            if not analyze(folder):
                raise RuntimeError(
                    "input/runtime/measurement admission failed; stop subsequent blocks"
                )
        started += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-blocks", type=int, default=6)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out)
    elif a.command == "analyze":
        analyze(a.out)
    else:
        assert 1 <= a.max_blocks <= 30
        run(a.out, a.max_blocks, dry=a.command == "preflight")


if __name__ == "__main__":
    main()
