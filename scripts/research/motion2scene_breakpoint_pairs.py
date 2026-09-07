#!/usr/bin/env python3
"""Twelve matched explicit commands on the first-wave development conditions."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from bundle_motion2scene_sources import closure
from motion2scene_layout_budget_review import review
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    pair_decisions,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)

DATA = ROOT.parent / "research-data/groot-wbc"
ORIGINAL = DATA / "m2s-independent-layout-v1"
PROTOCOL = ROOT / "docs/motion2scene/SELECTOR_BREAKPOINT_STUDY_V1.md"
AUDITOR = ROOT / "scripts/research/audit_motion2scene_comparison_capture.py"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_comparison_execution.py"


def prepare(out):
    parent_path = DATA / "m2s-comparative-acquisition-v1/manifest.json"
    parent = json.loads(parent_path.read_text())
    master = json.loads((ORIGINAL / "master.json").read_text())
    out.mkdir(exist_ok=False)
    refs = [artifact(PROTOCOL), artifact(parent_path), artifact(ORIGINAL / "prepared.json")]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME, AUDITOR]))]
    cells, originals = [], []
    for block in master["blocks"][:6]:
        folder = Path(block["directory"])
        admission = json.loads((folder / "admission.json").read_text())
        assert admission["admitted"]
        result = json.loads(
            checked(Path(admission["result"]["path"]), admission["result"]["sha256"]).read_text()
        )
        old = result["rows"][0]
        assert old["readout"]["requested_action"] == 0
        refs += [artifact(folder / "admission.json"), admission["result"]]
        originals.append({"block": block["index"], "row": old})
        for action in (0, 1):
            c = copy.deepcopy(next(c for c in parent["cells"] if c["encounter_action"] == action))
            group = f"{block['spec']['id']}_p{block['physics_seed']}"
            c.update(
                cell_id=f"{group}_a{action}",
                group_id=group,
                arm="diagnostic",
                beam=block["spec"]["beam"],
                scene=block["spec"]["scene"],
                runtime_seed=block["physics_seed"],
                original_block=block["index"],
                role="matched_action_diagnostic",
                output=str(out / "rollouts" / f"{group}_a{action}"),
            )
            c["hydra_overrides"] = [s for s in c["hydra_overrides"] if not s.startswith("++seed=")]
            c["hydra_overrides"].append(f"++seed={block['physics_seed']}")
            cells.append(c)
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "originals": originals,
            "original_assignment": artifact(ORIGINAL / "master.json"),
            "new_executions": 12,
            "original_pending": 480,
            "scope": "post hoc development diagnostic; v1 paused unchanged",
        },
    )
    m = copy.deepcopy(parent)
    m.update(
        experiment="M2S-selector-breakpoint-pairs-v1",
        cells=cells,
        purpose="Measure both supported commands on six inspected conditions",
        registered_predictions=artifact(PROTOCOL),
    )
    m["new_dependencies"] += refs + [artifact(out / "registration.json")]
    m["execution_policy"]["cost_ceiling"].update(rollouts=12, gpu_hours_contended=1.25)
    m["execution_policy"][
        "timing_override"
    ] = "User-prioritized matched command diagnostic; fresh cumulative budget gate."
    write_new(out / "manifest.json", m)
    subprocess.run([sys.executable, str(AUDITOR), "register", "--out", str(out)], check=True)
    print(json.dumps({"new_executions": len(cells), "reserved_gpu_hours": 1.25}))


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "run_record.json").read_text())
    reg = json.loads((out / "registration.json").read_text())
    assert (
        rec["status"] == "completed"
        and rec["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    rows, captures, decisions = [], {}, {}
    for c in m["cells"]:
        _, sampled, row = read_cell({**c, "condition": "present"}, rec)
        p = load_reset_capture(
            checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        )
        folder = Path(c["output"]) / "trajectories"
        d = json.loads((folder / "decision_capture.json").read_text())
        s = json.loads((folder / "reactive_interface.json").read_text())
        with np.load(folder / "physics_beam_contacts.npz") as f:
            _, forces, sync = physics_windows(f, sampled)
        beam = c["beam"]
        matrix = np.array(row["imported_beam"]["local_to_world_at_capture_start"])
        yaw = beam["yaw_rad"]
        rotation = np.array(
            [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        placement = np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
            rtol=0,
        )
        placement &= np.allclose(
            matrix[:3, :3],
            np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])[:, None] * rotation,
            atol=1e-6,
            rtol=0,
        )
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(all(o["transition"][k] for k in flags) for o in s["observations"]) and all(
            all(x[k] for k in flags)
            and x["joint_reference_jump_rad"] <= 0.05
            and x["root_reference_jump_m"] <= 0.01
            and (0.2 <= x["time_s"] <= 0.4 if x["to"] else 3.3 <= x["time_s"] <= 3.5)
            for x in s["switches"]
        )
        action = c["encounter_action"]
        executed = (
            not s["switches"]
            if action == 0
            else any(x["to"] == 1 and x["time_s"] == 0.3 for x in s["switches"])
        )
        row.update(
            score_passage(p, forces, beam),
            group_id=c["group_id"],
            action=action,
            physics_seed=c["runtime_seed"],
            beam=beam,
            original_block=c["original_block"],
            legal=legal,
            command_executed=executed,
            placement_matches=bool(placement),
            contact_sync_error_n=sync,
            return_logged=any(x["to"] == 0 for x in s["switches"]),
            decision=artifact(folder / "decision_capture.json"),
            sensor=artifact(folder / "reactive_interface.json"),
            physics=artifact(folder / "physics_beam_contacts.npz"),
        )
        rows.append(row)
        captures[c["cell_id"]], decisions[c["cell_id"]] = p, d
    pairs, repeats = [], []
    for original in reg["originals"]:
        a, b = [r for r in rows if r["original_block"] == original["block"]]
        prefix = paired_prefix(captures[a["cell_id"]], captures[b["cell_id"]], 0.3)
        direct = pair_decisions(decisions[a["cell_id"]], decisions[b["cell_id"]])
        valid = (
            prefix["exact_match"]
            and direct["valid"]
            and all(
                r["legal"]
                and r["command_executed"]
                and r["placement_matches"]
                and r["contact_sync_error_n"] <= 1e-6
                for r in (a, b)
            )
        )
        pairs.append(
            {
                "group_id": a["group_id"],
                "block": original["block"],
                "valid": bool(valid),
                "prefix": prefix,
                "direct": direct,
                "outcomes": [a["pass"], b["pass"]] if valid else [None, None],
                "features": decisions[a["cell_id"]]["features"],
                "training_eligible": False,
            }
        )
        old = original["row"]
        old_p = load_reset_capture(
            checked(Path(old["trajectory"]["path"]), old["trajectory"]["sha256"])
        )
        old_d = json.loads(
            checked(Path(old["decision"]["path"]), old["decision"]["sha256"]).read_text()
        )
        common = paired_prefix(old_p, captures[a["cell_id"]], 0.3)
        keys = common["max_abs_errors"]
        repeats.append(
            {
                "block": original["block"],
                "prefix": common,
                "features_exact": old_d["features"] == decisions[a["cell_id"]]["features"],
                "outcome_matches": old["pass"] == a["pass"],
                "full_trace_exact": all(
                    np.array_equal(old_p[k], captures[a["cell_id"]][k]) for k in keys
                ),
            }
        )
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "pairs": pairs,
            "historical_walk_repeats": repeats,
            "actual_contended_gpu_hours": rec["budget"]["actual_contended_gpu_hours"],
            "scientific_prediction_useful_adaptation": any(
                p["outcomes"] == [False, True] for p in pairs
            ),
            "scope": "twelve new commands on six inspected conditions; no v1 policy score replacement",
        },
    )
    subprocess.run([sys.executable, str(AUDITOR), "analyze", "--out", str(out)], check=True)
    audit = json.loads((out / "capture_audit.json").read_text())
    admitted = all(p["valid"] for p in pairs) and all(audit["predictions"].values())
    write_new(
        out / "admission.json",
        {
            "admitted": admitted,
            "result": artifact(out / "result.json"),
            "capture_audit": artifact(out / "capture_audit.json"),
        },
    )
    print(
        json.dumps(
            {"admitted": admitted, "pairs": [p["outcomes"] for p in pairs], "repeats": repeats}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "analyze"):
        globals()[args.command](args.out)
        return
    m = json.loads((args.out / "manifest.json").read_text())
    for ref in m["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    record = args.out / "run_record.json"
    if args.command == "run":
        old = json.loads(record.read_text()) if record.exists() else {"cells": {}}
        assert all(c["status"] in ("not_started", "completed") for c in old["cells"].values())
        gate = review(DATA, 12 - sum(c["status"] == "completed" for c in old["cells"].values()))
        write_new(
            args.out / f"budget_gate_{len(list(args.out.glob('budget_gate_*.json'))):03d}.json",
            gate,
        )
        if not gate["admitted"]:
            print("BUDGET YIELD")
            return
    cmd = [
        sys.executable,
        str(DRIVER),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(record),
    ]
    if args.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if args.command == "run" and json.loads(record.read_text())["status"] == "completed":
        analyze(args.out)


if __name__ == "__main__":
    main()
