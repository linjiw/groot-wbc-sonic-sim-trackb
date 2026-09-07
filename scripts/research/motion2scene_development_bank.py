#!/usr/bin/env python3
"""Acquire and qualify source-bound, matched empty-scene command transitions."""

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

from bundle_motion2scene_sources import closure
import joblib
from motion2scene_layout_budget_review import review
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import beam_at, read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    pair_decisions,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    decision_features,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_source_bank_audit import (
    source_bank_errors,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import qpos_to_sonic_motion_entry

DATA = ROOT.parent / "research-data/groot-wbc"
BANK = DATA / "cg-wbc-v2-shared-seed-confirmatory/e1_controlled_duck_ladders_v1"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_comparison_execution.py"
PROTOCOL = ROOT / "docs/motion2scene/DEVELOPMENT_TRANSITION_BANK_V1.md"
SOURCES = (41001, 41002, 41003)


def source_refs(source):
    refs = []
    for label in ("neutral", "d040"):
        p = BANK / f"duck_seed_{source}" / f"{label}.pkl"
        provenance = p.with_suffix(".pkl.manifest.json")
        meta = json.loads(provenance.read_text())
        entry = next(iter(joblib.load(checked(p, meta["output"]["sha256"])).values()))
        raw = np.loadtxt(
            checked(Path(meta["input"]["path"]), meta["input"]["sha256"]), delimiter=","
        )
        expected = qpos_to_sonic_motion_entry(raw, source_fps=30)
        errors = {
            k: float(abs(np.asarray(entry[k]) - np.asarray(expected[k])).max()) for k in expected
        }
        assert set(entry) == set(expected) and all(
            v <= (1e-6 if k == "root_trans_offset" else 1e-7) for k, v in errors.items()
        )
        refs.append(
            {
                **artifact(p),
                "conversion_provenance": str(provenance),
                "conversion_provenance_sha256": artifact(provenance)["sha256"],
                "scene_start_xyz": [0.0, 0.0, 0.0],
                "reference": meta["input"],
                "array_errors": errors,
            }
        )
    return refs


def prepare(out):
    parent_path = DATA / "m2s-comparative-acquisition-v1/manifest.json"
    parent = json.loads(parent_path.read_text())
    template = next(c for c in parent["cells"] if c["condition"] == "absent")
    out.mkdir(exist_ok=False)
    refs = [artifact(PROTOCOL), artifact(parent_path)]
    refs += [
        artifact(p)
        for p in sorted((DATA / "m2s-development-transition-bank-v1").iterdir())
        if p.is_file()
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    cells = []
    for source in SOURCES:
        neutral, alternate = source_refs(source)
        for ref in (neutral, alternate):
            refs += [ref, ref["reference"], artifact(Path(ref["conversion_provenance"]))]
        e = next(iter(joblib.load(neutral["path"]).values()))
        route = e["root_trans_offset"][:, :2]
        yaw = np.array(math.atan2(*(route[-1] - route[0])[::-1]))
        beam = {**beam_at({"route": route, "yaw": yaw}, 0.5, 1.5), "thickness_m": 0.1}
        for seed in (8721, 8722):
            for action in (0, 1):
                c = copy.deepcopy(template)
                group = f"dev_{source}_p{seed}"
                c.update(
                    cell_id=f"{group}_a{action}",
                    group_id=group,
                    generation_seed=source,
                    ladder_group_id=f"duck_seed_{source}",
                    runtime_seed=seed,
                    beam=beam,
                    motion=neutral,
                    alternate_motion=alternate,
                    reference=neutral["reference"],
                    encounter_action=action,
                    role="development_transition_qualification",
                    output=str(out / "rollouts" / f"{group}_a{action}"),
                )
                c["hydra_overrides"] = [
                    s
                    for s in c["hydra_overrides"]
                    if not any(
                        k in s for k in ("++seed=", "reactive_alternate_path=", "encounter_action=")
                    )
                ]
                c["hydra_overrides"] += [
                    f"++seed={seed}",
                    f"++manager_env.config.encounter_action={action}",
                    f"++manager_env.config.reactive_alternate_path={alternate['path']}",
                ]
                cells.append(c)
    m = copy.deepcopy(parent)
    m.update(
        experiment="M2S-development-transition-bank-v1",
        cells=cells,
        purpose="Qualify three development carriers under the deployed command contract",
        registered_predictions=artifact(PROTOCOL),
        alternate_motion={"role": "per-cell source-bound alternate_motion"},
    )
    m["new_dependencies"] += refs
    m["execution_policy"]["cost_ceiling"].update(rollouts=12, gpu_hours_contended=1.25)
    for ref in m["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(out / "manifest.json", m)
    print(json.dumps({"sources": SOURCES, "assigned": len(cells)}))


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "run_record.json").read_text())
    assert (
        rec["status"] == "completed"
        and rec["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    rows, payloads, decisions, banks = [], {}, {}, {}
    for c in m["cells"]:
        _, sampled, r = read_cell(c, rec)
        p = load_reset_capture(Path(r["trajectory"]["path"]))
        folder = Path(c["output"]) / "trajectories"
        d = json.loads((folder / "decision_capture.json").read_text())
        s = json.loads((folder / "reactive_interface.json").read_text())
        with np.load(folder / "physics_beam_contacts.npz") as f:
            _, forces, sync = physics_windows(f, sampled)
        r.update(score_passage(p, forces, c["beam"]))
        entries = [
            next(iter(joblib.load(checked(Path(ref["path"]), ref["sha256"])).values()))
            for ref in (c["motion"], c["alternate_motion"])
        ]
        with np.load(folder / "loaded_reference_bank.npz") as f:
            bank = {k: f[k].copy() for k in f.files}
        errors = source_bank_errors(bank, entries, d["joint_names"])
        source = c["generation_seed"]
        if source not in banks:
            if c.get("qualified_bank"):
                with np.load(
                    checked(Path(c["qualified_bank"]["path"]), c["qualified_bank"]["sha256"])
                ) as f:
                    banks[source] = {k: f[k].copy() for k in f.files}
            else:
                banks[source] = bank
        feature = decision_features(
            d["packet"], d["state"], d["phase_s"], d["active_before"], d["observation_age_s"]
        )
        i = d["capture_frame"]
        state_error = max(
            float(abs(np.array(v) - p[k][i]).max())
            for k, v in {
                **d["state"],
                "root_pos_w": d["root_pos_w"],
                "root_quat_w": d["root_quat_w"],
            }.items()
        )
        w, x, y, z = d["root_quat_w"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        origin = np.array(d["root_pos_w"]) + [0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4]
        flags = ("root_state_unchanged", "joint_state_unchanged", "clock_unchanged")
        legal = all(all(o["transition"][k] for k in flags) for o in s["observations"]) and all(
            all(x[k] for k in flags)
            and x["joint_reference_jump_rad"] <= 0.05
            and x["root_reference_jump_m"] <= 0.01
            and (0.2 <= x["time_s"] <= 0.4 if x["to"] else 3.3 <= x["time_s"] <= 3.5)
            for x in s["switches"]
        )
        entered = any(x["to"] == 1 and x["time_s"] == 0.3 for x in s["switches"])
        returned = any(x["to"] == 0 for x in s["switches"])
        predicates = {
            "features_exact": bool(np.array_equal(feature, np.array(d["features"], np.float32))),
            "state_bound": state_error <= 1e-7,
            "ray_origin_bound": bool(abs(origin - d["packet"]["origin"]).max() <= 1e-6),
            "source_joint_bound": all(e["joint_scalar_interpolation_rad"] <= 0.002 for e in errors),
            "source_bank_repeat": all(np.array_equal(bank[k], banks[source][k]) for k in bank),
            "contact_sync": sync <= 1e-6,
            "state_unchanged": legal,
            "decision_contract": d["phase_s"] == 0.3
            and d["requested_action"] == c["encounter_action"],
        }
        if c["condition"] == "present":
            matrix = np.array(r["imported_beam"]["local_to_world_at_capture_start"])
            b = c["beam"]
            yaw = b["yaw_rad"]
            rot = np.array(
                [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            predicates["placement"] = bool(
                np.allclose(
                    matrix[3, :3], [*b["center_xy_m"], b["underside_m"] + 0.05], atol=1e-6, rtol=0
                )
                and np.allclose(
                    matrix[:3, :3], np.array([0.1, 1.2, 0.1])[:, None] * rot, atol=1e-6, rtol=0
                )
            )
        action = c["encounter_action"]
        executed = entered if action else not s["switches"]
        qualified = (
            all(predicates.values())
            and r["pass"]
            and r["reset_count"] == 0
            and not r["fall_observed"]
            and executed
            and (returned if action else True)
        )
        r.update(
            group_id=c["group_id"],
            arm=c["arm"],
            action=action,
            physics_seed=c["runtime_seed"],
            beam=c["beam"],
            predicates=predicates,
            source_bank_errors=errors,
            state_error=state_error,
            entry_logged=entered,
            return_logged=returned,
            command_executed=executed,
            qualified=qualified,
            decision=artifact(folder / "decision_capture.json"),
            sensor=artifact(folder / "reactive_interface.json"),
            physics=artifact(folder / "physics_beam_contacts.npz"),
            bank=artifact(folder / "loaded_reference_bank.npz"),
        )
        rows.append(r)
        payloads[c["cell_id"]] = p
        decisions[c["cell_id"]] = d
    pairs = []
    for group in dict.fromkeys(c["group_id"] for c in m["cells"]):
        a, b = sorted([r for r in rows if r["group_id"] == group], key=lambda r: r["action"])
        prefix = paired_prefix(payloads[a["cell_id"]], payloads[b["cell_id"]], 0.3)
        direct = pair_decisions(decisions[a["cell_id"]], decisions[b["cell_id"]])
        valid = (
            prefix["exact_match"]
            and direct["valid"]
            and all(all(r["predicates"].values()) for r in (a, b))
        )
        pairs.append(
            {
                "group_id": group,
                "source": a["source"],
                "physics_seed": a["physics_seed"],
                "valid": valid,
                "qualified": valid and a["qualified"] and b["qualified"],
                "outcomes": [a["pass"], b["pass"]],
                "commands_executed": [a["command_executed"], b["command_executed"]],
                "features": decisions[a["cell_id"]]["features"],
                "prefix": prefix,
                "direct": direct,
            }
        )
    source_qualified = {
        str(source): all(p["qualified"] for p in pairs if p["source"] == source)
        for source in sorted({p["source"] for p in pairs})
    }
    result = {
        "manifest": artifact(out / "manifest.json"),
        "run_record": artifact(out / "run_record.json"),
        "rows": rows,
        "pairs": pairs,
        "source_qualified": source_qualified,
        "actual_contended_gpu_hours": rec["budget"]["actual_contended_gpu_hours"],
        "scope": "Development command qualification; measurement admission separate from scientific success",
    }
    write_new(out / "result.json", result)
    write_new(
        out / "admission.json",
        {
            "admitted": all(p["valid"] for p in pairs),
            "result": artifact(out / "result.json"),
            "qualified_sources": [s for s, q in source_qualified.items() if q],
        },
    )
    print(
        json.dumps(
            {
                "qualified_sources": source_qualified,
                "admitted_pairs": sum(p["valid"] for p in pairs),
                "pairs": len(pairs),
            }
        )
    )


def execute(out, preflight=False):
    m = json.loads((out / "manifest.json").read_text())
    for ref in m["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    record = out / "run_record.json"
    old = json.loads(record.read_text()) if record.exists() else {"cells": {}}
    assert all(c["status"] in ("not_started", "completed") for c in old["cells"].values())
    if not preflight:
        gate = review(
            DATA, len(m["cells"]) - sum(c["status"] == "completed" for c in old["cells"].values())
        )
        write_new(out / f"budget_gate_{len(list(out.glob('budget_gate_*.json'))):03d}.json", gate)
        if not gate["admitted"]:
            print("BUDGET YIELD")
            return
    cmd = [
        sys.executable,
        str(DRIVER),
        "--manifest",
        str(out / "manifest.json"),
        "--run-record",
        str(record),
    ]
    if preflight:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if not preflight and json.loads(record.read_text())["status"] == "completed":
        analyze(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command in ("prepare", "analyze"):
        globals()[a.command](a.out)
    else:
        execute(a.out, a.command == "preflight")
