#!/usr/bin/env python3
"""Frozen four-arm d040 proposal assignment and first direct-capture action pairs."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from bundle_motion2scene_sources import closure
import joblib
from motion2scene_beam_teacher import capsules
from motion2scene_carrier_learning import DOMAIN, RECORDS
from motion2scene_distill_study import checkpoint, independent, query_for
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_inverse_learning import inputs
from motion2scene_reactive_interface import physics_windows
from motion2scene_source_execution import beam_at, read_cell, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import make_clouds, numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import paired_prefix
from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_distinct import (
    distinct_indices,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search
from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import shared_origin_pair
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    pair_decisions,
    reserved_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import analytic_proposals
from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes
from gear_sonic.dataset_generation.hallucination.motion2scene_no_contrast import target_only_pattern
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search
from gear_sonic.dataset_generation.kimodo_motion_adapter import qpos_to_sonic_motion_entry
from gear_sonic.dataset_generation.reference_payload import payload_from_reference

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-action-label-completion-v1"
PROTOCOL = ROOT / "docs/motion2scene/COMPARATIVE_ACQUISITION_V1.md"
RUNTIME = ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_comparison_execution.py"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")


def reference_pair(prior):
    base = next(c for c in prior["cells"] if c["role"] == "timing")
    alternate = prior["alternate_motion"]
    provenance = json.loads(
        checked(
            Path(alternate["conversion_provenance"]), alternate["conversion_provenance_sha256"]
        ).read_text()
    )
    refs = [base["reference"], provenance["input"]]
    qpos = [np.loadtxt(checked(Path(r["path"]), r["sha256"]), delimiter=",") for r in refs]
    audits = []
    for raw, ref in zip(qpos, [base["motion"], alternate]):
        entry = next(iter(joblib.load(checked(Path(ref["path"]), ref["sha256"])).values()))
        reconstructed = qpos_to_sonic_motion_entry(raw, source_fps=30)
        differences = {
            k: float(np.max(np.abs(np.asarray(entry[k]) - np.asarray(reconstructed[k]))))
            for k in reconstructed
        }
        if (
            max(differences.values()) > 1e-7
            or differences["fps"] != 0
            or set(entry) != set(reconstructed)
        ):
            raise ValueError(f"reference conversion differs: {differences}")
        audits.append(
            {
                "reference": artifact(Path(refs[len(audits)]["path"])),
                "motion": ref,
                "array_errors": differences,
            }
        )
    neutral, target = shared_origin_pair(*qpos)
    states = {
        "neutral": {**capsules(payload_from_reference(neutral)), "label": "neutral"},
        "d040": {**capsules(payload_from_reference(target)), "label": "d040"},
    }
    # Compatibility identifiers only: every target array here comes from bound d040.
    legacy = {
        "reference_neutral": states["neutral"],
        "reference_d055": {**states["d040"], "label": "d055"},
    }
    route = neutral[:, :2]
    feature, _, rt, progress, yaw, mask, costs = inputs(legacy, route, RECORDS)
    case = dict(
        states=states,
        route=route,
        feature=feature,
        clouds=make_clouds(states),
        rt=rt,
        progress=progress,
        yaw=yaw,
        mask=mask,
        costs=costs,
    )
    return case, legacy, audits


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((PRIOR / "manifest.json").read_text())
    contract = DATA / "m2s-learning-contract-v2/registration.json"
    frozen = json.loads(contract.read_text())
    baseline_result = DATA / "m2s-no-contrast-baseline-v1/result.json"
    baseline = json.loads(baseline_result.read_text())
    alternate = prior["alternate_motion"]
    provenance = json.loads(Path(alternate["conversion_provenance"]).read_text())
    base = next(c for c in prior["cells"] if c["role"] == "timing")
    refs = [
        artifact(PROTOCOL),
        artifact(contract),
        artifact(PRIOR / "manifest.json"),
        frozen["generator"]["checkpoint"],
        artifact(baseline_result),
        baseline["checkpoint"],
        artifact(DATA / "m2s-refinement-v1/registration.json"),
        base["reference"],
        base["motion"],
        artifact(Path(alternate["conversion_provenance"])),
        alternate,
        provenance["input"],
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "arms": ARMS,
            "generator": frozen["generator"]["checkpoint"],
            "baseline": baseline["checkpoint"],
            "layouts": frozen["independent_common_bank_layouts"],
            "proposal_seed": 8601,
            "physics_seed": 8602,
            "decision_time_s": 0.3,
            "requested_per_arm": 12,
            "generated_slots_per_arm": 9,
            "outputs_per_arm": 16,
            "cpu_ceiling_s": 600,
            "scope": "development acquisition; no selector fitting or final tests",
        },
    )


def prepare(out):
    start = time.monotonic()
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    prior = json.loads((PRIOR / "manifest.json").read_text())
    parent = json.loads((DATA / "m2s-refinement-v1/registration.json").read_text())
    case, legacy, binding = reference_pair(prior)
    saved, model = checkpoint(reg["generator"])
    _, baseline = checkpoint(reg["baseline"])
    feature = normalized(raw_features({"bound_d040": case}), saved["config"])["bound_d040"]
    query = query_for(case, tensor(parent["search_offsets"]))
    setup_seconds = time.monotonic() - start
    rows, traces = [], {}
    for arm in ARMS:
        if time.monotonic() - start > reg["cpu_ceiling_s"]:
            raise TimeoutError("CPU proposal ceiling")
        tick = time.monotonic()
        rng = torch.Generator().manual_seed(reg["proposal_seed"])
        with torch.no_grad():
            if arm == "uniform":
                output = (
                    tensor([0.1, 1.1])
                    + torch.rand((16, 2), dtype=torch.float64, generator=rng) * tensor([0.8, 0.35])
                ).numpy()
                trace = {}
                search_queries = 0
            elif arm == "analytic":
                ranked, envelopes = analytic_proposals(
                    legacy,
                    case["route"],
                    case["yaw"].item(),
                    depth=DOMAIN.depth,
                    width=DOMAIN.width,
                    count=20,
                )
                nominal = np.array([[r["station"], r["height"]] for r in envelopes])
                ranking = np.array([int(np.argmin(abs(nominal[:, 0] - s))) for s in ranked[:, 0]])
                _, trace = global_search(
                    nominal, ranking, lambda scenes: query(tensor(scenes)).numpy()
                )
                trace["selected"] = distinct_indices(trace, count=16)
                output = trace["candidates"][trace["selected"]]
                search_queries = 4624
            else:
                initial = sample_scenes(
                    (model if arm == "motion2scene" else baseline)(feature), 16, rng
                )
                if arm == "motion2scene":
                    output, trace = station_search(initial, query, "pattern")
                else:
                    # Only the target cloud is queried; no neutral selection signal.
                    target_case = {**case, "clouds": case["clouds"][1:]}
                    target_query = query_for(target_case, tensor(parent["search_offsets"]))
                    output, trace = target_only_pattern(initial, lambda s: target_query(s)[..., 0])
                output = output.numpy()
                search_queries = 16 * 17 * 17 * (2 if arm == "motion2scene" else 1)
        synthesis = time.monotonic() - tick
        tick = time.monotonic()
        values, error = independent(output, case, np.array(parent["audit_offsets"]))
        initial_states = {
            k: {**v, "starts": v["starts"][:1], "ends": v["ends"][:1]}
            for k, v in case["states"].items()
        }
        initial_clear = numpy_perturbed(
            output, np.zeros((1, 4)), initial_states, case["route"], case["yaw"].item()
        )[:, 0, :]
        critical = verdict(values, case["mask"].numpy()).all(1)
        target_clear = values[:, :, 1].min(1) >= 0.01
        seen = set()
        for i, point in enumerate(output):
            reasons = []
            if (
                not np.isfinite(point).all()
                or np.any(point < [0.1, 1.1])
                or np.any(point > [0.9, 1.45])
            ):
                reasons.append("domain")
            if initial_clear[i].min() < 0:
                reasons.append("initial_penetration")
            if reserved_layout(*point, reg["layouts"]):
                reasons.append("reserved_test_neighborhood")
            if tuple(point) in seen:
                reasons.append("duplicate_within_arm")
            seen.add(tuple(point))
            if arm in ("motion2scene", "analytic") and not critical[i]:
                reasons.append("contrast_audit")
            if arm == "no_contrast" and not target_clear[i]:
                reasons.append("target_audit")
            rows.append(
                {
                    "id": f"{arm}_{i:02d}",
                    "arm": arm,
                    "slot": i,
                    "assigned": i < 9,
                    "station": float(point[0]),
                    "underside_m": float(point[1]),
                    "eligible": not reasons,
                    "rejection_reasons": reasons,
                    "critical_geometry": bool(critical[i]),
                    "target_clear_geometry": bool(target_clear[i]),
                    "beam": {**beam_at(case, *point), "thickness_m": 0.1},
                }
            )
        audit_seconds = time.monotonic() - tick
        arrays = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in trace.items()}
        np.savez_compressed(
            out / f"{arm}.npz",
            output=output,
            audit_clearances=values,
            initial_clearances=initial_clear,
            **arrays,
        )
        traces[arm] = {
            "raw": artifact(out / f"{arm}.npz"),
            "synthesis_seconds": synthesis,
            "audit_seconds": audit_seconds,
            "search_queries": search_queries,
            "audit_queries": int(values.size),
            "crosscheck_queries": 452,
            "initial_queries": int(initial_clear.size),
            "query_error_m": error,
        }
    proposal = {
        "registration": artifact(out / "registration.json"),
        "binding": binding,
        "target_identity": "d040",
        "legacy_slot_alias": "d055 identifier receives d040 arrays only",
        "setup_seconds": setup_seconds,
        "total_preparation_seconds": time.monotonic() - start,
        "rows": rows,
        "costs": traces,
        "backgrounds": ["absent", "raised", "blocked"],
        "scope": "48 assigned arm groups; shared backgrounds; no selector fit",
    }
    write_new(out / "proposals.json", proposal)
    (out / "scenes").mkdir()
    base = next(c for c in prior["cells"] if c["role"] == "timing")
    specs = []
    for row in rows:
        if row["slot"] < 2 and row["eligible"]:
            scene = scene_file(out / "scenes" / f"{row['id']}.usda", row["beam"])
            specs.append(
                {
                    "group_id": row["id"],
                    "condition": "generated",
                    "arm": row["arm"],
                    "beam": row["beam"],
                    "scene": scene,
                }
            )
    variation = json.loads((DATA / "m2s-overhang-variation-v1/manifest.json").read_text())
    for name in ("absent", "blocked"):
        old = next(c for c in variation["cells"] if c["condition"] == name)
        specs.append(
            {
                "group_id": f"shared_{name}",
                "condition": name,
                "arm": "shared",
                "beam": old["beam"],
                "scene": old["scene"],
            }
        )
    cells = []
    prefix = "gear_sonic.dataset_generation.hallucination.motion2scene_comparison_execution"
    for spec in specs:
        for action in (0, 1):
            c = {
                **copy.deepcopy(base),
                **spec,
                "encounter_action": action,
                "runtime_seed": 8602,
                "decision_time_s": 0.3,
                "role": "comparative_acquisition",
            }
            c["cell_id"] = f"{spec['group_id']}_a{action}"
            c["output"] = str(out / "rollouts" / c["cell_id"])
            c["hydra_overrides"] = [
                s
                for s in base["hydra_overrides"]
                if not any(
                    k in s
                    for k in (
                        "++seed=",
                        "manager_env._target_=",
                        "recorders.trajectory._target_=",
                        "encounter_action=",
                        "decision_time_s=",
                    )
                )
            ]
            c["hydra_overrides"] += [
                "++seed=8602",
                f"++manager_env._target_={prefix}.ComparisonEnvCfg",
                f"++manager_env.recorders.trajectory._target_={prefix}.ComparisonRecorderCfg",
                f"++manager_env.config.encounter_action={action}",
                "++manager_env.config.decision_time_s=0.3",
            ]
            cells.append(c)
    m = copy.deepcopy(prior)
    m.update(
        experiment="M2S-comparative-acquisition-v1-slice1",
        cells=cells,
        purpose="First fixed slots per data arm; direct 0.30 s pre-command state capture",
        registered_predictions=artifact(PROTOCOL),
        proposal_assignment=artifact(out / "proposals.json"),
    )
    m["new_dependencies"] += [
        *reg["references"],
        artifact(out / "registration.json"),
        artifact(out / "proposals.json"),
    ]
    m["execution_policy"]["cost_ceiling"].update(
        rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
    )
    m["execution_policy"]["timing_override"] = (
        "User requests next stage; at most 20 cells, 2.083334 GPU h; "
        "checked against standing daily/weekly envelope before launch."
    )
    write_new(out / "manifest.json", m)
    print(
        json.dumps(
            {
                "eligible_assigned": {
                    a: sum(r["eligible"] and r["assigned"] for r in rows if r["arm"] == a)
                    for a in ARMS
                },
                "slice_cells": len(cells),
                "costs": traces,
            },
            indent=2,
        )
    )


def analyze(out):
    m = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "run_record.json").read_text())
    assert (
        rec["status"] == "completed"
        and rec["manifest_sha256"] == artifact(out / "manifest.json")["sha256"]
    )
    rows = []
    payloads = {}
    decisions = {}
    for c in m["cells"]:
        _, sampled, row = read_cell(
            {**c, "condition": "absent" if c["condition"] == "absent" else "present"}, rec
        )
        p = load_reset_capture(
            checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        )
        folder = Path(c["output"]) / "trajectories"
        d = json.loads((folder / "decision_capture.json").read_text())
        s = json.loads((folder / "reactive_interface.json").read_text())
        with np.load(folder / "physics_beam_contacts.npz") as f:
            blocks, aggregate, error = physics_windows(f, sampled)
        assert blocks.shape == (199, 4, 30, 3) and len(s["observations"]) == 199
        matrix = np.array(row["imported_beam"]["local_to_world_at_capture_start"])
        beam = c["beam"]
        yaw = beam["yaw_rad"]
        assert np.allclose(
            matrix[3, :3],
            [*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2],
            atol=1e-6,
        )
        rotation = np.array(
            [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        assert np.allclose(
            matrix[:3, :3],
            np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]])[:, None] * rotation,
            atol=1e-6,
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
            score_passage(p, aggregate, beam),
            group_id=c["group_id"],
            arm=c["arm"],
            condition=c["condition"],
            action=action,
            legal=legal,
            command_executed=executed,
            decision=artifact(folder / "decision_capture.json"),
            sensor=artifact(folder / "reactive_interface.json"),
            physics=artifact(folder / "physics_beam_contacts.npz"),
            contact_sync_error_n=error,
        )
        rows.append(row)
        payloads[c["cell_id"]] = p
        decisions[c["cell_id"]] = d
    pairs = []
    for group in dict.fromkeys(r["group_id"] for r in rows):
        a, b = [r for r in rows if r["group_id"] == group]
        prefix = paired_prefix(payloads[a["cell_id"]], payloads[b["cell_id"]], 0.3)
        direct = pair_decisions(decisions[a["cell_id"]], decisions[b["cell_id"]])
        valid = (
            prefix["exact_match"]
            and direct["valid"]
            and all(r["legal"] and r["command_executed"] for r in (a, b))
        )
        pairs.append(
            {
                "group_id": group,
                "arm": a["arm"],
                "valid": bool(valid),
                "prefix": prefix,
                "direct": direct,
                "outcomes": [a["pass"], b["pass"]] if valid else [None, None],
                "mask": [bool(valid)] * 2,
                "features": decisions[a["cell_id"]]["features"] if valid else None,
                "training_eligible": False,
            }
        )
    predictions = {
        "p1_matching_pairs": all(
            p["prefix"]["exact_match"] and p["direct"]["valid"] for p in pairs
        ),
        "p2_legal_commands": all(r["legal"] and r["command_executed"] for r in rows),
        "p3_backgrounds": all(
            next(p for p in pairs if p["group_id"] == f"shared_{name}")["outcomes"] == expected
            for name, expected in [("absent", [True, True]), ("blocked", [False, False])]
        ),
        "p4_useful_contrast": any(
            p["arm"] in ("analytic", "motion2scene") and p["outcomes"] == [False, True]
            for p in pairs
        ),
    }
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "proposals": artifact(out / "proposals.json"),
            "rows": rows,
            "pairs": pairs,
            "predictions": predictions,
            "actual_contended_gpu_hours": rec["budget"]["actual_contended_gpu_hours"],
            "robot_data_selector_fits": 0,
            "scope": "first assigned paired data slice, one observed source; full comparative corpus pending",
        },
    )
    print(
        json.dumps(
            {"predictions": predictions, "pairs": [(p["group_id"], p["outcomes"]) for p in pairs]},
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["register", "prepare", "preflight", "run", "analyze"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command in ("register", "prepare", "analyze"):
        globals()[a.command](a.out)
        return
    m = json.loads((a.out / "manifest.json").read_text())
    for r in m["new_dependencies"]:
        checked(Path(r["path"]), r["sha256"])
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
