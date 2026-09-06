#!/usr/bin/env python3
"""Frozen learned-scene source pilot with a complete qualification/refusal funnel."""

import argparse
import copy
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

from motion2scene_distill_study import (
    DATA,
    checkpoint,
    independent,
    load,
    original_cells,
    query_for,
)
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes
from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search
from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)

SOURCES = [41005, 41006, 41007, 41008, 42005, 42006, 42007, 42008]
PRIOR = DATA / "m2s-beam-intervention-resource-v3"
DRIVER = ROOT / "scripts/research/hallucination/run_approved_manifest.py"


def distinct_indices(scenes, count=3):
    chosen = []
    for i, point in enumerate(scenes):
        if all(np.linalg.norm(point - scenes[j]) > 1e-9 for j in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return chosen


def beam_at(case, station, height):
    route = case["route"]
    progress = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    progress /= progress[-1]
    return {
        "route_progress": float(station),
        "center_xy_m": [float(np.interp(station, progress, route[:, i])) for i in range(2)],
        "yaw_rad": case["yaw"].item(),
        "length_m": 0.1,
        "width_m": 1.2,
        "underside_m": float(height),
    }


def scene_file(path, beam):
    text = (PRIOR / "beam_present.usda").read_text()
    prefix, text = text.split('    def Cube "CounterfactualBeam"', 1)
    text = re.sub(
        r"double3 xformOp:translate = \([^\n]+\)",
        f"double3 xformOp:translate = ({beam['center_xy_m'][0]}, "
        f"{beam['center_xy_m'][1]}, {beam['underside_m'] + 0.05})",
        text,
    )
    text = re.sub(
        r"double xformOp:rotateZ = [^\n]+",
        f"double xformOp:rotateZ = {np.degrees(beam['yaw_rad'])}",
        text,
    )
    with path.open("x") as f:
        f.write(prefix + '    def Cube "CounterfactualBeam"' + text)
    return {**artifact(path), "scene_id": path.stem}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    parent_path = DATA / "m2s-distillation-v1/registration.json"
    _, parent, _, dev = load(parent_path)
    selected = {f"{s}_event3": dev[f"{s}_event3"] for s in SOURCES}
    original = next(c for c in original_cells(parent) if c["seed"] == 8421)
    prior_path = PRIOR / "paired_manifest.json"
    prior = json.loads(prior_path.read_text())
    control_ref = artifact(PRIOR / "controls_result.json")
    control = json.loads(Path(control_ref["path"]).read_text())
    assert control["controls_pass"]
    refs = [
        artifact(Path(__file__)),
        artifact(parent_path),
        artifact(prior_path),
        artifact(ROOT / "docs/motion2scene/SOURCE_EXECUTION_V1.md"),
        original["checkpoint"],
        control_ref,
        control["manifest"],
        control["run_record"],
        artifact(PRIOR / "beam_present.usda"),
        artifact(PRIOR / "beam_absent.usda"),
        *prior["new_dependencies"],
        *[
            artifact(ROOT / "gear_sonic/dataset_generation" / name)
            for name in ("kimodo_motion_adapter.py", "capsule_box_exact.py")
        ],
    ]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {"dependencies": refs, "sources": SOURCES, "event_index": 3, "physics_seed": 7921},
    )
    start = time.monotonic()
    saved, model = checkpoint(original["checkpoint"])
    features = normalized(raw_features(selected), saved["config"])
    template = {
        k: copy.deepcopy(prior[k])
        for k in (
            "implementation",
            "execution_policy",
            "authorization",
            "schema_version",
            "promotion",
        )
    }
    template.update(
        registered_predictions=artifact(ROOT / "docs/motion2scene/SOURCE_EXECUTION_V1.md"),
        new_dependencies=refs,
        analysis_role="observed_reference_source_learned_beam_execution_pilot",
        purpose="Eight previously observed development source pairs; one physics seed; complete refusal funnel",
        stop_conditions=[
            "Serial; 7500 MiB startup floor; 375 seconds per cell",
            "Stop infrastructure failures; retain scientific refusals without retries",
        ],
    )
    template["authorization"][
        "note"
    ] = "User requests advancing the research plan and running available simulation budget."
    template["execution_policy"][
        "timing_override"
    ] = "Frozen source-execution development pilot registered before outcomes."
    cells, potential, rows = [], [], []
    (out / "motions").mkdir()
    (out / "scenes").mkdir()
    templates = {
        c["label"]: c
        for c in prior["cells"]
        if c["runtime_seed"] == 7911 and c["condition"] == "absent"
    }
    for name, case in selected.items():
        if time.monotonic() - start > 600:
            raise TimeoutError("proposal/conversion budget exceeded")
        source = case["metadata"]["carrier_seed"]
        tick = time.monotonic()
        with torch.no_grad():
            initial = sample_scenes(
                model(features[name]), 8, torch.Generator().manual_seed(310000 + source)
            )
        scenes, trace = station_search(
            initial, query_for(case, tensor(parent["search_offsets"])), "pattern"
        )
        chosen = distinct_indices(scenes.numpy())
        values, error = independent(scenes.numpy(), case, np.array(parent["audit_offsets"]))
        passing = verdict(values, case["mask"].numpy()).all(1)
        raw = out / f"proposals_{source}.npz"
        np.savez_compressed(
            raw,
            output=scenes.numpy(),
            audit_clearances=values,
            **{k: v.numpy() for k, v in trace.items()},
        )
        slots = [
            {
                "slot": j,
                "output_index": i,
                "geometry_pass": bool(passing[i]),
                "beam": beam_at(case, *scenes[i].tolist()),
            }
            for j, i in enumerate(chosen)
        ]
        slots += [
            {"slot": j, "output_index": None, "geometry_pass": False, "beam": None}
            for j in range(len(slots), 3)
        ]
        row = {
            "source": source,
            "case_id": name,
            "canonical_beam": beam_at(case, 0.35, 2.0),
            "slots": slots,
            "all_eight_pass": passing.tolist(),
            "raw": artifact(raw),
            "query_error_m": error,
            "proposal_seconds": time.monotonic() - tick,
            "queries": 6884,
            "reference_metadata": case["metadata"],
        }
        scene_refs = {
            s["slot"]: scene_file(out / "scenes" / f"beam_{source}_{s['slot']}.usda", s["beam"])
            for s in slots
            if s["beam"] is not None
        }
        for label, key in (("neutral", "parent"), ("d055", "target")):
            ref = case["metadata"][key]
            gate = case["metadata"]["neutral_gate" if label == "neutral" else "target_gate"]
            assert gate["q0_pass"] and gate["q1_pass"]
            qpos = np.loadtxt(checked(Path(ref["path"]), ref["sha256"]), delimiter=",")
            motion = out / "motions" / f"{source}_{label}.pkl"
            save_sonic_motion_file(
                motion, motion_key=f"{name}__{label}", motion_entry=qpos_to_sonic_motion_entry(qpos)
            )
            provenance = motion.with_suffix(".json")
            write_new(
                provenance,
                {
                    "scene_start_xyz": [0.0, 0.0, 0.0],
                    "scene_yaw": 0.0,
                    "source_fps": 30,
                    "canonicalize_horizontal_origin": True,
                    "input": ref,
                    "output": artifact(motion),
                    "registration": artifact(out / "registration.json"),
                },
            )
            cell = copy.deepcopy(templates[label])
            cell_id = f"source_{source}_absent_{label}"
            cell.update(
                cell_id=cell_id,
                generation_seed=source,
                base_carrier_id=str(source),
                ladder_group_id=name,
                runtime_seed=7921,
                reference=ref,
                motion={
                    **artifact(motion),
                    "conversion_provenance": str(provenance),
                    "conversion_provenance_sha256": artifact(provenance)["sha256"],
                    "scene_start_xyz": [0.0, 0.0, 0.0],
                },
                output=str(out / "rollouts" / cell_id),
            )
            cell["hydra_overrides"][0] = "++seed=7921"
            cell.pop("depends_on_acceptance_of", None)
            cells.append(cell)
            for slot in slots:
                if not slot["geometry_pass"]:
                    continue
                c = copy.deepcopy(cell)
                cell_id = f"source_{source}_present{slot['slot']}_{label}"
                c.update(
                    cell_id=cell_id,
                    condition="present",
                    slot=slot["slot"],
                    scene=scene_refs[slot["slot"]],
                    output=str(out / "rollouts" / cell_id),
                )
                potential.append(c)
        rows.append(row)
        print(
            f"FROZEN {source}: {sum(passing)}/8 reference accept; "
            f"{sum(s['geometry_pass'] for s in slots)}/3 selected",
            flush=True,
        )
    write_new(
        out / "proposals.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "queries": 55072,
            "preparation_seconds": time.monotonic() - start,
        },
    )
    template["eligibility"] = {
        "reference_gate": artifact(out / "proposals.json"),
        "rule": "all eight Q0/Q1 development sources, no outcome replacement",
    }
    absent = copy.deepcopy(template)
    absent.update(experiment="M2S-source-execution-v1-absent", cells=cells)
    absent["execution_policy"]["cost_ceiling"].update(
        rollouts=16, gpu_hours_contended=16 * 375 / 3600
    )
    write_new(out / "absent_manifest.json", absent)
    write_new(
        out / "present_template.json",
        {"manifest": template, "potential_cells": potential, "maximum_cells": 48},
    )


def read_cell(cell, record):
    science = record["cells"][cell["cell_id"]]["scientific"]
    ref = science["artifacts"]
    with checked(Path(ref["trajectory"]), ref["trajectory_sha256"]).open("rb") as f:
        payload = pickle.load(f)
    folder = Path(cell["output"]) / "trajectories"
    contacts = folder / "beam_contacts.npz"
    inventory_path = folder / "native_collision_inventory.json"
    with np.load(contacts, allow_pickle=False) as data:
        forces = data["force_w"].copy()
        filters = list(data["filter_paths"])
        assert float(data["fps"]) == float(payload["fps"])
    inventory = json.loads(inventory_path.read_text())
    assert filters == inventory["filter_paths"] and len(filters) == forces.shape[1] == 30
    beam = next(r for r in inventory["shapes"] if r["path"] == inventory["beam_path"])
    assert beam["attributes"]["physics:collisionEnabled"] == str(cell["condition"] == "present")
    return (
        payload,
        forces,
        {
            "cell_id": cell["cell_id"],
            "source": cell["generation_seed"],
            "label": cell["label"],
            "condition": cell["condition"],
            "tracker_outcome": science["outcome"],
            "tracker_rejection_reasons": science["rejection_reasons"],
            "trajectory": {"path": ref["trajectory"], "sha256": ref["trajectory_sha256"]},
            "contacts": artifact(contacts),
            "inventory": artifact(inventory_path),
            "imported_beam": beam,
        },
    )


def completed(out, batch):
    path = out / f"{batch}_manifest.json"
    manifest = json.loads(path.read_text())
    record = json.loads((out / f"{batch}_run_record.json").read_text())
    if record["status"] != "completed" or record["manifest_sha256"] != artifact(path)["sha256"]:
        raise ValueError("complete matching batch required")
    return manifest, record


def qualification_map(rows):
    expected = {(s, label) for s in SOURCES for label in ("neutral", "d055")}
    if len(rows) != len(expected) or {(r["source"], r["label"]) for r in rows} != expected:
        raise ValueError("requires the exact complete source/label panel")
    return {
        str(s): all(
            r["pass"] and r["tracker_outcome"] == "accepted" for r in rows if r["source"] == s
        )
        for s in SOURCES
    }


def qualify(out):
    manifest, record = completed(out, "absent")
    proposals = json.loads((out / "proposals.json").read_text())
    by_source = {r["source"]: r for r in proposals["rows"]}
    rows = []
    for cell in manifest["cells"]:
        payload, forces, row = read_cell(cell, record)
        source = by_source[row["source"]]
        row.update(score_passage(payload, forces, source["canonical_beam"]))
        row["slot_passages"] = {
            str(s["slot"]): score_passage(payload, forces, s["beam"])
            for s in source["slots"]
            if s["beam"] is not None
        }
        rows.append(row)
    qualified = qualification_map(rows)
    result = {
        "manifest": artifact(out / "absent_manifest.json"),
        "run_record": artifact(out / "absent_run_record.json"),
        "rows": rows,
        "qualified": qualified,
        "p1_at_least_four_qualified": sum(qualified.values()) >= 4,
    }
    write_new(out / "qualification.json", result)
    template = json.loads((out / "present_template.json").read_text())
    present = template["manifest"]
    cells = [c for c in template["potential_cells"] if qualified[str(c["generation_seed"])]]
    present.update(experiment="M2S-source-execution-v1-present", cells=cells)
    present["new_dependencies"] += [
        artifact(out / "qualification.json"),
        artifact(out / "present_template.json"),
    ]
    present["execution_policy"]["cost_ceiling"].update(
        rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
    )
    write_new(out / "present_manifest.json", present)
    print(f"QUALIFIED {sum(qualified.values())}/8; {len(cells)} frozen present runs", flush=True)


def analyze(out):
    manifest, record = completed(out, "present")
    qualification = json.loads((out / "qualification.json").read_text())
    proposals = json.loads((out / "proposals.json").read_text())
    by_source = {r["source"]: r for r in proposals["rows"]}
    rows, slots = [], []
    for cell in manifest["cells"]:
        payload, forces, row = read_cell(cell, record)
        beam = by_source[row["source"]]["slots"][cell["slot"]]["beam"]
        matrix = np.asarray(row["imported_beam"]["local_to_world_at_capture_start"])
        assert np.allclose(matrix[3, :3], [*beam["center_xy_m"], beam["underside_m"] + 0.05])
        row.update(score_passage(payload, forces, beam), slot=cell["slot"])
        rows.append(row)
    for source in proposals["rows"]:
        source_id = source["source"]
        for s in source["slots"]:
            slot = {"source": source_id, "slot": s["slot"], "separation": False}
            if not s["geometry_pass"]:
                slot["status"] = "reference_refusal"
            elif not qualification["qualified"][str(source_id)]:
                slot["status"] = "source_qualification_refusal"
            else:
                pair = {
                    r["label"]: r
                    for r in rows
                    if r["source"] == source_id and r["slot"] == s["slot"]
                }
                assert set(pair) == {"neutral", "d055"}
                absent = {
                    r["label"]: r["slot_passages"][str(s["slot"])]
                    for r in qualification["rows"]
                    if r["source"] == source_id
                }
                rates = {
                    **{f"{k}_present": int(v["pass"]) for k, v in pair.items()},
                    **{f"{k}_absent": int(v["pass"]) for k, v in absent.items()},
                }
                slot.update(
                    status="executed",
                    rates=rates,
                    separation=bool(
                        pair["d055"]["pass"]
                        and pair["neutral"]["observed_beam_contact"]
                        and all(v["pass"] for v in absent.values())
                    ),
                    counterfactual_interaction=rates["d055_present"]
                    - rates["neutral_present"]
                    - rates["d055_absent"]
                    + rates["neutral_absent"],
                )
            slots.append(slot)
    qualified = sum(qualification["qualified"].values())
    source_counts = {
        str(s): sum(r["separation"] for r in slots if r["source"] == s) for s in SOURCES
    }
    accepted = sum(source_counts.values())
    result = {
        "manifest": artifact(out / "present_manifest.json"),
        "run_record": artifact(out / "present_run_record.json"),
        "qualification": artifact(out / "qualification.json"),
        "proposals": artifact(out / "proposals.json"),
        "rows": rows,
        "slots": slots,
        "separated_slots_per_source": source_counts,
        "qualified_sources": qualified,
        "separated_slots": accepted,
        "requested_slots": 24,
        "predictions": {
            "p1": qualification["p1_at_least_four_qualified"],
            "p2": sum(v == 3 for v in source_counts.values()) >= max(1, np.ceil(qualified / 2)),
            "p3": accepted >= 12,
        },
    }
    result["predictions"]["p2"] = bool(result["predictions"]["p2"])
    write_new(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "slots")}), flush=True)


def run(out):
    for batch in ("absent", "present"):
        if batch == "present" and not (out / "qualification.json").exists():
            qualify(out)
        manifest_path = out / f"{batch}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for ref in manifest["new_dependencies"]:
            checked(Path(ref["path"]), ref["sha256"])
        record_path = out / f"{batch}_run_record.json"
        deadline = time.monotonic() + 3600
        while True:
            status = (
                json.loads(record_path.read_text())["status"] if record_path.exists() else "new"
            )
            if status == "completed":
                break
            if status not in ("new", "preflight_passed", "yielded_gpu_contention"):
                raise RuntimeError(f"refuse automatic retry: {status}")
            if time.monotonic() > deadline:
                raise TimeoutError("resource monitoring window exhausted")
            rc = subprocess.call(
                [
                    sys.executable,
                    str(DRIVER),
                    "--manifest",
                    str(manifest_path),
                    "--run-record",
                    str(record_path),
                ]
            )
            if rc:
                if json.loads(record_path.read_text())["status"] != "yielded_gpu_contention":
                    raise RuntimeError("runner failed outside resource contention")
                time.sleep(20)
    if not (out / "result.json").exists():
        analyze(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
