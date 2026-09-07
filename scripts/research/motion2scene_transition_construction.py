#!/usr/bin/env python3
"""Frozen-initializer construction using achieved command geometry, with charged refusals."""

import argparse
import copy
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_beam_teacher import capsules
from motion2scene_carrier_learning import DOMAIN, RECORDS
from motion2scene_development_bank import DATA, RUNTIME, execute, source_refs
from motion2scene_distill_study import checkpoint, independent, query_for
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_inverse_learning import inputs
from motion2scene_source_execution import beam_at, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import make_clouds, numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_distinct import (
    distinct_indices,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search
from gear_sonic.dataset_generation.hallucination.motion2scene_beam_visibility import beam_ray_hits
from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import shared_origin_pair
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    reserved_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import analytic_proposals
from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search
from gear_sonic.dataset_generation.reference_payload import payload_from_reference

BANK = DATA / "m2s-development-transition-bank-v2"
PROTOCOL = ROOT / "docs/motion2scene/TRANSITION_CONSTRUCTION_PILOT_V1.md"


def register(out):
    admission = json.loads((BANK / "admission.json").read_text())
    assert admission["admitted"]
    frozen = json.loads((DATA / "m2s-learning-contract-v2/registration.json").read_text())
    refs = [
        artifact(PROTOCOL),
        artifact(BANK / "manifest.json"),
        artifact(BANK / "result.json"),
        artifact(BANK / "admission.json"),
        artifact(DATA / "m2s-refinement-v1/registration.json"),
        artifact(DATA / "m2s-learning-contract-v2/registration.json"),
        frozen["generator"]["checkpoint"],
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    out.mkdir(exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "qualified_sources": admission["qualified_sources"],
            "requested_sources": [41001, 41002, 41003],
            "arms": ["analytic", "motion2scene"],
            "slots_per_arm": 2,
            "generator": frozen["generator"]["checkpoint"],
            "layouts": frozen["independent_common_bank_layouts"],
            "proposal_seed": 8731,
            "cpu_ceiling_s": 600,
        },
    )


def cases(source, bank_result):
    refs = source_refs(source)
    qpos = [np.loadtxt(r["reference"]["path"], delimiter=",") for r in refs]
    neutral, target = shared_origin_pair(*qpos)
    refstates = {
        "reference_neutral": {**capsules(payload_from_reference(neutral)), "label": "neutral"},
        "reference_d055": {**capsules(payload_from_reference(target)), "label": "d055"},
    }
    feature, _, rt, progress, yaw, mask, costs = inputs(refstates, neutral[:, :2], RECORDS)
    rows = sorted(
        [r for r in bank_result["rows"] if r["source"] == source and r["physics_seed"] == 8721],
        key=lambda r: r["action"],
    )
    assert len(rows) == 2 and all(r["qualified"] for r in rows)
    payloads = [
        load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        for r in rows
    ]
    states = {
        name: {**capsules(p), "label": label}
        for name, p, label in zip(("neutral", "d040"), payloads, ("neutral", "d040"))
    }
    case = {
        "feature": feature,
        "route": neutral[:, :2],
        "states": states,
        "rt": rt,
        "progress": progress,
        "yaw": yaw,
        "mask": mask,
        "costs": costs,
        "clouds": make_clouds(states),
    }
    legacy = {
        "reference_neutral": states["neutral"],
        "reference_d055": {**states["d040"], "label": "d055"},
    }
    packet = json.loads(Path(rows[0]["decision"]["path"]).read_text())["packet"]
    return case, legacy, packet, rows


def prepare(out):
    start = time.monotonic()
    torch.set_num_threads(2)
    reg = json.loads((out / "registration.json").read_text())
    for ref in reg["references"]:
        checked(Path(ref["path"]), ref["sha256"])
    bank_result = json.loads((BANK / "result.json").read_text())
    parent = json.loads((BANK / "manifest.json").read_text())
    search = json.loads((DATA / "m2s-refinement-v1/registration.json").read_text())
    saved, model = checkpoint(reg["generator"])
    rows = []
    costs = []
    batches = []
    (out / "scenes").mkdir()
    for source in reg["requested_sources"]:
        if str(source) not in reg["qualified_sources"]:
            for arm in reg["arms"]:
                for slot in range(2):
                    rows.append(
                        {
                            "source": source,
                            "arm": arm,
                            "slot": slot,
                            "assigned": True,
                            "eligible": False,
                            "reasons": ["source_not_qualified"],
                        }
                    )
            continue
        case, legacy, packet, bank_rows = cases(source, bank_result)
        feature = normalized(raw_features({"case": case}), saved["config"])["case"]
        base_query = query_for(case, tensor(search["search_offsets"]))
        source_cells = []
        batch_folder = out.with_name(out.name + f"-s{source}")
        for arm in reg["arms"]:
            if time.monotonic() - start > reg["cpu_ceiling_s"]:
                raise TimeoutError("construction preparation ceiling")
            tick = time.monotonic()
            charged = []

            def query(scenes):
                charged.append(len(scenes) * len(search["search_offsets"]) * 2)
                return base_query(scenes)

            with torch.no_grad():
                if arm == "analytic":
                    ranked, envelopes = analytic_proposals(
                        legacy,
                        case["route"],
                        case["yaw"].item(),
                        depth=DOMAIN.depth,
                        width=DOMAIN.width,
                        count=20,
                    )
                    nominal = np.array([[r["station"], r["height"]] for r in envelopes])
                    ranking = np.array(
                        [int(np.argmin(abs(nominal[:, 0] - s))) for s in ranked[:, 0]]
                    )
                    _, trace = global_search(nominal, ranking, lambda x: query(tensor(x)).numpy())
                    trace["selected"] = distinct_indices(trace, count=16)
                    output = trace["candidates"][trace["selected"]]
                else:
                    initial = sample_scenes(
                        model(feature), 16, torch.Generator().manual_seed(reg["proposal_seed"])
                    )
                    output, trace = station_search(initial, query, "pattern")
                    output = output.numpy()
            values, error = independent(output, case, np.array(search["audit_offsets"]))
            pre_states = {
                k: {**v, "starts": v["starts"][:16], "ends": v["ends"][:16]}
                for k, v in case["states"].items()
            }
            preclear = numpy_perturbed(
                output, np.zeros((1, 4)), pre_states, case["route"], case["yaw"].item()
            )[:, 0, :]
            valid = verdict(values, case["mask"].numpy()).all(1)
            seen = set()
            np.savez_compressed(
                out / f"{source}_{arm}.npz",
                output=output,
                audit=values,
                predecision_clearance=preclear,
                **{k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in trace.items()},
            )
            costs.append(
                {
                    "source": source,
                    "arm": arm,
                    "search_queries": sum(charged),
                    "audit_queries": int(values.size),
                    "crosscheck_queries": 452,
                    "predecision_queries": int(preclear.size),
                    "seconds": time.monotonic() - tick,
                    "trace": artifact(out / f"{source}_{arm}.npz"),
                    "independent_error_m": error,
                    "bank_inputs": [r["trajectory"] for r in bank_rows],
                }
            )
            for slot, point in enumerate(output):
                beam = {**beam_at(case, *point), "thickness_m": 0.1}
                hits = beam_ray_hits(packet, beam)
                reasons = []
                if (
                    not np.isfinite(point).all()
                    or np.any(point < [0.1, 1.1])
                    or np.any(point > [0.9, 1.45])
                ):
                    reasons.append("domain")
                if not valid[slot]:
                    reasons.append("achieved_contrast_audit")
                if preclear[slot].min() < 0.01:
                    reasons.append("predecision_clearance")
                if not hits:
                    reasons.append("box_visibility")
                if reserved_layout(*point, reg["layouts"]):
                    reasons.append("reserved_layout_neighborhood")
                if tuple(point) in seen:
                    reasons.append("duplicate_within_arm")
                seen.add(tuple(point))
                row = {
                    "source": source,
                    "arm": arm,
                    "slot": slot,
                    "assigned": slot < 2,
                    "eligible": not reasons,
                    "reasons": reasons,
                    "beam": beam,
                    "visibility_proposal_hits": hits,
                    "target_min_clearance_m": float(values[slot, :, 1].min()),
                    "walk_max_clearance_m": float(values[slot, :, 0].max()),
                    "predecision_min_clearance_m": float(preclear[slot].min()),
                }
                rows.append(row)
                if slot >= 2 or reasons:
                    continue
                scene = scene_file(out / "scenes" / f"transition_{source}_{arm}_{slot}.usda", beam)
                for seed in (8721, 8722):
                    for action in (0, 1):
                        old = next(
                            c
                            for c in parent["cells"]
                            if c["generation_seed"] == source
                            and c["runtime_seed"] == seed
                            and c["encounter_action"] == action
                        )
                        prior = next(
                            r for r in bank_result["rows"] if r["cell_id"] == old["cell_id"]
                        )
                        c = copy.deepcopy(old)
                        group = f"transition_{source}_{arm}_{slot}_p{seed}"
                        c.update(
                            cell_id=f"{group}_a{action}",
                            group_id=group,
                            condition="present",
                            arm=arm,
                            role="transition_aware_construction",
                            scene=scene,
                            beam=beam,
                            qualified_bank=prior["bank"],
                            empty_comparator=prior,
                            output=str(batch_folder / f"rollouts/{group}_a{action}"),
                        )
                        source_cells.append(c)
        if source_cells:
            folder = batch_folder
            folder.mkdir()
            m = copy.deepcopy(parent)
            m.update(
                cells=source_cells,
                experiment=f"M2S-transition-construction-{source}-v1",
                registered_predictions=artifact(PROTOCOL),
                purpose="Paired physical labels for fixed achieved-transition proposals",
            )
            m["new_dependencies"] += (
                reg["references"]
                + [c["scene"] for c in source_cells]
                + [c["qualified_bank"] for c in source_cells]
            )
            m["execution_policy"]["cost_ceiling"].update(
                rollouts=len(source_cells), gpu_hours_contended=len(source_cells) * 375 / 3600
            )
            write_new(folder / "manifest.json", m)
            batches.append(
                {
                    "source": source,
                    "directory": str(folder),
                    "manifest": artifact(folder / "manifest.json"),
                    "assigned": len(source_cells),
                }
            )
    write_new(
        out / "proposals.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "costs": costs,
            "seconds": time.monotonic() - start,
            "requested_slots": 12,
            "scope": "Proposal screening, not physical outcome labels",
        },
    )
    write_new(
        out / "prepared.json",
        {
            "proposals": artifact(out / "proposals.json"),
            "batches": batches,
            "physics_assigned": sum(b["assigned"] for b in batches),
        },
    )
    print(
        json.dumps(
            {
                "eligible_assigned_slots": sum(r["assigned"] and r["eligible"] for r in rows),
                "requested_slots": 12,
                "physics_assigned": sum(b["assigned"] for b in batches),
            }
        )
    )


def summarize(out):
    prep = json.loads((out / "prepared.json").read_text())
    results = []
    for b in prep["batches"]:
        p = Path(b["directory"]) / "admission.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        assert a["admitted"]
        r = json.loads(Path(a["result"]["path"]).read_text())
        for row in r["rows"]:
            d = json.loads(Path(row["decision"]["path"]).read_text())
            row["observed_beam_overhang"] = any(
                x["overhang"] and x["hit"] and x["hit"]["path"].endswith("CounterfactualBeam")
                for x in d["packet"]["rays"]
            )
        results.append(r)
    pairs = []
    for result in results:
        for pair in result["pairs"]:
            rr = [r for r in result["rows"] if r["group_id"] == pair["group_id"]]
            pairs.append(
                {
                    **pair,
                    "arm": rr[0]["arm"],
                    "observable": all(r["observed_beam_overhang"] for r in rr),
                    "useful": pair["commands_executed"] == [True, True]
                    and pair["outcomes"] == [False, True],
                }
            )
    write_new(
        out / f"physical_summary_{sum(len(r['rows']) for r in results):03d}.json",
        {
            "prepared": artifact(out / "prepared.json"),
            "pairs": pairs,
            "completed": sum(len(r["rows"]) for r in results),
            "assigned": prep["physics_assigned"],
            "gpu_hours": sum(r["actual_contended_gpu_hours"] for r in results),
            "scope": (
                "Paired-label mechanism pilot; source_qualified fields in batch scorer refer to "
                "both-command success across its scenes, not empty-bank qualification"
            ),
        },
    )
    print(
        json.dumps(
            {
                "completed": sum(len(r["rows"]) for r in results),
                "assigned": prep["physics_assigned"],
                "useful_observable_pairs": sum(p["useful"] and p["observable"] for p in pairs),
                "pairs": len(pairs),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("register", "prepare", "run", "summarize"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command != "run":
        globals()[a.command](a.out)
    else:
        prep = json.loads((a.out / "prepared.json").read_text())
        for b in prep["batches"]:
            folder = Path(b["directory"])
            if (folder / "admission.json").exists():
                continue
            execute(folder, preflight=True)
            execute(folder)
            if not (folder / "admission.json").exists():
                break
            assert json.loads((folder / "admission.json").read_text())["admitted"]
        summarize(a.out)
