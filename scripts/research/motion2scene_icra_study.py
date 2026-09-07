#!/usr/bin/env python3
"""Register and acquire the fixed four-arm ICRA study, preserving every refusal."""

import argparse
import copy
import json
from pathlib import Path
import re
import time

from bundle_motion2scene_sources import closure
from motion2scene_carrier_learning import DOMAIN
from motion2scene_development_bank import DATA, RUNTIME
from motion2scene_distill_study import checkpoint, independent, query_for
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_icra_audit import execute
from motion2scene_source_execution import beam_at, scene_file
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_transition_construction import BANK, cases
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_distinct import (
    distinct_indices,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search
from gear_sonic.dataset_generation.hallucination.motion2scene_beam_visibility import beam_ray_hits
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    reserved_layout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import analytic_proposals
from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes
from gear_sonic.dataset_generation.hallucination.motion2scene_no_contrast import target_only_pattern
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search

PROTOCOL = ROOT / "docs/motion2scene/M2S_ICRA_V1.md"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")
BACKGROUNDS = {
    41001: ("absent", "raised"),
    41002: ("absent", "blocked"),
    41003: ("raised", "blocked"),
}


def eligibility(arm, point, preclear, critical, target_clear, duplicate, layouts):
    reasons = []
    if not np.isfinite(point).all() or np.any(point < [0.1, 1.1]) or np.any(point > [0.9, 1.45]):
        reasons.append("domain")
    if preclear < 0.01:
        reasons.append("predecision_clearance")
    if reserved_layout(*point, layouts):
        reasons.append("reserved_layout")
    if duplicate:
        reasons.append("duplicate_within_arm")
    if arm in ("analytic", "motion2scene") and not critical:
        reasons.append("contrast_audit")
    if arm == "no_contrast" and not target_clear:
        reasons.append("target_audit")
    return reasons


def register(out):
    out.mkdir(exist_ok=False)
    for name in ("fable.md", "fable.html"):
        (out / name).write_bytes((ROOT / name).read_bytes())
    frozen = json.loads((DATA / "m2s-learning-contract-v2/registration.json").read_text())
    baseline = json.loads((DATA / "m2s-no-contrast-baseline-v1/result.json").read_text())
    refs = [
        artifact(PROTOCOL),
        artifact(BANK / "manifest.json"),
        artifact(BANK / "result.json"),
        artifact(BANK / "admission.json"),
        artifact(DATA / "m2s-refinement-v1/registration.json"),
        artifact(DATA / "m2s-learning-contract-v2/registration.json"),
        artifact(DATA / "m2s-no-contrast-baseline-v1/result.json"),
        frozen["generator"]["checkpoint"],
        baseline["checkpoint"],
    ]
    refs += [artifact(out / name) for name in ("fable.md", "fable.html")]
    refs += [artifact(p) for p in sorted(closure([Path(__file__), RUNTIME]))]
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    assert json.loads((BANK / "admission.json").read_text())["qualified_sources"] == [
        "41001",
        "41002",
        "41003",
    ]
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "sources": list(BACKGROUNDS),
            "arms": ARMS,
            "generator": frozen["generator"]["checkpoint"],
            "baseline": baseline["checkpoint"],
            "layouts": frozen["independent_common_bank_layouts"],
            "proposal_seed": 8841,
            "physics_seed": 8722,
            "generated_slots_per_source_arm": 6,
            "requested_per_arm": 24,
            "cpu_ceiling_s": 900,
            "maximum_label_cells": 156,
            "refitting": False,
            "evaluation_seeds": [8511, 8512],
            "evaluation_traversal_assignments": 432,
            "scope": "Inspected development carriers and reserved layouts; complete-count goal may fail",
        },
    )


def physical_scene(path, beam, absent=False):
    # Standard room and beam; change only declared box dimensions and enabled state.
    scene_file(path, beam)
    text = path.read_text()
    prefix, tail = text.split('    def Cube "CounterfactualBeam"', 1)
    tail = re.sub(
        r"double3 xformOp:translate = \([^\n]+\)",
        f"double3 xformOp:translate = ({beam['center_xy_m'][0]}, {beam['center_xy_m'][1]}, "
        f"{beam['underside_m'] + beam['thickness_m'] / 2})",
        tail,
    )
    tail = re.sub(
        r"double3 xformOp:scale = \([^\n]+\)",
        f"double3 xformOp:scale = ({beam['length_m']}, {beam['width_m']}, {beam['thickness_m']})",
        tail,
    )
    if absent:
        tail = re.sub(
            r"bool physics:collisionEnabled = (true|1)",
            "bool physics:collisionEnabled = false",
            tail,
        )
    path.write_text(prefix + '    def Cube "CounterfactualBeam"' + tail)
    return {**artifact(path), "scene_id": path.stem}


def prepare(out):
    tick = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reg = json.loads((out / "registration.json").read_text())
    for r in reg["references"]:
        checked(Path(r["path"]), r["sha256"])
    bank = json.loads((BANK / "result.json").read_text())
    parent = json.loads((BANK / "manifest.json").read_text())
    search = json.loads((DATA / "m2s-refinement-v1/registration.json").read_text())
    saved, model = checkpoint(reg["generator"])
    _, baseline = checkpoint(reg["baseline"])
    (out / "scenes").mkdir()
    rows, costs, specs = [], [], []
    for source in reg["sources"]:
        case, legacy, packet, _ = cases(source, bank)
        feature = normalized(raw_features({"case": case}), saved["config"])["case"]
        for arm in ARMS:
            if time.monotonic() - tick > reg["cpu_ceiling_s"]:
                raise TimeoutError("registered preparation ceiling")
            start = time.monotonic()
            charged = []
            cloud_case = {**case, "clouds": case["clouds"][1:]} if arm == "no_contrast" else case
            base_query = query_for(cloud_case, tensor(search["search_offsets"]))

            def query(points):
                charged.append(len(points) * 17 * (1 if arm == "no_contrast" else 2))
                return base_query(points)

            rng = torch.Generator().manual_seed(reg["proposal_seed"])
            with torch.no_grad():
                if arm == "uniform":
                    output = (
                        tensor([0.1, 1.1])
                        + torch.rand((16, 2), generator=rng, dtype=torch.float64)
                        * tensor([0.8, 0.35])
                    ).numpy()
                    trace = {}
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
                    ranking = np.array(
                        [int(np.argmin(abs(nominal[:, 0] - s))) for s in ranked[:, 0]]
                    )
                    _, trace = global_search(nominal, ranking, lambda x: query(tensor(x)).numpy())
                    trace["selected"] = distinct_indices(trace, count=16)
                    output = trace["candidates"][trace["selected"]]
                else:
                    initial = sample_scenes(
                        (baseline if arm == "no_contrast" else model)(feature), 16, rng
                    )
                    if arm == "no_contrast":
                        output, trace = target_only_pattern(initial, lambda x: query(x)[..., 0])
                    else:
                        output, trace = station_search(initial, query, "pattern")
                    output = output.numpy()
            values, error = independent(output, case, np.array(search["audit_offsets"]))
            pre = {
                k: {**v, "starts": v["starts"][:16], "ends": v["ends"][:16]}
                for k, v in case["states"].items()
            }
            preclear = numpy_perturbed(
                output, np.zeros((1, 4)), pre, case["route"], case["yaw"].item()
            )[:, 0, :]
            critical = verdict(values, case["mask"].numpy()).all(1)
            targets = values[:, :, 1].min(1) >= 0.01
            seen = set()
            for slot, point in enumerate(output):
                reasons = eligibility(
                    arm,
                    point,
                    preclear[slot].min(),
                    critical[slot],
                    targets[slot],
                    tuple(point) in seen,
                    reg["layouts"],
                )
                seen.add(tuple(point))
                beam = {**beam_at(case, *point), "thickness_m": 0.1}
                row = {
                    "group_id": f"icra_{source}_{arm}_{slot:02d}",
                    "source": source,
                    "arm": arm,
                    "slot": slot,
                    "assigned": slot < 6,
                    "eligible": not reasons,
                    "reasons": reasons,
                    "beam": beam,
                    "critical_geometry": bool(critical[slot]),
                    "target_clear_geometry": bool(targets[slot]),
                    "predecision_clearance_m": float(preclear[slot].min()),
                    "visibility_proposal_hits": beam_ray_hits(packet, beam),
                    "suite": "generated",
                }
                rows.append(row)
                if row["assigned"] and row["eligible"]:
                    specs.append(row)
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
                    "seconds": time.monotonic() - start,
                    "trace": artifact(out / f"{source}_{arm}.npz"),
                    "independent_error_m": error,
                }
            )
        for kind in BACKGROUNDS[source]:
            beam = {
                **beam_at(case, 0.6, {"absent": 1.5, "raised": 2.0, "blocked": 0.0}[kind]),
                "thickness_m": 1.4 if kind == "blocked" else 0.1,
            }
            specs.append(
                {
                    "group_id": f"icra_{source}_shared_{kind}",
                    "source": source,
                    "arm": "shared",
                    "beam": beam,
                    "suite": kind,
                    "assigned": True,
                    "eligible": True,
                }
            )
    batches = []
    for bi, start in enumerate(range(0, len(specs), 4)):
        folder = out.with_name(out.name + f"-labels-{bi:02d}")
        folder.mkdir()
        cells = []
        for spec in specs[start : start + 4]:
            scene = physical_scene(
                out / "scenes" / (spec["group_id"] + ".usda"),
                spec["beam"],
                spec["suite"] == "absent",
            )
            for action in (0, 1):
                old = next(
                    c
                    for c in parent["cells"]
                    if c["generation_seed"] == spec["source"]
                    and c["runtime_seed"] == reg["physics_seed"]
                    and c["encounter_action"] == action
                )
                prior = next(r for r in bank["rows"] if r["cell_id"] == old["cell_id"])
                c = copy.deepcopy(old)
                ident = f"{spec['group_id']}_a{action}"
                c.update(
                    cell_id=ident,
                    group_id=spec["group_id"],
                    arm=spec["arm"],
                    suite=spec["suite"],
                    condition="absent" if spec["suite"] == "absent" else "present",
                    beam=spec["beam"],
                    scene=scene,
                    qualified_bank=prior["bank"],
                    role="icra_training_label",
                    output=str(folder / "rollouts" / ident),
                )
                cells.append(c)
        manifest = copy.deepcopy(parent)
        manifest.update(
            cells=cells,
            experiment=f"M2S-ICRA-v1-labels-{bi:02d}",
            registered_predictions=artifact(PROTOCOL),
            purpose="Fixed four-arm paired labels",
        )
        manifest["new_dependencies"] += (
            reg["references"] + [c["scene"] for c in cells] + [c["qualified_bank"] for c in cells]
        )
        manifest["execution_policy"]["cost_ceiling"].update(
            rollouts=len(cells), gpu_hours_contended=len(cells) * 375 / 3600
        )
        write_new(folder / "manifest.json", manifest)
        batches.append(
            {
                "directory": str(folder),
                "manifest": artifact(folder / "manifest.json"),
                "assigned": len(cells),
            }
        )
    write_new(
        out / "proposals.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "costs": costs,
            "backgrounds": [s for s in specs if s["arm"] == "shared"],
            "requested_generated": 72,
            "seconds": time.monotonic() - tick,
        },
    )
    write_new(
        out / "prepared.json",
        {
            "batches": batches,
            "physics_assigned": sum(b["assigned"] for b in batches),
            "proposals": artifact(out / "proposals.json"),
        },
    )
    print(
        json.dumps(
            {
                "eligible_generated": {
                    a: sum(r["assigned"] and r["eligible"] and r["arm"] == a for r in rows)
                    for a in ARMS
                },
                "background_pairs": 6,
                "physics_assigned": len(specs) * 2,
            }
        ),
        flush=True,
    )


def run(out):
    prepared = json.loads((out / "prepared.json").read_text())
    for b in prepared["batches"]:
        folder = Path(b["directory"])
        if (folder / "admission.json").exists():
            assert json.loads((folder / "admission.json").read_text())["admitted"]
            continue
        execute(folder, preflight=True)
        execute(folder)
        if not (folder / "admission.json").exists():
            return
        assert json.loads((folder / "admission.json").read_text())["admitted"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("register", "prepare", "run"))
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    globals()[a.command](a.out)
