#!/usr/bin/env python3
"""Finite witness map to distinguish proposal reach from robust geometric support."""

import argparse
import json
from pathlib import Path
import time

from bundle_motion2scene_sources import closure
from motion2scene_development_bank import DATA
from motion2scene_distill_study import query_for
from motion2scene_source_execution import beam_at
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_transition_construction import BANK, cases
from motion2scene_uncertainty_learning import numpy_perturbed, tensor
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_beam_visibility import beam_ray_hits
from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    reserved_layout,
)

PILOT = DATA / "m2s-transition-construction-v1"
PROTOCOL = ROOT / "docs/motion2scene/TRANSITION_SUPPORT_DIAGNOSTIC_V1.md"


def run(out):
    refs = [
        artifact(PROTOCOL),
        artifact(PILOT / "proposals.json"),
        artifact(PILOT / "registration.json"),
        artifact(BANK / "result.json"),
        artifact(DATA / "m2s-refinement-v1/registration.json"),
    ]
    refs += [artifact(p) for p in sorted(closure([Path(__file__)]))]
    out.mkdir(exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "cpu_ceiling_s": 300,
            "role": "post hoc finite development map, no new placements admitted",
        },
    )
    for r in refs:
        checked(Path(r["path"]), r["sha256"])
    start = time.monotonic()
    torch.set_num_threads(2)
    bank = json.loads((BANK / "result.json").read_text())
    reg = json.loads((PILOT / "registration.json").read_text())
    offsets = json.loads((DATA / "m2s-refinement-v1/registration.json").read_text())
    grid = np.array(
        [[s, h] for s in 0.12 + np.arange(20) * 0.04 for h in 1.10 + np.arange(71) * 0.005]
    )
    rows = []
    for source in reg["requested_sources"]:
        if str(source) not in reg["qualified_sources"]:
            continue
        case, _, packet, _ = cases(source, bank)
        nominal = numpy_perturbed(
            grid, np.zeros((1, 4)), case["states"], case["route"], case["yaw"].item()
        )
        values = []
        with torch.no_grad():
            query = query_for(case, tensor(offsets["search_offsets"]))
            for chunk in np.array_split(grid, 40):
                if time.monotonic() - start > 300:
                    raise TimeoutError("support-map CPU ceiling")
                values.append(query(tensor(chunk)).numpy())
        search = np.concatenate(values)

        def slack(v):
            return np.minimum(v[:, :, 1].min(1) - 0.01, -v[:, :, 0].max(1) - 0.01)

        ns, ss = slack(nominal), slack(search)
        ids = np.flatnonzero(ss >= 0)
        audit = np.full(len(grid), np.nan)
        if len(ids):
            v = numpy_perturbed(
                grid[ids],
                np.array(offsets["audit_offsets"]),
                case["states"],
                case["route"],
                case["yaw"].item(),
            )
            audit[ids] = slack(v)
        pre = {
            k: {**v, "starts": v["starts"][:16], "ends": v["ends"][:16]}
            for k, v in case["states"].items()
        }
        pc = numpy_perturbed(grid, np.zeros((1, 4)), pre, case["route"], case["yaw"].item())[
            :, 0, :
        ].min(1)
        visible = np.array(
            [bool(beam_ray_hits(packet, {**beam_at(case, *p), "thickness_m": 0.1})) for p in grid]
        )
        reserved = np.array([reserved_layout(*p, reg["layouts"]) for p in grid])
        with np.load(PILOT / f"{source}_motion2scene.npz") as f:
            initial = f["initial"]
        reachable = np.any(
            np.all(
                abs(grid[:, None, :] - initial[None, :, :]) <= [0.05 + 1e-12, 0.03 + 1e-12], axis=2
            ),
            axis=1,
        )
        path = out / f"{source}.npz"
        np.savez_compressed(
            path,
            grid=grid,
            nominal=nominal,
            search=search,
            nominal_slack=ns,
            search_slack=ss,
            audit_slack=audit,
            visible=visible,
            reserved=reserved,
            predecision_clearance=pc,
            learned_reachable=reachable,
        )
        background = visible & ~reserved & (pc >= 0.01)
        rows.append(
            {
                "source": source,
                "grid_centres": len(grid),
                "nominal_witnesses": int((ns >= 0).sum()),
                "search_witnesses": len(ids),
                "audit_witnesses": int((audit >= 0).sum()),
                "nominal_visible_unreserved_preclear": int(((ns >= 0) & background).sum()),
                "audit_visible_unreserved_preclear": int(((audit >= 0) & background).sum()),
                "nominal_reachable_by_original_draws": int(((ns >= 0) & reachable).sum()),
                "max_nominal_slack_m": float(ns.max()),
                "max_search_slack_m": float(ss.max()),
                "arrays": artifact(path),
            }
        )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "cpu_seconds": time.monotonic() - start,
            "physics_executions": 0,
            "fitted_models": 0,
            "scope": "Finite witness counts, not feasible volume or physical labels",
        },
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.out)
