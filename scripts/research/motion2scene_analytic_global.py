#!/usr/bin/env python3
"""Registered global analytic development solver at the learned pattern query budget."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_distill_study import DATA, independent, load, query_for
from motion2scene_timing_diagnostic import ROOT, artifact, write_new
from motion2scene_uncertainty_learning import tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_analytic_global import global_search
from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import (
    analytic_proposals,
    reference_coverage,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    parent_path = DATA / "m2s-distillation-v1/registration.json"
    _, parent, _, cases = load(parent_path)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/ANALYTIC_GLOBAL_V1.md"),
            "driver": artifact(Path(__file__)),
            "helper": artifact(
                ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_analytic_global.py"
            ),
            "envelope_helper": artifact(
                ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_coverage.py"
            ),
            "parent": artifact(parent_path),
            "case_ids": list(cases),
            "queries_per_job": 6884,
            "cap_seconds": 180,
            "map_role": "already inspected development maps; never supplied to solver",
            "geometry_only": True,
            "gpu_seconds": 0,
        },
    )
    start = time.monotonic()
    rows = []
    for name, case in cases.items():
        if time.monotonic() - start > 180:
            raise TimeoutError("global baseline time cap")
        tick = time.monotonic()
        ranked, envelopes = analytic_proposals(
            case["states"],
            case["route"],
            case["yaw"].item(),
            depth=DOMAIN.depth,
            width=DOMAIN.width,
            count=20,
        )
        nominal = np.array([[r["station"], r["height"]] for r in envelopes])
        ranking = np.array([int(np.argmin(abs(nominal[:, 0] - s))) for s in ranked[:, 0]])
        preprocessing = time.monotonic() - tick
        query = query_for(case, tensor(parent["search_offsets"]))
        tick = time.monotonic()
        with torch.no_grad():
            output, trace = global_search(
                nominal, ranking, lambda scenes: query(tensor(scenes)).numpy()
            )
        search = time.monotonic() - tick
        tick = time.monotonic()
        values, error = independent(output, case, np.array(parent["audit_offsets"]))
        audit = time.monotonic() - tick
        accepted = verdict(values, case["mask"].numpy()).all(1)
        path = args.out / f"{name}.npz"
        np.savez_compressed(path, **trace, output=output, audit_clearances=values)
        row = {
            "case_id": name,
            "carrier_seed": case["metadata"]["carrier_seed"],
            "scenes": output.tolist(),
            "accepted": accepted.tolist(),
            "raw": artifact(path),
            "passing_search_stations": int(trace["passing_station_count"]),
            "unique_outputs": len(np.unique(output, axis=0)),
            "preprocessing_seconds": preprocessing,
            "search_seconds": search,
            "audit_seconds": audit,
            "search_queries": 4624,
            "audit_queries": 1808,
            "crosscheck_queries": 452,
            "query_error_m": error,
        }
        rows.append(row)
        write_new(args.out / f"{name}.json", row)
        print(
            json.dumps(
                {
                    "case": name,
                    "accepted": int(accepted.sum()),
                    "passing_stations": row["passing_search_stations"],
                }
            ),
            flush=True,
        )
    # Output freeze precedes reference-map reads. Maps do not alter selected outputs.
    write_new(args.out / "frozen_outputs.json", {"rows": rows})
    maps_path = DATA / "m2s-coverage-diagnostic-v1/result.json"
    maps = json.loads(maps_path.read_text())
    for row in rows:
        ref = next(r for r in maps["maps"] if r["case_id"] == row["case_id"])
        grid = np.load(ref["raw"]["path"])
        row["coverage"] = reference_coverage(
            grid["scenes"], grid["passing"], row["scenes"], np.array(row["accepted"], dtype=bool)
        )
    accepted = sum(sum(r["accepted"]) for r in rows)
    coverage = float(np.mean([r["coverage"]["reference_relative_station_coverage"] for r in rows]))
    original = [r for r in maps["comparisons"] if r["arm"] == "original_pattern17"]
    original_coverage = float(np.mean([r["reference_relative_station_coverage"] for r in original]))
    result = {
        "registration": artifact(args.out / "registration.json"),
        "maps": artifact(maps_path),
        "rows": rows,
        "accepted": accepted,
        "requested": 128,
        "mean_reference_station_coverage": coverage,
        "original_pattern17_reference_station_coverage": original_coverage,
        "prediction_match_yield_increase_coverage": accepted == 128
        and coverage > original_coverage,
        "elapsed_seconds": time.monotonic() - start,
        "gpu_seconds": 0,
    }
    write_new(args.out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
