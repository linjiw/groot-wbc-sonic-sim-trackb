#!/usr/bin/env python3
"""Bounded, independently checked teacher maps and event-aware analytic comparison."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_distill_study import DATA, independent, load, query_for
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import tensor, verdict
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_coverage import (
    analytic_proposals,
    reference_coverage,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    parent_path = DATA / "m2s-distillation-v1/registration.json"
    reg, parent, train, dev = load(parent_path)
    teacher_path = parent_path.parent / "teacher/complete.json"
    result_path = parent_path.parent / "result.json"
    teacher = json.loads(teacher_path.read_text())
    old = json.loads(result_path.read_text())
    teachers = {r["case_id"]: r for r in teacher["rows"]}
    grid = np.array(
        [
            (0.1 + (s + 0.5) * 0.04, 1.1 + (h + 0.5) * 0.35 / 18)
            for s in range(20)
            for h in range(18)
        ]
    )
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/COUNTERFACTUAL_STAGE_V1.md"),
            "driver": artifact(Path(__file__)),
            "helper": artifact(
                ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_coverage.py"
            ),
            "parent": artifact(parent_path),
            "teacher": artifact(teacher_path),
            "existing_outputs": artifact(result_path),
            "training_ids": list(train),
            "development_ids": list(dev),
            "grid": grid.tolist(),
            "cap_seconds": 1800,
            "threads": 2,
            "gpu_seconds": 0,
            "excluded_sources": reg["excluded_source_ids"],
            "reference_maps_are_certified_cells": False,
            "query_unit": "one whole-motion clearance at one beam placement",
        },
    )
    start = time.monotonic()
    maps, comparisons, analytic = [], [], []
    # Analytic outputs are frozen BEFORE independent maps are constructed.
    for name, case in dev.items():
        tick = time.monotonic()
        initial, envelopes = analytic_proposals(
            case["states"],
            case["route"],
            case["yaw"].item(),
            depth=DOMAIN.depth,
            width=DOMAIN.width,
        )
        preprocessing = time.monotonic() - tick
        tick = time.monotonic()
        output, trace = station_search(
            tensor(initial), query_for(case, tensor(parent["search_offsets"])), "pattern"
        )
        search_seconds = time.monotonic() - tick
        tick = time.monotonic()
        values, error = independent(output.numpy(), case, np.array(parent["audit_offsets"]))
        audit_seconds = time.monotonic() - tick
        accepted = verdict(values, case["mask"].numpy()).all(1)
        raw_path = args.out / f"analytic_{name}.npz"
        np.savez_compressed(
            raw_path,
            **{k: v.numpy() for k, v in trace.items()},
            output=output.numpy(),
            audit_clearances=values,
        )
        row = {
            "case_id": name,
            "carrier_seed": case["metadata"]["carrier_seed"],
            "scenes": output.tolist(),
            "accepted": accepted.tolist(),
            "raw": artifact(raw_path),
            "envelopes": envelopes,
            "preprocessing_seconds": preprocessing,
            "search_seconds": search_seconds,
            "audit_seconds": audit_seconds,
            "query_error_m": error,
            "search_queries": 4624,
            "audit_queries": 1808,
            "crosscheck_queries": 452,
        }
        analytic.append(row)
    write_new(args.out / "analytic.json", {"rows": analytic})
    for role, cases in [("development", dev), ("train", train)]:
        for name, case in cases.items():
            if time.monotonic() - start > 1800:
                raise TimeoutError("diagnostic cap; completed per-case artifacts retained")
            tick = time.monotonic()
            chunks, errors = [], []
            for batch in np.array_split(grid, 45):
                values, error = independent(batch, case, np.array(parent["audit_offsets"]))
                chunks.append(values)
                errors.append(error)
            values = np.concatenate(chunks)
            passing = verdict(values, case["mask"].numpy()).all(1)
            raw_path = args.out / f"map_{name}.npz"
            np.savez_compressed(raw_path, scenes=grid, clearances=values, passing=passing)
            row = {
                "case_id": name,
                "role": role,
                "carrier_seed": case["metadata"]["carrier_seed"],
                "passing_centres": int(passing.sum()),
                "centres": len(grid),
                "raw": artifact(raw_path),
                "query_error_m": max(errors),
                "seconds": time.monotonic() - tick,
                "audit_queries": len(grid) * 113 * 2,
                "crosscheck_queries": 45 * 2 * 113 * 2,
            }
            if role == "train":
                teacher_row = teachers[name]
                checked(Path(teacher_row["trace"]["path"]), teacher_row["trace"]["sha256"])
                targets = np.array(teacher_row["targets"]).reshape(-1, 2)
                row["teacher_support"] = reference_coverage(
                    grid, passing, targets, np.ones(len(targets), dtype=bool)
                )
            else:
                for previous in old["rows"]:
                    if previous["case_id"] != name:
                        continue
                    checked(Path(previous["raw"]["path"]), previous["raw"]["sha256"])
                    comparisons.append(
                        {
                            "case_id": name,
                            "carrier_seed": row["carrier_seed"],
                            "arm": previous["arm"],
                            "seed": previous["seed"],
                            **reference_coverage(
                                grid,
                                passing,
                                previous["scenes"],
                                np.array(previous["per_proposal_valid"], dtype=bool),
                            ),
                        }
                    )
                a = next(a for a in analytic if a["case_id"] == name)
                a["coverage"] = reference_coverage(
                    grid, passing, a["scenes"], np.array(a["accepted"], dtype=bool)
                )
            maps.append(row)
            write_new(args.out / f"map_{name}.json", row)
            print(
                json.dumps(
                    {
                        "case": name,
                        "role": role,
                        "passing": row["passing_centres"],
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
    result = {
        "registration": artifact(args.out / "registration.json"),
        "maps": maps,
        "comparisons": comparisons,
        "analytic": analytic,
        "elapsed_seconds": time.monotonic() - start,
        "gpu_seconds": 0,
        "execution_eligible": False,
        "map_audit_queries": sum(r["audit_queries"] for r in maps),
        "map_crosscheck_queries": sum(r["crosscheck_queries"] for r in maps),
    }
    write_new(args.out / "result.json", result)


if __name__ == "__main__":
    main()
