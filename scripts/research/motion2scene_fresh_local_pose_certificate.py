#!/usr/bin/env python3
"""Certify local full-pose neighborhoods of every frozen fresh-audit pattern output."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

from motion2scene_fresh_sources import load_cases
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_local_pose_certificate import (
    local_pose_certificate,
    rotvec_matrix,
)

HALF = np.array([0.05, 0.6, 0.05])


def query(states, center, rotation):
    values, witnesses = [], []
    for state in states.values():
        starts = (state["starts"] - center) @ rotation
        ends = (state["ends"] - center) @ rotation
        radii = np.broadcast_to(state["radii"], starts.shape[:-1])
        reach = radii[..., None] + 0.02
        keep = np.all(np.minimum(starts, ends) - reach <= HALF, -1) & np.all(
            np.maximum(starts, ends) + reach >= -HALF, -1
        )
        value, witness = 0.02, (-1, -1)
        if keep.any():
            scores = capsule_box_clearance(starts[keep], ends[keep], radii[keep], -HALF, HALF)
            index = int(np.argmin(scores))
            if scores[index] < 0.02:
                value = float(scores[index])
                witness = tuple(int(indices[index]) for indices in np.where(keep))
        values.append(value)
        witnesses.append(witness)
    return np.array(values), witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = ROOT.parent / "research-data/groot-wbc/m2s-fresh-source-v1"
    result_path = source / "result.json"
    result = json.loads(result_path.read_text())
    ref = result["registration"]
    checked(Path(ref["path"]), ref["sha256"])
    jobs = [r for r in result["rows"] if r["arm"] == "pattern"]
    keys = {(r["seed"], r["case_id"]) for r in jobs}
    if (
        len(jobs) != 48
        or len(keys) != 48
        or any(r["valid"] != 8 or len(r["scenes"]) != 8 for r in jobs)
    ):
        raise ValueError("requires all 48 original eight-output accepted pattern jobs")
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/FRESH_LOCAL_POSE_CERTIFICATE_V1.md"),
            "source_result": artifact(result_path),
            "source_registration": ref,
            "raw_jobs": [r["raw"] for r in jobs],
            "implementations": [
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_fresh_sources.py"),
                artifact(
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_local_pose_certificate.py"
                ),
                artifact(ROOT / "gear_sonic/dataset_generation/capsule_box_exact.py"),
            ],
            "requested_outputs": 384,
            "half_extents_m": HALF.tolist(),
            "margin_m": 0.01,
            "numerical_allowance_m": 1e-8,
            "training_or_method_selection": False,
        },
    )
    start = time.monotonic()
    _, _, cases = load_cases(Path(ref["path"]))
    rng = np.random.default_rng(6941)
    rows, queries = [], 0
    for job in jobs:
        raw_ref = job["raw"]
        with np.load(checked(Path(raw_ref["path"]), raw_ref["sha256"]), allow_pickle=False) as raw:
            scenes = raw["scenes"].copy()
            zero = np.flatnonzero((raw["offsets"] == 0).all(1))
            if len(zero) != 1 or not np.array_equal(scenes, job["scenes"]):
                raise ValueError("missing nominal audit or changed scene coordinates")
            saved_nominal = raw["clearances"][:, zero[0]].copy()
        case = cases[job["case_id"]]
        mask = case["mask"].numpy().astype(bool)
        arc = np.r_[0, np.linalg.norm(np.diff(case["route"], axis=0), axis=-1).cumsum()]
        arc /= arc[-1]
        yaw = case["yaw"].item()
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for i, (station, height) in enumerate(scenes):
            if time.monotonic() - start > 600:
                raise TimeoutError("local pose certificate CPU budget exhausted")
            center = np.array(
                [np.interp(station, arc, case["route"][:, j]) for j in range(2)] + [height + 0.05]
            )
            nominal, _ = query(case["states"], center, rotation)
            queries += len(nominal)
            cert = local_pose_certificate(
                float(nominal[mask].min()), float(nominal[~mask].max()), HALF
            )
            scores, witnesses, perturbations = [], [], np.empty((0, 6))
            if cert["status"] == "conditional_pass":
                scale = np.array(
                    [cert["translation_halfwidth_m"]] * 3
                    + [cert["rotvec_component_halfwidth_rad"]] * 3
                )
                perturbations = (
                    np.r_[np.array(list(product((-1, 1), repeat=6))), rng.uniform(-1, 1, (8, 6))]
                    * scale
                )
                for perturbation in perturbations:
                    values, witness = query(
                        case["states"],
                        center + perturbation[:3],
                        rotvec_matrix(perturbation[3:]) @ rotation,
                    )
                    scores.append(values)
                    witnesses.append(witness)
                    queries += len(values)
            scores = np.array(scores).reshape(-1, len(nominal))
            pass_mask = (
                (scores[:, mask].min(-1) >= 0.01) & (scores[:, ~mask].max(-1) <= -0.01)
                if len(scores)
                else np.zeros(0, dtype=bool)
            )
            path = args.out / f"{job['seed']}_{job['case_id']}_{i}.npz"
            np.savez_compressed(
                path,
                nominal=nominal,
                perturbations=perturbations,
                scores=scores,
                witnesses=witnesses,
            )
            rows.append(
                {
                    "source": job["carrier_seed"],
                    "case_id": job["case_id"],
                    "seed": job["seed"],
                    "output_index": i,
                    "scene": [float(station), float(height)],
                    "certificate": cert,
                    "nominal_reproduction_error_m": float(np.abs(nominal - saved_nominal[i]).max()),
                    "diagnostic_probes": len(scores),
                    "passing_probes": int(pass_mask.sum()),
                    "raw": artifact(path),
                }
            )
        print(f"checked {len(rows)}/384", flush=True)
    summaries = {}
    for source_id in sorted({r["source"] for r in rows}):
        source_rows = [r for r in rows if r["source"] == source_id]
        passing = [
            r["certificate"]
            for r in source_rows
            if r["certificate"]["status"] == "conditional_pass"
        ]
        summaries[str(source_id)] = {
            "requested": len(source_rows),
            "conditional_pass": len(passing),
            "translation_halfwidth_range_m": (
                [
                    min(r["translation_halfwidth_m"] for r in passing),
                    max(r["translation_halfwidth_m"] for r in passing),
                ]
                if passing
                else None
            ),
            "rotvec_halfwidth_range_rad": (
                [
                    min(r["rotvec_component_halfwidth_rad"] for r in passing),
                    max(r["rotvec_component_halfwidth_rad"] for r in passing),
                ]
                if passing
                else None
            ),
            "chart_volume_range_m3_rad3": (
                [
                    min(r["chart_volume_m3_rad3"] for r in passing),
                    max(r["chart_volume_m3_rad3"] for r in passing),
                ]
                if passing
                else None
            ),
        }
    write_new(
        args.out / "result.json",
        {
            "registration": artifact(args.out / "registration.json"),
            "rows": rows,
            "sources": summaries,
            "predictions": {
                "p1": all(r["nominal_reproduction_error_m"] <= 1e-6 for r in rows),
                "p2": len(rows) == 384
                and all(r["certificate"]["status"] == "conditional_pass" for r in rows),
                "p3": all(r["diagnostic_probes"] == r["passing_probes"] == 72 for r in rows),
            },
            "whole_motion_queries": queries,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_seconds": 0,
            "entire_original_uncertainty_domain_certified": False,
            "native_body_certified": False,
            "continuous_time_certified": False,
        },
    )


if __name__ == "__main__":
    main()
