#!/usr/bin/env python3
"""Register explicit scene-margin training and audit every proposal independently."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import time

from motion2scene_inverse_learning import DOMAIN, inputs
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import (
    load as load_uncertainty,
    make_clouds,
    numpy_perturbed,
    tensor,
    verdict,
)
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamMixture,
    sample_latents,
    stratified_latents,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_margins import margin_terms
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)

ARMS = {
    "margin_full": {"preference": 1.0, "kl": 0.02},
    "margin_no_kl": {"preference": 1.0, "kl": 0.0},
    "constraints_full": {"preference": 0.0, "kl": 0.02},
    "constraints_no_kl": {"preference": 0.0, "kl": 0.0},
}


def prepare(args):
    previous = json.loads(args.previous.read_text())
    if not previous["analysis_complete"]:
        raise ValueError("requires completed uncertainty experiment")
    ref = previous["manifest"]
    parent_path = checked(Path(ref["path"]), ref["sha256"])
    parent, _, _, _ = load_uncertainty(parent_path)
    baseline_refs = []
    for row in previous["rows"]:
        if row["arm"].startswith("robust_"):
            baseline_refs.append(row["raw"])
            cell = parent_path.parent / f'runs/{row["arm"]}_{row["seed"]}/result.json'
            saved = json.loads(cell.read_text())
            baseline_refs.extend([artifact(cell), saved["checkpoint"], saved["samples"]])
    for ref in baseline_refs:
        checked(Path(ref["path"]), ref["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        key: parent[key]
        for key in (
            "seeds",
            "samples_per_component",
            "training_corner_count",
            "training_corners",
            "evaluation_offsets",
            "evaluation_samples",
            "learning_rate",
            "feasibility_weight",
            "threads",
            "dtype",
            "device",
            "torch_version",
            "numpy_version",
        )
    }
    manifest.update(
        {
            "schema_version": "motion2scene_margin_v1",
            "role": args.role,
            "parent_manifest": artifact(parent_path),
            "previous_result": artifact(args.previous),
            "baseline_artifacts": baseline_refs,
            "protocol": artifact(args.protocol),
            "implementations": [
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_uncertainty_learning.py"),
                artifact(
                    ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_margins.py"
                ),
            ],
            "arms": ARMS,
            "steps": args.steps,
            "interference_weight": 5.0,
            "margin_m": 0.01,
            "max_run_seconds": 3600,
            "training_eligible": False,
            "execution_eligible": False,
        }
    )
    write_new(args.out / "manifest.json", manifest)
    print(json.dumps({"manifest": str(args.out / "manifest.json")}), flush=True)


def load(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for ref in [
        manifest["parent_manifest"],
        manifest["previous_result"],
        manifest["protocol"],
        *manifest["implementations"],
        *manifest["baseline_artifacts"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    _, source, states, route = load_uncertainty(Path(manifest["parent_manifest"]["path"]))
    return manifest, source, states, route


def run(args):
    manifest, source, states, route = load(args.manifest)
    torch.set_num_threads(manifest["threads"])
    torch.use_deterministic_algorithms(True)
    feature, _, rt, progress, yaw, mask, costs = inputs(states, route, source["records"])
    clouds = [
        cloud
        for cloud, record in zip(make_clouds(states), source["records"])
        if record["gradient_input"]
    ]
    root = args.manifest.parent / "runs"
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    for arm, seed in product(manifest["arms"], manifest["seeds"]):
        cell = root / f"{arm}_{seed}"
        cell.mkdir()
        write_new(cell / "started.json", {"manifest": artifact(args.manifest)})
        torch.manual_seed(seed)
        proposal_rng = torch.Generator().manual_seed(seed + 10000)
        offset_rng = torch.Generator().manual_seed(seed + 30000)
        model = BeamMixture(feature.shape[1]).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=manifest["learning_rate"])
        corners = tensor(manifest["training_corners"])
        history = []
        cell_start = time.monotonic()
        try:
            for step in range(manifest["steps"]):
                if time.monotonic() - started > manifest["max_run_seconds"]:
                    raise TimeoutError("registered training time budget exceeded")
                optimizer.zero_grad(set_to_none=True)
                parameters = model(feature)
                latent, weights, kl = stratified_latents(
                    parameters, manifest["samples_per_component"], proposal_rng
                )
                scenes = DOMAIN.physical(latent.reshape(-1, 2))
                chosen = torch.randperm(len(corners), generator=offset_rng)[
                    : manifest["training_corner_count"]
                ]
                offsets = torch.cat((torch.zeros(1, 4, dtype=torch.float64), corners[chosen]))
                select, feasible, interference, _, _ = margin_terms(
                    perturbed_clearances(scenes, offsets, clouds, rt, progress, yaw, DOMAIN),
                    costs,
                    mask,
                )
                loss = (
                    weights.flatten()
                    * (
                        manifest["arms"][arm]["preference"] * select
                        + manifest["feasibility_weight"] * feasible
                        + manifest["interference_weight"] * interference
                        + manifest["arms"][arm]["kl"] * kl.flatten()
                    )
                ).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                if step % 50 == 0 or step == manifest["steps"] - 1:
                    row = {
                        "step": step,
                        "loss": loss.item(),
                        "selection": (weights.flatten() * select).sum().item(),
                        "feasibility": (weights.flatten() * feasible).sum().item(),
                        "interference": (weights.flatten() * interference).sum().item(),
                        "kl_mc": (weights * kl).sum().item(),
                        "elapsed_s": time.monotonic() - cell_start,
                    }
                    history.append(row)
                    print(json.dumps({"cell": cell.name, **row}), flush=True)
            with torch.no_grad():
                parameters = model(feature)
                # Draw the same 256-sized batch as V1, then evaluate the fixed first 64.
                scenes = DOMAIN.physical(
                    sample_latents(parameters, 256, torch.Generator().manual_seed(seed + 20000))
                ).numpy()[: manifest["evaluation_samples"]]
            with (cell / "checkpoint.pt").open("xb") as handle:
                torch.save({"model": model.state_dict(), "arm": arm, "seed": seed}, handle)
            write_new(cell / "samples.json", {"scenes": scenes.tolist(), "history": history})
            write_new(
                cell / "result.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": artifact(cell / "checkpoint.pt"),
                    "samples": artifact(cell / "samples.json"),
                    "training_seconds": time.monotonic() - cell_start,
                },
            )
        except Exception as exc:
            write_new(cell / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
            raise
    write_new(
        root / "complete.json",
        {
            "manifest": artifact(args.manifest),
            "elapsed_s": time.monotonic() - started,
            "cells": [artifact(p) for p in sorted(root.glob("*/result.json"))],
        },
    )


def summarize(values, scenes, source):
    target_mask = np.array([r["label"] == "d055" for r in source["records"]])
    gradient = np.array([r["gradient_input"] for r in source["records"]])
    valid = verdict(values, target_mask)
    train_valid = verdict(values[..., gradient], target_mask[gradient])
    target = values[..., target_mask].min(axis=(1, 2))
    upright = values[..., ~target_mask].max(axis=(1, 2))
    accepted = valid.all(-1)
    bins = np.floor((scenes[accepted] - [0.35, 1.10]) / [0.30, 0.35] * [30, 70]).astype(int)
    return {
        "sample_count": len(scenes),
        "nominal_valid": int(valid[:, 0].sum()),
        "grid_81_valid": int(valid[:, :81].all(-1).sum()),
        "interior_32_valid": int(valid[:, 81:].all(-1).sum()),
        "combined_113_valid": int(accepted.sum()),
        "gradient_sources_113_valid": int(train_valid.all(-1).sum()),
        "target_clear_113": int((target >= 0.01).sum()),
        "upright_interference_113": int((upright <= -0.01).sum()),
        "failure_counts": {
            "target_only": int(((target < 0.01) & (upright <= -0.01)).sum()),
            "upright_only": int(((target >= 0.01) & (upright > -0.01)).sum()),
            "both": int(((target < 0.01) & (upright > -0.01)).sum()),
        },
        "occupied_accepted_bins": len(np.unique(bins, axis=0)),
        "per_proposal_113_valid": accepted.tolist(),
        "scenes": scenes.tolist(),
        "worst_target_clearance_m": target.tolist(),
        "worst_upright_clearance_m": upright.tolist(),
    }


def analyze(args):
    manifest, source, states, route = load(args.manifest)
    root = args.manifest.parent
    completion_path = root / "runs/complete.json"
    completion = json.loads(completion_path.read_text())
    checked(args.manifest, completion["manifest"]["sha256"])
    expected = len(manifest["arms"]) * len(manifest["seeds"])
    if len(completion["cells"]) != expected:
        raise ValueError("incomplete training cells")
    cells = []
    for ref in completion["cells"]:
        saved = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ("checkpoint", "samples"):
            checked(Path(saved[key]["path"]), saved[key]["sha256"])
        cells.append(saved)
    previous = json.loads(Path(manifest["previous_result"]["path"]).read_text())
    rows = []
    for row in previous["rows"]:
        if row["arm"].startswith("robust_"):
            ref = row["raw"]
            raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
            np.testing.assert_array_equal(raw["offsets"], manifest["evaluation_offsets"])
            summary = summarize(raw["clearance_m"], raw["scenes"], source)
            if summary["combined_113_valid"] != row["combined_113_valid"]:
                raise ValueError("baseline raw evidence disagrees")
            rows.append({**row, **summary, "evaluation_reused": True})
    torch.set_num_threads(manifest["threads"])
    _, _, rt, progress, yaw, _, _ = inputs(states, route, source["records"])
    clouds = make_clouds(states)
    offsets = np.array(manifest["evaluation_offsets"])
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    for cell in cells:
        if time.monotonic() - started > manifest["max_run_seconds"]:
            raise TimeoutError("registered evaluation budget exceeded")
        scenes = np.array(json.loads(Path(cell["samples"]["path"]).read_text())["scenes"])
        values = numpy_perturbed(scenes, offsets, states, route, yaw.item())
        with torch.no_grad():
            torch_values = torch.cat(
                [
                    perturbed_clearances(
                        tensor(scenes[:8]), tensor(chunk), clouds, rt, progress, yaw, DOMAIN
                    )
                    for chunk in np.array_split(offsets, 23)
                ],
                dim=1,
            ).numpy()
        error = float(np.abs(torch_values - values[:8]).max())
        if error > 1e-8:
            raise ValueError(f"independent geometry mismatch: {error}")
        raw_path = out / f'{cell["arm"]}_{cell["seed"]}.npz'
        with raw_path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearance_m=values)
        row = {
            **cell,
            **summarize(values, scenes, source),
            "raw": artifact(raw_path),
            "independent_query_max_error_m": error,
            "evaluation_reused": False,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "arm",
                        "seed",
                        "nominal_valid",
                        "combined_113_valid",
                        "target_clear_113",
                        "upright_interference_113",
                        "occupied_accepted_bins",
                        "failure_counts",
                    )
                }
            ),
            flush=True,
        )
    comparisons = []
    for family, seed in product(("full", "no_kl"), manifest["seeds"]):
        old = next(r for r in rows if r["arm"] == f"robust_{family}" and r["seed"] == seed)
        new = next(r for r in rows if r["arm"] == f"margin_{family}" and r["seed"] == seed)
        comparisons.append(
            {
                "family": family,
                "seed": seed,
                "valid_count_change": new["combined_113_valid"] - old["combined_113_valid"],
                "target_clear_count_change": new["target_clear_113"] - old["target_clear_113"],
            }
        )
    write_new(
        root / "result.json",
        {
            "schema_version": "motion2scene_margin_result_v1",
            "analysis_complete": True,
            "manifest": artifact(args.manifest),
            "training_completion": artifact(completion_path),
            "rows": rows,
            "comparisons": comparisons,
            "prediction_both_families_improve_at_least_two_seeds": all(
                sum(c["valid_count_change"] > 0 for c in comparisons if c["family"] == f) >= 2
                for f in ("full", "no_kl")
            ),
            "training_seconds": sum(c["training_seconds"] for c in cells),
            "evaluation_seconds": time.monotonic() - started,
            "independent_carriers": 1,
            "training_eligible": False,
            "execution_eligible": False,
            "continuous_placement_certified": False,
            "continuous_time_certified": False,
            "imported_robot_geometry_certified": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--role", default="registered_one_carrier_margin_development")
    for name in ("run", "analyze"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.steps <= 0:
        parser.error("steps must be positive")
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
