#!/usr/bin/env python3
"""Register and run a CPU placement-uncertainty experiment on a frozen motion bank."""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
import time

from motion2scene_inverse_learning import DOMAIN, inputs, load_bank
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamMixture,
    sample_latents,
    stratified_latents,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
    uncertainty_capsules,
    uncertainty_terms,
)

LIMITS = np.array([0.02, 0.02, 0.01, 0.02])


def offset_sets():
    corners = np.array(list(product((-1, 1), repeat=4))) * LIMITS
    grid = np.array(list(product((-1, 0, 1), repeat=4))) * LIMITS
    grid = np.vstack((np.zeros((1, 4)), grid[np.any(grid != 0, axis=1)]))
    interior = np.random.default_rng(92301).uniform(-1, 1, (32, 4)) * LIMITS
    return corners, np.vstack((grid, interior))


def prepare(args):
    source, _, _ = load_bank(args.source)
    baseline_root = args.source.parent / "runs"
    completion = json.loads((baseline_root / "complete.json").read_text())
    checked(args.source, completion["manifest"]["sha256"])
    baselines = []
    for arm, seed in product(("full", "no_kl"), source["seeds"]):
        path = baseline_root / f"{arm}_{seed}" / "result.json"
        result = json.loads(path.read_text())
        refs = [artifact(path), result["samples_artifact"], result["checkpoint"]]
        for ref in refs:
            checked(Path(ref["path"]), ref["sha256"])
        baselines.append({"arm": f"nominal_{arm}", "seed": seed, "artifacts": refs})
    timing_path = args.source.parent.parent / "m2s-timing-diagnostic-v1/result.json"
    repeat_path = args.source.parent.parent / "m2s-repeatability-v1/result.json"
    audit_sources, carrier_rows = [], []
    for path in (timing_path, repeat_path):
        result = json.loads(path.read_text())
        for key in ("manifest", "run_record"):
            checked(Path(result[key]["path"]), result[key]["sha256"])
        audit_sources.append(artifact(path))
        if path == timing_path:
            carrier_rows = [
                {
                    "generation_seed": row["generation_seed"],
                    "condition": row["condition"],
                    "tracker_accepted": row["tracker_accepted"],
                    "route_retained": row["route_retained"],
                    "paired_behavior_retained": row["paired_behavior_retained"],
                }
                for row in result["rows"]
                if row["label"] == "d055"
            ]
        else:
            repeat_summary = result["summary"]
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "carrier_audit.json",
        {
            "scope": "Existing timing and repeated-pair studies, not a repository-wide census",
            "sources": audit_sources,
            "timing_crouch_rows": carrier_rows,
            "repeatability_summary": repeat_summary,
            "independent_repeated_development_carriers": repeat_summary["independent_carriers"],
            "q4_admitted_ladders": repeat_summary["q4_admitted_ladders"],
            "independent_carrier_generalization_evaluable": False,
        },
    )
    corners, evaluation = offset_sets()
    write_new(
        args.out / "manifest.json",
        {
            "schema_version": "motion2scene_uncertainty_v1",
            "role": args.role,
            "source_bank_manifest": artifact(args.source),
            "protocol": artifact(args.protocol),
            "implementations": [
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_inverse_learning.py"),
                artifact(
                    ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_uncertainty.py"
                ),
            ],
            "baselines": baselines,
            "carrier_audit": artifact(args.out / "carrier_audit.json"),
            "arms": {"robust_full": 0.02, "robust_no_kl": 0.0},
            "seeds": source["seeds"],
            "steps": args.steps,
            "samples_per_component": 2,
            "training_corner_count": 4,
            "training_corners": corners.tolist(),
            "evaluation_offsets": evaluation.tolist(),
            "evaluation_samples": 64,
            "learning_rate": 0.002,
            "feasibility_weight": 5.0,
            "threads": 2,
            "max_run_seconds": 3600,
            "dtype": "float64",
            "device": "cpu",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json")}), flush=True)


def load(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    refs = [
        manifest["source_bank_manifest"],
        manifest["protocol"],
        manifest["carrier_audit"],
        *manifest["implementations"],
        *[ref for baseline in manifest["baselines"] for ref in baseline["artifacts"]],
    ]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    source, states, route = load_bank(Path(manifest["source_bank_manifest"]["path"]))
    return manifest, source, states, route


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def make_clouds(states):
    return [
        uncertainty_capsules(
            {key: tensor(state[key]) for key in ("starts", "ends", "radii")}, DOMAIN, LIMITS[2]
        )
        for state in states.values()
    ]


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
                select, feasible, _, _ = uncertainty_terms(
                    perturbed_clearances(scenes, offsets, clouds, rt, progress, yaw, DOMAIN),
                    costs,
                    mask,
                )
                loss = (
                    weights.flatten()
                    * (
                        select
                        + manifest["feasibility_weight"] * feasible
                        + manifest["arms"][arm] * kl.flatten()
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
        {"manifest": artifact(args.manifest), "elapsed_s": time.monotonic() - started},
    )


def numpy_perturbed(scenes, offsets, states, route, yaw):
    """Independent NumPy full-axis oracle with per-pose broad phase, all frames."""
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
    arc /= arc[-1]
    half = np.array([DOMAIN.depth, DOMAIN.width, DOMAIN.thickness]) / 2
    values = np.empty((len(scenes), len(offsets), len(states)))
    for k, offset in enumerate(offsets):
        angle = yaw + offset[3]
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for i, (station, height) in enumerate(scenes):
            center = (
                np.array(
                    [np.interp(station, arc, route[:, a]) for a in range(2)]
                    + [height + DOMAIN.thickness / 2]
                )
                + offset[:3]
            )
            for j, state in enumerate(states.values()):
                starts = (state["starts"] - center) @ rotation
                ends = (state["ends"] - center) @ rotation
                radii = np.broadcast_to(state["radii"], starts.shape[:-1])
                reach = radii[..., None] + DOMAIN.clearance_cap
                keep = np.all(np.minimum(starts, ends) - reach <= half, -1) & np.all(
                    np.maximum(starts, ends) + reach >= -half, -1
                )
                values[i, k, j] = (
                    min(
                        DOMAIN.clearance_cap,
                        float(
                            capsule_box_clearance(
                                starts[keep], ends[keep], radii[keep], -half, half
                            ).min()
                        ),
                    )
                    if keep.any()
                    else DOMAIN.clearance_cap
                )
    return values


def verdict(values, target_mask):
    return (values[..., target_mask].min(-1) >= 0.01) & (values[..., ~target_mask].max(-1) <= -0.01)


def analyze(args):
    manifest, source, states, route = load(args.manifest)
    completion_path = args.manifest.parent / "runs/complete.json"
    completion = json.loads(completion_path.read_text())
    checked(args.manifest, completion["manifest"]["sha256"])
    torch.set_num_threads(manifest["threads"])
    _, _, rt, progress, yaw, _, _ = inputs(states, route, source["records"])
    clouds = make_clouds(states)
    offsets = np.array(manifest["evaluation_offsets"])
    cells = []
    for baseline in manifest["baselines"]:
        samples = json.loads(Path(baseline["artifacts"][1]["path"]).read_text())["scenes"]
        cells.append({**baseline, "scenes": samples[: manifest["evaluation_samples"]]})
    training_seconds = 0.0
    for arm, seed in product(manifest["arms"], manifest["seeds"]):
        path = args.manifest.parent / f"runs/{arm}_{seed}/result.json"
        result = json.loads(path.read_text())
        for key in ("samples", "checkpoint"):
            checked(Path(result[key]["path"]), result[key]["sha256"])
        scenes = json.loads(Path(result["samples"]["path"]).read_text())["scenes"]
        cells.append({**result, "scenes": scenes, "artifacts": [artifact(path)]})
        training_seconds += result["training_seconds"]
    out = args.manifest.parent / "evaluation"
    out.mkdir(exist_ok=False)
    target = np.array([record["label"] == "d055" for record in source["records"]])
    gradient = np.array([record["gradient_input"] for record in source["records"]])
    rows = []
    started = time.monotonic()
    for cell in cells:
        if time.monotonic() - started > manifest["max_run_seconds"]:
            raise TimeoutError("registered evaluation time budget exceeded")
        scenes = np.array(cell["scenes"])
        values = numpy_perturbed(scenes, offsets, states, route, yaw.item())
        # Check the first eight proposals at every evaluation offset independently.
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
        all_valid = verdict(values, target)
        train_valid = verdict(values[..., gradient], target[gradient])
        target_clear = values[..., target].min(-1) >= 0.01
        name = f'{cell["arm"]}_{cell["seed"]}'
        raw_path = out / f"{name}.npz"
        with raw_path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearance_m=values)
        row = {
            "arm": cell["arm"],
            "seed": cell["seed"],
            "sample_count": len(scenes),
            "nominal_valid": int(all_valid[:, 0].sum()),
            "grid_81_valid": int(all_valid[:, :81].all(-1).sum()),
            "interior_32_valid": int(all_valid[:, 81:].all(-1).sum()),
            "combined_113_valid": int(all_valid.all(-1).sum()),
            "gradient_sources_113_valid": int(train_valid.all(-1).sum()),
            "target_clear_113": int(target_clear.all(-1).sum()),
            "per_proposal_113_valid": all_valid.all(-1).tolist(),
            "independent_query_max_error_m": error,
            "raw": artifact(raw_path),
            "sources": cell["artifacts"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("raw", "sources", "per_proposal_113_valid")
                }
            ),
            flush=True,
        )
    comparisons = []
    for family, seed in product(("full", "no_kl"), manifest["seeds"]):
        nominal = next(r for r in rows if r["arm"] == f"nominal_{family}" and r["seed"] == seed)
        robust = next(r for r in rows if r["arm"] == f"robust_{family}" and r["seed"] == seed)
        comparisons.append(
            {
                "family": family,
                "seed": seed,
                "valid_count_change": robust["combined_113_valid"] - nominal["combined_113_valid"],
            }
        )
    write_new(
        args.manifest.parent / "result.json",
        {
            "schema_version": "motion2scene_uncertainty_result_v1",
            "analysis_complete": True,
            "manifest": artifact(args.manifest),
            "training_completion": artifact(completion_path),
            "rows": rows,
            "comparisons": comparisons,
            "training_seconds": training_seconds,
            "evaluation_seconds": time.monotonic() - started,
            "prediction_both_families_improve_at_least_two_seeds": all(
                sum(c["valid_count_change"] > 0 for c in comparisons if c["family"] == family) >= 2
                for family in ("full", "no_kl")
            ),
            "independent_carriers": 1,
            "continuous_time_certified": False,
            "continuous_placement_certified": False,
            "imported_robot_geometry_certified": False,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--role", default="registered_one_carrier_uncertainty_development")
    for name in ("run", "analyze"):
        p = commands.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
