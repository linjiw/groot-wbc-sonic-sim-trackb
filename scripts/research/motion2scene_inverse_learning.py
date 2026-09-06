#!/usr/bin/env python3
"""Register, train, and independently evaluate a motion-only inverse beam diagnostic.

CPU-only. One selected carrier; two execution seeds in gradients, a third excluded
from gradients. This is sampled capsule geometry, not qualified scene data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
import math
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

from motion2scene_beam_teacher import capsules, jitter_audit
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

sys.path.insert(0, str(ROOT))
from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (  # noqa: E402
    BeamDomain,
    BeamMixture,
    domain_capsules,
    inverse_terms,
    sample_latents,
    scene_clearances,
    stratified_latents,
)
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload  # noqa: E402

DOMAIN = BeamDomain()
ARMS = {
    "full": {"selection_weight": 1.0, "kl_weight": 0.02, "conditioned": True},
    "no_kl": {"selection_weight": 1.0, "kl_weight": 0.0, "conditioned": True},
    "clearance_only": {"selection_weight": 0.0, "kl_weight": 0.02, "conditioned": True},
    "per_motion": {"selection_weight": 1.0, "kl_weight": 0.02, "conditioned": False},
}


def implementations():
    paths = [
        Path(__file__),
        ROOT / "scripts/research/motion2scene_beam_teacher.py",
        ROOT / "scripts/research/motion2scene_timing_diagnostic.py",
        *[
            ROOT / "gear_sonic/dataset_generation" / name
            for name in (
                "capsule_box_exact.py",
                "capsule_box_torch.py",
                "swept_volume.py",
                "reference_payload.py",
                "trajectory_segments.py",
                "hallucination/motion2scene_inverse.py",
            )
        ],
    ]
    return [artifact(path) for path in paths]


def read_sources(repeatability):
    report = json.loads(repeatability.read_text())
    if not report["analysis_complete"] or not report["summary"]["strict_development_repeatability"]:
        raise ValueError("requires the complete repeated development pair")
    source = report["manifest"]
    manifest = json.loads(checked(Path(source["path"]), source["sha256"]).read_text())
    states, sources, route = {}, [artifact(repeatability), source], None
    for cell in manifest["cells"]:
        label = cell["label"]
        key = f"reference_{label}"
        if key in states:
            continue
        ref = cell["reference"]
        path = checked(Path(ref["path"]), ref["sha256"])
        provenance = artifact(
            checked(
                Path(cell["motion"]["conversion_provenance"]),
                cell["motion"]["conversion_provenance_sha256"],
            )
        )
        conversion = json.loads(Path(provenance["path"]).read_text())
        if conversion["scene_start_xyz"] != [0, 0, 0] or conversion.get("scene_yaw", 0) != 0:
            raise ValueError("requires the registered shared world frame")
        qpos = np.loadtxt(path, delimiter=",")
        qpos[:, :2] -= qpos[0, :2]
        states[key] = {**capsules(payload_from_reference(qpos, fps=30)), "label": label}
        sources.extend((ref, provenance))
        if label == "neutral":
            route = qpos[:, :2]
    for row in report["rows"]:
        ref = row["trajectory"]
        with checked(Path(ref["path"]), ref["sha256"]).open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        states[row["cell_id"]] = {**capsules(payload), "label": row["label"]}
        sources.append(ref)
    return states, sources, route


def geometry_inventory(states):
    urdf = ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
    tree = ET.parse(urdf)
    inventory = []
    for link in tree.getroot().findall("link"):
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            shape = list(geometry)[0]
            origin = collision.find("origin")
            inventory.append(
                {
                    "link": link.attrib["name"],
                    "type": shape.tag,
                    "shape": shape.attrib,
                    "origin": {} if origin is None else origin.attrib,
                }
            )
    owners = set(next(iter(states.values()))["owners"])
    return {
        "urdf": artifact(urdf),
        "robot_configuration": artifact(ROOT / "gear_sonic/envs/manager_env/robots/g1.py"),
        "urdf_collision_count": len(inventory),
        "shape_counts": dict(Counter(x["type"] for x in inventory)),
        "links_with_urdf_collision_but_no_diagnostic_capsule": sorted(
            {x["link"] for x in inventory} - owners
        ),
        "inventory": inventory,
        "imported_shape_equivalence_established": False,
        "outer_enclosure_certified": False,
        "inner_intersection_certified": False,
        "reason": (
            "URDF inventory is audited, but importer conversions and mesh enclosure are not "
            "certified; all experiment verdicts remain capsule-proxy geometry"
        ),
    }


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    states, sources, route = read_sources(args.repeatability)
    values = {"route": route}
    records = []
    for index, (name, state) in enumerate(states.items()):
        for key in ("starts", "ends", "radii"):
            values[f"{index}_{key}"] = state[key]
        records.append(
            {
                "name": name,
                "label": state["label"],
                "index": index,
                "gradient_input": "_7903__" not in name,
                "owners": list(state["owners"]),
                "frame_count": state["starts"].shape[0],
                "capsule_count": len(state["radii"]),
            }
        )
    with (args.out / "bank.npz").open("xb") as handle:
        np.savez_compressed(handle, **values)
    write_new(args.out / "geometry_inventory.json", geometry_inventory(states))
    manifest = {
        "schema_version": "motion2scene_inverse_diagnostic_v1",
        "role": args.role,
        "bank": artifact(args.out / "bank.npz"),
        "sources": sources,
        "records": records,
        "implementations": implementations(),
        "geometry_inventory": artifact(args.out / "geometry_inventory.json"),
        "protocol": artifact(args.protocol),
        "domain": asdict(DOMAIN),
        "arms": ARMS,
        "seeds": [8121, 8122, 8123],
        "steps": args.steps,
        "samples_per_component": 2,
        "evaluation_samples": 256,
        "learning_rate": 0.002,
        "feasibility_weight": 5.0,
        "threads": 2,
        "device": "cpu",
        "dtype": "float64",
        "max_run_seconds": 1800,
        "margin_m": 0.01,
        "grid": {"station_bins": 30, "height_bins": 70},
        "training_eligible": False,
        "execution_eligible": False,
        "split_note": (
            "One selected development carrier. Seed 7903 excluded from gradients, previously "
            "observed in other development studies; not a confirmatory holdout or independent carrier."
        ),
        "geometry_note": (
            "All recorded frames and capsule axes; fixed abstract beam without support structure. "
            "No actual-robot or continuous-time certificate."
        ),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    write_new(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.out / "manifest.json"),
                "records": len(records),
                "steps": args.steps,
            }
        ),
        flush=True,
    )


def load_bank(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for ref in [
        manifest["bank"],
        manifest["protocol"],
        *manifest["implementations"],
        *manifest["sources"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    if manifest["domain"] != asdict(DOMAIN):
        raise ValueError("domain changed after registration")
    data = np.load(manifest["bank"]["path"], allow_pickle=False)
    states = {}
    for record in manifest["records"]:
        state = {key: data[f'{record["index"]}_{key}'] for key in ("starts", "ends", "radii")}
        states[record["name"]] = {**state, "label": record["label"], "owners": record["owners"]}
    return manifest, states, data["route"]


def inputs(states, route, records):
    def to_tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    target = states["reference_d055"]
    starts, ends = target["starts"], target["ends"]
    feature = to_tensor(
        np.concatenate(
            (
                (starts + ends) / 2,
                ends - starts,
                np.broadcast_to(target["radii"][None, :, None], starts.shape[:2] + (1,)),
            ),
            axis=-1,
        )
    )
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(length) > 0]
    route_tensor, progress = to_tensor(route[keep]), to_tensor(length[keep] / length[-1])
    yaw = to_tensor(math.atan2(*(route[-1] - route[0])[::-1]))
    training = [record for record in records if record["gradient_input"]]
    clouds = [
        domain_capsules(
            {key: to_tensor(states[record["name"]][key]) for key in ("starts", "ends", "radii")},
            DOMAIN,
        )
        for record in training
    ]
    mask = torch.tensor([record["label"] == "d055" for record in training])
    costs = mask.to(
        torch.float64
    )  # frozen semantic cost: walk=0, crouch=1; independent of loss label
    return feature, clouds, route_tensor, progress, yaw, mask, costs


def numpy_clearances(scenes, states, route, yaw):
    """Independent finite-box evaluation, retaining all recorded frames and capsules.

    Only proposal-specific horizontal broad phase is used; omitted pairs have
    separation at least clearance_cap. No trained model or teacher interval is read.
    """
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
    arc /= arc[-1]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    values = np.empty((len(scenes), len(states)))
    for i, (station, height) in enumerate(scenes):
        center = np.array(
            [np.interp(station, arc, route[:, axis]) for axis in range(2)]
            + [height + DOMAIN.thickness / 2]
        )
        half = np.array([DOMAIN.depth, DOMAIN.width, DOMAIN.thickness]) / 2
        for j, state in enumerate(states.values()):
            starts = (state["starts"] - center) @ rotation
            ends = (state["ends"] - center) @ rotation
            radii = np.broadcast_to(state["radii"], starts.shape[:-1])
            reach = radii[..., None] + DOMAIN.clearance_cap
            keep = np.all(np.minimum(starts, ends) - reach <= half, axis=-1) & np.all(
                np.maximum(starts, ends) + reach >= -half, axis=-1
            )
            values[i, j] = (
                DOMAIN.clearance_cap
                if not keep.any()
                else min(
                    DOMAIN.clearance_cap,
                    float(
                        capsule_box_clearance(
                            starts[keep], ends[keep], radii[keep], -half, half
                        ).min()
                    ),
                )
            )
    return values


def evaluate(scenes, states, route, yaw, records):
    values = numpy_clearances(scenes, states, route, yaw)
    target_mask = np.array([s["label"] == "d055" for s in states.values()])
    gradient_mask = np.array([r["gradient_input"] for r in records])
    target, weaker = values[:, target_mask].min(-1), values[:, ~target_mask].max(-1)
    train_target = values[:, target_mask & gradient_mask].min(-1)
    train_weaker = values[:, ~target_mask & gradient_mask].max(-1)
    return {
        "sample_count": len(scenes),
        "scenes": scenes.tolist(),
        "per_source_clearance_m": values.tolist(),
        "target_min_m": target.tolist(),
        "weaker_max_m": weaker.tolist(),
        "valid": ((target >= 0.01) & (weaker <= -0.01)).tolist(),
        "target_clear": (target >= 0.01).tolist(),
        "gradient_sources_valid": ((train_target >= 0.01) & (train_weaker <= -0.01)).tolist(),
        "gradient_source_margin_m": np.minimum(train_target - 0.01, -0.01 - train_weaker).tolist(),
    }


def run(args):
    manifest, states, route = load_bank(args.manifest)
    torch.set_num_threads(manifest["threads"])
    torch.use_deterministic_algorithms(True)
    feature, clouds, rt, progress, yaw, mask, costs = inputs(states, route, manifest["records"])
    all_clouds = [
        domain_capsules(
            {
                key: torch.as_tensor(state[key], dtype=torch.float64)
                for key in ("starts", "ends", "radii")
            },
            DOMAIN,
        )
        for state in states.values()
    ]
    root = args.manifest.parent / "runs"
    root.mkdir(exist_ok=False)
    start = time.monotonic()
    for arm, seed in product(manifest["arms"], manifest["seeds"]):
        spec = manifest["arms"][arm]
        cell = root / f"{arm}_{seed}"
        cell.mkdir()
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 10000)
        model = BeamMixture(feature.shape[1], motion_conditioned=spec["conditioned"]).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=manifest["learning_rate"])
        history = []
        cell_start = time.monotonic()
        write_new(
            cell / "started.json", {"manifest": artifact(args.manifest), "arm": arm, "seed": seed}
        )
        try:
            for step in range(manifest["steps"]):
                if time.monotonic() - start > manifest["max_run_seconds"]:
                    raise TimeoutError("registered total run wall-clock limit")
                optimizer.zero_grad(set_to_none=True)
                parameters = model(feature)
                latent, weights, kl = stratified_latents(
                    parameters, manifest["samples_per_component"], generator
                )
                scenes = DOMAIN.physical(latent.reshape(-1, 2))
                clearances = scene_clearances(scenes, clouds, rt, progress, yaw, DOMAIN)
                select, feasibility, _, _ = inverse_terms(clearances, costs, mask)
                loss = (
                    weights.flatten()
                    * (
                        spec["selection_weight"] * select
                        + manifest["feasibility_weight"] * feasibility
                        + spec["kl_weight"] * kl.flatten()
                    )
                ).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                if not all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise FloatingPointError("nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                if step % 50 == 0 or step == manifest["steps"] - 1:
                    row = {
                        "step": step,
                        "loss": loss.item(),
                        "selection": (weights.flatten() * select).sum().item(),
                        "feasibility": (weights.flatten() * feasibility).sum().item(),
                        "kl_mc": (weights * kl).sum().item(),
                        "elapsed_s": time.monotonic() - cell_start,
                    }
                    history.append(row)
                    print(json.dumps({"cell": cell.name, **row}), flush=True)
            with torch.no_grad():
                parameters = model(feature)
                eval_rng = torch.Generator().manual_seed(seed + 20000)
                samples = DOMAIN.physical(
                    sample_latents(parameters, manifest["evaluation_samples"], eval_rng)
                ).numpy()
                density = {
                    "weights": parameters[0].softmax(-1).tolist(),
                    "latent_means": parameters[1].tolist(),
                    "latent_cholesky": parameters[2].tolist(),
                }
            training_seconds = time.monotonic() - cell_start
            # Freeze checkpoint/samples before excluded-recording evaluation.
            with (cell / "checkpoint.pt").open("xb") as handle:
                torch.save({"model": model.state_dict(), "arm": arm, "seed": seed}, handle)
            write_new(
                cell / "samples.json",
                {"density": density, "scenes": samples.tolist(), "history": history},
            )
            result = evaluate(samples, states, route, yaw.item(), manifest["records"])
            with torch.no_grad():
                torch_values = torch.cat(
                    [
                        scene_clearances(chunk, all_clouds, rt, progress, yaw, DOMAIN)
                        for chunk in torch.as_tensor(samples).split(16)
                    ]
                ).numpy()
            oracle_error = float(np.max(np.abs(torch_values - result["per_source_clearance_m"])))
            if oracle_error > 1e-8:
                raise ValueError(f"PyTorch/NumPy query disagreement: {oracle_error} m")
            result.update(
                {
                    "arm": arm,
                    "seed": seed,
                    "training_seconds": training_seconds,
                    "independent_query_max_error_m": oracle_error,
                    "density": density,
                    "history": history,
                    "checkpoint": artifact(cell / "checkpoint.pt"),
                    "samples_artifact": artifact(cell / "samples.json"),
                    "training_eligible": False,
                }
            )
            write_new(cell / "result.json", result)
            print(
                json.dumps(
                    {"cell": cell.name, "valid": sum(result["valid"]), "samples": len(samples)}
                ),
                flush=True,
            )
        except Exception as exc:
            write_new(
                cell / "failure.json",
                {"error": str(exc), "type": type(exc).__name__, "history": history},
            )
            raise
    write_new(
        root / "complete.json",
        {
            "manifest": artifact(args.manifest),
            "expected_cells": len(ARMS) * len(manifest["seeds"]),
            "elapsed_s": time.monotonic() - start,
        },
    )


def analyze(args):
    manifest, states, route = load_bank(args.manifest)
    root = args.manifest.parent
    completion = json.loads((root / "runs/complete.json").read_text())
    checked(args.manifest, completion["manifest"]["sha256"])
    yaw = math.atan2(*(route[-1] - route[0])[::-1])
    rows = []
    for arm, seed in product(manifest["arms"], manifest["seeds"]):
        path = root / f"runs/{arm}_{seed}/result.json"
        result = json.loads(path.read_text())
        checked(Path(result["checkpoint"]["path"]), result["checkpoint"]["sha256"])
        rows.append({**result, "artifact": artifact(path)})
    for seed in manifest["seeds"]:
        rng = np.random.default_rng(seed + 20000)
        latent = torch.as_tensor(
            rng.normal(size=(manifest["evaluation_samples"], 2)), dtype=torch.float64
        )
        result = evaluate(DOMAIN.physical(latent).numpy(), states, route, yaw, manifest["records"])
        rows.append({**result, "arm": "random_prior", "seed": seed})
    # Independent midpoint grid, evaluated only after every model is frozen.
    ns, nh = manifest["grid"]["station_bins"], manifest["grid"]["height_bins"]
    stations = DOMAIN.station_low + (np.arange(ns) + 0.5) / ns * (
        DOMAIN.station_high - DOMAIN.station_low
    )
    heights = DOMAIN.height_low + (np.arange(nh) + 0.5) / nh * (
        DOMAIN.height_high - DOMAIN.height_low
    )
    grid_scenes = np.array(list(product(stations, heights)))
    grid = evaluate(grid_scenes, states, route, yaw, manifest["records"])
    grid_valid = np.asarray(grid["valid"]).reshape(ns, nh)
    grid_training_valid = np.asarray(grid["gradient_sources_valid"])
    # Oracle sampler uses gradient-source grid only, never excluded records for selection.
    for seed in manifest["seeds"]:
        if not grid_training_valid.any():
            rows.append(
                {
                    "arm": "grid_oracle",
                    "seed": seed,
                    "sample_count": 0,
                    "refusal": "no gradient-source feasible grid centers",
                    "valid": [],
                    "target_clear": [],
                    "gradient_sources_valid": [],
                }
            )
            continue
        rng = np.random.default_rng(seed + 20000)
        samples = grid_scenes[
            rng.choice(
                np.flatnonzero(grid_training_valid), manifest["evaluation_samples"], replace=True
            )
        ]
        rows.append(
            {
                **evaluate(samples, states, route, yaw, manifest["records"]),
                "arm": "grid_oracle",
                "seed": seed,
            }
        )
    for row in rows:
        row["valid_count"] = sum(row["valid"])
        row["target_clear_count"] = sum(row["target_clear"])
        row["gradient_sources_valid_count"] = sum(row["gradient_sources_valid"])
        occupied = set()
        for scene, valid in zip(row.get("scenes", []), row["valid"]):
            si = min(
                ns - 1,
                int(
                    (scene[0] - DOMAIN.station_low)
                    / (DOMAIN.station_high - DOMAIN.station_low)
                    * ns
                ),
            )
            hi = min(
                nh - 1,
                int((scene[1] - DOMAIN.height_low) / (DOMAIN.height_high - DOMAIN.height_low) * nh),
            )
            if valid and grid_valid[si, hi]:
                occupied.add((si, hi))
        row["valid_grid_center_bins_covered"] = len(occupied)
        # Freeze representative by best gradient-source margin; no post-hoc search on 7903.
        if row.get("scenes"):
            chosen = int(np.argmax(row["gradient_source_margin_m"]))
            station, height = row["scenes"][chosen]
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
            arc /= arc[-1]
            candidate = {
                "center_xy_m": [
                    float(np.interp(station, arc, route[:, axis])) for axis in range(2)
                ],
                "yaw_rad": yaw,
                "length_m": DOMAIN.depth,
                "width_m": DOMAIN.width,
            }
            row["representative"] = {
                "sample_index": chosen,
                "scene": [station, height],
                "selection": "maximum gradient-source minimum margin",
                "jitter_audit": jitter_audit(states, candidate, height),
            }
    summaries = [
        {
            key: row[key]
            for key in (
                "arm",
                "seed",
                "sample_count",
                "valid_count",
                "target_clear_count",
                "gradient_sources_valid_count",
                "valid_grid_center_bins_covered",
            )
        }
        for row in rows
    ]
    result = {
        "schema_version": "motion2scene_inverse_result_v1",
        "manifest": artifact(args.manifest),
        "complete": True,
        "rows": rows,
        "summaries": summaries,
        "grid": grid,
        "grid_valid_centers": int(grid_valid.sum()),
        "grid_gradient_valid_centers": int(grid_training_valid.sum()),
        "training_eligible": False,
        "physics_rollouts": 0,
        "limits": [
            manifest["split_note"],
            manifest["geometry_note"],
            "grid-center coverage is resolution dependent, not exact continuous support coverage",
            "grid oracle has a separate 2100-query search cost; it is not a matched-compute learner",
        ],
    }
    write_new(root / "result.json", result)
    print(
        json.dumps(
            {"grid_valid_centers": result["grid_valid_centers"], "summaries": summaries}, indent=2
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repeatability", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--steps", type=int, default=300)
    prep.add_argument("--role", default="registered_one_carrier_development")
    for command in ("run", "analyze"):
        sub.add_parser(command).add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
