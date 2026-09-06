#!/usr/bin/env python3
"""Reference-only grouped test of motion conditioning with varied crouch locations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import time

from motion2scene_beam_teacher import capsules
from motion2scene_inverse_learning import implementations as inverse_implementations, inputs
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import (
    make_clouds,
    numpy_perturbed,
    offset_sets,
    tensor,
    verdict,
)
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import (
    carrier_split,
    shared_origin_pair,
    training_feature_case,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamDomain,
    BeamMixture,
    sample_latents,
    stratified_latents,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_margins import margin_terms
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)
from gear_sonic.dataset_generation.local_adaptation import local_crouch
from gear_sonic.dataset_generation.reference_gate import screen_reference
from gear_sonic.dataset_generation.reference_payload import payload_from_reference

DOMAIN = BeamDomain(station_low=0.10, station_high=0.90)
STATIONS = [0.25, 0.50, 0.75]
MODES = ["conditioned", "constant", "shuffled"]
RECORDS = [
    {"name": "reference_neutral", "label": "neutral", "gradient_input": True},
    {"name": "reference_d055", "label": "d055", "gradient_input": True},
]


def register(args):
    source = json.loads(args.carriers.read_text())
    if [r["generation_seed"] for r in source["base_carriers"]] != list(range(41001, 41009)):
        raise ValueError("requires the eight registered base carriers in seed order")
    refs = [
        artifact(args.carriers),
        artifact(checked(Path(source["mjcf"]["path"]), source["mjcf"]["sha256"])),
    ]
    for row in source["base_carriers"]:
        refs.append(artifact(checked(Path(row["csv"]), row["csv_sha256"])))
    paths = [
        Path(__file__),
        ROOT / "scripts/research/motion2scene_uncertainty_learning.py",
        *[
            ROOT / "gear_sonic/dataset_generation" / name
            for name in (
                "local_adaptation.py",
                "motion_prefilter.py",
                "self_intersection.py",
                "reference_gate.py",
                "hallucination/motion2scene_carriers.py",
                "hallucination/motion2scene_uncertainty.py",
                "hallucination/motion2scene_margins.py",
            )
        ],
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    corners, offsets = offset_sets()
    write_new(
        args.out / "registration.json",
        {
            "schema_version": "motion2scene_reference_carriers_v1",
            "protocol": artifact(args.protocol),
            "sources": refs,
            "implementations": inverse_implementations() + [artifact(p) for p in paths],
            "base_carriers": source["base_carriers"],
            "mjcf": source["mjcf"],
            "splits": {str(seed): carrier_split(seed) for seed in range(41001, 41009)},
            "event_stations": STATIONS,
            "operator_drop_m": 0.055,
            "operator_window": 0.18,
            "domain": asdict(DOMAIN),
            "modes": MODES,
            "seeds": [8221, 8222, 8223],
            "steps": 1200,
            "direct_steps": 300,
            "learning_rate": 0.002,
            "direct_learning_rate": 0.03,
            "samples_per_component": 2,
            "evaluation_samples": 32,
            "training_corners": corners.tolist(),
            "evaluation_offsets": offsets.tolist(),
            "threads": 2,
            "max_stage_seconds": 3600,
            "training_eligible": False,
            "execution_eligible": False,
            "role": "new_reference_only_development_diagnostic_not_original_bank_admission",
        },
    )
    print(json.dumps({"registration": str(args.out / "registration.json")}), flush=True)


def registration(path):
    data = json.loads(path.read_text())
    for ref in [data["protocol"], *data["sources"], *data["implementations"]]:
        checked(Path(ref["path"]), ref["sha256"])
    if data["domain"] != asdict(DOMAIN):
        raise ValueError("domain changed")
    return data


def gate(qpos, name, mjcf):
    result = screen_reference(qpos, name, "walk", mjcf_path=mjcf)
    return {
        "q0_pass": result.embodiment_feasible,
        "q1_pass": result.self_collision_free,
        "reference_semantic_valid": result.reference_semantic_valid,
        "diagnosis": result.diagnosis,
    }


def build(args):
    reg = registration(args.registration)
    root = args.registration.parent
    out = root / "references"
    out.mkdir(exist_ok=False)
    values, rows = {}, []
    started = time.monotonic()
    for carrier in reg["base_carriers"]:
        seed = carrier["generation_seed"]
        neutral = np.loadtxt(carrier["csv"], delimiter=",")
        neutral_gate = gate(neutral, f"{seed}_neutral", reg["mjcf"]["path"])
        for event_index, station in enumerate(reg["event_stations"]):
            if time.monotonic() - started > reg["max_stage_seconds"]:
                raise TimeoutError("reference construction budget exceeded")
            case_id = f"{seed}_event{event_index}"
            target, report = local_crouch(
                neutral,
                station,
                target_drop_m=reg["operator_drop_m"],
                window=reg["operator_window"],
                mjcf_path=reg["mjcf"]["path"],
            )
            if not np.allclose(target[[0, -1]], neutral[[0, -1]], atol=1e-8, rtol=0):
                raise ValueError("event changed the start or final pose")
            first, second = shared_origin_pair(neutral, target)
            target_gate = gate(target, case_id, reg["mjcf"]["path"])
            reference = out / f"{case_id}.csv"
            with reference.open("x") as handle:
                np.savetxt(handle, target, delimiter=",", fmt="%.10f")
            # Build FK from the exact persisted reference, not a hidden higher-precision copy.
            first, second = shared_origin_pair(neutral, np.loadtxt(reference, delimiter=","))
            states = [
                capsules(payload_from_reference(q, fps=30, mjcf_path=reg["mjcf"]["path"]))
                for q in (first, second)
            ]
            values[f"{case_id}_route"] = first[:, :2]
            for i, state in enumerate(states):
                for key in ("starts", "ends", "radii"):
                    values[f"{case_id}_{i}_{key}"] = state[key]
            row = {
                "case_id": case_id,
                "carrier_seed": seed,
                "split": carrier_split(seed),
                "event_station": station,
                "parent": {"path": carrier["csv"], "sha256": carrier["csv_sha256"]},
                "target": artifact(reference),
                "neutral_gate": neutral_gate,
                "target_gate": target_gate,
                "operator": asdict(report),
                "owners": list(states[0]["owners"]),
                "q4_admitted": False,
                "execution_qualified": False,
            }
            rows.append(row)
            print(json.dumps({"case": case_id, "target_gate": target_gate}), flush=True)
    with (root / "bank.npz").open("xb") as handle:
        np.savez_compressed(handle, **values)
    write_new(
        root / "carrier_registry.json",
        {
            "rows": rows,
            "construction_seconds": time.monotonic() - started,
            "source_carriers": 8,
            "target_cases": 24,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    all_pass = all(
        r[g][k]
        for r in rows
        for g in ("neutral_gate", "target_gate")
        for k in ("q0_pass", "q1_pass")
    )
    write_new(
        root / "manifest.json",
        {
            "registration": artifact(args.registration),
            "bank": artifact(root / "bank.npz"),
            "registry": artifact(root / "carrier_registry.json"),
            "reference_gates_pass": all_pass,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    if not all_pass:
        raise ValueError(
            "reference gate failed; registry retained, training stopped without replacements"
        )


def load(path):
    manifest = json.loads(path.read_text())
    for ref in manifest.values():
        if isinstance(ref, dict) and "sha256" in ref:
            checked(Path(ref["path"]), ref["sha256"])
    reg = registration(Path(manifest["registration"]["path"]))
    registry = json.loads(Path(manifest["registry"]["path"]).read_text())
    if not manifest["reference_gates_pass"]:
        raise ValueError("reference bank is not eligible even for this diagnostic")
    bank = np.load(manifest["bank"]["path"], allow_pickle=False)
    cases = {}
    for row in registry["rows"]:
        checked(Path(row["target"]["path"]), row["target"]["sha256"])
        name = row["case_id"]
        states = {
            record["name"]: {
                **{key: bank[f"{name}_{i}_{key}"] for key in ("starts", "ends", "radii")},
                "label": record["label"],
                "owners": row["owners"],
            }
            for i, record in enumerate(RECORDS)
        }
        route = bank[f"{name}_route"]
        feature, _, rt, progress, yaw, mask, costs = inputs(states, route, RECORDS)
        cases[name] = {
            "metadata": row,
            "states": states,
            "route": route,
            "feature": feature,
            "clouds": make_clouds(states),
            "rt": rt,
            "progress": progress,
            "yaw": yaw,
            "mask": mask,
            "costs": costs,
        }
    return reg, cases


def geometric_loss(model, feature, case, reg, proposal_rng, offset_rng):
    latent, weights, _ = stratified_latents(
        model(feature), reg["samples_per_component"], proposal_rng
    )
    scenes = DOMAIN.physical(latent.reshape(-1, 2))
    corners = tensor(reg["training_corners"])
    offsets = torch.cat(
        (
            torch.zeros(1, 4, dtype=torch.float64),
            corners[torch.randperm(16, generator=offset_rng)[:4]],
        )
    )
    values = perturbed_clearances(
        scenes, offsets, case["clouds"], case["rt"], case["progress"], case["yaw"], DOMAIN
    )
    selection, target, upright, _, _ = margin_terms(values, case["costs"], case["mask"])
    return (weights.flatten() * (selection + 5 * target + 5 * upright)).sum()


def fit(model, optimizer, jobs, features, reg, steps, seed, start):
    rng = torch.Generator().manual_seed(seed + 10000)
    pose_rng = torch.Generator().manual_seed(seed + 30000)
    order_rng = torch.Generator().manual_seed(seed + 40000)
    history, order = [], []
    for step in range(steps):
        if time.monotonic() - start > reg["max_stage_seconds"]:
            raise TimeoutError("training stage time budget")
        if step % len(jobs) == 0:
            order = torch.randperm(len(jobs), generator=order_rng).tolist()
        index = order[step % len(jobs)]
        optimizer.zero_grad(set_to_none=True)
        loss = geometric_loss(model, features[index], jobs[index], reg, rng, pose_rng)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if step % 200 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss.item()})
    return history


def save_cell(root, name, model, seed, history, seconds, mode, cases):
    cell = root / name
    cell.mkdir()
    with (cell / "checkpoint.pt").open("xb") as handle:
        torch.save({"model": model.state_dict(), "mode": mode, "seed": seed}, handle)
    predictions = []
    with torch.no_grad():
        for case_id, case in cases.items():
            feature = case["feature"]
            if mode == "constant":
                feature = torch.zeros_like(feature)
            scenes = DOMAIN.physical(
                sample_latents(model(feature), 256, torch.Generator().manual_seed(seed + 20000))
            ).numpy()[:32]
            predictions.append({"case_id": case_id, "scenes": scenes.tolist()})
            if mode == "conditioned":
                source_case = (
                    case_id.rsplit("event", 1)[0] + "event" + str((int(case_id[-1]) + 1) % 3)
                )
                swapped = DOMAIN.physical(
                    sample_latents(
                        model(cases[source_case]["feature"]),
                        256,
                        torch.Generator().manual_seed(seed + 20000),
                    )
                ).numpy()[:32]
                predictions.append(
                    {
                        "case_id": case_id,
                        "mode": "swapped_input",
                        "input_case": source_case,
                        "scenes": swapped.tolist(),
                    }
                )
    write_new(cell / "samples.json", {"mode": mode, "seed": seed, "predictions": predictions})
    write_new(
        cell / "result.json",
        {
            "mode": mode,
            "seed": seed,
            "checkpoint": artifact(cell / "checkpoint.pt"),
            "samples": artifact(cell / "samples.json"),
            "history": history,
            "training_seconds": seconds,
        },
    )
    print(json.dumps({"cell": name, "training_seconds": seconds, "last": history[-1]}), flush=True)


def train(args):
    reg, cases = load(args.manifest)
    torch.set_num_threads(reg["threads"])
    torch.use_deterministic_algorithms(True)
    train_ids = [k for k, v in cases.items() if v["metadata"]["split"] == "train"]
    test_cases = {k: v for k, v in cases.items() if v["metadata"]["split"] != "train"}
    smoke = args.command == "smoke"
    seeds = reg["seeds"][:1] if smoke else reg["seeds"]
    steps = 12 if smoke else reg["steps"]
    root = args.manifest.parent / ("smoke" if smoke else "runs")
    root.mkdir(exist_ok=False)
    write_new(
        root / "started.json",
        {
            "manifest": artifact(args.manifest),
            "role": "engineering_smoke" if smoke else "registered_run",
        },
    )
    start = time.monotonic()
    try:
        for mode, seed in product(reg["modes"], seeds):
            torch.manual_seed(seed)
            model = BeamMixture(29).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=reg["learning_rate"])
            features = []
            for case_id in train_ids:
                feature_id = training_feature_case(mode, case_id, train_ids)
                features.append(
                    torch.zeros_like(cases[case_id]["feature"])
                    if feature_id is None
                    else cases[feature_id]["feature"]
                )
            cell_start = time.monotonic()
            history = fit(
                model, optimizer, [cases[k] for k in train_ids], features, reg, steps, seed, start
            )
            save_cell(
                root,
                f"{mode}_{seed}",
                model,
                seed,
                history,
                time.monotonic() - cell_start,
                mode,
                test_cases,
            )
        direct_cases = dict(list(test_cases.items())[:1]) if smoke else test_cases
        for (case_id, case), seed in product(direct_cases.items(), seeds):
            torch.manual_seed(seed)
            model = BeamMixture(29, motion_conditioned=False).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=reg["direct_learning_rate"])
            cell_start = time.monotonic()
            history = fit(
                model,
                optimizer,
                [case],
                [case["feature"]],
                reg,
                2 if smoke else reg["direct_steps"],
                seed,
                start,
            )
            save_cell(
                root,
                f"direct_{seed}_{case_id}",
                model,
                seed,
                history,
                time.monotonic() - cell_start,
                "direct",
                {case_id: case},
            )
    except Exception as exc:
        write_new(root / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    write_new(
        root / "complete.json",
        {
            "manifest": artifact(args.manifest),
            "elapsed_s": time.monotonic() - start,
            "cells": [artifact(p) for p in sorted(root.glob("*/result.json"))],
        },
    )


def analyze(args):
    reg, cases = load(args.manifest)
    root = args.manifest.parent
    completion_path = root / "runs/complete.json"
    completion = json.loads(completion_path.read_text())
    checked(args.manifest, completion["manifest"]["sha256"])
    if len(completion["cells"]) != 45:
        raise ValueError("requires 9 amortized and 36 per-case optimization cells")
    torch.set_num_threads(reg["threads"])
    predictions = []
    for ref in completion["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ("checkpoint", "samples"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        samples = json.loads(Path(cell["samples"]["path"]).read_text())
        predictions.extend(
            {**p, "mode": p.get("mode", cell["mode"]), "seed": cell["seed"], "source": ref}
            for p in samples["predictions"]
        )
    for case_id, case in cases.items():
        if case["metadata"]["split"] == "train":
            continue
        for seed in reg["seeds"]:
            latent = torch.randn(
                256, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 20000)
            )
            predictions.append(
                {
                    "mode": "random",
                    "seed": seed,
                    "case_id": case_id,
                    "scenes": DOMAIN.physical(latent).numpy()[:32].tolist(),
                }
            )
    offsets = np.array(reg["evaluation_offsets"])
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    rows = []
    start = time.monotonic()
    for prediction in predictions:
        if time.monotonic() - start > reg["max_stage_seconds"]:
            raise TimeoutError("evaluation stage time budget")
        case = cases[prediction["case_id"]]
        scenes = np.array(prediction["scenes"])
        values = numpy_perturbed(scenes, offsets, case["states"], case["route"], case["yaw"].item())
        with torch.no_grad():
            other = torch.cat(
                [
                    perturbed_clearances(
                        tensor(scenes[:4]),
                        tensor(chunk),
                        case["clouds"],
                        case["rt"],
                        case["progress"],
                        case["yaw"],
                        DOMAIN,
                    )
                    for chunk in np.array_split(offsets, 23)
                ],
                dim=1,
            ).numpy()
        error = float(np.abs(values[:4] - other).max())
        if error > 1e-8:
            raise ValueError("NumPy/Torch geometry mismatch")
        valid = verdict(values, np.array([False, True]))
        target, upright = values[..., 1].min(1), values[..., 0].max(1)
        name = f'{prediction["mode"]}_{prediction["seed"]}_{prediction["case_id"]}'
        path = out / f"{name}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
        row = {
            **prediction,
            "split": case["metadata"]["split"],
            "carrier_seed": case["metadata"]["carrier_seed"],
            "event_station": case["metadata"]["event_station"],
            "nominal_valid": int(valid[:, 0].sum()),
            "all_113_valid": int(valid.all(1).sum()),
            "target_clear": int((target >= 0.01).sum()),
            "upright_interference": int((upright <= -0.01).sum()),
            "sample_count": len(scenes),
            "per_proposal_valid": valid.all(1).tolist(),
            "query_error_m": error,
            "raw": artifact(path),
        }
        rows.append(row)
        print(
            json.dumps({k: row[k] for k in ("mode", "seed", "case_id", "all_113_valid")}),
            flush=True,
        )
    summary = []
    for mode, seed in product(MODES + ["swapped_input", "direct", "random"], range(41005, 41009)):
        selected = [r for r in rows if r["mode"] == mode and r["carrier_seed"] == seed]
        assert len(selected) == 9
        summary.append(
            {
                "mode": mode,
                "carrier_seed": seed,
                "split": carrier_split(seed),
                "valid": sum(r["all_113_valid"] for r in selected),
                "proposals": sum(r["sample_count"] for r in selected),
            }
        )

    def score(mode, seed):
        return next(r["valid"] for r in summary if r["mode"] == mode and r["carrier_seed"] == seed)

    write_new(
        root / "result.json",
        {
            "manifest": artifact(args.manifest),
            "training_completion": artifact(completion_path),
            "analysis_complete": True,
            "rows": rows,
            "summary": summary,
            "prediction_conditioned_beats_constant_and_shuffled_on_both_test_carriers": all(
                score("conditioned", seed) > max(score("constant", seed), score("shuffled", seed))
                for seed in (41007, 41008)
            ),
            "prediction_swapped_input_lowers_validity_on_both_test_carriers": all(
                score("conditioned", seed) > score("swapped_input", seed) for seed in (41007, 41008)
            ),
            "evaluation_seconds": time.monotonic() - start,
            "training_eligible": False,
            "execution_eligible": False,
            "reference_only": True,
            "test_source_carriers": 2,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("register")
    for name in ("carriers", "protocol", "out"):
        a.add_argument(f"--{name}", type=Path, required=True)
    a = commands.add_parser("build")
    a.add_argument("--registration", type=Path, required=True)
    for command in ("smoke", "train", "analyze"):
        a = commands.add_parser(command)
        a.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    {"register": register, "build": build, "smoke": train, "train": train, "analyze": analyze}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
