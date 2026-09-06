#!/usr/bin/env python3
"""Registered reference-only representation, compute, capacity and data comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN, load as load_carriers
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    event_features,
    fit_normalization,
    sample_scenes,
    stratified_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_margins import margin_terms
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)

ARMS = {
    "local": {"width": 32, "input": "correct", "pooled": False, "parents": 4},
    "pooled": {"width": 32, "input": "correct", "pooled": True, "parents": 4},
    "constant": {"width": 32, "input": "constant", "pooled": False, "parents": 4},
    "shuffled": {"width": 32, "input": "shuffled", "pooled": False, "parents": 4},
    "wide": {"width": 128, "input": "correct", "pooled": False, "parents": 4},
    "half_data": {"width": 32, "input": "correct", "pooled": False, "parents": 2},
}


def raw_features(cases):
    return {k: event_features(v["feature"].numpy(), v["route"]) for k, v in cases.items()}


def training_ids(cases, parents):
    return [
        k
        for k, v in cases.items()
        if v["metadata"]["split"] == "train" and v["metadata"]["carrier_seed"] <= 41000 + parents
    ]


def register(args):
    previous, cases = load_carriers(args.carriers)
    features = raw_features(cases)
    arms = {}
    for name, config in ARMS.items():
        ids = training_ids(cases, config["parents"])
        mean, std = fit_normalization(features, ids)
        arms[name] = {
            **config,
            "training_ids": ids,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "parameters": sum(p.numel() for p in EventMixture(width=config["width"]).parameters()),
        }
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(args.protocol),
            "carrier_manifest": artifact(args.carriers),
            "previous_result": artifact(args.previous),
            "implementations": [
                artifact(p)
                for p in [
                    Path(__file__),
                    ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_events.py",
                ]
            ],
            "arms": arms,
            "seeds": [8321, 8322, 8323],
            "budgets": [600, 2400],
            "learning_rate": 0.001,
            "threads": 2,
            "max_training_seconds": 7200,
            "max_analysis_seconds": 3600,
            "evaluation_samples": 16,
            "training_corners": previous["training_corners"],
            "evaluation_offsets": previous["evaluation_offsets"],
            "role": "reference_only_development_scale_diagnostic",
            "execution_eligible": False,
            "training_eligible": False,
        },
    )


def load(path):
    reg = json.loads(path.read_text())
    for ref in [
        reg["protocol"],
        reg["carrier_manifest"],
        reg["previous_result"],
        *reg["implementations"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    _, cases = load_carriers(Path(reg["carrier_manifest"]["path"]))
    features = raw_features(cases)
    for config in reg["arms"].values():
        assert config["training_ids"] == training_ids(cases, config["parents"])
        mean, std = fit_normalization(features, config["training_ids"])
        np.testing.assert_array_equal(mean, config["mean"])
        np.testing.assert_array_equal(std, config["std"])
    return reg, cases, features


def normalized(features, config):
    return {k: tensor((v - config["mean"]) / config["std"]) for k, v in features.items()}


def loss_for(model, feature, case, corners, proposal_rng, pose_rng):
    scenes, weights = stratified_scenes(model(feature), proposal_rng)
    offsets = torch.cat(
        (
            torch.zeros(1, 4, dtype=torch.float64),
            corners[torch.randperm(16, generator=pose_rng)[:4]],
        )
    )
    values = perturbed_clearances(
        scenes, offsets, case["clouds"], case["rt"], case["progress"], case["yaw"], DOMAIN
    )
    selection, target, upright, _, _ = margin_terms(values, case["costs"], case["mask"])
    return (weights * (selection + 5 * target + 5 * upright)).sum()


def save_model(
    root, arm, seed, budget, model, optimizer, history, seconds, cases, features, config
):
    out = root / f"{arm}_{seed}" / str(budget)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "checkpoint.pt").open("xb") as handle:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "arm": arm,
                "seed": seed,
                "budget": budget,
                "config": config,
            },
            handle,
        )
    predictions = []
    start = time.monotonic()
    with torch.no_grad():
        for case_id in cases:
            feature = features[case_id]
            if config["input"] == "constant":
                feature = torch.zeros_like(feature)
            scenes = sample_scenes(model(feature), 256, torch.Generator().manual_seed(seed + 20000))
            predictions.append({"case_id": case_id, "scenes": scenes.numpy()[:16].tolist()})
            if arm == "local":
                other = case_id.rsplit("event", 1)[0] + "event" + str((int(case_id[-1]) + 1) % 3)
                scenes = sample_scenes(
                    model(features[other]), 256, torch.Generator().manual_seed(seed + 20000)
                )
                predictions.append(
                    {
                        "case_id": case_id,
                        "input_case": other,
                        "arm": "swapped",
                        "scenes": scenes.numpy()[:16].tolist(),
                    }
                )
    write_new(out / "samples.json", {"predictions": predictions})
    write_new(
        out / "result.json",
        {
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "history": history,
            "cumulative_fitting_seconds": seconds,
            "sampling_seconds": time.monotonic() - start,
            "checkpoint": artifact(out / "checkpoint.pt"),
            "samples": artifact(out / "samples.json"),
        },
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "budget": budget,
                "fit_seconds": seconds,
                "last": history[-1],
            }
        ),
        flush=True,
    )


def train(args):
    reg, cases, raw = load(args.registration)
    torch.set_num_threads(reg["threads"])
    torch.use_deterministic_algorithms(True)
    smoke = args.command == "smoke"
    seeds = reg["seeds"][:1] if smoke else reg["seeds"]
    budgets = [12] if smoke else reg["budgets"]
    root = args.registration.parent / ("smoke" if smoke else "runs")
    root.mkdir(exist_ok=False)
    write_new(root / "started.json", {"registration": artifact(args.registration), "smoke": smoke})
    started = time.monotonic()
    try:
        for arm, config in reg["arms"].items():
            features = normalized(raw, config)
            ids = config["training_ids"]
            jobs = []
            for i, case_id in enumerate(ids):
                source = ids[(i + 1) % len(ids)] if config["input"] == "shuffled" else case_id
                feature = features[source]
                if config["input"] == "constant":
                    feature = torch.zeros_like(feature)
                jobs.append((cases[case_id], feature))
            evaluation = {k: v for k, v in cases.items() if v["metadata"]["split"] != "train"}
            if smoke:
                evaluation = {ids[0]: cases[ids[0]]}
            for seed in seeds:
                torch.manual_seed(seed)
                model = EventMixture(width=config["width"], pooled=config["pooled"]).double()
                optimizer = torch.optim.Adam(model.parameters(), lr=reg["learning_rate"])
                rng = torch.Generator().manual_seed(seed + 10000)
                pose_rng = torch.Generator().manual_seed(seed + 30000)
                order_rng = torch.Generator().manual_seed(seed + 40000)
                history, order, seconds = [], [], 0.0
                corners = tensor(reg["training_corners"])
                for step in range(max(budgets)):
                    if time.monotonic() - started > reg["max_training_seconds"]:
                        raise TimeoutError("registered training budget")
                    start = time.monotonic()
                    if step % len(jobs) == 0:
                        order = torch.randperm(len(jobs), generator=order_rng).tolist()
                    case, feature = jobs[order[step % len(jobs)]]
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_for(model, feature, case, corners, rng, pose_rng)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite loss")
                    loss.backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("nonfinite gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    seconds += time.monotonic() - start
                    if step % 200 == 0 or step + 1 in budgets:
                        history.append({"step": step + 1, "loss": loss.item()})
                    if step + 1 in budgets:
                        save_model(
                            root,
                            arm,
                            seed,
                            step + 1,
                            model,
                            optimizer,
                            list(history),
                            seconds,
                            evaluation,
                            features,
                            config,
                        )
    except Exception as exc:
        write_new(root / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    write_new(
        root / "complete.json",
        {
            "registration": artifact(args.registration),
            "elapsed_seconds": time.monotonic() - started,
            "cells": [artifact(p) for p in sorted(root.glob("*/*/result.json"))],
        },
    )


def analyze(args):
    reg, cases, _ = load(args.registration)
    root = args.registration.parent
    complete_path = root / "runs/complete.json"
    complete = json.loads(complete_path.read_text())
    checked(args.registration, complete["registration"]["sha256"])
    if len(complete["cells"]) != 36:
        raise ValueError("requires all 18 runs and both fixed budget snapshots")
    torch.set_num_threads(reg["threads"])
    predictions = []
    for ref in complete["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ["checkpoint", "samples"]:
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        predictions.extend(
            {
                **p,
                "arm": p.get("arm", cell["arm"]),
                "seed": cell["seed"],
                "budget": cell["budget"],
                "source": ref,
            }
            for p in json.loads(Path(cell["samples"]["path"]).read_text())["predictions"]
        )
    if len(predictions) != 504:
        raise ValueError("incomplete predictions")
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    offsets = np.array(reg["evaluation_offsets"])
    rows = []
    for i, prediction in enumerate(predictions):
        if time.monotonic() - start > reg["max_analysis_seconds"]:
            raise TimeoutError("registered analysis budget")
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
        error = float(abs(values[:4] - other).max())
        if error > 1e-8:
            raise ValueError("NumPy/Torch disagreement")
        name = f'{prediction["arm"]}_{prediction["seed"]}_{prediction["budget"]}_{prediction["case_id"]}'
        path = out / f"{name}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
        rows.append(measures(prediction, case, values, error, artifact(path)))
        if i % 12 == 11:
            print(
                json.dumps(
                    {
                        "audited_rows": i + 1,
                        "total": len(predictions),
                        "last_arm": prediction["arm"],
                    }
                ),
                flush=True,
            )
    # Frozen, previously observed comparators: reuse the first 16 of their 32 fixed draws.
    old = json.loads(Path(reg["previous_result"]["path"]).read_text())
    for row in old["rows"]:
        if row["mode"] not in ["direct", "random"]:
            continue
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["offsets"], offsets)
        np.testing.assert_array_equal(raw["scenes"], row["scenes"])
        prediction = {
            "arm": "previous_" + row["mode"],
            "seed": row["seed"],
            "budget": 0,
            "case_id": row["case_id"],
            "scenes": raw["scenes"][:16].tolist(),
            "reused_previous_draw_prefix": 16,
        }
        rows.append(
            measures(
                prediction, cases[row["case_id"]], raw["clearances"][:16], row["query_error_m"], ref
            )
        )
    summary = []
    for arm, budget in [(a, b) for a in list(reg["arms"]) + ["swapped"] for b in reg["budgets"]] + [
        ("previous_direct", 0),
        ("previous_random", 0),
    ]:
        for carrier in range(41005, 41009):
            selected = [
                r
                for r in rows
                if r["arm"] == arm and r["budget"] == budget and r["carrier_seed"] == carrier
            ]
            assert len(selected) == 9
            summary.append(
                {
                    "arm": arm,
                    "budget": budget,
                    "carrier_seed": carrier,
                    "split": selected[0]["split"],
                    "valid": sum(r["all_113_valid"] for r in selected),
                    "proposals": 144,
                }
            )

    def score(arm, budget, carrier):
        return next(
            r["valid"]
            for r in summary
            if (r["arm"], r["budget"], r["carrier_seed"]) == (arm, budget, carrier)
        )

    predictions = {
        "local_beats_pooled_constant_shuffled_at_2400_both_test": all(
            score("local", 2400, c)
            > max(score(a, 2400, c) for a in ["pooled", "constant", "shuffled"])
            for c in [41007, 41008]
        ),
        "swapped_lowers_local_at_2400_both_test": all(
            score("local", 2400, c) > score("swapped", 2400, c) for c in [41007, 41008]
        ),
        "four_times_updates_improves_local_both_test": all(
            score("local", 2400, c) > score("local", 600, c) for c in [41007, 41008]
        ),
        "wider_improves_over_local_at_2400_both_test": all(
            score("wide", 2400, c) > score("local", 2400, c) for c in [41007, 41008]
        ),
        "four_parents_improve_over_two_at_2400_both_test": all(
            score("local", 2400, c) > score("half_data", 2400, c) for c in [41007, 41008]
        ),
    }
    write_new(
        root / "result.json",
        {
            "registration": artifact(args.registration),
            "training_completion": artifact(complete_path),
            "analysis_complete": True,
            "rows": rows,
            "summary": summary,
            "predictions": predictions,
            "evaluation_seconds": time.monotonic() - start,
            "new_audit_rows": 504,
            "reused_baseline_rows": 72,
            "reference_only": True,
            "execution_eligible": False,
            "training_eligible": False,
        },
    )


def measures(prediction, case, values, error, raw):
    valid = verdict(values, np.array([False, True]))
    return {
        **prediction,
        "split": case["metadata"]["split"],
        "carrier_seed": case["metadata"]["carrier_seed"],
        "event_station": case["metadata"]["event_station"],
        "sample_count": len(values),
        "nominal_valid": int(valid[:, 0].sum()),
        "all_113_valid": int(valid.all(1).sum()),
        "target_clear": int((values[..., 1].min(1) >= 0.01).sum()),
        "upright_interference": int((values[..., 0].max(1) <= -0.01).sum()),
        "per_proposal_valid": valid.all(1).tolist(),
        "query_error_m": error,
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("register")
    for name in ["carriers", "previous", "protocol", "out"]:
        p.add_argument(f"--{name}", type=Path, required=True)
    for command in ["smoke", "train", "analyze"]:
        commands.add_parser(command).add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "smoke": train, "train": train, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
