#!/usr/bin/env python3
"""Frozen source-diversity / unseen-phase reference diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from motion2scene_beam_teacher import capsules
from motion2scene_carrier_learning import DOMAIN, RECORDS, gate
from motion2scene_event_scaling import loss_for, normalized, raw_features
from motion2scene_inverse_learning import inputs
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import make_clouds, numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import shared_origin_pair
from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    fit_normalization,
    sample_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)
from gear_sonic.dataset_generation.local_adaptation import local_crouch
from gear_sonic.dataset_generation.reference_payload import payload_from_reference

BASE = list(range(41001, 41005))
ARMS = {
    "base4": {"parents": BASE, "budgets": [600, 1200]},
    "add_a6": {"parents": BASE + [42001, 42002], "budgets": [600, 900]},
    "add_b6": {"parents": BASE + [42003, 42004], "budgets": [600, 900]},
    "all8": {"parents": BASE + list(range(42001, 42005)), "budgets": [600, 1200]},
}
SEEN = [0.25, 0.5, 0.75]


def study_split(seed):
    if seed not in list(range(41001, 41009)) + list(range(42001, 42009)):
        raise ValueError("unregistered source")
    return "train" if seed % 1000 <= 4 else "validation" if seed % 1000 <= 6 else "test"


def selected_ids(cases, parents):
    if any(study_split(p) != "train" for p in parents) or len(set(parents)) != len(parents):
        raise ValueError("training parents contain excluded or duplicate sources")
    ids = [
        k
        for k, v in cases.items()
        if v["metadata"]["carrier_seed"] in parents and v["metadata"]["event_station"] in SEEN
    ]
    if len(ids) != 3 * len(parents):
        raise ValueError("incomplete training derivatives")
    return ids


def register(args):
    prior = ROOT.parent / "research-data/groot-wbc/m2s-carrier-learning-v1/registration.json"
    previous = json.loads(prior.read_text())
    inventory_path = ROOT / "docs/motion2scene/evidence/data-scale-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    carriers = list(previous["base_carriers"])
    for r in inventory["additional_previously_observed_neutral_parents"]:
        p = ROOT.parent / "research-data/groot-wbc" / r["reference"]["path"]
        checked(p, r["reference"]["sha256"])
        carriers.append(
            {
                "generation_seed": r["generation_seed"],
                "csv": str(p),
                "csv_sha256": r["reference"]["sha256"],
            }
        )
    if [r["generation_seed"] for r in carriers] != list(range(41001, 41009)) + list(
        range(42001, 42009)
    ):
        raise ValueError("source inventory mismatch")
    implementations = previous["implementations"] + [
        artifact(p)
        for p in [
            Path(__file__),
            ROOT / "scripts/research/motion2scene_event_scaling.py",
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_events.py",
        ]
    ]
    sources = (
        previous["sources"]
        + [artifact(prior), artifact(inventory_path)]
        + [artifact(Path(r["csv"])) for r in carriers[8:]]
    )
    for ref in sources + implementations:
        checked(Path(ref["path"]), ref["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(args.protocol),
            "sources": sources,
            "implementations": implementations,
            "base_carriers": carriers,
            "splits": {
                str(r["generation_seed"]): study_split(r["generation_seed"]) for r in carriers
            },
            "mjcf": previous["mjcf"],
            "event_stations": SEEN + [0.35, 0.65],
            "operator_drop_m": 0.055,
            "operator_window": 0.18,
            "domain": asdict(DOMAIN),
            "arms": ARMS,
            "seeds": [8421, 8422, 8423],
            "learning_rate": 0.001,
            "training_corners": previous["training_corners"],
            "evaluation_offsets": previous["evaluation_offsets"],
            "threads": 2,
            "max_stage_seconds": 600,
            "max_training_seconds": 3600,
            "max_analysis_seconds": 1800,
            "evaluation_samples": 8,
            "role": "reference_only_development_new_roles_for_previously_observed_420xx",
            "training_eligible": False,
            "execution_eligible": False,
        },
    )


def registration(path):
    data = json.loads(path.read_text())
    for ref in [data["protocol"], *data["sources"], *data["implementations"]]:
        checked(Path(ref["path"]), ref["sha256"])
    if data["domain"] != asdict(DOMAIN) or data["arms"] != ARMS:
        raise ValueError("registered design changed")
    return data


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
            reference = out / f"{case_id}.csv"
            with reference.open("x") as handle:
                np.savetxt(handle, target, delimiter=",", fmt="%.10f")
            # Build FK from the exact persisted reference, not a hidden higher-precision copy.
            first, second = shared_origin_pair(neutral, np.loadtxt(reference, delimiter=","))
            target_gate = gate(np.loadtxt(reference, delimiter=","), case_id, reg["mjcf"]["path"])
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
                "split": study_split(seed),
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
            "source_carriers": 16,
            "target_cases": 80,
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


def save_cell(root, arm, seed, budget, model, config, history, seconds, features, cases):
    out = root / f"{arm}_{seed}" / str(budget)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "checkpoint.pt").open("xb") as handle:
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "seed": seed,
                "arm": arm,
                "budget": budget,
            },
            handle,
        )
    predictions = []
    with torch.no_grad():
        for name, case in cases.items():
            if case["metadata"]["split"] == "train":
                continue
            scenes = sample_scenes(
                model(features[name]), 256, torch.Generator().manual_seed(seed + 20000)
            )
            predictions.append({"case_id": name, "scenes": scenes[:8].numpy().tolist()})
    write_new(out / "samples.json", {"predictions": predictions})
    write_new(
        out / "result.json",
        {
            "arm": arm,
            "seed": seed,
            "budget": budget,
            "history": list(history),
            "cumulative_fitting_seconds": seconds,
            "checkpoint": artifact(out / "checkpoint.pt"),
            "samples": artifact(out / "samples.json"),
        },
    )
    print(json.dumps({"arm": arm, "seed": seed, "budget": budget, "seconds": seconds}), flush=True)


def train(args):
    reg, cases = load(args.registration.parent / "manifest.json")
    raw = raw_features(cases)
    torch.set_num_threads(reg["threads"])
    torch.use_deterministic_algorithms(True)
    root = args.registration.parent / "runs"
    root.mkdir(exist_ok=False)
    configs = {}
    for arm, config in reg["arms"].items():
        ids = selected_ids(cases, config["parents"])
        mean, std = fit_normalization(raw, ids)
        configs[arm] = {**config, "training_ids": ids, "mean": mean.tolist(), "std": std.tolist()}
    write_new(
        root / "started.json", {"registration": artifact(args.registration), "configs": configs}
    )
    start = time.monotonic()
    for arm, config in configs.items():
        features = normalized(raw, config)
        ids = config["training_ids"]
        for seed in reg["seeds"]:
            torch.manual_seed(seed)
            model = EventMixture(width=32).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=reg["learning_rate"])
            rng = torch.Generator().manual_seed(seed + 10000)
            pose_rng = torch.Generator().manual_seed(seed + 30000)
            order_rng = torch.Generator().manual_seed(seed + 40000)
            history, order, seconds = [], [], 0.0
            for step in range(max(config["budgets"])):
                if time.monotonic() - start > reg["max_training_seconds"]:
                    raise TimeoutError("registered training cap")
                tick = time.monotonic()
                if step % len(ids) == 0:
                    order = torch.randperm(len(ids), generator=order_rng).tolist()
                name = ids[order[step % len(ids)]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(
                    model,
                    features[name],
                    cases[name],
                    tensor(reg["training_corners"]),
                    rng,
                    pose_rng,
                )
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
                seconds += time.monotonic() - tick
                if step % 200 == 0 or step + 1 in config["budgets"]:
                    history.append({"step": step + 1, "loss": loss.item()})
                if step + 1 in config["budgets"]:
                    save_cell(
                        root, arm, seed, step + 1, model, config, history, seconds, features, cases
                    )
    write_new(
        root / "complete.json",
        {
            "registration": artifact(args.registration),
            "elapsed_seconds": time.monotonic() - start,
            "cells": [artifact(p) for p in sorted(root.glob("*/*/result.json"))],
        },
    )


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["arm"], row["budget"], row["carrier_seed"], row["phase"])
        if key not in groups:
            groups[key] = {k: row[k] for k in ("arm", "budget", "carrier_seed", "phase", "split")}
            groups[key].update(
                valid=0, proposals=0, target_failures=0, neutral_failures=0, by_seed={}
            )
        group = groups[key]
        for field in ("valid", "proposals", "target_failures", "neutral_failures"):
            group[field] += row[field]
        group["by_seed"].setdefault(str(row["seed"]), {"valid": 0, "proposals": 0})
        for field in ("valid", "proposals"):
            group["by_seed"][str(row["seed"])][field] += row[field]
    return list(groups.values())


def analyze(args):
    reg, cases = load(args.registration.parent / "manifest.json")
    root = args.registration.parent
    complete = json.loads((root / "runs/complete.json").read_text())
    checked(args.registration, complete["registration"]["sha256"])
    expected = {
        (a, s, b) for a, c in reg["arms"].items() for s in reg["seeds"] for b in c["budgets"]
    }
    cells, predictions = set(), []
    for ref in complete["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        key = tuple(cell[k] for k in ("arm", "seed", "budget"))
        if key in cells:
            raise ValueError("duplicate cell")
        cells.add(key)
        for k in ("checkpoint", "samples"):
            checked(Path(cell[k]["path"]), cell[k]["sha256"])
        batch = json.loads(Path(cell["samples"]["path"]).read_text())["predictions"]
        expected_ids = {k for k, v in cases.items() if v["metadata"]["split"] != "train"}
        if len(batch) != 40 or {p["case_id"] for p in batch} != expected_ids:
            raise ValueError("incomplete excluded-case predictions")
        predictions += [
            {**p, **{k: cell[k] for k in ("arm", "seed", "budget")}, "cell": ref} for p in batch
        ]
    if cells != expected or len(predictions) != 960:
        raise ValueError("incomplete experiment")
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(reg["threads"])
    offsets = np.array(reg["evaluation_offsets"])
    rows, start = [], time.monotonic()
    for i, prediction in enumerate(predictions):
        if time.monotonic() - start > reg["max_analysis_seconds"]:
            raise TimeoutError("registered analysis cap")
        case = cases[prediction["case_id"]]
        scenes = np.array(prediction["scenes"])
        if scenes.shape != (8, 2) or not np.isfinite(scenes).all():
            raise ValueError("invalid proposal batch")
        values = numpy_perturbed(scenes, offsets, case["states"], case["route"], case["yaw"].item())
        with torch.no_grad():
            other = torch.cat(
                [
                    perturbed_clearances(
                        tensor(scenes[:2]),
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
        error = float(abs(values[:2] - other).max())
        if not np.isfinite(values).all() or not np.isfinite(error) or error > 1e-8:
            raise ValueError("invalid or disagreeing independent audit")
        filename = (
            "_".join(str(prediction[k]) for k in ("arm", "seed", "budget", "case_id")) + ".npz"
        )
        path = out / filename
        with path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
        meta = case["metadata"]
        rows.append(
            {
                **prediction,
                "carrier_seed": meta["carrier_seed"],
                "split": meta["split"],
                "event_station": meta["event_station"],
                "phase": "seen" if meta["event_station"] in SEEN else "unseen",
                "valid": int(verdict(values, case["mask"].numpy()).all(axis=1).sum()),
                "proposals": 8,
                "target_failures": int((values[:, :, 1].min(axis=1) < 0.01).sum()),
                "neutral_failures": int((values[:, :, 0].max(axis=1) > -0.01).sum()),
                "query_error_m": error,
                "raw": artifact(path),
            }
        )
        if (i + 1) % 40 == 0:
            print(json.dumps({"audited_rows": i + 1, "total": len(predictions)}), flush=True)
    summary = summarize(rows)

    def score(arm, budget, parent):
        return next(
            r["valid"]
            for r in summary
            if (r["arm"], r["budget"], r["carrier_seed"], r["phase"])
            == (arm, budget, parent, "unseen")
        )

    predicates = {}
    for label, new_budget, old_budget in [
        ("equal_queries", 600, 600),
        ("equal_visits", 1200, 600),
        ("extra_update_control", 1200, 1200),
    ]:
        differences = {
            str(p): score("all8", new_budget, p) - score("base4", old_budget, p)
            for p in [41007, 41008, 42007, 42008]
        }
        predicates[label] = {
            "unseen_test_per_parent_delta": differences,
            "improves_each_parent": all(d > 0 for d in differences.values()),
        }
    write_new(
        root / "result.json",
        {
            "registration": artifact(args.registration),
            "manifest": artifact(root / "manifest.json"),
            "runs": artifact(root / "runs/complete.json"),
            "rows": rows,
            "summary": summary,
            "predicates": predicates,
            "analysis_seconds": time.monotonic() - start,
            "training_seconds": complete["elapsed_seconds"],
            "training_queries": 3 * sum(max(c["budgets"]) for c in reg["arms"].values()) * 80,
            "audit_queries": len(rows) * 8 * 113 * 2,
            "crosscheck_queries": len(rows) * 2 * 113 * 2,
            "max_query_error_m": max(r["query_error_m"] for r in rows),
            "execution_eligible": False,
            "training_eligible": False,
        },
    )
    print(json.dumps(predicates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "build", "train", "analyze"])
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/motion2scene/SOURCE_PHASE_V1.md"
    )
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if args.registration is not None:
            failure = args.registration.parent / f"{args.command}_failure.json"
            if not failure.exists():
                write_new(failure, {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
