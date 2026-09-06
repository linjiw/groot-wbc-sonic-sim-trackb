#!/usr/bin/env python3
"""Registered training-source set distillation; the 430xx pool is never loaded."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import loss_for, normalized, raw_features
from motion2scene_refinement_study import load as load_parent, measure
from motion2scene_source_phase import load as load_bank, selected_ids, summarize
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_distillation import (
    pattern_five,
    set_distance,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    sample_scenes,
    stratified_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)

DATA = ROOT.parent / "research-data/groot-wbc"
FITS = {"geometry600": 600, "geometry2666": 2666, "set600": 600, "hybrid600": 600}
ARMS = [
    "original_raw",
    *[a + "_raw" for a in FITS],
    "original_pattern17",
    "hybrid_pattern17",
    "hybrid_pattern5",
    "uniform_pattern5",
]


def register(args):
    parent_path = DATA / "m2s-refinement-v1/registration.json"
    parent, dev, _ = load_parent(parent_path)
    source, bank = load_bank(Path(parent["bank"]["path"]))
    ids = selected_ids(bank, source["arms"]["all8"]["parents"])
    args.out.mkdir(exist_ok=False, parents=True)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/DISTILLATION_V1.md"),
            "parent": artifact(parent_path),
            "training_ids": ids,
            "development_ids": list(dev),
            "seeds": parent["seeds"],
            "fits": FITS,
            "arms": ARMS,
            "implementations": [
                artifact(p)
                for p in [
                    Path(__file__),
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_distillation.py",
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_station_search.py",
                ]
            ],
            "max_teacher_seconds": 600,
            "max_fit_seconds": 1800,
            "max_evaluation_seconds": 1800,
            "role": "observed_source_reference_only_development_distillation",
            "excluded_source_ids": list(range(43001, 43009)),
            "execution_eligible": False,
            "original_bank_training_eligible": False,
            "gpu_seconds": 0,
        },
    )


def load(path):
    reg = json.loads(path.read_text())
    for ref in [reg["protocol"], reg["parent"], *reg["implementations"]]:
        checked(Path(ref["path"]), ref["sha256"])
    parent, dev, _ = load_parent(Path(reg["parent"]["path"]))
    source, bank = load_bank(Path(parent["bank"]["path"]))
    ids = selected_ids(bank, source["arms"]["all8"]["parents"])
    if (
        ids != reg["training_ids"]
        or list(dev) != reg["development_ids"]
        or reg["fits"] != FITS
        or reg["arms"] != ARMS
    ):
        raise ValueError("registered design changed")
    train = {k: bank[k] for k in ids}
    if any(
        v["metadata"]["carrier_seed"] in reg["excluded_source_ids"]
        for v in [*train.values(), *dev.values()]
    ):
        raise ValueError("fresh source leakage")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return reg, parent, train, dev


def checkpoint(ref):
    checked(Path(ref["path"]), ref["sha256"])
    saved = torch.load(ref["path"], map_location="cpu", weights_only=False)
    model = EventMixture(width=32).double()
    model.load_state_dict(saved["model"])
    model.eval()
    return saved, model


def original_cells(parent):
    cells = []
    for ref in parent["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        checked(Path(cell["checkpoint"]["path"]), cell["checkpoint"]["sha256"])
        cells.append(cell)
    return cells


def query_for(case, offsets):
    return lambda scenes: perturbed_clearances(
        scenes, offsets, case["clouds"], case["rt"], case["progress"], case["yaw"], DOMAIN
    )


def independent(scenes, case, offsets):
    values = numpy_perturbed(scenes, offsets, case["states"], case["route"], case["yaw"].item())
    with torch.no_grad():
        other = torch.cat(
            [
                query_for(case, tensor(chunk))(tensor(scenes[:2]))
                for chunk in np.array_split(offsets, 23)
            ],
            dim=1,
        ).numpy()
    error = float(abs(values[:2] - other).max())
    if not np.isfinite(values).all() or not np.isfinite(error) or error > 1e-8:
        raise ValueError("nonfinite or disagreeing independent geometry")
    return values, error


def retained_indices(scenes, accepted):
    bins, indices = set(), []
    for i, (scene, valid) in enumerate(zip(scenes, accepted)):
        key = tuple(np.floor((scene - [0.1, 1.1]) / [0.02, 0.01]).astype(int))
        if valid and key not in bins:
            bins.add(key)
            indices.append(i)
    return indices


def teacher(args):
    start = time.monotonic()
    reg, parent, train, _ = load(args.registration)
    out = args.registration.parent / "teacher"
    out.mkdir(exist_ok=False)
    cell = original_cells(parent)[0]
    assert cell["seed"] == 8421
    saved, model = checkpoint(cell["checkpoint"])
    features = normalized(raw_features(train), saved["config"])
    rows = []
    for i, (name, case) in enumerate(train.items()):
        if time.monotonic() - start > reg["max_teacher_seconds"]:
            raise TimeoutError("teacher cap")
        tick = time.monotonic()
        with torch.no_grad():
            learned = sample_scenes(
                model(features[name]), 256, torch.Generator().manual_seed(118421 + i)
            )[:4]
            rng = torch.Generator().manual_seed(138421 + i)
            uniform = tensor([0.1, 1.1]) + torch.rand(
                (2, 2), dtype=torch.float64, generator=rng
            ) * tensor([0.8, 0.35])
            noise = torch.rand((2, 2), dtype=torch.float64, generator=rng)
            stratified = torch.stack(
                (0.1 + 0.4 * (torch.arange(2) + noise[:, 0]), 1.1 + 0.35 * noise[:, 1]), -1
            )
            initial = torch.cat([learned, uniform, stratified])
        sampling = time.monotonic() - tick
        tick = time.monotonic()
        scenes, trace = station_search(
            initial, query_for(case, tensor(parent["search_offsets"])), "pattern"
        )
        search_seconds = time.monotonic() - tick
        tick = time.monotonic()
        values, error = independent(scenes.numpy(), case, np.array(parent["audit_offsets"]))
        audit_seconds = time.monotonic() - tick
        accepted = verdict(values, case["mask"].numpy()).all(1)
        indices = retained_indices(scenes.numpy(), accepted)
        path = out / f"{name}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{k: v.numpy() for k, v in trace.items()},
                output=scenes.numpy(),
                audit_clearances=values,
                offsets=parent["audit_offsets"],
                retained=indices,
            )
        rows.append(
            {
                "case_id": name,
                "carrier_seed": case["metadata"]["carrier_seed"],
                "accepted": accepted.tolist(),
                "retained_indices": indices,
                "targets": scenes.numpy()[indices].tolist(),
                "trace": artifact(path),
                "sampling_seconds": sampling,
                "search_seconds": search_seconds,
                "audit_seconds": audit_seconds,
                "query_error_m": error,
            }
        )
        print(json.dumps({"teacher_cases": len(rows), "retained": len(indices)}), flush=True)
    write_new(
        out / "complete.json",
        {
            "registration": artifact(args.registration),
            "checkpoint": cell["checkpoint"],
            "rows": rows,
            "elapsed_seconds": time.monotonic() - start,
            "search_queries": 110976,
            "audit_queries": 43392,
            "crosscheck_queries": 10848,
            "total_queries": 165216,
            "all_cases_nonempty": all(r["targets"] for r in rows),
        },
    )
    if not any(r["targets"] for r in rows):
        raise ValueError("all teacher sets empty; fitting stopped")


def read_teacher(root, reg):
    path = root / "teacher/complete.json"
    data = json.loads(path.read_text())
    checked(root / "registration.json", data["registration"]["sha256"])
    if [r["case_id"] for r in data["rows"]] != reg["training_ids"]:
        raise ValueError("teacher source roles changed")
    for row in data["rows"]:
        checked(Path(row["trace"]["path"]), row["trace"]["sha256"])
    return data, {r["case_id"]: tensor(r["targets"]).reshape(-1, 2) for r in data["rows"]}


def fit(args):
    start = time.monotonic()
    reg, parent, train, _ = load(args.registration)
    teacher_data, targets = read_teacher(args.registration.parent, reg)
    out = args.registration.parent / "runs"
    out.mkdir(exist_ok=False)
    write_new(
        out / "started.json",
        {
            "registration": artifact(args.registration),
            "teacher": artifact(args.registration.parent / "teacher/complete.json"),
        },
    )
    raw = raw_features(train)
    cells = []
    for original in original_cells(parent):
        seed = original["seed"]
        for arm, steps in FITS.items():
            saved, model = checkpoint(original["checkpoint"])
            if saved["config"]["training_ids"] != reg["training_ids"]:
                raise ValueError("training lineage differs from original checkpoint")
            features = normalized(raw, saved["config"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            rng, pose_rng, order_rng, set_rng = [
                torch.Generator().manual_seed(seed + n) for n in [210000, 220000, 230000, 240000]
            ]
            ids, order, history, skips, queries = reg["training_ids"], [], [], 0, 0
            tick = time.monotonic()
            for step in range(steps):
                if time.monotonic() - start > reg["max_fit_seconds"]:
                    raise TimeoutError("fit cap")
                if step % len(ids) == 0:
                    order = torch.randperm(len(ids), generator=order_rng).tolist()
                name = ids[order[step % len(ids)]]
                if arm == "set600" and len(targets[name]) == 0:
                    skips += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                geom, energy = tensor(0.0), tensor(0.0)
                if arm != "set600":
                    geom = loss_for(
                        model,
                        features[name],
                        train[name],
                        tensor(parent["search_offsets"][1:]),
                        rng,
                        pose_rng,
                    )
                    queries += 80
                if arm in ["set600", "hybrid600"]:
                    points, weights = stratified_scenes(model(features[name]), set_rng)
                    energy = set_distance(points, weights, targets[name])
                loss = (
                    energy if arm == "set600" else geom + (10 * energy if arm == "hybrid600" else 0)
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
                if (step + 1) % 200 == 0 or step + 1 == steps:
                    history.append(
                        {
                            "step": step + 1,
                            "loss": loss.item(),
                            "geometry": geom.item(),
                            "set_distance": energy.item(),
                        }
                    )
            seconds = time.monotonic() - tick
            path = out / f"{arm}_{seed}.pt"
            with path.open("xb") as handle:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": saved["config"],
                        "seed": seed,
                        "arm": arm,
                        "steps": steps,
                    },
                    handle,
                )
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "steps": steps,
                    "skipped_empty_updates": skips,
                    "checkpoint": artifact(path),
                    "original": original["checkpoint"],
                    "history": history,
                    "training_queries": queries,
                    "teacher_queries_charged_per_student": (
                        teacher_data["total_queries"] if arm in ["set600", "hybrid600"] else 0
                    ),
                    "fitting_seconds": seconds,
                }
            )
            print(json.dumps({"fitted": arm, "seed": seed, "seconds": seconds}), flush=True)
    write_new(
        out / "complete.json",
        {
            "registration": artifact(args.registration),
            "teacher": artifact(args.registration.parent / "teacher/complete.json"),
            "cells": cells,
            "elapsed_seconds": time.monotonic() - start,
            "actual_training_queries": sum(c["training_queries"] for c in cells),
        },
    )


def predicates(rows, teacher_data):
    summary = summarize(rows)
    sources = sorted({r["carrier_seed"] for r in rows})

    def delta(a, b):
        return {
            str(s): sum(r["valid"] for r in rows if r["arm"] == a and r["carrier_seed"] == s)
            - sum(r["valid"] for r in rows if r["arm"] == b and r["carrier_seed"] == s)
            for s in sources
        }

    d1, d2, d3 = (
        delta("hybrid600_raw", "original_raw"),
        delta("hybrid600_raw", "geometry2666_raw"),
        delta("hybrid_pattern5", "original_pattern17"),
    )
    bins = {
        a: sum(r["occupied_accepted_bins"] for r in rows if r["arm"] == a) / 48
        for a in ["hybrid600_raw", "original_raw"]
    }
    return summary, {
        "P0_all_teacher_cases_nonempty": teacher_data["all_cases_nonempty"],
        "P1_hybrid_raw_no_source_loss_and_total_gain": min(d1.values()) >= 0
        and sum(d1.values()) > 0,
        "P2_hybrid_raw_beats_query_control": min(d2.values()) >= 0 and sum(d2.values()) > 0,
        "P3_reduced_search_no_source_loss": min(d3.values()) >= 0,
        "P4_raw_accepted_bin_mean_preserved": bins["hybrid600_raw"] >= bins["original_raw"],
        "hybrid_minus_original_raw": d1,
        "hybrid_minus_query_control": d2,
        "reduced_minus_original_pattern": d3,
        "raw_mean_accepted_bins": bins,
    }


def evaluate(args):
    start = time.monotonic()
    reg, parent, _, dev = load(args.registration)
    root = args.registration.parent
    teacher_data, _ = read_teacher(root, reg)
    fitted = json.loads((root / "runs/complete.json").read_text())
    checked(args.registration, fitted["registration"]["sha256"])
    checked(root / "teacher/complete.json", fitted["teacher"]["sha256"])
    if len(fitted["cells"]) != 12 or {(c["arm"], c["seed"]) for c in fitted["cells"]} != {
        (a, s) for a in FITS for s in reg["seeds"]
    }:
        raise ValueError("incomplete fitting panel")
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    raw, rows, sampling_seconds, setup_seconds = raw_features(dev), [], 0.0, 0.0
    for original in original_cells(parent):
        seed = original["seed"]
        tick = time.monotonic()
        refs = {
            "original": original["checkpoint"],
            **{c["arm"]: c["checkpoint"] for c in fitted["cells"] if c["seed"] == seed},
        }
        models = {a: checkpoint(ref) for a, ref in refs.items()}
        for saved, _ in models.values():
            if saved["config"] != models["original"][0]["config"]:
                raise ValueError("normalization or training roles changed")
        features = normalized(raw, models["original"][0]["config"])
        setup_seconds += time.monotonic() - tick
        for name, case in dev.items():
            tick = time.monotonic()
            with torch.no_grad():
                draws = {
                    a: sample_scenes(
                        model(features[name]), 256, torch.Generator().manual_seed(seed + 250000)
                    )[:8]
                    for a, (_, model) in models.items()
                }
                draws["uniform"] = tensor([0.1, 1.1]) + torch.rand(
                    (8, 2),
                    dtype=torch.float64,
                    generator=torch.Generator().manual_seed(seed + 260000),
                ) * tensor([0.8, 0.35])
            sampling_seconds += time.monotonic() - tick
            for arm in ARMS:
                if time.monotonic() - start > reg["max_evaluation_seconds"]:
                    raise TimeoutError("evaluation cap")
                base = arm.split("_")[0]
                if base == "hybrid":
                    base = "hybrid600"
                initial = draws[base]
                tick = time.monotonic()
                query = query_for(case, tensor(parent["search_offsets"]))
                if arm.endswith("raw"):
                    scenes, trace, queries = initial, {"initial": initial, "scenes": initial}, 0
                elif arm.endswith("17"):
                    scenes, trace = station_search(initial, query, "pattern")
                    queries = 4624
                else:
                    scenes, trace = pattern_five(initial, query)
                    queries = 1360
                search_seconds = time.monotonic() - tick
                tick = time.monotonic()
                values, error = independent(scenes.numpy(), case, np.array(parent["audit_offsets"]))
                audit_seconds = time.monotonic() - tick
                path = out / f"{arm}_{seed}_{name}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(
                        handle,
                        **{k: v.numpy() for k, v in trace.items()},
                        output=scenes.numpy(),
                        audit_clearances=values,
                        offsets=parent["audit_offsets"],
                    )
                prediction = {
                    "arm": arm,
                    "seed": seed,
                    "case_id": name,
                    "scenes": scenes.tolist(),
                    "draw_seed": seed + 250000 if base != "uniform" else seed + 260000,
                    "queries": queries,
                    "search_seconds": search_seconds,
                    "audit_seconds": audit_seconds,
                    "checkpoint": refs.get(base),
                    "trace": artifact(path),
                }
                rows.append(measure(prediction, case, values, error, artifact(path)))
            print(json.dumps({"evaluation_jobs": len(rows) // 9, "total": 48}), flush=True)
    summary, outcomes = predicates(rows, teacher_data)
    write_new(
        root / "result.json",
        {
            "registration": artifact(args.registration),
            "teacher": artifact(root / "teacher/complete.json"),
            "fitting": artifact(root / "runs/complete.json"),
            "rows": rows,
            "summary": summary,
            "predicates": outcomes,
            "elapsed_seconds": time.monotonic() - start,
            "sampling_seconds": sampling_seconds,
            "checkpoint_setup_seconds": setup_seconds,
            "search_queries": sum(r["queries"] for r in rows),
            "audit_queries": 432 * 8 * 113 * 2,
            "crosscheck_queries": 432 * 2 * 113 * 2,
            "max_query_error_m": max(r["query_error_m"] for r in rows),
            "gpu_seconds": 0,
            "fresh_source_evidence": False,
            "execution_eligible": False,
        },
    )
    print(json.dumps(outcomes), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "teacher", "fit", "evaluate"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if args.registration:
            path = args.registration.parent / f"{args.command}_failure.json"
            if not path.exists():
                write_new(path, {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
