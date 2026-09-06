#!/usr/bin/env python3
"""Registered matched-query search and bounded proposal refinement."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_source_phase import load as load_bank, summarize
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor, verdict
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    sample_scenes,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import (
    bounded_refine,
    margin_slack,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)

ARMS = ["learned_rank", "uniform_rank", "learned_refine", "uniform_refine"]


def register(args):
    prior = args.source
    previous, cases = load_bank(prior / "manifest.json")
    result = json.loads((prior / "result.json").read_text())
    ids = [
        k
        for k, v in cases.items()
        if v["metadata"]["split"] != "train" and v["metadata"]["event_station"] in [0.35, 0.65]
    ]
    raw = [
        r
        for r in result["rows"]
        if r["arm"] == "all8" and r["budget"] == 1200 and r["case_id"] in ids
    ]
    if len(ids) != 16 or len(raw) != 48:
        raise ValueError("incomplete raw comparison")
    cells = [artifact(prior / f"runs/all8_{seed}/1200/result.json") for seed in previous["seeds"]]
    for row in raw:
        checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(args.protocol),
            "bank": artifact(prior / "manifest.json"),
            "previous": artifact(prior / "result.json"),
            "cells": cells,
            "case_ids": ids,
            "seeds": previous["seeds"],
            "arms": ARMS,
            "search_offsets": [[0.0, 0.0, 0.0, 0.0]] + previous["training_corners"],
            "audit_offsets": previous["evaluation_offsets"],
            "search_queries_per_job": 4624,
            "max_search_seconds": 1800,
            "max_audit_seconds": 1200,
            "implementations": [
                artifact(p)
                for p in [
                    Path(__file__),
                    ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_refinement.py",
                ]
            ],
            "role": "reference_only_previously_observed_development_inference_comparison",
            "execution_eligible": False,
            "training_eligible": False,
        },
    )


def load(path):
    reg = json.loads(path.read_text())
    for ref in [
        reg["protocol"],
        reg["bank"],
        reg["previous"],
        *reg["cells"],
        *reg["implementations"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    _, cases = load_bank(Path(reg["bank"]["path"]))
    cases = {k: cases[k] for k in reg["case_ids"]}
    previous = json.loads(Path(reg["previous"]["path"]).read_text())
    raw = {
        (r["seed"], r["case_id"]): r
        for r in previous["rows"]
        if r["arm"] == "all8" and r["budget"] == 1200 and r["case_id"] in cases
    }
    for row in raw.values():
        checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
    if len(raw) != 48 or len(cases) != 16:
        raise ValueError("incomplete prior outputs")
    return reg, cases, raw


def search(args):
    start = time.monotonic()
    reg, cases, raw = load(args.registration)
    features = raw_features(cases)
    root = args.registration.parent / "search"
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    preprocessing_seconds = time.monotonic() - start
    offsets = tensor(reg["search_offsets"])
    rows = []
    checkpoint_setup_seconds = 0.0
    for ref in reg["cells"]:
        setup_start = time.monotonic()
        cell = json.loads(Path(ref["path"]).read_text())
        for key in ["checkpoint", "samples"]:
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        saved = torch.load(cell["checkpoint"]["path"], map_location="cpu", weights_only=False)
        model = EventMixture(width=32).double()
        model.load_state_dict(saved["model"])
        model.eval()
        normalized_features = normalized(features, saved["config"])
        seed = cell["seed"]
        checkpoint_setup_seconds += time.monotonic() - setup_start
        for case_id, case in cases.items():
            if time.monotonic() - start > reg["max_search_seconds"]:
                raise TimeoutError("search cap")
            tick = time.monotonic()
            with torch.no_grad():
                learned = sample_scenes(
                    model(normalized_features[case_id]),
                    256,
                    torch.Generator().manual_seed(seed + 20000),
                )[:136]
            sampling_seconds = time.monotonic() - tick
            np.testing.assert_array_equal(learned[:8].numpy(), raw[(seed, case_id)]["scenes"])
            tick = time.monotonic()
            uniform = tensor(
                np.random.default_rng(seed + 50000).uniform([0.1, 1.1], [0.9, 1.45], size=(136, 2))
            )

            uniform_sampling_seconds = time.monotonic() - tick

            def query(scenes):
                return perturbed_clearances(
                    scenes,
                    offsets,
                    case["clouds"],
                    case["rt"],
                    case["progress"],
                    case["yaw"],
                    DOMAIN,
                )

            for arm in reg["arms"]:
                if time.monotonic() - start > reg["max_search_seconds"]:
                    raise TimeoutError("search cap")
                tick = time.monotonic()
                candidates = learned if arm.startswith("learned") else uniform
                if arm.endswith("rank"):
                    with torch.no_grad():
                        values = torch.cat([query(chunk) for chunk in candidates.split(8)])
                        indices = torch.argsort(margin_slack(values), descending=True, stable=True)[
                            :8
                        ]
                        scenes = candidates[indices]
                    trace = {"scenes": candidates, "clearances": values, "selected": indices}
                else:
                    scenes, trace = bounded_refine(candidates[:8], query)
                seconds = time.monotonic() - tick
                name = f"{arm}_{seed}_{case_id}"
                path = root / f"{name}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, **{k: v.detach().numpy() for k, v in trace.items()})
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "case_id": case_id,
                        "scenes": scenes.detach().numpy().tolist(),
                        "search_seconds": seconds,
                        "sampling_seconds": (
                            sampling_seconds
                            if arm.startswith("learned")
                            else uniform_sampling_seconds
                        ),
                        "queries": 4624,
                        "trace": artifact(path),
                    }
                )
            print(json.dumps({"finished_jobs": len(rows) // 4, "total_jobs": 48}), flush=True)
    write_new(
        root / "complete.json",
        {
            "registration": artifact(args.registration),
            "rows": rows,
            "preprocessing_seconds": preprocessing_seconds,
            "checkpoint_setup_seconds": checkpoint_setup_seconds,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def measure(prediction, case, values, error, raw_ref, prior_valid=None):
    valid = verdict(values, case["mask"].numpy()).all(1)
    row = {
        **prediction,
        "carrier_seed": case["metadata"]["carrier_seed"],
        "split": case["metadata"]["split"],
        "event_station": case["metadata"]["event_station"],
        "phase": "unseen",
        "budget": 0,
        "valid": int(valid.sum()),
        "proposals": 8,
        "per_proposal_valid": valid.tolist(),
        "target_failures": int((values[:, :, 1].min(1) < 0.01).sum()),
        "neutral_failures": int((values[:, :, 0].max(1) > -0.01).sum()),
        "query_error_m": error,
        "raw": raw_ref,
    }
    if prior_valid is not None:
        row.update(
            rescued=int((valid & ~prior_valid).sum()),
            newly_failed=int((~valid & prior_valid).sum()),
        )
    accepted = np.array(row["scenes"])[valid]
    bins = np.floor((accepted - [0.1, 1.1]) / [0.02, 0.01]).astype(int)
    row["occupied_accepted_bins"] = len(np.unique(bins, axis=0))
    return row


def analyze(args):
    reg, cases, raw = load(args.registration)
    root = args.registration.parent
    complete = json.loads((root / "search/complete.json").read_text())
    checked(args.registration, complete["registration"]["sha256"])
    expected = {(a, s, c) for a in reg["arms"] for s in reg["seeds"] for c in cases}
    if (
        len(complete["rows"]) != 192
        or {(r["arm"], r["seed"], r["case_id"]) for r in complete["rows"]} != expected
    ):
        raise ValueError("incomplete search")
    out = root / "audit"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    start = time.monotonic()
    offsets = np.array(reg["audit_offsets"])
    rows = []
    for i, prediction in enumerate(complete["rows"]):
        if time.monotonic() - start > reg["max_audit_seconds"]:
            raise TimeoutError("audit cap")
        checked(Path(prediction["trace"]["path"]), prediction["trace"]["sha256"])
        case = cases[prediction["case_id"]]
        scenes = np.array(prediction["scenes"])
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
            raise ValueError("nonfinite or disagreeing geometry audit")
        path = out / f'{prediction["arm"]}_{prediction["seed"]}_{prediction["case_id"]}.npz'
        with path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
        previous = raw[(prediction["seed"], prediction["case_id"])]
        prior = np.load(previous["raw"]["path"], allow_pickle=False)
        prior_valid = verdict(prior["clearances"], case["mask"].numpy()).all(1)
        rows.append(
            measure(
                prediction,
                case,
                values,
                error,
                artifact(path),
                prior_valid if prediction["arm"] == "learned_refine" else None,
            )
        )
        if (i + 1) % 16 == 0:
            print(json.dumps({"audited_rows": i + 1, "total": 192}), flush=True)
    for (seed, name), previous in raw.items():
        prior = np.load(previous["raw"]["path"], allow_pickle=False)
        np.testing.assert_array_equal(prior["offsets"], offsets)
        np.testing.assert_array_equal(prior["scenes"], previous["scenes"])
        rows.append(
            measure(
                {
                    "arm": "raw",
                    "seed": seed,
                    "case_id": name,
                    "scenes": previous["scenes"],
                    "queries": 0,
                },
                cases[name],
                prior["clearances"],
                previous["query_error_m"],
                previous["raw"],
            )
        )
    summary = summarize(rows)

    def score(arm, parent):
        return next(r["valid"] for r in summary if r["arm"] == arm and r["carrier_seed"] == parent)

    predicates = {}
    for comparator in ["raw", "uniform_refine", "learned_rank"]:
        delta = {
            str(p): score("learned_refine", p) - score(comparator, p)
            for p in [41007, 41008, 42007, 42008]
        }
        predicates["beats_" + comparator] = {
            "per_test_parent_delta": delta,
            "improves_each_parent": all(d > 0 for d in delta.values()),
        }
    hard = sum(
        r["valid"]
        for r in rows
        if (r["arm"], r["carrier_seed"], r["event_station"]) == ("learned_refine", 42007, 0.35)
    )
    predicates["hard_case"] = {"valid": hard, "proposals": 24, "gains_any": hard > 0}
    write_new(
        root / "result.json",
        {
            "registration": artifact(args.registration),
            "search": artifact(root / "search/complete.json"),
            "rows": rows,
            "summary": summary,
            "predicates": predicates,
            "audit_seconds": time.monotonic() - start,
            "search_seconds": complete["elapsed_seconds"],
            "search_queries": 192 * 4624,
            "audit_queries": 192 * 8 * 113 * 2,
            "crosscheck_queries": 192 * 2 * 113 * 2,
            "max_query_error_m": max(r["query_error_m"] for r in rows),
            "execution_eligible": False,
            "training_eligible": False,
        },
    )
    print(json.dumps(predicates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "search", "analyze"])
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--source", type=Path, default=ROOT.parent / "research-data/groot-wbc/m2s-source-phase-v1"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/motion2scene/REFINEMENT_V1.md"
    )
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if args.registration is not None:
            path = args.registration.parent / f"{args.command}_failure.json"
            if not path.exists():
                write_new(path, {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
