#!/usr/bin/env python3
"""Registered fresh-draw station-search comparison with a separate failure replay."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_refinement_study import load as load_parent, measure
from motion2scene_source_phase import summarize
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed, tensor
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
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import (
    METHODS,
    station_search,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)

ARMS = ["raw", "rank", "gradient", *METHODS]


def register(args):
    parent, cases, _ = load_parent(args.parent)
    replay_path = args.parent.parent / "fresh-sampler/result.json"
    replay = json.loads(replay_path.read_text())
    for key in ["trace", "sampler", "checkpoint"]:
        checked(Path(replay[key]["path"]), replay[key]["sha256"])
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(args.protocol),
            "parent": artifact(args.parent),
            "replay": artifact(replay_path),
            "replay_plan": artifact(args.parent.parent / "sampler-plan.json"),
            "implementations": [
                artifact(p)
                for p in [
                    Path(__file__),
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_station_search.py",
                ]
            ],
            "case_ids": list(cases),
            "seeds": parent["seeds"],
            "arms": ARMS,
            "main_draw_seed_offset": 60000,
            "max_search_seconds": 1800,
            "max_audit_seconds": 1200,
            "execution_eligible": False,
            "training_eligible": False,
            "role": "observed_source_development_fresh_draws_with_separate_known_failure_replay",
        },
    )


def load(path):
    reg = json.loads(path.read_text())
    for ref in [reg[k] for k in ["protocol", "parent", "replay", "replay_plan"]] + reg[
        "implementations"
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    parent, cases, _ = load_parent(Path(reg["parent"]["path"]))
    replay = json.loads(Path(reg["replay"]["path"]).read_text())
    for k in ["trace", "sampler", "checkpoint"]:
        checked(Path(replay[k]["path"]), replay[k]["sha256"])
    assert (
        list(cases) == reg["case_ids"] and parent["seeds"] == reg["seeds"] and reg["arms"] == ARMS
    )
    return reg, parent, cases, replay


def search(args):
    start = time.monotonic()
    reg, parent, cases, replay = load(args.registration)
    root = args.registration.parent / "search"
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw_features_by_case = raw_features(cases)
    offsets = tensor(parent["search_offsets"])
    rows = []
    for ref in parent["cells"]:
        cell = json.loads(Path(ref["path"]).read_text())
        checked(Path(cell["checkpoint"]["path"]), cell["checkpoint"]["sha256"])
        saved = torch.load(cell["checkpoint"]["path"], map_location="cpu", weights_only=False)
        model = EventMixture(width=32).double()
        model.load_state_dict(saved["model"])
        model.eval()
        features = normalized(raw_features_by_case, saved["config"])
        seed = cell["seed"]
        jobs = [("main", name, seed + reg["main_draw_seed_offset"]) for name in cases]
        if seed == replay["checkpoint_seed"]:
            jobs.append(("replay", replay["case_id"], replay["draw_seed"]))
        for panel, name, draw_seed in jobs:
            case = cases[name]
            tick = time.monotonic()
            with torch.no_grad():
                candidates = sample_scenes(
                    model(features[name]), 256, torch.Generator().manual_seed(draw_seed)
                )[:136]
            sampling_seconds = time.monotonic() - tick
            initial = candidates[:8]
            if panel == "replay":
                np.testing.assert_array_equal(initial.numpy(), replay["initial_scenes"])

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

            for arm in ARMS:
                if time.monotonic() - start > reg["max_search_seconds"]:
                    raise TimeoutError("search cap")
                tick = time.monotonic()
                if arm == "raw":
                    scenes = initial
                    trace = {"initial": initial, "scenes": initial}
                elif arm == "rank":
                    with torch.no_grad():
                        values = torch.cat([query(chunk) for chunk in candidates.split(8)])
                        indices = torch.argsort(margin_slack(values), descending=True, stable=True)[
                            :8
                        ]
                        scenes = candidates[indices]
                    trace = {
                        "initial": initial,
                        "scenes": candidates,
                        "clearances": values,
                        "selected": indices,
                    }
                elif arm == "gradient":
                    scenes, trace = bounded_refine(initial, query)
                    trace["initial"] = initial
                else:
                    scenes, trace = station_search(initial, query, arm)
                seconds = time.monotonic() - tick
                if arm in ["gradient", *METHODS]:
                    difference = abs(trace["scenes"] - initial)
                    if torch.any(difference > initial.new_tensor([0.05, 0.03]) + 1e-12):
                        raise ValueError("original trust region exceeded")
                if panel == "replay" and arm == "gradient":
                    np.testing.assert_allclose(
                        scenes.numpy(), replay["refined_scenes"], rtol=0, atol=1e-12
                    )
                path = root / f"{panel}_{arm}_{seed}_{name}.npz"
                with path.open("xb") as handle:
                    np.savez_compressed(handle, **{k: v.detach().numpy() for k, v in trace.items()})
                rows.append(
                    {
                        "panel": panel,
                        "arm": arm,
                        "seed": seed,
                        "draw_seed": draw_seed,
                        "case_id": name,
                        "scenes": scenes.detach().numpy().tolist(),
                        "search_seconds": seconds,
                        "sampling_seconds": sampling_seconds,
                        "queries": 0 if arm == "raw" else 4624,
                        "trace": artifact(path),
                    }
                )
            print(json.dumps({"jobs": len(rows) // 6, "total": 49}), flush=True)
    write_new(
        root / "complete.json",
        {
            "registration": artifact(args.registration),
            "rows": rows,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def predicates(rows):
    summaries = summarize([r for r in rows if r["panel"] == "main"])
    results = {}
    for arm in METHODS:
        delta = {}
        for p in [41007, 41008, 42007, 42008]:
            counts = [
                next(r["valid"] for r in summaries if r["arm"] == a and r["carrier_seed"] == p)
                for a in [arm, "gradient"]
            ]
            delta[str(p)] = counts[0] - counts[1]
        selected = [
            r for r in rows if r["panel"] == "main" and r["arm"] == arm and r["split"] == "test"
        ]
        replay = next(r["valid"] for r in rows if r["panel"] == "replay" and r["arm"] == arm)
        results[arm] = {
            "test_parent_deltas": delta,
            "no_parent_loss_and_total_gain": min(delta.values()) >= 0 and sum(delta.values()) > 0,
            "rescued_gradient_failures": sum(r["rescued_gradient"] for r in selected),
            "lost_gradient_successes": sum(r["lost_gradient"] for r in selected),
            "no_paired_loss": sum(r["lost_gradient"] for r in selected) == 0,
            "replay_valid": replay,
            "replay_improves": replay > 5,
        }
    return summaries, results


def analyze(args):
    reg, parent, cases, replay = load(args.registration)
    root = args.registration.parent
    complete = json.loads((root / "search/complete.json").read_text())
    checked(args.registration, complete["registration"]["sha256"])
    expected = {("main", a, s, c) for a in ARMS for s in reg["seeds"] for c in cases}
    expected |= {("replay", a, replay["checkpoint_seed"], replay["case_id"]) for a in ARMS}
    if (
        len(complete["rows"]) != 294
        or {(r["panel"], r["arm"], r["seed"], r["case_id"]) for r in complete["rows"]} != expected
    ):
        raise ValueError("incomplete search")
    out = root / "audit"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    offsets = np.array(parent["audit_offsets"])
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
            raise ValueError("nonfinite or disagreeing geometry")
        path = (
            out
            / f'{prediction["panel"]}_{prediction["arm"]}_{prediction["seed"]}_{prediction["case_id"]}.npz'
        )
        with path.open("xb") as handle:
            np.savez_compressed(handle, scenes=scenes, offsets=offsets, clearances=values)
        rows.append(measure(prediction, case, values, error, artifact(path)))
        if (i + 1) % 24 == 0:
            print(json.dumps({"audited": i + 1, "total": 294}), flush=True)
    gradients = {
        (r["panel"], r["seed"], r["case_id"]): np.array(r["per_proposal_valid"])
        for r in rows
        if r["arm"] == "gradient"
    }
    for r in rows:
        if r["arm"] in METHODS:
            g = gradients[(r["panel"], r["seed"], r["case_id"])]
            v = np.array(r["per_proposal_valid"])
            r.update(rescued_gradient=int((v & ~g).sum()), lost_gradient=int((~v & g).sum()))
    assert (
        next(r["valid"] for r in rows if r["panel"] == "replay" and r["arm"] == "gradient")
        == replay["accepted_count"]
    )
    summary, results = predicates(rows)
    write_new(
        root / "result.json",
        {
            "registration": artifact(args.registration),
            "search": artifact(root / "search/complete.json"),
            "rows": rows,
            "summary": summary,
            "predicates": results,
            "audit_seconds": time.monotonic() - start,
            "search_seconds": complete["elapsed_seconds"],
            "search_queries": 49 * 5 * 4624,
            "audit_queries": 294 * 8 * 113 * 2,
            "crosscheck_queries": 294 * 2 * 113 * 2,
            "max_query_error_m": max(r["query_error_m"] for r in rows),
            "execution_eligible": False,
            "training_eligible": False,
        },
    )
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "search", "analyze"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT.parent / "research-data/groot-wbc/m2s-refinement-v1/registration.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/motion2scene/STATION_SEARCH_V1.md"
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
