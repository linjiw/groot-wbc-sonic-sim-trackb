#!/usr/bin/env python3
"""Frozen seven-arm inference comparison on eight fresh source motions."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_carrier_learning import DOMAIN
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_fresh_sources import SOURCE_IDS, load_cases
from motion2scene_refinement_study import measure
from motion2scene_source_phase import summarize
from motion2scene_timing_diagnostic import artifact, checked, write_new
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

ARMS = ["raw", "rank", "gradient", *METHODS, "uniform_pattern"]


def uniform_initial(seed):
    return tensor([0.1, 1.1]) + torch.rand(
        (8, 2), dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
    ) * tensor([0.8, 0.35])


def load(path):
    reg, parent, cases = load_cases(path)
    if reg["arms"] != ARMS:
        raise ValueError("registered arms changed")
    return reg, parent, cases


def search(args):
    start = time.monotonic()
    reg, parent, cases = load(args.registration)
    root = args.registration.parent / "search"
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw_features_by_case = raw_features(cases)
    offsets = tensor(parent["search_offsets"])
    rows = []
    preprocessing_seconds = time.monotonic() - start
    checkpoint_setup_seconds = 0.0
    inherited_training_seconds = 0.0
    for ref in parent["cells"]:
        setup_start = time.monotonic()
        cell = json.loads(Path(ref["path"]).read_text())
        checked(Path(cell["checkpoint"]["path"]), cell["checkpoint"]["sha256"])
        saved = torch.load(cell["checkpoint"]["path"], map_location="cpu", weights_only=False)
        model = EventMixture(width=32).double()
        model.load_state_dict(saved["model"])
        model.eval()
        features = normalized(raw_features_by_case, saved["config"])
        if saved["arm"] != "all8" or saved["budget"] != 1200:
            raise ValueError("unexpected selected checkpoint")
        if set(saved["config"]["parents"]) & set(SOURCE_IDS):
            raise ValueError("audit source leaked into fitting")
        checkpoint_setup_seconds += time.monotonic() - setup_start
        inherited_training_seconds += cell["cumulative_fitting_seconds"]
        seed = cell["seed"]
        jobs = [("main", name, seed + reg["main_draw_seed_offset"]) for name in cases]
        for panel, name, draw_seed in jobs:
            case = cases[name]
            tick = time.monotonic()
            with torch.no_grad():
                candidates = sample_scenes(
                    model(features[name]), 256, torch.Generator().manual_seed(draw_seed)
                )[:136]
            sampling_seconds = time.monotonic() - tick
            initial = candidates[:8]
            uniform_start = time.monotonic()
            uniform_seed = seed + reg["uniform_seed_offset"]
            uniform = uniform_initial(uniform_seed)
            uniform_sampling_seconds = time.monotonic() - uniform_start

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
                elif arm == "uniform_pattern":
                    scenes, trace = station_search(uniform, query, "pattern")
                else:
                    scenes, trace = station_search(initial, query, arm)
                seconds = time.monotonic() - tick
                if arm in ["gradient", *METHODS, "uniform_pattern"]:
                    difference = abs(trace["scenes"] - trace["initial"])
                    if torch.any(difference > initial.new_tensor([0.05, 0.03]) + 1e-12):
                        raise ValueError("original trust region exceeded")
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
                        "uniform_sampling_seconds": uniform_sampling_seconds,
                        "uniform_seed": uniform_seed,
                        "queries": 0 if arm == "raw" else 4624,
                        "trace": artifact(path),
                    }
                )
            print(json.dumps({"jobs": len(rows) // 7, "total": 48}), flush=True)
    write_new(
        root / "complete.json",
        {
            "registration": artifact(args.registration),
            "rows": rows,
            "elapsed_seconds": time.monotonic() - start,
            "preprocessing_seconds": preprocessing_seconds,
            "checkpoint_setup_seconds": checkpoint_setup_seconds,
            "inherited_training_seconds": inherited_training_seconds,
        },
    )


def predicates(rows):
    summaries = summarize(rows)

    def counts(arm, source):
        return next(
            r["valid"] for r in summaries if r["arm"] == arm and r["carrier_seed"] == source
        )

    uniform_delta = {
        str(s): counts("pattern", s) - counts("uniform_pattern", s) for s in SOURCE_IDS
    }
    gradient_delta = {str(s): counts("pattern", s) - counts("gradient", s) for s in SOURCE_IDS}
    pattern = [r for r in rows if r["arm"] == "pattern"]
    return summaries, {
        "pattern_minus_uniform_by_source": uniform_delta,
        "learned_pattern_strictly_better_on_every_source": min(uniform_delta.values()) > 0,
        "pattern_minus_gradient_by_source": gradient_delta,
        "no_source_loss_and_total_gain_over_gradient": min(gradient_delta.values()) >= 0
        and sum(gradient_delta.values()) > 0,
        "rescued_gradient_failures": sum(r["rescued_gradient"] for r in pattern),
        "lost_gradient_successes": sum(r["lost_gradient"] for r in pattern),
        "no_paired_gradient_loss": sum(r["lost_gradient"] for r in pattern) == 0,
    }


def analyze(args):
    reg, parent, cases = load(args.registration)
    root = args.registration.parent
    complete = json.loads((root / "search/complete.json").read_text())
    checked(args.registration, complete["registration"]["sha256"])
    expected = {("main", a, s, c) for a in ARMS for s in reg["seeds"] for c in cases}
    if (
        len(complete["rows"]) != 336
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
            print(json.dumps({"audited": i + 1, "total": 336}), flush=True)
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
            "preprocessing_seconds": complete["preprocessing_seconds"],
            "checkpoint_setup_seconds": complete["checkpoint_setup_seconds"],
            "inherited_training_seconds": complete["inherited_training_seconds"],
            "sampling_seconds": sum(r["sampling_seconds"] for r in rows if r["arm"] == "raw"),
            "uniform_sampling_seconds": sum(
                r["uniform_sampling_seconds"] for r in rows if r["arm"] == "uniform_pattern"
            ),
            "search_queries": 48 * 6 * 4624,
            "audit_queries": 336 * 8 * 113 * 2,
            "crosscheck_queries": 336 * 2 * 113 * 2,
            "max_query_error_m": max(r["query_error_m"] for r in rows),
            "execution_eligible": False,
            "training_eligible": False,
        },
    )
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "analyze"])
    parser.add_argument("--registration", type=Path, required=True)
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
