#!/usr/bin/env python3
"""How good can a hallucinated scene possibly be? The closed-form window, measured in 3-D.

A scene expresses a counterfactual only if it does two things at once: the observed motion must
clear every obstacle, and the cheapest rival must be struck by one. The largest symmetric margin
that can hold both is bounded by the separation between the two bodies -- which is exactly the
closed-form window ``W = [R_adapted + delta_clear, R_nominal - delta_strike]`` the geometric
pipeline solves, and it does not depend on the model at all.

This script brute-forces a single box per clip over station, lateral offset and height and reports
the best achievable ``min(clearance to observed, penetration of cheapest rival)``. That number is
the **ceiling**: no hallucinator, learned or otherwise, can exceed it. Reporting a learned model's
margin without it is meaningless, because a low margin may be the task rather than the model.

Distances are true hard minima over the capsule cloud, never the training-time soft minimum.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.sdf_decoder import (  # noqa: E402
    SdfChoiceDecoder,
)
from scripts.research.hallucination.train_lflh_sdf import build_clip  # noqa: E402

#: A slab wide enough to span the route and thin enough along it to be a doorway lintel rather than
#: a tunnel. Fixed, so the sweep reports a ceiling for one obstacle *shape* and stays comparable.
HALF_ALONG_M = 0.10
HALF_LATERAL_M = 0.45
HALF_VERTICAL_M = 0.15


#: These scripts do many small tensor operations, where torch's default of one thread per core
#: costs more in contention than it buys in parallelism -- measured at load average 96 on a 20-core
#: box with three of them running. Override with LFH_TORCH_THREADS if a run has the machine to
#: itself.
def _cap_threads() -> None:
    torch.set_num_threads(int(os.environ.get("LFH_TORCH_THREADS", "2")))


def ceiling_for_clip(
    clip: dict, decoder: SdfChoiceDecoder, *, laterals, heights, epsilons=()
) -> dict:
    observed = clip["observed"]
    rivals = [
        index
        for index in range(len(clip["clouds"]))
        if index != observed and clip["costs"][index] <= clip["costs"][observed] + 1e-9
    ]
    #: Necessity is measured against the candidate the robot would have chosen in an empty scene.
    cheapest = int(min(range(len(clip["costs"])), key=lambda index: float(clip["costs"][index])))
    points, radii = clip["clouds"][observed]
    best = {"score_m": -9.0, "score_all_m": -9.0}
    for epsilon in epsilons:
        best[f"score_eps_{epsilon:g}_m"] = -9.0
    for station in range(len(clip["yaw"])):
        yaw = float(clip["yaw"][station])
        base_x, base_y = clip["station_xy"][station]
        for lateral in laterals:
            centre_x = base_x - np.sin(yaw) * lateral
            centre_y = base_y + np.cos(yaw) * lateral
            for height in heights:
                box = {
                    "centre_x": torch.tensor([centre_x], dtype=torch.float32),
                    "centre_y": torch.tensor([centre_y], dtype=torch.float32),
                    "centre_z": torch.tensor([height + HALF_VERTICAL_M], dtype=torch.float32),
                    "half_along_m": torch.tensor([HALF_ALONG_M], dtype=torch.float32),
                    "half_lateral_m": torch.tensor([HALF_LATERAL_M], dtype=torch.float32),
                    "half_vertical_m": torch.tensor([HALF_VERTICAL_M], dtype=torch.float32),
                    "yaw": torch.tensor([yaw], dtype=torch.float32),
                }
                clear = float(decoder.exact_clearance(points, radii, box).min())
                # The score can never exceed the clearance, so a clearance already below the best
                # score cannot win; skipping the rival pass here is exact, not an approximation.
                if clear <= best["score_m"] and clear <= best["score_all_m"]:
                    continue
                depths = [
                    float(-decoder.exact_clearance(*clip["clouds"][rival], box).min())
                    for rival in rivals
                ]
                # Two readings of "the scene excludes the alternative", and they differ a lot once
                # the edit ladder has rungs between the observed motion and the nominal:
                #   any -- some cheaper candidate is struck (the weakest useful requirement);
                #   all -- every cheaper candidate is struck, which is what the training loss asks
                #          for and what makes the observed motion the *only* survivor.
                score_any = min(clear, max(depths))
                score_all = min(clear, min(depths))
                if score_any > best["score_m"]:
                    best.update(
                        {
                            "score_m": score_any,
                            "clearance_m": clear,
                            "strike_m": max(depths),
                            "station": station,
                            "lateral_m": float(lateral),
                            "underside_height_m": float(height),
                        }
                    )
                if score_all > best["score_all_m"]:
                    best.update(
                        {
                            "score_all_m": score_all,
                            "clearance_all_m": clear,
                            "strike_all_m": min(depths),
                            "station_all": station,
                            "lateral_all_m": float(lateral),
                            "underside_height_all_m": float(height),
                        }
                    )
                # The reading that matches the project's own definition of the inverse set,
                # S_{eps,delta}(tau) = {S : Feasible, Regret <= eps, Necessity >= delta}. The two
                # rules above are its endpoints. Necessity is measured against the *cheapest*
                # candidate -- the nominal, the motion that would have been executed had the scene
                # been empty. Regret asks whether some cheaper candidate is also feasible, and by
                # how much cost: an intermediate rung of the same operator is not a counterexample
                # unless the robot could have saved more than eps by taking it.
                if epsilons:
                    clears = [
                        float(decoder.exact_clearance(*clip["clouds"][index], box).min())
                        for index in range(len(clip["clouds"]))
                    ]
                    feasible_costs = [
                        float(clip["costs"][index])
                        for index, value in enumerate(clears)
                        if value > 0.0
                    ]
                    regret = (
                        float(clip["costs"][observed]) - min(feasible_costs)
                        if feasible_costs
                        else float("inf")
                    )
                    necessity = -clears[cheapest]
                    score_eps = min(clear, necessity)
                    for epsilon in epsilons:
                        key = f"score_eps_{epsilon:g}_m"
                        if regret <= epsilon and score_eps > best.get(key, -9.0):
                            best[key] = score_eps
                            best[f"regret_eps_{epsilon:g}"] = regret
    best["motion_index"] = clip["motion_index"]
    best["body_mode"] = clip["body_mode"]
    best["rivals"] = [clip["labels"][r] for r in rivals]
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--targets", nargs="+", default=["crouch_040", "crouch_055", "crouch_070"])
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--height-step", type=float, default=0.01)
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.25, 0.5],
        help="regret tolerances for the S_{eps,delta} reading of the ceiling",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/scene_ceiling.json"
    )
    args = parser.parse_args()
    _cap_threads()

    payload = json.loads(args.candidates.read_text())
    decoder = SdfChoiceDecoder()
    laterals = np.arange(-0.4, 0.41, 0.1)
    heights = np.arange(0.8, 2.2, args.height_step)

    results = {}
    for target in args.targets:
        rows, built = [], 0
        for item in payload["candidate_sets"]:
            if built >= args.clips:
                break
            clip = build_clip(item, target, stride=args.stride)
            if clip is None:
                continue
            built += 1
            row = ceiling_for_clip(
                clip, decoder, laterals=laterals, heights=heights, epsilons=tuple(args.epsilon)
            )
            rows.append(row)
            print(
                f"{target} clip {row['motion_index']:03d} "
                f"ceiling(any) {1000 * row['score_m']:6.1f} mm  "
                f"ceiling(all) {1000 * row['score_all_m']:6.1f} mm  "
                f"rivals {len(row['rivals'])}"
            )
        scores = [1000 * r["score_m"] for r in rows]
        scores_all = [1000 * r["score_all_m"] for r in rows]
        results[target] = {
            "clips": rows,
            "rivals": rows[0]["rivals"],
            "median_ceiling_mm": float(np.median(scores)),
            "min_ceiling_mm": float(np.min(scores)),
            "max_ceiling_mm": float(np.max(scores)),
            "median_ceiling_all_rivals_mm": float(np.median(scores_all)),
            "median_ceiling_by_epsilon_mm": {
                f"{epsilon:g}": float(
                    np.median([1000 * r[f"score_eps_{epsilon:g}_m"] for r in rows])
                )
                for epsilon in args.epsilon
            },
            "min_ceiling_all_rivals_mm": float(np.min(scores_all)),
            "max_ceiling_all_rivals_mm": float(np.max(scores_all)),
        }
        print(
            f"  -> {target}: any-rival median {np.median(scores):.1f} mm "
            f"({min(scores):.1f}..{max(scores):.1f}); "
            f"all-rival median {np.median(scores_all):.1f} mm "
            f"({min(scores_all):.1f}..{max(scores_all):.1f}), "
            f"{len(rows[0]['rivals'])} rivals"
        )
        for epsilon in args.epsilon:
            values = [1000 * r[f"score_eps_{epsilon:g}_m"] for r in rows]
            print(f"       regret <= {epsilon:4g}: median {np.median(values):6.1f} mm")
        print()

    amplitudes = [int(t.rsplit("_", 1)[1]) for t in args.targets if t.rsplit("_", 1)[1].isdigit()]
    medians = [results[t]["median_ceiling_mm"] for t in args.targets]
    fit = np.polyfit(amplitudes, medians, 1) if len(amplitudes) >= 2 else [float("nan")] * 2
    print(f"ceiling_mm ~= {fit[0]:.3f} * amplitude_mm + {fit[1]:.1f}")

    args.out.write_text(
        json.dumps(
            {
                "schema_version": "scene_ceiling_v1",
                "obstacle_half_extents_m": [HALF_ALONG_M, HALF_LATERAL_M, HALF_VERTICAL_M],
                "targets": results,
                "slope_mm_per_mm": float(fit[0]),
                "intercept_mm": float(fit[1]),
                "contract": (
                    "exhaustive single-box search over station x lateral x height; distances are "
                    "true hard minima over the capsule cloud. This is an upper bound on the "
                    "two-sided margin any hallucinator can achieve for this obstacle shape."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
