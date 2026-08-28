#!/usr/bin/env python3
"""Train multi-obstacle LfLH on real clips, and score it against controls that decide the question.

The retracted first attempt compared a crippled learner against an analytic answer, which was not
a comparison. The controls here are the ones the review asked for:

* **random obstacles** -- the same parameterisation, drawn from the prior. If the learner cannot
  beat this it has learned nothing.
* **input-ablated learner** -- the trained model fed the corpus-mean profile instead of the clip's
  own. If this matches the learner, the model is not conditional and nothing about `q(C|p)` was
  tested. This is the control that exposed the retracted version as a constant.
* **no-KL learner** -- the same model without the term that opposes contraction, to separate a
  property of the objective from a property of the model class.

Two numbers per arm, never one: **reconstruction rate** (how often a sampled scene actually makes
the observed motion the preferred one) and **diversity** (how spread the sampled scenes are).
Reporting reconstruction alone is what let a constant look like a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.lflh import (  # noqa: E402
    ChoiceDecoder,
    ObstacleGeometry,
    sample_scenes,
    train,
)


def _diversity(parameters: np.ndarray) -> dict[str, float]:
    """Spread of sampled scenes, in interpretable units.

    ``station_sd`` and ``lateral_sd`` say whether the model puts obstacles in genuinely different
    places; ``side_entropy`` says whether it commits to one side of the route. A collapsed
    hallucinator scores near zero on all three while still reconstructing perfectly.
    """
    stations = parameters[..., 0].reshape(-1)
    lateral = parameters[..., 1].reshape(-1)
    height = parameters[..., 2].reshape(-1)
    left = float((lateral > 0.02).mean())
    right = float((lateral < -0.02).mean())
    centre = max(1e-9, 1.0 - left - right)
    entropy = -sum(share * np.log(share) for share in (max(left, 1e-9), max(right, 1e-9), centre))
    return {
        "station_sd": float(stations.std()),
        "lateral_sd_m": float(lateral.std()),
        "height_sd_m": float(height.std()),
        "side_entropy": float(entropy),
    }


def _score(model, sets, geometry, decoder, *, count: int, seed: int, profile_override=None):
    """Score scenes; ``profile_override`` replaces only the encoder input, never the candidates."""
    rates, diversity, chosen = [], [], []
    for row, item in enumerate(sets):
        extents = np.asarray(item["extents"], dtype=np.float64)
        costs = np.asarray(item["costs"], dtype=np.float64)
        batch = sample_scenes(
            model,
            extents,
            costs,
            item["observed_index"],
            str(item["motion_index"]),
            count=count,
            geometry=geometry,
            decoder=decoder,
            seed=seed + row,
            profile_override=profile_override,
        )
        rates.append(batch.reconstruction_rate)
        diversity.append(_diversity(batch.parameters))
        chosen.append(batch)
    summary = {
        "reconstruction_rate": float(np.mean(rates)),
        **{
            key: float(np.mean([item[key] for item in diversity]))
            for key in ("station_sd", "lateral_sd_m", "height_sd_m", "side_entropy")
        },
    }
    return summary, chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_training.json"
    )
    parser.add_argument("--scenes-out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text())
    sets = []
    for item in payload["candidate_sets"]:
        labels = item["labels"]
        # Ask the scene to explain a *non-nominal* motion; rotate the target across clips so the
        # corpus contains crouches and both tucks rather than one edit direction.
        wanted = [index for index, name in enumerate(labels) if name != "nominal"]
        if not wanted:
            continue
        item = dict(item)
        item["observed_index"] = wanted[len(sets) % len(wanted)]
        item["observed_label"] = labels[item["observed_index"]]
        sets.append(item)
    if not sets:
        raise SystemExit("no candidate sets available")

    extents = [np.asarray(item["extents"], dtype=np.float64) for item in sets]
    costs = [np.asarray(item["costs"], dtype=np.float64) for item in sets]
    observed = [item["observed_index"] for item in sets]
    stations = payload["stations"]
    geometry = ObstacleGeometry(stations=stations)
    decoder = ChoiceDecoder()

    print(f"training on {len(sets)} clips, {args.obstacles} obstacles, {args.steps} steps ...")
    model, report = train(
        extents,
        costs,
        observed,
        obstacles=args.obstacles,
        steps=args.steps,
        samples=args.samples,
        kl_weight=args.kl_weight,
        seed=args.seed,
        geometry=geometry,
        decoder=decoder,
    )
    print(f"  reconstruction {report.reconstruction:.4f}   mean sigma {report.mean_sigma:.4f}")

    learned, scenes = _score(model, sets, geometry, decoder, count=args.draws, seed=args.seed)

    # Control 1: the trained model fed the corpus-mean profile instead of each clip's own -- and
    # nothing else changed. Overwriting `extents` corrupts the scored candidates as well as the
    # encoder input, which conflates the encoder's input-dependence with a scoring artifact; the
    # artifact alone was measured larger than the effect. Only the profile is replaced here.
    mean_nominal = np.mean(np.stack([extents[row][0] for row in range(len(sets))]), axis=0)
    mean_observed = np.mean(
        np.stack([extents[row][observed[row]] for row in range(len(sets))]), axis=0
    )
    mean_profile = np.stack((mean_nominal, mean_observed), axis=0)
    ablated, _ = _score(
        model,
        sets,
        geometry,
        decoder,
        count=args.draws,
        seed=args.seed,
        profile_override=mean_profile,
    )

    # Control 2: obstacles drawn from the prior, no learning at all.
    class _Prior:
        def __call__(self, profiles):
            batch = profiles.shape[0]
            generator = torch.Generator().manual_seed(args.seed)
            mean = torch.randn(batch, args.obstacles, 6, generator=generator)
            return mean, torch.zeros_like(mean)

        def eval(self):
            return self

    random_arm, _ = _score(_Prior(), sets, geometry, decoder, count=args.draws, seed=args.seed)

    # Control 3: the same objective without the term that opposes contraction.
    bare_model, bare_report = train(
        extents,
        costs,
        observed,
        obstacles=args.obstacles,
        steps=args.steps,
        samples=args.samples,
        kl_weight=0.0,
        seed=args.seed,
        geometry=geometry,
        decoder=decoder,
    )
    bare, _ = _score(bare_model, sets, geometry, decoder, count=args.draws, seed=args.seed)

    summary = {
        "schema_version": "lflh_training_v2",
        "clips": len(sets),
        "obstacles": args.obstacles,
        "steps": args.steps,
        "kl_weight": args.kl_weight,
        "observed_labels": sorted({item["observed_label"] for item in sets}),
        "training": {
            "reconstruction": report.reconstruction,
            "mean_sigma": report.mean_sigma,
            "history": report.history,
        },
        "arms": {
            "learned": learned,
            "learned_input_ablated": ablated,
            "random_from_prior": random_arm,
            "learned_without_kl": {**bare, "mean_sigma": bare_report.mean_sigma},
        },
        "per_clip": [
            {
                "motion_index": item["motion_index"],
                "body_mode": item.get("body_mode", ""),
                "observed_label": item["observed_label"],
                "reconstruction_rate": batch.reconstruction_rate,
            }
            for item, batch in zip(sets, scenes)
        ],
        "contract": (
            "reconstruction is measured against a decoder that re-decides which candidate motion "
            "the scene prefers; it is not a physics verdict"
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\n%-28s %14s %10s %10s %10s" % ("arm", "reconstruct", "station sd", "lat sd", "side H"))
    print("-" * 78)
    for name, values in summary["arms"].items():
        print(
            "%-28s %13.1f%% %10.3f %10.3f %10.3f"
            % (
                name,
                100 * values["reconstruction_rate"],
                values["station_sd"],
                values["lateral_sd_m"],
                values["side_entropy"],
            )
        )
    if args.scenes_out:
        args.scenes_out.write_text(
            json.dumps(
                {
                    "schema_version": "lflh_scenes_v1",
                    "scenes": [
                        {
                            "motion_index": item["motion_index"],
                            "observed_label": item["observed_label"],
                            "station_xy_m": item["station_xy_m"],
                            "yaw_rad": item["yaw_rad"],
                            "obstacles": batch.parameters.tolist(),
                            "winner": batch.winner.tolist(),
                            "labels": item["labels"],
                        }
                        for item, batch in zip(sets, scenes)
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nscenes -> {args.scenes_out}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
