#!/usr/bin/env python3
"""Sample and freeze evaluation scenes before anything is tuned against them.

Every counterfactual family built so far is *fitted*: the obstacle is placed where a
particular executed trajectory's swept volume separates from another's. That makes those
scenes training material and nothing else. Evaluating on them answers a question nobody
asked -- whether a model can recover geometry the generator derived from the very
trajectories it was trained on.

So the test scenes are sampled independently of any motion, and frozen now, while there is
still nothing to tune them against. Later is too late: once an adaptation operator and a
selector exist, any resampling of these parameters is indistinguishable from choosing the
ones that flatter them, including to the person doing it.

**What freezing does and does not permit.** Running any number of candidate motions through
these scenes to obtain physics labels is fine and expected -- that is how the ground truth is
made. Moving an obstacle, changing a height, or dropping a scene because a result came out
badly is not. The manifest carries a SHA-256 over the parameters so a later run can prove the
geometry is unchanged.

Parameters are drawn from ranges that bracket what the corpus can execute, deliberately
including values where *no* behaviour succeeds and values where *every* behaviour does. A
test set holding only discriminative scenes would measure something easier than the real
task, in which most geometry is uninformative and the model has to notice which is not.

Usage::

    python scripts/research/freeze_scene_first_testset.py --out data/splits/scene_first_v1.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

#: Fixed seed. The point of a frozen set is that anyone can regenerate it exactly.
SEED = 20260818

#: Ranges per regime. Chosen to bracket the corpus's measured envelopes rather than to
#: guarantee interesting scenes: a plain walk's silhouette peak is 1.248 m and a retargeted
#: crouch reaches 1.036 m, so shelves from 0.95 to 1.45 m span "nothing fits" through
#: "everything fits". Lateral half-widths run 0.203 m for a walk to 0.389 m for a carry.
RANGES = {
    "overhead": {
        "shelf_underside_m": (0.95, 1.45),
        "shelf_depth_m": (0.30, 0.80),
        "station_fraction": (0.25, 0.75),
    },
    "lateral": {
        "gap_half_width_m": (0.18, 0.55),
        "gap_length_m": (0.40, 1.20),
        "station_fraction": (0.25, 0.75),
    },
    "floor": {
        "obstacle_height_m": (0.05, 0.30),
        "obstacle_depth_m": (0.10, 0.40),
        "station_fraction": (0.25, 0.75),
    },
}


@dataclass(frozen=True)
class TestScene:
    scene_id: str
    regime: str
    parameters: dict
    #: Route length the station fraction is measured along, in metres.
    route_length_m: float


def sample(regime: str, count: int, rng: np.random.Generator) -> list[TestScene]:
    ranges = RANGES[regime]
    scenes = []
    for index in range(count):
        route = float(rng.uniform(3.5, 5.5))
        parameters = {
            name: round(float(rng.uniform(low, high)), 4)
            for name, (low, high) in ranges.items()
        }
        scenes.append(
            TestScene(
                scene_id=f"sf_{regime}_{index:03d}",
                regime=regime,
                parameters=parameters,
                route_length_m=round(route, 4),
            )
        )
    return scenes


def fingerprint(scenes: list[TestScene]) -> str:
    payload = json.dumps([asdict(s) for s in scenes], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-regime", type=int, default=10)
    parser.add_argument(
        "--regimes", nargs="+", default=["overhead", "lateral", "floor"],
        choices=sorted(RANGES),
    )
    args = parser.parse_args()

    if args.out.exists():
        existing = json.loads(args.out.read_text())
        print(f"REFUSING to overwrite a frozen test set: {args.out}")
        print(f"  frozen at   {existing.get('frozen_at_commit', 'unknown commit')}")
        print(f"  fingerprint {existing.get('fingerprint')}")
        print("  A frozen set that can be resampled is not frozen. Delete it deliberately,")
        print("  in a commit that says why, if it genuinely has to change.")
        return 2

    rng = np.random.default_rng(SEED)
    scenes: list[TestScene] = []
    for regime in args.regimes:
        scenes.extend(sample(regime, args.per_regime, rng))

    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "regimes": args.regimes,
        "per_regime": args.per_regime,
        "scenes": [asdict(s) for s in scenes],
        "fingerprint": fingerprint(scenes),
        "rules": [
            "Running any candidate motion through these scenes to obtain a physics label is"
            " expected; that is how the ground truth is made.",
            "Moving an obstacle, changing a parameter, or dropping a scene because a result"
            " was unfavourable is not permitted.",
            "These scenes must never be used for training or model selection.",
            "Fitted counterfactual families are training material only, because their"
            " geometry is derived from executed swept volumes.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"froze {len(scenes)} scene(s) across {len(args.regimes)} regime(s)")
    for regime in args.regimes:
        group = [s for s in scenes if s.regime == regime]
        key = next(iter(RANGES[regime]))
        values = [s.parameters[key] for s in group]
        print(f"  {regime:9s} {len(group):2d} scenes, {key} spans "
              f"{min(values):.3f} to {max(values):.3f}")
    print(f"\nfingerprint {manifest['fingerprint']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
