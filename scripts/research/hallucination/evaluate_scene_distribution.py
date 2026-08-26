#!/usr/bin/env python3
"""Score q(S | executed pair) against baselines, on the eps-delta membership test.

The lesson of the refuted archetype model is that a proposal distribution is only as good as the
baseline it is measured against. So every number here is reported beside three controls that need
no trajectory conditioning at all:

* **random placement** -- a face height drawn from the range the corpus's obstacles occupy, with
  no knowledge of the executed pair. This is the "put an object near the motion" strawman the
  method has to beat.
* **route midpoint** -- correct height, arbitrary station. Isolates *where along the route*.
* **uniform in support** -- the full closed-form support, sampled flat. Isolates what the shaped
  proposal adds over merely knowing the support.

A proposal is admitted when it is in support, geometrically feasible for the adaptation, carries
regret <= eps, necessity >= delta, and realism >= r0. None of these is a physics verdict; they are
the computable necessary conditions, and physics still adjudicates every promoted scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.scene_distribution import (  # noqa: E402
    ARCHETYPE_DIMENSION_PRIORS,
    CriticalAtom,
    SceneDistribution,
    realism_score,
    score_atom,
)

ENGINEERING_MARGIN_M = 0.018044
#: Range of face heights the corpus's authored obstacles occupy, for the uninformed baseline.
RANDOM_COORDINATE_RANGE_M = (1.10, 1.45)


def atoms_from_reports(paths: list[Path]) -> list[CriticalAtom]:
    atoms: list[CriticalAtom] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entry in payload.get("scenes", []) + payload.get("selected", []):
            spec_path = REPO_ROOT / entry["spec"] if "spec" in entry else None
            if spec_path is None or not spec_path.exists():
                continue
            spec = json.loads(spec_path.read_text())
            binding = spec["binding"]
            atoms.append(
                CriticalAtom(
                    source_id=entry.get("pair_id") or entry.get("source_pair_id"),
                    station_xy_m=tuple(spec["binding_station_xy_m"]),
                    route_axis=spec["route_axis"],
                    route_progress=float(entry.get("station_progress", 0.55)),
                    face_along_route_m=float(spec["face_extent"]["along_route_m"]),
                    face_across_route_m=float(spec["face_extent"]["across_route_m"]),
                    reach_nominal_m=float(binding["reach_orig_m"]),
                    reach_adapted_m=float(binding["reach_edit_m"]),
                    delta_clear_m=ENGINEERING_MARGIN_M,
                    delta_strike_m=ENGINEERING_MARGIN_M,
                    binding_keypoint=binding["keypoint"],
                )
            )
    unique: dict[str, CriticalAtom] = {}
    for atom in atoms:
        unique.setdefault(atom.source_id, atom)
    return list(unique.values())


def _score_coordinate(atom: CriticalAtom, coordinate: float, archetype: str, rng) -> dict:
    """Score an arbitrary coordinate, which may fall outside the support."""
    prior = ARCHETYPE_DIMENSION_PRIORS[archetype]
    thickness = float(rng.uniform(*prior["thickness_m"]))
    across = float(rng.uniform(*prior["across_m"]))
    in_support = atom.lower_m <= coordinate <= atom.upper_m and atom.width_m > 0
    if in_support:
        xi = (coordinate - atom.lower_m) / atom.width_m
        score = score_atom(atom, archetype=archetype, xi=xi, thickness_m=thickness, across_m=across)
        return {
            "in_support": True,
            "regret_mm": score.regret_mm,
            "necessity_mm": score.necessity_mm,
            "realism": score.realism,
            "feasible_geometric": score.feasible_geometric,
        }
    # Out of support: still report *why*, because the failure mode is the interesting part.
    return {
        "in_support": False,
        "regret_mm": float("nan"),
        "necessity_mm": 1000 * (atom.reach_nominal_m - coordinate),
        "realism": realism_score(archetype, thickness, across),
        "feasible_geometric": bool(coordinate - atom.delta_clear_m >= atom.reach_adapted_m),
        "nominal_would_clear": bool(coordinate >= atom.reach_nominal_m),
        "adapted_would_strike": bool(coordinate < atom.reach_adapted_m),
    }


def evaluate(
    atoms: list[CriticalAtom],
    *,
    samples: int,
    seed: int,
    epsilon_mm: float,
    delta_mm: float,
    realism_floor: float,
    regret_budget_mm: float | None,
) -> dict:
    counts = {"shelf_plank": 4, "ibeam": 4, "door_lintel": 3, "hanging_panel": 1}
    shaped = SceneDistribution(atoms, counts, regret_budget_mm=regret_budget_mm)
    rng = np.random.default_rng(seed)

    def summarise(rows: list[dict], label: str) -> dict:
        admitted = [
            row
            for row in rows
            if row["in_support"]
            and row["feasible_geometric"]
            and row["regret_mm"] <= epsilon_mm
            and row["necessity_mm"] >= delta_mm
            and row["realism"] >= realism_floor
        ]
        in_support = [row for row in rows if row["in_support"]]
        regrets = [row["regret_mm"] for row in in_support]
        return {
            "sampler": label,
            "samples": len(rows),
            "in_support_rate": len(in_support) / max(len(rows), 1),
            "admitted_rate": len(admitted) / max(len(rows), 1),
            "median_regret_mm": float(np.median(regrets)) if regrets else None,
            "median_necessity_mm": (
                float(np.median([row["necessity_mm"] for row in in_support]))
                if in_support
                else None
            ),
            "median_realism": float(np.median([row["realism"] for row in rows])),
            "nominal_would_clear_rate": float(
                np.mean([row.get("nominal_would_clear", False) for row in rows])
            ),
            "adapted_would_strike_rate": float(
                np.mean([row.get("adapted_would_strike", False) for row in rows])
            ),
        }

    results = []

    shaped_rows = []
    for sample in shaped.sample(samples, seed=seed):
        atom = next(a for a in shaped.atoms if a.source_id == sample.source_id)
        score = score_atom(
            atom,
            archetype=sample.archetype,
            xi=sample.xi,
            thickness_m=sample.thickness_m,
            across_m=sample.face_across_route_m,
        )
        shaped_rows.append(
            {
                "in_support": score.in_support,
                "regret_mm": score.regret_mm,
                "necessity_mm": score.necessity_mm,
                "realism": score.realism,
                "feasible_geometric": score.feasible_geometric,
            }
        )
    results.append(summarise(shaped_rows, "q_LFH (shaped, regret-budgeted)"))

    usable = shaped.atoms
    archetypes = sorted(ARCHETYPE_DIMENSION_PRIORS)

    uniform_rows = []
    for _ in range(samples):
        atom = usable[int(rng.integers(len(usable)))]
        xi = float(rng.uniform(0.0, 1.0))
        uniform_rows.append(
            _score_coordinate(
                atom, atom.coordinate_at(xi), archetypes[int(rng.integers(len(archetypes)))], rng
            )
        )
    results.append(summarise(uniform_rows, "uniform in support"))

    random_rows = []
    for _ in range(samples):
        atom = usable[int(rng.integers(len(usable)))]
        coordinate = float(rng.uniform(*RANDOM_COORDINATE_RANGE_M))
        random_rows.append(
            _score_coordinate(atom, coordinate, archetypes[int(rng.integers(len(archetypes)))], rng)
        )
    results.append(summarise(random_rows, "random placement (no pair knowledge)"))

    midpoint_rows = []
    for _ in range(samples):
        atom = usable[int(rng.integers(len(usable)))]
        # Correct height band, but the station is not the one the executed pair separates at, so
        # the face is evaluated against a body that is not crossing it: model that as the nominal
        # passing under an arbitrary offset of its own reach.
        coordinate = atom.reach_nominal_m + float(rng.uniform(-0.05, 0.10))
        midpoint_rows.append(
            _score_coordinate(atom, coordinate, archetypes[int(rng.integers(len(archetypes)))], rng)
        )
    results.append(summarise(midpoint_rows, "route-midpoint station"))

    return {
        "schema_version": "lfh_scene_distribution_eval_v1",
        "atoms": len(atoms),
        "atoms_with_positive_window": len(usable),
        "atoms_refused_by_margin": shaped.refused,
        "admission_rule": {
            "epsilon_mm": epsilon_mm,
            "delta_mm": delta_mm,
            "realism_floor": realism_floor,
            "regret_budget_mm": regret_budget_mm,
        },
        "note": (
            "geometric necessary conditions only; physics adjudicates every promoted scene. "
            "Regret and necessity share one budget fixed by the executed window, so a sampler "
            "cannot improve both at once."
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        default=None,
        help="e17/e6-style scene report(s) carrying spec paths",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon-mm", type=float, default=15.0)
    parser.add_argument("--delta-mm", type=float, default=20.0)
    parser.add_argument("--realism-floor", type=float, default=1.0)
    parser.add_argument("--regret-budget-mm", type=float, default=15.0)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/scene_distribution_eval.json"
    )
    args = parser.parse_args()

    reports = args.report or [
        REPO_ROOT / "docs/hallucination/e17_ladder_scenes.json",
        REPO_ROOT / "docs/hallucination/e6_crouch_cpu.json",
    ]
    atoms = atoms_from_reports(list(reports))
    if not atoms:
        raise SystemExit("no critical atoms recovered from the supplied reports")
    report = evaluate(
        atoms,
        samples=args.samples,
        seed=args.seed,
        epsilon_mm=args.epsilon_mm,
        delta_mm=args.delta_mm,
        realism_floor=args.realism_floor,
        regret_budget_mm=args.regret_budget_mm,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"{report['atoms']} atoms ({report['atoms_with_positive_window']} with a window, "
        f"{report['atoms_refused_by_margin']} refused by margin)"
    )
    print(
        f"{'sampler':38s} {'in-support':>11s} {'admitted':>9s} "
        f"{'regret':>10s} {'necessity':>11s}"
    )
    print("-" * 82)
    for row in report["results"]:
        regret = row["median_regret_mm"]
        necessity = row["median_necessity_mm"]
        regret_text = f"{regret:.1f} mm" if regret is not None else "-"
        necessity_text = f"{necessity:.1f} mm" if necessity is not None else "-"
        print(
            f"{row['sampler']:38s} {row['in_support_rate']:10.1%} {row['admitted_rate']:9.1%} "
            f"{regret_text:>10s} {necessity_text:>11s}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
