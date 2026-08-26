#!/usr/bin/env python3
"""Does a learned LfLH-style hallucinator recover the humanoid inverse set, or collapse onto it?

LfLH and Dyna-LfLH both report that the learned obstacle distribution mode-collapses and cannot
cover the space of configurations that explain a trajectory. In the 2-D mobile-robot case that is
hard to quantify, because the true inverse set is not known in closed form.

In the humanoid overhead case it is. The set of face coordinates that make the nominal strike and
the adaptation clear is exactly `[R_adapted(u) + delta, R_nominal(u) - delta]` at every station, so
a learned distribution can be scored against ground truth rather than against itself.

This runs the LfLH objective -- reparameterised sampling through a fixed differentiable decoder,
with reconstruction, prior and clearance terms -- on reach profiles measured from real clips, and
compares it against uniform sampling over the exact support on two axes: how often a sample is a
valid explanation, and how much of the explanation set it reaches.

The surrogate decoder is a study instrument, not a proposal mechanism. No scene is authored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.learned_hallucinator import (  # noqa: E402
    ReachPair,
    coverage,
    sample_placements,
    train,
)
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402
from scripts.research.hallucination.screen_crouch_ladder import WINDOW_FRACTION  # noqa: E402

MARGIN_M = 0.018044
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 2.0
STATIONS = 24


def reach_pair(index: int, path: Path, target_drop_m: float) -> ReachPair | None:
    qpos = np.loadtxt(path, delimiter=",")
    adapted_qpos, report = local_crouch(
        qpos, 0.55, target_drop_m=target_drop_m, window=WINDOW_FRACTION
    )
    if not report.root_path_preserved or report.excursion_capped:
        return None
    root = np.asarray(qpos[:, :2], dtype=np.float64)
    progress = route_progress(root)
    span = np.ptp(root, axis=0)
    axis = "x" if span[0] >= span[1] else "y"
    nominal_tracks = extract_keypoints(payload_from_reference(qpos))
    adapted_tracks = extract_keypoints(payload_from_reference(adapted_qpos))

    fractions = np.linspace(0.25, 0.85, STATIONS)
    stations, nominal, adapted = [], [], []
    for fraction in fractions:
        frame = int(np.argmin(np.abs(progress - fraction)))
        station = (float(root[frame, 0]), float(root[frame, 1]))
        values = []
        for tracks in (nominal_tracks, adapted_tracks):
            face = overhead_face_reach(
                tracks, station, axis, FACE_ALONG_M, FACE_ACROSS_M, require_all_groups=False
            )
            values.append(float(face.reach_m) if np.isfinite(face.reach_m) else np.nan)
        if not all(np.isfinite(values)):
            continue
        stations.append(float(fraction))
        nominal.append(values[0])
        adapted.append(values[1])
    if len(stations) < STATIONS:
        return None
    return ReachPair(
        source_id=f"clip_{index:03d}",
        stations_m=np.asarray(stations),
        nominal_reach_m=np.asarray(nominal),
        adapted_reach_m=np.asarray(adapted),
    )


def uniform_over_support(pair: ReachPair, count: int, rng) -> np.ndarray:
    """The sampler we actually ship: pick a feasible station, then uniform inside its window."""
    feasible = np.flatnonzero(pair.feasible_mask(MARGIN_M))
    if not len(feasible):
        return np.zeros((0, 2))
    picks = rng.choice(feasible, size=count)
    lower = pair.adapted_reach_m[picks] + MARGIN_M
    upper = pair.nominal_reach_m[picks] - MARGIN_M
    return np.stack((picks.astype(float), rng.uniform(lower, upper)), axis=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument(
        "--case-study", type=Path, default=REPO_ROOT / "docs/hallucination/case_study.json"
    )
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--target-drop-mm", type=float, default=70.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_comparison.json"
    )
    args = parser.parse_args()

    cases = json.loads(args.case_study.read_text())["cases"]
    wanted = [case["motion_index"] for case in cases if case["decision"] == "propose"]
    pairs: list[ReachPair] = []
    for index in wanted:
        if len(pairs) >= args.clips:
            break
        matches = sorted(args.source_dir.glob(f"{index:03d}_*.csv"))
        if not matches:
            continue
        pair = reach_pair(index, matches[0], args.target_drop_mm / 1000.0)
        if pair is not None and pair.feasible_mask(MARGIN_M).any():
            pairs.append(pair)
            print(f"  profiled {pair.source_id}", flush=True)
    if len(pairs) < 4:
        raise SystemExit("too few usable reach pairs for the comparison")

    print(f"\ntraining LfLH-style hallucinator on {len(pairs)} clips ...")
    model, report = train(pairs, steps=args.steps, seed=args.seed)
    print(
        f"  final loss {report.final_loss:.4f}   "
        f"sigma_station {report.mean_sigma_station:.4f}   "
        f"sigma_coordinate {report.mean_sigma_coordinate_mm:.3f} mm"
    )

    learned = sample_placements(model, pairs, count=args.samples, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    rows = []
    for pair in pairs:
        learned_score = coverage(learned[pair.source_id], pair, margin_m=MARGIN_M)
        uniform_score = coverage(
            uniform_over_support(pair, args.samples, rng), pair, margin_m=MARGIN_M
        )
        rows.append(
            {
                "source_id": pair.source_id,
                "feasible_stations": int(pair.feasible_mask(MARGIN_M).sum()),
                "support_area_mm_m": 1000 * pair.support_area_m2(MARGIN_M),
                "learned": learned_score,
                "uniform_over_support": uniform_score,
            }
        )

    def mean(name: str, key: str) -> float:
        return float(np.mean([row[name][key] for row in rows]))

    summary = {
        "schema_version": "lfh_lflh_comparison_v1",
        "clips": len(pairs),
        "stations_per_clip": STATIONS,
        "commanded_drop_mm": args.target_drop_mm,
        "engineering_margin_mm": 1000 * MARGIN_M,
        "training": {
            "steps": report.steps,
            "final_loss": report.final_loss,
            "mean_sigma_station": report.mean_sigma_station,
            "mean_sigma_coordinate_mm": report.mean_sigma_coordinate_mm,
            "history": report.history,
        },
        "learned_hallucinator": {
            "valid_rate": mean("learned", "valid_rate"),
            "occupancy": mean("learned", "occupancy"),
        },
        "uniform_over_closed_form_support": {
            "valid_rate": mean("uniform_over_support", "valid_rate"),
            "occupancy": mean("uniform_over_support", "occupancy"),
        },
        "interpretation": (
            "valid_rate is the share of sampled placements that actually separate the pair; "
            "occupancy is the share of the exact feasible set reached. A single Gaussian cannot "
            "hold both high, which is the mode collapse LfLH and Dyna-LfLH report."
        ),
        "surrogate_decoder_is_a_study_instrument": True,
        "per_clip": rows,
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\n%-34s %10s %10s" % ("sampler", "valid", "occupancy"))
    print("-" * 56)
    for name, key in (
        ("learned LfLH-style hallucinator", "learned_hallucinator"),
        ("uniform over closed-form support", "uniform_over_closed_form_support"),
    ):
        print(
            "%-34s %9.1f%% %9.1f%%"
            % (name, 100 * summary[key]["valid_rate"], 100 * summary[key]["occupancy"])
        )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
