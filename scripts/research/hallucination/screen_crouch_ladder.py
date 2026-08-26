#!/usr/bin/env python3
"""CPU-screen a crouch amplitude ladder over the whole gated clip pool.

The CAL3 screen swept four target drops and then discarded three of them, recommending only the
historically verified 80 mm command.  E9a shows the cost of that: a fresh motion was carried to
physics at 80 mm and rejected on tracking, with no evidence about whether 40 or 55 mm would have
delivered.  This screen keeps the ladder, and it screens the *whole* gated pool rather than one
body mode, because independent sources are the binding constraint on every downstream claim.

Reference-side evidence only.  Nothing here predicts a physics verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    _signed_half_widths,
    local_crouch,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402
from scripts.research.hallucination.screen_crouch_strength import route_summary  # noqa: E402

LADDER_M = (0.040, 0.055, 0.070, 0.085)
STATION_FRACTION = 0.55
WINDOW_FRACTION = 0.30
MIN_STRAIGHTNESS = 0.95
#: Clips whose executed pair already anchors verified critical support.
ALREADY_SOURCED = (56, 86, 89, 90, 95)


def screen_clip(qpos: np.ndarray, index: int) -> dict:
    route = route_summary(qpos)
    base_left, base_right = _signed_half_widths(qpos, DEFAULT_G1_MJCF)
    rungs = []
    for target in LADDER_M:
        adapted, report = local_crouch(
            qpos, STATION_FRACTION, target_drop_m=target, window=WINDOW_FRACTION
        )
        gate = screen_reference(adapted, f"ladder_{index:03d}_{int(1000 * target):03d}", "walk")
        left, right = _signed_half_widths(adapted, DEFAULT_G1_MJCF)
        rungs.append(
            {
                "target_drop_mm": 1000 * target,
                "reference_drop_mm": 1000 * report.silhouette_drop_m,
                "delivered_fraction": report.silhouette_drop_m / target,
                "max_joint_change_rad": report.max_joint_change_rad,
                "excursion_capped": report.excursion_capped,
                "root_path_preserved": report.root_path_preserved,
                # A crouch that also narrows the arms is not a pure lower-body edit; motion 095's
                # 32.4 mm lateral coupling is the reason this is measured rather than assumed.
                "lateral_coupling_mm": 1000
                * float(max(abs((base_left - left).min()), abs((base_right - right).min()))),
                "worth_a_rollout": bool(gate.worth_a_rollout),
                "diagnosis": gate.diagnosis,
            }
        )
    usable = [
        rung
        for rung in rungs
        if rung["worth_a_rollout"]
        and rung["root_path_preserved"]
        and not rung["excursion_capped"]
        and rung["reference_drop_mm"] >= 0.9 * rung["target_drop_mm"]
    ]
    return {
        "motion_index": index,
        "route": route,
        "rungs": rungs,
        "usable_rungs": len(usable),
        "deepest_usable_mm": max((rung["reference_drop_mm"] for rung in usable), default=0.0),
        "min_lateral_coupling_mm": min(
            (rung["lateral_coupling_mm"] for rung in usable), default=None
        ),
        "already_sourced": index in ALREADY_SOURCED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument(
        "--reference-gate",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/coverage/reference_gate.json",
    )
    parser.add_argument("--min-straightness", type=float, default=MIN_STRAIGHTNESS)
    parser.add_argument("--limit", type=int, default=0, help="0 screens every eligible clip")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/crouch_ladder_screen.json"
    )
    args = parser.parse_args()

    gate = {row["index"]: row for row in json.loads(args.reference_gate.read_text())}
    eligible = []
    for index, row in sorted(gate.items()):
        if not row.get("worth_a_rollout"):
            continue
        matches = sorted(args.source_dir.glob(f"{index:03d}_*.csv"))
        if len(matches) != 1:
            continue
        qpos = np.loadtxt(matches[0], delimiter=",")
        route = route_summary(qpos)
        if route["straightness"] < args.min_straightness:
            continue
        eligible.append((index, matches[0], qpos, row.get("body_mode", "")))
    if args.limit:
        eligible = eligible[: args.limit]

    results = []
    for position, (index, path, qpos, mode) in enumerate(eligible, start=1):
        record = screen_clip(qpos, index)
        record["body_mode"] = mode
        record["source_csv"] = str(path)
        results.append(record)
        print(
            f"[{position}/{len(eligible)}] {index:03d} {mode:<14} "
            f"usable={record['usable_rungs']}/4 deepest={record['deepest_usable_mm']:.1f} mm",
            flush=True,
        )

    report = {
        "schema_version": "lfh_crouch_ladder_screen_v1",
        "ladder_target_drops_mm": [1000 * value for value in LADDER_M],
        "station_fraction": STATION_FRACTION,
        "window_fraction": WINDOW_FRACTION,
        "minimum_route_straightness": args.min_straightness,
        "screened_clips": len(results),
        "clips_with_any_usable_rung": sum(1 for row in results if row["usable_rungs"]),
        "interpretation": (
            "reference-side ladder support; a usable rung licenses an empty-scene physics cell "
            "and nothing more"
        ),
        "clips": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(results)} clips screened -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
