#!/usr/bin/env python3
"""Adjudicate LFH-E16: how reproducible is the executed critical window under seed alone?

E16 assumed the pairs it re-rolls would accept, and measured the window range across seeds. That
assumption is itself testable, and this analyzer reports it first: a pair whose adapted cell does
not reproducibly accept has no reproducible window to measure, and saying so is the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import route_progress  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
E1A_MARGIN_MM = 18.044
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 3.0
STATION_FRACTION = 0.55


def _tracks(path: Path):
    result = best_evaluable_payload(pickle.load(Path(path).open("rb")))
    return extract_keypoints(result[0] if isinstance(result, tuple) else result)


def _station_and_axis(tracks) -> tuple[tuple[float, float], str]:
    root = tracks.root_pos_w[:, :2]
    span = np.ptp(root, axis=0)
    axis = "x" if span[0] >= span[1] else "y"
    progress = route_progress(np.asarray(root, dtype=np.float64))
    index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    return (float(root[index, 0]), float(root[index, 1])), axis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-record",
        type=Path,
        default=DATA_ROOT / "hallucination/run_records/E16_WINDOW_REPEATABILITY_2026-08-26.json",
    )
    parser.add_argument(
        "--ladder-record",
        type=Path,
        default=DATA_ROOT / "hallucination/run_records/E12_CROUCH_LADDER_2026-08-26.json",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e16_window_repeatability.json",
    )
    args = parser.parse_args()

    record = json.loads(args.run_record.read_text())
    ladder = json.loads(args.ladder_record.read_text())
    cells = record["cells"]

    pairs: dict[str, dict] = {}
    for cell_id, cell in cells.items():
        scientific = cell.get("scientific") or {}
        outcome = scientific.get("outcome")
        if outcome is None:
            continue
        pair_id, role, seed_text = cell_id.rsplit("__", 2)
        seed = int(seed_text.lstrip("s"))
        entry = pairs.setdefault(pair_id, {"nominal": {}, "adapted": {}, "rung": None})
        slot = "nominal" if role == "nominal" else "adapted"
        if slot == "adapted":
            entry["rung"] = role
        entry[slot][seed] = {
            "outcome": outcome,
            "endpoint_error_m": (scientific.get("diagnostics") or {}).get("endpoint_error_m"),
            "trajectory": (scientific.get("artifacts") or {}).get("trajectory"),
        }

    # The single-seed E12 observation of the same pair, for a four-point view where it exists.
    ladder_cells = ladder["cells"]

    results = []
    for pair_id, entry in sorted(pairs.items()):
        seeds = sorted(set(entry["nominal"]) & set(entry["adapted"]))
        nominal_accepts = [
            s for s in entry["nominal"] if entry["nominal"][s]["outcome"] == "accepted"
        ]
        adapted_accepts = [
            s for s in entry["adapted"] if entry["adapted"][s]["outcome"] == "accepted"
        ]

        endpoints = {
            "nominal": [entry["nominal"][s]["endpoint_error_m"] for s in sorted(entry["nominal"])],
            "adapted": [entry["adapted"][s]["endpoint_error_m"] for s in sorted(entry["adapted"])],
        }
        rung = entry["rung"]
        for cell_id, cell in ladder_cells.items():
            scientific = cell.get("scientific") or {}
            if not scientific or not cell_id.startswith(pair_id + "__"):
                continue
            suffix = cell_id.split("__", 1)[1]
            if suffix == "nominal":
                endpoints["nominal"].append(
                    (scientific.get("diagnostics") or {}).get("endpoint_error_m")
                )
            elif rung and suffix == rung:
                endpoints["adapted"].append(
                    (scientific.get("diagnostics") or {}).get("endpoint_error_m")
                )

        windows = {}
        for seed in seeds:
            if (
                entry["nominal"][seed]["outcome"] != "accepted"
                or entry["adapted"][seed]["outcome"] != "accepted"
            ):
                continue
            nominal_tracks = _tracks(Path(entry["nominal"][seed]["trajectory"]))
            adapted_tracks = _tracks(Path(entry["adapted"][seed]["trajectory"]))
            station, axis = _station_and_axis(nominal_tracks)
            try:
                nominal_face = overhead_face_reach(
                    nominal_tracks,
                    station,
                    axis,
                    FACE_ALONG_M,
                    FACE_ACROSS_M,
                    require_all_groups=False,
                )
                adapted_face = overhead_face_reach(
                    adapted_tracks,
                    station,
                    axis,
                    FACE_ALONG_M,
                    FACE_ACROSS_M,
                    require_all_groups=False,
                )
            except ValueError:
                continue
            windows[seed] = 1000 * float(nominal_face.reach_m - adapted_face.reach_m)

        adapted_endpoints = [value for value in endpoints["adapted"] if value is not None]
        nominal_endpoints = [value for value in endpoints["nominal"] if value is not None]
        results.append(
            {
                "pair_id": pair_id,
                "rung": rung,
                "seeds_graded": len(seeds),
                "nominal_accept_rate": f"{len(nominal_accepts)}/{len(entry['nominal'])}",
                "adapted_accept_rate": f"{len(adapted_accepts)}/{len(entry['adapted'])}",
                "endpoint_error_m": {
                    "nominal": sorted(nominal_endpoints),
                    "adapted": sorted(adapted_endpoints),
                    "adapted_range_mm": (
                        1000 * (max(adapted_endpoints) - min(adapted_endpoints))
                        if len(adapted_endpoints) > 1
                        else None
                    ),
                    "nominal_range_mm": (
                        1000 * (max(nominal_endpoints) - min(nominal_endpoints))
                        if len(nominal_endpoints) > 1
                        else None
                    ),
                },
                "windows_mm_by_seed": windows,
                "window_range_mm": (
                    max(windows.values()) - min(windows.values()) if len(windows) > 1 else None
                ),
                "window_measurable": len(windows) > 1,
            }
        )

    measurable = [row for row in results if row["window_measurable"]]
    report = {
        "schema_version": "lfh_e16_window_repeatability_v1",
        "run_record": str(args.run_record),
        "run_status": record.get("status"),
        "pairs_attempted": len(results),
        "pairs_with_a_measurable_window_range": len(measurable),
        "e1a_margin_mm_each_side": E1A_MARGIN_MM,
        "predictions": {
            "P1_range_below_18_044mm": (
                None
                if not measurable
                else all(row["window_range_mm"] < E1A_MARGIN_MM for row in measurable)
            ),
            "P3_all_cells_accept": (
                all(
                    row["nominal_accept_rate"].split("/")[0]
                    == row["nominal_accept_rate"].split("/")[1]
                    and row["adapted_accept_rate"].split("/")[0]
                    == row["adapted_accept_rate"].split("/")[1]
                    for row in results
                )
                if results
                else None
            ),
        },
        "pairs": results,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"run status: {record.get('status')}; pairs graded: {len(results)}")
    for row in results:
        print(
            f"  {row['pair_id']} ({row['rung']}): nominal {row['nominal_accept_rate']} accepted, "
            f"adapted {row['adapted_accept_rate']} accepted"
        )
        endpoint = row["endpoint_error_m"]
        print(
            f"    endpoint error m -- nominal {[round(v, 4) for v in endpoint['nominal']]}, "
            f"adapted {[round(v, 4) for v in endpoint['adapted']]}"
        )
        if endpoint["adapted_range_mm"] is not None:
            print(
                f"    adapted endpoint range {endpoint['adapted_range_mm']:.1f} mm "
                f"(nominal {endpoint['nominal_range_mm']:.1f} mm)"
            )
        if row["window_measurable"]:
            print(
                f"    windows {[round(v, 2) for v in row['windows_mm_by_seed'].values()]} mm, "
                f"range {row['window_range_mm']:.2f} mm"
            )
        else:
            print("    window range NOT measurable: fewer than two seeds accept both cells")
    print(f"-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
