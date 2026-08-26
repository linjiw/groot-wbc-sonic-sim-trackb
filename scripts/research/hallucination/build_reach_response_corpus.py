#!/usr/bin/env python3
"""Pair commanded (reference-kinematic) overhead reach with executed reach, per rollout.

This is the supervision `docs/lfh/lfh.md` §4.1 calls ``D_phi`` and §4.1's E-D1 marks unvalidated.
The window that defines a critical scene is an order statistic over *executed* reach, so today a
scene cannot be proposed for a generated motion without first spending two rollouts.  If executed
reach is predictable from the reference clip, the proposal can be made before the spend -- which
is what makes motion-to-scene inference possible for a freshly generated motion.

Both sides are measured through the same instrument at the same face, so the residual is
delivery, not a change of definition.  Nothing here predicts or replaces a physics verdict.
"""

from __future__ import annotations

import argparse
import csv
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
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
RUN_RECORDS = DATA_ROOT / "hallucination/run_records"
STATION_FRACTION = 0.55
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 3.0


def _station(root_xy: np.ndarray, fraction: float) -> tuple[float, float]:
    progress = route_progress(np.asarray(root_xy, dtype=np.float64))
    index = int(np.argmin(np.abs(progress - fraction)))
    return float(root_xy[index, 0]), float(root_xy[index, 1])


def _route_axis(root_xy: np.ndarray) -> str:
    span = np.ptp(np.asarray(root_xy, dtype=np.float64), axis=0)
    return "x" if span[0] >= span[1] else "y"


def _face_reach(tracks, station, axis) -> float | None:
    try:
        face = overhead_face_reach(
            tracks, station, axis, FACE_ALONG_M, FACE_ACROSS_M, require_all_groups=False
        )
    except ValueError:
        return None
    return float(face.reach_m) if np.isfinite(face.reach_m) else None


def _payload(path: Path):
    result = best_evaluable_payload(pickle.load(path.open("rb")))
    return result[0] if isinstance(result, tuple) else result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RUN_RECORDS)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/reach_response_corpus.csv"
    )
    args = parser.parse_args()

    rows: list[dict] = []
    refused: dict[str, int] = {}
    for record_path in sorted(args.records.glob("*.json")):
        record = json.loads(record_path.read_text())
        for cell_id, cell in (record.get("cells") or {}).items():
            scientific = cell.get("scientific") or {}
            if scientific.get("outcome") is None:
                continue
            trajectory = (scientific.get("artifacts") or {}).get("trajectory")
            if not trajectory or not Path(trajectory).exists():
                refused["no_trajectory"] = refused.get("no_trajectory", 0) + 1
                continue
            command = cell.get("command") or []
            motion = scene = None
            for index, token in enumerate(command):
                if token == "--motion" and index + 1 < len(command):
                    motion = Path(command[index + 1])
                if token == "--scene" and index + 1 < len(command):
                    scene = command[index + 1]
            if motion is None:
                continue
            manifest = motion.with_suffix(motion.suffix + ".manifest.json")
            reference_csv = None
            if manifest.exists():
                candidate = json.loads(manifest.read_text()).get("input", {}).get("path")
                if candidate and Path(candidate).exists():
                    reference_csv = Path(candidate)
            if reference_csv is None:
                sibling = motion.with_suffix(".csv")
                reference_csv = sibling if sibling.exists() else None
            if reference_csv is None:
                refused["no_reference"] = refused.get("no_reference", 0) + 1
                continue

            qpos = np.loadtxt(reference_csv, delimiter=",")
            try:
                reference_tracks = extract_keypoints(payload_from_reference(qpos))
                executed_tracks = extract_keypoints(_payload(Path(trajectory)))
            except Exception as error:  # noqa: BLE001 - refusal is the result
                refused[type(error).__name__] = refused.get(type(error).__name__, 0) + 1
                continue

            axis = _route_axis(reference_tracks.root_pos_w[:, :2])
            # Both reaches are taken at the *commanded* station: the executed route drifts, and
            # a scene is authored in world coordinates from the reference, so the station is a
            # property of the plan rather than of the execution.
            station = _station(reference_tracks.root_pos_w[:, :2], STATION_FRACTION)
            commanded = _face_reach(reference_tracks, station, axis)
            executed = _face_reach(executed_tracks, station, axis)
            if commanded is None or executed is None:
                refused["face_misses_route"] = refused.get("face_misses_route", 0) + 1
                continue

            diagnostics = scientific.get("diagnostics") or {}
            rows.append(
                {
                    "experiment": record_path.stem,
                    "cell_id": cell_id,
                    "reference_csv": str(reference_csv),
                    "scene": scene or "",
                    "outcome": scientific["outcome"],
                    "accepted": int(scientific["outcome"] == "accepted"),
                    "route_axis": axis,
                    "station_x_m": station[0],
                    "station_y_m": station[1],
                    "commanded_reach_m": commanded,
                    "executed_reach_m": executed,
                    "delivery_residual_mm": 1000 * (executed - commanded),
                    "reference_frames": int(qpos.shape[0]),
                    "executed_frames": int(executed_tracks.frames),
                    "endpoint_error_m": diagnostics.get("endpoint_error_m"),
                    "schedule_error_p95_m": diagnostics.get("schedule_error_p95_m"),
                }
            )

    if not rows:
        raise SystemExit("no reach-response rows recovered")
    fields = list(rows[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    residual = np.asarray([row["delivery_residual_mm"] for row in rows])
    print(f"{len(rows)} paired reach rows -> {args.out}")
    print(
        f"delivery residual mm: mean {residual.mean():+.1f} "
        f"sd {residual.std(ddof=1):.1f} min {residual.min():+.1f} max {residual.max():+.1f}"
    )
    if refused:
        print("refused:", dict(sorted(refused.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
