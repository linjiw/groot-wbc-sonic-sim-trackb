#!/usr/bin/env python3
"""Measure the behaviour spread of generated reference motions, before any rollout.

Rollouts are expensive; generation is not. Measuring the reference library directly answers
"did the prompt taxonomy actually produce different behaviours?" in seconds, and separates
that question from "can the tracker follow them?", which only physics can answer.

Keeping the two apart matters. A narrow corpus can mean the prompts were narrow or that the
tracker rejected everything interesting, and those call for opposite responses -- rewrite
the taxonomy, or lower the reach of what we ask for. This script measures only the first.

Usage::

    python scripts/research/analyze_reference_motions.py --csv-dir /data/.../motions \\
        [--taxonomy /data/.../taxonomy.json] [--json reference_spread.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    KIMODO_G1_JOINT_NAMES,
    KimodoQposError,
    load_kimodo_qpos_csv,
)

#: Kimodo's output frame rate. Speed is meaningless without it.
SOURCE_FPS = 30.0

JOINT_GROUPS = {
    "lower_body": tuple(
        index
        for index, name in enumerate(KIMODO_G1_JOINT_NAMES)
        if any(part in name for part in ("hip", "knee", "ankle"))
    ),
    "waist": tuple(index for index, name in enumerate(KIMODO_G1_JOINT_NAMES) if "waist" in name),
    "upper_body": tuple(
        index
        for index, name in enumerate(KIMODO_G1_JOINT_NAMES)
        if any(part in name for part in ("shoulder", "elbow", "wrist"))
    ),
}


def joint_excursion_rms(qpos: np.ndarray, indices: tuple[int, ...]) -> float:
    """Return the RMS range of motion for a named group of joints."""
    joint_positions = qpos[:, 7:]
    excursion = np.ptp(joint_positions[:, indices], axis=0)
    return float(np.sqrt(np.mean(np.square(excursion))))


def summarise(qpos: np.ndarray) -> dict:
    path_xy = qpos[:, :2]
    steps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    duration = max(len(steps), 1) / SOURCE_FPS
    displacement = float(np.linalg.norm(path_xy[-1] - path_xy[0]))

    deltas = np.diff(path_xy, axis=0)
    moving = deltas[np.linalg.norm(deltas, axis=1) > 1e-6]
    if len(moving) > 1:
        headings = np.unwrap(np.arctan2(moving[:, 1], moving[:, 0]))
        heading_change = float(headings[-1] - headings[0])
    else:
        heading_change = 0.0

    path_length = float(steps.sum())
    return {
        "frames": int(qpos.shape[0]),
        "path_length_m": path_length,
        "net_displacement_m": displacement,
        "mean_speed_mps": path_length / duration,
        "heading_change_rad": heading_change,
        "abs_heading_change_rad": abs(heading_change),
        "root_height_min_m": float(qpos[:, 2].min()),
        "root_height_range_m": float(qpos[:, 2].max() - qpos[:, 2].min()),
        # Straight line is 1.0; guard a motion that returns to where it started.
        "tortuosity": path_length / displacement if displacement > 1e-6 else math.inf,
        "joint_excursion_rms_rad": joint_excursion_rms(
            qpos, tuple(range(len(KIMODO_G1_JOINT_NAMES)))
        ),
        **{
            f"{group}_excursion_rms_rad": joint_excursion_rms(qpos, indices)
            for group, indices in JOINT_GROUPS.items()
        },
    }


def spread(values: list[float]) -> dict:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0}
    mean = float(finite.mean())
    return {
        "n": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        # Scale-free, so speed and path length are comparable. This is the number that
        # was 0.037 when the accepted corpus was a single behaviour.
        "cv": float(finite.std() / abs(mean)) if abs(mean) > 1e-12 else math.nan,
    }


def grouped_spreads(
    motions: dict[str, dict], specs: list[dict], axis: str, metric_keys: tuple[str, ...]
) -> dict[str, dict]:
    """Summarise every metric for each requested taxonomy value on one axis."""
    grouped: dict[str, list[dict]] = {}
    for stem, summary in motions.items():
        try:
            index = int(stem.split("_", 1)[0])
        except ValueError:
            continue
        if index < len(specs) and axis in specs[index]:
            grouped.setdefault(str(specs[index][axis]), []).append(summary)
    return {
        label: {
            "n": len(summaries),
            "metrics": {
                key: spread([summary[key] for summary in summaries]) for key in metric_keys
            },
        }
        for label, summaries in sorted(grouped.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, help="taxonomy.json, to group by axis")
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    csvs = sorted(args.csv_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.csv_dir}")

    motions: dict[str, dict] = {}
    for path in csvs:
        try:
            qpos = load_kimodo_qpos_csv(path)
        except KimodoQposError as error:
            print(f"  UNREADABLE {path.name}: {error}")
            continue
        motions[path.stem] = summarise(qpos)

    keys = (
        "mean_speed_mps",
        "path_length_m",
        "heading_change_rad",
        "abs_heading_change_rad",
        "root_height_min_m",
        "root_height_range_m",
        "tortuosity",
        "joint_excursion_rms_rad",
        "lower_body_excursion_rms_rad",
        "waist_excursion_rms_rad",
        "upper_body_excursion_rms_rad",
    )
    spreads = {key: spread([m[key] for m in motions.values()]) for key in keys}

    print(f"{len(motions)} reference motion(s)\n")
    for key, stats in spreads.items():
        if stats.get("n"):
            print(
                f"  {key:24s} min {stats['min']:8.3f}  median {stats['median']:8.3f}  "
                f"max {stats['max']:8.3f}  cv {stats['cv']:.3f}"
            )

    by_axis: dict[str, dict] = {}
    speed_by_requested_style: dict[str, dict] = {}
    if args.taxonomy is not None and args.taxonomy.exists():
        taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
        # Prompts were written to the CSV stem in taxonomy order, so index recovers the axis.
        specs = taxonomy["specs"]
        for axis in ("body_mode", "speed", "turn"):
            if any(axis in spec for spec in specs):
                by_axis[axis] = grouped_spreads(motions, specs, axis, keys)

        print("\nachieved kinematics by requested axis (group medians):")
        print("  axis/value                    n   speed   heading     zmin  upper-ROM")
        for axis, groups in by_axis.items():
            for label, group in groups.items():
                metrics = group["metrics"]
                print(
                    f"  {axis + '/' + label:28s} {group['n']:3d}  "
                    f"{metrics['mean_speed_mps']['median']:6.3f}  "
                    f"{metrics['heading_change_rad']['median']:8.3f}  "
                    f"{metrics['root_height_min_m']['median']:7.3f}  "
                    f"{metrics['upper_body_excursion_rms_rad']['median']:9.3f}"
                )

        # Preserve the original JSON field for downstream readers while exposing all axes.
        speed_by_requested_style = {
            label: group["metrics"]["mean_speed_mps"]
            for label, group in by_axis.get("speed", {}).items()
        }

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {
                    "motions": len(motions),
                    "spreads": spreads,
                    "by_requested_axis": by_axis,
                    "speed_by_requested_style": speed_by_requested_style,
                    "per_motion": motions,
                },
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
