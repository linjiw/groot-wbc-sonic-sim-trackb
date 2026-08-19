#!/usr/bin/env python
"""Flat, filterable indexes over everything the project has produced.

The JSON release is faithful but nested, and nested is the wrong shape for "show me every episode
where the torso came within 20 mm of a shelf and the controller still held" -- which is the question
someone will actually ask. So the same data is emitted as three CSVs at three grains, joined by
stable ids:

* ``episodes.csv``  -- one row per graded rollout. The finest grain, and the one to filter on.
* ``families.csv``  -- one row per counterfactual family, with its four cell outcomes side by side.
* ``motions.csv``   -- one row per reference clip, carrying generation-side properties.

Every column is either measured or a path. Nothing is inferred, and where a value is unknown the
cell is empty rather than zero, because a zero force and an unmeasured force are different facts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import build_counterfactual_family as cf  # noqa: E402

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.scene_route_check import (  # noqa: E402
    capsule_box_clearance,
    check_route_meets_obstacle,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"

BEHAVIOURS = (
    ("crouch", r"crouch|duck|lower|stoop"),
    ("side_narrow", r"sideways|side[_ ]step|narrow|squeeze|shuffle"),
    ("arms", r"arm|reach|carry|hold"),
    ("turn", r"turn|curv"),
    ("stop_start", r"stop|stand[_ ]still|begins"),
    ("walk", r"walk"),
)

EPISODE_COLUMNS = [
    "episode_id",
    "family_id",
    "cell_role",
    "motion_role",
    "operator",
    "behaviour_class",
    "scene_id",
    "obstacle_underside_m",
    "route_reaches_obstacle",
    "frames_inside_obstacle",
    "outcome",
    "rejection_reasons",
    "min_clearance_mm",
    "clearance_frame",
    "closest_body",
    "external_force_n",
    "overhead_force_n",
    "lateral_force_n",
    "self_force_n",
    "contact_body",
    "contact_frame",
    "drift_rate_mps",
    "frames",
    "trajectory_path",
    "scene_path",
    "video_room",
    "video_side",
    "video_ego",
    "video_front",
    "video_top",
    "n_videos",
    # What the operator asked for, and what the controller actually did. A label names the
    # commanded behaviour; only the executed amplitude says how much of it happened, and a consumer
    # filtering for real crouches needs the second. Blank where the cell has no matched nominal.
    "commanded_amplitude_rad",
    "executed_amplitude_rad",
    "operator_survival",
    "survival_drift_margin",
]


def survival_columns(cell_dir: Path) -> dict:
    """Commanded vs executed amplitude for an adapted cell, or blanks.

    Reuses measure_operator_survival so the release and the paper cannot disagree: one
    implementation, one definition of which joints and frames count.
    """
    from scripts.research.measure_operator_survival import nominal_for, survival

    nominal = nominal_for(cell_dir)
    if nominal is None:
        return {}
    result = survival(cell_dir, nominal)
    if result is None:
        return {}
    return {
        "commanded_amplitude_rad": round(result["commanded_rad"], 4),
        "executed_amplitude_rad": round(result["executed_rad"], 4),
        "operator_survival": round(result["survival"], 3),
        "survival_drift_margin": round(result["margin"], 1),
    }


def behaviour_of(text: str) -> str:
    for tag, pattern in BEHAVIOURS:
        if re.search(pattern, text):
            return tag
    return "other"


def role_of(cell: str) -> tuple[str, str]:
    """(cell_role, motion_role) read from the cell name, handling both naming conventions."""
    adapted = any(key in cell for key in ("crouch", "tuck", "adapted"))
    motion = "adapted" if adapted else "nominal"
    if cell.endswith("_easy"):
        return f"{motion}_easy", motion
    if cell.endswith("_hard"):
        return f"{motion}_hard", motion
    return "probe", motion


def scene_of(cell_dir: Path) -> str | None:
    for name in (f"{cell_dir.name}.runner.log", f"{cell_dir.name}.log"):
        log = cell_dir.parent / "logs" / name
        if log.exists():
            match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
            if match:
                return match.group(1)
    return None


def episode_row(cell_dir: Path, family_id: str, videos: Path | None) -> dict | None:
    paths = sorted(cell_dir.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return None
    try:
        with open(paths[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except SegmentError:
        payload = None
    except Exception:  # noqa: BLE001
        return None

    cell_role, motion_role = role_of(cell_dir.name)
    operator = (
        "local_crouch"
        if "crouch" in cell_dir.name
        else "local_arm_tuck" if "tuck" in cell_dir.name else ""
    )
    scene_id = scene_of(cell_dir)
    row = {c: "" for c in EPISODE_COLUMNS}
    row.update(
        {
            "episode_id": f"{family_id}/{cell_dir.name}",
            "family_id": family_id,
            "cell_role": cell_role,
            "motion_role": motion_role,
            "operator": operator,
            "behaviour_class": behaviour_of(cell_dir.name),
            "scene_id": scene_id or "",
            "trajectory_path": str(paths[0]),
        }
    )
    if scene_id and (SCENES / f"{scene_id}.usda").exists():
        row["scene_path"] = str(SCENES / f"{scene_id}.usda")

    row.update(survival_columns(cell_dir))

    if payload is None:
        row["outcome"] = "unevaluable"
        return row

    outcome = classify_episode(cell_dir.name, payload)
    diagnostics = outcome.diagnostics or {}
    decomposition = diagnostics.get("contact_decomposition") or {}
    bodies = diagnostics.get("max_nonfoot_contact_bodies") or []
    row.update(
        {
            "outcome": outcome.outcome,
            "rejection_reasons": ";".join(outcome.rejection_reasons),
            "external_force_n": round(
                float(decomposition.get("max_external_contact_force_n", 0.0)), 1
            ),
            "overhead_force_n": round(
                float(decomposition.get("max_overhead_contact_force_n", 0.0)), 1
            ),
            "lateral_force_n": round(
                float(decomposition.get("max_lateral_contact_force_n", 0.0)), 1
            ),
            "self_force_n": round(float(decomposition.get("max_self_contact_force_n", 0.0)), 1),
            "contact_body": ";".join(bodies),
            "contact_frame": diagnostics.get("max_nonfoot_contact_frame", ""),
            "drift_rate_mps": round(float(diagnostics.get("drift_rate_mps", 0.0)), 4),
            "frames": int(len(payload["root_pos_w"])),
        }
    )

    if row["scene_path"]:
        try:
            box = cf.rendered_shelf_box(Path(row["scene_path"]))
            starts, ends, radii, names = body_capsules_world(
                np.asarray(payload["body_pos_w"], dtype=np.float64),
                np.asarray(payload["body_quat_w"], dtype=np.float64),
                list(payload["body_names"]),
                capsules=G1_COLLISION_CAPSULES,
            )
            gap, frame, capsule = capsule_box_clearance(starts, ends, radii, box)
            check = check_route_meets_obstacle(
                np.asarray(payload["root_pos_w"])[:, :2], (box[0], box[1], box[3], box[4])
            )
            row.update(
                {
                    "obstacle_underside_m": round(box[2], 4),
                    "min_clearance_mm": round(gap * 1000, 1),
                    "clearance_frame": frame,
                    "closest_body": names[capsule],
                    "route_reaches_obstacle": int(check.passes),
                    "frames_inside_obstacle": check.frames_inside,
                }
            )
        except Exception:  # noqa: BLE001 - a scene without a shelf carries no clearance
            pass

    if videos:
        for key, suffix in (
            ("video_room", "room"),
            ("video_side", "side"),
            ("video_ego", "ego"),
            ("video_front", "front"),
            ("video_top", "top"),
        ):
            candidate = videos / f"{cell_dir.name}__{suffix}.mp4"
            if candidate.exists():
                row[key] = str(candidate)
        row["n_videos"] = sum(
            1
            for k in ("video_room", "video_side", "video_ego", "video_front", "video_top")
            if row[k]
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families-root", type=Path, nargs="+", required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--screen", type=Path)
    args = ap.parse_args()

    index = args.release / "index"
    index.mkdir(parents=True, exist_ok=True)

    episodes, families = [], []
    for root in args.families_root:
        for work in sorted(p for p in root.iterdir() if p.is_dir()):
            family_id = work.name
            videos = args.release / "families" / family_id / "renders"
            rows = []
            for cell in sorted(p for p in work.iterdir() if p.is_dir()):
                if cell.name in ("logs", "motions", "clips", "renders"):
                    continue
                row = episode_row(cell, family_id, videos if videos.exists() else None)
                if row:
                    rows.append(row)
            if not rows:
                continue
            episodes += rows

            by_role = {r["cell_role"]: r["outcome"] for r in rows}
            verified = (
                by_role.get("nominal_easy") == "accepted"
                and by_role.get("adapted_easy") == "accepted"
                and by_role.get("nominal_hard") == "rejected"
                and by_role.get("adapted_hard") == "accepted"
            )
            manifest = root / f"{family_id}.json"
            meta = json.loads(manifest.read_text()) if manifest.exists() else {}
            if not meta and (work / "family.json").exists():
                meta = json.loads((work / "family.json").read_text())
            families.append(
                {
                    "family_id": family_id,
                    "status": "verified" if verified else "not_verified",
                    "cells": len(rows),
                    "window_predicted_mm": round(float(meta.get("window_m", 0.0)) * 1000, 1) or "",
                    "nominal_easy": by_role.get("nominal_easy", ""),
                    "nominal_hard": by_role.get("nominal_hard", ""),
                    "adapted_easy": by_role.get("adapted_easy", ""),
                    "adapted_hard": by_role.get("adapted_hard", ""),
                    "operators": ";".join(sorted({r["operator"] for r in rows if r["operator"]})),
                    "scenes": ";".join(sorted({r["scene_id"] for r in rows if r["scene_id"]})),
                }
            )

    with open(index / "episodes.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(episodes)
    with open(index / "families.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(families[0].keys()))
        writer.writeheader()
        writer.writerows(families)

    motions = []
    if args.screen and args.screen.exists():
        for record in json.loads(args.screen.read_text()).get("records", []):
            stem = re.sub(r"(_s\d+)+$", "", record["csv"][:-4])
            motions.append(
                {
                    "clip": stem,
                    "prompt_intent": re.sub(r"^\d+_", "", stem).replace("_", " "),
                    "behaviour_class": behaviour_of(stem),
                    "frames": record.get("frames", ""),
                    "embodiment_feasible": int(bool(record.get("passed"))),
                    "rejection_reasons": ";".join(record.get("reasons", [])),
                    "root_height_min_m": round(float(record.get("root_height_min_m", 0.0)), 4),
                    "saturated_frame_fraction": record.get("saturated_frame_fraction", ""),
                    "reference_semantic_valid": "",
                    "executed_semantic_valid": "",
                }
            )
        with open(index / "motions.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(motions[0].keys()))
            writer.writeheader()
            writer.writerows(motions)

    print(f"episodes.csv  {len(episodes):4d} rows x {len(EPISODE_COLUMNS)} columns")
    print(f"families.csv  {len(families):4d} rows")
    print(f"motions.csv   {len(motions):4d} rows")
    graded = [e for e in episodes if e["outcome"] in ("accepted", "rejected")]
    with_gap = [e for e in graded if e["min_clearance_mm"] != ""]
    print(f"\ngraded episodes {len(graded)}, of which {len(with_gap)} carry a measured clearance")
    print(f"verified families {sum(1 for f in families if f['status'] == 'verified')}")
    print(f"wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
