"""Score banded 2x2 families: does each one show the counterfactual, and if not, why not?

A family holds when the same journey survives an easy scene under both motions, and a hard scene
only under the adapted one. Four cells, one pattern:

    nominal_easy  accepted     adapted_easy  accepted
    nominal_hard  REJECTED     adapted_hard  accepted

Reporting that as a single Boolean throws away the interesting part. A family can miss in ways that
mean opposite things: the hard scene failing to stop the nominal is a placement problem, while an
adapted cell failing on tracking is the adaptation costing more than the controller will spend --
and only the second is a statement about the method. So each family is scored into a named miss
mode, and the endpoint error is carried alongside because it is the quantity P6 predicts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from gear_sonic.dataset_generation.trajectory_acceptance import evaluate_locomotion_trajectory
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

CELLS = ("nominal_easy", "adapted_easy", "nominal_hard", "adapted_hard")

#: The pattern a family must show to count as verified.
WANTED = {
    "nominal_easy": True,
    "adapted_easy": True,
    "nominal_hard": False,
    "adapted_hard": True,
}

TRACKING = ("reference_endpoint_tracking_error", "reference_path_tracking_error")


def cell_report(cell: Path) -> dict | None:
    """Verdict, reasons, endpoint error and peak overhead force for one cell."""
    found = sorted(cell.glob("trajectories/*.trajectory.pkl"))
    if not found:
        return None
    try:
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except (Exception,):  # noqa: BLE001
        return None
    if payload is None:
        return None
    report = evaluate_locomotion_trajectory(payload)
    reasons = tuple(report.rejection_reasons)

    endpoint = float("nan")
    if "reference_g1_qpos" in payload and "root_pos_w" in payload:
        reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, :3]
        executed = np.asarray(payload["root_pos_w"], dtype=np.float64)
        endpoint = float(np.linalg.norm(executed[-1] - reference[-1]))

    peak = float("nan")
    if "robot_contact_force_w" in payload:
        forces = np.asarray(payload["robot_contact_force_w"], dtype=np.float64)
        names = list(payload.get("contact_body_names", []))
        feet = set(payload.get("allowed_foot_contact_body_names", ()))
        keep = [i for i, name in enumerate(names) if name not in feet]
        if keep:
            peak = float(np.linalg.norm(forces[:, keep, :], axis=2).max())

    return {
        "accepted": report.accepted,
        "reasons": reasons,
        "endpoint_error_m": endpoint,
        "peak_nonfoot_n": peak,
        "tracking_failed": any(r in TRACKING for r in reasons),
        "contact_failed": "disallowed_robot_contact" in reasons,
    }


def miss_mode(cells: dict[str, dict]) -> str:
    """Why a family does not hold, named so that opposite causes are not merged."""
    if all(cells[c]["accepted"] == WANTED[c] for c in CELLS):
        return "verified"
    if cells["nominal_hard"]["accepted"]:
        return "hard scene did not stop the nominal"
    adapted = [c for c in ("adapted_easy", "adapted_hard") if not cells[c]["accepted"]]
    # Contact is checked first on purpose. A cell that both struck the obstacle and drifted has
    # failed at the thing the family is about, and reporting it as a tracking cost would credit
    # the adaptation with a clearance it did not achieve.
    if adapted and any(cells[c]["contact_failed"] for c in adapted):
        return "adapted motion still struck the obstacle"
    if adapted and all(cells[c]["tracking_failed"] for c in adapted):
        return "adaptation cost more progress than the gate allows"
    if not cells["nominal_easy"]["accepted"]:
        return "nominal does not survive the easy scene"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory of family directories")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    families = sorted(p for p in args.root.iterdir() if p.is_dir())
    rows, partial = [], 0
    for family in families:
        cells = {c: cell_report(family / c) for c in CELLS}
        if any(v is None for v in cells.values()):
            partial += 1
            continue
        rows.append({"family": family.name, "cells": cells, "mode": miss_mode(cells)})

    if not rows:
        print(f"no complete families yet ({partial} partial)")
        return 0

    print(f"{'family':>34s} {'n_easy':>7s} {'a_easy':>7s} {'n_hard':>7s} {'a_hard':>7s}  outcome")
    for row in rows:
        marks = " ".join(
            f"{('ok' if row['cells'][c]['accepted'] else 'no'):>7s}" for c in CELLS
        )
        print(f"{row['family']:>34s} {marks}  {row['mode']}")

    verified = [r for r in rows if r["mode"] == "verified"]
    print(f"\n{len(verified)} of {len(rows)} complete families verified; {partial} still running")

    modes: dict[str, int] = {}
    for row in rows:
        if row["mode"] != "verified":
            modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"  {count:2d}  {mode}")

    # P6 asks whether adapted endpoint error separates by band, so print it by band.
    print(f"\n{'band':>10s} {'families':>9s} {'adapted endpoint error (m)':>30s}")
    bands: dict[str, list[float]] = {}
    for row in rows:
        band = "overhead" if "overhead" in row["family"] else (
            "chest" if "chest" in row["family"] else "waist"
        )
        for cell in ("adapted_easy", "adapted_hard"):
            value = row["cells"][cell]["endpoint_error_m"]
            if value == value:
                bands.setdefault(band, []).append(value)
    for band, values in sorted(bands.items()):
        print(
            f"{band:>10s} {len(values) // 2:9d} "
            f"{min(values):9.3f} - {max(values):.3f}  (median {sorted(values)[len(values) // 2]:.3f})"
        )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
