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

from gear_sonic.dataset_generation.contact_decomposition import decompose_contact_forces
from gear_sonic.dataset_generation.swept_volume import G1_COLLISION_CAPSULES, body_capsules_world
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


def delivered_window(family: Path, plan: dict) -> float | None:
    """How much clearance the operator actually bought, at the moment the obstacle binds.

    Measured on the binding body the plan names, at the place along the route where the obstacle
    binds -- never over the whole episode, and never at matched frame indices.

    Both restrictions were learned by getting them wrong. A maximum over the episode is dominated by
    whatever the robot does furthest from the obstacle, the part the operator never touched, and it
    reported a real 27 mm crouch as *negative* delivery. Matching by frame index is just as wrong for
    a different reason: a nominal that the obstacle stops falls behind, so at the frame it strikes,
    the adapted run is half a metre further down the room and is being compared at a place the
    obstacle is not. Both runs are therefore sampled where each one reaches the binding x.
    """
    body, vertical = plan["binding_body"], plan["obstacle"] == "ceiling"
    # Lateral extent has a sign, and the obstacle decides it. Measuring +y on a right-side
    # configuration measures the *left* arm -- which that tuck never touches -- and reported a
    # correct 10.6 mm retraction as -33% delivery.
    outward = -1.0 if plan.get("side") == "right" else 1.0

    def reach(cell: Path, frames: slice | None) -> float | None:
        found = sorted(cell.glob("trajectories/*.trajectory.pkl"))
        if not found:
            return None
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        if payload is None or "body_pos_w" not in payload:
            return None
        starts, ends, radii, owner = body_capsules_world(
            np.asarray(payload["body_pos_w"], dtype=np.float64),
            np.asarray(payload["body_quat_w"], dtype=np.float64),
            list(payload["body_names"]),
            capsules=G1_COLLISION_CAPSULES,
        )
        keep = [k for k in range(starts.shape[1]) if owner[k] == body]
        if not keep:
            return None
        window = frames or slice(None)
        if vertical:
            return float(
                max(
                    (P[window, k, 2] + radii[k]).max()
                    for k in keep
                    for P in (starts, ends)
                )
            )
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        quat = np.asarray(payload["root_quat_w"], dtype=np.float64)
        w, x, y, z = (quat[:, i] for i in range(4))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos, sin = np.cos(-yaw), np.sin(-yaw)
        best = -np.inf
        for k in keep:
            for P in (starts, ends):
                delta = P[:, k, :] - root
                lateral = outward * (sin * delta[:, 0] + cos * delta[:, 1]) + radii[k]
                best = max(best, float(lateral[window].max()))
        return best

    # Locate the binding moment from the nominal's own contact on that body.
    found = sorted((family / "nominal_hard").glob("trajectories/*.trajectory.pkl"))
    if not found:
        return None
    with open(found[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None or "robot_contact_force_w" not in payload:
        return None
    # The binding moment comes from the decomposed *external* contact, never from raw force
    # magnitude. Two reasons, both met in this batch. A plan naming left_elbow_link describes a
    # capsule the elbow shares with left_shoulder_yaw_link, so the strike is reported against a
    # body the plan does not name -- timing therefore cannot be read off the named body. And every
    # episode here carries a standing 45 N self-contact at the hip, which is larger than a real
    # 31 N graze against a wall; thresholding raw magnitude finds the self-contact and misses the
    # obstacle entirely. Reach is still measured on the body the plan names.
    decomposition = decompose_contact_forces(
        np.asarray(payload["robot_contact_force_w"], dtype=np.float64),
        list(payload["contact_body_names"]),
        foot_body_names=list(payload.get("allowed_foot_contact_body_names", ())),
    )
    external = decomposition.external_contact_by_frame
    if not external.size or float(external.max()) <= 1.0:
        return None
    peak = int(external.argmax())
    binding_x = float(np.asarray(payload["root_pos_w"], dtype=np.float64)[peak, 0])

    def near_binding_x(cell: Path) -> slice | None:
        """Frames where this run is within 10 cm of where the obstacle binds."""
        found = sorted(cell.glob("trajectories/*.trajectory.pkl"))
        if not found:
            return None
        with open(found[0], "rb") as handle:
            data, _ = best_evaluable_payload(pickle.load(handle))
        if data is None or "root_pos_w" not in data:
            return None
        x = np.asarray(data["root_pos_w"], dtype=np.float64)[:, 0]
        close = np.flatnonzero(np.abs(x - binding_x) < 0.10)
        if not len(close):
            return None
        return slice(int(close[0]), int(close[-1]) + 1)

    spans = {c: near_binding_x(family / c) for c in ("nominal_hard", "adapted_hard")}
    if any(s is None for s in spans.values()):
        return None
    nominal = reach(family / "nominal_hard", spans["nominal_hard"])
    adapted = reach(family / "adapted_hard", spans["adapted_hard"])
    if nominal is None or adapted is None:
        return None
    return nominal - adapted


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
    ap.add_argument("--plans", type=Path, default=None, help="batch.json, to measure delivery")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    plans = {}
    if args.plans and args.plans.exists():
        for plan in json.loads(args.plans.read_text())["plans"]:
            plans[(plan["nominal"], plan["band"], plan["side"])] = plan

    families = sorted(p for p in args.root.iterdir() if p.is_dir())
    rows, partial = [], 0
    for family in families:
        cells = {c: cell_report(family / c) for c in CELLS}
        if any(v is None for v in cells.values()):
            partial += 1
            continue
        row = {"family": family.name, "cells": cells, "mode": miss_mode(cells)}
        parts = family.name.split("_")
        plan = plans.get((f"{parts[0]}_{parts[1]}", parts[3], parts[4])) if len(parts) > 4 else None
        if plan:
            delivered = delivered_window(family, plan)
            row["predicted_window_m"] = plan["window_m"]
            row["binding_body"] = plan["binding_body"]
            row["operator"] = plan["operator"]
            row["delivered_window_m"] = delivered
            row["delivery_ratio"] = (
                delivered / plan["window_m"] if delivered is not None and plan["window_m"] else None
            )
        rows.append(row)

    if not rows:
        print(f"no complete families yet ({partial} partial)")
        return 0

    print(f"{'family':>34s} {'n_easy':>7s} {'a_easy':>7s} {'n_hard':>7s} {'a_hard':>7s}  outcome")
    for row in rows:
        marks = " ".join(
            f"{('ok' if row['cells'][c]['accepted'] else 'no'):>7s}" for c in CELLS
        )
        print(f"{row['family']:>34s} {marks}  {row['mode']}")

    measured = [r for r in rows if r.get("delivery_ratio") is not None]
    if measured:
        print(f"\n{'family':>30s} {'binding body':>22s} {'pred':>8s} {'deliv':>8s} {'ratio':>6s}")
        for row in measured:
            print(
                f"{row['family']:>30s} {row['binding_body']:>22s} "
                f"{1000 * row['predicted_window_m']:7.1f}mm {1000 * row['delivered_window_m']:7.1f}mm "
                f"{100 * row['delivery_ratio']:5.0f}%"
            )

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
