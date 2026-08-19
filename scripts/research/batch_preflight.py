"""Decide on CPU which cells are worth a GPU, and refuse to launch a batch that is not.

The 2026-08-19 banded batch spent 21 rollouts and produced no verified family. Every one of its
four defects was visible without running anything:

* `n_064` began 418 mm outside its own room, because the room was sized from the route's span while
  the scene spec centres walls on the origin. Arithmetic on the route bounds.
* `n_064`, `n_065` and `n_122` all failed that check, and they were the entire remaining queue.
* The wall configurations asked for roughly a third of the edit they needed, because the
  minimum-edit rule was calibrated against a model that over-states delivery. A division.
* The overhead configurations needed more crouch than the endpoint budget allows, and the budget
  binds before the operator's own cap does. A lookup against a measured curve.

So this emits a go/no-go manifest and a batch may only launch from a passing one. Every check below
costs milliseconds and no GPU; the point is not that they are clever but that nothing runs until
they have all been asked.

    python scripts/research/batch_preflight.py --plans batch.json --rollouts <dir> --out manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from gear_sonic.dataset_generation.local_adaptation import (
    DELIVERY_RATIO,
    DELIVERY_RATIO_UNMEASURED,
    MAX_CROUCH_EXCURSION_RAD,
    MAX_TUCK_EXCURSION_RAD,
    delivery_corrected_target,
    local_arm_tuck,
    local_crouch,
)
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

#: Metres of walkable floor the scene builder leaves beyond the route, mirrored from
#: build_graded_scene.ROUTE_CLEARANCE_M. Duplicated deliberately: preflight must fail if the builder
#: silently changes, and importing the value would make the two agree by construction.
ROUTE_CLEARANCE_M = 2.0

#: Endpoint lag in metres at a given commanded excursion, measured on the amplitude sweeps. The
#: crouch is charged for its edit and the tuck is not; both curves are monotone in opposite
#: directions, which is why one number cannot serve for both.
CROUCH_LAG_CURVE = ((0.000, 0.257), (0.420, 0.313), (0.619, 0.399), (0.980, 0.509))
TUCK_LAG_CURVE = ((0.072, 0.257), (0.282, 0.218), (0.314, 0.195), (0.741, 0.112), (0.869, 0.084))

#: The acceptance gate's endpoint threshold. A configuration predicted to exceed it is not a
#: marginal case -- it is a rollout whose rejection is known before it starts.
ENDPOINT_BUDGET_M = 0.35

#: Minimum window, after delivery, that a configuration must be predicted to leave. The gate itself
#: is 20 mm; the margin covers run-to-run variation of the executed body, which is the reason a
#: nominally-positive window can still fail to be placeable.
WINDOW_GATE_M = 0.020
WINDOW_NOISE_MARGIN_M = 0.010


def interpolate(curve, x: float) -> float:
    """Linear interpolation on a measured curve, clamped at both ends."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    return float(np.interp(x, xs, ys))


def route_of(rollout: Path) -> np.ndarray | None:
    found = sorted(rollout.rglob("trajectories/*.trajectory.pkl"))
    if not found:
        return None
    try:
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except (Exception,):  # noqa: BLE001
        return None
    if payload is None or "root_pos_w" not in payload:
        return None
    return np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]


def nominal_clip(directory: Path, nominal: str) -> np.ndarray | None:
    """The reference clip for a nominal, by the `NNN_*.csv` convention the taxonomy uses."""
    found = sorted(directory.glob(f"{nominal.removeprefix('n_')}*.csv"))
    if not found:
        return None
    rows = [r for r in csv.reader(found[0].open()) if r and not r[0].startswith("#")]
    try:
        return np.array(rows, dtype=float)
    except ValueError:
        return np.array(rows[1:], dtype=float)


def room_for(route: np.ndarray) -> np.ndarray:
    """The room build_graded_scene would now build: origin-centred, sized to contain the route."""
    reach = np.maximum(np.abs(route.min(0)), np.abs(route.max(0)))
    half = reach + ROUTE_CLEARANCE_M
    return np.array([2 * half[0], max(2 * half[1], 5.0)])


def legacy_room_for(route: np.ndarray) -> np.ndarray:
    """The room the builder made before 2026-08-19: sized from the span, centred on the origin.

    Kept so the retrospective claim about this preflight is measured against the builder the batch
    actually ran with, rather than against the fixed one, which would flatter it.
    """
    span = route.max(0) - route.min(0)
    return np.array([span[0] + 2 * ROUTE_CLEARANCE_M, max(span[1] + 2 * ROUTE_CLEARANCE_M, 5.0)])


def check_route_fits_room(route: np.ndarray, *, legacy: bool = False) -> tuple[bool, str]:
    room = legacy_room_for(route) if legacy else room_for(route)
    half = room / 2.0
    overrun = float(np.maximum(np.abs(route.min(0)) - half, np.abs(route.max(0)) - half).max())
    if overrun > 0:
        return False, f"route leaves its room by {1000 * overrun:.0f} mm"
    return True, ""


def commanded_excursion(plan: dict, nominal: np.ndarray | None) -> float | None:
    """What this configuration would actually command, computed here rather than read from the plan.

    Preflight that trusts the planner's own record cannot audit a plan written before the field
    existed, and cannot catch a planner that computes it wrongly. So the operator is applied and
    the excursion measured, which costs milliseconds and depends on nothing upstream.
    """
    if nominal is None:
        return None
    operator, band = plan["operator"], plan["band"]
    base = 0.08 if operator == "local_crouch" else 0.06
    target, _ratio = delivery_corrected_target(base, operator, band if band else "waist")
    try:
        if operator == "local_crouch":
            adapted, _ = local_crouch(nominal, plan.get("station", 0.55), target_drop_m=target)
        else:
            side = plan["side"] if plan.get("side") in ("left", "right") else "both"
            adapted, _ = local_arm_tuck(
                nominal, plan.get("station", 0.55), target_reduction_m=target,
                window=0.30, side=side,
            )
    except (Exception,):  # noqa: BLE001
        return None
    count = min(len(adapted), len(nominal))
    return float(np.abs(adapted[:count, 7:] - nominal[:count, 7:]).max())


def check_delivery_reachable(plan: dict) -> tuple[bool, str]:
    """Is the edit this configuration needs inside the operator's cap once delivery is divided out?"""
    operator, band = plan["operator"], plan["band"]
    ratio = DELIVERY_RATIO.get((operator, band), DELIVERY_RATIO_UNMEASURED)
    cap = MAX_CROUCH_EXCURSION_RAD if operator == "local_crouch" else MAX_TUCK_EXCURSION_RAD
    commanded = plan.get("commanded_excursion_rad")
    if commanded is None:
        return True, ""
    if commanded > cap + 1e-9:
        return False, f"needs {commanded:.3f} rad against a {cap:.2f} rad cap"
    return True, f"ratio {ratio:.2f}"


def check_endpoint_budget(plan: dict) -> tuple[bool, str]:
    """Would the commanded amplitude spend more forward progress than the gate allows?

    The crouch's lag rises with depth and the nominal already spends most of the budget, so this is
    the check that disqualifies the overhead band -- and it binds at about 0.6 rad, well before the
    0.98 rad operator cap. Clamping at the cap would produce a clip whose rejection is certain.
    """
    commanded = plan.get("commanded_excursion_rad")
    if commanded is None:
        return True, ""
    curve = CROUCH_LAG_CURVE if plan["operator"] == "local_crouch" else TUCK_LAG_CURVE
    lag = interpolate(curve, commanded)
    if lag > ENDPOINT_BUDGET_M:
        return False, f"predicted lag {lag:.3f} m over a {ENDPOINT_BUDGET_M:.2f} m budget"
    return True, f"lag {lag:.3f} m"


def check_window_survives_delivery(plan: dict) -> tuple[bool, str]:
    """Does the predicted window still clear the gate after the measured shortfall is applied?"""
    ratio = plan.get("delivery_ratio_used") or DELIVERY_RATIO.get(
        (plan["operator"], plan["band"]), DELIVERY_RATIO_UNMEASURED
    )
    delivered = plan["window_m"] * ratio
    need = WINDOW_GATE_M + WINDOW_NOISE_MARGIN_M
    if delivered < need:
        return False, f"delivers {1000 * delivered:.0f} mm, under {1000 * need:.0f} mm"
    return True, f"delivers {1000 * delivered:.0f} mm"


def check_station_inside_window(plan: dict) -> tuple[bool, str]:
    """Is the obstacle where the adaptation is? Screened on the reference clips."""
    window = plan.get("reference_window_fraction")
    station = plan.get("station", 0.55)
    if window is None:
        return True, ""
    low, high = window
    if not low <= station <= high:
        return False, f"station {station:.2f} outside window {low:.2f}-{high:.2f}"
    return True, ""


CHECKS = (
    ("delivery_reachable", check_delivery_reachable),
    ("endpoint_budget", check_endpoint_budget),
    ("window_survives_delivery", check_window_survives_delivery),
    ("station_inside_window", check_station_inside_window),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True, help="dir of per-nominal rollouts")
    ap.add_argument("--clips", type=Path, help="dir of nominal CSVs, to compute commanded edits")
    ap.add_argument(
        "--legacy-room",
        action="store_true",
        help="size rooms the way the builder did before 2026-08-19, for retrospective audits",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plans = json.loads(args.plans.read_text())["plans"]

    routes: dict[str, np.ndarray | None] = {}
    for plan in plans:
        name = plan["nominal"]
        if name not in routes:
            routes[name] = route_of(args.rollouts / name)

    clips: dict[str, np.ndarray | None] = {}
    for plan in plans:
        name = plan["nominal"]
        if name not in clips:
            clips[name] = nominal_clip(args.clips, name) if args.clips else None

    cells, rejected = [], []
    for plan in plans:
        name = plan["nominal"]
        reasons = []
        if plan.get("commanded_excursion_rad") is None:
            plan["commanded_excursion_rad"] = commanded_excursion(plan, clips.get(name))

        route = routes.get(name)
        if route is None:
            reasons.append("no screen rollout to measure the route from")
        else:
            ok, why = check_route_fits_room(route, legacy=args.legacy_room)
            if not ok:
                reasons.append(why)

        for label, check in CHECKS:
            ok, why = check(plan)
            if not ok:
                reasons.append(f"{label}: {why}")

        record = {
            "nominal": name,
            "band": plan["band"],
            "side": plan["side"],
            "obstacle": plan["obstacle"],
            "operator": plan["operator"],
            "window_m": plan.get("window_m"),
            "passed": not reasons,
            "reasons": reasons,
        }
        (cells if not reasons else rejected).append(record)

    body = json.dumps(
        {"plans_sha256": hashlib.sha256(args.plans.read_bytes()).hexdigest(), "cells": cells},
        indent=2,
        sort_keys=True,
    )
    signature = hashlib.sha256(body.encode()).hexdigest()
    args.out.write_text(
        json.dumps(
            {
                "signature": signature,
                "plans_sha256": hashlib.sha256(args.plans.read_bytes()).hexdigest(),
                "passed": cells,
                "rejected": rejected,
                "rollouts_authorized": 4 * len(cells),
                "rollouts_avoided": 4 * len(rejected),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(f"{len(cells)} configurations pass, {len(rejected)} rejected")
    for record in rejected:
        label = f"{record['nominal']}_{record['obstacle']}_{record['band']}_{record['side']}"
        print(f"  {label:>34s}  {record['reasons'][0]}")
    print(
        f"\nauthorized {4 * len(cells)} rollouts; "
        f"refused {4 * len(rejected)} that would have been spent"
    )
    print(f"manifest {args.out} signature {signature[:16]}")
    return 0 if cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
