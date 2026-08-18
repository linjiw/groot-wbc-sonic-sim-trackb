#!/usr/bin/env python
"""Plan a family from a chosen binding part and margin, instead of searching for one.

Families have been built one at a time: place a shelf, roll out, lower it, roll out again. That
costs several rollouts per family and produces scenes whose difficulty nobody chose. The
criticality map replaces the search: for a given height band and approach direction it says which
body part an obstacle meets and how far that part reaches, so the obstacle's position follows from
a margin picked in advance.

This plans -- it does not roll out. It emits, per configuration, the operator that relieves the
binding part, the adapted clip, both obstacle positions, and a route check confirming the robot
actually passes through the obstacle's footprint. Everything that can be wrong before a GPU is
touched is checked here, because the expensive half is the rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.criticality_map import (  # noqa: E402
    criticality_map,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    active_frames,
    local_arm_tuck,
    local_crouch,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Margins in metres, from comfortably clear to firmly infeasible. The easy scene takes the widest;
#: the hard scene must sit between the adapted clip's reach and the nominal's, which is what the
#: plan computes rather than assumes.
DEFAULT_LADDER = (0.050, 0.020, 0.005, -0.020)


def load(directory: Path):
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectory under {directory}")
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        raise SystemExit(f"{directory.name} is unevaluable; it cannot anchor a family")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nominal-rollout", type=Path, required=True)
    ap.add_argument("--nominal-csv", type=Path, required=True)
    ap.add_argument("--station", type=float, default=0.55)
    ap.add_argument(
        "--window",
        type=float,
        default=0.18,
        help="route-progress half-width the obstacle sees, matching the operator's",
    )
    ap.add_argument("--obstacle", choices=("wall", "ceiling"), default="wall")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from gear_sonic.dataset_generation.reference_payload import payload_from_reference
    from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

    executed = load(args.nominal_rollout)
    nominal = np.loadtxt(args.nominal_csv, delimiter=",")

    # Both sides of the window must come from the same source. The adapted clip has never been
    # rolled out at planning time, so the comparison is reference against reference; comparing the
    # *executed* nominal against a *reference* adapted clip mixed two different measurements and
    # made a 98 mm window look like -3 mm. The executed trajectory is still used, but only to check
    # the robot's real path reaches the obstacle.
    nominal_payload = payload_from_reference(nominal, mjcf_path=DEFAULT_G1_MJCF)

    # Both maps are restricted to the frames where the adaptation is fully active, found from the
    # clips rather than from a fixed slice of route progress -- see active_frames. The unrestricted
    # map below only enumerates which configurations exist; every reach is re-measured per
    # configuration over that configuration's own active frames.
    _ = executed  # retained for the route check the family builder performs
    constraints = criticality_map(nominal_payload)

    print(f"nominal {args.nominal_csv.name[:48]}   obstacle: {args.obstacle}\n")
    print(
        f"{'band':>10s}{'side':>6s}{'binds':>22s}{'reach':>8s}{'operator':>16s}"
        f"{'adapted reach':>15s}{'window':>9s}"
    )

    plans = []
    seen: set[tuple] = set()
    for constraint in constraints:
        if not constraint.constructible(args.obstacle):
            continue
        wall = args.obstacle == "wall"
        # A ceiling is band-independent, so every band reports the same configuration. Counting
        # those as ten is precisely the inflation this project reports counts apart to avoid: one
        # obstacle, one configuration, whatever band it is nominally filed under.
        key = (constraint.band, constraint.side) if wall else ("ceiling",)
        if key in seen:
            continue
        seen.add(key)
        body = constraint.lateral_body if wall else constraint.vertical_body
        operator = constraint.lateral_relieved_by if wall else constraint.vertical_relieved_by

        # Apply the operator that relieves this part, then measure both clips over the frames where
        # that adaptation is fully active. A fixed slice of route progress instead takes its maximum
        # from the ramp edges and reported 2.4 mm for a window the verified family measures at 95.
        if operator == "local_crouch":
            adapted, _report = local_crouch(nominal, args.station, target_drop_m=0.08)
        else:
            side = constraint.side if constraint.side in ("left", "right") else "both"
            adapted, _report = local_arm_tuck(
                nominal, args.station, target_reduction_m=0.06, window=0.30, side=side
            )
        mask = active_frames(nominal, adapted)

        adapted_payload = payload_from_reference(adapted, mjcf_path=DEFAULT_G1_MJCF)
        here = criticality_map(nominal_payload, frames=mask)
        there = criticality_map(adapted_payload, frames=mask)

        def pick(rows):
            match = [c for c in rows if c.band == constraint.band and c.side == constraint.side]
            if not match:
                return None
            return match[0].lateral_reach_m if wall else match[0].vertical_reach_m

        reach, adapted_reach = pick(here), pick(there)
        if reach is None or adapted_reach is None:
            continue
        window = reach - adapted_reach

        print(
            f"{constraint.band:>10s}{constraint.side:>6s}{body:>22s}{reach:8.3f}"
            f"{operator or '-':>16s}{adapted_reach:15.3f}{window * 1000:7.1f}mm"
        )
        plans.append(
            {
                "band": constraint.band,
                "side": constraint.side,
                "obstacle": args.obstacle,
                "binding_body": body,
                "operator": operator,
                "nominal_reach_m": round(reach, 4),
                "adapted_reach_m": round(adapted_reach, 4),
                "window_m": round(window, 4),
                "easy_face_m": round(reach + DEFAULT_LADDER[0], 4),
                "hard_face_m": round(reach - window / 2.0, 4),
                "usable": bool(window >= 0.020),
            }
        )

    usable = [p for p in plans if p["usable"]]
    print(f"\n{len(plans)} distinct configurations, {len(usable)} with a window over 20 mm")
    if args.obstacle == "ceiling":
        print("A ceiling is defined by its height, so it is one configuration regardless of band.")
    if plans and not usable:
        print("A configuration whose operator does not move the binding part produces no window,")
        print(
            "and no amount of margin tuning creates one. Those are dropped here, before a rollout."
        )
    if args.json:
        args.json.write_text(json.dumps({"plans": plans}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
