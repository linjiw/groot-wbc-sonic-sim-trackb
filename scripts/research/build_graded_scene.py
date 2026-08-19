#!/usr/bin/env python
"""Write a scene whose difficulty was chosen, and verify it before a rollout is spent.

Takes a configuration from ``plan_graded_families`` -- a binding body part, the obstacle type that
binds it, and a margin -- and places the solid so the executed body clears it by exactly that
margin. A ceiling spans the corridor at a chosen height; a wall stands beside the route at a chosen
lateral offset, occupying the band that selects the part it binds.

Nothing here is searched. The placement follows from the reach the criticality map measured, and the
result is checked against the executed capsules before it is written, because a scene the robot
never meets costs a rollout to discover and teaches nothing.
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))


from gear_sonic.dataset_generation.clutter_scene_builder import (  # noqa: E402
    ClutterSceneSpec,
    FurniturePiece,
    render_scene_usda,
)
from gear_sonic.dataset_generation.criticality_map import DEFAULT_BANDS  # noqa: E402
from gear_sonic.dataset_generation.scene_route_check import (  # noqa: E402
    capsule_box_clearance,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
#: A wall is thin across the route and long along it, so the robot passes beside a surface rather
#: than clipping a post. Metres.
WALL_THICKNESS = 0.12
WALL_LENGTH = 1.20
CEILING_SPAN = 3.0
CEILING_DEPTH = 0.5
CEILING_THICKNESS = 0.1


def executed(directory: Path):
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        raise SystemExit(f"{directory.name} is unevaluable")
    return payload


def place(payload, starts, ends, radii, plan: dict, margin_m: float, station: float):
    """Where the solid goes, in world coordinates, for the chosen margin.

    The criticality map's reach is measured **from the root at each frame**, which is the right
    frame for asking "which part sticks out furthest" and the wrong one for placing a solid that
    does not move. The robot's own lateral position varies as it walks past, so a wall placed at
    ``station_root + reach`` was met at −36.8 mm when +50 mm was asked for.

    So the placement is computed in world coordinates: the furthest any capsule surface projects
    along the lateral axis, over the frames the obstacle is beside, plus the margin.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    index = int(np.clip(round(station * (len(root) - 1)), 0, len(root) - 1))
    here = root[index, :2]

    quat = np.asarray(payload["root_quat_w"], dtype=np.float64)[index]
    w, x, y, z = quat
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])

    high = np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]
    low = np.minimum(starts[:, :, 2], ends[:, :, 2]) - radii[None, :]

    if plan["obstacle"] == "ceiling":
        # Only the frames beneath the solid matter, so restrict to where the route runs under it.
        along = (root[:, :2] - here) @ forward
        near = np.abs(along) <= CEILING_DEPTH / 2.0 + 0.1
        if not near.any():
            near = np.ones(len(root), dtype=bool)
        z_base = float(high[near].max()) + margin_m
        return (
            (float(here[0]), float(here[1])),
            (CEILING_DEPTH, CEILING_SPAN, CEILING_THICKNESS),
            z_base,
        )

    band = {name: (z0, z1) for name, z0, z1 in DEFAULT_BANDS}[plan["band"]]
    sign = 1.0 if plan["side"] == "left" else -1.0
    in_band = (high > band[0]) & (low < band[1])

    # World-frame lateral extent, over the frames the wall spans and the capsules inside the band.
    centres = 0.5 * (starts + ends)
    along = (centres[:, :, :2] - here) @ forward
    beside = np.abs(along) <= WALL_LENGTH / 2.0
    projected = (centres[:, :, :2] - here) @ lateral * sign + radii[None, :]
    usable = in_band & beside
    if not usable.any():
        raise SystemExit(f"nothing occupies {plan['band']} beside the station; cannot place a wall")
    extent = float(projected[usable].max())

    offset = extent + margin_m + WALL_THICKNESS / 2.0
    centre = here + lateral * sign * offset
    return (
        (float(centre[0]), float(centre[1])),
        (WALL_LENGTH, WALL_THICKNESS, float(band[1] - band[0])),
        float(band[0]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--rollout", type=Path, required=True, help="executed nominal, for placement")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--station", type=float, default=0.55)
    # "centre" is not a fixed margin: it resolves per configuration to half that configuration's
    # own window, which is where the calibration says a hard shelf belongs. A single number cannot
    # express it, because each configuration has a different window.
    ap.add_argument("--margins", nargs="+", default=["0.050", "centre"])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    payload = executed(args.rollout)
    plans = json.loads(args.plans.read_text())["plans"]
    usable = [p for p in plans if p["usable"]]
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]
    span = root.max(0) - root.min(0)
    room = (float(span[0] + 4.0), float(max(span[1] + 4.0, 5.0)))

    starts, ends, radii, names = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )

    print(f"{len(usable)} usable configurations x {len(args.margins)} margins\n")
    print(f"{'scene':>34s}{'margin':>9s}{'measured':>11s}{'binds':>22s}")
    written = 0
    for plan in usable:
        for spec in args.margins:
            if spec == "centre":
                margin = -plan["window_m"] / 2.0
                tag_override = "hard"
            else:
                margin = float(spec)
                tag_override = None
            # Placement from a projected extent lands 5-36 mm short, because a box meets the body
            # at its corners rather than along the lateral axis and no simple projection accounts
            # for that. Rather than derive the corner geometry, the placement is corrected against
            # the measurement it is trying to achieve: two or three passes converge to well under a
            # millimetre, and the margin is then exact by construction rather than approximately
            # right.
            adjust = 0.0
            for _ in range(6):
                centre, size, z_base = place(
                    payload, starts, ends, radii, plan, margin + adjust, args.station
                )
                piece = FurniturePiece(
                    name="LowShelf_00",
                    kind="WallShelf",
                    center_xy=centre,
                    size=size,
                    color=(0.46, 0.34, 0.22),
                    z_base=z_base,
                    band=plan["band"],
                )
                gap, frame, capsule = capsule_box_clearance(starts, ends, radii, piece.box)
                error = margin - gap
                if abs(error) < 0.001:
                    break
                # Clearance saturates once a capsule is wholly inside the solid: the distance
                # clamps to zero and the reading becomes -radius, about -50 mm for a wrist. A
                # requested margin deeper than that cannot be hit and the loop would chase it
                # forever. Deep overlaps only need the nominal to fail, which they do, so stop.
                if gap <= -0.049 and margin < gap:
                    break
                adjust += error
            tag = tag_override or ("easy" if margin > 0.03 else f"m{int(round(margin*1000)):+d}")
            scene_id = f"{args.prefix}_{plan['obstacle']}_{plan['band']}_{plan['side']}_{tag}"
            print(
                f"{scene_id[:34]:>34s}{margin * 1000:8.0f}mm{gap * 1000:10.1f}mm"
                f"{names[capsule]:>22s}"
            )
            if not args.write:
                continue
            spec = ClutterSceneSpec(
                scene_id=scene_id,
                split_group=f"{scene_id}_family_v1",
                room_size_xy=room,
                wall_height=2.8,
                pieces=[piece],
                path_xy=root,
                clearance_m=0.0,
                seed=0,
                metrics={
                    "scene_start_xy": [float(root[0, 0]), float(root[0, 1])],
                    "placed_pieces": 1,
                    "clutter_occupancy": 0.0,
                    "min_distance_to_path_m": 0.0,
                    "path_length_m": float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum()),
                    # What physics will actually meet, not what the plan predicted. The plan
                    # reasons from root-relative reach and can name a neighbouring capsule -- it
                    # predicts the elbow where the shoulder turns out to be nearest -- and a
                    # manifest that records the prediction would mislabel the scene it describes.
                    "binding_body": names[capsule],
                    "predicted_binding_body": plan["binding_body"],
                    "intended_margin_m": margin,
                    "measured_clearance_m": round(gap, 4),
                    "closest_frame": frame,
                },
            )
            (SCENES / f"{scene_id}.usda").write_text(render_scene_usda(spec), encoding="utf-8")
            written += 1

    print("\nmeasured clearance is the real gap to the executed body; it should track the margin.")
    if args.write:
        print(f"wrote {written} scenes to {SCENES}")
    else:
        print("dry run -- pass --write to emit the scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
