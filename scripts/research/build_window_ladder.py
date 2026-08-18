#!/usr/bin/env python
"""Place a ladder of shelf heights across a family's predicted window.

The swept-volume search predicts each motion's own clearance boundary, and a family is built by
putting one shelf between them. That is a *centre* placement -- maximally robust, and silent
about where the real transitions are. If the prediction is off by 40 mm, the family still works
and nobody finds out.

This writes scenes at several heights spanning the window so the predicted boundaries can be
compared against the observed ones. Against the family's own scene each rung differs only in the
shelf's height and its scene id, with one exception worth recording: the room's walls move by
0.2 mm, because the floor's width is stored to three decimals and re-deriving wall positions from
it rounds. The walls stand 5.37 m from a shelf interaction at x = 2.23 m, so this cannot reach
the result, but a reader comparing the files should not have to wonder. It also repairs the family manifest, which recorded the two
shelf heights but not the executed root path or the room size, so the scenes it described could
not be rebuilt from it.
"""

from __future__ import annotations

import argparse
import json
import re
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import build_counterfactual_family as cf  # noqa: E402

from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload  # noqa: E402

#: Offsets inward from each predicted boundary, in metres. Near-threshold probes come first
#: because they are the informative ones: a family at the window's centre is already known to
#: work, so only the edges can move the calibration.
NEAR = (0.008, 0.025)
#: Clearance added around the executed path when sizing the room. Rooms sized from one motion
#: have caused a family to fail on a wall rather than on the obstacle.
ROOM_MARGIN_M = 1.6


def executed_path(rollout: Path) -> np.ndarray:
    paths = sorted(rollout.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectory under {rollout}")
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        raise SystemExit(f"{rollout.name} is unevaluable, so its path cannot size a room")
    return np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--nominal-rollout", type=Path, required=True)
    ap.add_argument("--adapted-rollout", type=Path)
    ap.add_argument("--write-scenes", action="store_true")
    ap.add_argument(
        "--match-scene", type=Path,
        help="An existing family scene whose room and shelf centre the ladder must reuse. "
             "Read back out of the USDA physics loads rather than recomputed, because a room "
             "resized between rungs would confound the ladder with a room change.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    lo = float(manifest["adapted_peak_m"])
    hi = float(manifest["nominal_peak_m"])
    station = float(manifest["station_x_m"])
    family = manifest["family_id"]

    paths = [executed_path(args.nominal_rollout)]
    if args.adapted_rollout and args.adapted_rollout.exists():
        paths.append(executed_path(args.adapted_rollout))
    else:
        print("note: adapted rollout absent, room sized from the nominal path alone")
    path = np.concatenate(paths)

    start = tuple(float(v) for v in paths[0][0])
    if args.match_scene:
        text = args.match_scene.read_text(encoding="utf-8")
        floor = text[text.index('Plane "Floor"'):]
        room = (float(re.search(r"double width = ([\d.]+)", floor).group(1)),
                float(re.search(r"double length = ([\d.]+)", floor).group(1)))
        box = cf.rendered_shelf_box(args.match_scene)
        centre = ((box[0] + box[3]) / 2.0, (box[1] + box[4]) / 2.0)
        print(f"matched {args.match_scene.name}: room {room[0]:.3f} x {room[1]:.3f} m, "
              f"shelf centre x={centre[0]:.3f} y={centre[1]:.3f}, underside {box[2]:.4f} m")
    else:
        span = path.max(0) - path.min(0)
        room = (float(span[0] + 2 * ROOM_MARGIN_M), float(max(span[1] + 2 * ROOM_MARGIN_M, 5.0)))
        centre = (station, float(np.interp(station, path[:, 0], path[:, 1])))

    rungs = []
    for d in NEAR:
        rungs.append(("near_nominal", round(hi - d, 4)))
        rungs.append(("near_adapted", round(lo + d, 4)))
    rungs.sort(key=lambda r: -r[1])

    print(f"family {family}   predicted window {lo:.4f} -> {hi:.4f} m ({(hi - lo) * 1000:.1f} mm)")
    print(f"room {room[0]:.1f} x {room[1]:.1f} m   shelf centre x={centre[0]:.3f} y={centre[1]:.3f}")
    print(f"\n{'rung':>14s}{'underside':>12s}  predicts")
    for kind, z in rungs:
        pred = ("nominal fails, crouch clears" if lo < z < hi
                else "both clear" if z >= hi else "both fail")
        print(f"{kind:>14s}{z:12.4f}  {pred}")
        if args.write_scenes:
            scene = cf.write_scene(
                f"{family}_z{int(round(z * 1000))}", path, z, start, room, centre)
            print(f"{'':14s}{'':12s}  wrote {Path(scene).name}")

    manifest["path_xy"] = [[round(float(x), 5), round(float(y), 5)] for x, y in path[::5]]
    manifest["room_size_xy_m"] = list(room)
    manifest["shelf_center_xy_m"] = list(centre)
    manifest["start_xy_m"] = list(start)
    manifest["ladder_heights_m"] = {k: z for k, z in rungs}
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nrepaired {args.manifest.name}: it now records the path, room and shelf centre, so "
          "these scenes can be rebuilt from it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
