#!/usr/bin/env python3
"""Build and verify one counterfactual family end to end, in physics.

The geometric prediction is cheap: binary-search a shelf against the executed swept volume
and read off the height at which one motion interferes and another does not. That prediction
is not a result. This runs the 2x2 that turns it into one.

Protocol, six rollouts:

1. **probe** -- run the nominal and adapted motions on a bare plane, to obtain the executed
   swept volume each one actually produces. The scene cannot be built from a reference,
   because the question is whether *this trajectory* collides.
2. **boundary** -- binary-search the shelf height at which the nominal motion's swept volume
   first interferes.
3. **verify** -- build an easy and a hard scene around that boundary and roll *both* motions
   through *both*, giving the 2x2 the claim rests on:

   ================  =============  =============
   scene             nominal        adapted
   ================  =============  =============
   easy              expect pass    expect pass
   hard              **expect fail**  **expect pass**
   ================  =============  =============

The bottom row is the family. If the hard scene stops the nominal motion and not the
adapted one, geometry decided the behaviour, and the pair is supervision no
scene-around-motion episode can provide.

Usage::

    python scripts/research/build_counterfactual_family.py \\
        --nominal 005 --adapted 053 --motions /data/.../taxonomy/motions_4s \\
        --work /data/.../counterfactual/duck_000
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.clutter_scene_builder import (  # noqa: E402
    ClutterSceneSpec,
    FurniturePiece,
    render_scene_usda,
)
from gear_sonic.dataset_generation.motion_envelope import (  # noqa: E402
    best_overhead_station,
)
from gear_sonic.dataset_generation.counterfactual_family import (  # noqa: E402
    CounterfactualError,
    ObstacleSpec,
    swept_clearance_to_box,
    build_paired_family,
)
from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.route_placement import canonical_path_xy  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

PYTHON = Path.home() / "miniconda3/envs/env_isaaclab/bin/python"
SCENES_ROOT = REPO_ROOT / "gear_sonic/data/assets/scenes"

#: Clearance kept between the furthest point either motion reaches and the wall, in metres.
#: The measured body half-width is 0.664 m; the rest is margin for tracking drift.
WALL_MARGIN_M = 1.2

#: Floor under which a room is not built, in metres, so a short motion still gets a room.
MIN_ROOM_SIZE_XY = (8.0, 5.0)
WALL_HEIGHT = 2.8
SHELF_SIZE = (0.5, 3.0, 0.10)


def run(command: list[str], log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return "PASS" in log.read_text(encoding="utf-8", errors="replace")


def convert(csv: Path, out: Path, key: str, start_xy) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(PYTHON), str(REPO_ROOT / "gear_sonic/data_process/convert_kimodo_to_motion_lib.py"),
            "--input", str(csv), "--output", str(out), "--motion-key", key,
            "--source-fps", "30",
            "--scene-start", str(start_xy[0]), str(start_xy[1]), "0.0", "--scene-yaw", "0.0",
        ],
        capture_output=True, text=True, env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    return result.returncode == 0


def rollout(scene: str, motion: Path, out: Path, log: Path, task: str) -> bool:
    return run(
        [
            str(REPO_ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
            "--scene", scene, "--motion", str(motion), "--out", str(out),
            "--max-steps", "auto", "--task", task,
        ],
        log,
    )


def executed_bodies(rollout_dir: Path):
    paths = sorted(rollout_dir.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectory recorded in {rollout_dir}")
    with paths[0].open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def room_size_for(paths: list[np.ndarray]) -> tuple[float, float]:
    """A room that holds every motion in the family, not just the nominal one.

    Sizing from the nominal path alone put the adapted motion into the far wall: it
    travelled 4.85 m against the nominal's 4.08 m, and a 10 m room centred on the shorter
    one leaves its wall at 5.0 m. The result was 710.9 N of leg-against-wall contact in
    *both* scenes -- identical in each, which is what gave it away as unrelated to the shelf.
    """
    extent_x = max(float(np.abs(p[:, 0]).max()) for p in paths)
    extent_y = max(float(np.abs(p[:, 1]).max()) for p in paths)
    return (
        max(2 * (extent_x + WALL_MARGIN_M), MIN_ROOM_SIZE_XY[0]),
        max(2 * (extent_y + WALL_MARGIN_M), MIN_ROOM_SIZE_XY[1]),
    )


def write_scene(
    scene_id: str,
    path_xy: np.ndarray,
    shelf_z_base: float,
    start_xy,
    room_size_xy: tuple[float, float],
    shelf_center_xy,
) -> Path:
    """A bare room whose only furniture is one shelf spanning the corridor.

    ``shelf_center_xy`` is required rather than recomputed. It used to be derived here as the
    path midpoint, which silently disagreed with the position the boundary search had
    optimised: the search placed its obstacle at the station where the two motions differ
    most, and the scene then rendered a shelf half a metre away where the duck had not begun.
    Both motions hit it, and a 0.178 m window produced no family at all. One shelf position
    must reach both the search and the renderer, so it is passed in.
    """
    mid = shelf_center_xy
    piece = FurniturePiece(
        name="LowShelf_00",
        kind="WallShelf",
        center_xy=(float(mid[0]), float(mid[1])),
        size=SHELF_SIZE,
        color=(0.46, 0.34, 0.22),
        z_base=float(shelf_z_base),
        band="overhead",
    )
    spec = ClutterSceneSpec(
        scene_id=scene_id,
        split_group=f"{scene_id}_family_v1",
        room_size_xy=room_size_xy,
        wall_height=WALL_HEIGHT,
        pieces=[piece],
        path_xy=path_xy,
        clearance_m=0.0,
        seed=0,
        metrics={
            "scene_start_xy": [float(start_xy[0]), float(start_xy[1])],
            "placed_pieces": 1,
            # render_scene_usda writes these into the file's provenance header, so they are
            # required even for a scene whose only furniture is a single shelf.
            "clutter_occupancy": 0.0,
            "min_distance_to_path_m": 0.0,
            "path_length_m": float(
                np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()
            ),
        },
    )
    directory = SCENES_ROOT / "g1_counterfactual"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{scene_id}.usda"
    destination.write_text(render_scene_usda(spec), encoding="utf-8")
    return destination


def rendered_shelf_box(usda: Path) -> tuple[float, ...]:
    """The shelf's world box, read back out of the file physics will load.

    Recomputing it from the builder's own variables would only confirm the builder agrees
    with itself, which is exactly what was true while the bug was live.
    """
    text = usda.read_text(encoding="utf-8")
    block = text[text.index("LowShelf_00"):]
    size = [float(v) for v in
            re.search(r"double3 xformOp:scale = \(([^)]+)\)", block).group(1).split(",")]
    trans = [float(v) for v in
             re.search(r"double3 xformOp:translate = \(([^)]+)\)", block).group(1).split(",")]
    half = [s / 2.0 for s in size]
    return (
        trans[0] - half[0], trans[1] - half[1], trans[2] - half[2],
        trans[0] + half[0], trans[1] + half[1], trans[2] + half[2],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", required=True, help="taxonomy index, e.g. 005")
    parser.add_argument("--adapted", required=True, help="taxonomy index, e.g. 053")
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()

    def find(index: str) -> Path:
        matches = sorted(glob.glob(str(args.motions / f"{index}_*.csv")))
        if not matches:
            raise SystemExit(f"no motion CSV for index {index} in {args.motions}")
        return Path(matches[0])

    nominal_csv, adapted_csv = find(args.nominal), find(args.adapted)
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"nominal  {nominal_csv.name[:60]}")
    print(f"adapted  {adapted_csv.name[:60]}")

    # The path both motions share, taken from the nominal one and recentred on the room.
    path_xy = canonical_path_xy(np.loadtxt(nominal_csv, delimiter=","))
    start_xy = (float(path_xy[0][0]), float(path_xy[0][1]))

    # --- 1. probe on a bare plane, to get each motion's executed swept volume -------------
    probes: dict[str, dict] = {}
    for label, csv in (("nominal", nominal_csv), ("adapted", adapted_csv)):
        out = args.work / f"probe_{label}"
        if not (out / "success_manifest.json").exists():
            motion = args.work / "motions" / f"probe_{label}.pkl"
            if not convert(csv, motion, f"probe_{label}", start_xy):
                raise SystemExit(f"conversion failed for {label}")
            print(f"  probing {label} on a bare plane ...")
            rollout("plane", motion, out, args.work / "logs" / f"probe_{label}.log",
                    "walk forward across the open floor")
        probes[label] = executed_bodies(out)
        print(f"  {label}: {probes[label]['total_frames']} frames recorded")

    # --- 2. choose where to put the shelf, then search its height ------------------------
    # Not the path midpoint. An obstacle only separates two motions where they actually
    # differ, and the adapted motion ducks over a stretch of its route rather than all of
    # it. On the first family the midpoint caught the leading edge of the duck and yielded a
    # 0.053 m window, while the station where the duck is deepest offers 0.18 m -- the
    # difference between a 3 N graze and a decisive collision.
    station_x, predicted_window = best_overhead_station(probes["nominal"], probes["adapted"])
    if np.isfinite(station_x):
        # Follow the nominal route's y at that station, so the shelf still spans the corridor.
        index = int(np.argmin(np.abs(path_xy[:, 0] - station_x)))
        mid = path_xy[index]
        print(f"\nshelf station x = {station_x:.2f} m "
              f"(predicted window {predicted_window:.3f} m); "
              f"path midpoint would have been x = {path_xy[len(path_xy) // 2][0]:.2f} m")
    else:
        mid = path_xy[len(path_xy) // 2]
        print("\nno station separates the motions; falling back to the path midpoint")

    shelf = ObstacleSpec(
        name="LowShelf", size=SHELF_SIZE,
        base_center=(float(mid[0]), float(mid[1]), 2.30),
        axis=(0.0, 0.0, -1.0), regime="overhead",
    )
    def bodies(label):
        payload = probes[label]
        return (
            np.asarray(payload["body_pos_w"]),
            np.asarray(payload["body_quat_w"]),
            list(payload["body_names"]),
        )

    try:
        family = build_paired_family(
            f"cf_{args.nominal}_{args.adapted}",
            bodies("nominal"), bodies("adapted"), shelf, search_high=1.6,
        )
    except CounterfactualError as error:
        raise SystemExit(f"no family for this pair: {error}") from error

    easy_z = shelf.box_at(family.easy_parameter)[2]
    hard_z = shelf.box_at(family.hard_parameter)[2]
    nominal_z = shelf.box_at(family.nominal_boundary.parameter)[2]
    adapted_z = shelf.box_at(family.adapted_boundary.parameter)[2]
    print(f"\nnominal clears a shelf down to {nominal_z:.3f} m")
    print(f"adapted clears a shelf down to {adapted_z:.3f} m")
    print(f"WINDOW {family.window_m:.3f} m -- a shelf between those heights should stop the "
          "nominal motion and pass the adapted one")
    print(f"  easy scene shelf underside {easy_z:.3f} m")
    print(f"  hard scene shelf underside {hard_z:.3f} m")

    # --- 3. build both scenes and run the 2x2 ---------------------------------------------
    # Both motions must fit: the room is sized from whichever travels furthest.
    executed_paths = [
        np.asarray(probes[label]["root_pos_w"])[:, :2] for label in ("nominal", "adapted")
    ]
    room = room_size_for([path_xy, *executed_paths])
    print(f"room sized {room[0]:.1f} x {room[1]:.1f} m to hold both motions")

    scenes = {}
    for label, z_base in (("easy", easy_z), ("hard", hard_z)):
        scene_id = f"{family.family_id}_{label}"
        write_scene(scene_id, path_xy, z_base, start_xy, room, mid)
        scenes[label] = scene_id
    print(f"\nwrote scenes: {', '.join(scenes.values())}")

    # --- 3a. check the scenes actually implement the family, before spending rollouts -----
    # The search reasons about an ObstacleSpec; the renderer writes a USD prim. Nothing used
    # to check the two agreed, and they silently did not: the shelf was rendered at the path
    # midpoint while the search had placed it at the station where the motions differ. Both
    # motions hit it and a 0.178 m window produced no family. Reading the geometry back out
    # of the file that physics will load is the only way to know what was really built.
    expectations = {
        ("nominal", "easy"): +1, ("nominal", "hard"): -1,
        ("adapted", "easy"): +1, ("adapted", "hard"): +1,
    }
    disagreements = []
    for (motion_label, scene_label), expected_sign in expectations.items():
        box = rendered_shelf_box(SCENES_ROOT / "g1_counterfactual" /
                                 f"{scenes[scene_label]}.usda")
        clearance, _ = swept_clearance_to_box(*bodies(motion_label), box)
        if np.sign(clearance) != expected_sign:
            disagreements.append(
                f"{motion_label}/{scene_label}: rendered clearance {clearance:+.4f} m, "
                f"expected {'clear' if expected_sign > 0 else 'interference'}"
            )
    if disagreements:
        print("\nthe scenes do not implement the family the search found:")
        for line in disagreements:
            print(f"  {line}")
        raise SystemExit(
            "refusing to roll out: the rendered geometry disagrees with the search"
        )
    print("scenes verified against the search: all four clearances have the intended sign")

    results: dict[str, dict] = {}
    for motion_label, csv in (("nominal", nominal_csv), ("adapted", adapted_csv)):
        for scene_label, scene_id in scenes.items():
            key = f"{motion_label}_{scene_label}"
            out = args.work / key
            if not (out / "success_manifest.json").exists():
                motion = args.work / "motions" / f"{key}.pkl"
                if not convert(csv, motion, key, start_xy):
                    print(f"  CONVFAIL {key}")
                    continue
                print(f"  rolling out {key} ...")
                rollout(scene_id, motion, out, args.work / "logs" / f"{key}.log",
                        "walk forward through the room")
            trajectories = sorted(out.glob("trajectories/*.trajectory.pkl"))
            if not trajectories:
                results[key] = {"outcome": "no_rollout"}
                continue
            with trajectories[0].open("rb") as handle:
                raw = pickle.load(handle)  # noqa: S301
            outcome = classify_episode(key, raw)
            from gear_sonic.dataset_generation.contact_decomposition import (
                decompose_payload_contacts,
            )
            payload, _ = best_evaluable_payload(raw)
            results[key] = {
                "outcome": outcome.outcome,
                "reasons": list(outcome.rejection_reasons),
                "lateral_contact_n": decompose_payload_contacts(payload).max_lateral_contact,
            }

    print(f"\n{'':>10} {'easy scene':>28} {'hard scene':>28}")
    for motion_label in ("nominal", "adapted"):
        row = []
        for scene_label in ("easy", "hard"):
            entry = results.get(f"{motion_label}_{scene_label}", {})
            row.append(
                f"{entry.get('outcome','?')} ({entry.get('lateral_contact_n',float('nan')):.1f} N)"
            )
        print(f"{motion_label:>10} {row[0]:>28} {row[1]:>28}")

    separated = (
        results.get("nominal_hard", {}).get("outcome") == "rejected"
        and results.get("adapted_hard", {}).get("outcome") == "accepted"
    )
    print(f"\ncounterfactual established: {separated}")
    if not separated:
        print("  the hard scene did not separate the two motions in physics; the family is a "
              "candidate, not a result.")

    (args.work / "family.json").write_text(
        json.dumps(
            {
                "family_id": family.family_id,
                "regime": family.regime,
                "nominal_motion": nominal_csv.name,
                "adapted_motion": adapted_csv.name,
                "nominal_boundary_parameter": family.nominal_boundary.parameter,
                "adapted_boundary_parameter": family.adapted_boundary.parameter,
                "window_m": family.window_m,
                "nominal_clears_to_m": nominal_z,
                "adapted_clears_to_m": adapted_z,
                "easy_shelf_underside_m": easy_z,
                "hard_shelf_underside_m": hard_z,
                "results": results,
                "counterfactual_established": separated,
            },
            indent=2, sort_keys=True, default=float,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.work / 'family.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
