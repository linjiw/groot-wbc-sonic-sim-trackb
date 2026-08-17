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
from gear_sonic.dataset_generation.counterfactual_family import (  # noqa: E402
    CounterfactualError,
    ObstacleSpec,
    build_family,
    swept_clearance_to_box,
)
from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.route_placement import canonical_path_xy  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

PYTHON = Path.home() / "miniconda3/envs/env_isaaclab/bin/python"
SCENES_ROOT = REPO_ROOT / "gear_sonic/data/assets/scenes"

#: Room large enough to hold a 4 s walk with margin on every side.
ROOM_SIZE_XY = (10.0, 6.0)
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


def write_scene(scene_id: str, path_xy: np.ndarray, shelf_z_base: float, start_xy) -> Path:
    """A bare room whose only furniture is one shelf spanning the corridor."""
    mid = path_xy[len(path_xy) // 2]
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
        room_size_xy=ROOM_SIZE_XY,
        wall_height=WALL_HEIGHT,
        pieces=[piece],
        path_xy=path_xy,
        clearance_m=0.0,
        seed=0,
        metrics={"scene_start_xy": [float(start_xy[0]), float(start_xy[1])], "placed_pieces": 1},
    )
    directory = SCENES_ROOT / "g1_counterfactual"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{scene_id}.usda"
    destination.write_text(render_scene_usda(spec), encoding="utf-8")
    return destination


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

    # --- 2. binary-search the shelf height the nominal motion first interferes with -------
    mid = path_xy[len(path_xy) // 2]
    shelf = ObstacleSpec(
        name="LowShelf", size=SHELF_SIZE,
        base_center=(float(mid[0]), float(mid[1]), 2.30),
        axis=(0.0, 0.0, -1.0), regime="overhead",
    )
    nominal = probes["nominal"]
    try:
        family = build_family(
            f"cf_{args.nominal}_{args.adapted}",
            np.asarray(nominal["body_pos_w"]), np.asarray(nominal["body_quat_w"]),
            list(nominal["body_names"]), shelf,
            search_low=0.0, search_high=1.5, margin_m=args.margin,
        )
    except CounterfactualError as error:
        raise SystemExit(f"no boundary for this pair: {error}") from error

    adapted = probes["adapted"]
    adapted_hard, _ = swept_clearance_to_box(
        np.asarray(adapted["body_pos_w"]), np.asarray(adapted["body_quat_w"]),
        list(adapted["body_names"]), shelf.box_at(family.hard_parameter),
    )
    easy_z = shelf.box_at(family.easy_parameter)[2]
    hard_z = shelf.box_at(family.hard_parameter)[2]
    print(f"\nboundary at {family.boundary.parameter:.3f} m of lowering "
          f"({family.boundary.iterations} iterations)")
    print(f"  easy shelf underside {easy_z:.3f} m   nominal clearance {family.easy_clearance_m:+.4f}")
    print(f"  hard shelf underside {hard_z:.3f} m   nominal clearance {family.hard_clearance_m:+.4f}"
          f"   adapted clearance {adapted_hard:+.4f}")
    if adapted_hard <= 0:
        print("  NOTE: the adapted motion also interferes at this height, so the pair does not "
              "separate geometrically. The 2x2 below will say what physics makes of it.")

    # --- 3. build both scenes and run the 2x2 ---------------------------------------------
    scenes = {}
    for label, z_base in (("easy", easy_z), ("hard", hard_z)):
        scene_id = f"{family.family_id}_{label}"
        write_scene(scene_id, path_xy, z_base, start_xy)
        scenes[label] = scene_id
    print(f"\nwrote scenes: {', '.join(scenes.values())}")

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
                "boundary_parameter": family.boundary.parameter,
                "easy_shelf_underside_m": easy_z,
                "hard_shelf_underside_m": hard_z,
                "nominal_easy_clearance_m": family.easy_clearance_m,
                "nominal_hard_clearance_m": family.hard_clearance_m,
                "adapted_hard_clearance_m": adapted_hard,
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
