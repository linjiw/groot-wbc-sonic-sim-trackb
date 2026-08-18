#!/usr/bin/env python3
"""Build a counterfactual family from *one* motion and its own local adaptation.

The existing families pair two separately generated clips -- a walk and a crouch that were
never the same motion. A reviewer can fairly ask whether the shelf separated the behaviours or
two different journeys, and the construction cannot answer.

Here the adapted motion is the nominal motion, bent locally around the obstacle station by
``local_crouch``. Root XY and yaw, duration, frame count, gait phase, start and goal are
identical by construction; the two clips differ by exactly the adaptation under test. That
also removes the mining step: the obstacle's station is not searched for, it is *chosen* first
and the adaptation is placed there.

The three gates run in order and the run stops at the first failure, so the reason is never
ambiguous:

1. **kinematic** -- the adapted reference is reachable, self-collision free, and its silhouette
   comes down by the requested amount
2. **trackable** -- both clips roll out and are accepted in the easy scene, with no scene
   contact. *Only then* is a hard scene built.
3. **counterfactual** -- the 2x2, and its attribution

Deliberately not done here: tuning the hard shelf until the family works. The hard height is
placed between the two measured boundaries and left alone; if the 2x2 fails, that is a result
about the pair rather than an invitation to search.

Usage::

    python scripts/research/build_matched_family.py \\
        --nominal 005 --motions /data/.../motions_4s --work /data/.../matched/mf_005 \\
        --station 0.55 --drop 0.18
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.gpu_capacity import wait_for_gpu  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    _silhouette,
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402

PYTHON = Path.home() / "miniconda3/envs/env_isaaclab/bin/python"

#: Metres of margin either side of a measured boundary. The easy shelf clears the nominal by
#: this much and the hard shelf sits this far below it.
BOUNDARY_MARGIN_M = 0.05

#: Depth of the shelf along the direction of travel, in metres. Matches the family builder's
#: SHELF_SIZE, and it is what decides which frames the obstacle actually sees.
SHELF_DEPTH_M = 0.5


def convert(csv: Path, out: Path, key: str, start_xy) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "gear_sonic/data_process/convert_kimodo_to_motion_lib.py"),
            "--input", str(csv), "--output", str(out), "--motion-key", key,
            "--source-fps", "30",
            "--scene-start", str(start_xy[0]), str(start_xy[1]), "0.0", "--scene-yaw", "0.0",
        ],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    return result.returncode == 0


def rollout(scene: str, motion: Path, out: Path, log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    wait_for_gpu()
    with log.open("w") as handle:
        subprocess.run(
            [
                str(REPO_ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
                "--scene", scene, "--motion", str(motion), "--out", str(out),
                "--max-steps", "auto", "--task", "walk forward through the room",
            ],
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    return "PASS" in log.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", required=True, help="taxonomy index, e.g. 005")
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--station", type=float, default=0.55)
    parser.add_argument("--drop", type=float, default=0.18)
    parser.add_argument("--stage", choices=("kinematic", "trackable", "all"), default="all")
    args = parser.parse_args()

    matches = sorted(glob.glob(str(args.motions / f"{args.nominal}_*.csv")))
    if not matches:
        raise SystemExit(f"no motion CSV for index {args.nominal} in {args.motions}")
    csv = Path(matches[0])
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"nominal  {csv.name[:64]}")

    nominal = np.loadtxt(csv, delimiter=",")

    # --- gate 1: kinematic ----------------------------------------------------------------
    print("\n=== gate 1: kinematic ===")
    adapted, report = local_crouch(nominal, args.station, target_drop_m=args.drop)
    verdict = screen_reference(adapted, f"{args.nominal}_adapted", "crouch_walk")
    print(f"  silhouette {report.nominal_silhouette_m:.3f} -> "
          f"{report.adapted_silhouette_m:.3f} m  (drop {report.silhouette_drop_m:.3f})")
    print(f"  waist change {report.waist_change_rad:.4f} rad, "
          f"foot shift {report.foot_height_shift_m:.4f} m, "
          f"active {report.active_fraction:.0%}")
    print(f"  reachable {verdict.embodiment_feasible}, "
          f"self-collision free {verdict.self_collision_free}, "
          f"root path preserved {report.root_path_preserved}")
    if not (verdict.embodiment_feasible and verdict.self_collision_free
            and report.root_path_preserved):
        print("\nFAILED gate 1; not spending a rollout")
        return 1
    print("  PASS")

    clips = args.work / "clips"
    clips.mkdir(exist_ok=True)
    np.savetxt(clips / "nominal.csv", nominal, delimiter=",", fmt="%.6f")
    np.savetxt(clips / "adapted.csv", adapted, delimiter=",", fmt="%.6f")

    # Where the obstacle goes, and the two boundaries it sits between. Both are read off the
    # reference silhouettes at the station rather than searched for -- the adaptation was
    # placed here, so this is where the motions differ by construction.
    progress = route_progress(nominal[:, :2])
    station_x = float(np.interp(args.station, progress, nominal[:, 0]))

    # Measure the peak over the frames the robot spends inside the *shelf's own x-span*, not
    # over a fixed slice of route progress. The adaptation ramps in and out, so a wider window
    # takes its maximum from the edges where the adapted clip is barely crouched -- which
    # reported a 0.014 m window from a 0.180 m drop.
    # The window is where the adaptation is *fully active*, found from the clips themselves.
    # A fixed slice of route progress takes its maximum from the ramp edges, where the
    # adaptation has barely begun -- that reported a 14 mm window for a 180 mm adaptation.
    departure = np.abs(adapted[:, 7:] - nominal[:, 7:]).max(axis=1)
    active = np.flatnonzero(departure > 0.5 * departure.max())
    margin = int(len(active) * 0.2)
    inside = np.zeros(len(nominal), dtype=bool)
    inside[active[margin:len(active) - margin] if margin else active] = True
    nominal_peak = float(_silhouette(nominal, DEFAULT_G1_MJCF)[inside].max())
    adapted_peak = float(_silhouette(adapted, DEFAULT_G1_MJCF)[inside].max())
    station_x = float(nominal[inside, 0].mean())
    easy_z = nominal_peak + BOUNDARY_MARGIN_M
    hard_z = adapted_peak + BOUNDARY_MARGIN_M

    summary = {
        "family_id": f"mf_{args.nominal}",
        "nominal_motion": csv.name,
        "station_fraction": args.station,
        "station_x_m": station_x,
        "target_drop_m": args.drop,
        "nominal_peak_at_station_m": nominal_peak,
        "adapted_peak_at_station_m": adapted_peak,
        "window_m": nominal_peak - adapted_peak,
        "frames_inside_shelf_span": int(inside.sum()),
        "easy_shelf_underside_m": easy_z,
        "hard_shelf_underside_m": hard_z,
        "kinematic": {
            "silhouette_drop_m": report.silhouette_drop_m,
            "waist_change_rad": report.waist_change_rad,
            "foot_height_shift_m": report.foot_height_shift_m,
            "active_fraction": report.active_fraction,
        },
    }
    print(f"\n  station x = {station_x:.2f} m")
    print(f"  nominal peak {nominal_peak:.3f} m, adapted peak {adapted_peak:.3f} m")
    print(f"  WINDOW {summary['window_m']:.3f} m")
    print(f"  easy shelf {easy_z:.3f} m, hard shelf {hard_z:.3f} m")
    (args.work / "family.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    if args.stage == "kinematic":
        return 0

    print("\n=== gate 2: trackable in an obstacle-free scene ===")
    print("  (not implemented as an automatic step yet -- run the clips through a")
    print("   scene with no interfering obstacle and confirm both are accepted before")
    print("   building a hard shelf. Tuning the hard scene before this passes is how a")
    print("   tracking failure gets misread as a counterfactual.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
