#!/usr/bin/env python3
"""Re-run a counterfactual family's 2x2 under small initial-state perturbations.

The family is discovered from an executed trajectory and verified by replaying that same
motion into a scene built around it. SONIC rollouts are highly deterministic under a fixed
configuration, which is exactly what makes scene-around-motion generation work -- and it is
also the obvious objection to the result: *you placed an obstacle on a trajectory you had
already recorded, then replayed that trajectory.*

Discovery and verification therefore must not share a single deterministic capture. This
runs the four cells again from perturbed initial states -- the robot placed a centimetre or
two off, turned a fraction of a degree -- and asks whether the same 2x2 comes back. A result
that survives is robust to placement; a result that does not was an artifact of one replay,
and it is far better to learn that here than from a reviewer.

The three claim levels this separates:

* **geometrically predicted** -- the swept volumes say a window exists
* **nominally physics verified** -- the unperturbed 2x2 came out as predicted
* **perturbation robust** -- and it still does when the start pose is jittered

Usage::

    python scripts/research/verify_family_robustness.py \\
        --family /data/.../counterfactual/duck_002 \\
        --motions /data/.../taxonomy/motions_4s
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.route_placement import canonical_path_xy  # noqa: E402

PYTHON = Path.home() / "miniconda3/envs/env_isaaclab/bin/python"

#: Start-pose jitters, in metres and radians. Small enough that the task is unchanged --
#: the robot is standing in the same place doing the same thing -- and large enough to break
#: bit-identical replay. A degree is about 0.017 rad.
PERTURBATIONS = (
    ("p1", 0.015, 0.010, math.radians(0.5)),
    ("p2", -0.012, 0.018, math.radians(-0.8)),
    ("p3", 0.008, -0.015, math.radians(1.0)),
)

#: Free GPU memory a rollout needs before it is worth starting, in MiB. PhysX asks for a
#: 256 MiB block up front and several more after; starting below this wastes a cell.
REQUIRED_GPU_MIB = 6000

#: What each cell is supposed to do. The family is the one False.
EXPECTED_ACCEPT = {
    "nominal_easy": True,
    "nominal_hard": False,
    "adapted_easy": True,
    "adapted_hard": True,
}


def convert(csv: Path, out: Path, key: str, start_xy, yaw: float) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(PYTHON), str(REPO_ROOT / "gear_sonic/data_process/convert_kimodo_to_motion_lib.py"),
            "--input", str(csv), "--output", str(out), "--motion-key", key,
            "--source-fps", "30",
            "--scene-start", str(start_xy[0]), str(start_xy[1]), "0.0",
            "--scene-yaw", str(yaw),
        ],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    return result.returncode == 0


class InfrastructureError(RuntimeError):
    """A rollout did not run. This is never a scientific result.

    The first attempt at this verification hit PhysX GPU out-of-memory on a shared card --
    ``PxgCudaDeviceMemoryAllocator failed to allocate 268435456 bytes`` while another job
    held 21 GB -- and every one of the twelve cells would have come back empty. Reported as
    outcomes, that reads as ``perturbation_robust: false``: the family disowned because a
    neighbour was using the GPU. Isaac also exits zero after printing a fatal traceback, so
    the exit code cannot be trusted either. An absent rollout must stop the run.
    """


def free_gpu_mib() -> int:
    """Free memory on the card, or -1 if it cannot be read."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return -1


def wait_for_gpu(required_mib: int, *, timeout_s: float = 3600.0) -> None:
    """Block until the card has room, rather than burning cells against a full GPU."""
    deadline = time.monotonic() + timeout_s
    while True:
        free = free_gpu_mib()
        if free < 0 or free >= required_mib:
            return
        if time.monotonic() > deadline:
            raise InfrastructureError(
                f"only {free} MiB free after waiting {timeout_s / 60:.0f} min; "
                f"need {required_mib} MiB"
            )
        print(f"    waiting for GPU: {free} MiB free, need {required_mib} MiB")
        time.sleep(60)


def rollout(scene: str, motion: Path, out: Path, log: Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
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


def outcome_of(directory: Path, key: str) -> str:
    trajectories = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not trajectories:
        return "no_rollout"
    with trajectories[0].open("rb") as handle:
        return classify_episode(key, pickle.load(handle)).outcome  # noqa: S301


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    args = parser.parse_args()

    family = json.loads((args.family / "family.json").read_text())
    csvs = {
        "nominal": args.motions / family["nominal_motion"],
        "adapted": args.motions / family["adapted_motion"],
    }
    for label, path in csvs.items():
        if not path.exists():
            raise SystemExit(f"missing {label} motion: {path}")

    path_xy = canonical_path_xy(np.loadtxt(csvs["nominal"], delimiter=","))
    base_xy = (float(path_xy[0][0]), float(path_xy[0][1]))
    work = args.family / "robustness"

    table: dict[str, dict[str, str]] = {}
    for name, dx, dy, dyaw in PERTURBATIONS:
        start = (base_xy[0] + dx, base_xy[1] + dy)
        print(f"\n=== {name}: start offset ({dx:+.3f}, {dy:+.3f}) m, yaw {math.degrees(dyaw):+.1f} deg ===")
        table[name] = {}
        for motion_label, csv in csvs.items():
            for difficulty in ("easy", "hard"):
                key = f"{motion_label}_{difficulty}"
                out = work / name / key
                if not (out / "success_manifest.json").exists():
                    motion = work / name / "motions" / f"{key}.pkl"
                    if not convert(csv, motion, f"{name}_{key}", start, dyaw):
                        table[name][key] = "convfail"
                        print(f"  CONVFAIL {key}")
                        continue
                    wait_for_gpu(REQUIRED_GPU_MIB)
                    print(f"  rolling out {key} ...")
                    log = work / name / "logs" / f"{key}.log"
                    if not rollout(
                        f"{family['family_id']}_{difficulty}", motion, out, log
                    ):
                        raise InfrastructureError(
                            f"{name}/{key} did not complete; see {log}"
                        )
                result = outcome_of(out, key)
                if result == "no_rollout":
                    raise InfrastructureError(f"{name}/{key} recorded no trajectory")
                table[name][key] = result
                mark = "ok" if (result == "accepted") == EXPECTED_ACCEPT[key] else "DEPARTS"
                print(f"    {key:16s} {result:12s} {mark}")

    holds = {
        name: all(
            (cells.get(key) == "accepted") == expected
            for key, expected in EXPECTED_ACCEPT.items()
        )
        for name, cells in table.items()
    }
    robust = all(holds.values())

    print("\n" + "=" * 70)
    print(f"{'perturbation':14s}{'nom/easy':>12s}{'nom/hard':>12s}"
          f"{'adp/easy':>12s}{'adp/hard':>12s}{'holds':>8s}")
    for name, cells in table.items():
        print(
            f"{name:14s}"
            + "".join(f"{cells.get(k, '-'):>12s}" for k in
                      ("nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard"))
            + f"{str(holds[name]):>8s}"
        )
    print(f"\nperturbation robust: {robust}")

    report = {
        "family_id": family["family_id"],
        "perturbations": [
            {"name": n, "dx_m": dx, "dy_m": dy, "dyaw_rad": dyaw}
            for n, dx, dy, dyaw in PERTURBATIONS
        ],
        "outcomes": table,
        "holds": holds,
        "perturbation_robust": robust,
        "claim_level": "perturbation robust" if robust else "nominally physics verified",
    }
    destination = args.family / "robustness.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {destination}")
    return 0 if robust else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InfrastructureError as error:
        print(f"\nINFRASTRUCTURE FAILURE: {error}")
        print("No verdict written -- this says nothing about the family.")
        raise SystemExit(2) from error
