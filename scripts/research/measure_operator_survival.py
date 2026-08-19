"""How much of a commanded adaptation actually appears in the executed body?

A clip labelled ``crouch18`` asserts that the robot crouched. What the dataset can honestly claim
is weaker: that a crouch was *commanded*. The frozen controller decides how much of it happens, and
a label describing motion the robot largely did not perform is a mislabel however cleanly the
physics gates pass.

The measurement is a ratio, per clip, computed only over the joints the operator actually moves and
only over the frames where it is active:

    survival = mean |executed_adapted - executed_nominal| / mean |reference_adapted - reference_nominal|

Both sides come from ``reference_g1_qpos`` and ``dof_pos``, which the rollout records on the same
frame grid, so nothing is resampled. Operator joints are discovered from the reference difference
rather than hardcoded, so the same code reads a crouch, a tuck or a combination.

Two rollouts of the same journey drift apart on their own, and that drift inflates the numerator.
The control column measures it directly on the joints the operator never touches -- a survival
figure is only worth reading when the operator column stands well clear of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np

from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

#: A joint counts as the operator's if the reference moves it by more than this, in radians.
OPERATOR_RAD = 0.05

#: Frames count as active where the operator's reference departure exceeds this share of its peak.
ACTIVE_SHARE = 0.5


def read(cell: Path):
    """Reference and executed joint angles on the same frame grid, or None."""
    found = sorted(cell.glob("trajectories/*.trajectory.pkl"))
    if not found:
        return None
    try:
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except (Exception,):  # noqa: BLE001 -- an unreadable cell is reported, not fatal
        return None
    if payload is None or "reference_g1_qpos" not in payload or "dof_pos" not in payload:
        return None
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, 7:]
    executed = np.asarray(payload["dof_pos"], dtype=np.float64)
    return reference, executed, list(payload.get("dof_joint_names", []))


def survival(adapted: Path, nominal: Path) -> dict | None:
    """Commanded and executed amplitude of one adaptation, with its own noise floor."""
    left, right = read(adapted), read(nominal)
    if left is None or right is None:
        return None
    (ref_a, exe_a, names), (ref_n, exe_n, _) = left, right
    count = min(len(ref_a), len(ref_n))
    d_ref = ref_a[:count] - ref_n[:count]
    d_exe = exe_a[:count] - exe_n[:count]

    amplitude = np.abs(d_ref).max(axis=0)
    operator = np.flatnonzero(amplitude > OPERATOR_RAD)
    control = np.flatnonzero(amplitude < 1e-6)
    if not len(operator):
        return None
    peak = np.abs(d_ref[:, operator]).max()
    active = np.abs(d_ref[:, operator]).max(axis=1) > ACTIVE_SHARE * peak
    if int(active.sum()) < 3:
        return None

    commanded = float(np.abs(d_ref[np.ix_(active, operator)]).mean())
    executed = float(np.abs(d_exe[np.ix_(active, operator)]).mean())
    drift = float(np.abs(d_exe[np.ix_(active, control)]).mean()) if len(control) else float("nan")
    return {
        "clip": adapted.name,
        "commanded_rad": commanded,
        "executed_rad": executed,
        "survival": executed / commanded if commanded > 1e-6 else float("nan"),
        "drift_rad": drift,
        "margin": executed / drift if drift > 1e-9 else float("inf"),
        "joints": int(len(operator)),
        "active_frames": int(active.sum()),
        "operator_joints": [names[j] for j in operator] if names else [],
    }


def nominal_for(cell: Path) -> Path | None:
    """The unadapted counterpart, by the layout conventions in use across the corpora."""
    name = cell.name
    if name.startswith("adapted_"):
        candidate = cell.parent / f"nominal_{name[len('adapted_'):]}"
        return candidate if candidate.is_dir() else None
    candidate = cell.parent / f"{name.split('_')[0]}_nominal"
    if candidate.is_dir() and candidate != cell:
        return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="directories of rollout cells")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for cell in sorted(p for p in root.iterdir() if p.is_dir()):
            nominal = nominal_for(cell)
            if nominal is None or nominal == cell:
                continue
            result = survival(cell, nominal)
            if result is not None:
                result["corpus"] = root.name
                rows.append(result)

    if not rows:
        print("no adapted/nominal pairs found")
        return 1

    print(f"{'clip':>20s} {'cmd':>7s} {'exec':>7s} {'survival':>9s} {'drift':>7s} {'margin':>7s}")
    for row in rows:
        print(
            f"{row['clip']:>20s} {row['commanded_rad']:7.3f} {row['executed_rad']:7.3f} "
            f"{100 * row['survival']:8.0f}% {row['drift_rad']:7.3f} {row['margin']:6.1f}x"
        )

    weak = [r for r in rows if r["margin"] < 3.0]
    if weak:
        print(f"\n{len(weak)} clip(s) within 3x of the drift floor; read their survival with care:")
        for row in weak:
            print(f"  {row['clip']} ({row['margin']:.1f}x)")

    if args.csv:
        import csv

        fields = [
            "corpus",
            "clip",
            "commanded_rad",
            "executed_rad",
            "survival",
            "drift_rad",
            "margin",
            "joints",
            "active_frames",
        ]
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
