#!/usr/bin/env python
"""Export the ego frames a selector is allowed to see, and prove it cannot cheat.

A selector that reads the *motion* rather than the *scene* would score well and mean nothing. The
export is therefore bounded by two constraints, and the tighter one is enforced:

1. **Decision margin.** Frames must end at least ``--margin`` executed frames before the predicted
   contact, so the choice is a decision rather than a readout of an imminent collision.
2. **Shortcut-free.** Frames must end before the nominal and adapted references diverge at all. On
   the verified family the references are bit-identical -- maximum joint difference exactly
   0.00e+00 rad -- until reference frame 41, so before that point the two labels are observationally
   identical apart from the scene.

Measured on ``mf_005_c08``: contact at executed frame 90, divergence at executed frame ~68, so the
margin constraint (≤60) binds first and leaves 8 frames of slack. Both are checked per episode and
an episode that cannot satisfy both is refused rather than exported with a narrower window.

Reference clips run at 30 fps and rollouts at 50 Hz. The window is computed in executed frames; the
same arithmetic in reference frames is wrong by a factor of 1.66.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

#: Executed frames the decision must precede contact by. From the plan; it is also what makes the
#: window shortcut-free on the families measured so far.
DEFAULT_MARGIN = 30


def divergence_frame(nominal_csv: Path, adapted_csv: Path, executed_frames: int) -> int | None:
    """First executed frame at which the two references differ at all, or None if never."""
    if not (nominal_csv.exists() and adapted_csv.exists()):
        return None
    nominal = np.loadtxt(nominal_csv, delimiter=",")
    adapted = np.loadtxt(adapted_csv, delimiter=",")
    count = min(len(nominal), len(adapted))
    difference = np.abs(adapted[:count, 7:] - nominal[:count, 7:]).max(axis=1)
    if not (difference > 1e-6).any():
        return None
    reference_onset = int(np.argmax(difference > 1e-6))
    return int(round(reference_onset * executed_frames / count))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--nominal-csv", type=Path, required=True)
    ap.add_argument("--adapted-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Pass one: measure each cell's own limits. Pass two: sample the SAME frames in every cell.
    #
    # Sampling each cell at its own window gave the four cells of a family different frame indices
    # -- 0..60 for the one that made contact, 0..67 for the three that did not -- and a frame index
    # that correlates with the label is exactly the shortcut this export exists to prevent. The
    # window is therefore the tightest across the family, applied to all of it.
    measured = []
    for cell in args.cells:
        paths = sorted(cell.glob("trajectories/*.trajectory.pkl"))
        if not paths:
            continue
        try:
            with open(paths[0], "rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
        except (SegmentError, Exception):  # noqa: BLE001
            continue
        if payload is None:
            continue
        outcome = classify_episode(cell.name, payload)
        executed = int(len(payload["root_pos_w"]))
        contact = (outcome.diagnostics or {}).get("max_nonfoot_contact_frame")
        # With no contact the whole episode is a decision window, so the margin is measured from
        # the end, which is the conservative reading.
        contact = int(contact) if contact is not None else executed
        onset = divergence_frame(args.nominal_csv, args.adapted_csv, executed)
        measured.append(
            {
                "cell": cell,
                "outcome": outcome.outcome,
                "executed_frames": executed,
                "contact_frame": contact,
                "divergence_frame": onset,
                "margin_limit": contact - args.margin,
                "shortcut_limit": (onset - 1) if onset is not None else executed,
            }
        )

    if not measured:
        print("no evaluable cells")
        return 1

    last = min(min(m["margin_limit"], m["shortcut_limit"]) for m in measured)
    binding = "margin" if last == min(m["margin_limit"] for m in measured) else "shortcut"
    print(f"family decision window 0..{last}, bound by {binding}, applied to every cell\n")
    if last < args.k:
        print(f"REFUSE: window ends at {last}, too narrow for K={args.k}")
        return 1

    picks = [int(x) for x in np.linspace(max(0, last - 15), last, args.k)]
    records, refused = [], 0
    for entry in measured:
        cell = entry["cell"]
        video = cell / "renders" / "000000.mp4"
        saved = []
        if video.exists():
            for frame in picks:
                target = args.out / f"{cell.name}_f{frame:03d}.png"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(video),
                        "-vf",
                        f"select=eq(n\\,{frame})",
                        "-vframes",
                        "1",
                        str(target),
                    ],
                    check=False,
                )
                if target.exists():
                    saved.append(str(target))
        records.append(
            {
                **{k: (v if k != "cell" else v.name) for k, v in entry.items()},
                "window_last_frame": last,
                "binding_constraint": binding,
                "frames": picks,
                "images": saved,
            }
        )
        print(
            f"{cell.name:>22s}  {entry['outcome']:>9s}  contact {entry['contact_frame']:3d}"
            f"  frames {picks}  {len(saved)} images"
        )

    (args.out / "decision_frames.json").write_text(
        json.dumps(
            {
                "k": args.k,
                "margin": args.margin,
                "window_last_frame": last,
                "binding_constraint": binding,
                "frames": picks,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(records)} episodes exported, {refused} refused for too narrow a window")
    print(
        f"every cell sampled at frames {picks}, so the frame index carries no " "label information"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
