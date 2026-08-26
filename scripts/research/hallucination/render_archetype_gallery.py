#!/usr/bin/env python3
"""Render one critical point realised by several archetypes, as a single gallery.

The two-stage factorisation claims the binding coordinate is determined by the executed pair while
the object realising it is free. This figure is that claim drawn: every panel places a different
obstacle at the *same* face coordinate, inferred from the same motion pair, with the nominal shown
at its binding frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.hallucination.render_critical_frame_figure import (  # noqa: E402
    DEFAULT_ROBOT_XML,
    _label,
    render_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-report", type=Path, action="append", required=True)
    parser.add_argument("--role", choices=("nominal", "adapted"), default="nominal")
    parser.add_argument("--cell", choices=("hard", "easy"), default="hard")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-5.0)
    parser.add_argument("--distance", type=float, default=2.6)
    parser.add_argument("--lookat-z", type=float, default=1.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panels = []
    for path in args.scene_report:
        entry = json.loads(Path(path).read_text())["scenes"][0]
        coordinate = entry[f"{args.cell}_coordinate_m"]
        spec = json.loads((REPO_ROOT / entry["spec"]).read_text())
        reach = spec["binding"]["reach_orig_m" if args.role == "nominal" else "reach_edit_m"]
        image = render_frame(
            Path(entry["reference_motions"][args.role]),
            REPO_ROOT / entry["scenes"][args.cell],
            entry["critical_frames"][args.role],
            robot_xml=DEFAULT_ROBOT_XML,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat_z=args.lookat_z,
        )
        gap = 1000 * (coordinate - reach)
        panels.append(
            _label(
                image,
                [
                    f"{entry['archetype']}   underside {coordinate:.4f} m",
                    f"{args.role} reach {reach:.4f} m  ->  "
                    f"{'strikes' if gap < 0 else 'clears'} by {abs(gap):.1f} mm",
                ],
            )
        )

    columns = max(1, args.columns)
    rows = []
    for start in range(0, len(panels), columns):
        row = panels[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.concatenate(row, axis=1))
    imageio.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"wrote {args.out} ({len(panels)} archetypes at one critical point)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
