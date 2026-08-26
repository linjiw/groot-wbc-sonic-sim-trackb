#!/usr/bin/env python3
"""Render a verified 2x2 family as one labelled contact sheet.

Each panel is the recorded execution of one cell at its binding frame, captioned with the
authoritative Isaac verdict. The point of the figure is that the two rows differ only in the height
of one plank, and only the nominal's fate changes.

Kinematic ``mj_forward`` replay of recorded states. MuJoCo draws; Isaac decided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio  # noqa: E402

from scripts.research.hallucination.render_critical_frame_figure import (  # noqa: E402
    DEFAULT_ROBOT_XML,
    _label,
    render_frame,
)

ORDER = (("easy", "nominal"), ("easy", "adapted"), ("hard", "nominal"), ("hard", "adapted"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument(
        "--scene-report", type=Path, default=REPO_ROOT / "docs/hallucination/e17_ladder_scenes.json"
    )
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-4.0)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--lookat-z", type=float, default=1.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    record = json.loads(args.run_record.read_text())
    report = json.loads(args.scene_report.read_text())
    entry = (
        next(scene for scene in report["scenes"] if scene["pair_id"] == args.pair_id)
        if args.pair_id
        else report["scenes"][0]
    )
    pair_id = entry["pair_id"]

    rows = []
    for difficulty in ("easy", "hard"):
        panels = []
        coordinate = entry[f"{difficulty}_coordinate_m"]
        for role in ("nominal", "adapted"):
            cell = record["cells"][f"{pair_id}__{difficulty}__{role}"]
            scientific = cell["scientific"]
            trajectory = Path(scientific["artifacts"]["trajectory"])
            contact = (scientific.get("diagnostics") or {}).get("contact_decomposition") or {}
            force = float(contact.get("max_external_contact_force_n") or 0.0)
            frame = (
                int(contact.get("max_external_contact_frame") or 0)
                or entry["critical_frames"][role]
            )
            stage = REPO_ROOT / entry["scenes"][difficulty]
            image = render_frame(
                trajectory,
                stage,
                frame,
                robot_xml=DEFAULT_ROBOT_XML,
                width=args.width,
                height=args.height,
                azimuth=args.azimuth,
                elevation=args.elevation,
                distance=args.distance,
                lookat_z=args.lookat_z,
            )
            verdict = scientific["outcome"].upper()
            reasons = ", ".join(scientific.get("rejection_reasons") or []) or "no external contact"
            panels.append(
                _label(
                    image,
                    [
                        f"{difficulty.upper()} scene, {role}  ->  {verdict}",
                        f"plank underside {coordinate:.4f} m   frame {frame}",
                        f"{reasons}   peak external {force:.1f} N",
                    ],
                )
            )
        rows.append(np.concatenate(panels, axis=1))

    figure = np.concatenate(rows, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
