#!/usr/bin/env python3
"""Render scenes the LfLH hallucinator generated, against the motions that produced them.

Unlike the earlier proposal videos -- whose obstacles came from a closed-form solver and a
hand-written inventory -- every box here is sampled from the trained hallucinator's own
distribution. The caption records which candidate motion the fixed decoder selects under that
scene, so a viewer can see whether the generated obstacle actually explains the observed edit.

Obstacles are emitted in the route frame at the station they were sampled for, then placed into
the world using that station's position and executed heading, so a scene follows a curving route.

MuJoCo draws; the differentiable decoder decides; Isaac physics is not involved and no verdict is
implied. These are generated *proposals*, not verified families.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.lflh import (  # noqa: E402
    ChoiceDecoder,
    ObstacleGeometry,
    sample_scenes,
    train,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_arm_tuck,
    local_crouch,
)
from scripts.research.hallucination.build_candidate_sets import (  # noqa: E402
    STATION_FRACTION,
    WINDOW_FRACTION,
)
from scripts.research.hallucination.render_proposal_video import (  # noqa: E402
    render_pair,
)


def world_boxes(parameters: np.ndarray, station_xy: np.ndarray, yaw: np.ndarray) -> list[dict]:
    """Route-frame obstacle parameters -> world boxes, following the executed heading."""
    boxes = []
    for row in parameters:
        station, lateral, height, half_along, half_lateral, half_vertical = row
        index = int(np.clip(round(station), 0, len(yaw) - 1))
        angle = float(yaw[index])
        centre = np.asarray(station_xy[index], dtype=np.float64) + np.asarray(
            (-math.sin(angle), math.cos(angle))
        ) * float(lateral)
        boxes.append(
            {
                "name": f"lflh_{len(boxes)}",
                "center_m": (float(centre[0]), float(centre[1]), float(height)),
                "full_size_m": (
                    2 * float(half_along),
                    2 * float(half_lateral),
                    2 * float(half_vertical),
                ),
                "yaw_rad": angle,
            }
        )
    return boxes


def rebuild_motion(qpos: np.ndarray, label: str) -> np.ndarray:
    """Recreate the adapted clip named by a candidate label."""
    if label == "nominal":
        return qpos
    kind, _, magnitude = label.rpartition("_")
    amount = int(magnitude) / 1000.0
    if kind == "crouch":
        return local_crouch(qpos, STATION_FRACTION, target_drop_m=amount, window=WINDOW_FRACTION)[0]
    side = "left" if "left" in kind else "right"
    return local_arm_tuck(
        qpos, STATION_FRACTION, target_reduction_m=amount, window=WINDOW_FRACTION, side=side
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    parser.add_argument("--target", default="crouch_040")
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--scenes-per-clip", type=int, default=2)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument(
        "--relaxed-extents",
        action="store_true",
        help="widen the per-axis size ranges. The model converges far more easily but emits "
        "boxes no one would call a shelf; use only to demonstrate the mechanism, never as a "
        "scene proposal.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.candidates.read_text())
    sets = [item for item in payload["candidate_sets"] if args.target in item["labels"]][
        : args.clips
    ]
    if not sets:
        raise SystemExit(f"no candidate set offers {args.target}")

    geometry = (
        ObstacleGeometry(
            stations=payload["stations"],
            half_along_range_m=(0.04, 0.80),
            half_lateral_range_m=(0.04, 0.80),
            half_vertical_range_m=(0.04, 0.80),
        )
        if args.relaxed_extents
        else ObstacleGeometry(stations=payload["stations"])
    )
    decoder = ChoiceDecoder()
    extents = [np.asarray(item["extents"], dtype=np.float64) for item in sets]
    costs = [np.asarray(item["costs"], dtype=np.float64) for item in sets]
    observed = [item["labels"].index(args.target) for item in sets]

    print(f"training operator-conditioned hallucinator for {args.target} on {len(sets)} clips ...")
    model, report = train(
        extents,
        costs,
        observed,
        obstacles=args.obstacles,
        steps=args.steps,
        samples=4,
        seed=0,
        kl_weight=0.004,
        geometry=geometry,
        decoder=decoder,
    )
    print(f"  reconstruction {report.reconstruction:.3f}")

    records = []
    for row, item in enumerate(sets):
        qpos = np.loadtxt(item["source_csv"], delimiter=",")
        adapted = rebuild_motion(qpos, args.target)
        batch = sample_scenes(
            model,
            extents[row],
            costs[row],
            observed[row],
            str(item["motion_index"]),
            count=args.scenes_per_clip,
            geometry=geometry,
            decoder=decoder,
            seed=row,
        )
        station_xy = np.asarray(item["station_xy_m"])
        yaw = np.asarray(item["yaw_rad"])
        for draw in range(args.scenes_per_clip):
            boxes = world_boxes(batch.parameters[draw], station_xy, yaw)
            selected = item["labels"][int(batch.winner[draw])]
            binding, context = boxes[0], boxes[1:]
            out_path = args.out_dir / f"lflh_{item['motion_index']:03d}_{args.target}_s{draw}.mp4"
            shared = [
                f"{item['motion_index']:03d} {item.get('body_mode', '')}  "
                f"LfLH-generated scene, draw {draw}",
                f"target {args.target}  decoder selects {selected}  "
                f"{'MATCH' if selected == args.target else 'MISS'}",
                f"{len(boxes)} sampled obstacles",
            ]
            frames = render_pair(
                {"a_nominal": qpos, "b_adapted": adapted},
                binding,
                out_path,
                width=args.width,
                height=args.height,
                fps=30,
                context=context,
                captions={
                    "a_nominal": ["NOMINAL"] + shared,
                    "b_adapted": [f"ADAPTED ({args.target})"] + shared,
                },
            )
            records.append(
                {
                    "motion_index": item["motion_index"],
                    "body_mode": item.get("body_mode", ""),
                    "target": args.target,
                    "decoder_selects": selected,
                    "match": selected == args.target,
                    "obstacles": [
                        {
                            "center_m": box["center_m"],
                            "full_size_m": box["full_size_m"],
                            "yaw_deg": math.degrees(box["yaw_rad"]),
                        }
                        for box in boxes
                    ],
                    "video": str(out_path),
                    "frames": frames,
                }
            )
            print(
                f"  {item['motion_index']:03d} draw {draw}: selects {selected} "
                f"({'match' if selected == args.target else 'MISS'}) -> {out_path.name}",
                flush=True,
            )

    if args.manifest:
        args.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "lflh_generated_scenes_v1",
                    "target": args.target,
                    "clips": len(sets),
                    "reconstruction": report.reconstruction,
                    "match_rate": float(np.mean([r["match"] for r in records])),
                    "contract": (
                        "obstacles sampled from the trained hallucinator; the decoder's selection "
                        "is a differentiable model of the choice rule, not a physics verdict"
                    ),
                    "videos": records,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"\n{len(records)} videos; match rate {np.mean([r['match'] for r in records]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
