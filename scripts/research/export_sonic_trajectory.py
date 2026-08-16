#!/usr/bin/env python3
"""Export an aligned SONIC/PhysX trajectory and Isaac RGB video to LeRobot."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic.dataset_generation.trajectory_export import export_sonic_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="schema-v2 trajectory recorder pickle")
    parser.add_argument("ego_video", type=Path, help="aligned 640x480, 50 Hz Isaac Sim RGB MP4")
    parser.add_argument("output", type=Path, help="output LeRobot dataset directory")
    parser.add_argument("--task", required=True, help="natural-language task annotation")
    parser.add_argument(
        "--camera-provenance",
        required=True,
        choices=("isaac_sim",),
        help="explicitly assert that frames came from the Isaac simulation render",
    )
    parser.add_argument(
        "--runtime-success-manifest",
        required=True,
        type=Path,
        help="schema-v2 Isaac success manifest binding the trajectory and video hashes",
    )
    parser.add_argument("--scene-id", required=True, help="scene registry identifier")
    parser.add_argument("--scene-hash", required=True, help="scene sha256 content hash")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="first source frame to export (inclusive; default: 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="last source frame boundary (exclusive; default: source length)",
    )
    parser.add_argument(
        "--upstream-provenance-json",
        type=Path,
        required=True,
        help="typed ConversionResult JSON in a request/generation/conversion bundle",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="replace an existing output dataset",
    )
    args = parser.parse_args()

    frame_count = export_sonic_trajectory(
        trajectory_path=args.trajectory,
        ego_video_path=args.ego_video,
        output_path=args.output,
        task=args.task,
        camera_provenance=args.camera_provenance,
        runtime_manifest_path=args.runtime_success_manifest,
        scene_id=args.scene_id,
        scene_hash=args.scene_hash,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        upstream_provenance_path=args.upstream_provenance_json,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"SONIC_LEROBOT_EXPORT_SUCCESS output={args.output.resolve()} frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
