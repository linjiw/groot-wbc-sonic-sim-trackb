#!/usr/bin/env python3
"""Preflight the repo-owned M0 G1 dataset scene package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.scene_asset_preflight import (  # noqa: E402
    DEFAULT_SCENE_PACKAGE_DIR,
    preflight_scene_package,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SCENE_PACKAGE_DIR,
        help="directory containing manifest.json and self-contained USDA scenes",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="override the manifest path (scene file paths remain relative to package_dir)",
    )
    parser.add_argument("--json", dest="json_path", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)

    report = preflight_scene_package(args.package_dir, manifest_path=args.manifest)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for note in report.notes:
        print(f"NOTE: {note}")
    for error in report.errors:
        print(f"ERROR: package: {error}")
    for scene in report.scenes:
        status = "PASS" if scene.ok else "FAIL"
        clearance = (
            f", route clearance={scene.minimum_route_clearance_m:.3f} m"
            if scene.minimum_route_clearance_m is not None
            else ""
        )
        print(
            f"{status}: {scene.scene_id}: {scene.collision_prim_count} collision prims{clearance}"
        )
        for note in scene.notes:
            print(f"NOTE: {scene.scene_id}: {note}")
        for error in scene.errors:
            print(f"ERROR: {scene.scene_id}: {error}")
    print("PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
