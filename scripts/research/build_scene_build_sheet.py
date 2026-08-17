#!/usr/bin/env python3
"""Turn released scene packages into build sheets for physically rebuilding them.

The corpus claims its scenes are replicable by construction. This is the tool that makes
that claim actionable: for each scene it emits a printable floor plan (SVG) and a
tape-measure placement table (Markdown), derived from the released ``.usda`` rather than
from any internal object, so anyone who downloads the package can rebuild the room.

Usage::

    python scripts/research/build_scene_build_sheet.py \\
        --package gear_sonic/data/assets/scenes/g1_clutter3d \\
        --out /data/.../build_sheets [--scene clutter3d_walk_s4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.scene_build_sheet import (  # noqa: E402
    build_sheet_from_scene,
    render_build_sheet_markdown,
    render_build_sheet_svg,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="dir holding manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", action="append", help="scene_id to emit; repeatable")
    args = parser.parse_args()

    manifest_path = args.package / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json in {args.package}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_id = manifest.get("package_id", args.package.name)

    entries = manifest["scenes"]
    if isinstance(entries, dict):
        entries = list(entries.values())
    wanted = set(args.scene) if args.scene else None

    args.out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for entry in entries:
        scene_id = entry.get("scene_id", "unknown")
        if wanted is not None and scene_id not in wanted:
            continue
        usda_path = args.package / entry["file"]
        if not usda_path.exists():
            print(f"  skip {scene_id}: missing {usda_path.name}")
            continue

        sheet = build_sheet_from_scene(
            usda_path.read_text(encoding="utf-8"),
            scene_entry=entry,
            package_id=package_id,
        )
        (args.out / f"{scene_id}.md").write_text(
            render_build_sheet_markdown(sheet), encoding="utf-8"
        )
        (args.out / f"{scene_id}.svg").write_text(render_build_sheet_svg(sheet), encoding="utf-8")

        tightest = sheet.tightest_tolerance_m
        summary.append(
            {
                "scene_id": scene_id,
                "package_id": package_id,
                "room_size_xy_m": list(sheet.room_size_xy_m),
                "pieces": len(sheet.items),
                "floor_pieces": len(sheet.floor_items),
                "raised_pieces": len(sheet.raised_items),
                "over_corridor_pieces": sum(1 for i in sheet.items if i.is_over_corridor),
                "tightest_placement_tolerance_m": tightest,
                "lowest_overhead_m": sheet.lowest_overhead_m,
            }
        )
        detail = f"{len(sheet.floor_items)} floor + {len(sheet.raised_items)} raised"
        tolerance = f"{tightest:.3f} m" if tightest is not None else "n/a"
        print(f"  {scene_id:28s} {detail:22s} tightest tolerance {tolerance}")

    if not summary:
        raise SystemExit("no scenes emitted -- check --scene / package contents")

    (args.out / "build_sheets.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(summary)} build sheet(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
