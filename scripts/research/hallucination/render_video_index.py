#!/usr/bin/env python3
"""Contact sheet over a batch of proposal videos, so 94 cases can be reviewed at a glance.

One tile per case, taken at the frame where the nominal is nearest the binding station, sorted by
route misalignment so the straight walks and the sharp turns sit at opposite ends of the sheet.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _tile(video: Path, fraction: float) -> np.ndarray | None:
    reader = imageio.get_reader(video)
    try:
        count = reader.count_frames()
        wanted = int(np.clip(round(fraction * count), 0, max(count - 1, 0)))
        frame = None
        for index, image in enumerate(reader):
            if index == wanted:
                frame = np.asarray(image)
                break
        return frame
    except Exception:  # noqa: BLE001 - a missing tile is a gap, not a failure
        return None
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=0.55)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sort-by", default="face_misalignment_deg")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    videos = json.loads(args.manifest.read_text())["videos"]
    videos.sort(key=lambda row: row.get(args.sort_by, 0.0))
    if args.limit:
        videos = videos[: args.limit]

    tiles = []
    for row in videos:
        frame = _tile(Path(row["video"]), args.fraction)
        if frame is None:
            continue
        if args.scale != 1.0:
            step = max(1, int(round(1.0 / args.scale)))
            frame = frame[::step, ::step]
        tiles.append(frame)
    if not tiles:
        raise SystemExit("no frames could be read")

    height = min(tile.shape[0] for tile in tiles)
    width = min(tile.shape[1] for tile in tiles)
    tiles = [tile[:height, :width] for tile in tiles]
    columns = min(args.columns, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=tiles[0].dtype)
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = tile
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, sheet)
    print(f"wrote {args.out} ({len(tiles)} tiles, {columns}x{rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
