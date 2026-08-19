"""Cut a render down to the moment it is about.

A four-second traversal spends most of its frames walking toward the thing that matters. On a page
the approach is context and the crossing is the argument, so clips are trimmed to the window around
the obstacle rather than shown whole. The window is given in frames because every other number in
this project is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg  # noqa: F401  (ensures the encoder is present)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    frames = []
    for index in range(args.start, args.end):
        try:
            frames.append(iio.imread(args.src, index=index))
        except (IndexError, Exception):  # noqa: BLE001 -- a short clip simply ends
            break
    if not frames:
        raise SystemExit(f"no frames in {args.start}..{args.end} of {args.src}")
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264")
    size = args.out.stat().st_size / 1e6
    print(f"{args.src.parent.name}: {len(frames)} frames -> {args.out} ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
