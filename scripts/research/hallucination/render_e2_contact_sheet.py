#!/usr/bin/env python3
"""Render preregistered E2 ego-view frames around hard/nominal contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

OFFSETS = (-60, -30, 0)
HEADER_PX = 34


def read_frame(video: Path, frame: int) -> Image.Image:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, pixels = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame} from {video}")
    return Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    variants = sorted(report["variant_results"], key=lambda row: row["archetype_id"])
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    panels: list[list[Image.Image]] = []

    for variant in variants:
        hard = next(cell for cell in variant["cells"] if cell["cell_role"] == "nominal_hard")
        contact_frame = hard["contact"]["first_frame"]
        if contact_frame is None:
            raise RuntimeError(f"{variant['archetype_id']} has no hard/nominal contact frame")
        video = Path(cells[hard["cell_id"]]["output"]) / "renders/000000.mp4"
        verdict = "verified" if variant["verified"] else "refused: pre-contact drift"
        row = []
        for offset in OFFSETS:
            frame = contact_frame + offset
            image = read_frame(video, frame)
            panel = Image.new("RGB", (image.width, image.height + HEADER_PX), "#111827")
            panel.paste(image, (0, HEADER_PX))
            label = f"{variant['archetype_id']} | {verdict} | f{frame} ({offset:+d})"
            ImageDraw.Draw(panel).text((10, 6), label, fill="white", font=font)
            row.append(panel)
        panels.append(row)

    width = sum(panel.width for panel in panels[0])
    height = sum(row[0].height for row in panels)
    sheet = Image.new("RGB", (width, height), "black")
    y = 0
    for row in panels:
        x = 0
        for panel in row:
            sheet.paste(panel, (x, y))
            x += panel.width
        y += row[0].height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(f"wrote {args.output} ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
