#!/usr/bin/env python3
"""Build a frozen, CPU-only LACE reference-length inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.reference_lengths import (  # noqa: E402
    DEFAULT_MAX_LEN,
    DEFAULT_MOTION_FPS_SCALE,
    DEFAULT_SIM_FPS,
    DEFAULT_TARGET_FPS,
    build_reference_length_inventory,
)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} at {path} must contain a JSON object")
    return payload


def _write_canonical_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    try:
        with path.open(mode, encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing inventory {path}; pass --force explicitly"
        ) from error


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen LACE split JSON")
    parser.add_argument(
        "--motion-root",
        type=Path,
        help="Optional directory containing <motion_key>.pkl; defaults to split robot_path values",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help=f"SONIC target FPS (default: {DEFAULT_TARGET_FPS})",
    )
    parser.add_argument(
        "--sim-fps",
        type=int,
        default=DEFAULT_SIM_FPS,
        help=f"Simulator FPS bound into the runtime contract (default: {DEFAULT_SIM_FPS})",
    )
    parser.add_argument(
        "--motion-fps-scale",
        type=float,
        default=DEFAULT_MOTION_FPS_SCALE,
        help="Runtime motion FPS scale; frozen inventories require exactly 1",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Motion crop limit; frozen inventories require -1 (disabled)",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("scientific", "pilot"),
        default="scientific",
    )
    parser.add_argument(
        "--selected-motion-key",
        action="append",
        dest="selected_motion_keys",
        help="Explicit pilot motion key; repeat for each member of the strict D_atlas subset",
    )
    parser.add_argument("--output", type=Path, required=True, help="New canonical inventory JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow replacement of an existing output file",
    )
    args = parser.parse_args(argv)

    split = _load_json_object(args.split, "split")
    inventory = build_reference_length_inventory(
        split,
        target_fps=args.target_fps,
        sim_fps=args.sim_fps,
        motion_fps_scale=args.motion_fps_scale,
        max_len=args.max_len,
        artifact_mode=args.artifact_mode,
        selected_motion_keys=args.selected_motion_keys,
        motion_root=args.motion_root,
    )
    _write_canonical_json(args.output, inventory, force=args.force)
    print(
        json.dumps(
            {
                "artifact_mode": inventory["artifact_mode"],
                "file_sha256": _sha256_file(args.output),
                "inventory_sha256": inventory["inventory_sha256"],
                "motion_count": inventory["motion_count"],
                "output": str(args.output),
                "target_fps": inventory["target_fps"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
