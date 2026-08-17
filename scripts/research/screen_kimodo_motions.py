#!/usr/bin/env python3
"""Screen generated motions for configurations the G1 cannot reach.

Reports by default and moves nothing. Pass ``--quarantine DIR`` to relocate screened-out
motions, which is the opt-in the validation argues for: a rollout wrongly run costs GPU
minutes, while a motion wrongly screened is a behaviour permanently absent from the corpus.

See docs/motion_prefilter_validation.md for what the signal does and does not predict.

Usage::

    python scripts/research/screen_kimodo_motions.py --csv-dir /data/.../motions \\
        --mjcf ~/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml --json screen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    KimodoQposError,
    load_kimodo_qpos_csv,
)
from gear_sonic.dataset_generation.motion_prefilter import (  # noqa: E402
    DEFAULT_SATURATION_LIMIT,
    load_joint_limits,
    screen_reference_motion,
)

DEFAULT_MJCF = Path.home() / "kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--limit", type=float, default=DEFAULT_SATURATION_LIMIT)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument(
        "--quarantine",
        type=Path,
        help="move screened-out motions here; omit to report only",
    )
    args = parser.parse_args()

    if not args.mjcf.exists():
        raise SystemExit(f"MJCF not found: {args.mjcf}")
    names, limits = load_joint_limits(args.mjcf)

    csvs = sorted(args.csv_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.csv_dir}")

    records, screened = [], []
    for path in csvs:
        try:
            qpos = load_kimodo_qpos_csv(path)
        except KimodoQposError as error:
            print(f"  UNREADABLE {path.name}: {error}")
            continue
        report = screen_reference_motion(qpos, names, limits, saturation_limit=args.limit)
        records.append(
            {
                "csv": path.name,
                "saturated_cell_fraction": report.saturated_cell_fraction,
                "saturated_frame_fraction": report.saturated_frame_fraction,
                "root_height_min_m": report.root_height_min_m,
                "is_crouch": report.is_crouch,
                "frames": report.frames,
                "passed": report.passed,
                "reasons": list(report.reasons),
                "worst_joints": [list(j) for j in report.worst_joints],
            }
        )
        if not report.passed:
            screened.append((path, report))

    records.sort(key=lambda r: -r["saturated_cell_fraction"])
    print(f"{len(csvs)} motion(s); {len(screened)} above saturation {args.limit:.3f}")
    for record in records[:10]:
        flag = "SCREEN" if not record["passed"] else "  ok  "
        worst = record["worst_joints"][0][0] if record["worst_joints"] else "-"
        print(
            f"  {flag} {record['saturated_cell_fraction']:.4f}  "
            f"zmin {record['root_height_min_m']:.3f}  worst {worst:24s} {record['csv'][:44]}"
        )

    if args.quarantine is not None and screened:
        args.quarantine.mkdir(parents=True, exist_ok=True)
        for path, _ in screened:
            shutil.move(str(path), args.quarantine / path.name)
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                shutil.move(str(sidecar), args.quarantine / sidecar.name)
        print(f"moved {len(screened)} motion(s) to {args.quarantine}")
    elif screened:
        print("(reporting only -- pass --quarantine DIR to move them)")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {
                    "saturation_limit": args.limit,
                    "motions": len(records),
                    "screened": len(screened),
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
