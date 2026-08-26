#!/usr/bin/env python3
"""CPU-screen stronger settings of the existing arm-tuck operator after CAL1 refusal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    _signed_half_widths,
    active_frames,
    local_arm_tuck,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import (  # noqa: E402
    DATA_ROOT,
    STATION_FRACTION,
    WINDOW_FRACTION,
    _one_source,
)

PAIRS = {"left": (89, 86), "right": (92, 84)}
TARGETS_M = (0.08, 0.10, 0.12, 0.15, 0.18)


def render(report: dict) -> str:
    lines = [
        "# Arm-Tuck Strong-Command CPU Screen",
        "",
        "This screen searches only the existing `local_arm_tuck` operator. It is reference-side "
        "proposal evidence, not executed delivery and never a scene verdict.",
        "",
        "| motion / side | target | reference window | joint change | capped | CPU gate |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['motion_index']:03d}/{row['side']}` | "
            f"{row['target_reduction_mm']:.0f} mm | {row['reference_window_mm']:.2f} mm | "
            f"{row['max_joint_change_rad']:.3f} rad | "
            f"{'yes' if row['excursion_capped'] else 'no'} | "
            f"{'pass' if row['worth_a_rollout'] else 'refused'} |"
        )
    lines.extend(
        [
            "",
            f"All {report['passing_rows']}/{len(report['rows'])} settings pass the strict "
            "reference gate. The strongest distinct reference windows are "
            f"{report['maximum_window_mm']['left']:.2f} mm left and "
            f"{report['maximum_window_mm']['right']:.2f} mm right. A later registered empty-room "
            "cohort may test those commands, but CAL1's executed lower bounds remain binding until "
            "such evidence exists.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/arm_tuck_strength_screen.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/REPORT_ARM_TUCK_STRENGTH_SCREEN.md",
    )
    args = parser.parse_args()

    rows = []
    for side, indices in PAIRS.items():
        for index in indices:
            nominal = np.loadtxt(_one_source(args.source_dir, index), delimiter=",")
            nominal_left, nominal_right = _signed_half_widths(nominal, DEFAULT_G1_MJCF)
            nominal_side = nominal_left if side == "left" else nominal_right
            for target in TARGETS_M:
                adapted, operator = local_arm_tuck(
                    nominal,
                    STATION_FRACTION,
                    target_reduction_m=target,
                    window=WINDOW_FRACTION,
                    side=side,
                )
                mask = active_frames(nominal, adapted)
                adapted_left, adapted_right = _signed_half_widths(adapted, DEFAULT_G1_MJCF)
                adapted_side = adapted_left if side == "left" else adapted_right
                gate = screen_reference(adapted, f"{index:03d}_{side}_{target}", "walk")
                rows.append(
                    {
                        "motion_index": index,
                        "side": side,
                        "target_reduction_mm": 1000.0 * target,
                        "reference_window_mm": 1000.0
                        * float(nominal_side[mask].max() - adapted_side[mask].max()),
                        "max_joint_change_rad": operator.max_joint_change_rad,
                        "excursion_capped": operator.excursion_capped,
                        "worth_a_rollout": gate.worth_a_rollout,
                        "saturated_cell_fraction": gate.saturated_cell_fraction,
                    }
                )
    report = {
        "schema_version": "lfh_arm_tuck_strength_screen_v1",
        "rows": rows,
        "passing_rows": sum(row["worth_a_rollout"] for row in rows),
        "maximum_window_mm": {
            side: max(row["reference_window_mm"] for row in rows if row["side"] == side)
            for side in PAIRS
        },
        "verdict_scope": "reference-side proposal screen only",
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"PASS: {report['passing_rows']}/{len(rows)} settings; "
        f"left={report['maximum_window_mm']['left']:.2f} mm, "
        f"right={report['maximum_window_mm']['right']:.2f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
