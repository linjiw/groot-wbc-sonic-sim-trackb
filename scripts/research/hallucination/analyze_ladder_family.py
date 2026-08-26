#!/usr/bin/env python3
"""Adjudicate a ladder-family 2x2 batch, one row per archetype.

A family completes only on accepted/accepted/rejected/accepted with the hard-cell rejection
attributed to the binding face and no external contact in any intended-clear cell. Outcomes are
copied from the immutable run record; nothing is re-graded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL = {
    ("easy", "nominal"): "accepted",
    ("easy", "adapted"): "accepted",
    ("hard", "nominal"): "rejected",
    ("hard", "adapted"): "accepted",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--endpoint-reference-m", type=float, default=0.3298)
    parser.add_argument("--endpoint-tolerance-mm", type=float, default=20.0)
    args = parser.parse_args()

    record = json.loads(args.run_record.read_text())
    groups: dict[str, dict] = {}
    for cell_id, cell in record["cells"].items():
        scientific = cell.get("scientific") or {}
        if not scientific:
            continue
        # The run record stores the runner's own fields, not the manifest's, so the grouping keys
        # are recovered from the cell id: "<pair>__<archetype>__<difficulty>__<role>".
        parts = cell_id.split("__")
        if len(parts) < 4:
            continue
        archetype, difficulty, role = parts[-3], parts[-2], parts[-1]
        diagnostics = scientific.get("diagnostics") or {}
        contact = diagnostics.get("contact_decomposition") or {}
        groups.setdefault(archetype, {})[(difficulty, role)] = {
            "outcome": scientific["outcome"],
            "reasons": scientific.get("rejection_reasons") or [],
            "endpoint_error_m": diagnostics.get("endpoint_error_m"),
            "external_force_n": contact.get("max_external_contact_force_n") or 0.0,
            "bodies": contact.get("external_contact_bodies") or [],
            "frame": contact.get("max_external_contact_frame"),
        }

    rows = []
    for archetype, cells in sorted(groups.items()):
        pattern_ok = all(
            cells.get(key, {}).get("outcome") == expected for key, expected in CANONICAL.items()
        )
        clear_cells = [key for key in CANONICAL if CANONICAL[key] == "accepted"]
        zero_contact = all(
            cells.get(key, {}).get("external_force_n", 1.0) == 0.0 for key in clear_cells
        )
        hard_nominal = cells.get(("hard", "nominal"), {})
        contact_only = hard_nominal.get("reasons") == ["disallowed_robot_contact"]
        hard_adapted = cells.get(("hard", "adapted"), {})
        endpoint = hard_adapted.get("endpoint_error_m")
        endpoint_stable = (
            endpoint is not None
            and abs(1000 * (endpoint - args.endpoint_reference_m)) <= args.endpoint_tolerance_mm
        )
        rows.append(
            {
                "archetype": archetype,
                "cells": {f"{d}/{r}": v["outcome"] for (d, r), v in sorted(cells.items())},
                "canonical_pattern": pattern_ok,
                "clear_cells_zero_contact": zero_contact,
                "hard_nominal_contact_only": contact_only,
                "hard_nominal_force_n": hard_nominal.get("external_force_n"),
                "hard_nominal_bodies": hard_nominal.get("bodies"),
                "hard_nominal_frame": hard_nominal.get("frame"),
                "hard_adapted_endpoint_m": endpoint,
                "hard_adapted_endpoint_stable": endpoint_stable,
                "family_complete": bool(pattern_ok and zero_contact and contact_only),
            }
        )

    report = {
        "schema_version": "lfh_ladder_family_v1",
        "run_record": str(args.run_record),
        "run_status": record.get("status"),
        "archetypes": len(rows),
        "families_complete": sum(row["family_complete"] for row in rows),
        "predictions": {
            "P1_all_archetypes_canonical": (
                all(row["canonical_pattern"] for row in rows) if rows else None
            ),
            "P2_binding_attribution": (
                all(row["hard_nominal_contact_only"] for row in rows) if rows else None
            ),
            "P3_clear_cells_zero_contact": (
                all(row["clear_cells_zero_contact"] for row in rows) if rows else None
            ),
            "P4_endpoint_stable": (
                all(row["hard_adapted_endpoint_stable"] for row in rows) if rows else None
            ),
        },
        "rows": rows,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"run status: {record.get('status')}; archetypes: {len(rows)}; "
        f"families complete: {report['families_complete']}/{len(rows)}"
    )
    for row in rows:
        print(
            f"  {row['archetype']:16s} {row['cells']}  "
            f"contact {row['hard_nominal_force_n'] or 0.0:.1f} N on {row['hard_nominal_bodies']} "
            f"frame {row['hard_nominal_frame']}  adapted endpoint {row['hard_adapted_endpoint_m']}"
        )
    for name, value in report["predictions"].items():
        print(f"  {name}: {value}")

    if args.md_out:
        lines = [
            "# LFH-E18 — archetype freedom at a fixed critical point",
            "",
            "| archetype | easy/nominal | easy/adapted | hard/nominal | hard/adapted "
            "| binding contact | adapted endpoint |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for row in rows:
            cells = row["cells"]
            lines.append(
                f"| `{row['archetype']}` | {cells.get('easy/nominal','-')} | "
                f"{cells.get('easy/adapted','-')} | {cells.get('hard/nominal','-')} | "
                f"{cells.get('hard/adapted','-')} | "
                f"{row['hard_nominal_force_n'] or 0.0:.1f} N on "
                f"{', '.join(row['hard_nominal_bodies'] or [])} | "
                f"{row['hard_adapted_endpoint_m'] or float('nan'):.4f} m |"
            )
        args.md_out.write_text("\n".join(lines) + "\n")
        print(f"-> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
