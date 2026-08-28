#!/usr/bin/env python3
"""Adjudicate the seed-matched E10b multi-obstacle context-survival pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from scripts.research.hallucination.analyze_e6_crouch_pilot import (  # noqa: E402
    contact_evidence,
    load,
)

EXPECTED = {
    "nominal_easy": "accepted",
    "adapted_easy": "accepted",
    "nominal_hard": "rejected",
    "adapted_hard": "accepted",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run.read_text())
    cpu = json.loads(args.cpu_report.read_text())
    if run["status"] != "completed" or run["manifest_sha256"] != sha256_file(args.manifest):
        raise SystemExit("E10b run is incomplete or does not match the approved manifest")

    outcomes = {}
    contacts = {}
    diagnostics = {}
    videos = {}
    for cell in manifest["cells"]:
        role = cell["cell_role"]
        scientific = run["cells"][cell["cell_id"]]["scientific"]
        outcomes[role] = scientific["outcome"]
        diagnostics[role] = scientific["diagnostics"]
        contacts[role] = contact_evidence(
            load(
                Path(scientific["artifacts"]["trajectory"]),
            ),
            REPO_ROOT / cell["scene"]["path"],
        )
        videos[role] = str(Path(cell["output"]) / "renders/000000.mp4")

    reasons = [
        f"{role}_outcome_mismatch"
        for role, expected in EXPECTED.items()
        if outcomes.get(role) != expected
    ]
    hard = contacts["nominal_hard"]
    if hard["attribution"] != "binding_constraint":
        reasons.append("nominal_hard_not_uniquely_binding")
    if hard["attributed_body"] != "torso_link":
        reasons.append("nominal_hard_binding_anatomy_mismatch")
    if hard["first_frame"] is None or (
        hard["drift_onset_frame"] is not None and hard["first_frame"] >= hard["drift_onset_frame"]
    ):
        reasons.append("nominal_hard_contact_not_before_drift")
    for role, evidence in contacts.items():
        if role != "nominal_hard" and evidence["first_frame"] is not None:
            reasons.append(f"{role}_unexpected_external_contact")
        if evidence["attribution"] in {"secondary_contact", "unattributed"}:
            reasons.append(f"{role}_{evidence['attribution']}")

    verified = not reasons
    report = {
        "schema_version": "lfh_e10_context_v1",
        "verified": verified,
        "refusal_reasons": sorted(set(reasons)),
        "outcomes": outcomes,
        "contacts": contacts,
        "diagnostics": diagnostics,
        "videos": videos,
        "binding_obstacles": 1,
        "context_obstacles": len(cpu["context"]),
        "context_critical_support": False,
        "minimum_cpu_context_clearance_mm": min(cpu["keepout_min_clearance_mm"].values()),
        "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
        "evidence": {
            "manifest_sha256": sha256_file(args.manifest),
            "run_sha256": sha256_file(args.run),
            "cpu_report_sha256": sha256_file(args.cpu_report),
        },
    }
    rows = [
        f"| `{role}` | {outcomes[role]} | {contacts[role]['attribution']} |" for role in EXPECTED
    ]
    markdown = (
        "\n".join(
            [
                "# E10b Multi-Obstacle Context Report",
                "",
                f"The seed-matched five-obstacle scene is **{'verified' if verified else 'refused'}**.",
                "",
                "| role | outcome | external-contact attribution |",
                "|---|---|---|",
                *rows,
                "",
                f"The nominal-hard torso contact occurs at frame {hard['first_frame']} and is uniquely "
                f"attributed to `{hard['attributed_prim_path']}` before drift at frame "
                f"{hard['drift_onset_frame']}. Four context obstacles remain non-causal; the CPU "
                f"minimum context clearance is {report['minimum_cpu_context_clearance_mm']:.2f} mm.",
                "",
                "The lateral, floor-level, and high-overhead faces demonstrate route-relative scene "
                "composition only. They are not promoted into verified critical-axis support.",
            ]
        )
        + "\n"
    )
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(markdown)
    print(f"{'PASS' if verified else 'REFUSED'}: E10b multi-obstacle context scene")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
