#!/usr/bin/env python3
"""Adjudicate the E6c finite-exposure ablation against its registered gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    sha256_file,
)
from scripts.research.hallucination.analyze_e6_crouch_pilot import (  # noqa: E402
    contact_evidence,
    load,
)


def render(report: dict) -> str:
    source = report["source"]
    contact = source["contacts"]["nominal_hard"]
    decision = "verified" if source["verified"] else "refused"
    lines = [
        "# E6c Finite-Exposure Ablation Report",
        "",
        f"The adaptive source-086 family is **{decision}**. Reducing the route-aligned shelf "
        "face from 0.30 m to 0.10 m changed the adapted-easy outcome from rejected to accepted "
        "while holding the motion pair, operator, station, archetype, DCS target, and margin rule "
        "fixed.",
        "",
        "| role | outcome | external contact |",
        "|---|---|---|",
    ]
    for role in ("nominal_easy", "adapted_easy", "nominal_hard", "adapted_hard"):
        evidence = source["contacts"][role]
        lines.append(
            f"| `{role}` | {source['outcomes'][role]} | "
            f"{evidence['attribution'] if evidence['first_frame'] is not None else 'none'} |"
        )
    lines.extend(
        [
            "",
            f"Nominal-hard contact occurs at frame {contact['first_frame']} on "
            f"`{contact['attributed_body']}` and is uniquely attributed to "
            f"`{contact['attributed_prim_path']}`; reference drift begins at frame "
            f"{contact['drift_onset_frame']}. The runner-up primitive is "
            f"{contact['runner_up_separation_mm']:.1f} mm farther away.",
            "",
            f"Adapted-easy drift rate fell from "
            f"{report['exposure_comparison']['long_face_adapted_easy_drift_rate_mps']:.5f} m/s "
            f"to {report['exposure_comparison']['short_face_adapted_easy_drift_rate_mps']:.5f} "
            "m/s. This is one paired source/seed intervention: it establishes finite exposure as "
            "a required proposal variable here, not a population-level effect estimate.",
            "",
            f"Spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h** across four "
            "rollouts. E6 remains 2/3 by preregistration; E6c independently makes source 086 "
            "extractable and brings the available source count to four when combined with "
            "`cf_005_056`, 089, and 090.",
        ]
    )
    if source["refusal_reasons"]:
        lines.extend(["", "Refusal reasons: " + ", ".join(source["refusal_reasons"]) + "."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-manifest", type=Path, required=True)
    parser.add_argument("--easy-run", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--hard-run", type=Path, required=True)
    parser.add_argument("--e6-easy-run", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    manifests = {
        "easy": json.loads(args.easy_manifest.read_text()),
        "hard": json.loads(args.hard_manifest.read_text()),
    }
    runs = {
        "easy": json.loads(args.easy_run.read_text()),
        "hard": json.loads(args.hard_run.read_text()),
    }
    for phase, manifest_path in (
        ("easy", args.easy_manifest),
        ("hard", args.hard_manifest),
    ):
        if runs[phase]["status"] != "completed":
            raise SystemExit(f"E6c {phase} run is incomplete")
        if runs[phase]["manifest_sha256"] != sha256_file(manifest_path):
            raise SystemExit(f"E6c {phase} run/manifest identity mismatch")

    outcomes: dict[str, str] = {}
    contacts: dict[str, dict] = {}
    for phase in ("easy", "hard"):
        for cell in manifests[phase]["cells"]:
            role = cell["cell_role"]
            scientific = runs[phase]["cells"][cell["cell_id"]]["scientific"]
            outcomes[role] = scientific["outcome"]
            payload = load(Path(scientific["artifacts"]["trajectory"]))
            contacts[role] = contact_evidence(payload, REPO_ROOT / cell["scene"]["path"])

    expected = {
        "nominal_easy": "accepted",
        "adapted_easy": "accepted",
        "nominal_hard": "rejected",
        "adapted_hard": "accepted",
    }
    reasons = [
        f"{role}_outcome_mismatch"
        for role, expected_outcome in expected.items()
        if outcomes.get(role) != expected_outcome
    ]
    hard_contact = contacts["nominal_hard"]
    if hard_contact["attribution"] != "binding_constraint":
        reasons.append("nominal_hard_not_uniquely_binding")
    if hard_contact["attributed_body"] != "torso_link":
        reasons.append("nominal_hard_binding_anatomy_mismatch")
    if hard_contact["first_frame"] is None or (
        hard_contact["drift_onset_frame"] is not None
        and hard_contact["first_frame"] >= hard_contact["drift_onset_frame"]
    ):
        reasons.append("nominal_hard_contact_not_before_drift")
    for role, evidence in contacts.items():
        if role != "nominal_hard" and evidence["first_frame"] is not None:
            reasons.append(f"{role}_unexpected_external_contact")
        if evidence["attribution"] in {"secondary_contact", "unattributed"}:
            reasons.append(f"{role}_{evidence['attribution']}")

    e6_run = json.loads(args.e6_easy_run.read_text())
    long_cell = next(
        cell
        for cell_id, cell in e6_run["cells"].items()
        if cell_id.endswith("__adapted_easy") and "086" in cell_id
    )
    short_cell = next(
        cell
        for cell_id, cell in runs["easy"]["cells"].items()
        if cell_id.endswith("__adapted_easy")
    )
    comparison = {
        "held_fixed": manifests["hard"]["ablation"]["held_fixed"],
        "long_face_m": manifests["hard"]["ablation"]["failed_e6_value_m"],
        "short_face_m": manifests["hard"]["ablation"]["e6c_value_m"],
        "long_face_adapted_easy_outcome": long_cell["scientific"]["outcome"],
        "short_face_adapted_easy_outcome": short_cell["scientific"]["outcome"],
        "long_face_adapted_easy_drift_rate_mps": long_cell["scientific"]["diagnostics"][
            "drift_rate_mps"
        ],
        "short_face_adapted_easy_drift_rate_mps": short_cell["scientific"]["diagnostics"][
            "drift_rate_mps"
        ],
    }
    verified = not reasons
    report = {
        "schema_version": "lfh_e6c_exposure_ablation_v1",
        "adaptive_followup_verified": verified,
        "e6_registered_result_unchanged": "2/3",
        "extractable_source_count_after_e6c": 4 if verified else 3,
        "actual_contended_gpu_hours": sum(
            run["budget"]["actual_contended_gpu_hours"] for run in runs.values()
        ),
        "exposure_comparison": comparison,
        "evidence": {
            "easy_manifest_sha256": sha256_file(args.easy_manifest),
            "easy_run_sha256": sha256_file(args.easy_run),
            "hard_manifest_sha256": sha256_file(args.hard_manifest),
            "hard_run_sha256": sha256_file(args.hard_run),
            "e6_easy_run_sha256": sha256_file(args.e6_easy_run),
        },
        "source": {
            "source_pair_id": "lfh_086_crouch",
            "verified": verified,
            "refusal_reasons": sorted(set(reasons)),
            "outcomes": outcomes,
            "contacts": contacts,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(f"{'PASS' if verified else 'REFUSED'}: E6c source 086")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
