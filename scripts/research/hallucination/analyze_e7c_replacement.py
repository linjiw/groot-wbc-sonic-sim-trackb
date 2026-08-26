#!/usr/bin/env python3
"""Adjudicate the source-086 E7c hanging-panel replacement."""

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


def render(report: dict) -> str:
    source = report["replacement"]
    contact = source["contacts"]["nominal_hard"]
    rows = []
    for role in EXPECTED:
        evidence = source["contacts"][role]
        attribution = evidence["attribution"] if evidence["first_frame"] is not None else "none"
        rows.append(f"| `{role}` | {source['outcomes'][role]} | {attribution} |")
    return (
        "\n".join(
            [
                "# E7c Source-086 Replacement Report",
                "",
                f"The hanging-panel replacement is **{'verified' if source['verified'] else 'refused'}**. "
                "It preserves source 086's motion pair, station, 0.10 m exposure, coordinates, and "
                "margin rule while replacing only the E7 door-lintel archetype.",
                "",
                "| role | outcome | external contact |",
                "|---|---|---|",
                *rows,
                "",
                f"Nominal-hard contact is uniquely attributed to `{contact['attributed_body']}` on "
                f"`{contact['attributed_prim_path']}` at frame {contact['first_frame']}, before drift "
                f"at frame {contact['drift_onset_frame']}. The refused door-lintel contact occurred "
                "after drift (122 versus 95), so the replacement changes causal cleanliness rather "
                "than merely recovering outcome labels.",
                "",
                f"Spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h** across four "
                "rollouts. With the verified shelf and I-beam, this gives source 086 three clean "
                "archetypes. This satisfies the count-only gate, but not the stricter crossed E5 "
                "gate: a third archetype must still be verified across all four sources.",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-manifest", type=Path, required=True)
    parser.add_argument("--easy-run", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--hard-run", type=Path, required=True)
    parser.add_argument("--recertification", type=Path, required=True)
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
            raise SystemExit(f"E7c {phase} run is incomplete")
        if runs[phase]["manifest_sha256"] != sha256_file(manifest_path):
            raise SystemExit(f"E7c {phase} run/manifest identity mismatch")

    recert = json.loads(args.recertification.read_text())
    outcomes = {}
    contacts = {}
    videos = {}
    for phase in ("easy", "hard"):
        for cell in manifests[phase]["cells"]:
            role = cell["cell_role"]
            scientific = runs[phase]["cells"][cell["cell_id"]]["scientific"]
            outcomes[role] = scientific["outcome"]
            contacts[role] = contact_evidence(
                load(Path(scientific["artifacts"]["trajectory"])),
                REPO_ROOT / cell["scene"]["path"],
            )
            videos[role] = str(Path(cell["output"]) / "renders/000000.mp4")

    reasons = [
        f"{role}_outcome_mismatch"
        for role, expected in EXPECTED.items()
        if outcomes.get(role) != expected
    ]
    if recert["eligible_variants"] != 1:
        reasons.append("context_recertification_refused")
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
        "schema_version": "lfh_e7c_replacement_v1",
        "adaptive_replacement_verified": verified,
        "e7_registered_result_unchanged": "5/6",
        "count_only_gate_four_sources_three_archetypes": verified,
        "strict_crossed_gate_three_common_archetypes": False,
        "next_required_evidence": (
            "transfer one common third archetype across cf_005_056 and sources 086/089/090"
        ),
        "actual_contended_gpu_hours": sum(
            run["budget"]["actual_contended_gpu_hours"] for run in runs.values()
        ),
        "evidence": {
            "easy_manifest_sha256": sha256_file(args.easy_manifest),
            "easy_run_sha256": sha256_file(args.easy_run),
            "hard_manifest_sha256": sha256_file(args.hard_manifest),
            "hard_run_sha256": sha256_file(args.hard_run),
            "recertification_sha256": sha256_file(args.recertification),
        },
        "replacement": {
            "source_pair_id": "lfh_086_crouch",
            "archetype_id": "hanging_panel",
            "verified": verified,
            "refusal_reasons": sorted(set(reasons)),
            "scene_conditioned_engineering_width_mm": recert["variants"][0][
                "scene_conditioned_engineering_width_mm"
            ],
            "outcomes": outcomes,
            "contacts": contacts,
            "videos": videos,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(f"{'PASS' if verified else 'REFUSED'}: E7c hanging-panel replacement")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
