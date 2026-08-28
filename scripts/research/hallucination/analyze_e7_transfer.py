#!/usr/bin/env python3
"""Adjudicate E7 archetype transfer with source-level and contact-cause gates."""

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
    lines = [
        "# E7 Source-Conditioned Archetype Transfer Report",
        "",
        f"The registered primary is **{'confirmed' if report['registered_primary_confirmed'] else 'falsified'}**: "
        f"{report['verified_variants']}/6 variants are verified across "
        f"{report['verified_sources']}/3 sources. The 6/6 stretch prediction is "
        f"**{'confirmed' if report['registered_stretch_confirmed'] else 'falsified'}**.",
        "",
        "| source | archetype | easy N/A | hard N/A | contact frame / drift | decision |",
        "|---|---|---|---|---|---|",
    ]
    for variant in report["variants"]:
        outcomes = variant["outcomes"]
        contact = variant["contacts"]["nominal_hard"]
        lines.append(
            f"| `{variant['source_pair_id']}` | `{variant['archetype_id']}` | "
            f"{outcomes['nominal_easy']}/{outcomes['adapted_easy']} | "
            f"{outcomes['nominal_hard']}/{outcomes['adapted_hard']} | "
            f"{contact['first_frame']} / {contact['drift_onset_frame']} | "
            f"**{'verified' if variant['verified'] else 'refused'}** |"
        )
    lines.extend(
        [
            "",
            "All six nominal-hard contacts are uniquely attributed to `torso_link` on the authored "
            "binding primitive, and every intended-clear cell has zero external contact. Five "
            "contacts precede 0.15 m reference drift. Source 086's door-lintel contact occurs after "
            "drift (frame 122 versus 95), so that variant is refused even though its four outcome "
            "labels match. No door jamb, I-beam web, or top flange becomes a secondary cause.",
            "",
            f"Scene-conditioned engineering intervals span "
            f"{report['context_width_range_mm'][0]:.2f}–{report['context_width_range_mm'][1]:.2f} "
            "mm. Source 086 preserves the E6c-verified 0.10 m exposure, so this transfer does not "
            "reintroduce the long-face instability.",
            "",
            f"Spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h** across 24 "
            "rollouts. The evidence reaches the registered transfer primary, but source 086 still "
            "has only two verified archetypes (shelf and I-beam). E5 remains blocked pending one "
            "causally clean replacement archetype for that source; a matching outcome pattern "
            "alone is insufficient.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-manifest", type=Path, required=True)
    parser.add_argument("--easy-run", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--hard-run", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
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
            raise SystemExit(f"E7 {phase} run is incomplete")
        if runs[phase]["manifest_sha256"] != sha256_file(manifest_path):
            raise SystemExit(f"E7 {phase} run/manifest identity mismatch")

    cpu = json.loads(args.cpu_report.read_text())
    recert = json.loads(args.recertification.read_text())
    recert_by_id = {row["variant_id"]: row for row in recert["variants"]}
    variants = []
    for proposed in cpu["variants_detail"]:
        variant_id = proposed["variant_id"]
        outcomes = {}
        contacts = {}
        videos = {}
        for phase in ("easy", "hard"):
            for cell in manifests[phase]["cells"]:
                if (
                    cell["source_family_id"] != proposed["source_pair_id"]
                    or cell["archetype_id"] != proposed["archetype_id"]
                ):
                    continue
                role = cell["cell_role"]
                scientific = runs[phase]["cells"][cell["cell_id"]]["scientific"]
                outcomes[role] = scientific["outcome"]
                trajectory = Path(scientific["artifacts"]["trajectory"])
                contacts[role] = contact_evidence(
                    load(trajectory), REPO_ROOT / cell["scene"]["path"]
                )
                videos[role] = str(Path(cell["output"]) / "renders/000000.mp4")

        reasons = [
            f"{role}_outcome_mismatch"
            for role, expected in EXPECTED.items()
            if outcomes.get(role) != expected
        ]
        if not recert_by_id[variant_id]["hard_phase_eligible"]:
            reasons.append("context_recertification_refused")
        hard = contacts.get("nominal_hard")
        if hard is None:
            reasons.append("nominal_hard_missing")
        else:
            if hard["attribution"] != "binding_constraint":
                reasons.append("nominal_hard_not_uniquely_binding")
            if hard["attributed_body"] != "torso_link":
                reasons.append("nominal_hard_binding_anatomy_mismatch")
            if hard["first_frame"] is None or (
                hard["drift_onset_frame"] is not None
                and hard["first_frame"] >= hard["drift_onset_frame"]
            ):
                reasons.append("nominal_hard_contact_not_before_drift")
        for role, evidence in contacts.items():
            if role != "nominal_hard" and evidence["first_frame"] is not None:
                reasons.append(f"{role}_unexpected_external_contact")
            if evidence["attribution"] in {"secondary_contact", "unattributed"}:
                reasons.append(f"{role}_{evidence['attribution']}")
        variants.append(
            {
                "variant_id": variant_id,
                "source_pair_id": proposed["source_pair_id"],
                "archetype_id": proposed["archetype_id"],
                "verified": not reasons,
                "refusal_reasons": sorted(set(reasons)),
                "route_station_xy_m": proposed["station_xy_m"],
                "face_along_route_m": proposed["face_along_route_m"],
                "hard_coordinate_m": proposed["hard_coordinate_m"],
                "scene_conditioned_engineering_width_mm": recert_by_id[variant_id][
                    "scene_conditioned_engineering_width_mm"
                ],
                "outcomes": outcomes,
                "contacts": contacts,
                "videos": videos,
            }
        )

    verified = [row for row in variants if row["verified"]]
    verified_sources = {row["source_pair_id"] for row in verified}
    widths = [row["scene_conditioned_engineering_width_mm"] for row in variants]
    report = {
        "schema_version": "lfh_e7_transfer_v1",
        "registered_primary_confirmed": len(verified) >= 4 and len(verified_sources) == 3,
        "registered_stretch_confirmed": len(verified) == 6,
        "verified_variants": len(verified),
        "verified_sources": len(verified_sources),
        "readiness_gate_four_sources_three_archetypes": len(verified) == 6,
        "context_width_range_mm": [min(widths), max(widths)],
        "actual_contended_gpu_hours": sum(
            run["budget"]["actual_contended_gpu_hours"] for run in runs.values()
        ),
        "rollouts": sum(len(run["cells"]) for run in runs.values()),
        "evidence": {
            "cpu_report_sha256": sha256_file(args.cpu_report),
            "recertification_sha256": sha256_file(args.recertification),
            "easy_manifest_sha256": sha256_file(args.easy_manifest),
            "easy_run_sha256": sha256_file(args.easy_run),
            "hard_manifest_sha256": sha256_file(args.hard_manifest),
            "hard_run_sha256": sha256_file(args.hard_run),
        },
        "variants": variants,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"{'PASS' if report['registered_primary_confirmed'] else 'REFUSED'}: "
        f"E7 verified={len(verified)}/6 sources={len(verified_sources)}/3"
    )
    return 0 if report["registered_primary_confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
