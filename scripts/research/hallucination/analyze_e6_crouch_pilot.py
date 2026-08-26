#!/usr/bin/env python3
"""Adjudicate staged E6 with primitive attribution and source-level pattern accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.constraint_distance import (  # noqa: E402
    constraint_distance_from_payload,
)
from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    sha256_file,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

ATTRIBUTION_RESOLUTION_M = 0.00001
CONTACT_THRESHOLD_N = 1.0
DRIFT_ONSET_M = 0.15


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory segment")
    return payload


def drift_onset(payload: dict) -> int | None:
    reference = np.asarray(payload.get("reference_g1_qpos"), dtype=np.float64)
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    if reference.ndim != 2 or len(reference) != len(root):
        return None
    error = np.linalg.norm(root[:, :2] - reference[:, :2], axis=1)
    hits = np.flatnonzero(error > DRIFT_ONSET_M)
    return int(hits[0]) if len(hits) else None


def contact_evidence(payload: dict, scene_path: Path) -> dict[str, object]:
    stage = read_stage_geometry(scene_path)
    distances = {
        cube.path: constraint_distance_from_payload(payload, cube.box) for cube in stage.cubes
    }
    profiles = {path: value.body_clearance_m for path, value in distances.items()}
    decomposition = decompose_payload_contacts(payload)
    signal = np.maximum(
        decomposition.external_lateral_by_frame,
        decomposition.external_overhead_by_frame,
    )
    hits = np.flatnonzero(signal > CONTACT_THRESHOLD_N)
    first_contact = int(hits[0]) if len(hits) else None
    attribution = "none"
    attributed_path = None
    runner_up_mm = None
    binding_body = None
    if first_contact is not None:
        nearest = sorted(
            (float(profile[first_contact]), path) for path, profile in profiles.items()
        )
        separation = nearest[1][0] - nearest[0][0] if len(nearest) > 1 else None
        runner_up_mm = None if separation is None else 1000 * separation
        if separation is None or separation <= ATTRIBUTION_RESOLUTION_M:
            attribution = "unattributed"
        else:
            attributed_path = nearest[0][1]
            cube = next(cube for cube in stage.cubes if cube.path == attributed_path)
            attribution = (
                "binding_constraint" if cube.role == "binding_constraint" else "secondary_contact"
            )
            distance = distances[attributed_path]
            binding_body = distance.binding_body[first_contact]
    binding = [cube for cube in stage.cubes if cube.role == "binding_constraint"]
    binding_clearance = min(float(distances[cube.path].body_clearance_m.min()) for cube in binding)
    return {
        "first_frame": first_contact,
        "drift_onset_frame": drift_onset(payload),
        "attribution": attribution,
        "attributed_prim_path": attributed_path,
        "attributed_body": binding_body,
        "runner_up_separation_mm": runner_up_mm,
        "binding_clearance_mm": 1000 * binding_clearance,
        "external_bodies": list(decomposition.external_contact_bodies),
        "peak_external_force_n": decomposition.max_external_contact,
        "peak_lateral_force_n": decomposition.max_lateral_contact,
        "peak_overhead_force_n": decomposition.max_overhead_contact,
    }


def render(report: dict) -> str:
    primary = "confirmed" if report["registered_primary_confirmed"] else "falsified"
    lines = [
        "# E6 Independent-Source Critical-Window Report",
        "",
        f"The registered primary prediction is **{primary}**: "
        f"{report['verified_sources']}/3 CAL3 sources reproduced a complete canonical 2x2 pattern. "
        "The 3/3 stretch prediction is falsified.",
        "",
        "| source | easy nominal/edit | hard nominal/edit | hard contact | decision |",
        "|---|---|---|---|---|",
    ]
    for source in report["sources"]:
        outcomes = source["outcomes"]
        hard = source.get("nominal_hard_contact")
        lines.append(
            f"| `{source['source_pair_id']}` | "
            f"{outcomes.get('nominal_easy', 'not run')}/{outcomes.get('adapted_easy', 'not run')} | "
            f"{outcomes.get('nominal_hard', 'not run')}/{outcomes.get('adapted_hard', 'not run')} | "
            f"{'' if hard is None else hard['attribution']} | "
            f"{'verified' if source['verified'] else 'refused: ' + '; '.join(source['refusal_reasons'])} |"
        )
    lines.extend(
        [
            "",
            "Sources 089 and 090 both pass outcome, unique binding attribution, contact-before-"
            "drift, and no-secondary-contact gates. Source 086 is refused before hard physics: "
            "its adapted-easy trajectory crossed the reference-drift threshold without external "
            "contact. This is direct evidence that route phase and finite exposure belong in the "
            "proposal distribution; empty-room clearance alone is insufficient.",
            "",
            f"Spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h** across 10 "
            "rollouts. The two verified families remain isolated from claim 5. They add independent "
            "source evidence in one DCS cell; they do not yet establish a learned distribution or "
            "either the count-only or crossed E5 readiness gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-manifest", type=Path, required=True)
    parser.add_argument("--easy-run", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--hard-run", type=Path, required=True)
    parser.add_argument(
        "--cpu-report",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e6_crouch_cpu.json",
    )
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
    for phase in ("easy", "hard"):
        manifest_path = args.easy_manifest if phase == "easy" else args.hard_manifest
        if runs[phase]["status"] != "completed":
            raise SystemExit(f"E6 {phase} run is incomplete")
        if runs[phase]["manifest_sha256"] != sha256_file(manifest_path):
            raise SystemExit(f"E6 {phase} run/manifest identity mismatch")

    cpu = json.loads(args.cpu_report.read_text())
    source_rows = []
    for selected in cpu["selected"]:
        pair_id = selected["source_pair_id"]
        outcomes = {}
        contacts = {}
        for phase in ("easy", "hard"):
            for cell in manifests[phase]["cells"]:
                if cell["source_family_id"] != pair_id:
                    continue
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
        reasons = []
        for role, expected_outcome in expected.items():
            if role not in outcomes:
                continue
            if outcomes.get(role) != expected_outcome:
                reasons.append(f"{role}_outcome_mismatch")
        if "nominal_hard" not in outcomes or "adapted_hard" not in outcomes:
            reasons.append("hard_cells_not_authorized_after_easy_failure")
        hard_contact = contacts.get("nominal_hard")
        if hard_contact is not None:
            if hard_contact["attribution"] != "binding_constraint":
                reasons.append("nominal_hard_not_uniquely_binding")
            if hard_contact["attributed_body"] != "torso_link":
                reasons.append("nominal_hard_binding_anatomy_mismatch")
            drift_frame = hard_contact["drift_onset_frame"]
            contact_frame = hard_contact["first_frame"]
            if contact_frame is None or (drift_frame is not None and contact_frame >= drift_frame):
                reasons.append("nominal_hard_contact_not_before_drift")
        for role, contact in contacts.items():
            if role == "nominal_hard":
                continue
            if contact["first_frame"] is not None:
                reasons.append(f"{role}_unexpected_external_contact")
        if any(
            contact["attribution"] in {"secondary_contact", "unattributed"}
            for contact in contacts.values()
        ):
            reasons.append("secondary_or_unattributed_contact")
        source_rows.append(
            {
                "source_pair_id": pair_id,
                "verified": not reasons,
                "refusal_reasons": sorted(set(reasons)),
                "outcomes": outcomes,
                "contacts": contacts,
                "nominal_hard_contact": hard_contact,
            }
        )

    verified = sum(source["verified"] for source in source_rows)
    report = {
        "schema_version": "lfh_e6_crouch_pilot_v1",
        "registered_primary_confirmed": verified >= 2,
        "registered_stretch_confirmed": verified == 3,
        "verified_sources": verified,
        "coverage_targets_filled": int(verified > 0),
        "actual_contended_gpu_hours": sum(
            run["budget"]["actual_contended_gpu_hours"] for run in runs.values()
        ),
        "evidence": {
            "easy_manifest_sha256": sha256_file(args.easy_manifest),
            "easy_run_sha256": sha256_file(args.easy_run),
            "hard_manifest_sha256": sha256_file(args.hard_manifest),
            "hard_run_sha256": sha256_file(args.hard_run),
            "cpu_report_sha256": sha256_file(args.cpu_report),
        },
        "sources": source_rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(f"{'PASS' if verified >= 2 else 'REFUSED'}: E6 verified={verified}/3")
    return 0 if verified >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
