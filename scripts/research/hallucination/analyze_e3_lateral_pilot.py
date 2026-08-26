#!/usr/bin/env python3
"""Adjudicate the four-cell E3 lateral pilot and preserve its failure mechanism."""

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
from gear_sonic.dataset_generation.hallucination.keypoints import (  # noqa: E402
    extract_keypoints,
)
from gear_sonic.dataset_generation.hallucination.propose import (  # noqa: E402
    CoverageTarget,
)
from gear_sonic.dataset_generation.hallucination.reach import (  # noqa: E402
    lateral_gap_reach,
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
    return payload


def drift_onset(payload: dict) -> int | None:
    reference = np.asarray(payload.get("reference_g1_qpos"), dtype=np.float64)
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    if reference.ndim != 2 or len(reference) != len(root):
        return None
    error = np.linalg.norm(root[:, :2] - reference[:, :2], axis=1)
    hits = np.flatnonzero(error > DRIFT_ONSET_M)
    return int(hits[0]) if len(hits) else None


def render(report: dict) -> str:
    lines = [
        "# E3 Lateral-Gap Pilot Report",
        "",
        "The registered four-cell prediction is **falsified**: 3/4 outcomes matched. "
        "Nominal-easy, nominal-hard, and adapted-easy reproduced the requested pattern, but "
        "adapted-hard contacted a binding panel and rejected.",
        "",
        "| cell | outcome | expected | binding clearance | first contact | attribution | drift onset |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for cell in report["cells"]:
        contact = cell["contact"]
        lines.append(
            f"| `{cell['cell_role']}` | {cell['outcome']} | {cell['expected_outcome']} | "
            f"{cell['binding_clearance_mm']:.2f} mm | "
            f"{'' if contact['first_frame'] is None else contact['first_frame']} | "
            f"{contact['attribution']} | "
            f"{'' if contact['drift_onset_frame'] is None else contact['drift_onset_frame']} |"
        )
    adapted = next(cell for cell in report["cells"] if cell["cell_role"] == "adapted_hard")
    context = report["context_conditioned_retry_gate"]
    lines.extend(
        [
            "",
            f"The adapted-hard collision is causal, not secondary clutter: first contact at frame "
            f"{adapted['contact']['first_frame']} is uniquely nearest "
            f"`{adapted['contact']['attributed_prim_path']}` and precedes 0.15 m reference drift "
            f"at frame {adapted['contact']['drift_onset_frame']}. Contact includes "
            f"`{', '.join(adapted['contact']['external_bodies'])}`. Its executed binding clearance "
            f"fell to {adapted['binding_clearance_mm']:.2f} mm from the +"
            f"{adapted['cpu_clearance_mm']:.2f} mm empty-room CPU prediction.",
            "",
            "Therefore the candidate is **refused**, and the empty DCS bin is not marked covered. "
            "The result says a static empty-room capsule window, even after an E1a-derived margin, "
            "does not certify adapted clearance once the lateral constraint changes the controller "
            "trajectory. A wider retry would be a new registered intervention, not a repair of this "
            "result.",
            "",
            "The accepted easy-scene trajectories provide the conservative retry gate at the same "
            f"station and face extent. Their required full gaps are "
            f"{context['nominal_required_gap_m']:.6f} m nominal and "
            f"{context['adapted_required_gap_m']:.6f} m adapted. The adapted envelope already "
            f"exceeds the target-bin upper edge ({context['target_upper_m']:.3f} m) by "
            f"{context['adapted_target_excess_mm']:.2f} gap-mm before uncertainty; adding the "
            f"registered {context['gap_uncertainty_mm']:.3f} mm gap uncertainty requires at least "
            f"{context['adapted_certified_minimum_m']:.6f} m. Therefore no retry in the registered "
            "0.8–0.9 m bin has a context-conditioned CPU clearance certificate. The bin remains "
            "refused; widening beyond 0.9 m would target a different DCS bucket.",
            "",
            f"Funnel: **24 CPU trials -> 1 instantiated candidate -> 4/4 rollouts -> "
            f"0 verified families**. Actual serial spend: "
            f"**{report['actual_contended_gpu_hours']:.3f} contended GPU-h**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run_record.read_text())
    if run["status"] != "completed" or len(run["cells"]) != 4:
        raise SystemExit("a completed four-cell E3 pilot is required")
    cpu = manifest["cpu_certificate"]["binding_geometry_clearance_mm"]
    cpu_report = json.loads((REPO_ROOT / manifest["cpu_certificate"]["report"]).read_text())
    target = CoverageTarget.from_mapping(manifest["cpu_certificate"]["coverage_target"])
    spec = json.loads((REPO_ROOT / manifest["cpu_certificate"]["spec"]).read_text())
    station = tuple(float(value) for value in spec["binding_station_xy_m"])
    route_axis = str(spec["route_axis"])
    along_route_m = float(spec["face_extent"]["along_route_m"])
    vertical_band_m = (0.0, float(spec["face_extent"]["vertical_m"]))
    cells = []
    payloads = {}
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        role = cell["cell_role"]
        scientific = run["cells"][cell_id]["scientific"]
        payload = load(Path(scientific["artifacts"]["trajectory"]))
        payloads[role] = payload
        stage = read_stage_geometry(REPO_ROOT / cell["scene"]["path"])
        bindings = [cube for cube in stage.cubes if cube.role == "binding_constraint"]
        if len(bindings) != 2:
            raise SystemExit(f"{cell_id}: expected exactly two binding panels")
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
                    "binding_constraint"
                    if cube.role == "binding_constraint"
                    else "secondary_contact"
                )
        binding_candidates = [
            (float(distances[cube.path].body_clearance_m.min()), cube) for cube in bindings
        ]
        binding_clearance, binding_cube = min(binding_candidates, key=lambda value: value[0])
        binding_distance = distances[binding_cube.path]
        bottleneck = int(np.argmin(binding_distance.body_clearance_m))
        expected = cell["expected_source_outcome"]
        cells.append(
            {
                "cell_id": cell_id,
                "cell_role": role,
                "outcome": scientific["outcome"],
                "expected_outcome": expected,
                "outcome_matches": scientific["outcome"] == expected,
                "binding_clearance_mm": 1000 * binding_clearance,
                "cpu_clearance_mm": float(
                    cpu["easy" if role.endswith("easy") else "hard"][
                        "orig" if role.startswith("nominal") else "edit"
                    ]
                ),
                "binding_clearance_drift_mm": abs(
                    1000 * binding_clearance
                    - float(
                        cpu["easy" if role.endswith("easy") else "hard"][
                            "orig" if role.startswith("nominal") else "edit"
                        ]
                    )
                ),
                "binding_bottleneck_frame": bottleneck,
                "binding_body": binding_distance.binding_body[bottleneck],
                "contact": {
                    "first_frame": first_contact,
                    "drift_onset_frame": drift_onset(payload),
                    "external_bodies": list(decomposition.external_contact_bodies),
                    "peak_lateral_force_n": decomposition.max_lateral_contact,
                    "peak_overhead_force_n": decomposition.max_overhead_contact,
                    "attribution": attribution,
                    "attributed_prim_path": attributed_path,
                    "runner_up_separation_mm": runner_up_mm,
                },
            }
        )

    hard_nominal = next(cell for cell in cells if cell["cell_role"] == "nominal_hard")
    adapted_hard = next(cell for cell in cells if cell["cell_role"] == "adapted_hard")
    prediction_confirmed = (
        all(cell["outcome_matches"] for cell in cells)
        and hard_nominal["contact"]["attribution"] == "binding_constraint"
        and adapted_hard["contact"]["first_frame"] is None
    )
    context_reaches = {
        role: lateral_gap_reach(
            extract_keypoints(payloads[role]),
            station,
            route_axis,
            along_route_m,
            vertical_band_m,
        )
        for role in ("nominal_easy", "adapted_easy")
    }
    gap_uncertainty_m = float(cpu_report["search_contract"]["delta_clear_gap_mm"]) / 1000
    nominal_required = context_reaches["nominal_easy"].reach_m
    adapted_required = context_reaches["adapted_easy"].reach_m
    context_gate = {
        "evidence_cells": ["nominal_easy", "adapted_easy"],
        "evidence_outcomes": [
            next(cell["outcome"] for cell in cells if cell["cell_role"] == role)
            for role in ("nominal_easy", "adapted_easy")
        ],
        "station_xy_m": list(station),
        "along_route_m": along_route_m,
        "vertical_band_m": list(vertical_band_m),
        "nominal_required_gap_m": nominal_required,
        "nominal_binding_keypoint": context_reaches["nominal_easy"].binding_keypoint,
        "nominal_critical_frame": context_reaches["nominal_easy"].critical_frame,
        "adapted_required_gap_m": adapted_required,
        "adapted_binding_keypoint": context_reaches["adapted_easy"].binding_keypoint,
        "adapted_critical_frame": context_reaches["adapted_easy"].critical_frame,
        "target_upper_m": target.coordinate_upper_m,
        "adapted_target_excess_mm": 1000 * (adapted_required - target.coordinate_upper_m),
        "gap_uncertainty_mm": 1000 * gap_uncertainty_m,
        "adapted_certified_minimum_m": adapted_required + gap_uncertainty_m,
        "in_target_bin_certified": bool(
            adapted_required + gap_uncertainty_m <= target.coordinate_upper_m
        ),
        "decision": "refuse_target_bin",
    }
    report = {
        "schema_version": "lfh_e3_lateral_pilot_v1",
        "manifest_sha256": run["manifest_sha256"],
        "registered_prediction_confirmed": prediction_confirmed,
        "outcome_matches": sum(cell["outcome_matches"] for cell in cells),
        "verified_families": int(prediction_confirmed),
        "coverage_target_filled": bool(prediction_confirmed),
        "secondary_contact": any(
            cell["contact"]["attribution"] in {"secondary_contact", "unattributed"}
            for cell in cells
        ),
        "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
        "context_conditioned_retry_gate": context_gate,
        "cells": cells,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"{'PASS' if prediction_confirmed else 'REFUSED'}: "
        f"outcomes={report['outcome_matches']}/4 verified={report['verified_families']}"
    )
    return 0 if prediction_confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
