#!/usr/bin/env python3
"""Build approval-ready, non-authorizing Phase-2 GPU request manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
SCENE_ROOT = REPO_ROOT / "gear_sonic/data/assets/scenes"
CONTENDED_SECONDS = 375


def _motion(path: Path, *, provenance: Path | None = None) -> dict[str, object]:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        start = payload.get("scene_start_xyz")
        manifest_path: Path | None = manifest
    elif provenance is not None and provenance.exists():
        start = [0.0, 0.0, 0.0]
        manifest_path = provenance
    else:
        raise ValueError(f"{path}: no conversion provenance")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "conversion_provenance": str(manifest_path),
        "conversion_provenance_sha256": sha256_file(manifest_path),
        "scene_start_xyz": start,
    }


def _scene(scene_id: str) -> dict[str, object]:
    matches = sorted(SCENE_ROOT.glob(f"*/{scene_id}.usda"))
    if len(matches) != 1:
        raise ValueError(f"{scene_id}: expected one scene file, found {len(matches)}")
    path = matches[0]
    return {
        "scene_id": scene_id,
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
    }


def _plane() -> dict[str, object]:
    return {"scene_id": "plane", "path": None, "sha256": None}


def _policy(rollouts: int) -> dict[str, object]:
    return {
        "driver": "scripts/research/run_kimodo_sonic_rollout.sh",
        "not_authorized": True,
        "requires_explicit_user_approval": True,
        "default_not_before": "2026-08-25T00:00:00-04:00",
        "claim5_inputs_untouched": True,
        "prediction_register_untouched": True,
        "camera_passes": ["ego"],
        "forbidden_camera_passes": ["overhead"],
        "runtime": {
            "free_gpu_mib_required": 9000,
            "hang_timeout_seconds": 1800,
            "success_requires": [
                "SONIC_EVAL_SUCCESS marker",
                "success_manifest.json",
                "at least one trajectory pickle",
                "recorded fps=50",
            ],
        },
        "cost_ceiling": {
            "rollouts": rollouts,
            "seconds_per_rollout_contended": CONTENDED_SECONDS,
            "gpu_hours_contended": round(rollouts * CONTENDED_SECONDS / 3600, 3),
            "peak_gpu_memory_gib_per_serial_rollout": 4.8,
        },
        "preflight": [
            "verify every scene, motion, and provenance hash",
            "require motion scene_start_xyz to match the cell's expected start exactly",
            "run only one rollout at a time after the 9000 MiB free-memory gate",
            "classify missing marker, timeout, OOM, or missing artifact as infrastructure_failure",
            "classify a completed physics rejection as a scientific outcome, never infrastructure",
        ],
        "postflight": [
            "derive verdict labels only through the existing physics scorer",
            "report capsule-proxy disagreement without changing the physics verdict",
            "use a 0.00001 m attribution-resolution floor and refuse ambiguous contact as unattributed",
            "refuse any contact attributed to non-binding authored geometry as secondary_contact",
        ],
    }


def _write(
    path: Path,
    experiment: str,
    purpose: str,
    cells: list[dict],
    stops: list[str],
    *,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "schema_version": "lfh_gpu_request_v1",
        "experiment": experiment,
        "purpose": purpose,
        "execution_policy": _policy(len(cells)),
        "stop_conditions": stops,
        "cells": cells,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o664)


def _e1a(out: Path) -> None:
    duck = DATA_ROOT / "counterfactual/duck_003"
    matched = DATA_ROOT / "matched/mf_005_c08"
    families = (
        (
            "duck_003",
            duck,
            {
                "nominal_easy": ("cf_005_056_easy", "nominal_easy.pkl", "accepted"),
                "nominal_hard": ("cf_005_056_hard", "nominal_hard.pkl", "rejected"),
                "adapted_easy": ("cf_005_056_easy", "adapted_easy.pkl", "accepted"),
                "adapted_hard": ("cf_005_056_hard", "adapted_hard.pkl", "accepted"),
            },
        ),
        (
            "mf_005_c08",
            matched,
            {
                "nominal_easy": ("mf_005_c08_easy", "w_nominal_easy.pkl", "accepted"),
                "nominal_hard": ("mf_005_c08_hard", "w_nominal_hard.pkl", "rejected"),
                "adapted_easy": ("mf_005_c08_easy", "w_crouch08_easy.pkl", "accepted"),
                "adapted_hard": ("mf_005_c08_hard", "w_crouch08_hard.pkl", "accepted"),
            },
        ),
    )
    cells = []
    for repeat, seed in ((1, 31001), (2, 31002)):
        for family_id, root, roles in families:
            for role, (scene_id, motion_name, expected) in roles.items():
                cells.append(
                    {
                        "cell_id": f"{family_id}__{role}__repeat_{repeat}",
                        "family_id": family_id,
                        "cell_role": role,
                        "repeat": repeat,
                        "runtime_seed": seed,
                        "hydra_overrides": [f"++seed={seed}"],
                        "scene": _scene(scene_id),
                        "motion": _motion(root / "motions" / motion_name),
                        "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                        "expected_source_outcome": expected,
                        "output": str(
                            DATA_ROOT
                            / "hallucination/e1a_repeatability"
                            / family_id
                            / f"repeat_{repeat}"
                            / role
                        ),
                    }
                )
    _write(
        out / "E1A_REPEATABILITY_PROPOSED.json",
        "E1a",
        "Measure a two-repeat per-cell clearance noise floor for both historical sources.",
        cells,
        [
            "stop the complete LFH program immediately on any source outcome flip",
            "stop the batch on an infrastructure failure; diagnose and request approval before retry",
            "do not reinterpret a completed rejection as an infrastructure retry",
        ],
    )


def _e1b(out: Path) -> None:
    root = DATA_ROOT / "counterfactual/duck_003"
    roles = {
        "nominal_easy": ("easy", "nominal_easy.pkl", "accepted"),
        "nominal_hard": ("hard", "nominal_hard.pkl", "rejected"),
        "adapted_easy": ("easy", "adapted_easy.pkl", "accepted"),
        "adapted_hard": ("hard", "adapted_hard.pkl", "accepted"),
    }
    scene_ids = {cell: f"cs_duck_003__shelf_plank__s00000017__{cell}" for cell in ("easy", "hard")}
    cells = []
    for role, (difficulty, motion_name, expected) in roles.items():
        cells.append(
            {
                "cell_id": f"cs_duck_003__shelf_plank__{role}",
                "source_family_id": "cf_005_056",
                "source_variant_id": "duck_003",
                "cell_role": role,
                "runtime_seed": 31101,
                "hydra_overrides": ["++seed=31101"],
                "scene": _scene(scene_ids[difficulty]),
                "motion": _motion(root / "motions" / motion_name),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expected_source_outcome": expected,
                "output": str(DATA_ROOT / "hallucination/e1b_duck003_shelf" / role),
            }
        )
    _write(
        out / "E1B_DUCK003_PHYSICS_PROPOSED_V2.json",
        "E1b-physics-duck_003",
        "Complete the physics half of the CPU-green shelf_plank golden regression.",
        cells,
        [
            "do not start until E1a completes with zero source outcome flips",
            "stop on infrastructure failure and preserve partial outputs",
            "refuse the golden if any 2x2 outcome differs from the shipped source pattern",
            "refuse hard/nominal unless contact is uniquely attributed to the binding primitive",
            "judge clearance drift only against the completed E1a noise floor",
        ],
        extra={
            "supersedes": "E1B_DUCK003_PHYSICS_PROPOSED.json",
            "supersession_reason": (
                "D2-005: the reviewed 1.0811 m face reversed the adapted-hard capsule sign; "
                "v2 preserves the source 0.5 m finite footprint and passed the four-sign CPU gate"
            ),
            "cpu_gate": "docs/hallucination/e1_cpu/summary.json",
        },
    )


def _probes(out: Path) -> None:
    candidate_manifest = DATA_ROOT / "lfh_probe_candidates/candidates.json"
    candidates = json.loads(candidate_manifest.read_text())["candidates"]
    matched = DATA_ROOT / "matched/mf_005_c08"
    cells = [
        {
            "cell_id": "mf_005_c08__probe_nominal",
            "pair_id": "mf_005_c08",
            "pair_role": "nominal",
            "depends_on_acceptance_of": None,
            "scene": _scene("screen_empty"),
            "motion": _motion(matched / "motions/w_nominal.pkl"),
            "scene_start_xyz_expected": [0.0, 0.0, 0.0],
            "expectation": {
                "status": "predicted",
                "outcome": "accepted",
                "basis": "historical nonempty-scene executions tracked this nominal motion",
            },
            "verdict_policy": "reference_trackability",
            "output": str(matched / "probe_nominal_gradeable"),
        },
        {
            "cell_id": "mf_005_c08__probe_adapted",
            "pair_id": "mf_005_c08",
            "pair_role": "adapted",
            "depends_on_acceptance_of": "mf_005_c08__probe_nominal",
            "scene": _scene("screen_empty"),
            "motion": _motion(matched / "motions/w_crouch08.pkl"),
            "scene_start_xyz_expected": [0.0, 0.0, 0.0],
            "expectation": {
                "status": "measurement_no_prediction",
                "outcome": None,
                "basis": (
                    "no plane execution exists and crouch transport-cost history makes an "
                    "endpoint-gate failure scientifically plausible"
                ),
            },
            "verdict_policy": "reference_trackability",
            "output": str(matched / "probe_adapted_gradeable"),
        },
    ]
    for candidate in candidates:
        artifacts = candidate["artifacts"]
        pair = f"lfh_{candidate['motion_index']:03d}_arm_tuck_{candidate['side']}"
        for role in ("nominal", "adapted"):
            cells.append(
                {
                    "cell_id": f"{pair}__probe_{role}",
                    "pair_id": pair,
                    "pair_role": role,
                    "coverage_target": "arms/local_arm_tuck/lateral_gap",
                    "operator": candidate["operator"],
                    "response_alpha": 0.0 if role == "nominal" else 1.0,
                    "cpu_reference_window_m": candidate["reference_window_m"],
                    "predicted_delivered_window_m": candidate["predicted_delivered_window_m"],
                    "depends_on_acceptance_of": (
                        None if role == "nominal" else f"{pair}__probe_nominal"
                    ),
                    "scene": _scene("screen_empty"),
                    "motion": _motion(
                        Path(artifacts[f"{role}_motion"]), provenance=candidate_manifest
                    ),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expectation": {
                        "status": "measurement_no_prediction",
                        "outcome": None,
                        "basis": (
                            "no accepted empty-room execution exists for this arm-tuck pair; "
                            "nonempty-scene successes are prioritization evidence only"
                        ),
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(DATA_ROOT / "hallucination/gradeable_empty_probes" / pair / role),
                }
            )
    _write(
        out / "MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V5.json",
        "LFH-empty-room-probes-1",
        "Close the mf source pair and unlock top-ranked arm-tuck targets in a gradeable empty room.",
        cells,
        [
            "skip an adapted cell when its paired nominal is rejected in physics",
            "record any completed rejection as a trackability refusal and continue with other pairs",
            "stop the batch on infrastructure failure; do not spend retry budget without approval",
            "do not use nonempty-scene successes as substitutes for any refused plane cell",
        ],
        extra={
            "supersedes": "MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V4.json",
            "supersession_reason": (
                "V5 pins the existing raw locomotion acceptance report as the trackability "
                "verdict, keeping reference path and endpoint gates binding. V4 named the scene "
                "correction but did not make this scorer choice explicit in each cell"
            ),
            "verdict_contract": {
                "policy": "reference_trackability",
                "scorer": "evaluate_locomotion_trajectory after best_evaluable_payload",
                "reference_path_and_endpoint_gates_bind": True,
                "completed_rejection_is_scientific": True,
                "missing_support_evidence_is_infrastructure": True,
                "zero_external_collision_required_for_acceptance": True,
            },
            "response_contract": {
                "artifact": "keypoint_response.csv",
                "columns": [
                    "motion_id",
                    "operator",
                    "α",
                    "keypoint",
                    "axis",
                    "commanded_mm",
                    "executed_mm",
                ],
                "alpha_definition": (
                    "dimensionless interpolation from the paired nominal command (0) to the "
                    "manifest-pinned edited command (1); legacy crouch08 remains blocked until "
                    "its metadata repair defines the operator amplitude"
                ),
                "append_rule": (
                    "append only after both paired plane rollouts are accepted and commanded "
                    "and executed per-keypoint displacements are evidence-backed"
                ),
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/hallucination/manifests")
    parser.add_argument("--only", choices=("e1a", "e1b", "probes"), action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out.chmod(0o775)
    selected = set(args.only or ("e1a", "e1b", "probes"))
    if "e1a" in selected:
        _e1a(args.out)
    if "e1b" in selected:
        _e1b(args.out)
    if "probes" in selected:
        _probes(args.out)
    print(f"wrote {len(selected)} proposed, non-authorizing manifest(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
