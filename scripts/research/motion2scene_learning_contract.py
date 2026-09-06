#!/usr/bin/env python3
"""Freeze the current generator and a prospective common learner/test-layout contract."""

import json
from pathlib import Path

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

DATA = ROOT.parent / "research-data/groot-wbc"


def main():
    out = DATA / "m2s-learning-contract-v1"
    parent_path = DATA / "m2s-refinement-v1/registration.json"
    parent = json.loads(parent_path.read_text())
    cell_ref = parent["cells"][0]
    cell = json.loads(checked(Path(cell_ref["path"]), cell_ref["sha256"]).read_text())
    assert cell["seed"] == 8421
    checked(Path(cell["checkpoint"]["path"]), cell["checkpoint"]["sha256"])
    names = [
        "scripts/research/motion2scene_event_scaling.py",
        "scripts/research/motion2scene_distill_study.py",
        "scripts/research/motion2scene_source_execution.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_events.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_station_search.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_analytic_global.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_analytic_distinct.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_coverage.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_action_contract.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_outcome_learner.py",
        "gear_sonic/dataset_generation/hallucination/motion2scene_policy_features.py",
        "docs/motion2scene/LEARNING_UTILITY_PLAN_V1.md",
    ]
    refs = [artifact(ROOT / n) for n in names]
    refs += [artifact(Path(__file__)), artifact(parent_path), cell_ref, cell["checkpoint"]]
    out.mkdir(parents=True, exist_ok=False)
    layouts = [
        {"id": f"layout_{i:02d}", "station": station, "underside_m": h, "suite": "traversal"}
        for i, (station, h) in enumerate(
            (s, h) for s in (0.35, 0.45, 0.55, 0.65) for h in (1.18, 1.27, 1.36)
        )
    ]
    write_new(
        out / "registration.json",
        {
            "role": (
                "prospective generator/learner freeze and independent layout reservation; "
                "no fitting or physics authorized by this file"
            ),
            "references": refs,
            "generator": {
                "checkpoint": cell["checkpoint"],
                "fit": "all8_8421/1200",
                "search": "pattern17",
                "refitting": False,
            },
            "learner": {
                "features": 214,
                "hidden": [64, 32],
                "heads": 2,
                "activation": "relu",
                "loss": "masked_binary_cross_entropy",
                "normalization": "train_only",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "steps": 1000,
                "seeds": [8501, 8502, 8503, 8504, 8505],
                "checkpoint": "fixed_endpoint",
            },
            "primary_comparison": "equal_complete_labeled_encounters_Motion2Scene_vs_strong_analytic",
            "secondary_comparison": "equal_total_acquisition_budget",
            "development_bank": {
                "ancestor": 41002,
                "skills": ["neutral", "d040"],
                "decision_time_s": 0.2,
                "qualification_pending": "m2s-action-label-completion-v1",
            },
            "independent_common_bank_layouts": layouts,
            "layout_physics_seeds": [8511, 8512],
            "training_exclusion": {
                "rule": (
                    "exclude if abs(station-test_station)<=0.01 AND abs(height-test_height)<=0.005 "
                    "for any reserved traversal layout; charge all rejected proposals equally"
                ),
                "scope": "common-bank layout evaluation; not fresh-source evidence",
            },
            "background_suites": ["absent", "raised", "blocked"],
            "prohibited_fitting_sources": list(range(43001, 43009)),
            "fresh_source_panel": "not yet acquired/reserved; requires distinct before-generation manifest",
            "action_label_requirement": (
                "common recorded pre-decision state/action/token history; "
                "exact deployed command; four binary outcomes; missing labels masked"
            ),
            "acquisition_manifest_pending": True,
            "training_eligible": False,
            "robot_data_fits": 0,
            "physics_runs": 0,
        },
    )
    print(
        json.dumps(
            {
                "generator": cell["checkpoint"],
                "reserved_layouts": len(layouts),
                "robot_data_fits": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
