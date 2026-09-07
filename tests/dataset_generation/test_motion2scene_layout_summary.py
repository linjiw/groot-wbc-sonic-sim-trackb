import importlib
from pathlib import Path


def block_rows(suite):
    return [
        {
            "cell_id": f"{suite}_{arm}_{seed}",
            "suite": suite,
            "arm": arm,
            "optimizer_seed": seed,
            "pass": arm == "analytic",
            "readout": {"requested_action": int(arm == "analytic"), "refusal": False},
        }
        for seed in range(8501, 8506)
        for arm in ("uniform", "analytic", "no_contrast", "motion2scene")
    ]


def test_partial_wave_keeps_missing_cells_out_of_failure_counts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    module = importlib.import_module("render_motion2scene_layout_results")
    r = module.summarize({"assigned_cells": 600}, [{"rows": block_rows("traversal")}])
    assert r["admitted_measurements"] == 20 and r["pending_or_unadmitted"] == 580
    assert not r["primary_traversal_panel_complete"]
    assert all(x["measured"] == 1 and x["assigned"] == 24 for x in r["per_fit"])
    assert all(
        x["motion2scene_minus_analytic_passes"] == -1 for x in r["paired_optimizer_differences"]
    )


def test_controls_do_not_enter_traversal_effect(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    module = importlib.import_module("render_motion2scene_layout_results")
    r = module.summarize({"assigned_cells": 600}, [{"rows": block_rows("absent")}])
    assert r["traversal_measured"] == 0 and r["control_measured"] == 20
    assert all(
        x["percentage_points_on_completed_blocks"] is None
        for x in r["paired_optimizer_differences"]
    )
