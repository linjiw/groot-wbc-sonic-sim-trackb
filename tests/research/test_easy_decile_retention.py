"""Easy-decile retention metric tests (research_plan_zpd_teacher.md §3.4.3, D8).

The metric is plumbing for the preregistered retention non-inferiority gate:
mean per-motion metrics over the easiest decile of a FROZEN SIM-D1 difficulty
ranking, computed from the eval callback's metrics_eval.json per-motion table.
"""

from __future__ import annotations

import json

import pytest

from scripts.research.summarize_sonic_logs import (
    compute_easy_decile_metrics,
    load_difficulty_ranking,
    summarize_logs,
)


def _metrics_eval(motion_keys, mpjpe_g, terminated):
    return {
        "eval/all_metrics_dict": {
            "motion_keys": motion_keys,
            "mpjpe_g": mpjpe_g,
            "terminated": terminated,
        }
    }


_RANKING = [f"m{i:02d}" for i in range(20)]  # m00 easiest ... m19 hardest


def test_easy_decile_selects_easiest_fraction_of_frozen_ranking() -> None:
    # 20-motion ranking -> decile = 2 motions (m00, m01).
    metrics = _metrics_eval(
        ["m01", "m19", "m00", "m10"],
        [1.0, 9.0, 3.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
    )
    record = compute_easy_decile_metrics(metrics, _RANKING)
    assert record["decile_size"] == 2
    assert record["evaluated_in_decile"] == 2
    assert record["motion_keys"] == ["m00", "m01"]
    assert record["mpjpe_g"] == pytest.approx(2.0)  # mean of 1.0 and 3.0
    assert record["success_rate"] == pytest.approx(1.0)
    assert record["ok"] is True


def test_easy_decile_reports_thin_coverage_instead_of_failing() -> None:
    # Eval covered only one of the two easiest motions.
    metrics = _metrics_eval(["m01", "m15"], [2.0, 8.0], [0.0, 1.0])
    record = compute_easy_decile_metrics(metrics, _RANKING)
    assert record["decile_size"] == 2
    assert record["evaluated_in_decile"] == 1
    assert record["mpjpe_g"] == pytest.approx(2.0)


def test_easy_decile_empty_overlap_is_not_ok() -> None:
    metrics = _metrics_eval(["m18", "m19"], [7.0, 9.0], [1.0, 1.0])
    record = compute_easy_decile_metrics(metrics, _RANKING)
    assert record["evaluated_in_decile"] == 0
    assert record["ok"] is False
    assert "mpjpe_g" not in record


def test_easy_decile_uses_ranking_not_metric_order() -> None:
    # A sampler that degrades easy motions must be caught even when the eval
    # table happens to list hard motions first.
    metrics = _metrics_eval(
        ["m19", "m18", "m00", "m01"],
        [0.5, 0.6, 50.0, 60.0],  # easy motions REGRESSED
        [0.0, 0.0, 1.0, 1.0],
    )
    record = compute_easy_decile_metrics(metrics, _RANKING)
    assert record["mpjpe_g"] == pytest.approx(55.0)
    assert record["success_rate"] == pytest.approx(0.0)


def test_load_difficulty_ranking_accepts_list_and_artifact_object(tmp_path) -> None:
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(_RANKING))
    assert load_difficulty_ranking(bare) == _RANKING

    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"dataset": "d_b_speed", "ranking": _RANKING}))
    assert load_difficulty_ranking(artifact) == _RANKING


def test_load_difficulty_ranking_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(["a", "b", "a"]))
    with pytest.raises(ValueError, match="duplicate"):
        load_difficulty_ranking(path)


def test_summarize_logs_attaches_easy_decile_under_eval(tmp_path) -> None:
    metrics_path = tmp_path / "metrics_eval.json"
    metrics_path.write_text(
        json.dumps(_metrics_eval(["m00", "m01", "m19"], [1.0, 2.0, 9.0], [0.0, 0.0, 1.0]))
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_RANKING))

    summary = summarize_logs(
        metrics_eval_json=metrics_path, difficulty_ranking_json=ranking_path
    )
    easy = summary["eval"]["easy_decile"]
    assert easy["mpjpe_g"] == pytest.approx(1.5)
    # The manifest comparison reads metrics.eval.easy_decile.mpjpe_g — the
    # nesting must match compare_sonic_manifests._METRIC_PATHS.
    from scripts.research.compare_sonic_manifests import _METRIC_PATHS

    assert ("metrics", "eval", "easy_decile", "mpjpe_g") in _METRIC_PATHS


def test_missing_per_motion_table_raises() -> None:
    with pytest.raises(ValueError, match="all_metrics_dict"):
        compute_easy_decile_metrics({"eval/foo": 1}, _RANKING)
