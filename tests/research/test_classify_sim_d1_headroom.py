from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.classify_sim_d1_headroom import classify_sim_d1
from scripts.research.summarize_sonic_logs import load_difficulty_ranking


def _metrics(motion_keys, mpjpe_g, terminated, progress):
    return {
        "eval/all_metrics_dict": {
            "motion_keys": motion_keys,
            "mpjpe_g": mpjpe_g,
            "terminated": terminated,
            "progress": progress,
        }
    }


def _passing_metrics():
    # Exactly 20% frontier motions and 30% mastered anchors.  The MPJPE spread
    # comfortably passes, and table order is deliberately unrelated to ease.
    return _metrics(
        ["m9", "m1", "m0", "m8", "m2", "m7", "m3", "m6", "m4", "m5"],
        [90, 10, 0, 80, 20, 70, 30, 60, 40, 50],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0.5, 1.0, 1.0, 0.8, 1.0, 0.95, 0.95, 0.95, 0.95, 0.95],
    )


def test_preregistered_gate_pass_and_deterministic_ranking() -> None:
    result = classify_sim_d1(
        _passing_metrics(),
        dataset="d_b_speed",
        dataset_kind="synthetic",
        expected_motion_count=10,
    )

    assert result["schema_version"] == 1
    assert result["kind"] == "sim_d1_headroom_classification"
    assert result["verdict"] == "PASS"
    assert result["pass"] is True
    assert result["eligible_for_effect_experiment"] is True
    assert result["failed_gates"] == []
    assert result["preregistered_defaults_used"] is True
    assert result["classification_mode"] == "preregistered"
    assert result["gates"]["frontier_exists"]["fraction"] == pytest.approx(0.2)
    assert result["gates"]["mastered_anchor_exists"]["fraction"] == pytest.approx(0.3)
    assert result["ranking"] == [f"m{i}" for i in range(10)]
    assert result["coverage"]["all_motion_coverage_independently_verified"] is True


def test_gate_fail_reports_every_failed_condition() -> None:
    metrics = _metrics(
        [f"m{i}" for i in range(10)],
        [10.0] * 10,
        [False] * 10,
        [1.0] * 10,
    )
    result = classify_sim_d1(metrics)

    assert result["verdict"] == "FAIL"
    assert result["pass"] is False
    assert result["failed_gates"] == ["difficulty_spread", "frontier_exists"]
    assert result["gates"]["mastered_anchor_exists"]["pass"] is True
    assert result["summary"]["mpjpe_g_p90_over_p10"] == pytest.approx(1.0)


def test_unverified_all_motion_claim_is_visible_and_blocks_eligibility() -> None:
    result = classify_sim_d1(_passing_metrics())

    assert result["pass"] is True
    assert result["eligible_for_effect_experiment"] is False
    assert "all_motion_coverage_not_independently_verified" in result["eligibility_blockers"]
    assert "warning" in result["coverage"]


def test_equal_mpjpe_uses_lexical_motion_key_tie_break() -> None:
    metrics = _passing_metrics()
    table = metrics["eval/all_metrics_dict"]
    table["mpjpe_g"][1] = table["mpjpe_g"][2] = 5.0
    result = classify_sim_d1(metrics)

    assert result["ranking"][:2] == ["m0", "m1"]


def test_spread_or_ratio_logic_and_zero_denominator_are_explicit() -> None:
    ratio_pass = classify_sim_d1(
        _metrics(
            [f"m{i}" for i in range(10)],
            [10.0] * 8 + [20.0] * 2,
            [True, True] + [False] * 8,
            [0.5, 0.8] + [1.0] * 8,
        ),
        mpjpe_spread_min=100.0,
    )
    assert ratio_pass["gates"]["difficulty_spread"]["absolute_condition_pass"] is False
    assert ratio_pass["gates"]["difficulty_spread"]["ratio_condition_pass"] is True

    zero_denominator = classify_sim_d1(
        _metrics(
            [f"m{i}" for i in range(10)],
            [0.0] * 9 + [1.0],
            [True, True] + [False] * 8,
            [0.5, 0.8] + [1.0] * 8,
        )
    )
    assert zero_denominator["summary"]["mpjpe_g_p90_over_p10"] is None
    assert zero_denominator["gates"]["difficulty_spread"]["ratio_evaluable"] is False
    assert zero_denominator["gates"]["difficulty_spread"]["ratio_condition_pass"] is False


def test_threshold_override_is_never_labeled_preregistered() -> None:
    result = classify_sim_d1(
        _passing_metrics(), expected_motion_count=10, mpjpe_spread_min=21.0
    )
    assert result["classification_mode"] == "exploratory_threshold_override"
    assert result["preregistered_defaults_used"] is False
    assert result["eligible_for_effect_experiment"] is False


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (_metrics(["a", "a"], [1, 2], [0, 0], [1, 1]), "duplicates"),
        (_metrics(["a", "b"], [1], [0, 0], [1, 1]), "length 2"),
        (_metrics(["a"], [float("nan")], [0], [1]), "finite"),
        (_metrics(["a"], [1], [0], [1.1]), "<= 1.0"),
        (_metrics(["a"], [1], [2], [1]), "boolean or numeric 0/1"),
    ],
)
def test_rejects_malformed_per_motion_tables(metrics, message) -> None:
    with pytest.raises(ValueError, match=message):
        classify_sim_d1(metrics)


def test_exact_expected_motion_keys_reject_incomplete_eval() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        classify_sim_d1(
            _passing_metrics(),
            expected_motion_keys=[f"m{i}" for i in range(11)],
        )


def test_cli_emits_artifact_consumable_as_frozen_ranking(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics_eval.json"
    output_path = tmp_path / "sim_d1.json"
    expected_path = tmp_path / "expected.json"
    metrics_path.write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    expected_path.write_text(
        json.dumps({"motion_keys": [f"m{i}" for i in range(10)]}), encoding="utf-8"
    )
    script = Path(__file__).parents[2] / "scripts/research/classify_sim_d1_headroom.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics-eval-json",
            str(metrics_path),
            "--output-json",
            str(output_path),
            "--dataset",
            "d_b_speed",
            "--dataset-kind",
            "synthetic",
            "--expected-motion-keys-json",
            str(expected_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["verdict"] == "PASS"
    assert len(artifact["source"]["metrics_eval_sha256"]) == 64
    assert artifact["coverage"]["exact_key_set_verified"] is True
    assert load_difficulty_ranking(output_path) == [f"m{i}" for i in range(10)]
