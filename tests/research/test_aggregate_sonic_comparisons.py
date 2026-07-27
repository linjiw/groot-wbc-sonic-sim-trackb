from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.aggregate_sonic_comparisons import (
    build_aggregate_comparison,
    write_aggregate_table_markdown,
)


def _comparison(
    path: Path,
    *,
    seed: int,
    ok: bool = True,
    warning: bool = False,
    treatment_mpjpe_delta: float = 1.0,
    treatment_mpjpe_l_delta: float | None = None,
    variant_a: str = "adaptive_sampling_micro",
    variant_b: str = "uniform_sampling_micro",
    telemetry: bool = False,
    retention_delta: float | None = None,
    retention_complete: bool = True,
) -> Path:
    if treatment_mpjpe_l_delta is None:
        treatment_mpjpe_l_delta = treatment_mpjpe_delta
    a_metrics = {
        "train.mean_rewards": 1.1 + seed,
        "eval.ok": True,
        "eval.all.mpjpe_g": 30.0 + seed + treatment_mpjpe_delta,
        "eval.all.mpjpe_l": 20.0 + seed + treatment_mpjpe_l_delta,
    }
    if telemetry:
        a_metrics["train.adp_samp_prob_max_over_uniform"] = 3.05
        a_metrics["train.adp_samp_num_concentrated_bins"] = 0.0
        a_metrics["train.adp_samp_effective_num_bins"] = 69.7
        a_metrics["train.adp_samp_episodes_max_over_mean"] = 1.8
    b_metrics = {
        "train.mean_rewards": 1.0 + seed,
        "eval.ok": True,
        "eval.all.mpjpe_g": 30.0 + seed,
        "eval.all.mpjpe_l": 20.0 + seed,
    }
    if retention_delta is not None:
        evaluated = 2 if retention_complete else 1
        retention_common = {
            "eval.easy_decile.ok": True,
            "eval.easy_decile.evaluated_in_decile": evaluated,
            "eval.easy_decile.decile_size": 2,
            "eval.easy_decile.motion_keys": ["easy_0", "easy_1"][:evaluated],
            "eval.easy_decile.difficulty_ranking_path": "/frozen/d1_ranking.json",
        }
        b_metrics.update(retention_common)
        b_metrics["eval.easy_decile.mpjpe_g"] = 10.0 + seed
        a_metrics.update(retention_common)
        a_metrics["eval.easy_decile.mpjpe_g"] = 10.0 + seed + retention_delta
    comparison = {
        "schema_version": 1,
        "kind": "sonic_manifest_comparison",
        "manifest_count": 2,
        "variants": sorted([variant_a, variant_b]),
        "seeds": [seed],
        "control_mismatches": [] if ok else [{"field": "datasets.robot_motion", "values": {}}],
        "metric_warnings": (
            [{"experiment": f"seed{seed}", "field": "eval.all.mpjpe_g", "problem": "missing"}]
            if warning
            else []
        ),
        "checkpoint_warnings": [],
        "validation_errors": [],
        "ok_for_causal_comparison": ok and not warning,
        "rows": [
            {
                "experiment_id": f"control_seed{seed}",
                "variant": variant_b,
                "seed": seed,
                "status": "needs_review",
                "git_commit": "abc1234",
                "metrics": b_metrics,
            },
            {
                "experiment_id": f"treatment_seed{seed}",
                "variant": variant_a,
                "seed": seed,
                "status": "needs_review",
                "git_commit": "abc1234",
                "metrics": a_metrics,
            },
        ],
    }
    path.write_text(json.dumps(comparison), encoding="utf-8")
    return path


def test_build_aggregate_comparison_preserves_seed_level_gates(tmp_path: Path) -> None:
    seed0 = _comparison(tmp_path / "seed0.json", seed=0)
    seed1 = _comparison(tmp_path / "seed1.json", seed=1, warning=True)
    seed2 = _comparison(tmp_path / "seed2.json", seed=2)

    aggregate = build_aggregate_comparison([seed0, seed1, seed2])

    assert aggregate["kind"] == "sonic_aggregate_comparison"
    assert aggregate["schema_version"] == 2
    assert aggregate["comparison_count"] == 3
    assert aggregate["seeds"] == [0, 1, 2]
    assert aggregate["variants"] == ["adaptive_sampling_micro", "uniform_sampling_micro"]
    assert aggregate["variant_a"] == "adaptive_sampling_micro"
    assert aggregate["variant_b"] == "uniform_sampling_micro"
    assert aggregate["seed_level_ok"] == {"0": True, "1": False, "2": True}
    assert aggregate["ok_for_causal_comparison"] is False
    assert aggregate["warning_counts"]["metric_warnings"] == 1
    assert aggregate["rows"][0]["seed"] == 0
    assert aggregate["rows"][0]["b.eval.all.mpjpe_g"] == 30.0
    assert aggregate["rows"][0]["a.eval.all.mpjpe_g"] == 31.0
    assert aggregate["rows"][0]["delta.eval.all.mpjpe_g.a_minus_b"] == 1.0
    assert aggregate["missing_variant_rows"] == []


def test_build_aggregate_comparison_applies_preregistered_effect_gate(tmp_path: Path) -> None:
    paths = [
        _comparison(tmp_path / "seed0.json", seed=0, treatment_mpjpe_delta=-0.75),
        _comparison(tmp_path / "seed1.json", seed=1, treatment_mpjpe_delta=-0.60),
        _comparison(tmp_path / "seed2.json", seed=2, treatment_mpjpe_delta=0.10),
    ]

    aggregate = build_aggregate_comparison(
        paths,
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    effect = aggregate["effect_summary"]
    assert effect["metric"] == "eval.all.mpjpe_g"
    assert effect["a_minus_b_threshold"] == -0.5
    assert effect["min_improved_seeds"] == 2
    assert effect["mean_delta_a_minus_b"] == pytest.approx(-0.4166666666666667)
    assert effect["improved_seed_count"] == 2
    assert effect["seed_count"] == 3
    assert effect["passes_preregistered_effect_gate"] is False
    assert aggregate["ok_for_causal_comparison"] is True


def test_build_aggregate_comparison_effect_gate_passes_only_when_mean_and_seed_count_pass(
    tmp_path: Path,
) -> None:
    paths = [
        _comparison(tmp_path / "seed0.json", seed=0, treatment_mpjpe_delta=-0.75),
        _comparison(tmp_path / "seed1.json", seed=1, treatment_mpjpe_delta=-0.60),
        _comparison(tmp_path / "seed2.json", seed=2, treatment_mpjpe_delta=-0.20),
    ]

    aggregate = build_aggregate_comparison(
        paths,
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert aggregate["effect_summary"]["mean_delta_a_minus_b"] < -0.5
    assert aggregate["effect_summary"]["improved_seed_count"] == 3
    assert aggregate["effect_summary"]["passes_preregistered_effect_gate"] is True


def test_build_aggregate_comparison_supports_mpjpe_l_effect_gate(tmp_path: Path) -> None:
    path = _comparison(
        tmp_path / "seed0.json",
        seed=0,
        treatment_mpjpe_delta=4.0,
        treatment_mpjpe_l_delta=-0.75,
    )

    aggregate = build_aggregate_comparison(
        [path],
        effect_metric="eval.all.mpjpe_l",
        a_minus_b_threshold=0.0,
        min_improved_seeds=1,
    )

    row = aggregate["rows"][0]
    assert row["delta.eval.all.mpjpe_l.a_minus_b"] == -0.75
    assert aggregate["effect_summary"]["passes_preregistered_effect_gate"] is True


def test_retention_gate_requires_complete_easy_decile_and_noninferiority(
    tmp_path: Path,
) -> None:
    paths = [
        _comparison(
            tmp_path / f"seed{seed}.json",
            seed=seed,
            treatment_mpjpe_delta=-1.0,
            retention_delta=delta,
        )
        for seed, delta in enumerate((0.3, 0.4, 0.5))
    ]

    aggregate = build_aggregate_comparison(
        paths,
        effect_metric="eval.all.mpjpe_g",
        retention_metric="eval.easy_decile.mpjpe_g",
        retention_a_minus_b_threshold=0.5,
    )

    retention = aggregate["retention_summary"]
    assert retention["complete"] is True
    assert retention["coverage_ok"] is True
    assert retention["mean_delta_a_minus_b"] == pytest.approx(0.4)
    assert retention["passes_preregistered_retention_gate"] is True
    assert aggregate["passes_all_preregistered_result_gates"] is True


def test_retention_gate_rejects_thin_easy_decile_coverage(tmp_path: Path) -> None:
    path = _comparison(
        tmp_path / "seed0.json",
        seed=0,
        retention_delta=0.1,
        retention_complete=False,
    )

    aggregate = build_aggregate_comparison(
        [path], retention_metric="eval.easy_decile.mpjpe_g"
    )

    assert aggregate["retention_summary"]["coverage_ok"] is False
    assert aggregate["retention_summary"]["passes_preregistered_retention_gate"] is False


def test_effect_summary_reproduces_recorded_sim_m3_statistics(tmp_path: Path) -> None:
    """Validity check preregistered in fable-next.md Phase 0 item 5: the recorded
    SIM-M3 seed-level MPJPE-G values must reproduce mean delta +0.110667 and the
    exact one-sided sign-flip permutation p of 7/8."""
    sim_m3 = {0: (31.835, 32.037), 1: (31.554, 31.696), 2: (34.244, 34.232)}
    paths = []
    for seed, (uniform_mpjpe, adaptive_mpjpe) in sim_m3.items():
        path = _comparison(tmp_path / f"seed{seed}.json", seed=seed)
        comparison = json.loads(path.read_text(encoding="utf-8"))
        comparison["rows"][0]["metrics"]["eval.all.mpjpe_g"] = uniform_mpjpe
        comparison["rows"][1]["metrics"]["eval.all.mpjpe_g"] = adaptive_mpjpe
        path.write_text(json.dumps(comparison), encoding="utf-8")
        paths.append(path)

    aggregate = build_aggregate_comparison(paths, effect_metric="eval.all.mpjpe_g")

    effect = aggregate["effect_summary"]
    assert effect["mean_delta_a_minus_b"] == pytest.approx(0.110667, abs=1e-6)
    assert effect["improved_seed_count"] == 1
    assert effect["passes_preregistered_effect_gate"] is False
    stats = effect["statistics"]
    assert stats["permutation_p_one_sided"] == pytest.approx(7 / 8)
    assert stats["min_achievable_p"] == pytest.approx(1 / 8)
    assert "screen only" in stats["power_note"]


def test_build_aggregate_comparison_supports_custom_variant_names(tmp_path: Path) -> None:
    paths = [
        _comparison(
            tmp_path / f"seed{seed}.json",
            seed=seed,
            variant_a="error_ema_micro",
            variant_b="uniform_sampling_micro",
            treatment_mpjpe_delta=-1.0,
        )
        for seed in (0, 1)
    ]

    aggregate = build_aggregate_comparison(
        paths, variant_a="error_ema_micro", variant_b="uniform_sampling_micro"
    )

    assert aggregate["variant_a"] == "error_ema_micro"
    assert aggregate["rows"][0]["a.eval.all.mpjpe_g"] == 29.0
    assert aggregate["rows"][0]["delta.eval.all.mpjpe_g.a_minus_b"] == -1.0
    assert aggregate["missing_variant_rows"] == []


def test_build_aggregate_comparison_flags_unmatched_variant_names(tmp_path: Path) -> None:
    path = _comparison(tmp_path / "seed0.json", seed=0)

    aggregate = build_aggregate_comparison(
        [path], variant_a="typo_variant", variant_b="uniform_sampling_micro"
    )

    row = aggregate["rows"][0]
    assert row["a.variant_matched"] is False
    assert row["b.variant_matched"] is True
    assert row["a.eval.all.mpjpe_g"] is None
    assert row["delta.eval.all.mpjpe_g.a_minus_b"] is None
    # A one-sided name mismatch is a validity failure, not a silent None column.
    assert aggregate["missing_variant_rows"] == ["0"]
    assert aggregate["ok_for_causal_comparison"] is False


def test_build_aggregate_comparison_rejects_identical_variants(tmp_path: Path) -> None:
    path = _comparison(tmp_path / "seed0.json", seed=0)

    with pytest.raises(ValueError, match="must differ"):
        build_aggregate_comparison([path], variant_a="same", variant_b="same")


def test_build_aggregate_comparison_rejects_duplicate_paths(tmp_path: Path) -> None:
    path = _comparison(tmp_path / "seed0.json", seed=0)

    with pytest.raises(ValueError, match="duplicate comparison paths"):
        build_aggregate_comparison([path, path])


def test_build_aggregate_comparison_flags_duplicate_seeds(tmp_path: Path) -> None:
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    first = _comparison(tmp_path / "run_a" / "seed0.json", seed=0, treatment_mpjpe_delta=-1.0)
    second = _comparison(tmp_path / "run_b" / "seed0.json", seed=0, treatment_mpjpe_delta=-1.0)

    aggregate = build_aggregate_comparison([first, second])

    assert aggregate["duplicate_seed_rows"] == ["0"]
    assert aggregate["ok_for_causal_comparison"] is False


def test_aggregate_rows_carry_sampler_telemetry_for_variant_a(tmp_path: Path) -> None:
    path = _comparison(tmp_path / "seed0.json", seed=0, telemetry=True)

    aggregate = build_aggregate_comparison([path])

    row = aggregate["rows"][0]
    assert row["a.train.adp_samp_prob_max_over_uniform"] == 3.05
    assert row["a.train.adp_samp_num_concentrated_bins"] == 0.0
    assert row["b.train.adp_samp_prob_max_over_uniform"] is None


def test_write_aggregate_table_markdown_contains_seed_rows_and_guardrail(tmp_path: Path) -> None:
    aggregate = build_aggregate_comparison(
        [_comparison(tmp_path / "seed0.json", seed=0, telemetry=True)]
    )
    output = tmp_path / "aggregate_table.md"

    write_aggregate_table_markdown(output, aggregate)

    text = output.read_text(encoding="utf-8")
    assert "# SONIC Aggregate Comparison" in text
    assert "No sampling-mechanism performance claim" in text
    assert "b.eval.all.mpjpe_g" in text
    assert "a.eval.all.mpjpe_g" in text
    assert "delta.eval.all.mpjpe_g.a_minus_b" in text
    assert "delta.eval.all.mpjpe_l.a_minus_b" in text
    assert "A = treatment, B = control" in text
    assert "Sampler telemetry" in text
