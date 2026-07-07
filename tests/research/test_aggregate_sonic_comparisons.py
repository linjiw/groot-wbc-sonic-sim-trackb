from __future__ import annotations

import json
from pathlib import Path

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
    adaptive_mpjpe_delta: float = 1.0,
) -> Path:
    comparison = {
        "schema_version": 1,
        "kind": "sonic_manifest_comparison",
        "manifest_count": 2,
        "variants": ["adaptive_sampling_micro", "uniform_sampling_micro"],
        "seeds": [seed],
        "control_mismatches": [] if ok else [{"field": "datasets.robot_motion", "values": {}}],
        "metric_warnings": [{"experiment": f"seed{seed}", "field": "eval.all.mpjpe_g", "problem": "missing"}]
        if warning
        else [],
        "checkpoint_warnings": [],
        "validation_errors": [],
        "ok_for_causal_comparison": ok and not warning,
        "rows": [
            {
                "experiment_id": f"uniform_seed{seed}",
                "variant": "uniform_sampling_micro",
                "seed": seed,
                "status": "needs_review",
                "git_commit": "abc1234",
                "metrics": {
                    "train.mean_rewards": 1.0 + seed,
                    "eval.ok": True,
                    "eval.all.mpjpe_g": 30.0 + seed,
                },
            },
            {
                "experiment_id": f"adaptive_seed{seed}",
                "variant": "adaptive_sampling_micro",
                "seed": seed,
                "status": "needs_review",
                "git_commit": "abc1234",
                "metrics": {
                    "train.mean_rewards": 1.1 + seed,
                    "eval.ok": True,
                    "eval.all.mpjpe_g": 30.0 + seed + adaptive_mpjpe_delta,
                },
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
    assert aggregate["comparison_count"] == 3
    assert aggregate["seeds"] == [0, 1, 2]
    assert aggregate["variants"] == ["adaptive_sampling_micro", "uniform_sampling_micro"]
    assert aggregate["seed_level_ok"] == {"0": True, "1": False, "2": True}
    assert aggregate["ok_for_causal_comparison"] is False
    assert aggregate["warning_counts"]["metric_warnings"] == 1
    assert aggregate["rows"][0]["seed"] == 0
    assert aggregate["rows"][0]["uniform.eval.all.mpjpe_g"] == 30.0
    assert aggregate["rows"][0]["adaptive.eval.all.mpjpe_g"] == 31.0
    assert aggregate["rows"][0]["delta.eval.all.mpjpe_g.adaptive_minus_uniform"] == 1.0


def test_build_aggregate_comparison_applies_preregistered_effect_gate(tmp_path: Path) -> None:
    paths = [
        _comparison(tmp_path / "seed0.json", seed=0, adaptive_mpjpe_delta=-0.75),
        _comparison(tmp_path / "seed1.json", seed=1, adaptive_mpjpe_delta=-0.60),
        _comparison(tmp_path / "seed2.json", seed=2, adaptive_mpjpe_delta=0.10),
    ]

    aggregate = build_aggregate_comparison(
        paths,
        effect_metric="eval.all.mpjpe_g",
        adaptive_minus_uniform_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert aggregate["effect_summary"] == {
        "metric": "eval.all.mpjpe_g",
        "adaptive_minus_uniform_threshold": -0.5,
        "min_improved_seeds": 2,
        "mean_delta_adaptive_minus_uniform": -0.4166666666666667,
        "improved_seed_count": 2,
        "seed_count": 3,
        "passes_preregistered_effect_gate": False,
    }
    assert aggregate["ok_for_causal_comparison"] is True


def test_build_aggregate_comparison_effect_gate_passes_only_when_mean_and_seed_count_pass(tmp_path: Path) -> None:
    paths = [
        _comparison(tmp_path / "seed0.json", seed=0, adaptive_mpjpe_delta=-0.75),
        _comparison(tmp_path / "seed1.json", seed=1, adaptive_mpjpe_delta=-0.60),
        _comparison(tmp_path / "seed2.json", seed=2, adaptive_mpjpe_delta=-0.20),
    ]

    aggregate = build_aggregate_comparison(
        paths,
        effect_metric="eval.all.mpjpe_g",
        adaptive_minus_uniform_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert aggregate["effect_summary"]["mean_delta_adaptive_minus_uniform"] < -0.5
    assert aggregate["effect_summary"]["improved_seed_count"] == 3
    assert aggregate["effect_summary"]["passes_preregistered_effect_gate"] is True


def test_write_aggregate_table_markdown_contains_seed_rows_and_guardrail(tmp_path: Path) -> None:
    aggregate = build_aggregate_comparison([_comparison(tmp_path / "seed0.json", seed=0)])
    output = tmp_path / "aggregate_table.md"

    write_aggregate_table_markdown(output, aggregate)

    text = output.read_text(encoding="utf-8")
    assert "# SONIC Aggregate Comparison" in text
    assert "No adaptive-sampling performance claim" in text
    assert "uniform.eval.all.mpjpe_g" in text
    assert "adaptive.eval.all.mpjpe_g" in text
    assert "delta.eval.all.mpjpe_g.adaptive_minus_uniform" in text
