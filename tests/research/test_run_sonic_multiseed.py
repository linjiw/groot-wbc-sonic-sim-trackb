from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.run_sonic_multiseed import (
    render_spec_for_seed,
    run_multiseed_experiment,
)


def _summary(path: Path, *, reward: float, mpjpe: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {
                    "kind": "sonic_training_log",
                    "ok": True,
                    "learning_iteration": 50,
                    "mean_rewards": reward,
                    "total_timesteps": 9600,
                    "traceback_count": 0,
                    "log_path": "train.log",
                },
                "eval": {
                    "kind": "sonic_eval_log",
                    "ok": True,
                    "all": {"mpjpe_g": mpjpe, "mpjpe_l": 18.0, "mpjpe_pa": 12.0},
                    "terminated_final": 0,
                    "success_rate_final": 1.0,
                    "traceback_count": 0,
                    "log_path": "eval.log",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _template(tmp_path: Path, seeds: list[int]) -> dict:
    variants = []
    for name, mpjpe_offset in [("uniform_sampling_micro", 0.0), ("adaptive_sampling_micro", 0.5)]:
        for seed in seeds:
            _summary(
                tmp_path / f"summaries/{name}_seed{seed}.json",
                reward=0.9,
                mpjpe=30.0 + seed + mpjpe_offset,
            )
        variants.append(
            {
                "name": name,
                "summary_json": str(tmp_path / "summaries" / f"{name}_seed{{seed}}.json"),
                "train_command": f"python gear_sonic/train_agent_trl.py seed={{seed}} exp_var={name}_seed{{seed}}",
                "eval_command": f"python gear_sonic/eval_agent_trl.py seed={{seed}} exp_var={name}_seed{{seed}}",
                "interpretation": f"multiseed_fixture_{name}",
            }
        )
    return {
        "experiment_group": "multiseed_fixture",
        "hypothesis": "fixture hypothesis",
        "seed": 0,
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "git_commit": "abc1234",
        "variants": variants,
    }


def test_render_spec_substitutes_seed_everywhere() -> None:
    template = {
        "seed": 0,
        "variants": [{"train_command": "train seed={seed} exp_var=v_seed{seed}"}],
        "nested": {"list": ["a_{seed}", 3]},
    }

    spec = render_spec_for_seed(template, 2)

    assert spec["seed"] == 2
    assert spec["variants"][0]["train_command"] == "train seed=2 exp_var=v_seed2"
    assert spec["nested"]["list"] == ["a_2", 3]
    # Template itself is untouched.
    assert "{seed}" in template["variants"][0]["train_command"]


def test_multiseed_dry_run_materializes_seeds_and_aggregate(tmp_path: Path) -> None:
    seeds = [0, 1, 2]
    template = _template(tmp_path, seeds)
    output_dir = tmp_path / "run"

    run = run_multiseed_experiment(
        template,
        seeds=seeds,
        output_dir=output_dir,
        dry_run=True,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert run["dry_run"] is True
    assert run["seeds"] == seeds
    for seed in seeds:
        assert (output_dir / f"seed{seed}" / "spec.json").exists()
        assert (output_dir / f"seed{seed}" / "comparison.json").exists()
        spec = json.loads((output_dir / f"seed{seed}" / "spec.json").read_text(encoding="utf-8"))
        assert spec["seed"] == seed
        assert f"seed{seed}" in spec["variants"][0]["train_command"]
    aggregate = json.loads((output_dir / "aggregate_comparison.json").read_text(encoding="utf-8"))
    assert aggregate["comparison_count"] == 3
    assert aggregate["seeds"] == seeds
    # Fixture puts adaptive 0.5 above uniform on every seed.
    effect = aggregate["effect_summary"]
    assert effect["mean_delta_a_minus_b"] == pytest.approx(0.5)
    assert effect["passes_preregistered_effect_gate"] is False
    assert (output_dir / "aggregate_table.md").exists()
    assert (output_dir / "multiseed_run.json").exists()


def test_multiseed_rejects_template_without_seed_placeholder(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])
    for variant in template["variants"]:
        for key in ("summary_json", "train_command", "eval_command"):
            variant[key] = variant[key].replace("{seed}", "0")

    with pytest.raises(ValueError, match="placeholder"):
        run_multiseed_experiment(
            template,
            seeds=[0, 1],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )


def test_multiseed_rejects_variant_names_absent_from_template(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])

    with pytest.raises(ValueError, match="not found in spec template"):
        run_multiseed_experiment(
            template,
            seeds=[0],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="error_ema_micro",  # template has adaptive/uniform only
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )


def test_multiseed_invalidates_run_when_a_seed_yields_no_comparison(tmp_path: Path) -> None:
    seeds = [0, 1]
    template = _template(tmp_path, seeds)
    # Break seed 1: point its summaries at nonexistent files so no manifests emerge.
    for name in ("uniform_sampling_micro", "adaptive_sampling_micro"):
        (tmp_path / "summaries" / f"{name}_seed1.json").unlink()

    run = run_multiseed_experiment(
        template,
        seeds=seeds,
        output_dir=tmp_path / "run",
        dry_run=True,
        repo_root=tmp_path,
        variant_a="adaptive_sampling_micro",
        variant_b="uniform_sampling_micro",
        effect_metric="eval.all.mpjpe_g",
        a_minus_b_threshold=-0.5,
        min_improved_seeds=2,
    )

    assert run["missing_comparison_seeds"] == [1]
    assert run["ok_for_causal_comparison"] is False


def test_multiseed_rejects_duplicate_seeds(tmp_path: Path) -> None:
    template = _template(tmp_path, [0])

    with pytest.raises(ValueError, match="duplicate"):
        run_multiseed_experiment(
            template,
            seeds=[0, 0],
            output_dir=tmp_path / "run",
            dry_run=True,
            repo_root=tmp_path,
            variant_a="adaptive_sampling_micro",
            variant_b="uniform_sampling_micro",
            effect_metric=None,
            a_minus_b_threshold=-0.5,
            min_improved_seeds=2,
        )
