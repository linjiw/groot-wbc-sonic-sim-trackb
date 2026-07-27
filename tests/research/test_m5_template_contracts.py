from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.research.run_sonic_multiseed import (
    render_spec_for_seed,
    validate_m5_multiseed_contract,
)
from scripts.research.run_sonic_paired_experiment import validate_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = (
    "sim_m5_l_learnability_multiseed_template.json",
    "sim_m5_t_threshold_multiseed_template.json",
)
_EXPECTED_VARIANTS = {
    "sim_m5_l_learnability_multiseed_template.json": (
        "learnability_m5l",
        "uniform_m5l",
    ),
    "sim_m5_t_threshold_multiseed_template.json": (
        "threshold_schedule_m5t",
        "release_threshold_m5t",
    ),
}
_NOMINAL_FULL_SEQUENCE_TOKENS = (
    "++seed=0",
    "++num_envs=1",
    "++callbacks.im_eval.max_eval_steps=null",
    "+eval_events=nominal_d1",
    "+manager_env/terminations=tracking/eval",
    "++manager_env.config.terrain_type=plane",
    "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
    "++manager_env.observations.policy.enable_corruption=false",
    "++manager_env.observations.tokenizer.enable_corruption=false",
    "+use_encoder=g1",
)


def _load_template(name: str) -> dict:
    path = REPO_ROOT / "configs" / "research" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_m5_templates_use_unique_deterministic_trained_checkpoints() -> None:
    all_checkpoints: set[str] = set()
    for template_name in _TEMPLATES:
        template = _load_template(template_name)
        assert (template["variant_a"], template["variant_b"]) == _EXPECTED_VARIANTS[template_name]
        assert template["checkpoint"] == "VARIANT_SPECIFIC_CHECKPOINTS"
        assert template["checkpoint_source"] == "trained_variant_checkpoint"
        assert template["requires_activation_gate"] is True
        assert template["activation_arm"] in {"m5_l", "m5_t"}
        assert "REPLACE_ME" in template["dataset_manifest_json"]
        assert "REPLACE_ME" in template["dataset_manifest_sha256"]
        assert "REPLACE_ME" in template["sim_d1_classification_json"]
        assert "REPLACE_ME" in template["sim_d1_classification_sha256"]
        assert "REPLACE_ME" in template["sim_d1_source_checkpoint_sha256"]
        assert template["training_python_executable"] == ("TRAINING_PYTHON_EXECUTABLE_REPLACE_ME")
        assert template["training_initialization_checkpoint"] == ("SIM_D1_SOURCE_CHECKPOINT_REPLACE_ME")
        assert "REPLACE_ME" in template["difficulty_ranking_sha256"]
        assert template["expected_training_iterations"] == 200
        assert template["training_num_envs"] == "DATASET_MOTION_COUNT_REPLACE_ME"
        assert template["difficulty_ranking_json"].endswith("REPLACE_ME.json")
        assert template["retention_metric"] == "eval.easy_decile.mpjpe_g"
        assert template["retention_a_minus_b_threshold"] == 0.5

        for seed in (0, 1, 2):
            spec = render_spec_for_seed(template, seed)
            assert validate_spec(spec) == []
            assert validate_m5_multiseed_contract(spec) == []
            for variant in spec["variants"]:
                checkpoint = variant["checkpoint"]
                checkpoint_dir = str(Path(checkpoint).parent)
                assert variant["checkpoint_source"] == "trained_variant_checkpoint"
                assert "num_envs=DATASET_MOTION_COUNT_REPLACE_ME" in variant["train_command"]
                assert "++algo.config.num_learning_iterations=200" in variant["train_command"]
                assert f"experiment_dir={checkpoint_dir}" in variant["train_command"]
                assert "++callbacks.model_save.save_last_frequency=200" in variant["train_command"]
                assert variant["train_command"].count("+checkpoint=SIM_D1_SOURCE_CHECKPOINT_REPLACE_ME") == 1
                assert variant["train_command"].count("+resume=false") == 1
                for command_field in ("train_command", "eval_command"):
                    assert "TRAINING_PYTHON_EXECUTABLE_REPLACE_ME" in variant[command_field]
                    assert " python " not in f" {variant[command_field]} "
                assert f"+checkpoint={checkpoint}" in variant["eval_command"]
                metrics_eval = variant["metrics_eval_json"]
                eval_output_dir = str(Path(metrics_eval).parent)
                assert f"++eval_output_dir={eval_output_dir}" in variant["eval_command"]
                for token in _NOMINAL_FULL_SEQUENCE_TOKENS:
                    assert token in variant["eval_command"]
                assert "algo.config.eval.num_eval_episodes" not in variant["eval_command"]
                assert "max_unique_motions" not in variant["eval_command"]
                assert "max_eval_steps=200" not in variant["eval_command"]
                assert checkpoint not in all_checkpoints
                all_checkpoints.add(checkpoint)


def test_m5_t_control_and_eval_threshold_semantics_are_explicit() -> None:
    template = _load_template("sim_m5_t_threshold_multiseed_template.json")
    variants = {variant["name"]: variant for variant in template["variants"]}

    assert set(variants) == {"release_threshold_m5t", "threshold_schedule_m5t"}
    assert "trainer=trl_threshold_curriculum" not in variants["release_threshold_m5t"]["train_command"]
    assert "trainer=trl_threshold_curriculum" in variants["threshold_schedule_m5t"]["train_command"]
    for variant in variants.values():
        # Both arms retain release failure-rate sampling; the only arm difference
        # is the threshold schedule.
        assert "adaptive_sampling.enable=false" not in variant["train_command"]
        assert "++manager_env.config.log_m5t_height_termination=true" in variant["train_command"]
        assert "manager_env.terminations.anchor_pos.params.threshold=0.15" in variant["eval_command"]
        assert "manager_env.terminations.ee_body_pos.params.threshold=0.15" in variant["eval_command"]
        assert "manager_env.terminations.anchor_pos.params.down_threshold=0.15" in variant["eval_command"]
        assert "manager_env.terminations.ee_body_pos.params.down_threshold=0.15" in variant["eval_command"]
        assert "trainer.schedule_dict=null" in variant["eval_command"]
    assert "matched fixed strict-vertical 0.15 verifier" in template["hypothesis"]
    assert "not a full release-policy verifier" in template["hypothesis"]
    assert "complete release position-termination policy" not in template["hypothesis"]


def test_m5_t_schedule_anneals_height_policy_with_release_aux_trainer() -> None:
    config_path = REPO_ROOT / "gear_sonic" / "config" / "trainer" / "trl_threshold_curriculum.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["_target_"] == (
        "gear_sonic.trl.trainer.ppo_trainer_aux_loss.TRLAuxLossPPOTrainer"
    )
    schedule = config["schedule_dict"]

    assert len(schedule) == 4
    for term in ("anchor_pos", "ee_body_pos"):
        ordinary = schedule[f"env@env@unwrapped@termination_manager@get_term_cfg('{term}')@params['threshold']"]
        low_root = schedule[
            f"env@env@unwrapped@termination_manager@get_term_cfg('{term}')@params['down_threshold']"
        ]
        assert ordinary == {
            "type": "linear",
            "seg_steps": [0, 150],
            "seg_vals": [0.30, 0.15],
        }
        assert low_root == {
            "type": "linear",
            "seg_steps": [0, 150],
            "seg_vals": [1.50, 0.75],
        }
        assert [low / normal for low, normal in zip(low_root["seg_vals"], ordinary["seg_vals"], strict=True)] == [
            5.0,
            5.0,
        ]


def test_m5_l_treatment_explicitly_binds_dump_sampler_config() -> None:
    template = _load_template("sim_m5_l_learnability_multiseed_template.json")
    treatment = next(variant for variant in template["variants"] if variant["name"] == template["variant_a"])
    command = treatment["train_command"]
    fields = (
        "enable",
        "signal",
        "optimism_k",
        "evidence_half_life",
        "advmass_n",
        "uniform_sampling_rate",
        "tripwire_max_prob_over_uniform",
        "bin_size",
        "paired_dataset_sha256",
    )
    for field in fields:
        assert command.count(f"adaptive_sampling.{field}=") == 1
    assert "adaptive_sampling.enable=true" in command
    assert "adaptive_sampling.signal=learnability" in command
    assert ("adaptive_sampling.paired_dataset_sha256=PAIRED_DATASET_SHA256_REPLACE_ME") in command


def test_activation_contract_rejects_duplicate_nominal_overrides_and_limiters() -> None:
    duplicate = render_spec_for_seed(_load_template("sim_m5_l_learnability_multiseed_template.json"), 0)
    duplicate["variants"][0]["eval_command"] += " ++num_envs=2"
    errors = validate_spec(duplicate)
    assert any("exactly one num_envs=1" in error and "['1', '2']" in error for error in errors)

    limited = render_spec_for_seed(_load_template("sim_m5_l_learnability_multiseed_template.json"), 0)
    limited["variants"][0]["eval_command"] += (
        " ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[motion]"
    )
    errors = validate_spec(limited)
    assert any("full-sequence limiter" in error for error in errors)

    wrong_training_budget = render_spec_for_seed(
        _load_template("sim_m5_l_learnability_multiseed_template.json"), 0
    )
    wrong_training_budget["variants"][0]["train_command"] += " ++algo.config.num_learning_iterations=199"
    errors = validate_spec(wrong_training_budget)
    assert any(
        "algo.config.num_learning_iterations=200" in error and "['200', '199']" in error for error in errors
    )

    missing_m5t_telemetry = render_spec_for_seed(
        _load_template("sim_m5_t_threshold_multiseed_template.json"), 0
    )
    missing_m5t_telemetry["variants"][0]["train_command"] = missing_m5t_telemetry[
        "variants"
    ][0]["train_command"].replace(
        " ++manager_env.config.log_m5t_height_termination=true", ""
    )
    errors = validate_spec(missing_m5t_telemetry)
    assert any("log_m5t_height_termination=true" in error for error in errors)
