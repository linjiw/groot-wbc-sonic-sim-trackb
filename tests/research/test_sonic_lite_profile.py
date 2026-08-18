from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gear_sonic.research.lace.sonic_lite import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    EXPERIMENT_CONFIG,
    G1_ENCODER_INPUTS,
    TOKENIZER_FEATURE_DIMS,
    SonicLiteProfileError,
    audit_lite_s_profile,
    checkpoint_iterations,
    dense_mlp_parameter_count,
    validate_runtime_dimensions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_lite_s_profile_has_exact_audited_trainable_counts() -> None:
    report = audit_lite_s_profile(REPO_ROOT)

    assert report.profile_id == "sonic_lite_s_v1"
    assert report.encoder_input_dim == 640
    assert report.decoder_input_dim == 994
    assert report.critic_input_dim == 1645
    assert report.tokenizer_total_dim == 641
    assert report.encoder == 385_856
    assert report.decoder == 841_629
    assert report.quantizer == 0
    assert report.exploration_noise == 29
    assert report.actor == 1_227_514
    assert report.critic == 1_171_329
    assert report.total == 2_398_843
    assert report.checkpoint_iterations == (10_000, 50_000, 100_000)
    assert len(report.config_sha256) == 64
    assert report.to_dict()["checkpoint_iterations"] == [10_000, 50_000, 100_000]


def test_lite_s_hydra_groups_compose_without_isaac() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    config_dir = REPO_ROOT / "gear_sonic/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(
            config_name="base",
            overrides=["+exp=manager/universal_token/g1_only/lace_lite_s"],
        )

    assert list(config.algo.config.actor.backbone.encoders) == ["g1"]
    assert list(config.algo.config.actor.backbone.decoders) == ["g1_dyn"]
    assert config.algo.config.actor.has_aux_loss is False
    assert config.trainer._target_.endswith("ppo_trainer.TRLPPOTrainer")
    assert config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file is None


def test_dense_mlp_count_includes_every_weight_and_bias() -> None:
    # Linear(3, 5) + Linear(5, 2): 15+5 + 10+2.
    assert dense_mlp_parameter_count(3, [5], 2) == 32

    with pytest.raises(SonicLiteProfileError, match="at least one hidden layer"):
        dense_mlp_parameter_count(3, [], 2)
    with pytest.raises(SonicLiteProfileError, match="positive integer"):
        dense_mlp_parameter_count(3, [True], 2)


def test_checkpoint_fractions_are_exact_and_preregistered() -> None:
    assert checkpoint_iterations(1000) == (100, 500, 1000)

    with pytest.raises(SonicLiteProfileError, match="would require rounding"):
        checkpoint_iterations(12)
    with pytest.raises(SonicLiteProfileError, match="unique and increasing"):
        checkpoint_iterations(1000, [50, 10, 100])


def test_profile_fails_closed_if_a_cross_modal_encoder_is_added(tmp_path: Path) -> None:
    actor_path = REPO_ROOT / "gear_sonic/config/actor_critic/universal_token/g1_lite_s.yaml"
    actor = _read_yaml(actor_path)
    mutated = deepcopy(actor)
    mutated["algo"]["config"]["actor"]["backbone"]["encoders"]["teleop"] = deepcopy(
        mutated["algo"]["config"]["actor"]["backbone"]["encoders"]["g1"]
    )
    fixture_path = tmp_path / "actor.yaml"
    _write_yaml(fixture_path, mutated)

    with pytest.raises(SonicLiteProfileError, match="encoder names"):
        audit_lite_s_profile(REPO_ROOT, actor_config=fixture_path)


def test_profile_fails_closed_on_width_or_checkpoint_drift(tmp_path: Path) -> None:
    actor_path = REPO_ROOT / "gear_sonic/config/actor_critic/universal_token/g1_lite_s.yaml"
    actor = _read_yaml(actor_path)
    actor["algo"]["config"]["actor"]["backbone"]["decoders"]["g1_dyn"]["params"][
        "module_config_dict"
    ]["layer_config"]["hidden_dims"] = [256, 256]
    actor_fixture = tmp_path / "actor.yaml"
    _write_yaml(actor_fixture, actor)
    with pytest.raises(SonicLiteProfileError, match="decoder hidden_dims"):
        audit_lite_s_profile(REPO_ROOT, actor_config=actor_fixture)

    experiment = _read_yaml(REPO_ROOT / EXPERIMENT_CONFIG)
    experiment["lace_research"]["checkpoints"]["percentages"] = [25, 50, 100]
    experiment_fixture = tmp_path / "experiment.yaml"
    _write_yaml(experiment_fixture, experiment)
    with pytest.raises(SonicLiteProfileError, match="checkpoint percentages"):
        audit_lite_s_profile(REPO_ROOT, experiment_config=experiment_fixture)


def test_runtime_dimension_gate_accepts_exact_contract_and_rejects_drift() -> None:
    validate_runtime_dimensions(
        actor_obs_dim=ACTOR_OBS_DIM,
        critic_obs_dim=CRITIC_OBS_DIM,
        action_dim=ACTION_DIM,
        tokenizer_feature_dims=TOKENIZER_FEATURE_DIMS,
    )

    drifted = dict(TOKENIZER_FEATURE_DIMS)
    drifted[G1_ENCODER_INPUTS[0]] = (10, 59)
    with pytest.raises(SonicLiteProfileError, match="dimension contract drifted"):
        validate_runtime_dimensions(
            actor_obs_dim=ACTOR_OBS_DIM,
            critic_obs_dim=CRITIC_OBS_DIM,
            action_dim=ACTION_DIM,
            tokenizer_feature_dims=drifted,
        )
