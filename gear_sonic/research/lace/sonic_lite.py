"""Fail-closed static audit for the G1-only SONIC-Lite profiles.

The Isaac environment discovers observation dimensions at runtime, after the
simulator has started.  LACE needs the model scale frozen before that expensive
step, so this module binds the recovered SONIC release dimensions to the
executable Hydra fragments and analytically counts every trainable dense-layer
parameter.  It intentionally imports neither Isaac Lab nor PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from math import prod
from pathlib import Path
from typing import Any


class SonicLiteProfileError(ValueError):
    """Raised when a Lite profile is incomplete, inconsistent, or ambiguous."""


ACTOR_CONFIG = Path("gear_sonic/config/actor_critic/universal_token/g1_lite_s.yaml")
TOKENIZER_CONFIG = Path("gear_sonic/config/manager_env/observations/tokenizer/unitoken_g1_noz.yaml")
EXPERIMENT_CONFIG = Path("gear_sonic/config/exp/manager/universal_token/g1_only/lace_lite_s.yaml")
QUANTIZER_CONFIG = Path("gear_sonic/config/actor_critic/quantizers/fsq.yaml")

PROFILE_ID = "sonic_lite_s_v1"
ACTOR_PARAMETER_BAND = (1_000_000, 1_500_000)
CHECKPOINT_PERCENTAGES = (10, 50, 100)
G1_ENCODER_INPUTS = (
    "command_multi_future_nonflat",
    "motion_anchor_ori_b_mf_nonflat",
)
TOKENIZER_TERMS = ("encoder_index", *G1_ENCODER_INPUTS)
TOKENIZER_FEATURE_DIMS = {
    "encoder_index": (1,),
    "command_multi_future_nonflat": (10, 58),
    "motion_anchor_ori_b_mf_nonflat": (10, 6),
}
ACTION_DIM = 29
ACTOR_OBS_DIM = 930
CRITIC_OBS_DIM = 1645
NUM_FUTURE_FRAMES = 10
TOKEN_DIM = 32
MAX_NUM_TOKENS = 2
ENCODER_HIDDEN_DIMS = (384, 256, 128)
DECODER_HIDDEN_DIMS = (512, 384, 256, 128)
CRITIC_HIDDEN_DIMS = (512, 384, 256, 128)


@dataclass(frozen=True)
class SonicLiteParameterReport:
    """Audited trainable parameter counts and their frozen dimension contract."""

    profile_id: str
    encoder: int
    decoder: int
    quantizer: int
    exploration_noise: int
    actor: int
    critic: int
    total: int
    encoder_input_dim: int
    decoder_input_dim: int
    critic_input_dim: int
    tokenizer_total_dim: int
    checkpoint_iterations: tuple[int, int, int]
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record."""

        result = asdict(self)
        result["checkpoint_iterations"] = list(self.checkpoint_iterations)
        return result


def dense_mlp_parameter_count(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> int:
    """Count weights and biases for ``BaseModule``'s plain MLP construction."""

    dimensions = (input_dim, *tuple(hidden_dims), output_dim)
    if len(dimensions) < 3:
        raise SonicLiteProfileError("an audited MLP must contain at least one hidden layer")
    for index, dimension in enumerate(dimensions):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise SonicLiteProfileError(
                f"MLP dimension {index} must be a positive integer, got {dimension!r}"
            )
    return sum((source + 1) * target for source, target in zip(dimensions, dimensions[1:]))


def checkpoint_iterations(
    planned_ppo_iterations: int,
    percentages: Sequence[int] = CHECKPOINT_PERCENTAGES,
) -> tuple[int, ...]:
    """Resolve preregistered progress fractions with no discretionary rounding."""

    if (
        isinstance(planned_ppo_iterations, bool)
        or not isinstance(planned_ppo_iterations, int)
        or planned_ppo_iterations <= 0
    ):
        raise SonicLiteProfileError("planned_ppo_iterations must be a positive integer")
    percentage_tuple = tuple(percentages)
    if percentage_tuple != tuple(sorted(set(percentage_tuple))):
        raise SonicLiteProfileError("checkpoint percentages must be unique and increasing")

    iterations: list[int] = []
    for percentage in percentage_tuple:
        if (
            isinstance(percentage, bool)
            or not isinstance(percentage, int)
            or not 0 < percentage <= 100
        ):
            raise SonicLiteProfileError(
                f"checkpoint percentage must be an integer in [1, 100], got {percentage!r}"
            )
        numerator = planned_ppo_iterations * percentage
        if numerator % 100:
            raise SonicLiteProfileError(
                "checkpoint fraction is not an exact PPO iteration; "
                f"{percentage}% of {planned_ppo_iterations} would require rounding"
            )
        iterations.append(numerator // 100)
    return tuple(iterations)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SonicLiteProfileError(f"{context} must be a mapping")
    return value


def _require_sequence(value: object, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SonicLiteProfileError(f"{context} must be a sequence")
    return value


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise SonicLiteProfileError(f"{context} must be {expected!r}, got {actual!r}")


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        current = _require_mapping(value, path)
        if part not in current or current[part] is None:
            raise SonicLiteProfileError(f"missing required config value: {path}")
        value = current[part]
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - exercised only in minimal installs
        raise SonicLiteProfileError(
            "PyYAML is required to audit SONIC-Lite configs; install gear_sonic[sim] "
            "or gear_sonic[training]"
        ) from error

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SonicLiteProfileError(f"cannot read valid YAML from {path}: {error}") from error
    return _require_mapping(loaded, str(path))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _default_choices(config: Mapping[str, Any], context: str) -> tuple[dict[str, Any], set[str]]:
    defaults = _require_sequence(config.get("defaults"), f"{context}.defaults")
    choices: dict[str, Any] = {}
    plain: set[str] = set()
    for entry in defaults:
        if isinstance(entry, str):
            plain.add(entry)
            continue
        entry_mapping = _require_mapping(entry, f"{context}.defaults entry")
        if len(entry_mapping) != 1:
            raise SonicLiteProfileError(f"{context}.defaults entries must have one choice")
        key, value = next(iter(entry_mapping.items()))
        choices[key] = value
    return choices, plain


def _hidden_dims(module_config: Mapping[str, Any], context: str) -> tuple[int, ...]:
    layer = _require_mapping(
        _nested(module_config, "params.module_config_dict.layer_config"),
        f"{context}.layer_config",
    )
    _require_equal(layer.get("type"), "MLP", f"{context}.type")
    _require_equal(layer.get("activation"), "SiLU", f"{context}.activation")
    values = tuple(_require_sequence(layer.get("hidden_dims"), f"{context}.hidden_dims"))
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SonicLiteProfileError(f"{context}.hidden_dims must contain positive integers")
    return values


def _audit_experiment(experiment: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    choices, plain = _default_choices(experiment, "experiment")
    expected_choices = {
        "/algo": "ppo_im_phc",
        "/manager_env": "base_env",
        "override /trainer": "trl",
        "override /actor_critic": "universal_token/g1_lite_s",
        "override /manager_env/observations/tokenizer": "unitoken_g1_noz",
        "override /manager_env/observations/policy": "local_dir_hist",
        "override /manager_env/observations/critic": "privileged_mf_hist",
        "override /manager_env/events": "tracking/level0_4",
        "override /manager_env/terminations": "tracking/base_adaptive_strict_ori_foot_xyz",
        "override /manager_env/rewards": "tracking/base_5point_local_feet_acc",
    }
    for key, expected in expected_choices.items():
        _require_equal(choices.get(key), expected, f"experiment default {key}")
    for callback in ("/callbacks/read_eval", "/callbacks/im_resample"):
        if callback not in plain:
            raise SonicLiteProfileError(f"experiment must preserve callback {callback}")
    if any("aux_losses" in key for key in (*choices, *plain)):
        raise SonicLiteProfileError("Lite-S must not compose a cross-modal auxiliary-loss group")

    _require_equal(_nested(experiment, "manager_env.config.robot.type"), "g1_model_12_dex", "robot")
    _require_equal(_nested(experiment, "manager_env.config.terrain_type"), "trimesh", "terrain")
    _require_equal(
        _nested(experiment, "manager_env.commands.motion.num_future_frames"),
        NUM_FUTURE_FRAMES,
        "num_future_frames",
    )
    _require_equal(
        _nested(experiment, "manager_env.commands.motion.teleop_sample_prob_when_smpl"),
        0.0,
        "teleop sampling probability",
    )
    motion_lib = _require_mapping(
        _nested(experiment, "manager_env.commands.motion.motion_lib_cfg"), "motion_lib_cfg"
    )
    _require_equal(
        motion_lib.get("motion_file"),
        "data/motion_lib_bones_seed/robot_filtered",
        "Lite-S motion library",
    )
    if motion_lib.get("smpl_motion_file", object()) is not None:
        raise SonicLiteProfileError("Lite-S smpl_motion_file must be explicitly null")
    _require_equal(
        _nested(motion_lib, "adaptive_sampling.bin_size"), 50, "adaptive sampling bin_size"
    )

    research = _require_mapping(experiment.get("lace_research"), "lace_research")
    _require_equal(research.get("profile_id"), PROFILE_ID, "profile_id")
    band = tuple(_require_sequence(research.get("actor_parameter_band"), "actor_parameter_band"))
    _require_equal(band, ACTOR_PARAMETER_BAND, "actor_parameter_band")

    dimension_contract = _require_mapping(
        research.get("dimension_contract"), "lace_research.dimension_contract"
    )
    expected_scalars = {
        "action_dim": ACTION_DIM,
        "actor_obs_dim": ACTOR_OBS_DIM,
        "critic_obs_dim": CRITIC_OBS_DIM,
        "num_future_frames": NUM_FUTURE_FRAMES,
        "token_dim": TOKEN_DIM,
        "max_num_tokens": MAX_NUM_TOKENS,
    }
    for name, expected in expected_scalars.items():
        _require_equal(dimension_contract.get(name), expected, f"dimension_contract.{name}")
    feature_dims = _require_mapping(
        dimension_contract.get("tokenizer_feature_dims"), "tokenizer_feature_dims"
    )
    normalized_feature_dims = {
        key: tuple(_require_sequence(value, f"tokenizer_feature_dims.{key}"))
        for key, value in feature_dims.items()
    }
    _require_equal(normalized_feature_dims, TOKENIZER_FEATURE_DIMS, "tokenizer_feature_dims")

    checkpoints = _require_mapping(research.get("checkpoints"), "lace_research.checkpoints")
    _require_equal(checkpoints.get("rounding"), "exact_only", "checkpoint rounding")
    _require_equal(
        checkpoints.get("progress_basis"),
        "planned_ppo_iterations_with_fixed_rollout_and_optimizer_schedule",
        "checkpoint progress_basis",
    )
    percentages = tuple(_require_sequence(checkpoints.get("percentages"), "checkpoint percentages"))
    _require_equal(percentages, CHECKPOINT_PERCENTAGES, "checkpoint percentages")
    planned_iterations = _nested(experiment, "algo.config.num_learning_iterations")
    return planned_iterations, checkpoint_iterations(planned_iterations, percentages)


def _audit_tokenizer(tokenizer: Mapping[str, Any]) -> None:
    _, plain = _default_choices(tokenizer, "tokenizer")
    terms = tuple(
        entry.removeprefix("../terms/").removesuffix("@_here_")
        for entry in _require_sequence(tokenizer.get("defaults"), "tokenizer.defaults")
        if isinstance(entry, str)
    )
    _require_equal(terms, TOKENIZER_TERMS, "G1 tokenizer terms")
    _require_equal(
        tokenizer.get("_target_"),
        "gear_sonic.envs.manager_env.mdp.observations.TokenizerCfg",
        "tokenizer target",
    )
    _require_equal(tokenizer.get("enable_corruption"), True, "tokenizer corruption")
    _require_equal(tokenizer.get("concatenate_terms"), False, "tokenizer concatenation")
    if len(plain) != len(TOKENIZER_TERMS):
        raise SonicLiteProfileError("G1 tokenizer contains duplicate or non-term defaults")


def _audit_quantizer(actor: Mapping[str, Any], quantizer: Mapping[str, Any]) -> None:
    _, defaults = _default_choices(actor, "actor config")
    required_default = "quantizers/fsq@algo.config.actor.backbone.quantizer"
    _require_equal(defaults, {required_default}, "actor quantizer defaults")
    _require_equal(quantizer.get("_target_"), "vector_quantize_pytorch.FSQ", "quantizer target")


def audit_lite_s_profile(
    repo_root: str | Path,
    *,
    actor_config: str | Path = ACTOR_CONFIG,
    tokenizer_config: str | Path = TOKENIZER_CONFIG,
    experiment_config: str | Path = EXPERIMENT_CONFIG,
    quantizer_config: str | Path = QUANTIZER_CONFIG,
) -> SonicLiteParameterReport:
    """Validate the complete Lite-S contract and return exact trainable counts.

    All paths are injectable so CPU-only tests can audit isolated fixtures.
    The function rejects extra encoders/decoders, auxiliary losses, dimension
    drift, non-G1 tokenizer terms, and ambiguous checkpoint rounding.
    """

    root = Path(repo_root)
    actor = _load_yaml(_resolve(root, actor_config))
    tokenizer = _load_yaml(_resolve(root, tokenizer_config))
    experiment = _load_yaml(_resolve(root, experiment_config))
    quantizer = _load_yaml(_resolve(root, quantizer_config))

    _audit_quantizer(actor, quantizer)
    planned_iterations, frozen_checkpoints = _audit_experiment(experiment)
    _audit_tokenizer(tokenizer)

    encoder_probs = _nested(actor, "manager_env.commands.motion.encoder_sample_probs")
    _require_equal(
        dict(_require_mapping(encoder_probs, "encoder_sample_probs")),
        {"g1": 1.0},
        "encoder_sample_probs",
    )

    actor_config_map = _require_mapping(_nested(actor, "algo.config.actor"), "actor")
    _require_equal(
        actor_config_map.get("_target_"),
        "gear_sonic.trl.modules.actor_critic_modules.Actor",
        "actor target",
    )
    _require_equal(actor_config_map.get("input_obs_dict"), True, "actor.input_obs_dict")
    _require_equal(actor_config_map.get("has_aux_loss"), False, "actor.has_aux_loss")

    backbone = _require_mapping(actor_config_map.get("backbone"), "actor.backbone")
    _require_equal(
        backbone.get("_target_"),
        "gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule",
        "actor backbone target",
    )
    _require_equal(backbone.get("num_fsq_levels"), TOKEN_DIM, "num_fsq_levels")
    _require_equal(backbone.get("fsq_level_list"), 32, "fsq_level_list")
    _require_equal(backbone.get("max_num_tokens"), MAX_NUM_TOKENS, "max_num_tokens")
    _require_equal(backbone.get("proprioception_features"), ["actor_obs"], "proprioception")
    _require_equal(backbone.get("reencode_smpl_g1_recon"), False, "SMPL re-encoding")
    _require_equal(
        dict(_require_mapping(backbone.get("aux_loss_func"), "aux_loss_func")),
        {},
        "aux_loss_func",
    )
    _require_equal(
        dict(_require_mapping(backbone.get("aux_loss_coef"), "aux_loss_coef")),
        {},
        "aux_loss_coef",
    )

    encoders = _require_mapping(backbone.get("encoders"), "encoders")
    _require_equal(tuple(encoders), ("g1",), "encoder names")
    g1_encoder = _require_mapping(encoders["g1"], "encoders.g1")
    _require_equal(tuple(g1_encoder.get("inputs", ())), G1_ENCODER_INPUTS, "G1 encoder inputs")
    encoder_hidden = _hidden_dims(g1_encoder, "G1 encoder")
    _require_equal(encoder_hidden, ENCODER_HIDDEN_DIMS, "G1 encoder hidden_dims")

    decoders = _require_mapping(backbone.get("decoders"), "decoders")
    _require_equal(tuple(decoders), ("g1_dyn",), "decoder names")
    g1_decoder = _require_mapping(decoders["g1_dyn"], "decoders.g1_dyn")
    _require_equal(
        g1_decoder.get("inputs"), ["token_flattened", "proprioception"], "decoder inputs"
    )
    _require_equal(g1_decoder.get("outputs"), ["action"], "decoder outputs")
    _require_equal(g1_decoder.get("has_temporal_dim"), False, "decoder temporal mode")
    decoder_hidden = _hidden_dims(g1_decoder, "G1 decoder")
    _require_equal(decoder_hidden, DECODER_HIDDEN_DIMS, "G1 decoder hidden_dims")

    critic_config = _require_mapping(_nested(actor, "algo.config.critic"), "critic")
    _require_equal(
        critic_config.get("_target_"),
        "gear_sonic.trl.modules.actor_critic_modules.Critic",
        "critic target",
    )
    critic_backbone = _require_mapping(critic_config.get("backbone"), "critic.backbone")
    critic_layer = _require_mapping(
        _nested(critic_backbone, "module_config_dict.layer_config"), "critic layer"
    )
    _require_equal(critic_layer.get("type"), "MLP", "critic type")
    _require_equal(critic_layer.get("activation"), "SiLU", "critic activation")
    critic_hidden = tuple(_require_sequence(critic_layer.get("hidden_dims"), "critic.hidden_dims"))
    _require_equal(critic_hidden, CRITIC_HIDDEN_DIMS, "critic hidden_dims")
    _require_equal(
        _nested(critic_backbone, "module_config_dict.input_dim"), ["critic_obs"], "critic inputs"
    )
    _require_equal(_nested(critic_backbone, "module_config_dict.output_dim"), [1], "critic outputs")

    encoder_input_dim = sum(TOKENIZER_FEATURE_DIMS[name][-1] for name in G1_ENCODER_INPUTS)
    encoder_input_dim *= NUM_FUTURE_FRAMES
    token_total_dim = TOKEN_DIM * MAX_NUM_TOKENS
    decoder_input_dim = token_total_dim + ACTOR_OBS_DIM
    tokenizer_total_dim = sum(prod(shape) for shape in TOKENIZER_FEATURE_DIMS.values())

    encoder_count = dense_mlp_parameter_count(encoder_input_dim, encoder_hidden, token_total_dim)
    decoder_count = dense_mlp_parameter_count(decoder_input_dim, decoder_hidden, ACTION_DIM)
    quantizer_count = 0  # vector_quantize_pytorch.FSQ has buffers but no Parameters.
    exploration_count = ACTION_DIM
    actor_count = encoder_count + decoder_count + quantizer_count + exploration_count
    critic_count = dense_mlp_parameter_count(CRITIC_OBS_DIM, critic_hidden, 1)
    total_count = actor_count + critic_count
    if not ACTOR_PARAMETER_BAND[0] <= actor_count <= ACTOR_PARAMETER_BAND[1]:
        raise SonicLiteProfileError(
            f"actor has {actor_count:,} trainable parameters, outside "
            f"the preregistered [{ACTOR_PARAMETER_BAND[0]:,}, {ACTOR_PARAMETER_BAND[1]:,}] band"
        )

    digest_payload = {
        "actor": actor,
        "tokenizer": tokenizer,
        "experiment": experiment,
        "quantizer": quantizer,
        "planned_ppo_iterations": planned_iterations,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SonicLiteParameterReport(
        profile_id=PROFILE_ID,
        encoder=encoder_count,
        decoder=decoder_count,
        quantizer=quantizer_count,
        exploration_noise=exploration_count,
        actor=actor_count,
        critic=critic_count,
        total=total_count,
        encoder_input_dim=encoder_input_dim,
        decoder_input_dim=decoder_input_dim,
        critic_input_dim=CRITIC_OBS_DIM,
        tokenizer_total_dim=tokenizer_total_dim,
        checkpoint_iterations=frozen_checkpoints,
        config_sha256=config_sha256,
    )


def validate_runtime_dimensions(
    *,
    actor_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    tokenizer_feature_dims: Mapping[str, Sequence[int]],
) -> None:
    """Fail before training if simulator-discovered dimensions drift from the audit."""

    actual = {
        "actor_obs_dim": actor_obs_dim,
        "critic_obs_dim": critic_obs_dim,
        "action_dim": action_dim,
        "tokenizer_feature_dims": {
            key: tuple(value) for key, value in tokenizer_feature_dims.items()
        },
    }
    expected = {
        "actor_obs_dim": ACTOR_OBS_DIM,
        "critic_obs_dim": CRITIC_OBS_DIM,
        "action_dim": ACTION_DIM,
        "tokenizer_feature_dims": TOKENIZER_FEATURE_DIMS,
    }
    if actual != expected:
        raise SonicLiteProfileError(
            "runtime SONIC-Lite dimension contract drifted: "
            f"expected {expected!r}, got {actual!r}"
        )
