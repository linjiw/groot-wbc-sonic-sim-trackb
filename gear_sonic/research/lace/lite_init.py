"""Environment-independent initialization bundle for SONIC-Lite-S.

The normal training entrypoint constructs an Isaac environment before the
actor and critic.  That makes a shared seed insufficient for an env-count
sweep because environment construction may consume an ``N_env``-dependent
number of random draws.  This module composes the exact Lite-S model on CPU,
seeds immediately before model construction, and writes a checkpoint before
any environment exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from types import SimpleNamespace
from typing import Any, Mapping
import uuid

from gear_sonic.research.lace.sonic_lite import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    PROFILE_ID,
    TOKENIZER_FEATURE_DIMS,
    audit_lite_s_profile,
)

RECEIPT_KIND = "lace_sonic_lite_initialization_receipt"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class LiteInitializationError(RuntimeError):
    """Raised when the shared Lite-S initialization is invalid or ambiguous."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LiteInitializationError(f"cannot read initialization receipt: {path}") from error
    if not isinstance(value, dict):
        raise LiteInitializationError("initialization receipt must contain a JSON object")
    return value


def _tensor_record(tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise LiteInitializationError(f"state-dict value is not a tensor: {type(tensor)!r}")
    value = tensor.detach().cpu().contiguous()
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
        raise LiteInitializationError("initialization state contains a non-finite tensor")
    # Flatten first because PyTorch rejects dtype-reinterpretation directly on
    # zero-dimensional tensors (the actor includes scalar state).  Reinterpreting
    # the contiguous 1-D storage also supports dtypes such as bfloat16 that NumPy
    # cannot represent natively.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "elements": value.numel(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _state_manifest(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    tensors = {name: _tensor_record(state_dict[name]) for name in sorted(state_dict)}
    return {
        "tensor_count": len(tensors),
        "element_count": sum(int(record["elements"]) for record in tensors.values()),
        "tensors": tensors,
        "state_sha256": _canonical_sha256(tensors),
    }


def _compose_lite_config(repo_root: Path, *, seed: int, destination: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    with initialize_config_dir(
        config_dir=str((repo_root / "gear_sonic/config").resolve()), version_base="1.1"
    ):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/g1_only/lace_lite_s",
                "num_envs=1",
                f"seed={seed}",
                "headless=true",
                "use_wandb=false",
                f"experiment_dir={destination}",
            ],
        )
    with open_dict(config):
        config.checkpoint = None
        config.resume = False
        config.exp_var = "shared_pre_env_initialization"
        config.algo.trl.output_dir = str(destination)
    # Resolve a detached copy only for fields that do not use Hydra runtime;
    # the saved training config intentionally preserves standard interpolations.
    return config, OmegaConf


def _synthetic_dimension_config():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "robot": {
                "actions_dim": ACTION_DIM,
                "algo_obs_dim_dict": {
                    "actor_obs": ACTOR_OBS_DIM,
                    "critic_obs": CRITIC_OBS_DIM,
                },
            },
            "obs": {
                "group_obs_dims": {
                    "tokenizer": {
                        name: list(dimensions)
                        for name, dimensions in TOKENIZER_FEATURE_DIMS.items()
                    }
                },
                "group_obs_names": {"tokenizer": list(TOKENIZER_FEATURE_DIMS)},
            },
        }
    )


def _model_spec(config: Any) -> dict[str, Any]:
    """Return the complete model-construction contract in canonical form."""

    from omegaconf import OmegaConf

    dimensions = _synthetic_dimension_config()
    return {
        "actor": OmegaConf.to_container(config.algo.config.actor, resolve=True),
        "critic": OmegaConf.to_container(config.algo.config.critic, resolve=True),
        "dimension_contract": OmegaConf.to_container(dimensions, resolve=True),
    }


def _instantiate_lite_models(config: Any):
    """Instantiate the exact actor and critic without constructing an environment."""

    from gear_sonic.trl.utils.common import custom_instantiate

    dimensions = _synthetic_dimension_config()
    common = {
        "env_config": dimensions,
        "algo_config": config.algo.config,
        "module_dim_dict": {},
        "backbone_kwargs": {},
        "_resolve": False,
    }
    return (
        custom_instantiate(config.algo.config.actor, **common).cpu(),
        custom_instantiate(config.algo.config.critic, **common).cpu(),
    )


def materialize_lite_initialization(
    *,
    repo_root: str | Path,
    destination: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Atomically create one environment-independent Lite-S checkpoint bundle."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LiteInitializationError("seed must be a non-negative integer")
    root = Path(repo_root).resolve()
    target = Path(destination).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"initialization destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"

    try:
        stage.mkdir()
        final_checkpoint = target / "last.pt"
        config, omega_conf = _compose_lite_config(root, seed=seed, destination=target)
        profile = audit_lite_s_profile(root)

        # No environment is imported or constructed above this point.  Seed
        # immediately before the only model constructors in this routine.
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        policy, value_model = _instantiate_lite_models(config)
        actor_parameters = sum(parameter.numel() for parameter in policy.parameters())
        critic_parameters = sum(parameter.numel() for parameter in value_model.parameters())
        if actor_parameters != profile.actor or critic_parameters != profile.critic:
            raise LiteInitializationError(
                "instantiated Lite-S parameter counts differ from the frozen profile"
            )
        import torch.nn as nn

        if any(isinstance(module, (nn.LazyLinear, nn.LazyConv2d)) for module in policy.modules()):
            raise LiteInitializationError("Lite-S initialization retained lazy actor parameters")
        if any(
            isinstance(module, (nn.LazyLinear, nn.LazyConv2d)) for module in value_model.modules()
        ):
            raise LiteInitializationError("Lite-S initialization retained lazy critic parameters")

        policy_state = policy.state_dict()
        value_state = value_model.state_dict()
        policy_manifest = _state_manifest(policy_state)
        value_manifest = _state_manifest(value_state)
        model_spec = _model_spec(config)
        checkpoint = {
            "policy_state_dict": policy_state,
            "value_state_dict": value_state,
            "optimizer_state_dict": None,
            "lr_scheduler_state_dict": None,
            "state": SimpleNamespace(global_step=0),
            "args": None,
            "env_state_dict": {},
            "lace_initialization": {
                "profile_id": PROFILE_ID,
                "seed": seed,
                "construction_order": "seed_then_actor_critic_without_environment",
                "policy_state_sha256": policy_manifest["state_sha256"],
                "value_state_sha256": value_manifest["state_sha256"],
            },
        }
        checkpoint_path = stage / "last.pt"
        torch.save(checkpoint, checkpoint_path)
        config_path = stage / "config.yaml"
        omega_conf.save(config=config, f=config_path, resolve=False)

        source_path = Path(__file__).resolve()
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "profile_id": PROFILE_ID,
            "seed": seed,
            "construction_order": "seed_then_actor_critic_without_environment",
            "environment_constructed": False,
            "destination": str(target),
            "checkpoint": {
                "path": str(final_checkpoint),
                "bytes": checkpoint_path.stat().st_size,
                "file_sha256": _file_sha256(checkpoint_path),
            },
            "config": {
                "path": str(target / "config.yaml"),
                "bytes": config_path.stat().st_size,
                "file_sha256": _file_sha256(config_path),
            },
            "policy": {
                "trainable_parameters": actor_parameters,
                **policy_manifest,
            },
            "value": {
                "trainable_parameters": critic_parameters,
                **value_manifest,
            },
            "profile_config_sha256": profile.config_sha256,
            "model_spec_sha256": _canonical_sha256(model_spec),
            "initializer_source": {
                "path": str(source_path.relative_to(root)),
                "file_sha256": _file_sha256(source_path),
            },
            "runtime": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        (stage / "initialization_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_lite_initialization(target, repo_root=root, expected_seed=seed)


def verify_lite_initialization(
    bundle: str | Path,
    *,
    repo_root: str | Path,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Verify bundle bytes, config/model identity, and every state tensor hash."""

    root = Path(repo_root).resolve()
    directory = Path(bundle).resolve()
    receipt_path = directory / "initialization_receipt.json"
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("kind") != RECEIPT_KIND:
        raise LiteInitializationError("unsupported initialization receipt")
    unsigned = dict(receipt)
    stored_receipt_sha = unsigned.pop("receipt_sha256", None)
    if stored_receipt_sha != _canonical_sha256(unsigned):
        raise LiteInitializationError("initialization receipt self-hash mismatch")
    if expected_seed is not None and receipt.get("seed") != expected_seed:
        raise LiteInitializationError("initialization seed differs from the protocol")
    if receipt.get("profile_id") != PROFILE_ID:
        raise LiteInitializationError("initialization profile is not Lite-S")
    if receipt.get("environment_constructed") is not False:
        raise LiteInitializationError("initialization must precede environment construction")
    if receipt.get("construction_order") != "seed_then_actor_critic_without_environment":
        raise LiteInitializationError("initialization construction order is not frozen")
    if Path(str(receipt.get("destination"))).resolve() != directory:
        raise LiteInitializationError("initialization receipt destination differs")

    checkpoint_path = directory / "last.pt"
    config_path = directory / "config.yaml"
    for label, path in (("checkpoint", checkpoint_path), ("config", config_path)):
        record = receipt.get(label)
        if not isinstance(record, dict):
            raise LiteInitializationError(f"receipt is missing {label} record")
        if Path(str(record.get("path"))).resolve() != path:
            raise LiteInitializationError(f"{label} path differs from receipt")
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise LiteInitializationError(f"{label} byte count differs from receipt")
        if _file_sha256(path) != record.get("file_sha256"):
            raise LiteInitializationError(f"{label} file digest differs from receipt")

    profile = audit_lite_s_profile(root)
    if receipt.get("profile_config_sha256") != profile.config_sha256:
        raise LiteInitializationError("Lite-S source configs changed after initialization")
    source = receipt.get("initializer_source")
    if not isinstance(source, dict):
        raise LiteInitializationError("initializer source receipt is missing")
    source_path = root / str(source.get("path"))
    if _file_sha256(source_path) != source.get("file_sha256"):
        raise LiteInitializationError("initializer source changed after initialization")

    from omegaconf import OmegaConf
    import torch

    try:
        config = OmegaConf.load(config_path)
        if OmegaConf.select(config, "lace_research.profile_id") != PROFILE_ID:
            raise LiteInitializationError("initialization config is not the Lite-S profile")
        if OmegaConf.select(config, "seed") != receipt.get("seed"):
            raise LiteInitializationError("initialization config seed differs from receipt")
        if receipt.get("model_spec_sha256") != _canonical_sha256(_model_spec(config)):
            raise LiteInitializationError("initialization model construction contract differs")
    except LiteInitializationError:
        raise
    except Exception as error:
        raise LiteInitializationError("cannot validate initialization config") from error

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise LiteInitializationError("cannot load initialization checkpoint") from error
    if not isinstance(checkpoint, dict):
        raise LiteInitializationError("initialization checkpoint must contain a mapping")
    required_keys = {
        "policy_state_dict",
        "value_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "state",
        "args",
        "env_state_dict",
        "lace_initialization",
    }
    if set(checkpoint) != required_keys:
        raise LiteInitializationError("initialization checkpoint key set differs")
    if (
        checkpoint["optimizer_state_dict"] is not None
        or checkpoint["lr_scheduler_state_dict"] is not None
    ):
        raise LiteInitializationError("pre-update initialization cannot contain optimizer state")
    if checkpoint["env_state_dict"] != {}:
        raise LiteInitializationError("pre-environment initialization cannot contain env state")
    if getattr(checkpoint["state"], "global_step", None) != 0:
        raise LiteInitializationError("initialization checkpoint global_step must be zero")

    policy_manifest = _state_manifest(checkpoint["policy_state_dict"])
    value_manifest = _state_manifest(checkpoint["value_state_dict"])
    for label, manifest, expected_parameters in (
        ("policy", policy_manifest, profile.actor),
        ("value", value_manifest, profile.critic),
    ):
        record = receipt.get(label)
        if not isinstance(record, dict):
            raise LiteInitializationError(f"receipt is missing {label} state")
        if record.get("trainable_parameters") != expected_parameters:
            raise LiteInitializationError(f"{label} parameter count differs")
        for field in ("tensor_count", "element_count", "tensors", "state_sha256"):
            if record.get(field) != manifest[field]:
                raise LiteInitializationError(f"{label} {field} differs from receipt")

    # Strictly loading the exact models proves this checkpoint is consumable by
    # SONIC's actor and critic constructors, rather than merely shape-counting
    # an arbitrary pair of state dictionaries.
    policy, value_model = _instantiate_lite_models(config)
    try:
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        value_model.load_state_dict(checkpoint["value_state_dict"], strict=True)
    except Exception as error:
        raise LiteInitializationError(
            "initialization state is not strictly loadable by Lite-S"
        ) from error
    lace = checkpoint["lace_initialization"]
    if not isinstance(lace, dict):
        raise LiteInitializationError("checkpoint initialization metadata is missing")
    if lace.get("seed") != receipt["seed"] or lace.get("profile_id") != PROFILE_ID:
        raise LiteInitializationError("checkpoint initialization metadata differs")
    if lace.get("policy_state_sha256") != policy_manifest["state_sha256"]:
        raise LiteInitializationError("checkpoint policy state hash differs")
    if lace.get("value_state_sha256") != value_manifest["state_sha256"]:
        raise LiteInitializationError("checkpoint value state hash differs")
    return receipt
