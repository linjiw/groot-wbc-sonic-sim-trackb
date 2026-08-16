#!/usr/bin/env python3
"""Run a real UNITREE_G1_SONIC sample through NVIDIA Isaac-GR00T's loader.

This optional runtime gate is intentionally separate from
``check_sonic_vla_dataset.py``. It requires the pinned ``gr00t`` package, decodes
the ego-view video, instantiates ``ShardedSingleStepDataset``, and extracts one
official 40-step SONIC training sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PINNED_GROOT_REVISION = "ab88b50c718f6528e1df9dcbaf75865d1b604760"
EXPECTED_STATE_SHAPES = {
    "left_leg": (1, 6),
    "right_leg": (1, 6),
    "waist": (1, 3),
    "left_arm": (1, 7),
    "right_arm": (1, 7),
    "left_hand": (1, 7),
    "right_hand": (1, 7),
    "projected_gravity": (1, 3),
}
EXPECTED_ACTION_SHAPES = {
    "motion_token": (40, 64),
    "left_hand_joints": (40, 7),
    "right_hand_joints": (40, 7),
}
EXPECTED_VIDEO_SHAPES = {"ego_view": (1, 480, 640, 3)}


class GrootUnavailableError(RuntimeError):
    """Raised when the optional Isaac-GR00T runtime is unavailable."""


class GrootLoaderSmokeError(RuntimeError):
    """Raised when the real loader cannot produce the expected SONIC sample."""


@dataclass(frozen=True)
class GrootApi:
    embodiment_tag: Any
    modality_configs: dict[str, Any]
    dataset_class: type
    extract_step_data: Any
    module_file: str | None


def load_groot_api() -> GrootApi:
    """Import only the official API needed by this runtime gate."""
    try:
        groot_module = import_module("gr00t")
        configs_module = import_module("gr00t.configs.data.embodiment_configs")
        dataset_module = import_module("gr00t.data.dataset.sharded_single_step_dataset")
        tags_module = import_module("gr00t.data.embodiment_tags")
    except ModuleNotFoundError as exc:
        if exc.name == "gr00t" or (exc.name and exc.name.startswith("gr00t.")):
            raise GrootUnavailableError(
                "NVIDIA Isaac-GR00T is not installed. Install the repository's "
                "pinned Python-3.10 inference extra with "
                "`bash install_scripts/install_inference.sh`. This runtime smoke "
                "is distinct from the file-only dataset validator."
            ) from exc
        raise GrootUnavailableError(
            "NVIDIA Isaac-GR00T was found, but one of its runtime dependencies is "
            f"missing: {exc.name!r}. Reinstall the pinned inference environment."
        ) from exc
    except ImportError as exc:
        raise GrootUnavailableError(
            "NVIDIA Isaac-GR00T could not import its loader API. Reinstall the "
            f"pinned inference environment: {exc}"
        ) from exc

    return GrootApi(
        embodiment_tag=tags_module.EmbodimentTag,
        modality_configs=configs_module.MODALITY_CONFIGS,
        dataset_class=dataset_module.ShardedSingleStepDataset,
        extract_step_data=dataset_module.extract_step_data,
        module_file=getattr(groot_module, "__file__", None),
    )


def get_groot_provenance(module_file: str | None) -> dict[str, Any]:
    """Report the installed VCS revision when direct-URL metadata is available."""
    result: dict[str, Any] = {
        "module_file": module_file,
        "package_version": None,
        "vcs_revision": None,
    }
    try:
        installed = distribution("gr00t")
    except PackageNotFoundError:
        return result

    result["package_version"] = installed.version
    direct_url_text = installed.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {}
        result["vcs_revision"] = direct_url.get("vcs_info", {}).get("commit_id")

    revision = result["vcs_revision"]
    if revision is not None and revision != PINNED_GROOT_REVISION:
        raise GrootUnavailableError(
            "Installed gr00t revision does not match the repository pin: "
            f"installed={revision}, expected={PINNED_GROOT_REVISION}. Reinstall "
            "with `bash install_scripts/install_inference.sh`."
        )
    return result


def _shape_map(values: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(np.asarray(value).shape) for key, value in values.items()}


def _assert_shape_map(
    modality: str,
    actual: dict[str, tuple[int, ...]],
    expected: dict[str, tuple[int, ...]],
) -> None:
    if actual != expected:
        raise GrootLoaderSmokeError(
            f"{modality} shapes do not match UNITREE_G1_SONIC: "
            f"actual={actual}, expected={expected}"
        )


def run_loader_smoke(
    dataset_path: Path,
    *,
    episode_index: int = 0,
    step_index: int = 0,
) -> dict[str, Any]:
    """Load and validate one model-facing sample through official GR00T code."""
    if not dataset_path.is_dir():
        raise GrootLoaderSmokeError(f"dataset directory does not exist: {dataset_path}")
    if episode_index < 0 or step_index < 0:
        raise GrootLoaderSmokeError("episode and step indices must be non-negative")

    api = load_groot_api()
    provenance = get_groot_provenance(api.module_file)
    try:
        tag = api.embodiment_tag.UNITREE_G1_SONIC
    except AttributeError as exc:
        raise GrootUnavailableError(
            "Installed gr00t does not register UNITREE_G1_SONIC; reinstall the "
            f"repository pin {PINNED_GROOT_REVISION}."
        ) from exc

    try:
        modality_config = api.modality_configs[tag.value]
    except KeyError as exc:
        raise GrootUnavailableError(
            f"Installed gr00t has no modality config for {tag.value!r}."
        ) from exc

    action_offsets = list(modality_config["action"].delta_indices)
    if action_offsets != list(range(40)):
        raise GrootLoaderSmokeError(
            "Unexpected UNITREE_G1_SONIC action horizon: "
            f"{action_offsets}; expected offsets 0..39"
        )

    try:
        dataset = api.dataset_class(
            dataset_path=dataset_path,
            embodiment_tag=tag,
            modality_configs=modality_config,
            shard_size=64,
            episode_sampling_rate=1.0,
            seed=0,
            allow_padding=False,
        )
    except Exception as exc:
        raise GrootLoaderSmokeError(
            "ShardedSingleStepDataset construction failed. Confirm meta/stats.json "
            "exists and at least one episode has 40 frames: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    loader = dataset.episode_loader
    if episode_index >= len(loader):
        raise GrootLoaderSmokeError(
            f"episode index {episode_index} is out of range for {len(loader)} episodes"
        )

    try:
        episode = loader[episode_index]
    except Exception as exc:
        raise GrootLoaderSmokeError(
            "GR00T failed to load parquet/video data for episode "
            f"{episode_index}: {type(exc).__name__}: {exc}"
        ) from exc

    max_action_offset = max(action_offsets)
    if step_index + max_action_offset >= len(episode):
        raise GrootLoaderSmokeError(
            f"step {step_index} needs frames through {step_index + max_action_offset}, "
            f"but episode {episode_index} has only {len(episode)} frames"
        )

    try:
        sample = api.extract_step_data(
            episode,
            step_index,
            modality_config,
            tag,
            allow_padding=False,
        )
    except Exception as exc:
        raise GrootLoaderSmokeError(
            "GR00T extract_step_data failed: " f"{type(exc).__name__}: {exc}"
        ) from exc

    state_shapes = _shape_map(sample.states)
    action_shapes = _shape_map(sample.actions)
    video_shapes = _shape_map(sample.images)
    _assert_shape_map("state", state_shapes, EXPECTED_STATE_SHAPES)
    _assert_shape_map("action", action_shapes, EXPECTED_ACTION_SHAPES)
    _assert_shape_map("video", video_shapes, EXPECTED_VIDEO_SHAPES)
    if not isinstance(sample.text, str) or not sample.text.strip():
        raise GrootLoaderSmokeError("GR00T sample has no non-empty task text")

    return {
        "ok": True,
        "gate": "isaac_groot_runtime_loader",
        "dataset": str(dataset_path.resolve()),
        "embodiment": tag.value,
        "episode_index": episode_index,
        "step_index": step_index,
        "action_horizon": len(action_offsets),
        "num_shards": len(dataset),
        "state_shapes": {key: list(value) for key, value in state_shapes.items()},
        "action_shapes": {key: list(value) for key, value in action_shapes.items()},
        "video_shapes": {key: list(value) for key, value in video_shapes.items()},
        "task_text_nonempty": True,
        "gr00t": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="GR00T/LeRobot dataset directory")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        report = run_loader_smoke(
            args.dataset.resolve(),
            episode_index=args.episode_index,
            step_index=args.step_index,
        )
    except GrootUnavailableError as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        return 2
    except GrootLoaderSmokeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
