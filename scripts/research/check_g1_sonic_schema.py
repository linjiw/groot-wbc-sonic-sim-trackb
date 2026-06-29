#!/usr/bin/env python3
"""Validate local G1 SONIC schema assumptions used by the research plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_GROUPS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)


def check_observation_config(path: Path) -> list[str]:
    import yaml

    errors: list[str] = []
    if not path.exists():
        return [f"missing deploy observation config: {path}"]

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("encoder", {}).get("dimension") != 64:
        errors.append("deploy observation config encoder.dimension is not 64")

    observations = [item.get("name") for item in config.get("observations", [])]
    for required in (
        "token_state",
        "his_base_angular_velocity_10frame_step1",
        "his_body_joint_positions_10frame_step1",
        "his_body_joint_velocities_10frame_step1",
        "his_last_actions_10frame_step1",
        "his_gravity_dir_10frame_step1",
    ):
        if required not in observations:
            errors.append(f"deploy observation config missing observation {required!r}")

    return errors


def main() -> int:
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_g1_robot_model,
        get_modality_config_sonic_vla,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs-config",
        "--observation-config",
        type=Path,
        default=Path("gear_sonic_deploy/policy/release/observation_config.yaml"),
        help="SONIC deploy observation config to validate",
    )
    args = parser.parse_args()

    errors: list[str] = []
    robot_model = get_g1_robot_model()
    features = get_features_sonic_vla(robot_model)
    modality = get_modality_config_sonic_vla(robot_model)

    print(f"robot joints: {robot_model.num_joints}")
    if robot_model.num_joints <= 0:
        errors.append("robot model has no joints")

    for group in EXPECTED_GROUPS:
        try:
            indices = robot_model.get_joint_group_indices(group)
        except Exception as exc:  # noqa: BLE001 - surface schema failures directly.
            errors.append(f"failed to read joint group {group!r}: {exc}")
            continue
        if not indices:
            errors.append(f"joint group {group!r} is empty")
        print(f"{group}: {len(indices)} joints")

    action_features = {
        "action.motion_token": 64,
        "teleop.left_hand_joints": 7,
        "teleop.right_hand_joints": 7,
    }
    for key, expected_dim in action_features.items():
        feature = features.get(key)
        if feature is None:
            errors.append(f"features missing {key}")
            continue
        dim = int(feature.get("shape", [0])[0])
        print(f"{key}: {dim}")
        if dim != expected_dim:
            errors.append(f"{key} dim={dim}, expected {expected_dim}")

    action_modality = modality.get("action", {})
    for key, expected_dim in (
        ("motion_token", 64),
        ("left_hand_joints", 7),
        ("right_hand_joints", 7),
    ):
        entry = action_modality.get(key)
        if entry is None:
            errors.append(f"modality action missing {key}")
            continue
        dim = entry.get("end", 0) - entry.get("start", 0)
        original_key = entry.get("original_key")
        print(f"modality action.{key}: dim={dim}, original_key={original_key}")
        if dim != expected_dim:
            errors.append(f"modality action.{key} dim={dim}, expected {expected_dim}")

    errors.extend(check_observation_config(args.obs_config))

    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    print("G1 SONIC schema validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
