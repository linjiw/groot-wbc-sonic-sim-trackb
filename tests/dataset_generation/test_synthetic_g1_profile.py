from __future__ import annotations

from gear_sonic.data.features_sonic_vla import (
    SYNTHETIC_G1_ACTION_MODALITIES,
    SYNTHETIC_G1_STATE_MODALITIES,
    get_features_synthetic_g1,
    get_modality_config_synthetic_g1,
)


class StubRobotModel:
    joint_names = [f"joint_{index}" for index in range(43)]
    num_joints = 43

    _groups = {
        "left_leg": range(0, 6),
        "right_leg": range(6, 12),
        "waist": range(12, 15),
        "left_arm": range(15, 22),
        "left_hand": range(22, 29),
        "right_arm": range(29, 36),
        "right_hand": range(36, 43),
    }

    def get_joint_group_indices(self, group: str) -> list[int]:
        return list(self._groups[group])


def test_synthetic_profile_matches_registered_g1_sonic_modalities() -> None:
    modality = get_modality_config_synthetic_g1(StubRobotModel())

    assert tuple(modality["state"]) == SYNTHETIC_G1_STATE_MODALITIES
    assert tuple(modality["action"]) == SYNTHETIC_G1_ACTION_MODALITIES
    assert tuple(modality["video"]) == ("ego_view",)
    assert tuple(modality["annotation"]) == ("human.task_description",)
    assert modality["action"]["motion_token"]["original_key"] == "action.motion_token"


def test_synthetic_features_omit_fake_human_and_vr_fields() -> None:
    features = get_features_synthetic_g1(StubRobotModel())

    assert set(features) == {
        "observation.images.ego_view",
        "observation.state",
        "observation.projected_gravity",
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
        "reference.g1_qpos",
    }
    assert features["reference.g1_qpos"]["shape"] == (36,)
    for key in (
        "observation.state",
        "observation.projected_gravity",
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
        "reference.g1_qpos",
    ):
        assert features[key]["dtype"] == "float32"
    assert not any("smpl" in key or "vr_" in key for key in features)
