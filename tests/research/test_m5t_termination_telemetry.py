from __future__ import annotations

import pytest
import torch

from gear_sonic.utils.m5t_telemetry import build_m5t_height_activation_telemetry


def test_m5t_telemetry_preserves_term_specific_masks_and_union() -> None:
    telemetry = build_m5t_height_activation_telemetry(
        {
            "anchor_pos": torch.tensor([True, False, True, False]),
            "ee_body_pos": torch.tensor([False, True, True, False]),
        },
        torch.tensor([True, True, True, False]),
    )

    assert telemetry["adp_samp/m5t_anchor_pos_termination_density"].tolist() == [
        1.0,
        0.0,
        1.0,
        0.0,
    ]
    assert telemetry["adp_samp/m5t_ee_body_pos_termination_density"].tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    assert telemetry["adp_samp/m5t_height_termination_density"].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert telemetry["adp_samp/m5t_episode_end_density"].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]


def test_m5t_telemetry_rejects_missing_or_nonterminating_masks() -> None:
    with pytest.raises(ValueError, match="missing M5-T termination masks"):
        build_m5t_height_activation_telemetry({"anchor_pos": torch.tensor([False])}, torch.tensor([False]))

    with pytest.raises(ValueError, match="without ending the episode"):
        build_m5t_height_activation_telemetry(
            {
                "anchor_pos": torch.tensor([True]),
                "ee_body_pos": torch.tensor([False]),
            },
            torch.tensor([False]),
        )
