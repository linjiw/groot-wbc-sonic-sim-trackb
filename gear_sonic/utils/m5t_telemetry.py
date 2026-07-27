"""M5-T height-termination activation telemetry helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

M5T_HEIGHT_TERMINATION_TERMS = ("anchor_pos", "ee_body_pos")


def record_m5t_height_termination(env: Any, term_name: str, term_mask: torch.Tensor) -> None:
    """Forward a raw scheduled-term mask to an opt-in environment recorder."""
    recorder = getattr(getattr(env, "wrapper", None), "record_m5t_height_termination", None)
    if callable(recorder):
        recorder(term_name, term_mask)


def build_m5t_height_activation_telemetry(
    term_masks: Mapping[str, torch.Tensor], dones: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return per-environment indicators whose rollout means form M5-T densities.

    The classifier divides the union density by the episode-end density. This
    produces a scheduled-height-termination fraction without bias from rollout
    steps that contain no episode end.
    """
    missing = [name for name in M5T_HEIGHT_TERMINATION_TERMS if name not in term_masks]
    if missing:
        raise ValueError(f"missing M5-T termination masks: {missing}")

    done_mask = dones.bool()
    anchor_mask = term_masks["anchor_pos"].bool()
    ee_mask = term_masks["ee_body_pos"].bool()
    if anchor_mask.shape != done_mask.shape or ee_mask.shape != done_mask.shape:
        raise ValueError("M5-T termination masks and dones must have the same per-environment shape")

    height_mask = anchor_mask | ee_mask
    if torch.any(height_mask & ~done_mask):
        raise ValueError("an M5-T height termination fired without ending the episode")

    return {
        "adp_samp/m5t_anchor_pos_termination_density": anchor_mask.float(),
        "adp_samp/m5t_ee_body_pos_termination_density": ee_mask.float(),
        "adp_samp/m5t_height_termination_density": height_mask.float(),
        "adp_samp/m5t_episode_end_density": done_mask.float(),
    }
