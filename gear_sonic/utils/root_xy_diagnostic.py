"""Pure helpers for optional root-XY evaluation diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

ROOT_XY_PER_MOTION_FIELDS = (
    "root_xy_error_mean_m",
    "root_xy_error_p95_m",
    "root_xy_error_max_m",
    "root_xy_guard_hit",
    "root_xy_first_crossing_frame",
    "root_xy_first_crossing_progress",
)


def validate_root_xy_threshold(value: float) -> float:
    """Return a finite, positive diagnostic threshold."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("root XY diagnostic threshold must be a real number")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("root XY diagnostic threshold must be finite and positive")
    return threshold


def root_xy_distance(reference_pos_w: torch.Tensor, robot_pos_w: torch.Tensor) -> torch.Tensor:
    """Compute horizontal world-frame distance for corresponding anchor positions."""
    if reference_pos_w.shape != robot_pos_w.shape:
        raise ValueError("reference and robot anchor tensors must have identical shapes")
    if reference_pos_w.ndim < 1 or reference_pos_w.shape[-1] < 2:
        raise ValueError("anchor tensors must have at least X and Y coordinates")
    return (reference_pos_w[..., :2] - robot_pos_w[..., :2]).norm(dim=-1)


def root_xy_guard_exceeded(error_m: torch.Tensor, *, threshold_m: float) -> torch.Tensor:
    """Apply the strict root-XY guard threshold used by the termination term."""
    threshold = validate_root_xy_threshold(threshold_m)
    return error_m.gt(threshold)


def summarize_root_xy_error_batch(
    error_by_step: torch.Tensor,
    motion_num_steps: Sequence[int] | torch.Tensor,
    *,
    threshold_m: float,
) -> dict[str, torch.Tensor]:
    """Summarize observed root-XY errors for each motion in an eval batch.

    The evaluated sequence mirrors ``ImEvalCallback``: at most ``num_steps - 1``
    recorded samples are used. Percentiles use linear interpolation. Crossing
    frames are zero-based; ``-1`` (and progress ``-1.0``) denotes no strict
    ``error > threshold`` crossing.
    """
    threshold = validate_root_xy_threshold(threshold_m)
    if error_by_step.ndim != 2:
        raise ValueError("root XY error batch must have shape (steps, motions)")

    if isinstance(motion_num_steps, torch.Tensor):
        steps_per_motion = motion_num_steps.detach().cpu().tolist()
    else:
        steps_per_motion = list(motion_num_steps)
    if len(steps_per_motion) != error_by_step.shape[1]:
        raise ValueError("motion step counts must match the error batch width")

    rows: dict[str, list[float | int | bool]] = {
        field: [] for field in ROOT_XY_PER_MOTION_FIELDS
    }
    errors_cpu = error_by_step.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(errors_cpu).all():
        raise ValueError("root XY errors must all be finite")

    for motion_index, raw_num_steps in enumerate(steps_per_motion):
        if isinstance(raw_num_steps, bool) or int(raw_num_steps) != raw_num_steps:
            raise ValueError("motion step counts must be integers")
        intended_samples = int(raw_num_steps) - 1
        if intended_samples <= 0:
            raise ValueError("each motion must contain at least two frames")
        sequence = errors_cpu[:intended_samples, motion_index]
        if sequence.numel() == 0:
            raise ValueError("each motion must have at least one observed diagnostic sample")

        crossings = torch.nonzero(sequence > threshold, as_tuple=False).flatten()
        guard_hit = bool(crossings.numel())
        first_frame = int(crossings[0].item()) if guard_hit else -1
        first_progress = (
            float(first_frame + 1) / float(intended_samples) if guard_hit else -1.0
        )
        rows["root_xy_error_mean_m"].append(float(sequence.mean().item()))
        rows["root_xy_error_p95_m"].append(
            float(torch.quantile(sequence, 0.95, interpolation="linear").item())
        )
        rows["root_xy_error_max_m"].append(float(sequence.max().item()))
        rows["root_xy_guard_hit"].append(guard_hit)
        rows["root_xy_first_crossing_frame"].append(first_frame)
        rows["root_xy_first_crossing_progress"].append(first_progress)

    return {
        "root_xy_error_mean_m": torch.tensor(rows["root_xy_error_mean_m"], dtype=torch.float64),
        "root_xy_error_p95_m": torch.tensor(rows["root_xy_error_p95_m"], dtype=torch.float64),
        "root_xy_error_max_m": torch.tensor(rows["root_xy_error_max_m"], dtype=torch.float64),
        "root_xy_guard_hit": torch.tensor(rows["root_xy_guard_hit"], dtype=torch.bool),
        "root_xy_first_crossing_frame": torch.tensor(
            rows["root_xy_first_crossing_frame"], dtype=torch.int64
        ),
        "root_xy_first_crossing_progress": torch.tensor(
            rows["root_xy_first_crossing_progress"], dtype=torch.float64
        ),
    }
