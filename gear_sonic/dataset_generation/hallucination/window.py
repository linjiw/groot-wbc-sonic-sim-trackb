"""Closed-form LFH interval solver over executed semantic reach."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .keypoints import SEMANTIC_GROUPS
from .reach import FaceReach

DEFAULT_MIN_WINDOW_M = 0.02
DEFAULT_EASY_MARGIN_M = 0.05


class WindowError(ValueError):
    """A proposed face cannot establish the requested counterfactual pattern."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class WindowSolution:
    lower_m: float
    upper_m: float
    width_m: float
    hard_coordinate_m: float
    easy_coordinate_m: float
    binding_keypoint: str
    critical_frame_orig: int
    critical_frame_edit: int
    per_keypoint_margins_m: dict[str, dict[str, float]]


def solve_window(
    orig: FaceReach,
    edit: FaceReach,
    *,
    delta_clear_m: float = 0.0,
    delta_strike_m: float = 0.0,
    min_window_m: float = DEFAULT_MIN_WINDOW_M,
    hard_coordinate_m: float | None = None,
    easy_coordinate_m: float | None = None,
) -> WindowSolution:
    """Solve the necessary/sufficient face interval and attribute every margin."""

    if orig.axis_type != edit.axis_type:
        raise WindowError("axis_mismatch", f"{orig.axis_type} != {edit.axis_type}")
    if min(delta_clear_m, delta_strike_m, min_window_m) < 0:
        raise ValueError("margins and min_window_m must be non-negative")
    lower = edit.reach_m + delta_clear_m
    upper = orig.reach_m - delta_strike_m
    width = upper - lower
    if width < min_window_m:
        raise WindowError(
            "window_below_min",
            f"{width * 1000:.3f} mm is below {min_window_m * 1000:.3f} mm",
        )
    hard = 0.5 * (lower + upper) if hard_coordinate_m is None else hard_coordinate_m
    if hard < lower or hard > upper:
        raise WindowError(
            "hard_face_outside_window",
            f"{hard:.6f} m is outside [{lower:.6f}, {upper:.6f}] m",
        )
    easy = orig.reach_m + DEFAULT_EASY_MARGIN_M if easy_coordinate_m is None else easy_coordinate_m
    if easy <= orig.reach_m:
        raise WindowError(
            "easy_face_not_clear",
            f"{easy:.6f} m does not clear original reach {orig.reach_m:.6f} m",
        )
    # A group that never occupies the face has no margin against it.  Carrying its -inf reach
    # into the subtraction produced +inf margins, which serialize as the non-JSON token
    # ``Infinity`` and are rejected downstream, so no such window could ever populate a spec.
    margins = {}
    for group in SEMANTIC_GROUPS:
        orig_reach = orig.per_keypoint_reach_m.get(group, -math.inf)
        edit_reach = edit.per_keypoint_reach_m.get(group, -math.inf)
        if not (math.isfinite(orig_reach) and math.isfinite(edit_reach)):
            continue
        margins[group] = {"orig_m": hard - orig_reach, "edit_m": hard - edit_reach}
    return WindowSolution(
        lower_m=lower,
        upper_m=upper,
        width_m=width,
        hard_coordinate_m=hard,
        easy_coordinate_m=easy,
        binding_keypoint=orig.binding_keypoint,
        critical_frame_orig=orig.critical_frame,
        critical_frame_edit=edit.critical_frame,
        per_keypoint_margins_m=margins,
    )
