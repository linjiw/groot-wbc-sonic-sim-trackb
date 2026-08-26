"""Coverage-targeted LFH tier-1 proposals with certified delivery bounds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

from .keypoints import SEMANTIC_GROUPS
from .reach import FaceReach
from .window import DEFAULT_MIN_WINDOW_M

COUPLING = {
    ("overhead", "local_crouch"): frozenset({"head_torso"}),
    ("lateral_gap", "local_arm_tuck"): frozenset(
        {"shoulder_left", "shoulder_right", "wrist_left", "wrist_right"}
    ),
}


class ProposalRefusal(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class CoverageTarget:
    behaviour_class: str
    operator: str
    axis_type: str
    binding_keypoint: str
    coordinate_lower_m: float
    coordinate_upper_m: float
    margin_bucket: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CoverageTarget":
        bucket = str(value.get("constraint_coordinate_bucket") or "")
        try:
            if bucket.startswith("under_"):
                lower, upper = 0.0, float(bucket.removeprefix("under_"))
            elif bucket.startswith("at_least_"):
                lower, upper = float(bucket.removeprefix("at_least_")), math.inf
            else:
                lower, upper = (float(item) for item in bucket.split("_", 1))
        except (TypeError, ValueError) as error:
            raise ProposalRefusal("coordinate_bucket_unknown", bucket) from error
        target = cls(
            behaviour_class=str(value.get("edit_behaviour_class") or ""),
            operator=str(value.get("operator") or ""),
            axis_type=str(value.get("constraint_axis") or ""),
            binding_keypoint=str(value.get("binding_keypoint") or ""),
            coordinate_lower_m=lower,
            coordinate_upper_m=upper,
            margin_bucket=str(value.get("margin_bucket") or ""),
        )
        target.validate()
        return target

    def validate(self) -> None:
        allowed = COUPLING.get((self.axis_type, self.operator))
        if allowed is None or self.binding_keypoint not in allowed:
            raise ProposalRefusal(
                "operator_keypoint_mismatch",
                f"{self.operator} cannot target {self.binding_keypoint} on {self.axis_type}",
            )
        if (
            not math.isfinite(self.coordinate_lower_m)
            or self.coordinate_lower_m < 0
            or math.isnan(self.coordinate_upper_m)
            or self.coordinate_upper_m <= self.coordinate_lower_m
        ):
            raise ProposalRefusal("coordinate_bucket_invalid", "upper bound must exceed lower")


@dataclass(frozen=True)
class Tier1Proposal:
    motion_id: str
    target: CoverageTarget
    alpha: float
    coordinate_m: float
    certified_window_m: tuple[float, float]
    per_keypoint_orig_margin_m: dict[str, float]
    per_keypoint_predicted_edit_margin_m: dict[str, float]
    tier: int = 1


def _body_reach(values: Mapping[str, float]) -> float:
    missing = set(SEMANTIC_GROUPS) - set(values)
    if missing:
        raise ProposalRefusal("incomplete_keypoint_screen", f"missing {sorted(missing)}")
    return max(float(values[group]) for group in SEMANTIC_GROUPS)


def minimum_certified_alpha(
    predict_upper_reach: Callable[[float], Mapping[str, float]],
    coordinate_m: float,
    *,
    delta_clear_m: float = 0.0,
    tolerance: float = 1e-4,
    max_alpha: float = 1.0,
) -> float:
    """Smallest alpha whose upper predictive body reach clears the target face."""

    if not math.isfinite(max_alpha) or max_alpha <= 0:
        raise ValueError("max_alpha must be finite and positive")
    threshold = coordinate_m - delta_clear_m
    if _body_reach(predict_upper_reach(0.0)) <= threshold:
        return 0.0
    if _body_reach(predict_upper_reach(max_alpha)) > threshold:
        raise ProposalRefusal("certified_window_empty", "maximum edit does not clear target")
    low, high = 0.0, max_alpha
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _body_reach(predict_upper_reach(middle)) <= threshold:
            high = middle
        else:
            low = middle
    return high


def propose(
    motion_id: str,
    target: CoverageTarget,
    orig: FaceReach,
    predict_upper_reach: Callable[[float], Mapping[str, float]],
    *,
    delta_clear_m: float = 0.0,
    delta_strike_m: float = 0.0,
    min_window_m: float = DEFAULT_MIN_WINDOW_M,
    max_alpha: float = 1.0,
) -> Tier1Proposal:
    """Intersect a DCS coordinate bin with W_cert and run the all-keypoint screen."""

    target.validate()
    if orig.axis_type != target.axis_type:
        raise ProposalRefusal("axis_mismatch", f"{orig.axis_type} != {target.axis_type}")
    if orig.binding_keypoint != target.binding_keypoint:
        raise ProposalRefusal(
            "binding_keypoint_mismatch",
            f"motion binds {orig.binding_keypoint}, target asks for {target.binding_keypoint}",
        )
    if not math.isfinite(max_alpha) or max_alpha <= 0:
        raise ValueError("max_alpha must be finite and positive")
    predicted_full = predict_upper_reach(max_alpha)
    lower = max(target.coordinate_lower_m, _body_reach(predicted_full) + delta_clear_m)
    upper = min(target.coordinate_upper_m, orig.reach_m - delta_strike_m)
    if upper - lower < min_window_m:
        raise ProposalRefusal(
            "certified_window_empty",
            f"bin/window intersection is {(upper - lower) * 1000:.3f} mm",
        )
    coordinate = 0.5 * (lower + upper)
    alpha = minimum_certified_alpha(
        predict_upper_reach,
        coordinate,
        delta_clear_m=delta_clear_m,
        max_alpha=max_alpha,
    )
    predicted = predict_upper_reach(alpha)
    edit_margins = {group: coordinate - float(predicted[group]) for group in SEMANTIC_GROUPS}
    if min(edit_margins.values()) < delta_clear_m - 1e-9:
        raise ProposalRefusal("multi_keypoint_screen", "a non-binding keypoint fails clearance")
    return Tier1Proposal(
        motion_id=motion_id,
        target=target,
        alpha=alpha,
        coordinate_m=coordinate,
        certified_window_m=(lower, upper),
        per_keypoint_orig_margin_m={
            group: coordinate - orig.per_keypoint_reach_m[group] for group in SEMANTIC_GROUPS
        },
        per_keypoint_predicted_edit_margin_m=edit_margins,
    )
