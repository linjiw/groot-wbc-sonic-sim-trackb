"""Validate reset-spanning measurement captures without dropping failed episodes."""

from pathlib import Path
import pickle

from gear_sonic.dataset_generation.trajectory_segments import split_payload
from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory

RESET_ERROR = "motion_time_s is not monotonic and must be split at reset boundaries"


def load_reset_capture(path):
    # Callers bind trusted local recorder files to a registered SHA before loading.
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted hash-bound recorder capture.
    report = validate_sonic_trajectory(payload)
    if report.ok:
        return payload
    if set(report.errors) != {RESET_ERROR}:
        raise ValueError(f"invalid measurement capture: {report.errors}")
    # Validate all segments, including short tails; never choose the best episode.
    segments = split_payload(payload, min_frames=1)
    if sum(s["total_frames"] for s in segments) != payload["total_frames"]:
        raise ValueError("reset split lost frames")
    for segment in segments:
        checked = validate_sonic_trajectory(segment)
        if not checked.ok:
            raise ValueError(f"invalid reset-separated segment: {checked.errors}")
    return payload
