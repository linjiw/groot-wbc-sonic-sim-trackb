"""Split a capture that spans an environment reset into independent passes.

A recorder captures a fixed number of frames. If the environment terminates early it
resets and starts the reference again, and the capture keeps going -- so one file can hold
the tail of one pass and the head of the next, stitched at a discontinuity. Every
downstream quantity computed over such a file is meaningless: path length sums a teleport,
endpoint error compares the wrong endpoints, contact peaks may belong to a different pass.

The acceptance evaluator already refuses these, with the message "motion_time_s is not
monotonic and must be split at reset boundaries". This module is the split it asks for.

Two reasons this is worth doing rather than discarding the file:

* **A reset-spanning capture usually contains one good pass.** Discarding it throws away a
  complete episode because a second, partial one was appended.
* **"Unevaluable" and "rejected" are different facts.** A capture that could not be
  evaluated says nothing about the motion; a rejected one says the motion failed. Counting
  the first as the second inflates the rejection rate and hides the recording bug behind
  what looks like a controller problem.

Splitting is done on ``motion_time_s``, which is the reference clock: it advances with the
motion and restarts at a reset. A drop in that series is the boundary, and it is exact --
no threshold or heuristic is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Payload keys that describe the capture as a whole rather than a per-frame value. These
#: are copied to every segment unchanged; everything else with a matching leading axis is
#: sliced.
_FRAME_COUNT_KEY = "total_frames"


class SegmentError(ValueError):
    """Raised when a capture cannot be split into passes."""


@dataclass(frozen=True)
class SegmentSpan:
    """One pass within a capture, as a half-open frame range."""

    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


def find_reset_boundaries(motion_time_s: np.ndarray) -> list[int]:
    """Frame indices at which a new pass begins.

    A reset is a strict decrease in the reference clock. Equality is not a reset: a paused
    or held reference legitimately repeats a timestamp, and treating that as a boundary
    would shred a stand-still motion into single frames.
    """
    times = np.asarray(motion_time_s, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise SegmentError("motion_time_s is empty")
    return [int(i) + 1 for i in np.flatnonzero(np.diff(times) < 0.0)]


def segment_spans(motion_time_s: np.ndarray) -> list[SegmentSpan]:
    """Split the frame range at every reset boundary."""
    times = np.asarray(motion_time_s, dtype=np.float64).reshape(-1)
    edges = [0, *find_reset_boundaries(times), int(times.size)]
    return [SegmentSpan(start, end) for start, end in zip(edges[:-1], edges[1:]) if end > start]


def spans_a_reset(payload: dict) -> bool:
    """Whether this capture contains more than one pass."""
    return len(find_reset_boundaries(payload["motion_time_s"])) > 0


def split_payload(payload: dict, *, min_frames: int = 40) -> list[dict]:
    """Return one payload per complete pass, longest first.

    Per-frame arrays are sliced; scalars and metadata are copied. ``total_frames`` is
    rewritten so a segment is self-describing rather than inheriting the parent's count --
    a segment carrying the wrong frame count is exactly the kind of quiet inconsistency
    that survives every downstream check.

    Segments shorter than ``min_frames`` are dropped: the tail of a capture is a partial
    pass that was cut by the recorder rather than by the environment, and it has no
    endpoint to compare against.
    """
    times = np.asarray(payload["motion_time_s"], dtype=np.float64).reshape(-1)
    frames = int(times.size)
    spans = segment_spans(times)

    def slice_value(value: Any, span: SegmentSpan) -> Any:
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == frames:
            return value[span.start : span.end]
        if isinstance(value, dict):
            # tracking_metrics and partial_tracking_metrics nest per-frame arrays one
            # level down. Copying them unchanged leaves the segment internally
            # inconsistent, and the acceptance evaluator catches it only because it
            # cross-checks frame counts -- a check not every consumer performs.
            return {name: slice_value(inner, span) for name, inner in value.items()}
        return value

    segments: list[dict] = []
    for span in spans:
        if span.frames < min_frames:
            continue
        segment: dict[str, Any] = {
            key: slice_value(value, span)
            for key, value in payload.items()
            if key != _FRAME_COUNT_KEY
        }
        segment[_FRAME_COUNT_KEY] = span.frames
        segments.append(segment)

    if not segments:
        raise SegmentError(
            f"no pass of at least {min_frames} frames in a {frames}-frame capture "
            f"split into {len(spans)} segment(s) of {[s.frames for s in spans]}"
        )
    # Longest first: the complete pass is the one worth evaluating, and it is the longest
    # unless the environment terminated the first pass early -- in which case neither is
    # complete and the caller should see the longest available rather than the first.
    segments.sort(key=lambda s: -int(s[_FRAME_COUNT_KEY]))
    return segments


def best_evaluable_payload(payload: dict, *, min_frames: int = 40) -> tuple[dict, bool]:
    """The payload to evaluate, and whether a split was needed.

    Returns the input unchanged when the capture holds a single pass, so callers can use
    this unconditionally without paying a copy for the common case.
    """
    if not spans_a_reset(payload):
        return payload, False
    return split_payload(payload, min_frames=min_frames)[0], True
