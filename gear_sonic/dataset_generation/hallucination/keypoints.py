"""Semantic capsule groups extracted from an executed humanoid trajectory.

The LFH "keypoints" are labels over the authoritative collision capsules, not invented
single-centre spheres.  Keeping every capsule makes group extrema attributable without losing
the feet, torso, or link segments that a point proxy would omit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np

from gear_sonic.dataset_generation.sweepcf_coverage import (
    SEMANTIC_GROUP_BY_BODY,
    UNKNOWN,
)
from gear_sonic.dataset_generation.swept_volume import (
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

from .constraint_spec import sha256_file

SEMANTIC_GROUPS = (
    "head_torso",
    "shoulder_left",
    "shoulder_right",
    "wrist_left",
    "wrist_right",
    "pelvis",
    "knee_left",
    "knee_right",
    "foot_left",
    "foot_right",
)


def validate_semantic_partition() -> dict[str, tuple[str, ...]]:
    """Return the accepted owner partition, refusing omissions or duplicate labels."""

    unknown = sorted(set(G1_COLLISION_CAPSULES) - set(SEMANTIC_GROUP_BY_BODY))
    extra = sorted(set(SEMANTIC_GROUP_BY_BODY) - set(G1_COLLISION_CAPSULES))
    invalid = sorted(set(SEMANTIC_GROUP_BY_BODY.values()) - set(SEMANTIC_GROUPS))
    if unknown or extra or invalid:
        raise ValueError(
            "semantic capsule partition is not exact: "
            f"unmapped={unknown}, noncollision={extra}, invalid_groups={invalid}"
        )
    return {
        group: tuple(
            owner for owner, assigned in SEMANTIC_GROUP_BY_BODY.items() if assigned == group
        )
        for group in SEMANTIC_GROUPS
    }


@dataclass(frozen=True)
class SemanticCapsuleTracks:
    """World-space tracks for all capsules, carrying one semantic label per capsule."""

    starts: np.ndarray
    ends: np.ndarray
    radii: np.ndarray
    owners: tuple[str, ...]
    groups: tuple[str, ...]
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray

    def __post_init__(self) -> None:
        frames, capsules, xyz = self.starts.shape
        if self.ends.shape != self.starts.shape or xyz != 3:
            raise ValueError("capsule starts/ends must share shape (T, C, 3)")
        if self.radii.shape != (capsules,):
            raise ValueError("one radius is required per capsule")
        if len(self.owners) != capsules or len(self.groups) != capsules:
            raise ValueError("one owner and semantic group are required per capsule")
        if self.root_pos_w.shape != (frames, 3) or self.root_quat_w.shape != (frames, 4):
            raise ValueError("root tracks must align with capsule frames")
        # The root tracks are validated too: every station selector in this package is an
        # ``argmin`` over root position, and NaN propagates through argmin silently, so one
        # bad root frame is selected as *every* station rather than raising.
        if not all(
            np.isfinite(array).all()
            for array in (self.starts, self.ends, self.root_pos_w, self.root_quat_w)
        ):
            raise ValueError("capsule and root tracks must be finite")
        if set(self.groups) - set(SEMANTIC_GROUPS):
            raise ValueError(
                f"unknown semantic groups: {sorted(set(self.groups) - set(SEMANTIC_GROUPS))}"
            )

    @property
    def frames(self) -> int:
        return int(self.starts.shape[0])

    @property
    def capsules(self) -> int:
        return int(self.starts.shape[1])

    def indices(self, group: str) -> np.ndarray:
        if group not in SEMANTIC_GROUPS:
            raise ValueError(f"unknown semantic group {group!r}")
        return np.asarray([index for index, value in enumerate(self.groups) if value == group])


def extract_keypoints(payload: Mapping[str, object]) -> SemanticCapsuleTracks:
    """Extract the lossless semantic capsule representation from executed body states."""

    validate_semantic_partition()
    starts, ends, radii, owners = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
    )
    groups = tuple(SEMANTIC_GROUP_BY_BODY.get(owner, UNKNOWN) for owner in owners)
    return SemanticCapsuleTracks(
        starts=starts,
        ends=ends,
        radii=radii,
        owners=owners,
        groups=groups,
        root_pos_w=np.asarray(payload["root_pos_w"], dtype=np.float64),
        root_quat_w=np.asarray(payload["root_quat_w"], dtype=np.float64),
    )


@lru_cache(maxsize=32)
def _load_cached(path_text: str, expected_sha256: str) -> SemanticCapsuleTracks:
    path = Path(path_text)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"executed artifact hash mismatch: {actual} != {expected_sha256}")
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return extract_keypoints(payload)


def load_keypoints(path: Path, expected_sha256: str) -> SemanticCapsuleTracks:
    """Hash-check and process one rollout, cached by immutable path/hash identity."""

    return _load_cached(str(path.resolve()), expected_sha256)
