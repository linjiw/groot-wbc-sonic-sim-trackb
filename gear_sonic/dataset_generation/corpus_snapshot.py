"""One immutable manifest every reported statistic must come from.

Three separate stale figures shipped in review pages built from ad-hoc passes over the
trajectory store, each computed independently and each free to disagree with the others. The
fix is not more care: it is a single snapshot that every consumer reads, so a number can only
enter a report by being in here first.

The snapshot also makes explicit two things that per-episode counting hides:

* **Episodes are not distinct behaviours.** Measured, 164 evaluable episodes carry only 131
  distinct executed state trajectories -- the same motion rolled out in different rooms
  produces different pixels and a bit-identical trajectory. Counting those as independent
  evidence inflates every per-family rate and any diversity claim built on episode counts.
* **Small denominators are not results.** A family at 2/2 is not "solved". Every rate carries
  a Wilson interval, which is wide exactly where the sample is thin.

A snapshot is named and frozen. Any gate change produces a new one rather than overwriting,
because a verdict without the rules that produced it cannot be compared across time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Bumped whenever the schema of a snapshot record changes.
SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotError(ValueError):
    """Raised when a snapshot cannot be built or is internally inconsistent."""


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Preferred over the normal approximation because the interesting cases here are exactly
    where it fails: small n, and rates at 0 or 1. A family at 2/2 gets [0.34, 1.00] rather
    than the [1.00, 1.00] a naive interval would give.
    """
    if trials <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > trials:
        raise SnapshotError(f"{successes} successes out of {trials} trials is impossible")
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    spread = (
        z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    ) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def trajectory_fingerprint(payload: dict) -> str:
    """A hash of the executed state, so identical rollouts collapse to one behaviour.

    Root pose and joint positions only: two episodes differing solely in which room they
    were rendered in have identical state and must not count as two behaviours.
    """
    parts: list[bytes] = []
    for key in ("root_pos_w", "root_quat_w", "dof_pos"):
        if key in payload:
            parts.append(
                np.ascontiguousarray(np.asarray(payload[key], dtype=np.float64)).tobytes()
            )
    if not parts:
        raise SnapshotError("payload carries no state arrays to fingerprint")
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()[:16]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class EpisodeRecord:
    """Everything about one episode that any report is allowed to quote."""

    episode_id: str
    trajectory_path: str
    trajectory_sha256: str
    trajectory_fingerprint: str
    frames: int
    fps: float
    outcome: str
    rejection_reasons: tuple[str, ...]
    demoted_failures: tuple[str, ...]
    errors: tuple[str, ...]
    recovered_from_split: bool
    policy: str
    behaviour: str
    speed_style: str
    video_path: str | None
    video_sha256: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_video(self) -> bool:
        return self.video_path is not None

    @property
    def evaluated(self) -> bool:
        return self.outcome in ("accepted", "rejected")


@dataclass
class CorpusSnapshot:
    """A frozen view of the corpus, with the counts that matter kept separate."""

    name: str
    gate_version: str
    records: list[EpisodeRecord]
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    @property
    def episodes(self) -> int:
        return len(self.records)

    @property
    def evaluable(self) -> list[EpisodeRecord]:
        return [r for r in self.records if r.evaluated]

    @property
    def accepted(self) -> list[EpisodeRecord]:
        return [r for r in self.records if r.outcome == "accepted"]

    @property
    def distinct_trajectories(self) -> int:
        """Episodes that are actually different. The headline number should be this one."""
        return len({r.trajectory_fingerprint for r in self.evaluable})

    @property
    def duplicate_episodes(self) -> int:
        return len(self.evaluable) - self.distinct_trajectories

    @property
    def missing_clips(self) -> list[EpisodeRecord]:
        """Episodes with no video. Named, because a reviewer should never have to guess
        which episode the gap between an episode count and a clip count refers to."""
        return [r for r in self.records if not r.has_video]

    def family_rates(self) -> dict[str, dict[str, Any]]:
        """Per-behaviour acceptance with its uncertainty and its duplicate count."""
        groups: dict[str, list[EpisodeRecord]] = {}
        for record in self.evaluable:
            groups.setdefault(record.behaviour, []).append(record)

        out: dict[str, dict[str, Any]] = {}
        for behaviour, group in groups.items():
            accepted = sum(1 for r in group if r.outcome == "accepted")
            low, high = wilson_interval(accepted, len(group))
            out[behaviour] = {
                "episodes": len(group),
                "distinct_trajectories": len({r.trajectory_fingerprint for r in group}),
                "accepted": accepted,
                "rate": accepted / len(group),
                "wilson_low": low,
                "wilson_high": high,
                # A rate whose interval spans more than half the unit line is not a result.
                "informative": (high - low) < 0.5,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        families = self.family_rates()
        return {
            "schema_version": self.schema_version,
            "snapshot": self.name,
            "gate_version": self.gate_version,
            "counts": {
                "episodes": self.episodes,
                "evaluable": len(self.evaluable),
                "unevaluable": self.episodes - len(self.evaluable),
                "accepted": len(self.accepted),
                "distinct_executed_trajectories": self.distinct_trajectories,
                "duplicate_episodes": self.duplicate_episodes,
                "with_video": sum(1 for r in self.records if r.has_video),
                "missing_video": len(self.missing_clips),
            },
            "missing_clips": [
                {"episode_id": r.episode_id, "outcome": r.outcome} for r in self.missing_clips
            ],
            "families": families,
            "uninformative_families": sorted(
                name for name, stats in families.items() if not stats["informative"]
            ),
            "episodes": [
                {
                    "episode_id": r.episode_id,
                    "trajectory_path": r.trajectory_path,
                    "trajectory_sha256": r.trajectory_sha256,
                    "trajectory_fingerprint": r.trajectory_fingerprint,
                    "frames": r.frames,
                    "fps": r.fps,
                    "outcome": r.outcome,
                    "rejection_reasons": list(r.rejection_reasons),
                    "demoted_failures": list(r.demoted_failures),
                    "errors": list(r.errors),
                    "recovered_from_split": r.recovered_from_split,
                    "policy": r.policy,
                    "behaviour": r.behaviour,
                    "speed_style": r.speed_style,
                    "video_path": r.video_path,
                    "video_sha256": r.video_sha256,
                    "diagnostics": r.diagnostics,
                }
                for r in self.records
            ],
        }


def check_consistency(snapshot: CorpusSnapshot) -> list[str]:
    """Internal contradictions a report built from this snapshot would inherit."""
    problems: list[str] = []
    counts = snapshot.to_dict()["counts"]
    if counts["accepted"] > counts["evaluable"]:
        problems.append("more accepted episodes than evaluable ones")
    if counts["with_video"] + counts["missing_video"] != counts["episodes"]:
        problems.append("video counts do not sum to the episode count")
    if counts["distinct_executed_trajectories"] > counts["evaluable"]:
        problems.append("more distinct trajectories than evaluable episodes")
    for behaviour, stats in snapshot.family_rates().items():
        if stats["accepted"] > stats["episodes"]:
            problems.append(f"{behaviour}: accepted exceeds episodes")
    return problems


def build_snapshot(
    name: str,
    gate_version: str,
    records: Iterable[EpisodeRecord],
) -> CorpusSnapshot:
    snapshot = CorpusSnapshot(name=name, gate_version=gate_version, records=list(records))
    if not snapshot.records:
        raise SnapshotError("refusing to build an empty snapshot")
    problems = check_consistency(snapshot)
    if problems:
        raise SnapshotError("snapshot is internally inconsistent: " + "; ".join(problems))
    return snapshot
