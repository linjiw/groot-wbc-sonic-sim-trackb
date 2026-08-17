"""Measure whether the corpus covers a behaviour space or repeats one behaviour.

The corpus's stated weakness is that every accepted episode is roughly the same walk. That
claim needs a number, and the obvious number is wrong in a way worth being explicit about.

**Effective rank depends on what you pool over, and the three answers mean different
things.** For a 64-dimensional action:

* *Within-episode* rank averages each episode's own covariance. It asks "how varied is one
  episode?" A single steady walk scores low no matter how many distinct behaviours sit
  beside it in the corpus, so this cannot answer the degeneracy question at all.
* *Between-episode* rank uses one mean action vector per episode. It asks "how many
  distinct behaviours are there?" -- the closest match to the concern, but it discards
  everything about how an episode evolves.
* *Pooled* rank stacks every frame of every episode. It asks "how much of the action space
  does the corpus occupy in total?" This is the number to report for coverage.

Measured on the corpus at the time of writing, these came out 3.48, 4.96 and 7.61. An
earlier datasheet quoted the within-episode figure as the coverage number, which understated
coverage by more than a factor of two while happening to support the same conclusion. All
three are computed and reported here so the distinction cannot quietly collapse again.

Rank alone is also not enough: it is a second-moment summary and says nothing about whether
the corpus contains a crouch. Behaviour spread over interpretable quantities -- speed, path
curvature, root height, tilt -- is reported alongside, because those are what a reader can
check against the videos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeBehaviour:
    """Interpretable behaviour summary for one episode."""

    episode_id: str
    mean_speed_mps: float
    path_length_m: float
    net_displacement_m: float
    #: Path length over straight-line displacement. 1.0 is a straight line.
    tortuosity: float
    #: Net heading change over the episode, in radians.
    heading_change_rad: float
    root_height_min_m: float
    root_height_range_m: float
    max_tilt_rad: float
    #: Mean action vector, used for the between-episode rank.
    action_mean: np.ndarray = field(repr=False)
    #: Rank of this episode's own action covariance.
    within_rank: float = 0.0


@dataclass
class DiversityReport:
    """Corpus-level diversity, with the three ranks kept distinct."""

    episodes: list[EpisodeBehaviour]
    within_episode_rank_mean: float
    between_episode_rank: float
    pooled_rank: float
    action_dim: int
    spreads: dict[str, dict[str, float]]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    def summary_lines(self) -> list[str]:
        lines = [
            f"episodes: {self.episode_count}   action dim: {self.action_dim}",
            f"effective rank  pooled {self.pooled_rank:.2f}   "
            f"between-episode {self.between_episode_rank:.2f}   "
            f"within-episode {self.within_episode_rank_mean:.2f}",
        ]
        for name, stats in self.spreads.items():
            lines.append(
                f"  {name:22s} min {stats['min']:8.3f}  median {stats['median']:8.3f}  "
                f"max {stats['max']:8.3f}  cv {stats['cv']:5.3f}"
            )
        return lines


def effective_rank(matrix: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum.

    ``(sum lambda)^2 / sum(lambda^2)``: 1.0 when all variance sits on one direction, and
    equal to the dimension when the spectrum is flat. Preferred over counting eigenvalues
    above a threshold, which needs an arbitrary cutoff.
    """
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D (samples, features) matrix, got {data.shape}")
    if data.shape[0] < 2:
        return 0.0
    covariance = np.cov(data.T) + 1e-12 * np.eye(data.shape[1])
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0.0)
    total = eigenvalues.sum()
    if total <= 0.0:
        return 0.0
    return float(total**2 / (eigenvalues**2).sum())


def _tilt(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return np.maximum(np.abs(roll), np.abs(pitch))


def _heading_change(path_xy: np.ndarray) -> float:
    """Total signed heading change along the path, unwrapped so turns accumulate."""
    deltas = np.diff(path_xy, axis=0)
    moving = deltas[np.linalg.norm(deltas, axis=1) > 1e-6]
    if len(moving) < 2:
        return 0.0
    headings = np.unwrap(np.arctan2(moving[:, 1], moving[:, 0]))
    return float(headings[-1] - headings[0])


def summarise_episode(episode_id: str, payload: dict) -> EpisodeBehaviour:
    """Reduce one recorded trajectory to its behaviour summary."""
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    action = np.asarray(payload["action_motion_token"], dtype=np.float64)
    fps = float(payload["fps"])
    steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    path_length = float(steps.sum())
    displacement = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    duration = max(len(steps), 1) / fps

    return EpisodeBehaviour(
        episode_id=episode_id,
        mean_speed_mps=path_length / duration,
        path_length_m=path_length,
        net_displacement_m=displacement,
        # A straight line is 1.0; guard the degenerate case of walking back to the start.
        tortuosity=path_length / displacement if displacement > 1e-6 else math.inf,
        heading_change_rad=_heading_change(root[:, :2]),
        root_height_min_m=float(root[:, 2].min()),
        root_height_range_m=float(root[:, 2].max() - root[:, 2].min()),
        max_tilt_rad=float(_tilt(payload["root_quat_w"]).max()),
        action_mean=action.mean(axis=0),
        within_rank=effective_rank(action),
    )


def _spread(values: Sequence[float]) -> dict[str, float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "min": math.nan, "median": math.nan, "max": math.nan, "cv": math.nan}
    mean = float(finite.mean())
    return {
        "n": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        # Coefficient of variation: a scale-free spread, so speed and path length are
        # comparable. This is the number that was 0.14 when the corpus was degenerate.
        "cv": float(finite.std() / abs(mean)) if abs(mean) > 1e-12 else math.nan,
    }


def build_diversity_report(
    episodes: Sequence[EpisodeBehaviour], actions: Sequence[np.ndarray]
) -> DiversityReport:
    """Combine per-episode summaries into the corpus-level report.

    ``actions`` holds each episode's full ``(T, D)`` action block, in the same order as
    ``episodes``; the pooled rank needs every frame, not just the means.
    """
    if len(episodes) != len(actions):
        raise ValueError(
            f"{len(episodes)} episode summaries but {len(actions)} action blocks"
        )
    if not episodes:
        raise ValueError("refusing to report diversity over zero episodes")

    dims = {np.asarray(a).shape[1] for a in actions}
    if len(dims) != 1:
        raise ValueError(f"inconsistent action dimensions across episodes: {sorted(dims)}")

    pooled = np.concatenate([np.asarray(a, dtype=np.float64) for a in actions], axis=0)
    means = np.stack([episode.action_mean for episode in episodes])

    spreads = {
        "mean_speed_mps": _spread([e.mean_speed_mps for e in episodes]),
        "path_length_m": _spread([e.path_length_m for e in episodes]),
        "tortuosity": _spread([e.tortuosity for e in episodes]),
        "heading_change_rad": _spread([e.heading_change_rad for e in episodes]),
        "root_height_min_m": _spread([e.root_height_min_m for e in episodes]),
        "root_height_range_m": _spread([e.root_height_range_m for e in episodes]),
        "max_tilt_rad": _spread([e.max_tilt_rad for e in episodes]),
    }

    return DiversityReport(
        episodes=list(episodes),
        within_episode_rank_mean=float(np.mean([e.within_rank for e in episodes])),
        # Needs at least two episodes to have a between-episode spread at all.
        between_episode_rank=effective_rank(means) if len(episodes) > 1 else 0.0,
        pooled_rank=effective_rank(pooled),
        action_dim=int(pooled.shape[1]),
        spreads=spreads,
    )
