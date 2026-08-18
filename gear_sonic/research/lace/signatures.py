"""Construct factorized LACE failure signatures from rollout aggregates."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import beta as beta_distribution

DEFAULT_MECHANISMS = (
    "contact_timing",
    "foot_slip",
    "base_drift",
    "balance_orientation",
    "actuation_saturation",
    "joint_pose_constraint",
)


def _validate_mechanisms(mechanism_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(mechanism_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("mechanism_names must contain non-empty strings")
    if len(names) < 2:
        raise ValueError("mechanism_names must contain at least two channels")
    if len(set(names)) != len(names):
        raise ValueError("mechanism_names must be unique")
    return names


def build_factorized_signatures(
    episodes: Iterable[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str] = DEFAULT_MECHANISMS,
    mechanism_scales: Mapping[str, float] | None = None,
    beta_prior: tuple[float, float] = (0.5, 0.5),
    minimum_resolved_failures: int = 3,
    mechanism_dirichlet_prior: float = 0.5,
    credible_interval_level: float = 0.95,
) -> list[dict[str, Any]]:
    """Aggregate rollout records into scalar difficulty and mechanism type.

    Each episode must provide ``motion_key``, ``probe_policy_id``, ``failed``, and
    ``mechanism_scores``. Scores are nonnegative, already normalized severities
    whose normalizer was fit only on ``D_atlas``. Only failed episodes contribute
    mechanism evidence. A failed episode with all-zero evidence remains a failure
    for difficulty but is marked unresolved rather than assigned a fabricated
    mechanism.

    ``difficulty`` is the empirical failure frequency. The primary joint mechanism
    mass ``f`` uses only attributed failures: ``f = a_resolved * q``. Multiplying
    ``q`` by all failures would silently assume unresolved failures are missing at
    random, so that quantity is exported only as an explicitly named sensitivity.
    Jeffreys-prior difficulty and Dirichlet-regularized mechanism summaries are
    exported separately for uncertainty-aware analyses.
    """

    names = _validate_mechanisms(mechanism_names)
    if mechanism_scales is None:
        scales = {name: 1.0 for name in names}
    else:
        if set(mechanism_scales) != set(names):
            raise ValueError("mechanism_scales must exactly match mechanism_names")
        scales = {}
        for name in names:
            scale = float(mechanism_scales[name])
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"mechanism_scales[{name!r}] must be finite and positive")
            scales[name] = scale
    alpha, beta = (float(beta_prior[0]), float(beta_prior[1]))
    if not (math.isfinite(alpha) and alpha > 0.0 and math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta_prior values must be finite and positive")
    if minimum_resolved_failures < 1:
        raise ValueError("minimum_resolved_failures must be at least one")
    mechanism_prior = float(mechanism_dirichlet_prior)
    if not math.isfinite(mechanism_prior) or mechanism_prior <= 0.0:
        raise ValueError("mechanism_dirichlet_prior must be finite and positive")
    interval_level = float(credible_interval_level)
    if not math.isfinite(interval_level) or not 0.0 < interval_level < 1.0:
        raise ValueError("credible_interval_level must lie strictly between zero and one")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        motion_key = episode.get("motion_key")
        policy_id = episode.get("probe_policy_id")
        if not isinstance(motion_key, str) or not motion_key:
            raise ValueError(f"episodes[{index}].motion_key missing")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError(f"episodes[{index}].probe_policy_id missing")
        if not isinstance(episode.get("failed"), bool):
            raise ValueError(f"episodes[{index}].failed must be bool")
        scores = episode.get("mechanism_scores")
        if not isinstance(scores, Mapping):
            raise ValueError(f"episodes[{index}].mechanism_scores must be a mapping")
        missing = [name for name in names if name not in scores]
        unexpected = [name for name in scores if name not in names]
        if missing or unexpected:
            raise ValueError(
                f"episodes[{index}].mechanism_scores channel mismatch; missing={missing}, unexpected={unexpected}"
            )
        for name in names:
            value = float(scores[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"episodes[{index}].mechanism_scores[{name!r}] is invalid")
        grouped[(motion_key, policy_id)].append(episode)

    if not grouped:
        raise ValueError("episodes must be non-empty")

    signatures: list[dict[str, Any]] = []
    for (motion_key, policy_id), group in sorted(grouped.items()):
        num_rollouts = len(group)
        failed_episodes = [episode for episode in group if episode["failed"]]
        num_failures = len(failed_episodes)
        evidence = [0.0 for _ in names]
        raw_evidence = [0.0 for _ in names]
        resolved_failures = 0
        for episode in failed_episodes:
            raw_episode_scores = [float(episode["mechanism_scores"][name]) for name in names]
            episode_scores = [
                value / scales[name] for name, value in zip(names, raw_episode_scores, strict=True)
            ]
            episode_total = sum(episode_scores)
            if episode_total <= 0.0:
                continue
            resolved_failures += 1
            for index, (raw_value, value) in enumerate(
                zip(raw_episode_scores, episode_scores, strict=True)
            ):
                raw_evidence[index] += raw_value
                evidence[index] += value / episode_total

        difficulty = num_failures / num_rollouts
        attributed_failure_rate = resolved_failures / num_rollouts
        unresolved_failure_probability = (num_failures - resolved_failures) / num_rollouts
        posterior_mean = (num_failures + alpha) / (num_rollouts + alpha + beta)
        evidence_total = sum(evidence)
        if resolved_failures >= minimum_resolved_failures and evidence_total > 0.0:
            q = [value / evidence_total for value in evidence]
            f = [attributed_failure_rate * value for value in q]
            posterior_alpha = [value + mechanism_prior for value in evidence]
            posterior_total = sum(posterior_alpha)
            q_posterior_mean = [value / posterior_total for value in posterior_alpha]
            tail_probability = (1.0 - interval_level) / 2.0
            q_credible_interval = [
                [
                    float(
                        beta_distribution.ppf(
                            tail_probability,
                            value,
                            posterior_total - value,
                        )
                    ),
                    float(
                        beta_distribution.ppf(
                            1.0 - tail_probability,
                            value,
                            posterior_total - value,
                        )
                    ),
                ]
                for value in posterior_alpha
            ]
            f_posterior_mean = [attributed_failure_rate * value for value in q_posterior_mean]
            f_all_failure_mar_sensitivity = [difficulty * value for value in q]
        else:
            q = None
            f = None
            posterior_alpha = None
            q_posterior_mean = None
            q_credible_interval = None
            f_posterior_mean = None
            f_all_failure_mar_sensitivity = None

        signatures.append(
            {
                "motion_key": motion_key,
                "probe_policy_id": policy_id,
                "num_rollouts": num_rollouts,
                "num_failures": num_failures,
                "num_resolved_failures": resolved_failures,
                "difficulty": difficulty,
                "attributed_failure_rate": attributed_failure_rate,
                "unresolved_failure_probability": unresolved_failure_probability,
                "difficulty_posterior_alpha": num_failures + alpha,
                "difficulty_posterior_beta": num_rollouts - num_failures + beta,
                "difficulty_posterior_mean": posterior_mean,
                "q": q,
                "f": f,
                "minimum_resolved_failures": minimum_resolved_failures,
                "mechanism_dirichlet_prior": mechanism_prior,
                "mechanism_posterior_alpha": posterior_alpha,
                "q_posterior_mean": q_posterior_mean,
                "q_credible_interval_level": interval_level,
                "q_credible_interval": q_credible_interval,
                "f_posterior_mean": f_posterior_mean,
                "f_all_failure_mar_sensitivity": f_all_failure_mar_sensitivity,
                "mechanism_scales": [scales[name] for name in names],
                "mechanism_raw_evidence": raw_evidence,
                "mechanism_evidence": evidence,
                "unresolved_failure_rate": (
                    (num_failures - resolved_failures) / num_failures if num_failures else 0.0
                ),
            }
        )
    return signatures
