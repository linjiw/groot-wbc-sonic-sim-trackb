"""Tests freezing the SIM-M5 activation-gate rules (plan v1.1 §4, Z6 amendment).

The synthetic fixtures encode the forecast's scenarios so the classifier's
behavior is pinned to what the gate re-freeze promised: a correctly-working
learnability teacher (frontier-loaded, sane posterior, hard-half ratio < 1.5)
must ACTIVATE under the ZPD rule and would have FAILED the old ratio rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.research.classify_m5_activation import (
    _checkpoint_summaries,
    aggregate_arm,
    bin_difficulty_ranks,
    classify_failure_rate_seed,
    classify_m5t_seed,
    classify_m5t_telemetry_points,
    classify_zpd_seed,
    derive_termination_rate_series,
    frontier_tercile_mask,
)

# 12 motions, easiest first; one bin per motion keeps fixtures readable.
_RANKING = [f"m{i:02d}" for i in range(12)]
_BIN_KEYS = list(_RANKING)
_UNIFORM_FRACTIONS = [1.0 / 12.0] * 12
_WORKING_FRACTIONS = [0.04] * 4 + [0.17] * 4 + [0.04] * 4
_DATASET_MANIFEST_SHA256 = "a" * 64
_PAIRED_DATASET_SHA256 = "b" * 64
_EXPECTED_ITERATIONS = 100


def _tripwire_points(values):
    return [[iteration, value] for iteration, value in enumerate(values, start=1)]


def _synthetic_gate(fractions, values=None):
    values = [0.0] * _EXPECTED_ITERATIONS if values is None else values
    return {
        "tripwire_binding_series": _tripwire_points(values),
        "expected_iterations": len(values),
        "sampled_fractions": fractions,
        "synthetic_sampled_count_total": 1_000,
    }


def _sampler_schema() -> dict:
    return {
        "kind": "zpd_adaptive_sampler_state",
        "version": 1,
        "sampling_mass_semantics": "selected_target_bin_pre_window_shift",
    }


def _sampler_config() -> dict:
    return {
        "signal": "learnability",
        "optimism_k": 0.0,
        "evidence_half_life": 10.0,
        "advmass_n": 16,
        "uniform_sampling_rate": 0.1,
        "tripwire_max_prob_over_uniform": 20.0,
        "bin_size": 50,
        "paired_dataset_sha256": _PAIRED_DATASET_SHA256,
    }


def _bins(episodes, failures):
    n = len(episodes)
    prob_uniform = 1.0 / n
    out = []
    for e, f in zip(episodes, failures):
        rate = f / e if e > 0 else 0.0
        out.append(
            {
                "num_episodes": float(e),
                "num_failures": float(f),
                "failure_rate": rate,
                "recomputed_prob_unweighted": prob_uniform,  # overwritten where needed
            }
        )
    return out


def _working_teacher_bins():
    """Mastered easy tercile, ~50% frontier tercile, impossible hard tercile."""
    episodes, failures = [], []
    for i in range(12):
        if i < 4:  # easy: high survival
            episodes.append(100.0)
            failures.append(2.0)
        elif i < 8:  # frontier: ~50%
            episodes.append(100.0)
            failures.append(50.0)
        else:  # impossible: hazard ~1
            episodes.append(100.0)
            failures.append(98.0)
    return _bins(episodes, failures)


def _with_empirical_mass(bins, counts):
    total = sum(counts)
    for bin_record, count in zip(bins, counts, strict=True):
        bin_record["bin_weight"] = 1.0
        bin_record["sampled_count"] = count
        bin_record["sampled_fraction"] = count / total if total else 0.0
    return bins


def test_ranks_and_frontier_tercile() -> None:
    ranks = bin_difficulty_ranks(_BIN_KEYS, _RANKING)
    mask = frontier_tercile_mask(ranks)
    assert mask.sum() == 4
    assert list(np.nonzero(mask)[0]) == [4, 5, 6, 7]


def test_ranks_reject_unknown_motion_keys() -> None:
    with pytest.raises(ValueError, match="absent from the difficulty ranking"):
        bin_difficulty_ranks(["m00", "not_in_ranking"], _RANKING)


def test_working_learnability_teacher_activates_under_zpd_rule() -> None:
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_WORKING_FRACTIONS),
    )
    assert result["verdict"] == "activated"
    ev = result["evidence"]
    assert ev["frontier_over_alloc"] >= 1.2
    assert ev["posterior_spearman_vs_easiness"] >= 0.4
    # The Z6 point: this same working teacher would FAIL the old hard-half rule
    # (utility near-zero on the impossible bins that fill the hard half).
    assert ev["pmax_over_uniform_empirical"] < 10.0  # diffuse, as forecast
    assert ev["sampling_mass_source"] == (
        "explicit_synthetic_selected_target_bin_fraction"
    )


def test_flat_posterior_is_inactive() -> None:
    # No evidence: uniform posterior everywhere -> no targeting, no sanity.
    result = classify_zpd_seed(
        _bins([0.0] * 12, [0.0] * 12),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_UNIFORM_FRACTIONS),
    )
    assert result["verdict"] == "inactive"


def test_insane_posterior_fails_even_if_frontier_loaded() -> None:
    # Difficulty INVERTED vs the ranking (easy motions fail, hard survive):
    # posterior sanity must catch it regardless of allocation shape.
    episodes = [100.0] * 12
    failures = [98.0] * 4 + [50.0] * 4 + [2.0] * 4
    result = classify_zpd_seed(
        _bins(episodes, failures),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_WORKING_FRACTIONS),
    )
    assert result["evidence"]["posterior_spearman_vs_easiness"] < 0
    assert result["verdict"] == "inactive"


def test_tripwire_binding_over_5pct_is_invalid_unstable() -> None:
    series_bad = [0.0] * 90 + [1.0] * 10  # 10% binding
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_WORKING_FRACTIONS, series_bad),
    )
    assert result["verdict"] == "invalid-unstable"
    series_ok = [0.0] * 98 + [1.0] * 2  # 2% binding
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_WORKING_FRACTIONS, series_ok),
    )
    assert result["verdict"] == "activated"


def test_tripwire_exactly_5pct_is_invalid_and_missing_is_incomplete() -> None:
    boundary = [0.0] * 95 + [1.0] * 5
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(_WORKING_FRACTIONS, boundary),
    )
    assert result["verdict"] == "invalid-unstable"

    missing = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        sampled_fractions=_WORKING_FRACTIONS,
        synthetic_sampled_count_total=1_000,
        expected_iterations=_EXPECTED_ITERATIONS,
    )
    assert missing["verdict"] == "incomplete"


def test_zpd_missing_empirical_mass_is_incomplete() -> None:
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
    )
    assert result["verdict"] == "incomplete"
    assert result["evidence"]["sampling_mass_source"] == (
        "empirical_selected_target_bin_draw_counts_pre_window_shift"
    )
    assert "sampled_fraction is missing" in result["evidence"]["reason"]


def test_zpd_zero_empirical_mass_is_incomplete() -> None:
    bins = _with_empirical_mass(_working_teacher_bins(), [0] * 12)
    result = classify_zpd_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
    )
    assert result["verdict"] == "incomplete"
    assert "sampling mass is zero" in result["evidence"]["reason"]


def test_zpd_inconsistent_empirical_mass_is_incomplete() -> None:
    bins = _with_empirical_mass(_working_teacher_bins(), [1] * 12)
    for bin_record, fraction in zip(bins, _WORKING_FRACTIONS, strict=True):
        bin_record["sampled_fraction"] = fraction
    result = classify_zpd_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
    )
    assert result["verdict"] == "incomplete"
    assert "inconsistent with sampled_count" in result["evidence"]["reason"]


def test_zpd_frontier_mass_uses_empirical_draws_not_recompute() -> None:
    # The count-derived posterior looks like a working teacher, but the actual
    # two-level draws were uniform. Recomputing utility would activate this
    # seed; empirical mass correctly leaves it inactive.
    bins = _with_empirical_mass(_working_teacher_bins(), [1] * 12)
    result = classify_zpd_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
    )
    assert result["verdict"] == "inactive"
    assert result["evidence"]["frontier_over_alloc"] == pytest.approx(1.0)
    assert result["evidence"]["sampling_mass_source"] == (
        "empirical_selected_target_bin_draw_counts_pre_window_shift"
    )


@pytest.mark.parametrize(
    ("points", "expected_iterations", "reason"),
    [
        ([[1, 0.0]], 1, "one-point certification"),
        ([[1, 0.0], [3, 0.0]], 3, "contiguous"),
        ([[1, 0.0], [1, 0.0]], 2, "duplicates"),
        ([[1, 0.0], [2, 0.5]], 2, "finite and binary"),
        ([[1, 0.0], [2, float("nan")]], 2, "finite and binary"),
    ],
)
def test_zpd_d9_requires_complete_indexed_binary_telemetry(
    points, expected_iterations: int, reason: str
) -> None:
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=points,
        expected_iterations=expected_iterations,
        sampled_fractions=_WORKING_FRACTIONS,
        synthetic_sampled_count_total=1_000,
    )
    assert result["verdict"] == "incomplete"
    assert reason in result["evidence"]["reason"]


def test_zpd_d9_seed_coverage_error_fails_closed() -> None:
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
        tripwire_seed_coverage_error="D9 telemetry seed coverage mismatch: missing=[2]",
        sampled_fractions=_WORKING_FRACTIONS,
        synthetic_sampled_count_total=1_000,
    )
    assert result["verdict"] == "incomplete"
    assert "seed coverage mismatch" in result["evidence"]["reason"]


def test_flat_posterior_with_ten_bins_per_motion_cannot_activate() -> None:
    # Average-rank Spearman must treat all 120 identical posterior means as tied.
    # The empirical mass deliberately passes the frontier criterion, so posterior
    # sanity is the only thing preventing a false activation.
    keys = [motion for motion in _RANKING for _ in range(10)]
    bins = _bins([0.0] * len(keys), [0.0] * len(keys))
    fractions = [0.004] * 40 + [0.017] * 40 + [0.004] * 40
    result = classify_zpd_seed(
        bins,
        keys,
        _RANKING,
        signal="learnability",
        **_synthetic_gate(fractions),
    )
    assert result["evidence"]["frontier_over_alloc"] > 1.2
    assert result["evidence"]["posterior_spearman_vs_easiness"] == pytest.approx(0.0)
    assert result["verdict"] == "inactive"


def test_zpd_requires_configured_minimum_selected_target_draws() -> None:
    bins = _with_empirical_mass(_working_teacher_bins(), [1] * 12)
    result = classify_zpd_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=_tripwire_points([0.0] * _EXPECTED_ITERATIONS),
        expected_iterations=_EXPECTED_ITERATIONS,
        min_actual_draws=13,
    )
    assert result["verdict"] == "incomplete"
    assert "observed 12, require >= 13" in result["evidence"]["reason"]


def test_failure_rate_rule_unchanged_hard_half_path() -> None:
    bins = _working_teacher_bins()
    # Failure-rate-shaped allocation: mass proportional to failure rate.
    rates = np.asarray([b["failure_rate"] for b in bins])
    prob = rates / rates.sum()
    for b, p in zip(bins, prob):
        b["recomputed_prob_unweighted"] = float(p)
    result = classify_failure_rate_seed(bins, _BIN_KEYS, _RANKING)
    assert result["evidence"]["hard_half_mass_ratio"] >= 1.5
    assert result["verdict"] == "activated"


def test_failure_rate_prefers_telemetry_peakedness() -> None:
    bins = _bins([100.0] * 12, [10.0] * 12)  # flat recompute
    result = classify_failure_rate_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        telemetry_last={"prob_max_over_uniform": 15.0, "num_concentrated_bins": 2.0},
    )
    assert result["evidence"]["peakedness_source"] == "telemetry"
    assert result["verdict"] == "activated"


def test_m5t_band_check() -> None:
    in_band = [0.3] * 60 + [0.05] * 40  # 60% in [0.15, 0.6]
    assert classify_m5t_seed(in_band)["verdict"] == "activated"
    out_of_band = [0.02] * 80 + [0.3] * 20  # schedule never bit
    assert classify_m5t_seed(out_of_band)["verdict"] == "invalid-inactive"
    assert classify_m5t_seed([])["verdict"] == "incomplete"


def _m5t_density_points(values: list[float]) -> list[list[float | int]]:
    return [[iteration, value] for iteration, value in enumerate(values, start=1)]


def _m5t_points_by_key(
    rates: list[float], *, episode_density: float = 0.2
) -> dict[str, list[list[float | int]]]:
    height = [rate * episode_density for rate in rates]
    return {
        "m5t_anchor_pos_termination_density": _m5t_density_points(
            [value * 0.75 for value in height]
        ),
        "m5t_ee_body_pos_termination_density": _m5t_density_points(
            [value * 0.25 for value in height]
        ),
        "m5t_height_termination_density": _m5t_density_points(height),
        "m5t_episode_end_density": _m5t_density_points(
            [episode_density] * len(rates)
        ),
    }


def test_m5t_term_specific_telemetry_requires_exact_finite_iterations() -> None:
    points = _m5t_points_by_key([0.3, 0.3, 0.05, 0.3])
    result = classify_m5t_telemetry_points(points, expected_iterations=4)
    assert result["verdict"] == "activated"
    assert result["evidence"]["telemetry_iterations"] == 4
    assert result["evidence"]["telemetry_source"] == (
        "anchor_pos_or_ee_body_pos_episode_end_fraction"
    )

    gap = _m5t_points_by_key([0.3, 0.3, 0.3, 0.3])
    gap["m5t_height_termination_density"][2][0] = 4
    result = classify_m5t_telemetry_points(gap, expected_iterations=4)
    assert result["verdict"] == "invalid-telemetry"
    assert "ordered and contiguous" in result["evidence"]["reason"]

    non_finite = _m5t_points_by_key([0.3, 0.3])
    non_finite["m5t_episode_end_density"][1][1] = None
    result = classify_m5t_telemetry_points(non_finite, expected_iterations=2)
    assert result["verdict"] == "invalid-telemetry"
    assert "non-numeric" in result["evidence"]["reason"]


def test_m5t_term_specific_telemetry_rejects_missing_and_inconsistent_series() -> None:
    missing = _m5t_points_by_key([0.3, 0.3])
    del missing["m5t_ee_body_pos_termination_density"]
    result = classify_m5t_telemetry_points(missing, expected_iterations=2)
    assert result["verdict"] == "invalid-telemetry"
    assert "is missing" in result["evidence"]["reason"]

    inconsistent = _m5t_points_by_key([0.3, 0.3])
    inconsistent["m5t_height_termination_density"][0][1] = 0.3
    result = classify_m5t_telemetry_points(inconsistent, expected_iterations=2)
    assert result["verdict"] == "invalid-telemetry"
    assert "exceeds episode-end density" in result["evidence"]["reason"]


def test_derive_termination_rate_from_cumulative_counts() -> None:
    eps = [1.0, 2.0, 4.0, 4.0, 8.0]  # per-iteration deltas: 1, 2, 0, 4
    fails = [1.0, 1.5, 2.5, 2.5, 3.5]  # deltas: 0.5, 1.0, 0.0, 1.0
    rates = derive_termination_rate_series(fails, eps)
    assert rates == pytest.approx([0.5, 0.5, 0.25])  # zero-delta iteration skipped


def test_aggregate_arm_two_thirds_rule() -> None:
    ok = {"verdict": "activated"}
    no = {"verdict": "inactive"}
    assert aggregate_arm({0: ok, 1: ok, 2: no})["arm_verdict"] == "activated"
    assert aggregate_arm({0: ok, 1: no, 2: no})["arm_verdict"] == "inactive"
    bad = {"verdict": "invalid-unstable"}
    assert aggregate_arm({0: ok, 1: ok, 2: bad})["arm_verdict"] == "invalid-unstable"
    invalid_inactive = {"verdict": "invalid-inactive"}
    assert aggregate_arm({0: ok, 1: ok, 2: invalid_inactive})["arm_verdict"] == (
        "invalid-inactive"
    )
    incomplete = {"verdict": "incomplete"}
    assert aggregate_arm({0: ok, 1: ok, 2: incomplete})["arm_verdict"] == "incomplete"
    assert aggregate_arm({})["arm_verdict"] == "incomplete"
    assert aggregate_arm({0: ok})["arm_verdict"] == "incomplete"
    assert aggregate_arm({0: ok, 1: ok})["arm_verdict"] == "incomplete"
    unexpected = aggregate_arm({0: ok, 1: ok, 3: no})
    assert unexpected["arm_verdict"] == "incomplete"
    assert unexpected["missing_seeds"] == [2]
    assert unexpected["unexpected_seeds"] == [3]


def _checkpoint_summary(seed: int) -> dict:
    return {
        "checkpoint_path": f"runs/learnability_seed{seed}/last.pt",
        "checkpoint_sha256": f"{seed + 1:064x}",
        "global_step": 2,
        "bins": _with_empirical_mass(
            _working_teacher_bins(), [4] * 4 + [17] * 4 + [4] * 4
        ),
        "bin_motion_keys": _BIN_KEYS,
        "sampler_schema": _sampler_schema(),
        "sampler_config": _sampler_config(),
        "sampling_mass_source": (
            "empirical_selected_target_bin_draw_counts_pre_window_shift"
        ),
        "sampling_mass_semantics": "selected_target_bin_pre_window_shift",
    }


def _required_cli_args() -> list[str]:
    return [
        "--expected-iterations",
        "2",
        "--dataset-manifest-sha256",
        _DATASET_MANIFEST_SHA256,
        "--dataset-manifest-sha256-kind",
        "file_bytes",
        "--paired-dataset-sha256",
        _PAIRED_DATASET_SHA256,
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m5t_cli_uses_term_specific_telemetry_and_non_zpd_provenance(
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "m5t_checkpoint_dump.json"
    dump_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "seed": seed,
                        "checkpoint_sha256": f"{seed + 1:064x}",
                        "global_step": 2,
                        "bins": [],
                    }
                    for seed in (0, 1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_RANKING), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "logs": [
                    {
                        "log_path": f"runs/threshold_schedule_m5t_seed{seed}/train.log",
                        "adaptive_telemetry_present": True,
                        "keys": {
                            key: {"series": points}
                            for key, points in _m5t_points_by_key([0.3, 0.3]).items()
                        },
                    }
                    for seed in (0, 1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "activation.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/classify_m5_activation.py",
            "--arm",
            "m5_t",
            "--checkpoint-dumps",
            str(dump_path),
            "--difficulty-ranking",
            str(ranking_path),
            "--telemetry-summary",
            str(telemetry_path),
            *_required_cli_args(),
            "--output-json",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    classification = json.loads(output_path.read_text(encoding="utf-8"))
    assert classification["arm_verdict"] == "activated"
    assert classification["seed_verdicts"] == {
        "0": "activated",
        "1": "activated",
        "2": "activated",
    }
    for seed_result in classification["seeds"].values():
        assert set(seed_result["provenance"]) == {
            "checkpoint_sha256",
            "checkpoint_global_step",
            "checkpoint_dump_sha256",
            "difficulty_ranking_sha256",
            "telemetry_summary_sha256",
            "dataset_manifest_sha256",
            "dataset_manifest_sha256_kind",
            "paired_dataset_sha256",
        }


def test_checkpoint_loader_accepts_official_wrapper_and_legacy_summary(tmp_path: Path) -> None:
    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(
        json.dumps(
            {
                "kind": "sampler_checkpoint_state_dump",
                "checkpoints": [_checkpoint_summary(0), _checkpoint_summary(1)],
            }
        ),
        encoding="utf-8",
    )
    legacy_path = tmp_path / "legacy_seed2.json"
    legacy_path.write_text(json.dumps(_checkpoint_summary(2)), encoding="utf-8")

    loaded = _checkpoint_summaries([wrapped_path, legacy_path])
    assert set(loaded) == {0, 1, 2}

    with pytest.raises(ValueError, match="duplicate"):
        _checkpoint_summaries([wrapped_path, wrapped_path])


def test_classifier_cli_consumes_official_dump_without_separate_bin_map(tmp_path: Path) -> None:
    dump_path = tmp_path / "checkpoint_dump.json"
    dump_path.write_text(
        json.dumps(
            {
                "kind": "sampler_checkpoint_state_dump",
                "checkpoints": [_checkpoint_summary(seed) for seed in (0, 1, 2)],
            }
        ),
        encoding="utf-8",
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_RANKING), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "logs": [
                    {
                        "log_path": f"runs/learnability_seed{seed}/train.log",
                        "adaptive_telemetry_present": True,
                        "keys": {
                            "tripwire_max_prob_binding": {
                                "series": [[1, 0.0], [2, 0.0]]
                            }
                        },
                    }
                    for seed in (0, 1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "activation.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/classify_m5_activation.py",
            "--arm",
            "m5_l",
            "--checkpoint-dumps",
            str(dump_path),
            "--difficulty-ranking",
            str(ranking_path),
            "--telemetry-summary",
            str(telemetry_path),
            *_required_cli_args(),
            "--output-json",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    classification = json.loads(output_path.read_text(encoding="utf-8"))
    assert classification["arm_verdict"] == "activated"
    assert set(classification["seeds"]) == {"0", "1", "2"}
    assert all(
        seed["evidence"]["sampling_mass_source"]
        == "empirical_selected_target_bin_draw_counts_pre_window_shift"
        for seed in classification["seeds"].values()
    )
    assert classification["expected_iterations"] == 2
    assert classification["seed_verdicts"] == {
        "0": "activated",
        "1": "activated",
        "2": "activated",
    }
    for seed_text, seed_result in classification["seeds"].items():
        seed = int(seed_text)
        provenance = seed_result["provenance"]
        assert provenance["checkpoint_sha256"] == f"{seed + 1:064x}"
        assert provenance["checkpoint_global_step"] == 2
        assert provenance["checkpoint_dump_sha256"] == _sha256(dump_path)
        assert provenance["difficulty_ranking_sha256"] == _sha256(ranking_path)
        assert provenance["telemetry_summary_sha256"] == _sha256(telemetry_path)
        assert provenance["dataset_manifest_sha256"] == _DATASET_MANIFEST_SHA256
        assert provenance["dataset_manifest_sha256_kind"] == "file_bytes"
        assert provenance["paired_dataset_sha256"] == _PAIRED_DATASET_SHA256
        assert provenance["sampler_schema"] == _sampler_schema()
        assert provenance["sampler_config"] == _sampler_config()


def test_classifier_cli_does_not_fallback_when_empirical_mass_is_missing(
    tmp_path: Path,
) -> None:
    summaries = []
    for seed in (0, 1, 2):
        summary = _checkpoint_summary(seed)
        summary["bins"] = _working_teacher_bins()
        summaries.append(summary)
    dump_path = tmp_path / "legacy_checkpoint_dump.json"
    dump_path.write_text(
        json.dumps({"kind": "sampler_checkpoint_state_dump", "checkpoints": summaries}),
        encoding="utf-8",
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_RANKING), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "logs": [
                    {
                        "log_path": f"runs/learnability_seed{seed}/train.log",
                        "adaptive_telemetry_present": True,
                        "keys": {
                            "tripwire_max_prob_binding": {
                                "series": [[1, 0.0], [2, 0.0]]
                            }
                        },
                    }
                    for seed in (0, 1, 2)
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "activation.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/classify_m5_activation.py",
            "--arm",
            "m5_l",
            "--checkpoint-dumps",
            str(dump_path),
            "--difficulty-ranking",
            str(ranking_path),
            "--telemetry-summary",
            str(telemetry_path),
            *_required_cli_args(),
            "--output-json",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    classification = json.loads(output_path.read_text(encoding="utf-8"))
    assert classification["arm_verdict"] == "incomplete"
    assert all(
        "sampled_fraction is missing" in seed["evidence"]["reason"]
        for seed in classification["seeds"].values()
    )


def test_classifier_cli_rejects_incomplete_d9_seed_coverage(tmp_path: Path) -> None:
    dump_path = tmp_path / "checkpoint_dump.json"
    dump_path.write_text(
        json.dumps(
            {
                "kind": "sampler_checkpoint_state_dump",
                "checkpoints": [_checkpoint_summary(seed) for seed in (0, 1, 2)],
            }
        ),
        encoding="utf-8",
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps(_RANKING), encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "logs": [
                    {
                        "log_path": f"runs/learnability_seed{seed}/train.log",
                        "adaptive_telemetry_present": True,
                        "keys": {
                            "tripwire_max_prob_binding": {
                                "series": [[1, 0.0], [2, 0.0]]
                            }
                        },
                    }
                    for seed in (0, 1)
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "activation.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/research/classify_m5_activation.py",
            "--arm",
            "m5_l",
            "--checkpoint-dumps",
            str(dump_path),
            "--difficulty-ranking",
            str(ranking_path),
            "--telemetry-summary",
            str(telemetry_path),
            *_required_cli_args(),
            "--output-json",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    classification = json.loads(output_path.read_text(encoding="utf-8"))
    assert classification["arm_verdict"] == "incomplete"
    assert all(
        "D9 telemetry seed coverage mismatch" in seed["evidence"]["reason"]
        for seed in classification["seeds"].values()
    )
