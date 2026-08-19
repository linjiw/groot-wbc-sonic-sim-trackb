"""Tests for the hygiene x sampler matrix adjudicator.

WHY these tests exist: the script they cover is written BEFORE the experiment runs, so nothing in
the real world can tell us whether its decision rule works. The synthetic dry-run is therefore the
only evidence that exists, and these tests are what keep it honest -- they re-derive each branch's
expected verdict here rather than trusting the script's own internal assertions, and they pin the
two pieces of machinery a reviewer is most likely to distrust: how the strata are cut, and what the
permutation test actually computes.

CPU only; no Isaac Lab, no GPU, no training log, no real motion bank.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.research.analyze_hygiene_matrix import (
    BASELINE_ARM,
    CONCENTRATION_RATIO,
    DEFAULT_SEED,
    EASY_DRIFT_TOL,
    EASY_SURVIVAL,
    FLAG_INFEASIBLE_FRAC,
    MIN_WORST_DECILE_EFFECT,
    REQUIRED_ARMS,
    SYNTHETIC_BRANCHES,
    SYNTHETIC_EXPECTATIONS,
    AnalysisInputError,
    analyze,
    build_strata,
    classify_direction,
    exposure_metrics,
    load_case,
    materialize_branch,
    read_stratified,
    render_markdown,
    run_synthetic,
    sign_flip_permutation_p,
    synthetic_eval_bank,
)

#: Small enough to keep the suite quick, large enough that the Monte-Carlo floor is not the story.
TEST_NPERM = 400


# --------------------------------------------------------------------------- the four branches


@pytest.fixture(scope="module")
def branch_reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Run every fabricated branch once; the branch tests read the cached reports."""
    root = tmp_path_factory.mktemp("hygiene_matrix_branches")
    reports = {}
    for branch, _blurb, effects in SYNTHETIC_BRANCHES:
        paths = materialize_branch(root / branch, effects)
        reports[branch] = (
            effects,
            analyze(**load_case(paths), nperm=TEST_NPERM, seed=DEFAULT_SEED),
        )
    return reports


@pytest.mark.parametrize("branch", [name for name, _, _ in SYNTHETIC_BRANCHES])
def test_every_fabricated_branch_reaches_the_designed_verdict(branch, branch_reports) -> None:
    """The four required branches plus the two that catch the sealed rule's blind spots."""
    effects, report = branch_reports[branch]
    for arm, effect in effects.items():
        want_pass, want_direction = SYNTHETIC_EXPECTATIONS[effect]
        conc = report["arms"][arm]["concentration"]
        assert bool(conc["pass"]) is want_pass, (branch, arm, conc)
        assert conc["direction"] == want_direction, (branch, arm, conc)
        assert report["decision"]["hygiene_claim_passes"][arm] is want_pass
        assert report["decision"]["direction"][arm] == want_direction


def test_hygiene_works_branch_lifts_the_worst_and_leaves_the_easy_stratum_alone(
    branch_reports,
) -> None:
    _effects, report = branch_reports["i_hygiene_works"]
    conc = report["arms"]["B_pruned"]["concentration"]
    assert conc["delta_worst_decile"] >= CONCENTRATION_RATIO * conc["delta_best_half"]
    assert conc["delta_worst_decile"] >= MIN_WORST_DECILE_EFFECT
    assert abs(conc["delta_easy"]) <= EASY_DRIFT_TOL
    assert "C_repaired preserves N" in report["decision"]["headline"]


def test_volume_confound_branch_fails_on_the_ratio_not_on_the_effect_size(branch_reports) -> None:
    """The whole point of the design: a uniform lift is a real lift and still must not pass."""
    _effects, report = branch_reports["ii_volume_confound"]
    conc = report["arms"]["B_pruned"]["concentration"]
    assert conc["materiality_pass"] is True  # the arm really did get better everywhere
    assert conc["ratio_pass"] is False  # ... uniformly, which is the confound's signature
    assert conc["easy_flat_pass"] is False
    assert conc["pass"] is False


def test_null_branch_is_reported_as_a_null_and_not_as_harm(branch_reports) -> None:
    _effects, report = branch_reports["iii_null"]
    conc = report["arms"]["C_repaired"]["concentration"]
    assert abs(conc["delta_worst_decile"]) < MIN_WORST_DECILE_EFFECT
    assert conc["direction"] == "null"
    assert "no arm shows the concentration signature" in report["decision"]["headline"]


def test_harm_branch_reports_the_sign(branch_reports) -> None:
    _effects, report = branch_reports["iv_harm"]
    conc = report["arms"]["D_raw_uniform"]["concentration"]
    assert conc["delta_worst_decile"] < 0
    assert conc["direction"] == "harm"
    assert conc["pass"] is False


def test_perverse_ratio_branch_is_caught_by_the_materiality_floor_alone(branch_reports) -> None:
    """Sealed rule as literally written would pass this; the added floor is what refuses it."""
    _effects, report = branch_reports["vi_perverse_ratio"]
    conc = report["arms"]["B_pruned"]["concentration"]
    assert conc["ratio_pass"] is True  # -0.05 >= 2 * -0.10 is arithmetically true
    assert conc["materiality_pass"] is False
    assert conc["pass"] is False
    assert conc["direction"] == "harm"


def test_arms_do_not_contaminate_each_other_inside_one_matrix(branch_reports) -> None:
    _effects, report = branch_reports["v_mixed_matrix"]
    directions = report["decision"]["direction"]
    assert directions == {
        "B_pruned": "improves-worst",
        "C_repaired": "null",
        "D_raw_uniform": "uniform-lift",
        "E_raw_cap": "harm",
    }
    assert report["decision"]["headline"].endswith("B_pruned")


def test_split_half_control_actually_reranks(branch_reports) -> None:
    """A split-half column that reproduces the sealed column exactly is not a control."""
    _effects, report = branch_reports["i_hygiene_works"]
    record = report["arms"]["B_pruned"]
    assert record["concentration_split_half"] is not None
    assert record["regression_to_mean_gap"] > 0.0
    assert (
        record["concentration_split_half"]["delta_worst_decile"]
        < record["concentration"]["delta_worst_decile"]
    )


# --------------------------------------------------------------------------- output contract


def test_report_is_json_serializable_and_declares_its_inference_limits(branch_reports) -> None:
    _effects, report = branch_reports["i_hygiene_works"]
    round_tripped = json.loads(json.dumps(report))
    assert round_tripped["analysis"]["rng"] == f"numpy.random.default_rng({DEFAULT_SEED})"
    assert round_tripped["analysis"]["nperm"] == TEST_NPERM
    assert "NO seed-level inference" in round_tripped["inference_note"]
    assert round_tripped["design"]["reported_not_adjudicated"] == [
        "exposure ledger",
        "torque health",
    ]
    assert len(round_tripped["caveats"]) >= 4


def test_markdown_names_every_arm_and_its_confound_status(branch_reports) -> None:
    _effects, report = branch_reports["i_hygiene_works"]
    markdown = render_markdown(report)
    for arm in REQUIRED_ARMS:
        assert f"`{arm}`" in markdown
    assert "volume-controlled" in markdown  # C_repaired
    assert "volume-confounded" in markdown  # B_pruned
    assert "Exposure ledger (reported, not adjudicated)" in markdown
    assert "Torque health (reported, not adjudicated)" in markdown
    assert "What a reviewer should challenge" in markdown


# --------------------------------------------------------------------------- refusals


@pytest.fixture(scope="module")
def guard_paths(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("hygiene_matrix_guards")
    return materialize_branch(root / "base", dict(SYNTHETIC_BRANCHES[0][2]))


def test_missing_arm_is_a_loud_error_not_a_dropped_column(guard_paths) -> None:
    kwargs = load_case(guard_paths)
    for table in ("stratified", "exposure", "torque"):
        kwargs[table] = {a: v for a, v in kwargs[table].items() if a != "C_repaired"}
    with pytest.raises(AnalysisInputError, match="incomplete matrix"):
        analyze(**kwargs, nperm=TEST_NPERM)


def test_partial_exposure_ledger_is_refused(guard_paths) -> None:
    kwargs = load_case(guard_paths)
    kwargs["exposure"] = {a: v for a, v in kwargs["exposure"].items() if a != "E_raw_cap"}
    with pytest.raises(AnalysisInputError, match="exposure ledger given for"):
        analyze(**kwargs, nperm=TEST_NPERM)


def test_missing_eval_screen_is_refused_unless_the_choice_is_recorded(guard_paths) -> None:
    with pytest.raises(AnalysisInputError, match="no --eval-screen"):
        analyze(**load_case(guard_paths, eval_flags=None), nperm=TEST_NPERM)
    report = analyze(
        **load_case(guard_paths, eval_flags=None), assume_all_feasible=True, nperm=TEST_NPERM
    )
    assert "ASSUMED all-feasible" in report["eval_set"]["feasibility_source"]


def test_repaired_eval_bank_is_refused(tmp_path: Path) -> None:
    """Grading the repair on repaired references would be a self-serving endpoint."""
    paths = materialize_branch(
        tmp_path / "repaired_eval",
        dict(SYNTHETIC_BRANCHES[0][2]),
        eval_bank_overrides={"C_repaired": "repaired_heldout_v1"},
    )
    with pytest.raises(AnalysisInputError, match="self-serving endpoint"):
        analyze(**load_case(paths), nperm=TEST_NPERM)


def test_ragged_offset_grid_is_refused_by_the_reader(tmp_path: Path) -> None:
    paths = materialize_branch(
        tmp_path / "ragged", dict(SYNTHETIC_BRANCHES[0][2]), ragged_arm="B_pruned"
    )
    with pytest.raises(AnalysisInputError, match="missing start offsets"):
        read_stratified(paths["strat"]["B_pruned"])


def test_mismatched_clip_sets_are_refused_but_can_be_intersected_on_purpose(tmp_path: Path) -> None:
    paths = materialize_branch(
        tmp_path / "mismatch", dict(SYNTHETIC_BRANCHES[0][2]), drop_clip_from="B_pruned"
    )
    with pytest.raises(AnalysisInputError, match="different clip set"):
        analyze(**load_case(paths), nperm=TEST_NPERM)
    report = analyze(**load_case(paths), allow_clip_mismatch=True, nperm=TEST_NPERM)
    assert report["eval_set"]["clip_sets_intersected"] is True


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AnalysisInputError, match="missing input file"):
        read_stratified(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- strata construction


def test_worst_decile_and_best_half_follow_the_sealed_definition() -> None:
    baseline = {f"c{i:02d}": i / 100.0 for i in range(40)}  # 0.00 .. 0.39, already ascending
    strata = build_strata(baseline)
    assert len(strata.feasible) == 40
    assert strata.worst_decile == tuple(f"c{i:02d}" for i in range(4))  # max(1, 40 // 10)
    assert strata.best_half == tuple(f"c{i:02d}" for i in range(20, 40))  # order[n // 2 :]
    assert strata.easy == ()  # nothing reaches 0.95


def test_worst_decile_is_never_empty_for_a_tiny_eval_set() -> None:
    strata = build_strata({"only": 0.4})
    assert strata.worst_decile == ("only",)
    assert strata.best_half == ("only",)


def test_easy_stratum_is_exactly_the_clips_at_or_above_the_cut() -> None:
    baseline = {"a": 0.10, "b": EASY_SURVIVAL - 1e-9, "c": EASY_SURVIVAL, "d": 0.99}
    strata = build_strata(baseline)
    assert set(strata.easy) == {"c", "d"}


def test_flagged_clips_are_excluded_from_the_concentration_support() -> None:
    baseline = {f"c{i:02d}": i / 100.0 for i in range(20)}
    flags = {"c00": FLAG_INFEASIBLE_FRAC + 0.01, "c01": FLAG_INFEASIBLE_FRAC}
    strata = build_strata(baseline, infeasible_frac=flags)
    assert strata.infeasible == ("c00",)  # strictly greater than the threshold
    assert "c01" in strata.feasible  # exactly at the threshold is still feasible
    assert "c00" not in strata.worst_decile


def test_all_flagged_eval_set_is_refused_rather_than_silently_empty() -> None:
    baseline = {"a": 0.1, "b": 0.2}
    with pytest.raises(AnalysisInputError, match="no support"):
        build_strata(baseline, infeasible_frac={"a": 0.9, "b": 0.9})


def test_ranking_may_come_from_a_disjoint_offset_split() -> None:
    """The split-half control: rank on one signal, score on another."""
    scored = {"a": 0.9, "b": 0.1}
    ranking = {"a": 0.1, "b": 0.9}  # deliberately the reverse order
    strata = build_strata(scored, ranking_means=ranking)
    assert strata.worst_decile == ("a",)  # chosen by the RANKING, not by what it will be scored on


def test_ranking_must_cover_every_scored_clip() -> None:
    with pytest.raises(AnalysisInputError, match="ranking is missing"):
        build_strata({"a": 0.5, "b": 0.5}, ranking_means={"a": 0.5})


def test_ground_contact_stratum_is_membership_not_a_survival_cut() -> None:
    baseline = {"kneel": 0.2, "walk": 0.8}
    strata = build_strata(baseline, ground_clips=["kneel", "not_in_eval"])
    assert strata.ground_contact == ("kneel",)


def test_strata_are_defined_by_the_baseline_for_every_arm(branch_reports) -> None:
    """No arm may be graded against strata cut on its own numbers."""
    _effects, report = branch_reports["v_mixed_matrix"]
    members = report["strata"]["worst_decile_members"]
    levels, flags, _ground = synthetic_eval_bank()
    feasible = sorted(
        (k for k in levels if flags[k] <= FLAG_INFEASIBLE_FRAC), key=lambda c: (levels[c], c)
    )
    assert members == feasible[: max(1, len(feasible) // 10)]


# --------------------------------------------------------------------------- permutation test


def test_permutation_is_deterministic_under_the_pinned_seed() -> None:
    deltas = [0.1, -0.02, 0.3, 0.05, -0.11, 0.07]
    first = sign_flip_permutation_p(deltas, nperm=TEST_NPERM, seed=0)
    second = sign_flip_permutation_p(deltas, nperm=TEST_NPERM, seed=0)
    assert first == second
    assert first["rng_seed"] == 0
    assert first["nperm"] == TEST_NPERM
    assert first["n_clips"] == len(deltas)
    assert first["observed_mean"] == pytest.approx(float(np.mean(deltas)))


def test_permutation_p_is_never_zero_and_names_both_of_its_floors() -> None:
    """Two different floors, and which one binds depends on the stratum size, not on the effect.

    With a large stratum the Monte-Carlo draw count is the limit. With a small one -- and the worst
    decile IS small -- there are only 2^n sign assignments, so no number of draws can push the
    p-value below 2^-n however large the effect is. A reviewer reading "p = 0.03" off a 5-clip
    worst decile is reading the lattice, not the evidence.
    """
    big = sign_flip_permutation_p([1.0] * 20, nperm=TEST_NPERM, seed=0, alternative="greater")
    assert big["p"] == pytest.approx(1.0 / (TEST_NPERM + 1))  # 2^-20 is far below the draw floor
    assert big["monte_carlo_floor_p"] == pytest.approx(1.0 / (TEST_NPERM + 1))
    assert big["discrete_floor_p"] == pytest.approx(2.0**-20)

    small = sign_flip_permutation_p([1.0] * 5, nperm=TEST_NPERM, seed=0, alternative="greater")
    assert small["discrete_floor_p"] == pytest.approx(2.0**-5)  # only 32 sign assignments exist
    assert small["p"] > small["monte_carlo_floor_p"]  # the lattice binds first
    assert small["p"] > big["p"]
    assert small["p"] == pytest.approx(2.0**-5, rel=0.6)
    assert small["p"] > 0.0


def test_permutation_on_an_exact_null_is_maximally_unsurprising() -> None:
    result = sign_flip_permutation_p([0.0] * 10, nperm=TEST_NPERM, seed=0)
    assert result["p"] == pytest.approx(1.0)


def test_permutation_alternatives_point_in_opposite_directions() -> None:
    deltas = [0.2, 0.25, 0.3, 0.18, 0.22, 0.27]
    greater = sign_flip_permutation_p(deltas, nperm=TEST_NPERM, seed=0, alternative="greater")
    less = sign_flip_permutation_p(deltas, nperm=TEST_NPERM, seed=0, alternative="less")
    assert greater["p"] < less["p"]
    assert less["p"] == pytest.approx(1.0)


def test_permutation_rejects_degenerate_input() -> None:
    with pytest.raises(AnalysisInputError, match="at least one paired delta"):
        sign_flip_permutation_p([])
    with pytest.raises(AnalysisInputError, match="non-finite"):
        sign_flip_permutation_p([0.1, float("nan")])
    with pytest.raises(ValueError, match="unknown alternative"):
        sign_flip_permutation_p([0.1], alternative="sideways")


def test_permutation_null_is_symmetric_so_a_sign_flip_flips_the_p() -> None:
    """H0 for this test is 'deltas are symmetric about zero' -- worth stating, and worth testing."""
    deltas = [0.3, 0.1, 0.25, 0.05, 0.2]
    up = sign_flip_permutation_p(deltas, nperm=TEST_NPERM, seed=0, alternative="greater")
    down = sign_flip_permutation_p(
        [-d for d in deltas], nperm=TEST_NPERM, seed=0, alternative="less"
    )
    assert up["p"] == pytest.approx(down["p"])


# --------------------------------------------------------------------------- direction labels


@pytest.mark.parametrize(
    ("worst", "best", "easy", "expected"),
    [
        (0.25, 0.008, 0.008, "improves-worst"),
        (0.06, 0.055, 0.035, "uniform-lift"),
        (0.004, 0.0, 0.0, "null"),
        (-0.15, -0.01, -0.01, "harm"),
        (-0.05, -0.10, -0.10, "harm"),
        (0.0, 0.05, 0.0, "uniform-lift"),
        (0.01, 0.0, 0.05, "uniform-lift"),
    ],
)
def test_direction_labels(worst, best, easy, expected) -> None:
    assert classify_direction(worst, best, easy) == expected


# --------------------------------------------------------------------------- exposure ledger


def test_uniform_exposure_has_normalized_entropy_one() -> None:
    metrics = exposure_metrics({f"c{i}": 0.25 for i in range(4)})
    assert metrics["normalized_entropy"] == pytest.approx(1.0)
    assert metrics["effective_num_clips"] == pytest.approx(4.0)
    assert metrics["top1_over_fair_share"] == pytest.approx(1.0)
    assert metrics["wasted_exposure_frac"] == pytest.approx(0.0)


def test_wasted_exposure_is_the_mass_on_flagged_clips() -> None:
    probs = {"good": 0.3, "bad": 0.5, "also_bad": 0.2}
    flags = {"good": 0.01, "bad": 0.4, "also_bad": FLAG_INFEASIBLE_FRAC + 1e-6}
    metrics = exposure_metrics(probs, infeasible_frac=flags)
    assert metrics["wasted_exposure_frac"] == pytest.approx(0.7)
    assert metrics["n_flagged_in_bank"] == 2
    assert metrics["top1_motion_key"] == "bad"


def test_empty_exposure_ledger_is_refused() -> None:
    with pytest.raises(AnalysisInputError, match="exposure ledger is empty"):
        exposure_metrics({})


def test_exposure_ledger_separates_the_arms_the_design_predicts_it_should(branch_reports) -> None:
    """Mechanism evidence, not adjudication: uniform sampling flattens, the cap binds at 5x."""
    _effects, report = branch_reports["i_hygiene_works"]
    ledgers = {arm: report["arms"][arm]["exposure"] for arm in REQUIRED_ARMS}
    assert ledgers["D_raw_uniform"]["normalized_entropy"] == pytest.approx(1.0)
    assert ledgers["E_raw_cap"]["top1_over_fair_share"] == pytest.approx(5.0, abs=0.3)
    assert (
        ledgers["A_raw"]["wasted_exposure_frac"] > ledgers["D_raw_uniform"]["wasted_exposure_frac"]
    )
    assert ledgers["B_pruned"]["wasted_exposure_frac"] == pytest.approx(0.0)
    assert ledgers[BASELINE_ARM]["top1_over_fair_share"] > 5.0


# --------------------------------------------------------------------------- the CLI itself


def test_synthetic_dry_run_returns_zero() -> None:
    assert run_synthetic(nperm=TEST_NPERM, seed=DEFAULT_SEED) == 0


def test_synthetic_dry_run_writes_both_output_formats(tmp_path: Path) -> None:
    out_json, out_md = tmp_path / "matrix.json", tmp_path / "matrix.md"
    assert run_synthetic(nperm=TEST_NPERM, out_json=str(out_json), out_md=str(out_md)) == 0
    report = json.loads(out_json.read_text())
    assert report["design"]["baseline_arm"] == BASELINE_ARM
    assert set(report["arms"]) == set(REQUIRED_ARMS)
    assert out_md.read_text().startswith("# Hygiene x sampler matrix")


def test_cli_synthetic_end_to_end_exits_zero_and_reports_every_branch() -> None:
    """The deliverable as a reviewer will run it: one command, no data, an exit status."""
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "research" / "analyze_hygiene_matrix.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--synthetic", "--nperm", str(TEST_NPERM)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for branch, _blurb, _effects in SYNTHETIC_BRANCHES:
        assert f"branch {branch}" in proc.stdout
    assert "FAIL" not in proc.stdout
    assert "SYNTHETIC DRY-RUN OK" in proc.stdout
    assert "No real data was touched" in proc.stdout


def test_cli_without_strat_fails_loudly() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "research" / "analyze_hygiene_matrix.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 2
    assert "--strat ARM=CSV is required" in proc.stderr
