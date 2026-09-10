"""Tests for the pilot's validation panel assignment set and its readout logic."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_support_pilot_readout as readout  # noqa: E402
import motion2scene_support_validation_panel as panel  # noqa: E402

OPTIONS = [
    "neutral",
    "short_e015_r265",
    "sustained_e015_r255",
    "prior_splice_e015_r265",
    "prior_splice_e050_r265",
    "short_e070_r265",
    "sustained_e070_r255",
]


def contexts(count=3):
    return [
        dict(
            scene_id=f"support_validation_short_low_{i:02d}",
            base_layout_id=f"support_validation_short_low_{i:02d}",
            stratum="short",
            underside_band="low",
            scene_definition={"path": f"/tmp/scene_{i}.json", "sha256": "sha256:0"},
        )
        for i in range(count)
    ]


def models():
    rows = []
    for seed in (93201, 93202, 93203):
        for arm in ("support_strict", "support_broad", "support_mixture"):
            rows.append(
                dict(
                    policy_id=f"seed{seed}_{arm}",
                    run_id=f"seed{seed}_{arm}",
                    arm=arm,
                    seed=seed,
                    origin="pilot",
                )
            )
    return rows


def test_panel_assigns_every_policy_and_comparator_to_every_context():
    rows = panel.assignments(models(), contexts(3), OPTIONS)
    # nine pilot policies + the strong script + all seven fixed schedules
    assert len({row["policy_id"] for row in rows}) == 9 + 1 + 7
    assert len(rows) == (9 + 1 + 7) * 3
    for scene in {row["scene_id"] for row in rows}:
        assert len([row for row in rows if row["scene_id"] == scene]) == 17
    forced = [row for row in rows if row["mode"] == "forced"]
    assert sorted({row["option_id"] for row in forced}) == sorted(OPTIONS)
    assert len([row for row in rows if row["mode"] == "scripted_multi"]) == 3
    assert {row["physics_seed"] for row in rows} == {panel.PHYSICS_SEED}


def test_panel_assignment_ids_are_dense_and_order_is_deterministic():
    first = panel.assignments(models(), contexts(3), OPTIONS)
    second = panel.assignments(models(), contexts(3), OPTIONS)
    assert [row["assignment_id"] for row in first] == [
        f"episode_{i:03d}" for i in range(len(first))
    ]
    assert first == second


def test_panel_requires_the_whole_repertoire_and_a_nonempty_sample():
    with pytest.raises(ValueError):
        panel.assignments(models(), contexts(1), OPTIONS[:6])
    with pytest.raises(ValueError):
        panel.assignments(models(), [], OPTIONS)


def test_driver_detection_ignores_a_shell_that_merely_names_the_driver():
    """The documented hazard: a command mentioning the path is not a simulator."""
    needle = panel.DRIVER_FILENAME
    process = subprocess.Popen(
        ["/bin/sh", "-c", f"echo {needle} >/dev/null; read _ 2>/dev/null || sleep 30"],
        stdin=subprocess.PIPE,
    )
    try:
        assert process.pid not in panel.native_driver_pids()
    finally:
        process.kill()
        process.wait()


def test_driver_detection_ignores_a_python_process_without_the_driver_argument():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        assert process.pid not in panel.native_driver_pids()
    finally:
        process.kill()
        process.wait()


def test_driver_detection_finds_a_real_python_invocation_of_the_driver(tmp_path):
    target = tmp_path / panel.DRIVER_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("import sys\nsys.stdin.read()\n")
    process = subprocess.Popen([sys.executable, str(target)], stdin=subprocess.PIPE)
    try:
        # Poll briefly: the kernel exposes cmdline as soon as exec completes.
        found = False
        for _ in range(200):
            if process.pid in panel.native_driver_pids():
                found = True
                break
            os.sched_yield()
        assert found
    finally:
        process.kill()
        process.wait()


def summaries():
    return {
        "a": dict(
            passages=2,
            per_context={
                "s0": dict(outcome="pass", passage_time_s=4.0),
                "s1": dict(outcome="pass", passage_time_s=5.0),
                "s2": dict(outcome="failure", passage_time_s=None),
            },
        ),
        "b": dict(
            passages=3,
            per_context={
                "s0": dict(outcome="pass", passage_time_s=4.5),
                "s1": dict(outcome="pass", passage_time_s=5.0),
                "s2": dict(outcome="pass", passage_time_s=6.0),
            },
        ),
    }


def test_matched_difference_pairs_only_mutually_successful_contexts():
    diff = readout.matched_difference("a", "b", summaries())
    assert diff["passage_difference"] == -1
    assert diff["mutually_successful"] == 2
    # (4.0-4.5) and (5.0-5.0) -> mean -0.25; s2 is excluded, not counted as 0.
    assert diff["mean_paired_passage_time_difference_s"] == -0.25


def test_matched_difference_reports_none_when_nothing_is_mutually_successful():
    only = {
        "a": dict(passages=1, per_context={"s0": dict(outcome="pass", passage_time_s=4.0)}),
        "b": dict(passages=0, per_context={"s0": dict(outcome="failure", passage_time_s=None)}),
    }
    diff = readout.matched_difference("a", "b", only)
    assert diff["mutually_successful"] == 0
    assert diff["mean_paired_passage_time_difference_s"] is None
    assert diff["passage_difference"] == 1


def test_matched_difference_is_absent_for_an_unmeasured_policy():
    assert readout.matched_difference("a", "missing", summaries()) is None


def test_bank_capability_separates_solvable_from_unsolved_contexts():
    rows = [
        dict(mode="forced", scene_id="s0", option_id="neutral", outcome="pass", passage_time_s=5.0),
        dict(
            mode="forced",
            scene_id="s0",
            option_id="short_e015_r265",
            outcome="failure",
            passage_time_s=None,
        ),
        dict(
            mode="forced",
            scene_id="s1",
            option_id="neutral",
            outcome="failure",
            passage_time_s=None,
        ),
    ]
    capability = readout.bank_capability(rows, ["s0", "s1"])
    assert capability["s0"]["bank_solvable"] is True
    assert capability["s0"]["passing_schedules"] == ["neutral"]
    assert capability["s0"]["best_bank_passage_time_s"] == 5.0
    # A context no schedule solves is retained, not dropped.
    assert capability["s1"]["bank_solvable"] is False
    assert capability["s1"]["best_bank_passage_time_s"] is None


def test_policy_summary_counts_a_selection_failure_only_where_the_bank_can_pass():
    rows = [
        dict(
            policy_id="p",
            mode="learned",
            scene_id="s0",
            outcome="failure",
            passage_time_s=None,
            chosen_option_id="neutral",
            stratum="short",
            underside_band="low",
        ),
        dict(
            policy_id="p",
            mode="learned",
            scene_id="s1",
            outcome="failure",
            passage_time_s=None,
            chosen_option_id="neutral",
            stratum="short",
            underside_band="low",
        ),
    ]
    capability = {
        "s0": dict(bank_solvable=True),
        "s1": dict(bank_solvable=False),
    }
    summary = readout.policy_summary(rows, ["s0", "s1"], capability)["p"]
    assert summary["selection_failures"] == ["s0"]
    assert summary["passages"] == 0
    assert summary["failures"] == 2
    assert summary["unmeasured"] == 0


def test_policy_summary_reports_unmeasured_assignments_rather_than_hiding_them():
    rows = [
        dict(
            policy_id="p",
            mode="learned",
            scene_id="s0",
            outcome="pass",
            passage_time_s=4.0,
            chosen_option_id="neutral",
            stratum="short",
            underside_band="low",
        )
    ]
    summary = readout.policy_summary(rows, ["s0", "s1"], {"s0": dict(bank_solvable=True)})["p"]
    assert summary["assigned"] == 2
    assert summary["measured"] == 1
    assert summary["unmeasured"] == 1


def test_validation_definitions_satisfy_the_frozen_collector_split_contract():
    """The collector accepts only two splits, and widening it is not an option.

    It is hash-pinned in three closure lists of the adopted runtime, so a
    validation context that stamps its own split would abort the panel's very
    first prepare -- after all the acquisition physics had already been spent.
    """
    import motion2scene_collect_timed_schedules as collection
    import motion2scene_support_validation_sample as sample

    source = Path(sample.__file__).read_text()
    assert 'split="development"' in source
    # The distinguishing role is retained, but out of the validated field.
    assert 'sample_role="development_validation"' in source
    assert 'split="development_validation"' not in source

    signature = Path(collection.__file__).read_text()
    assert '("development", "reserved_evaluation_v3")' in signature
    # The panel must assert the same value the collector will accept.
    panel_source = Path(panel.__file__).read_text()
    assert 'common["split"] != "development"' in panel_source


def synthetic_readout():
    """A deliberately awkward readout: an unsolved context, a null time, a gap."""
    return {
        "assigned_episodes": 200,
        "measured_episodes": 40,
        "measured_physics_steps": 47680,
        "unmeasured_assignments": ["episode_199"],
        "contexts": [
            {
                "scene_id": "support_validation_short_low_00",
                "stratum": "short",
                "underside_band": "low",
                "base_layout_id": "support_validation_short_low_00",
            },
            {
                "scene_id": "support_validation_sustained_high_00",
                "stratum": "sustained",
                "underside_band": "high",
                "base_layout_id": "support_validation_sustained_high_00",
            },
        ],
        "bank_capability": {
            "support_validation_short_low_00": {
                "bank_solvable": True,
                "passing_schedules": ["neutral"],
                "best_bank_passage_time_s": 5.1,
            },
            "support_validation_sustained_high_00": {
                "bank_solvable": False,
                "passing_schedules": [],
                "best_bank_passage_time_s": None,
            },
        },
        "bank_unsolved_contexts": ["support_validation_sustained_high_00"],
        "policies": {
            "seed93201_support_strict": {
                "arm": "support_strict",
                "seed": 93201,
                "passages": 1,
                "measured": 2,
                "selection_failures": [],
                "distinct_schedules_used": 1,
            },
            "seed93201_support_broad": {
                "arm": "support_broad",
                "seed": 93201,
                "passages": 0,
                "measured": 2,
                "selection_failures": ["support_validation_short_low_00"],
                "distinct_schedules_used": 1,
            },
        },
        "primary_contrasts": [
            {
                "seed": 93201,
                "first": "seed93201_support_mixture",
                "second": "seed93201_support_strict",
                "passage_difference": 0,
                "mutually_successful": 0,
                "mean_paired_passage_time_difference_s": None,
            },
        ],
        "encounter_mechanism": [
            {
                "executed": True,
                "inside_positive_screen": True,
                "inside_negative_screen": False,
                "usable_continuation_target": True,
                "all_branches_failed": False,
            },
            {
                "executed": True,
                "inside_positive_screen": False,
                "inside_negative_screen": True,
                "usable_continuation_target": False,
                "all_branches_failed": True,
            },
            {"executed": False, "inside_positive_screen": True, "inside_negative_screen": True},
        ],
        "cost_denominator_warning": {
            "rule": "compare at matched assigned episodes",
            "why": "early failures cost fewer steps",
            "per_corpus": {
                "seed93201_support_broad": {
                    "arm": "support_broad",
                    "seed": 93201,
                    "new_recorded_physics_steps": 5404,
                    "executed_encounters": 1,
                    "encounters_with_no_passing_branch": 1,
                    "steps_in_encounters_that_taught_nothing": 4564,
                },
            },
        },
        "shared_prefix": {"rounds": 4, "source_arm": "analytic_contrast", "note": "shared"},
        "interpretation_rules": ["passage is the criterion"],
    }


def test_render_reports_partial_measurement_and_unsolved_contexts_honestly():
    text = readout.render(synthetic_readout())
    assert "40 of 200 assigned episodes" in text
    # A context no schedule solves is named and emphasised, never dropped.
    assert "**no**" in text
    assert "Contexts no schedule in the bank solves: **1**" in text
    assert "sustained_high_00" in text
    # Unmeasured assignments are counted rather than silently omitted.
    assert "never dropped: 1" in text


def test_render_shows_a_null_paired_time_rather_than_a_zero():
    text = readout.render(synthetic_readout())
    row = [ln for ln in text.splitlines() if "mixture − support_strict" in ln]
    assert row, "the declared contrast row must be rendered"
    assert row[0].rstrip().endswith("| 0 | — |")
    assert "| +0 |" in row[0]


def test_render_stratifies_encounters_by_pre_registered_gate_membership():
    text = readout.render(synthetic_readout())
    assert "inside positive · outside negative | 1 | 1 | 0" in text
    assert "outside positive · inside negative | 1 | 0 | 1" in text


def test_render_pairs_cheapness_with_encounters_that_taught_nothing():
    text = readout.render(synthetic_readout())
    assert "compare at matched assigned episodes" in text
    assert "| 5404 | 1 | 1 | 4564 |" in text


def test_render_survives_a_readout_with_no_executed_encounter():
    empty = synthetic_readout()
    empty["encounter_mechanism"] = []
    text = readout.render(empty)
    assert "_no encounter executed yet_" in text
