"""Eval-strip guard tests (research_plan_zpd_teacher.md §3.2, Change B precondition).

The pitfall: ``eval_agent_trl.py:466-470`` re-applies ``trainer.schedule_dict``
at the checkpoint's global step, and ``:134-142`` strips only
``train_only_events``-scoped entries — so a threshold-curriculum schedule used
in training silently reshapes the termination verifier at eval, corrupting every
comparison. The guard lives in ``validate_spec`` (the choke point every
preregistered experiment spec passes through, both in run_sonic_paired_experiment
and run_sonic_multiseed): any spec whose materialized commands use
``schedule_dict`` must strip it from the eval invocation.
"""

from __future__ import annotations

from scripts.research.run_sonic_multiseed import render_spec_for_seed
from scripts.research.run_sonic_paired_experiment import validate_spec


def _spec(train_command: str, eval_command: str) -> dict:
    return {
        "experiment_group": "sim_m5_t",
        "hypothesis": "threshold schedule manufactures a frontier",
        "seed": 0,
        "dataset_robot": "data/robot",
        "dataset_smpl": "data/smpl",
        "checkpoint": "ckpt/last.pt",
        "variants": [
            {
                "name": "threshold_schedule",
                "train_command": train_command,
                "eval_command": eval_command,
                "interpretation": "arm",
            }
        ],
    }


_SCHEDULE_OVERRIDE = (
    "'+trainer.schedule_dict={\"env@x@get_term_cfg(anchor_pos)@params@threshold\":"
    "{type:linear,seg_steps:[0,150],seg_vals:[0.30,0.15]}}'"
)


def test_schedule_in_train_without_eval_strip_is_rejected() -> None:
    spec = _spec(
        f"python gear_sonic/train_agent_trl.py +exp=e {_SCHEDULE_OVERRIDE}",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c",
    )
    errors = validate_spec(spec)
    assert any("schedule_dict" in e and "strip" in e for e in errors)


def test_schedule_in_train_with_tilde_strip_in_eval_passes() -> None:
    spec = _spec(
        f"python gear_sonic/train_agent_trl.py +exp=e {_SCHEDULE_OVERRIDE}",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c '~trainer.schedule_dict'",
    )
    assert validate_spec(spec) == []


def test_schedule_in_train_with_null_strip_in_eval_passes() -> None:
    spec = _spec(
        f"python gear_sonic/train_agent_trl.py +exp=e {_SCHEDULE_OVERRIDE}",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c trainer.schedule_dict=null",
    )
    assert validate_spec(spec) == []


def test_schedule_set_directly_in_eval_is_rejected() -> None:
    spec = _spec(
        "python gear_sonic/train_agent_trl.py +exp=e",
        f"python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c {_SCHEDULE_OVERRIDE}",
    )
    errors = validate_spec(spec)
    assert any("schedule_dict" in e for e in errors)


def test_spec_without_schedules_is_unaffected() -> None:
    spec = _spec(
        "python gear_sonic/train_agent_trl.py +exp=e",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c",
    )
    assert validate_spec(spec) == []


def test_schedule_bearing_trainer_config_is_detected_without_literal_key() -> None:
    # trainer=trl_threshold_curriculum carries a schedule_dict inside the YAML;
    # the command never contains the literal string "schedule_dict".
    spec = _spec(
        "python gear_sonic/train_agent_trl.py +exp=e trainer=trl_threshold_curriculum",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c",
    )
    errors = validate_spec(spec)
    assert any("schedule-bearing trainer config" in e for e in errors)
    spec = _spec(
        "python gear_sonic/train_agent_trl.py +exp=e trainer=trl_threshold_curriculum",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c ++trainer.schedule_dict=null",
    )
    assert validate_spec(spec) == []


def test_m5_spec_templates_pass_guard_for_all_seeds() -> None:
    # The committed launch templates must stay guard-clean end to end.
    import json
    from pathlib import Path

    from scripts.research.run_sonic_multiseed import _template_has_placeholder

    repo_root = Path(__file__).resolve().parents[2]
    for name in (
        "sim_m5_l_learnability_multiseed_template",
        "sim_m5_t_threshold_multiseed_template",
    ):
        template = json.loads((repo_root / "configs" / "research" / f"{name}.json").read_text())
        assert _template_has_placeholder(template), name
        for seed in (0, 1, 2):
            assert validate_spec(render_spec_for_seed(template, seed)) == [], name


def test_guard_applies_to_multiseed_rendered_specs() -> None:
    # The multiseed orchestrator renders {seed} templates through the same
    # validate_spec choke point; a bad template must fail for every seed.
    template = _spec(
        f"python gear_sonic/train_agent_trl.py +exp=e seed={{seed}} {_SCHEDULE_OVERRIDE}",
        "python gear_sonic/eval_agent_trl.py +exp=e +checkpoint=c seed={seed}",
    )
    for seed in (0, 1, 2):
        rendered = render_spec_for_seed(template, seed)
        errors = validate_spec(rendered)
        assert any("schedule_dict" in e for e in errors)
