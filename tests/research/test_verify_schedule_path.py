"""Tests for the Change B schedule-path dry-run tool (plan §3.2 precondition).

Everything here runs through the REAL scheduler engine
(gear_sonic.trl.utils.scheduler) against mock wrapper chains, so the path
syntax that ships in the M5-T spec is the syntax the trainer will execute.
"""

from __future__ import annotations

import pytest

from scripts.research.verify_schedule_path import (
    _MockTerminationManager,
    _Obj,
    build_mock_trainer,
    verify_candidate_paths,
    verify_one_path,
)


def test_self_test_chain_verifies_and_restores() -> None:
    trainer = build_mock_trainer()
    result = verify_candidate_paths(trainer, parameters=("threshold", "down_threshold"))
    assert result["ok"] is True
    for term in ("anchor_pos", "ee_body_pos"):
        for parameter in ("threshold", "down_threshold"):
            assert f"params['{parameter}']" in result["verified_paths"][f"{term}.{parameter}"]
    # Round-trip restored the original values.
    manager = trainer.env.env.unwrapped.termination_manager
    assert manager.get_term_cfg("anchor_pos").params["threshold"] == pytest.approx(0.15)
    assert manager.get_term_cfg("ee_body_pos").params["threshold"] == pytest.approx(0.15)
    assert manager.get_term_cfg("anchor_pos").params["down_threshold"] == pytest.approx(0.75)
    assert manager.get_term_cfg("ee_body_pos").params["down_threshold"] == pytest.approx(0.75)


def test_bare_attr_tail_would_fail_dict_params() -> None:
    # The trap the tool exists to catch: '@params@threshold' resolves params
    # (a dict) then getattr's 'threshold' -> AttributeError. Documented pitfall.
    from gear_sonic.trl.utils import scheduler

    trainer = build_mock_trainer()
    with pytest.raises(AttributeError):
        scheduler._navigate_object_path(
            trainer, "env@env@unwrapped@termination_manager@get_term_cfg('anchor_pos')@params@threshold"
        )


def test_wrong_chain_reports_error_not_crash() -> None:
    trainer = build_mock_trainer()
    record = verify_one_path(trainer, "env@unwrapped", "anchor_pos")
    assert record["resolved"] is False
    assert "error" in record


def test_shallower_chain_is_found() -> None:
    # trainer.env.termination_manager directly (no gym wrapper).
    trainer = _Obj(env=_Obj(termination_manager=_MockTerminationManager()))
    result = verify_candidate_paths(trainer)
    assert result["ok"] is True
    assert result["verified_paths"]["anchor_pos"].startswith("env@termination_manager")


def test_unknown_term_fails_cleanly() -> None:
    trainer = build_mock_trainer()
    result = verify_candidate_paths(trainer, terms=("not_a_term",))
    assert result["ok"] is False
    assert "not_a_term" not in result["verified_paths"]
