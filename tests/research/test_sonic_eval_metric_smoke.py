from __future__ import annotations

import json
from pathlib import Path

from scripts.research.diagnose_sonic_eval_logs import diagnose_eval_log
from scripts.research.run_sonic_eval_metric_smoke import (
    build_eval_command,
    run_eval_metric_smoke,
    validate_eval_smoke_spec,
)


def _spec(tmp_path: Path, command: str = "printf ''") -> dict:
    return {
        "name": "sonic_eval_micro_metric_complete",
        "goal": "SIM-M1 bounded eval metric completeness on sample_data",
        "repo_root": str(tmp_path),
        "dataset_robot": "sample_data/robot_filtered",
        "dataset_smpl": "sample_data/smpl_filtered",
        "checkpoint": "sonic_release/last.pt",
        "eval": {
            "enabled": True,
            "command": command,
            "max_sequences": 1,
            "max_steps_per_sequence": 200,
            "num_envs": 2,
            "headless": True,
            "timeout_seconds": 5,
        },
        "success_criteria": {
            "traceback": False,
            "all_mpjpe_present": True,
            "eval_ok": True,
            "primary_mpjpe_metric": "mpjpe_g",
        },
    }


def test_validate_eval_smoke_spec_requires_eval_enabled(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["eval"]["enabled"] = False

    errors = validate_eval_smoke_spec(spec)

    assert "eval.enabled must be true" in errors


def test_build_eval_command_includes_bounded_eval_knobs(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    del spec["eval"]["command"]

    command = build_eval_command(spec)

    assert "++callbacks.im_eval.max_eval_steps=200" in command
    assert "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1" in command
    assert "+manager_env/terminations=tracking/eval" in command


def test_run_eval_metric_smoke_passes_when_all_mpjpe_is_emitted(tmp_path: Path) -> None:
    command = (
        "printf 'Success Rate: 1.0000000000\\nProgress Rate: 1.0000000000\\n"
        "All:  mpjpe_g: 16.493 mpjpe_l: 13.255 mpjpe_pa: 9.168\\n"
        "Succ:  mpjpe_g: 16.493 mpjpe_l: 13.255 mpjpe_pa: 9.168\\n'"
    )
    result = run_eval_metric_smoke(
        _spec(tmp_path, command=command),
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert result["ok"] is True
    assert result["eval_ok"] is True
    assert result["primary_mpjpe_value"] == 16.493
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval"]["all"]["mpjpe_g"] == 16.493


def test_run_eval_metric_smoke_fails_without_all_mpjpe(tmp_path: Path) -> None:
    result = run_eval_metric_smoke(
        _spec(tmp_path, command="printf 'Terminated: 1 | Succ rate: 0.000 | Mpjpe: nan\\n'"),
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert result["ok"] is False
    assert result["eval_ok"] is False
    assert result["primary_mpjpe_value"] is None


def test_diagnose_eval_log_reports_metric_presence(tmp_path: Path) -> None:
    eval_log = tmp_path / "eval.log"
    eval_log.write_text(
        "Sequence progress: 100%\n"
        "Success Rate: 1.0000000000\n"
        "All:  mpjpe_g: 1.0 mpjpe_l: 2.0\n"
        "Succ:  mpjpe_g: 1.0\n",
        encoding="utf-8",
    )

    diag = diagnose_eval_log("baseline", eval_log, tmp_path / "diag")

    assert diag["contains_all_mpjpe"] is True
    assert diag["contains_any_mpjpe"] is True
    assert diag["eval_traceback"] is False
    assert diag["num_sequences_started"] == 1
    assert diag["candidate_metric_lines"][-1].startswith("Succ:")
