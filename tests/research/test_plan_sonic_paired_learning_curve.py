from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

import scripts.research.plan_sonic_paired_learning_curve as curve_module
from scripts.research.plan_sonic_paired_learning_curve import (
    CHECKPOINT_ITERATIONS,
    aggregate_learning_curve,
    build_learning_curve_plan,
    execute_learning_curve,
    write_curve_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "configs/research/bones_seed_official_failure_rate_vs_zpd_template.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _execution_spec(tmp_path: Path) -> dict:
    spec = _spec()
    old_initialization = spec["training_initialization_checkpoint"]
    initialization = tmp_path / "checkpoints/release.pt"
    initialization.parent.mkdir(parents=True)
    initialization.write_bytes(b"release")
    spec["training_initialization_checkpoint"] = str(initialization)
    spec["training_initialization_checkpoint_sha256"] = hashlib.sha256(b"release").hexdigest()

    for variant in spec["variants"]:
        name = variant["name"]
        old_checkpoint = variant["checkpoint"]
        old_experiment_dir = old_checkpoint.removesuffix("/last.pt")
        experiment_dir = tmp_path / "training" / name
        checkpoint = experiment_dir / "last.pt"
        variant["checkpoint"] = str(checkpoint)
        variant["train_command"] = variant["train_command"].replace(
            f"+checkpoint={old_initialization}", f"+checkpoint={initialization}"
        ).replace(
            f"experiment_dir={old_experiment_dir}", f"experiment_dir={experiment_dir}"
        )
        variant["eval_command"] = variant["eval_command"].replace(
            f"+checkpoint={old_checkpoint}", f"+checkpoint={checkpoint}"
        )
    return spec


def _prepared_execution(tmp_path: Path) -> tuple[dict, dict]:
    spec = _execution_spec(tmp_path)
    plan = build_learning_curve_plan(spec, output_dir=tmp_path / "curve")
    for checkpoint_text in {row["checkpoint"] for row in plan["rows"]}:
        checkpoint = Path(checkpoint_text)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            checkpoint.write_bytes(f"checkpoint:{checkpoint.name}".encode())
    return spec, plan


def _verified_provenance() -> dict:
    return {
        "dataset_manifest": {
            "path": "dataset_manifest.json",
            "sha256": "1" * 64,
            "paired_dataset_sha256": "2" * 64,
            "motion_count": 2,
            "motion_keys": ["motion_a", "motion_b"],
            "content_hashes_verified": True,
        }
    }


def test_plans_exact_paired_official_learning_curve(tmp_path: Path) -> None:
    spec = _spec()

    plan = build_learning_curve_plan(spec, output_dir=tmp_path / "curve")

    assert plan["checkpoint_iterations"] == list(CHECKPOINT_ITERATIONS)
    assert plan["eval_timeout_seconds"] == 7200
    assert plan["commands_equal_outside_checkpoint_and_eval_output_dir"] is True
    assert plan["metric_sources"] == {
        "success_rate": "eval/success/success_rate",
        "progress_rate": "eval/success/progress_rate",
        "mpjpe_l": "eval/all/mpjpe_l",
        "mpjpe_g": "eval/all/mpjpe_g",
    }
    assert len(plan["rows"]) == 10
    assert {
        (row["variant"], row["checkpoint_iteration"]) for row in plan["rows"]
    } == {
        (variant, iteration)
        for variant in ("official_failure_rate", "zpd_learnability")
        for iteration in CHECKPOINT_ITERATIONS
    }

    initial = [row for row in plan["rows"] if row["checkpoint_iteration"] == 0]
    assert len({row["checkpoint"] for row in initial}) == 1
    assert initial[0]["checkpoint"] == spec["training_initialization_checkpoint"]
    assert {row["shared_evaluation_id"] for row in initial} == {
        "release_initialization"
    }
    assert plan["gpu_eval_invocations_expected"] == 9

    step_50 = next(
        row
        for row in plan["rows"]
        if row["variant"] == "official_failure_rate" and row["checkpoint_iteration"] == 50
    )
    assert step_50["checkpoint"].endswith(
        "seed0/official_failure_rate/model_step_000050.pt"
    )
    assert step_50["sample_budget_env_transitions"] == 50 * 128 * 24
    assert "+checkpoint=" in step_50["eval_command"]
    assert "++eval_callbacks=im_eval" in step_50["eval_command"]
    assert "++callbacks.im_eval.max_eval_steps=null" in step_50["eval_command"]
    assert "robot_filtered" in step_50["eval_command"]
    assert "smpl_filtered" in step_50["eval_command"]

    final_rows = [row for row in plan["rows"] if row["checkpoint_iteration"] == 200]
    assert {row["sample_budget_env_transitions"] for row in final_rows} == {614_400}
    assert all(row["checkpoint"].endswith("model_step_000200.pt") for row in final_rows)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda spec: spec["learning_curve"].__setitem__(
                "checkpoint_iterations", [0, 50, 100, 200]
            ),
            "checkpoint_iterations must be exactly",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "train_command",
                    variant["train_command"].replace(
                        "algo.config.num_steps_per_env=24",
                        "algo.config.num_steps_per_env=12",
                    ),
                )
                for variant in spec["variants"]
            ],
            "num_steps_per_env=24",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "train_command",
                    variant["train_command"].replace(
                        "callbacks.model_save.save_frequency=50",
                        "callbacks.model_save.save_frequency=100",
                    ),
                )
                for variant in spec["variants"]
            ],
            "save_frequency=50",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "eval_command",
                    variant["eval_command"].replace(
                        "callbacks.im_eval.max_eval_steps=null",
                        "callbacks.im_eval.max_eval_steps=64",
                    ),
                )
                for variant in spec["variants"]
            ],
            "callbacks.im_eval.max_eval_steps=null",
        ),
        (
            lambda spec: [
                variant.__setitem__(
                    "eval_command",
                    variant["eval_command"]
                    + " ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=1",
                )
                for variant in spec["variants"]
            ],
            "full-sequence limiter",
        ),
    ],
)
def test_rejects_incomplete_or_unmaterializable_curve_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    spec = _spec()
    mutate(spec)

    with pytest.raises(ValueError, match=message):
        build_learning_curve_plan(spec, output_dir=tmp_path / "curve")


def _write_metrics(
    path: Path,
    *,
    success: float,
    progress: float | None = None,
    mpjpe_l: float,
    mpjpe_g: float,
    motion_keys: list[str] | None = None,
) -> None:
    motion_keys = motion_keys or ["motion"]
    progress = success if progress is None else progress
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval/success/success_rate": success,
                "eval/success/progress_rate": progress,
                "eval/all/mpjpe_l": mpjpe_l,
                "eval/all/mpjpe_g": mpjpe_g,
                "eval/all_metrics_dict": {
                    "motion_keys": motion_keys,
                    "terminated": [1.0 - success] * len(motion_keys),
                    "mpjpe_l": [mpjpe_l] * len(motion_keys),
                    "mpjpe_g": [mpjpe_g] * len(motion_keys),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_aggregates_paired_metrics_sample_budget_and_auc(tmp_path: Path) -> None:
    plan = build_learning_curve_plan(_spec(), output_dir=tmp_path / "curve")
    baseline_success = [0.0, 0.2, 0.4, 0.6, 0.8]
    treatment_success = [0.0, 0.3, 0.6, 0.8, 1.0]
    baseline_mpjpe_l = [100.0, 80.0, 60.0, 40.0, 20.0]
    treatment_mpjpe_l = [100.0, 70.0, 50.0, 30.0, 10.0]

    for row in plan["rows"]:
        index = list(CHECKPOINT_ITERATIONS).index(row["checkpoint_iteration"])
        treatment = row["variant"] == "zpd_learnability"
        success = treatment_success[index] if treatment else baseline_success[index]
        mpjpe_l = treatment_mpjpe_l[index] if treatment else baseline_mpjpe_l[index]
        _write_metrics(
            Path(row["metrics_eval_json"]),
            success=success,
            mpjpe_l=mpjpe_l,
            mpjpe_g=mpjpe_l * 2.0,
        )

    result = aggregate_learning_curve(plan, repo_root=REPO_ROOT)

    assert result["complete"] is True
    assert len(result["rows"]) == 5
    assert result["rows"][0]["sample_budget_env_transitions"] == 0
    assert result["rows"][-1]["sample_budget_env_transitions"] == 614_400
    assert result["rows"][2]["delta_treatment_minus_baseline"]["success_rate"] == pytest.approx(
        0.2
    )
    assert result["rows"][2]["delta_treatment_minus_baseline"]["mpjpe_l"] == -10.0
    assert result["auc"]["success_rate"]["baseline_normalized_auc"] == pytest.approx(0.4)
    assert result["auc"]["success_rate"]["treatment_normalized_auc"] == pytest.approx(0.55)
    assert result["auc"]["success_rate"]["normalized_auc_zpd_advantage"] == pytest.approx(
        0.15
    )
    assert result["auc"]["progress_rate"]["normalized_auc_zpd_advantage"] == pytest.approx(
        0.15
    )
    assert result["auc"]["mpjpe_l"]["baseline_normalized_auc"] == pytest.approx(60.0)
    assert result["auc"]["mpjpe_l"]["treatment_normalized_auc"] == pytest.approx(51.25)
    assert result["auc"]["mpjpe_l"]["normalized_auc_zpd_advantage"] == pytest.approx(8.75)

    write_curve_outputs(tmp_path / "curve", result)
    assert (tmp_path / "curve/learning_curve.json").is_file()
    assert "sample_budget_env_transitions" in (
        tmp_path / "curve/learning_curve.csv"
    ).read_text(encoding="utf-8")
    assert "Normalized AUC" in (tmp_path / "curve/learning_curve.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eval/success/success_rate", math.nan, "non-finite"),
        ("eval/success/success_rate", 1.1, r"must be in \[0, 1\]"),
        ("eval/success/progress_rate", 1.1, r"must be in \[0, 1\]"),
        ("eval/all/mpjpe_l", -1.0, "must be non-negative"),
    ],
)
def test_rejects_invalid_official_metric_values(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    plan = build_learning_curve_plan(_spec(), output_dir=tmp_path / "curve")
    for row in plan["rows"]:
        path = Path(row["metrics_eval_json"])
        _write_metrics(path, success=0.5, mpjpe_l=20.0, mpjpe_g=30.0)
    first_path = Path(plan["rows"][0]["metrics_eval_json"])
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload[field] = value
    first_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        aggregate_learning_curve(plan, repo_root=REPO_ROOT)


def test_execute_runs_all_rows_shell_free_and_requires_exact_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    preflight_calls: list[tuple[dict, Path]] = []
    subprocess_calls: list[list[str]] = []

    def fake_preflight(spec_arg: dict, *, repo_root: Path) -> dict:
        preflight_calls.append((spec_arg, repo_root))
        return _verified_provenance()

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        subprocess_calls.append(argv)
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 7200
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["cwd"] == str(tmp_path)
        assert isinstance(argv, list)
        assert argv[1] == "gear_sonic/eval_agent_trl.py"
        output_token = next(
            token for token in argv if token.lstrip("+~").startswith("eval_output_dir=")
        )
        output_dir = Path(output_token.split("=", 1)[1])
        _write_metrics(
            output_dir / "metrics_eval.json",
            success=0.5,
            mpjpe_l=20.0,
            mpjpe_g=30.0,
            motion_keys=["motion_a", "motion_b"],
        )
        kwargs["stdout"].write("official eval fixture\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(curve_module, "_verify_official_zpd_pair_preflight", fake_preflight)
    monkeypatch.setattr(curve_module.subprocess, "run", fake_run)

    result = execute_learning_curve(
        plan,
        spec=spec,
        output_dir=tmp_path / "curve",
        repo_root=tmp_path,
    )

    assert preflight_calls == [(spec, tmp_path)]
    assert len(subprocess_calls) == 9
    assert result["complete"] is True
    assert result["execution"]["all_10_rows_passed"] is True
    assert result["execution"]["mode"] == "sequential_shell_free_with_shared_initialization"
    assert result["execution"]["gpu_eval_invocations"] == 9
    assert result["execution"]["shared_initialization_reuses"] == 1
    assert len(result["execution"]["rows"]) == 10
    assert sum(
        row["reused_shared_initialization"] for row in result["execution"]["rows"]
    ) == 1
    assert all(
        row["metrics_coverage"]["exact_motion_key_coverage"] is True
        for row in result["execution"]["rows"]
    )
    log_texts = [Path(row["eval_log"]).read_text() for row in plan["rows"]]
    assert log_texts.count("official eval fixture\n") == 9
    assert sum(text.startswith("Reused content-identical official ImEval metrics") for text in log_texts) == 1
    zero_rows = [
        row for row in result["execution"]["rows"] if row["checkpoint_iteration"] == 0
    ]
    assert len({row["metrics_sha256"] for row in zero_rows}) == 1

    write_curve_outputs(tmp_path / "curve", result)
    assert (tmp_path / "curve/learning_curve.json").is_file()


def test_execute_preflights_every_checkpoint_before_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    missing = next(row for row in plan["rows"] if row["checkpoint_iteration"] == 50)
    Path(missing["checkpoint"]).unlink()
    subprocess_called = False

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )

    def forbidden_run(*args, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("subprocess must not start after failed checkpoint preflight")

    monkeypatch.setattr(curve_module.subprocess, "run", forbidden_run)

    with pytest.raises(ValueError, match="checkpoint does not exist"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )
    assert subprocess_called is False
    assert not any(Path(row["eval_log"]).exists() for row in plan["rows"])


def test_execute_refuses_any_existing_eval_route_before_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    existing_route = Path(plan["rows"][-1]["eval_log"]).parent
    existing_route.mkdir(parents=True)

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )
    monkeypatch.setattr(
        curve_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run when a route exists"),
    )

    with pytest.raises(ValueError, match="refusing to overwrite existing eval output route"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )


def test_execute_stops_on_nonzero_return_without_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    calls = 0

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )

    def failed_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        kwargs["stdout"].write("fixture failure\n")
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(curve_module.subprocess, "run", failed_run)

    with pytest.raises(ValueError, match="return code 7"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )
    assert calls == 1
    assert not (tmp_path / "curve/learning_curve.json").exists()


def test_execute_rejects_zero_return_without_fresh_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    calls = 0

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )

    def empty_success(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        kwargs["stdout"].write("returned zero without official callback output\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(curve_module.subprocess, "run", empty_success)

    with pytest.raises(ValueError, match="did not create fresh official metrics"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )
    assert calls == 1
    assert not (tmp_path / "curve/learning_curve.json").exists()


def test_execute_stops_after_configured_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    calls = 0

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )

    def timed_out(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(curve_module.subprocess, "run", timed_out)

    with pytest.raises(ValueError, match="exceeded timeout 7200s"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )
    assert calls == 1
    assert not (tmp_path / "curve/learning_curve.json").exists()


def test_execute_stops_on_inexact_manifest_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, plan = _prepared_execution(tmp_path)
    calls = 0

    monkeypatch.setattr(
        curve_module,
        "_verify_official_zpd_pair_preflight",
        lambda spec_arg, *, repo_root: _verified_provenance(),
    )

    def incomplete_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        output_token = next(
            token for token in argv if token.lstrip("+~").startswith("eval_output_dir=")
        )
        output_dir = Path(output_token.split("=", 1)[1])
        _write_metrics(
            output_dir / "metrics_eval.json",
            success=0.5,
            mpjpe_l=20.0,
            mpjpe_g=30.0,
            motion_keys=["motion_a"],
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(curve_module.subprocess, "run", incomplete_run)

    with pytest.raises(ValueError, match="do not exactly cover"):
        execute_learning_curve(
            plan,
            spec=spec,
            output_dir=tmp_path / "curve",
            repo_root=tmp_path,
        )
    assert calls == 1
    assert not (tmp_path / "curve/learning_curve.json").exists()
