#!/usr/bin/env python3
"""Run the isolated, diagnostic-only SIM-D2 root-XY evaluation.

SIM-D2 reuses the frozen SIM-D1 checkpoint, paired-dataset, coverage-order,
and full-sequence preflight. It adds a 0.25 m horizontal root guard alongside
the existing Z-height guard, records exact guard-aligned telemetry, and never
produces an eligibility classification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from gear_sonic.utils.root_xy_diagnostic import ROOT_XY_PER_MOTION_FIELDS  # noqa: E402
from scripts.research.run_sim_d1_all_motion_eval import (  # noqa: E402
    _file_snapshot,
    _load_json_object,
    _resolve_path,
    _sha256,
    _validate_spec,
    _write_json,
    run_sim_d1_all_motion_eval,
)

SCHEMA_VERSION = 1
ROOT_XY_THRESHOLD_M = 0.25
DIAGNOSTIC_ROOT = Path("outputs/research/sim_d2_root_xy")
ROOT_XY_TERM_OVERRIDE = (
    "++manager_env.terminations.anchor_pos_xy="
    "{_target_:isaaclab.managers.TerminationTermCfg,"
    "func:gear_sonic.envs.manager_env.mdp:exceeded_anchor_pos_xy,"
    "params:{command_name:motion,threshold:0.25}}"
)
ROOT_XY_CALLBACK_OVERRIDE = (
    "++callbacks.im_eval.root_xy_diagnostic_threshold_m=0.25"
)
_D2_RESERVED_OVERRIDE_FRAGMENTS = (
    "anchor_pos_xy",
    "root_xy_diagnostic_threshold_m",
)


def _new_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_within(left, right) or _is_within(right, left)


def _validate_isolated_output_path(
    *, output_dir: Path, canonical_d1_output_dir: Path, repo_root: Path
) -> Path:
    diagnostic_root = (repo_root / DIAGNOSTIC_ROOT).resolve()
    if _paths_overlap(output_dir, canonical_d1_output_dir):
        raise ValueError(
            "refusing SIM-D2 output that is a canonical SIM-D1 path or overlaps it: "
            f"d2={output_dir}, d1={canonical_d1_output_dir}"
        )
    if output_dir == diagnostic_root or not _is_within(output_dir, diagnostic_root):
        raise ValueError(
            f"SIM-D2 output must be a child of the isolated root {diagnostic_root}"
        )
    return diagnostic_root


def _validate_source_contract(spec: dict[str, Any]) -> None:
    if spec.get("motion_order") != "coverage_manifest":
        raise ValueError("SIM-D2 requires motion_order=coverage_manifest")
    if spec.get("num_envs") != 1:
        raise ValueError("SIM-D2 requires num_envs=1 for exact frozen order")
    if not isinstance(spec.get("dataset_manifest"), dict):
        raise ValueError("SIM-D2 requires a paired dataset_manifest binding")
    for override in spec.get("extra_overrides", []):
        lowered = override.lower()
        if any(fragment in lowered for fragment in _D2_RESERVED_OVERRIDE_FRAGMENTS):
            raise ValueError(
                f"SIM-D2 owns the root-XY guard and callback override: {override!r}"
            )


def _tree_snapshot(root: Path) -> dict[str, Any]:
    """Hash a tree so a D2 attempt can prove canonical D1 immutability."""
    if not root.exists():
        return {"exists": False, "files": {}, "directories": []}
    if not root.is_dir():
        return {
            "exists": True,
            "not_a_directory": True,
            "snapshot": _file_snapshot(root),
        }
    files: dict[str, dict[str, Any]] = {}
    directories: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            files[relative] = {"symlink_target": os.readlink(path)}
        elif path.is_file():
            snapshot = _file_snapshot(path)
            assert snapshot is not None
            files[relative] = snapshot
        elif path.is_dir():
            directories.append(relative)
    return {"exists": True, "files": files, "directories": directories}


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "exists": snapshot.get("exists", False),
        "file_count": len(snapshot.get("files", {})),
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _retarget_eval_output(command: list[str], eval_output_dir: Path) -> list[str]:
    prefix = "++eval_output_dir="
    matches = [index for index, value in enumerate(command) if value.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("validated D1 command must contain exactly one eval_output_dir")
    result = list(command)
    result[matches[0]] = f"{prefix}{eval_output_dir}"
    return result


def _preflight_via_d1(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    python_executable: str,
    diagnostic_root: Path,
    eval_output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Reuse the canonical D1 preflight without entering its execute path."""
    validation_spec = json.loads(json.dumps(spec))
    validation_spec["name"] = f"sim_d2_preflight__{spec['name']}"
    validation_spec["extra_overrides"] = [
        *spec.get("extra_overrides", []),
        ROOT_XY_TERM_OVERRIDE,
        ROOT_XY_CALLBACK_OVERRIDE,
    ]
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".preflight-", dir=diagnostic_root
    ) as temporary_output:
        validation_spec["output_dir"] = temporary_output
        validation = run_sim_d1_all_motion_eval(
            validation_spec,
            repo_root=repo_root,
            python_executable=python_executable,
            dry_run=True,
        )
        command = _retarget_eval_output(validation["command"], eval_output_dir)
    return validation, command


def _archive_d2_outputs(
    paths: dict[str, Path], *, output_dir: Path, attempt_id: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    archive_dir = output_dir / "archive" / attempt_id / "prior"
    for label, path in paths.items():
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"refusing to archive non-file D2 output: {path}")
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{label}__{path.name}"
        if destination.exists():
            raise ValueError(f"refusing to overwrite D2 archive: {destination}")
        snapshot = _file_snapshot(path)
        path.rename(destination)
        records.append(
            {
                "label": label,
                "original_path": str(path),
                "archived_path": str(destination),
                "snapshot": snapshot,
            }
        )
    return records


def _quarantine_metrics(
    metrics_path: Path, *, output_dir: Path, attempt_id: str, reason: str
) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    destination_dir = output_dir / "archive" / attempt_id / "failed_attempt"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "metrics_eval.json"
    metrics_path.rename(destination)
    return {
        "reason": reason,
        "original_path": str(metrics_path),
        "archived_path": str(destination),
        "snapshot": _file_snapshot(destination),
    }


def _real_vector(
    table: dict[str, Any], field: str, count: int, *, nonnegative: bool = False
) -> list[float]:
    values = table.get(field)
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{field} must be a list of length {count}")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field}[{index}] must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field}[{index}] must be finite")
        if nonnegative and numeric < 0.0:
            raise ValueError(f"{field}[{index}] must be nonnegative")
        result.append(numeric)
    return result


def _validate_root_xy_metrics(
    metrics_eval: dict[str, Any], *, expected_motion_keys: list[str]
) -> dict[str, list[Any]]:
    """Validate exact callback telemetry and frozen coverage/order."""
    threshold = metrics_eval.get("eval/root_xy_diagnostic_threshold_m")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError("metrics missing numeric root XY diagnostic threshold")
    if float(threshold) != ROOT_XY_THRESHOLD_M:
        raise ValueError(
            f"root XY threshold must be exactly {ROOT_XY_THRESHOLD_M}, got {threshold}"
        )
    if metrics_eval.get("eval/root_xy_diagnostic_exact_guard_samples") is not True:
        raise ValueError("root XY telemetry was not sampled from the exact guard tensor")

    table = metrics_eval.get("eval/all_metrics_dict")
    if not isinstance(table, dict):
        raise ValueError("metrics missing eval/all_metrics_dict")
    if table.get("motion_keys") != expected_motion_keys:
        raise ValueError("metrics motion_keys do not exactly match frozen coverage order")
    count = len(expected_motion_keys)
    mean = _real_vector(table, "root_xy_error_mean_m", count, nonnegative=True)
    p95 = _real_vector(table, "root_xy_error_p95_m", count, nonnegative=True)
    maximum = _real_vector(table, "root_xy_error_max_m", count, nonnegative=True)
    hit = table.get("root_xy_guard_hit")
    frame = table.get("root_xy_first_crossing_frame")
    progress = _real_vector(table, "root_xy_first_crossing_progress", count)
    if not isinstance(hit, list) or len(hit) != count or not all(
        isinstance(value, bool) for value in hit
    ):
        raise ValueError(f"root_xy_guard_hit must be a boolean list of length {count}")
    if not isinstance(frame, list) or len(frame) != count or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= -1
        for value in frame
    ):
        raise ValueError(
            f"root_xy_first_crossing_frame must be an integer list of length {count}"
        )

    for index in range(count):
        if mean[index] > maximum[index] or p95[index] > maximum[index]:
            raise ValueError(f"root XY aggregate ordering is invalid at motion {index}")
        if hit[index]:
            if frame[index] < 0 or not 0.0 < progress[index] <= 1.0:
                raise ValueError(f"root XY crossing metadata is invalid at motion {index}")
            if maximum[index] <= ROOT_XY_THRESHOLD_M:
                raise ValueError(f"root XY guard hit lacks a strict crossing at motion {index}")
        elif (
            frame[index] != -1
            or progress[index] != -1.0
            or maximum[index] > ROOT_XY_THRESHOLD_M
        ):
            raise ValueError(f"root XY no-crossing sentinel is invalid at motion {index}")

    return {
        "root_xy_error_mean_m": mean,
        "root_xy_error_p95_m": p95,
        "root_xy_error_max_m": maximum,
        "root_xy_guard_hit": list(hit),
        "root_xy_first_crossing_frame": list(frame),
        "root_xy_first_crossing_progress": progress,
    }


def _build_diagnostic_artifact(
    *,
    spec: dict[str, Any],
    telemetry: dict[str, list[Any]],
    metrics_path: Path,
    coverage: dict[str, Any],
    dataset_manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    attempt_id: str,
    canonical_d1_output_dir: Path,
    canonical_before: dict[str, Any],
    canonical_after: dict[str, Any],
) -> dict[str, Any]:
    keys = coverage["motion_keys"]
    per_motion = []
    for index, motion_key in enumerate(keys):
        per_motion.append(
            {
                "motion_key": motion_key,
                **{field: telemetry[field][index] for field in ROOT_XY_PER_MOTION_FIELDS},
            }
        )
    hit_count = sum(telemetry["root_xy_guard_hit"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sim_d2_root_xy_diagnostic",
        "scientific_status": "diagnostic_only_not_an_effect_experiment",
        "eligible_for_effect_experiment": False,
        "attempt_id": attempt_id,
        "name": spec["name"],
        "dataset": spec["dataset"],
        "root_xy_threshold_m": ROOT_XY_THRESHOLD_M,
        "definitions": {
            "guard": "world-frame root XY L2 error > 0.25 m",
            "percentile": "linear interpolation",
            "first_crossing_frame": "zero-based evaluated-sample index; -1 means none",
            "first_crossing_progress": "(frame + 1) / (motion_frames - 1); -1.0 means none",
        },
        "summary": {
            "motion_count": len(keys),
            "guard_hit_count": hit_count,
            "guard_hit_fraction": hit_count / len(keys),
        },
        "per_motion": per_motion,
        "source": {
            "metrics_eval_path": str(metrics_path),
            "metrics_eval_sha256": _sha256(metrics_path),
            "checkpoint": checkpoint,
            "coverage_manifest": coverage,
            "dataset_manifest": dataset_manifest,
        },
        "canonical_d1_immutability": {
            "path": str(canonical_d1_output_dir),
            "before": _snapshot_summary(canonical_before),
            "after": _snapshot_summary(canonical_after),
            "unchanged": canonical_before == canonical_after,
        },
        "safeguards": {
            "d1_classifier_invoked": False,
            "diagnostic_output_isolated": True,
            "z_height_guard_preserved": True,
            "root_xy_guard_added": True,
            "full_sequences": True,
            "frozen_coverage_order_verified": True,
            "exact_guard_tensor_telemetry": True,
        },
    }


def run_sim_d2_root_xy_diagnostic(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    repo_root: Path,
    python_executable: str = sys.executable,
    dry_run: bool,
) -> dict[str, Any]:
    """Materialize an isolated D2 plan or execute the diagnostic."""
    _validate_spec(spec)
    _validate_source_contract(spec)
    repo_root = repo_root.expanduser().resolve()
    output_dir = _resolve_path(output_dir, repo_root)
    canonical_d1_output_dir = _resolve_path(spec["output_dir"], repo_root)
    diagnostic_root = _validate_isolated_output_path(
        output_dir=output_dir,
        canonical_d1_output_dir=canonical_d1_output_dir,
        repo_root=repo_root,
    )
    canonical_before = _tree_snapshot(canonical_d1_output_dir)

    eval_output_dir = output_dir / "eval_metrics"
    metrics_path = eval_output_dir / "metrics_eval.json"
    plan_path = output_dir / ("dry_run_plan.json" if dry_run else "run_plan.json")
    result_path = output_dir / "result.json"
    artifact_path = output_dir / "root_xy_diagnostic.json"
    eval_log_path = output_dir / "eval.log"
    attempt_id = None if dry_run else _new_attempt_id()

    validation, command = _preflight_via_d1(
        spec,
        repo_root=repo_root,
        python_executable=python_executable,
        diagnostic_root=diagnostic_root,
        eval_output_dir=eval_output_dir,
    )
    canonical_after_preflight = _tree_snapshot(canonical_d1_output_dir)
    if canonical_after_preflight != canonical_before:
        raise RuntimeError("canonical SIM-D1 output changed during D2 preflight")

    full_sequences = all(
        "max_eval_steps" not in value.lower() or value.lower().endswith("=null")
        for value in command
    )
    term_added = command.count(ROOT_XY_TERM_OVERRIDE) == 1
    callback_added = command.count(ROOT_XY_CALLBACK_OVERRIDE) == 1
    z_guard_preserved = command.count("+manager_env/terminations=tracking/eval") == 1
    contract_errors = list(validation["preflight_errors"])
    if not full_sequences:
        contract_errors.append("D2 command contains a finite max_eval_steps")
    if not term_added:
        contract_errors.append("D2 command does not contain exactly one root XY guard")
    if not callback_added:
        contract_errors.append("D2 command does not contain exactly one root XY callback override")
    if not z_guard_preserved:
        contract_errors.append("D2 command does not preserve the tracking/eval Z guard")

    ready = validation["ready_to_execute"] and not contract_errors
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sim_d2_root_xy_diagnostic_plan",
        "scientific_status": "diagnostic_only_not_an_effect_experiment",
        "eligible_for_effect_experiment": False,
        "mode": "dry_run" if dry_run else "execute",
        "attempt_id": attempt_id,
        "name": spec["name"],
        "dataset": spec["dataset"],
        "repo_root": str(repo_root),
        "command": command,
        "command_display": shlex.join(command),
        "root_xy_threshold_m": ROOT_XY_THRESHOLD_M,
        "checkpoint": validation["checkpoint"],
        "datasets": validation["datasets"],
        "coverage_manifest": validation["coverage_manifest"],
        "dataset_manifest": validation["dataset_manifest"],
        "motion_order": validation["motion_order"],
        "canonical_d1_output": {
            "path": str(canonical_d1_output_dir),
            "snapshot": _snapshot_summary(canonical_before),
        },
        "outputs": {
            "directory": str(output_dir),
            "metrics_eval_json": str(metrics_path),
            "diagnostic_json": str(artifact_path),
            "result_json": str(result_path),
            "eval_log": str(eval_log_path),
        },
        "safeguards": {
            "d1_classifier_invoked": False,
            "diagnostic_output_isolated": True,
            "canonical_d1_output_unchanged": True,
            "z_height_guard_preserved": z_guard_preserved,
            "root_xy_guard_added": term_added,
            "root_xy_callback_enabled": callback_added,
            "full_sequences": full_sequences,
            "frozen_coverage_order_verified": (
                validation["safeguards"]["coverage_order_exact_filter_verified"]
            ),
            "checkpoint_hash_verified": (
                validation["checkpoint"]["actual_sha256"]
                == validation["checkpoint"]["expected_sha256"]
            ),
            "paired_dataset_content_hashed": validation["safeguards"][
                "paired_dataset_content_is_hashed"
            ],
        },
        "preflight_errors": contract_errors,
        "ready_to_execute": ready,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        _write_json(plan_path, plan)
        return {**plan, "plan_path": str(plan_path), "exit_code": 0}

    assert attempt_id is not None
    if not ready:
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sim_d2_root_xy_diagnostic_result",
            "ok": False,
            "status": "preflight_failed",
            "eligible_for_effect_experiment": False,
            "attempt_id": attempt_id,
            "errors": contract_errors,
            "result_json": str(result_path),
            "exit_code": 2,
        }
        _write_json(result_path, result)
        return result

    archived = _archive_d2_outputs(
        {
            "metrics_eval": metrics_path,
            "diagnostic": artifact_path,
            "result": result_path,
            "eval_log": eval_log_path,
            "run_plan": plan_path,
        },
        output_dir=output_dir,
        attempt_id=attempt_id,
    )
    plan["archived_prior_outputs"] = archived
    _write_json(plan_path, plan)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "HYDRA_FULL_ERROR": "1",
            "LOGURU_LEVEL": "ERROR",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    timed_out = False
    with eval_log_path.open("w", encoding="utf-8") as eval_log:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(spec["timeout_seconds"]),
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            eval_log.write(
                f"\n[SIM_D2_TIMEOUT] timed out after {spec['timeout_seconds']}s: {exc}\n"
            )

    base_result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sim_d2_root_xy_diagnostic_result",
        "eligible_for_effect_experiment": False,
        "attempt_id": attempt_id,
        "name": spec["name"],
        "dataset": spec["dataset"],
        "eval_returncode": returncode,
        "timed_out": timed_out,
        "plan_path": str(plan_path),
        "result_json": str(result_path),
        "diagnostic_json": str(artifact_path),
        "archived_prior_outputs": archived,
    }
    if returncode != 0:
        quarantine = _quarantine_metrics(
            metrics_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="evaluation_returned_nonzero_or_timed_out",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "evaluation_failed",
            "quarantined_metrics": quarantine,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    metrics_snapshot = _file_snapshot(metrics_path)
    if metrics_snapshot is None:
        result = {
            **base_result,
            "ok": False,
            "status": "metrics_missing",
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    canonical_after_eval = _tree_snapshot(canonical_d1_output_dir)
    if canonical_after_eval != canonical_before:
        quarantine = _quarantine_metrics(
            metrics_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="canonical_d1_output_changed_during_d2",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "canonical_d1_output_changed",
            "quarantined_metrics": quarantine,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    try:
        metrics_eval = _load_json_object(metrics_path)
        telemetry = _validate_root_xy_metrics(
            metrics_eval,
            expected_motion_keys=validation["coverage_manifest"]["motion_keys"],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        quarantine = _quarantine_metrics(
            metrics_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="root_xy_telemetry_validation_failed",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "telemetry_validation_failed",
            "error": str(exc),
            "quarantined_metrics": quarantine,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    canonical_final = _tree_snapshot(canonical_d1_output_dir)
    if canonical_final != canonical_before:
        quarantine = _quarantine_metrics(
            metrics_path,
            output_dir=output_dir,
            attempt_id=attempt_id,
            reason="canonical_d1_output_changed_during_artifact_validation",
        )
        result = {
            **base_result,
            "ok": False,
            "status": "canonical_d1_output_changed",
            "quarantined_metrics": quarantine,
            "exit_code": 1,
        }
        _write_json(result_path, result)
        return result

    artifact = _build_diagnostic_artifact(
        spec=spec,
        telemetry=telemetry,
        metrics_path=metrics_path,
        coverage=validation["coverage_manifest"],
        dataset_manifest=validation["dataset_manifest"],
        checkpoint=validation["checkpoint"],
        attempt_id=attempt_id,
        canonical_d1_output_dir=canonical_d1_output_dir,
        canonical_before=canonical_before,
        canonical_after=canonical_final,
    )
    _write_json(artifact_path, artifact)
    result = {
        **base_result,
        "ok": True,
        "status": "diagnostic_complete",
        "root_xy_threshold_m": ROOT_XY_THRESHOLD_M,
        "motion_count": artifact["summary"]["motion_count"],
        "guard_hit_count": artifact["summary"]["guard_hit_count"],
        "artifact_sha256": _sha256(artifact_path),
        "exit_code": 0,
    }
    _write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Frozen SIM-D1 spec")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    try:
        result = run_sim_d2_root_xy_diagnostic(
            _load_json_object(args.spec),
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            python_executable=args.python_executable,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"wrote SIM-D2 dry-run plan to {result['plan_path']}")
        print(f"ready_to_execute={result['ready_to_execute']}")
        for error in result["preflight_errors"]:
            print(f"preflight: {error}")
    else:
        print(
            f"SIM-D2 status={result['status']} ok={result['ok']} "
            f"result={result['result_json']}"
        )
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
