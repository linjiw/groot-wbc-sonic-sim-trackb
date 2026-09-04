#!/usr/bin/env python3
"""Execute one self-approved LFH manifest serially with resumable run records."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DRIVER = REPO_ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"
GOVERNANCE = "docs/hallucination/GOVERNANCE.md"
DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
DAILY_GPU_HOURS = 8.0


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    path.chmod(0o664)


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: {actual} != {expected}")


def verify_cell(cell: dict) -> None:
    motion = cell["motion"]
    motion_path = Path(motion["path"])
    provenance_path = Path(motion["conversion_provenance"])
    verify_hash(motion_path, motion["sha256"], f"{cell['cell_id']} motion")
    verify_hash(
        provenance_path,
        motion["conversion_provenance_sha256"],
        f"{cell['cell_id']} provenance",
    )
    expected_start = cell["scene_start_xyz_expected"]
    if motion["scene_start_xyz"] != expected_start:
        raise ValueError(f"{cell['cell_id']}: cell/motion scene-start mismatch")
    provenance = json.loads(provenance_path.read_text())
    if "scene_start_xyz" not in provenance:
        raise ValueError(
            f"{cell['cell_id']}: provenance does not independently record scene_start_xyz"
        )
    if provenance["scene_start_xyz"] != expected_start:
        raise ValueError(f"{cell['cell_id']}: provenance scene-start mismatch")

    scene = cell["scene"]
    if scene["scene_id"] == "plane":
        if scene["path"] is not None or scene["sha256"] is not None:
            raise ValueError(f"{cell['cell_id']}: plane unexpectedly has a scene artifact")
    else:
        verify_hash(REPO_ROOT / scene["path"], scene["sha256"], f"{cell['cell_id']} scene")

    output = Path(cell["output"])
    if not output.is_absolute():
        raise ValueError(f"{cell['cell_id']}: output must be absolute")
    protected = {"scene_first_v1", "scene_first_v2", "claim5", "sweepcf_release"}
    if protected.intersection(output.parts):
        raise ValueError(f"{cell['cell_id']}: protected output path {output}")


def _resolve_artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_manifest_artifacts(manifest: dict) -> None:
    """Verify the code, controller, and prediction record pinned by a manifest."""
    implementation = manifest.get("implementation", {})
    for label in ("rollout_driver", "manifest_driver", "checkpoint"):
        artifact = implementation.get(label)
        if not isinstance(artifact, dict):
            raise ValueError(f"manifest implementation.{label} is missing")
        verify_hash(
            _resolve_artifact_path(artifact["path"]),
            artifact["sha256"],
            f"implementation {label}",
        )
    prediction = manifest.get("registered_predictions")
    if not isinstance(prediction, dict):
        raise ValueError("manifest registered_predictions must be a hash-pinned artifact")
    verify_hash(
        _resolve_artifact_path(prediction["path"]),
        prediction["sha256"],
        "registered predictions",
    )
    reference_gate = manifest.get("eligibility", {}).get("reference_gate")
    if reference_gate is not None:
        if not isinstance(reference_gate, dict):
            raise ValueError("manifest eligibility.reference_gate must be a hash-pinned artifact")
        verify_hash(
            _resolve_artifact_path(reference_gate["path"]),
            reference_gate["sha256"],
            "eligibility reference gate",
        )


def resolve_python(manifest: dict, override: Path | None) -> Path:
    """Resolve the rollout interpreter, preferring an explicit CLI override.

    Physics manifests pin the environment used to execute their cells.  A stale developer-machine
    default must not silently replace that recorded environment.
    """
    value = override
    if value is None:
        value = manifest.get("implementation", {}).get("python")
    if value is None:
        raise ValueError("manifest implementation.python is missing and --python was not supplied")
    path = _resolve_artifact_path(str(value))
    if not path.is_file():
        raise ValueError(f"rollout Python does not exist: {path}")
    return path


def capture_mode_arguments(policy: dict) -> list[str]:
    """Translate the registered capture mode into rollout-driver arguments."""
    if policy.get("runtime", {}).get("render_ego", True):
        return []
    return ["--trajectory-only"]


def free_gpu_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one GPU, found {len(values)}")
    return values[0]


def artifacts(output: Path) -> tuple[Path, Path, Path]:
    trajectories = sorted((output / "trajectories").glob("*.trajectory.pkl"))
    if len(trajectories) != 1:
        raise RuntimeError(f"expected exactly one trajectory, found {len(trajectories)}")
    log = output / "rollout.log"
    success = output / "success_manifest.json"
    if not log.exists() or "SONIC_EVAL_SUCCESS" not in log.read_text(errors="replace"):
        raise RuntimeError("missing SONIC_EVAL_SUCCESS marker")
    if not success.exists():
        raise RuntimeError("missing success_manifest.json")
    return trajectories[0], log, success


def score(cell: dict) -> dict:
    trajectory, log, success = artifacts(Path(cell["output"]))
    with trajectory.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted, hash-recorded local rollout
    if abs(float(payload.get("fps", 0.0)) - 50.0) > 1e-6:
        raise RuntimeError(f"recorded fps is {payload.get('fps')}, expected 50")
    if cell.get("verdict_policy") == "reference_trackability":
        payload, recovered = best_evaluable_payload(payload)
        report = evaluate_locomotion_trajectory(payload)
        if report.errors:
            raise RuntimeError(f"trajectory is unevaluable: {report.errors}")
        scientific = {
            "outcome": "accepted" if report.accepted else "rejected",
            "rejection_reasons": list(report.rejection_reasons),
            "demoted_failures": [],
            "frames": int(payload["total_frames"]),
            "recovered_from_split": recovered,
            "policy": "reference_trackability",
            "diagnostics": dict(report.diagnostics),
        }
    else:
        outcome = classify_episode(cell["cell_id"], payload)
        if outcome.outcome == "unevaluable":
            raise RuntimeError(f"trajectory is unevaluable: {outcome.errors}")
        scientific = {
            "outcome": outcome.outcome,
            "rejection_reasons": list(outcome.rejection_reasons),
            "demoted_failures": list(outcome.demoted_failures),
            "frames": outcome.frames,
            "recovered_from_split": outcome.recovered_from_split,
            "policy": outcome.policy,
            "diagnostics": outcome.diagnostics,
        }
    return {
        **scientific,
        "artifacts": {
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256(trajectory),
            "rollout_log": str(log),
            "rollout_log_sha256": sha256(log),
            "success_manifest": str(success),
            "success_manifest_sha256": sha256(success),
        },
    }


def run_with_process_group(command: list[str], *, timeout: int) -> int:
    """Run the rollout driver so a timeout kills the *whole* job, not just the wrapper.

    ``subprocess.run(timeout=...)`` kills its direct child. The driver is a shell script that
    execs Isaac in a further process, so on timeout the wrapper died while the Isaac process
    survived, orphaned, holding its VRAM on a shared card indefinitely -- measured once at 4.65 GB
    held for 35 minutes after the runner had already moved on and recorded the cell as an
    infrastructure failure. Running the driver in its own session and signalling the process group
    means the timeout reclaims the memory it was supposed to.
    """
    process = subprocess.Popen(command, cwd=REPO_ROOT, text=True, start_new_session=True)
    try:
        return int(process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        raise
    except BaseException:
        _terminate_group(process)
        raise


def _terminate_group(process: subprocess.Popen) -> None:
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    for signal_number, grace in ((signal.SIGTERM, 15.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(group, signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Override implementation.python from the manifest.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rollout_python = resolve_python(manifest, args.python)
    authorization = manifest.get("authorization", {})
    policy = manifest["execution_policy"]
    if policy.get("not_authorized") is not False:
        raise SystemExit("manifest is not approved")
    if (
        authorization.get("status") != "self_approved"
        or authorization.get("authority") != GOVERNANCE
    ):
        raise SystemExit("manifest lacks the LFH governance authorization record")
    projected = float(policy["cost_ceiling"]["gpu_hours_contended"])
    if projected > DAILY_GPU_HOURS:
        raise SystemExit(f"manifest projects {projected:.3f} GPU-h, beyond the daily budget")
    verify_manifest_artifacts(manifest)
    for cell in manifest["cells"]:
        verify_cell(cell)

    manifest_hash = sha256(args.manifest)
    if args.run_record.exists():
        record = json.loads(args.run_record.read_text())
        if record["manifest_sha256"] != manifest_hash:
            raise SystemExit("run record belongs to a different manifest hash")
    else:
        record = {
            "schema_version": "lfh_physics_run_v1",
            "experiment": manifest["experiment"],
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash,
            "authority": GOVERNANCE,
            "started_at": now(),
            "ended_at": None,
            "status": "preflight_passed",
            "claim5_isolation": authorization["claim5_isolation_verified"],
            "timing_override": policy["timing_override"],
            "budget": {
                "daily_limit_gpu_hours": DAILY_GPU_HOURS,
                "projected_gpu_hours": projected,
                "actual_contended_gpu_hours": 0.0,
            },
            "cells": {},
        }
        write_record(args.run_record, record)

    if args.dry_run:
        print(f"PREFLIGHT PASS: {len(manifest['cells'])} cells; {projected:.3f} GPU-h ceiling")
        return 0

    completed = record["cells"]
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        prior = completed.get(cell_id)
        if prior and prior.get("status") in {"completed", "skipped_dependency"}:
            continue
        dependency = cell.get("depends_on_acceptance_of")
        if (
            dependency
            and completed.get(dependency, {}).get("scientific", {}).get("outcome") != "accepted"
        ):
            completed[cell_id] = {
                "status": "skipped_dependency",
                "dependency": dependency,
                "recorded_at": now(),
            }
            write_record(args.run_record, record)
            continue

        output = Path(cell["output"])
        if output.exists() and any(output.iterdir()):
            try:
                scientific = score(cell)
            except Exception as error:
                record["status"] = "infrastructure_failure"
                record["ended_at"] = now()
                completed[cell_id] = {
                    "status": "infrastructure_failure",
                    "error": f"pre-existing incomplete output: {type(error).__name__}: {error}",
                }
                write_record(args.run_record, record)
                raise SystemExit(f"INFRASTRUCTURE FAILURE {cell_id}: {error}")
        else:
            available = free_gpu_mib()
            required = int(policy["runtime"]["free_gpu_mib_required"])
            if available < required:
                record["status"] = "yielded_gpu_contention"
                completed[cell_id] = {
                    "status": "not_started",
                    "free_gpu_mib": available,
                    "required_gpu_mib": required,
                }
                write_record(args.run_record, record)
                raise SystemExit(f"GPU contention: {available} MiB free, {required} required")
            command = [
                str(DRIVER),
                "--scene",
                cell["scene"]["scene_id"],
                "--motion",
                cell["motion"]["path"],
                "--out",
                cell["output"],
                "--python",
                str(rollout_python),
                "--checkpoint",
                str(_resolve_artifact_path(manifest["implementation"]["checkpoint"]["path"])),
            ]
            command.extend(capture_mode_arguments(policy))
            task = cell.get("task_prompt")
            if task:
                command.extend(["--task", task])
            scene_path = cell["scene"].get("path")
            if cell["scene"]["scene_id"] != "plane" and scene_path:
                scene_package = Path(scene_path)
                if not scene_package.is_absolute():
                    scene_package = REPO_ROOT / scene_package
                command.extend(["--scene-package", str(scene_package.parent)])
            for override in cell.get("hydra_overrides", []):
                command.extend(["--extra", override])
            started = time.monotonic()
            completed[cell_id] = {
                "status": "running",
                "started_at": now(),
                "free_gpu_mib_at_start": available,
                "command": command,
            }
            record["status"] = "running"
            write_record(args.run_record, record)
            try:
                returncode = run_with_process_group(
                    command,
                    timeout=int(policy["runtime"]["hang_timeout_seconds"]),
                )
                if returncode != 0:
                    raise RuntimeError(f"driver exited {returncode}")
                scientific = score(cell)
            except Exception as error:
                elapsed = time.monotonic() - started
                record["budget"]["actual_contended_gpu_hours"] += elapsed / 3600.0
                record["status"] = "infrastructure_failure"
                record["ended_at"] = now()
                completed[cell_id].update(
                    {
                        "status": "infrastructure_failure",
                        "ended_at": now(),
                        "elapsed_seconds": round(elapsed, 3),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                write_record(args.run_record, record)
                raise SystemExit(f"INFRASTRUCTURE FAILURE {cell_id}: {error}")
            elapsed = time.monotonic() - started
            record["budget"]["actual_contended_gpu_hours"] += elapsed / 3600.0
            completed[cell_id].update(
                {
                    "status": "completed",
                    "ended_at": now(),
                    "elapsed_seconds": round(elapsed, 3),
                    "scientific": scientific,
                }
            )

        expected = cell.get("expected_source_outcome")
        if expected is None:
            expected = cell.get("expectation", {}).get("outcome")
        completed[cell_id]["expected_outcome"] = expected
        completed[cell_id]["prediction_matches"] = (
            None if expected is None else scientific["outcome"] == expected
        )
        if manifest["experiment"] == "E1a" and scientific["outcome"] != expected:
            record["status"] = "stop_line_source_outcome_flip"
            record["ended_at"] = now()
            record["stop_line"] = {
                "cell_id": cell_id,
                "expected": expected,
                "observed": scientific["outcome"],
            }
            write_record(args.run_record, record)
            raise SystemExit(
                f"STOP LINE: {cell_id} expected {expected}, observed {scientific['outcome']}"
            )
        write_record(args.run_record, record)
        print(f"CELL COMPLETE {cell_id}: {scientific['outcome']}", flush=True)

    record["status"] = "completed"
    record["ended_at"] = now()
    write_record(args.run_record, record)
    print(
        f"BATCH COMPLETE {manifest['experiment']}: {len(completed)} cells; "
        f"{record['budget']['actual_contended_gpu_hours']:.3f} contended GPU-h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
