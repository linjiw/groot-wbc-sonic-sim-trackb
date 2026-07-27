#!/usr/bin/env python3
"""Run a bounded SONIC eval-only metric smoke from a JSON spec.

This harness validates SIM-M1: eval must complete, emit parseable final All: MPJPE
metrics, and produce a finite primary MPJPE without relying on training logs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from scripts.research.summarize_sonic_logs import (  # noqa: E402
    summarize_logs,
    write_markdown,
)

_REQUIRED_TOP_LEVEL = (
    "name",
    "goal",
    "repo_root",
    "dataset_robot",
    "dataset_smpl",
    "checkpoint",
    "eval",
    "success_criteria",
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_eval_smoke_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in _REQUIRED_TOP_LEVEL:
        if key not in spec or spec[key] in (None, ""):
            errors.append(f"missing {key}")
    eval_cfg = spec.get("eval")
    if not isinstance(eval_cfg, dict):
        errors.append("eval must be an object")
        return errors
    if eval_cfg.get("enabled") is False:
        errors.append("eval.enabled must be true")
    timeout = eval_cfg.get("timeout_seconds")
    if timeout is not None and int(timeout) <= 0:
        errors.append("eval.timeout_seconds must be positive")
    max_steps = eval_cfg.get("max_steps_per_sequence")
    if max_steps is not None and int(max_steps) <= 0:
        errors.append("eval.max_steps_per_sequence must be positive")
    return errors


def build_eval_command(spec: dict[str, Any]) -> str:
    eval_cfg = spec["eval"]
    if eval_cfg.get("command"):
        return str(eval_cfg["command"])

    max_sequences = int(eval_cfg.get("max_sequences", 1))
    max_steps = int(eval_cfg.get("max_steps_per_sequence", 200))
    num_envs = int(eval_cfg.get("num_envs", 2))
    headless = str(bool(eval_cfg.get("headless", True))).lower()
    return " ".join(
        [
            "WANDB_MODE=disabled HYDRA_FULL_ERROR=1 LOGURU_LEVEL=ERROR",
            "python gear_sonic/eval_agent_trl.py",
            f"+checkpoint={spec['checkpoint']}",
            f"+headless={headless}",
            "++eval_callbacks=im_eval",
            "++run_eval_loop=False",
            f"++num_envs={num_envs}",
            f"++callbacks.im_eval.max_eval_steps={max_steps}",
            f"++algo.config.eval.num_eval_episodes={max_sequences * num_envs}",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={spec['dataset_robot']}",
            f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={spec['dataset_smpl']}",
            f"++manager_env.commands.motion.motion_lib_cfg.max_unique_motions={max_sequences}",
            "+manager_env/terminations=tracking/eval",
        ]
    )


def _finite_primary_mpjpe(summary: dict[str, Any], metric_name: str) -> bool:
    eval_summary = summary.get("eval", {})
    if not isinstance(eval_summary, dict):
        return False
    all_metrics = eval_summary.get("all", {})
    if not isinstance(all_metrics, dict):
        return False
    value = all_metrics.get(metric_name)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def run_eval_metric_smoke(
    spec: dict[str, Any], *, output_dir: Path, repo_root: Path, dry_run: bool = False
) -> dict[str, Any]:
    errors = validate_eval_smoke_spec(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_log = output_dir / "eval.log"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    result_json = output_dir / "result.json"
    command = build_eval_command(spec)
    timeout_seconds = int(spec["eval"].get("timeout_seconds", 300))

    returncode: int | None = None
    timed_out = False
    if not dry_run:
        with eval_log.open("w", encoding="utf-8") as f:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(repo_root),
                    shell=True,
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = 124
                f.write(f"\n[SIM_M1_TIMEOUT] timed out after {timeout_seconds}s: {exc}\n")

    summary = summarize_logs(eval_log=eval_log) if eval_log.exists() else {"schema_version": 1}
    _write_json(summary_json, summary)
    write_markdown(summary_md, summary)

    success_criteria = spec.get("success_criteria", {})
    if not isinstance(success_criteria, dict):
        success_criteria = {}
    primary_metric = str(success_criteria.get("primary_mpjpe_metric", "mpjpe_g"))
    eval_summary_raw = summary.get("eval", {})
    eval_summary = eval_summary_raw if isinstance(eval_summary_raw, dict) else {}
    ok = (
        returncode == 0
        and not timed_out
        and bool(eval_summary.get("ok"))
        and _finite_primary_mpjpe(summary, primary_metric)
    )
    result = {
        "schema_version": 1,
        "kind": "sonic_eval_metric_smoke",
        "name": spec["name"],
        "goal": spec["goal"],
        "command": command,
        "eval_log": str(eval_log),
        "summary_json": str(summary_json),
        "returncode": returncode,
        "timed_out": timed_out,
        "primary_mpjpe_metric": primary_metric,
        "primary_mpjpe_value": eval_summary.get("all", {}).get(primary_metric),
        "eval_ok": bool(eval_summary.get("ok")),
        "ok": ok,
    }
    _write_json(result_json, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = _load_json(args.spec)
    repo_root = args.repo_root or Path(spec.get("repo_root", Path.cwd()))
    try:
        result = run_eval_metric_smoke(spec, output_dir=args.output_dir, repo_root=repo_root, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote eval metric smoke result to {args.output_dir / 'result.json'}")
    print(f"ok={result['ok']} eval_ok={result['eval_ok']} primary={result['primary_mpjpe_value']}")
    return 0 if result["ok"] or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
