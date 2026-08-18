#!/usr/bin/env python3
"""Plan, launch one cell, or aggregate the LACE SONIC-Lite throughput sweep.

With no action flag this command is plan-only.  GPU work requires an explicit
``--launch-cell`` and an already written, launch-ready plan.  Cell outputs are
never resumed or overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_text = str(REPO_ROOT)
if repo_root_text in sys.path:
    sys.path.remove(repo_root_text)
sys.path.insert(0, repo_root_text)

from gear_sonic.research.lace.lite_init import (  # noqa: E402
    LiteInitializationError,
    materialize_lite_initialization,
    verify_lite_initialization,
)
from gear_sonic.research.lace.throughput import (  # noqa: E402
    EXPECTED_ENV_COUNTS,
    ThroughputLaunchError,
    ThroughputProtocolError,
    aggregate_throughput_plan,
    build_throughput_plan,
    launch_throughput_cell,
    load_json_object,
    materialize_partition_subset,
    resolve_materialization_targets,
    validate_plan,
    write_new_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--launch-cell", type=int, choices=EXPECTED_ENV_COUNTS)
    actions.add_argument("--aggregate", action="store_true")
    actions.add_argument("--materialize-subset", action="store_true")
    actions.add_argument("--materialize-init", action="store_true")
    actions.add_argument("--verify-init", action="store_true")
    parser.add_argument("--report", type=Path, help="Required with --aggregate")
    return parser


def _require_plan_path(plan: Path | None) -> Path:
    if plan is None:
        raise ThroughputProtocolError("--plan is required for planning, launch, and aggregation")
    return plan.resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_json_object(protocol_path)

    if args.materialize_init or args.verify_init:
        if args.report is not None or args.plan is not None:
            raise ThroughputProtocolError(
                "--plan/--report are not valid with initialization actions"
            )
        targets = resolve_materialization_targets(protocol, protocol_path=protocol_path)
        repo_root = targets["repo_root"]
        bundle = targets["initialization_dir"]
        seed = targets["seed"]
        receipt = (
            materialize_lite_initialization(
                repo_root=repo_root,
                destination=bundle,
                seed=seed,
            )
            if args.materialize_init
            else verify_lite_initialization(
                bundle,
                repo_root=repo_root,
                expected_seed=seed,
            )
        )
        print(
            json.dumps(
                {
                    "bundle": str(bundle),
                    "environment_constructed": receipt["environment_constructed"],
                    "model_spec_sha256": receipt["model_spec_sha256"],
                    "policy_state_sha256": receipt["policy"]["state_sha256"],
                    "value_state_sha256": receipt["value"]["state_sha256"],
                    "verified": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.materialize_subset:
        if args.report is not None:
            raise ThroughputProtocolError("--report is only valid with --aggregate")
        targets = resolve_materialization_targets(protocol, protocol_path=protocol_path)
        split_path = targets["split_path"]
        partition = targets["partition"]
        subset_root = targets["subset_root"]
        manifest = materialize_partition_subset(
            split_path,
            partition=partition,
            destination=subset_root,
        )
        print(
            json.dumps(
                {
                    "subset_root": str(subset_root),
                    "motion_count": manifest["motion_count"],
                    "subset_sha256": manifest["subset_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.launch_cell is not None or args.aggregate:
        plan_path = _require_plan_path(args.plan)
        if not plan_path.is_file():
            raise ThroughputProtocolError(f"existing plan is required: {plan_path}")
        plan = load_json_object(plan_path)
        validate_plan(plan)
        if Path(plan["protocol_path"]).resolve() != protocol_path:
            raise ThroughputProtocolError("plan protocol path differs from --protocol")
        if args.launch_cell is not None:
            if args.report is not None:
                raise ThroughputProtocolError("--report is only valid with --aggregate")
            result = launch_throughput_cell(plan, num_envs=args.launch_cell)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.report is None:
            raise ThroughputProtocolError("--aggregate requires --report")
        report = aggregate_throughput_plan(plan)
        write_new_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.report is not None:
        raise ThroughputProtocolError("--report is only valid with --aggregate")
    plan_path = _require_plan_path(args.plan)
    plan = build_throughput_plan(protocol, protocol_path=protocol_path)
    if not plan["launch_ready"]:
        print(json.dumps({"launch_ready": False, "errors": plan["readiness_errors"]}, indent=2))
        return 2
    write_new_json(plan_path, plan)
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "launch_ready": True,
                "cells": [cell["cell_id"] for cell in plan["cells"]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ThroughputProtocolError,
        ThroughputLaunchError,
        LiteInitializationError,
        FileExistsError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
