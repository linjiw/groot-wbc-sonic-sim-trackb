#!/usr/bin/env python3
"""Build, CPU-compose, or receipt one fixed-distribution LACE RQ1 run.

This command intentionally has no launch action.  GPU execution remains gated
on the separate exclusive-GPU preflight and externally anchored one-attempt
registry; the generated plan contains the exact argv for that future runner.
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

from gear_sonic.research.lace.rq1_training import (  # noqa: E402
    RQ1TrainingError,
    build_rq1_training_plan,
    build_rq1_training_receipt,
    dry_compose_rq1_training_plan,
    load_json_object,
    validate_rq1_training_plan,
    validate_rq1_training_receipt,
    write_new_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-compose", action="store_true")
    actions.add_argument("--build-receipt", action="store_true")
    actions.add_argument("--verify-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    plan_path = args.plan.resolve()
    protocol = load_json_object(protocol_path)

    if args.dry_compose or args.build_receipt or args.verify_receipt is not None:
        if not plan_path.is_file():
            raise RQ1TrainingError(f"existing RQ1 plan is required: {plan_path}")
        plan = load_json_object(plan_path)
        validate_rq1_training_plan(plan)
        if Path(plan["protocol"]["path"]).resolve() != protocol_path:
            raise RQ1TrainingError("--protocol differs from the plan protocol")
        if args.dry_compose:
            readback = dry_compose_rq1_training_plan(plan)
            print(json.dumps(readback, indent=2, sort_keys=True))
            return 0
        if args.build_receipt:
            receipt = build_rq1_training_receipt(plan, plan_path=plan_path)
            receipt_path = Path(plan["runtime_artifacts"]["receipt"])
            write_new_json(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "arm_id": receipt["arm_id"],
                        "exact_budget_pass": receipt["exact_budget_pass"],
                        "receipt": str(receipt_path),
                        "receipt_sha256": receipt["receipt_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        receipt = load_json_object(args.verify_receipt.resolve())
        validate_rq1_training_receipt(receipt, plan)
        print(
            json.dumps(
                {
                    "arm_id": receipt["arm_id"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "verified": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    plan = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=plan_path,
    )
    readback = dry_compose_rq1_training_plan(plan)
    write_new_json(plan_path, plan)
    print(
        json.dumps(
            {
                "arm_id": plan["arm_id"],
                "cpu_dry_compose_pass": True,
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "readback_sha256": readback["readback_sha256"],
                "scientific_launch_ready": False,
                "remaining_external_gates": [
                    "exclusive GPU preflight",
                    "externally anchored append-only one-attempt registry",
                    "end-to-end runtime callback validation",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RQ1TrainingError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
