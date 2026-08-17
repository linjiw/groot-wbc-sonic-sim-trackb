#!/usr/bin/env python3
"""Record both verdicts for every episode, and list the ones a re-grade flipped.

A re-grade changes the verdict on identical data, so the old verdict is evidence and
overwriting it destroys the audit trail. This writes a record per episode carrying the
pre-policy verdict, the post-policy verdict, the gate version, and the reasons on each side.

The flipped-to-accepted list is the sample worth human review: those are the episodes the
corpus gained without any new physics, so they are exactly where a false accept would hide.
That review is not overhead on top of the calibration pilot -- it *is* the false-accept
audit, run on the most informative subset instead of a random one.

Usage::

    python scripts/research/audit_gate_regrade.py --root /data/.../groot-wbc-kimodo-m0 \\
        --json regrade_audit.json [--sample 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.gate_policy import (  # noqa: E402
    SCENE_AROUND_MOTION,
    apply_policy,
)
from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Bump when any gate threshold or policy changes, so a stored verdict says which rules
#: produced it. A verdict without its gate version cannot be compared across time.
GATE_VERSION = "2026-08-17.policy-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=15, help="flipped episodes to list")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records: list[dict] = []
    for path in sorted(args.root.rglob("*.trajectory.pkl")):
        episode_id = path.parent.parent.name
        try:
            with path.open("rb") as handle:
                raw = pickle.load(handle)  # noqa: S301 - local recorder artifact
            payload, recovered = best_evaluable_payload(raw)
        except Exception as error:  # noqa: BLE001 - a bad file must not end the audit
            records.append(
                {
                    "episode_id": episode_id,
                    "gate_version": GATE_VERSION,
                    "unevaluable": True,
                    "error": f"{type(error).__name__}: {error}",
                    "source": str(path.parent.parent),
                }
            )
            continue

        report = evaluate_locomotion_trajectory(payload)
        if report.errors:
            records.append(
                {
                    "episode_id": episode_id,
                    "gate_version": GATE_VERSION,
                    "unevaluable": True,
                    "error": report.errors[0],
                    "source": str(path.parent.parent),
                }
            )
            continue

        graded = apply_policy(tuple(report.rejection_reasons), payload, SCENE_AROUND_MOTION)
        records.append(
            {
                "episode_id": episode_id,
                "gate_version": GATE_VERSION,
                "unevaluable": False,
                "recovered_from_split": bool(recovered),
                "before": {
                    "accepted": bool(report.accepted),
                    "reasons": list(report.rejection_reasons),
                },
                "after": {
                    "accepted": bool(graded.accepted),
                    "reasons": list(graded.rejection_reasons),
                    "policy": graded.policy,
                    "demoted": list(graded.demoted_failures),
                },
                "diagnostics": dict(graded.diagnostics),
                "source": str(path.parent.parent),
            }
        )

    graded_records = [r for r in records if not r["unevaluable"]]
    to_accepted = [
        r for r in graded_records if not r["before"]["accepted"] and r["after"]["accepted"]
    ]
    to_rejected = [
        r for r in graded_records if r["before"]["accepted"] and not r["after"]["accepted"]
    ]

    rng = random.Random(args.seed)
    sample = rng.sample(to_accepted, min(args.sample, len(to_accepted)))

    payload_out = {
        "gate_version": GATE_VERSION,
        "episodes": len(records),
        "evaluable": len(graded_records),
        "accepted_before": sum(1 for r in graded_records if r["before"]["accepted"]),
        "accepted_after": sum(1 for r in graded_records if r["after"]["accepted"]),
        "flipped_to_accepted": len(to_accepted),
        "flipped_to_rejected": len(to_rejected),
        "review_sample": [r["episode_id"] for r in sample],
        "records": records,
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(payload_out, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"gate version {GATE_VERSION}")
    print(f"  evaluable {len(graded_records)} of {len(records)}")
    print(
        f"  accepted {payload_out['accepted_before']} -> {payload_out['accepted_after']}"
        f"   flipped to accepted {len(to_accepted)}, to rejected {len(to_rejected)}"
    )
    if to_rejected:
        print("  NOTE: some episodes lost acceptance; those are the ones to check first")
        for record in to_rejected[:5]:
            print(f"     {record['episode_id'][:56]}  {record['after']['reasons']}")
    print(f"\n  human-review sample ({len(sample)} of {len(to_accepted)} flipped to accepted):")
    for record in sample:
        demoted = ",".join(record["after"]["demoted"]) or "-"
        drift = record["diagnostics"].get("drift_rate_mps")
        drift_text = f"{float(drift):.4f}" if isinstance(drift, (int, float)) else "n/a"
        print(f"     {record['episode_id'][:50]:52s} demoted={demoted[:36]:38s} drift={drift_text}")
    print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
