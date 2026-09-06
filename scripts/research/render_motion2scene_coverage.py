#!/usr/bin/env python3
"""Summarize completed finite reference maps without changing the registered experiments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    report = ROOT / "docs/motion2scene"
    sources = sorted({r["carrier_seed"] for r in data["comparisons"]})
    arms = sorted({r["arm"] for r in data["comparisons"]})
    summary, source_rows = {}, []
    for arm in [*arms, "analytic_pattern17"]:
        rows = (
            [{"carrier_seed": r["carrier_seed"], **r["coverage"]} for r in data["analytic"]]
            if arm == "analytic_pattern17"
            else [r for r in data["comparisons"] if r["arm"] == arm]
        )
        for source in sources:
            group = [r for r in rows if r["carrier_seed"] == source]
            source_rows.append(
                {
                    "arm": arm,
                    "source": source,
                    "accepted": sum(r["accepted"] for r in group),
                    "requested": sum(r["requested"] for r in group),
                    "mean_reference_station_coverage": float(
                        np.mean(
                            [
                                r["reference_relative_station_coverage"]
                                for r in group
                                if not r["empty_reference_map"]
                            ]
                        )
                    ),
                }
            )
        summary[arm] = {
            "accepted": sum(r["accepted"] for r in rows),
            "requested": sum(r["requested"] for r in rows),
            "mean_reference_station_coverage": float(
                np.mean(
                    [
                        r["reference_relative_station_coverage"]
                        for r in rows
                        if not r["empty_reference_map"]
                    ]
                )
            ),
            "empty_reference_jobs": sum(r["empty_reference_map"] for r in rows),
            "accepted_bins_outside_reference_support": sum(
                len(r["accepted_bins_outside_reference_support"]) for r in rows
            ),
            "online_queries_per_eight_requests": 2260
            + (4624 if "17" in arm else 1360 if "5" in arm else 0),
        }
    rng = np.random.default_rng(9062026)
    contrasts = {}
    for a, b in [("hybrid600_raw", "original_raw"), ("analytic_pattern17", "original_pattern17")]:
        difference = np.array(
            [
                next(r for r in source_rows if r["source"] == s and r["arm"] == a)[
                    "mean_reference_station_coverage"
                ]
                - next(r for r in source_rows if r["source"] == s and r["arm"] == b)[
                    "mean_reference_station_coverage"
                ]
                for s in sources
            ]
        )
        bootstrap = difference[rng.integers(0, len(sources), (10000, len(sources)))].mean(1)
        contrasts[f"{a}_minus_{b}"] = {
            "paired_source_mean": float(difference.mean()),
            "source_bootstrap_95_percentile_interval": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
        }
    training = [r for r in data["maps"] if r["role"] == "train"]
    coverage = [r["teacher_support"]["reference_relative_station_coverage"] for r in training]
    teacher = {
        "retained_points": sum(r["teacher_support"]["requested"] for r in training),
        "training_cases": len(training),
        "cases_missing_reference_station_bins": sum(v is not None and v < 1 for v in coverage),
        "empty_reference_cases": sum(v is None for v in coverage),
        "mean_reference_station_support": float(np.mean([v for v in coverage if v is not None])),
        "rows": [
            {k: r[k] for k in ("case_id", "carrier_seed", "passing_centres", "teacher_support")}
            for r in training
        ],
    }
    compact = {
        "source": artifact(args.result),
        "protocol": data["registration"],
        "maps": len(data["maps"]),
        "grid_centres": sum(r["centres"] for r in data["maps"]),
        "map_audit_queries": data["map_audit_queries"],
        "map_crosscheck_queries": data["map_crosscheck_queries"],
        "elapsed_seconds": data["elapsed_seconds"],
        "gpu_seconds": 0,
        "summary": summary,
        "source_rows": source_rows,
        "coverage_contrasts": contrasts,
        "teacher": teacher,
        "execution_result": None,
        "limits": [
            "finite proxy geometry, passing centres only",
            "previously observed development sources",
            "historical learned outputs; current analytic outputs",
            "analytic deterministic; one job per case",
            "station-bin support, not feasible volume or natural scene frequency",
        ],
    }
    (report / "evidence/coverage-diagnostic.json").write_text(json.dumps(compact, indent=2) + "\n")

    old_path = Path(
        json.loads((args.result.parent / "registration.json").read_text())["existing_outputs"][
            "path"
        ]
    )
    old = json.loads(old_path.read_text())
    dev = [r for r in data["maps"] if r["role"] == "development"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True, sharey=True)
    for ax, row in zip(axes.flat, dev):
        raw = np.load(row["raw"]["path"])
        points, passing = raw["scenes"], raw["passing"]
        ax.scatter(*points.T, c="#e6e6e6", s=3)
        ax.scatter(*points[passing].T, color="#228a59", s=25, marker="s")
        for arm, color, marker in [
            ("original_raw", "#3668b0", "o"),
            ("hybrid600_raw", "#d27621", "x"),
        ]:
            group = [r for r in old["rows"] if r["case_id"] == row["case_id"] and r["arm"] == arm]
            points = np.concatenate([np.array(r["scenes"])[r["per_proposal_valid"]] for r in group])
            if len(points):
                ax.scatter(*points.T, c=color, s=18, marker=marker, alpha=0.65)
        ax.set_title(row["case_id"], fontsize=10)
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(1.1, 1.45)
    fig.supxlabel("Route station (fixed world placement)")
    fig.supylabel("Beam underside (m)")
    fig.suptitle(
        "Passing grid centres (green); accepted original raw (blue) and hybrid raw (orange)\n"
        "24 requests per model/case across three fitting seeds; finite proxy geometry",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
    fig.savefig(report / "assets/coverage-reference.svg")
    fig.savefig(report / "assets/coverage-reference.png", dpi=120)
    plt.close(fig)
    print(
        json.dumps(
            {
                "summary": summary,
                "teacher": {k: v for k, v in teacher.items() if k != "rows"},
                "coverage_contrasts": contrasts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
