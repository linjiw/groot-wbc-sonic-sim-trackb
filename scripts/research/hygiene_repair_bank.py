#!/usr/bin/env python
"""Materialise a contact-repaired copy of a SONIC motion bank, one .pkl at a time.

WHY this script exists
----------------------
``gear_sonic.research.hygiene.screen`` tells us which reference clips demand forces no controller
could supply.  Knowing that is only half a result: the interesting experimental arm is not "drop the
bad clips" but "repair the cheaply repairable ones and keep them", because pruning shrinks the
corpus while repair preserves it.  Running that arm needs a *bank*, not a report -- a directory of
.pkl files a training config can point ``motion_file`` at without any other change.  This script
produces exactly that, plus the census that says what the operator did to every clip.

Three properties the output bank is built to have, each for a concrete downstream reason:

* **Complete.**  Every clip in the input bank appears in the output bank, repaired or not.  The
  repaired arm and the raw arm therefore have identical clip counts and identical motion keys, so a
  difference in training outcome cannot be a difference in corpus size.
* **Real files, never symlinks.**  ``gear_sonic/research/lace/throughput.py``'s
  ``verify_partition_subset`` resolves every entry with ``Path.resolve()`` and demands it equal the
  recorded source path, then re-checks the sha256.  A symlink farm passes neither test.  Clips that
  need no repair are copied byte-for-byte with :func:`shutil.copy2` so their digests still match the
  source bank exactly; only genuinely modified clips are re-serialised.
* **Resumable.**  A clip whose output .pkl *and* per-clip report both exist is skipped.  The census
  is rebuilt from the per-clip reports at the end, so an interrupted run resumes without redoing
  work and without producing a half-written census.

Sampling-rate note: the operator runs at the bank's native 30 fps, and the output bank is 30 fps, so
it is a drop-in replacement.  This is safe rather than merely convenient -- SONIC's load-time
30 -> 50 Hz resample lerps ``root_trans_offset``, and lerp is affine, so lerping a repaired root
equals lerping the raw root and then lerping the offset profile.  The projection commutes with the
resample up to the resample of the offset itself; it does not have to be redone at 50 Hz.

Usage
-----
    hygiene_repair_bank.py --bank BANK_DIR --out-bank OUT_DIR \
        --screen-dir SCREEN_JSON_DIR --out-reports REPORT_DIR --workers 8 [--limit N]

``--screen-dir`` is the screen's output; clips it reports as feasible are copied through without
paying for a second screen.  Without it (or with ``--repair-all``) every clip is screened here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.hygiene.motion_io import load_motion, save_motion  # noqa: E402
from gear_sonic.research.hygiene.repair import (  # noqa: E402
    REPAIR_REASONS,
    REPAIR_SCHEMA_VERSION,
    RepairBudget,
    repair_motion,
)
from gear_sonic.research.hygiene.screen import ScreenThresholds, load_model  # noqa: E402

#: Written into ``--out-reports`` once every clip has an output .pkl and a report.
COMPLETED_SENTINEL = "COMPLETED"
CENSUS_FILENAME = "repair_census.csv"
SUMMARY_FILENAME = "repair_summary.json"

#: What happened to the bytes on disk, as opposed to what the operator concluded.
ACTION_WRITTEN = "written"  # re-serialised: the root was actually moved
ACTION_COPIED = "copied"  # byte-identical passthrough of the source .pkl

CENSUS_COLUMNS = (
    "motion_key",
    "action",
    "success",
    "reason",
    "screened_here",
    "airborne_frac_before",
    "airborne_frac_after",
    "infeasible_frac_before",
    "infeasible_frac_after",
    "offset_max_m",
    "offset_mean_m",
    "seconds",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bank", required=True, type=Path, help="source bank directory of .pkl clips"
    )
    parser.add_argument(
        "--out-bank", required=True, type=Path, help="destination bank directory (real files)"
    )
    parser.add_argument(
        "--screen-dir",
        type=Path,
        default=None,
        help="directory of per-clip screen JSONs; clips already feasible there are copied through",
    )
    parser.add_argument(
        "--out-reports", required=True, type=Path, help="per-clip reports, census CSV, summary"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel worker processes (1 = in-process)"
    )
    parser.add_argument("--limit", type=int, default=0, help="process at most N clips (0 = all)")
    parser.add_argument(
        "--repair-all", action="store_true", help="ignore --screen-dir and screen every clip here"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="redo clips that already have outputs"
    )
    parser.add_argument("--max-offset-m", type=float, default=RepairBudget.max_offset_m)
    parser.add_argument(
        "--max-infeasible-frac-after", type=float, default=RepairBudget.max_infeasible_frac_after
    )
    parser.add_argument("--clearance-m", type=float, default=RepairBudget.clearance_m)
    parser.add_argument("--smooth-s", type=float, default=RepairBudget.smooth_s)
    parser.add_argument("--gap-m", type=float, default=ScreenThresholds.gap_m)
    parser.add_argument("--mu", type=float, default=ScreenThresholds.mu)
    parser.add_argument(
        "--unsupported-force-frac", type=float, default=ScreenThresholds.unsupported_force_frac
    )
    return parser.parse_args(argv)


def read_screen_reports(screen_dir: Path) -> dict[str, dict[str, Any]]:
    """Index the screen's output by motion key.

    Accepts a directory of ``<motion_key>.json`` files (one :class:`ClipScreen` each) and also a
    directory holding aggregate JSONs -- a bare list of clip dicts, or an object with a ``clips``,
    ``screens`` or ``motions`` array.  Being permissive here is cheap and keeps this script from
    breaking on a naming choice made in the screen module.
    """

    reports: dict[str, dict[str, Any]] = {}
    if not screen_dir.is_dir():
        return reports
    for path in sorted(screen_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        records: list[Any] = []
        if isinstance(payload, dict):
            if "motion_key" in payload:
                records = [payload]
            else:
                for field in ("clips", "screens", "motions"):
                    if isinstance(payload.get(field), list):
                        records = list(payload[field])
                        break
        elif isinstance(payload, list):
            records = list(payload)
        for record in records:
            if isinstance(record, dict) and "motion_key" in record and "infeasible_frac" in record:
                reports[str(record["motion_key"])] = record
    return reports


def needs_repair(report: dict[str, Any] | None, max_infeasible_frac_after: float) -> bool:
    """True when the screen says this clip is over the bar, or said nothing usable about it.

    ``infeasible_frac`` is ``null`` in a screen record whose every LP failed -- the screen refuses
    to coerce an unsolved frame to zero.  ``float(None)`` raises, which lands here as "screen it
    again", which is the safe reading: an unscoreable clip is not a clean clip.
    """

    if report is None:
        return True
    try:
        value = float(report["infeasible_frac"])
    except (KeyError, TypeError, ValueError):
        return True
    return not (value <= max_infeasible_frac_after)  # NaN falls through to True


def _copy_through(source: Path, destination: Path) -> None:
    """Byte-identical passthrough, written atomically so a kill cannot leave a truncated clip."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    shutil.copy2(source, staging)
    os.replace(staging, destination)


def process_clip(
    source: Path,
    out_bank: Path,
    out_reports: Path,
    *,
    screen_report: dict[str, Any] | None,
    budget: RepairBudget,
    thresholds: ScreenThresholds,
) -> dict[str, Any]:
    """Repair (or pass through) one clip and return its census row.

    The model is loaded through :func:`load_model`, which caches per process, so a worker pays the
    MJCF compile once and every subsequent clip reuses it.
    """

    started = time.process_time()
    destination = out_bank / source.name
    report_path = out_reports / f"{source.stem}.json"

    if not needs_repair(screen_report, budget.max_infeasible_frac_after):
        _copy_through(source, destination)
        row = {
            "motion_key": source.stem,
            "action": ACTION_COPIED,
            "success": True,
            "reason": "screen_feasible",
            "screened_here": False,
            "airborne_frac_before": (
                float(screen_report.get("airborne_frac", float("nan")))
                if screen_report
                else float("nan")
            ),
            "airborne_frac_after": (
                float(screen_report.get("airborne_frac", float("nan")))
                if screen_report
                else float("nan")
            ),
            "infeasible_frac_before": (
                float(screen_report["infeasible_frac"]) if screen_report else 0.0
            ),
            "infeasible_frac_after": (
                float(screen_report["infeasible_frac"]) if screen_report else 0.0
            ),
            "offset_max_m": 0.0,
            "offset_mean_m": 0.0,
            "seconds": time.process_time() - started,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(row, indent=1, sort_keys=True))
        return row

    model = load_model()
    motion = load_motion(source)
    repaired, result = repair_motion(motion, model=model, budget=budget, thresholds=thresholds)

    moved = result.success and result.offset_max_m > 0.0 and repaired is not motion
    if moved:
        save_motion(repaired, destination)
        action = ACTION_WRITTEN
    else:
        _copy_through(source, destination)
        action = ACTION_COPIED

    row = {
        "motion_key": result.motion_key,
        "action": action,
        "success": bool(result.success),
        "reason": result.reason,
        "screened_here": True,
        **{
            field: value
            for field, value in asdict(result).items()
            if field not in {"motion_key", "success", "reason"}
        },
        "seconds": time.process_time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(row, indent=1, sort_keys=True))
    return row


#: (source, out_bank, out_reports, screen_report, budget_kwargs, threshold_kwargs)
WorkPayload = tuple[str, str, str, "dict[str, Any] | None", dict[str, float], dict[str, float]]


def _worker(payload: WorkPayload) -> dict[str, Any]:
    """Module-level entry point so ``joblib`` can pickle the unit of work."""

    source, out_bank, out_reports, screen_report, budget_kw, threshold_kw = payload
    return process_clip(
        Path(source),
        Path(out_bank),
        Path(out_reports),
        screen_report=screen_report,
        budget=RepairBudget(**budget_kw),
        thresholds=ScreenThresholds(**threshold_kw),
    )


def summarise(
    rows: list[dict[str, Any]], *, budget: RepairBudget, thresholds: ScreenThresholds
) -> dict[str, Any]:
    """Aggregate the census into the numbers a reader of the experiment actually wants."""

    attempted = [row for row in rows if row.get("screened_here")]
    successes = [row for row in attempted if row.get("success")]
    moved = [row for row in rows if row.get("action") == ACTION_WRITTEN]

    def _mean(values: list[float]) -> float:
        finite = [value for value in values if value == value]
        return float(sum(finite) / len(finite)) if finite else 0.0

    by_reason: dict[str, int] = {reason: 0 for reason in REPAIR_REASONS}
    for row in rows:
        reason = str(row.get("reason", ""))
        by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "n_clips": len(rows),
        "n_copied_unscreened": len(rows) - len(attempted),
        "n_attempted": len(attempted),
        "n_success": len(successes),
        "n_bank_files_rewritten": len(moved),
        "success_rate": float(len(successes) / len(attempted)) if attempted else 0.0,
        "by_reason": by_reason,
        "offset_max_m_mean": _mean([float(row["offset_max_m"]) for row in attempted]),
        "offset_max_m_max": max([float(row["offset_max_m"]) for row in attempted], default=0.0),
        "offset_mean_m_mean": _mean([float(row["offset_mean_m"]) for row in attempted]),
        "infeasible_frac_before_mean": _mean(
            [float(row["infeasible_frac_before"]) for row in attempted]
        ),
        "infeasible_frac_after_mean": _mean(
            [float(row["infeasible_frac_after"]) for row in attempted]
        ),
        "airborne_frac_before_mean": _mean(
            [float(row["airborne_frac_before"]) for row in attempted]
        ),
        "airborne_frac_after_mean": _mean([float(row["airborne_frac_after"]) for row in attempted]),
        "seconds_per_clip_mean": _mean([float(row["seconds"]) for row in attempted]),
        "seconds_total": float(sum(float(row["seconds"]) for row in rows)),
        "budget": asdict(budget),
        "thresholds": asdict(thresholds),
    }


def write_census(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the per-clip census CSV, one row per clip in the output bank."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CENSUS_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["motion_key"])):
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budget = RepairBudget(
        max_offset_m=args.max_offset_m,
        max_infeasible_frac_after=args.max_infeasible_frac_after,
        clearance_m=args.clearance_m,
        smooth_s=args.smooth_s,
    )
    thresholds = ScreenThresholds(
        gap_m=args.gap_m, mu=args.mu, unsupported_force_frac=args.unsupported_force_frac
    )

    sources = sorted(Path(args.bank).glob("*.pkl"))
    if not sources:
        raise SystemExit(f"no .pkl clips under {args.bank}")
    if args.limit:
        sources = sources[: args.limit]

    screen_reports = {} if args.repair_all else read_screen_reports(Path(args.screen_dir or ""))
    args.out_bank.mkdir(parents=True, exist_ok=True)
    args.out_reports.mkdir(parents=True, exist_ok=True)
    (args.out_reports / COMPLETED_SENTINEL).unlink(missing_ok=True)

    pending = []
    done_rows: list[dict[str, Any]] = []
    for source in sources:
        report_path = args.out_reports / f"{source.stem}.json"
        if not args.overwrite and (args.out_bank / source.name).is_file() and report_path.is_file():
            try:
                done_rows.append(json.loads(report_path.read_text()))
                continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(source)

    print(
        f"[repair] {len(sources)} clips: {len(done_rows)} already done, {len(pending)} pending "
        f"({sum(1 for s in pending if needs_repair(screen_reports.get(s.stem), budget.max_infeasible_frac_after))}"
        f" to screen here)",
        flush=True,
    )

    budget_kw = asdict(budget)
    threshold_kw = asdict(thresholds)
    started = time.time()
    rows: list[dict[str, Any]] = list(done_rows)
    if args.workers > 1 and pending:
        from joblib import Parallel, delayed

        payloads = [
            (
                str(source),
                str(args.out_bank),
                str(args.out_reports),
                screen_reports.get(source.stem),
                budget_kw,
                threshold_kw,
            )
            for source in pending
        ]
        pool = Parallel(n_jobs=args.workers, backend="loky", verbose=5)
        rows.extend(pool(delayed(_worker)(payload) for payload in payloads))
    else:
        for index, source in enumerate(pending, start=1):
            rows.append(
                process_clip(
                    source,
                    args.out_bank,
                    args.out_reports,
                    screen_report=screen_reports.get(source.stem),
                    budget=budget,
                    thresholds=thresholds,
                )
            )
            if index % 25 == 0 or index == len(pending):
                print(f"[repair] {index}/{len(pending)} ({time.time() - started:.0f}s)", flush=True)

    census_path = args.out_reports / CENSUS_FILENAME
    write_census(rows, census_path)
    summary = summarise(rows, budget=budget, thresholds=thresholds)
    summary.update(
        {
            "bank": str(Path(args.bank).resolve()),
            "out_bank": str(Path(args.out_bank).resolve()),
            "screen_dir": str(Path(args.screen_dir).resolve()) if args.screen_dir else None,
            "wall_seconds": time.time() - started,
            "workers": int(args.workers),
        }
    )
    (args.out_reports / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=1, sort_keys=True))

    written = {path.name for path in args.out_bank.glob("*.pkl")}
    expected = {source.name for source in sources}
    complete = written >= expected
    if complete:
        (args.out_reports / COMPLETED_SENTINEL).write_text(
            json.dumps(
                {
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_clips": len(expected),
                    "census": str(census_path.resolve()),
                    "summary": str((args.out_reports / SUMMARY_FILENAME).resolve()),
                },
                indent=1,
            )
        )
    else:
        print(
            f"[repair] INCOMPLETE: {len(expected - written)} clips missing from {args.out_bank}",
            flush=True,
        )

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "thresholds"}, indent=1, sort_keys=True
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
