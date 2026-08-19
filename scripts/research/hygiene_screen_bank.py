#!/usr/bin/env python3
"""Run the dynamic-feasibility screen over a bank of SONIC motion clips (CPU only).

WHY a script and not a notebook: screening 4950 clips is a multi-hour CPU job
that will be interrupted.  Every clip therefore gets its own JSON, the run is
resumable by construction (an existing JSON is proof that clip is done), and a
``COMPLETED.json`` sentinel records wall-clock and exit code so a downstream
aggregator can tell "finished clean" from "still running" from "died".

Clips are resampled to SONIC's training rate (``--target-fps``, default 50)
before screening, because that is the timeline the policy actually tracks.  Pass
``--no-resample`` to screen the raw 30 Hz bank instead.

Examples:
    # 40-clip pilot
    python scripts/research/hygiene_screen_bank.py \
        --bank /data/.../robot_filtered --out out/hygiene/screen_pilot \
        --limit 40 --workers 8

    # full bank, resuming an interrupted run
    python scripts/research/hygiene_screen_bank.py \
        --bank /data/.../robot_filtered --out out/hygiene/screen_full --workers 16
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields as dataclass_fields
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.hygiene.motion_io import load_motion, resample_to  # noqa: E402
from gear_sonic.research.hygiene.screen import (  # noqa: E402
    DEFAULT_G1_MJCF,
    SCREEN_SCHEMA_VERSION,
    ClipScreen,
    ScreenThresholds,
    load_model,
    model_sha256,
    screen_motion,
)

_WORKER: dict[str, Any] = {}


def _csv_columns() -> list[str]:
    """Scalar ClipScreen fields, in declaration order, plus flattened thresholds."""
    scalar = [f.name for f in dataclass_fields(ClipScreen) if f.name != "thresholds"]
    return scalar + ["thr_gap_m", "thr_mu", "thr_unsupported_force_frac"]


def _init_worker(mjcf: str, thresholds: dict[str, float], target_fps: int | None) -> None:
    """Compile the robot once per worker process; MjModel compilation is not free.

    Thread caps are set here, before anything imports torch, because the resample
    step is BLAS/OpenMP-parallel by default: on a 20-core box one worker will
    happily burn 10 cores on a 90k-element interpolation and then fight the other
    workers for them.  Screening is embarrassingly parallel *across* clips, so one
    thread per worker and ``--workers`` as the only knob is both faster and makes
    the per-clip CPU numbers mean something.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    _WORKER["model"] = load_model(mjcf)
    _WORKER["thresholds"] = ScreenThresholds(**thresholds)
    _WORKER["target_fps"] = target_fps


def _screen_one(path_str: str) -> tuple[str, dict[str, Any] | None, str | None]:
    """Screen one clip.  Returns ``(motion_key, screen_dict, error_text)``."""
    path = Path(path_str)
    key = path.stem
    try:
        motion = load_motion(path)
        key = motion.key
        target_fps = _WORKER["target_fps"]
        if target_fps is not None:
            motion = resample_to(motion, target_fps)
        result = screen_motion(motion, _WORKER["model"], _WORKER["thresholds"])
        payload = result.to_dict()
        payload["source_path"] = str(path)
        return key, payload, None
    except Exception:  # noqa: BLE001 - one bad clip must not kill a multi-hour sweep
        return key, None, traceback.format_exc()


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.motions:
        paths = [Path(p) for p in args.motions]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"missing motion files: {missing[:5]}")
    else:
        bank = Path(args.bank)
        if not bank.is_dir():
            raise SystemExit(f"--bank is not a directory: {bank}")
        paths = sorted(bank.glob(args.pattern))
        if not paths:
            raise SystemExit(f"no files matching {args.pattern!r} under {bank}")
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def _write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write so an interrupted run never leaves a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _write_csv(out_dir: Path, clip_dir: Path) -> int:
    """Rebuild the bank-level CSV from every per-clip JSON currently on disk."""
    columns = _csv_columns()
    rows: list[dict[str, Any]] = []
    for json_path in sorted(clip_dir.glob("*.json")):
        with json_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        thresholds = record.get("thresholds", {})
        row = {name: record.get(name) for name in columns if not name.startswith("thr_")}
        row["thr_gap_m"] = thresholds.get("gap_m")
        row["thr_mu"] = thresholds.get("mu")
        row["thr_unsupported_force_frac"] = thresholds.get("unsupported_force_frac")
        rows.append(row)
    csv_path = out_dir / "hygiene_screen.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", help="directory of motion-library .pkl clips")
    source.add_argument("--motions", nargs="+", help="explicit list of clip .pkl paths")
    parser.add_argument(
        "--pattern", default="*.pkl", help="glob used with --bank (default: %(default)s)"
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--workers", type=int, default=1, help="worker processes (default: %(default)s)"
    )
    parser.add_argument("--limit", type=int, default=None, help="screen at most N clips")
    parser.add_argument(
        "--force", action="store_true", help="re-screen clips that already have a JSON"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=50,
        help="resample to this rate before screening; SONIC trains at 50 (default: %(default)s)",
    )
    parser.add_argument(
        "--no-resample", action="store_true", help="screen the clip at its stored fps instead"
    )
    parser.add_argument("--mjcf", default=None, help="override the G1 MJCF path")
    parser.add_argument("--gap-m", type=float, default=ScreenThresholds.gap_m)
    parser.add_argument("--mu", type=float, default=ScreenThresholds.mu)
    parser.add_argument(
        "--unsupported-force-frac", type=float, default=ScreenThresholds.unsupported_force_frac
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    out_dir = Path(args.out)
    clip_dir = out_dir / "clips"
    error_dir = out_dir / "errors"
    clip_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    thresholds = ScreenThresholds(
        gap_m=args.gap_m, mu=args.mu, unsupported_force_frac=args.unsupported_force_frac
    )
    mjcf = args.mjcf if args.mjcf is not None else str(DEFAULT_G1_MJCF)
    target_fps = None if args.no_resample else int(args.target_fps)

    paths = _resolve_inputs(args)
    pending = [p for p in paths if args.force or not (clip_dir / f"{p.stem}.json").exists()]
    skipped = len(paths) - len(pending)
    print(
        f"[hygiene-screen] {len(paths)} clips, {skipped} already done, {len(pending)} to screen",
        flush=True,
    )

    completed = 0
    failed: list[str] = []
    exit_code = 0
    try:
        if pending:
            init_args = (mjcf, thresholds.to_dict(), target_fps)
            if args.workers > 1:
                context = mp.get_context("spawn")
                with context.Pool(
                    args.workers, initializer=_init_worker, initargs=init_args
                ) as pool:
                    stream = pool.imap_unordered(
                        _screen_one, [str(p) for p in pending], chunksize=1
                    )
                    for index, (key, payload, error) in enumerate(stream, start=1):
                        _record(clip_dir, error_dir, key, payload, error, failed)
                        completed += payload is not None
                        _progress(index, len(pending), started)
            else:
                _init_worker(*init_args)
                for index, path in enumerate(pending, start=1):
                    key, payload, error = _screen_one(str(path))
                    _record(clip_dir, error_dir, key, payload, error, failed)
                    completed += payload is not None
                    _progress(index, len(pending), started)
    except KeyboardInterrupt:
        exit_code = 130
        print("[hygiene-screen] interrupted", flush=True)
    except Exception:
        exit_code = 1
        traceback.print_exc()

    rows = _write_csv(out_dir, clip_dir)
    elapsed = time.time() - started
    sentinel = {
        "kind": "hygiene_screen_bank",
        "schema_version": SCREEN_SCHEMA_VERSION,
        "exit_code": exit_code,
        "wall_clock_s": round(elapsed, 3),
        "started_unix": round(started, 3),
        "finished_unix": round(time.time(), 3),
        "requested_clips": len(paths),
        "already_done": skipped,
        "screened_this_run": completed,
        "failed_this_run": len(failed),
        "failed_keys": sorted(failed)[:50],
        "csv_rows": rows,
        "csv_path": str(out_dir / "hygiene_screen.csv"),
        "workers": int(args.workers),
        "target_fps": target_fps,
        "thresholds": thresholds.to_dict(),
        "mjcf_path": mjcf,
        "robot_sha256": model_sha256(load_model(mjcf)),
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    _write_json(out_dir / "COMPLETED.json", sentinel)
    print(
        f"[hygiene-screen] done: {completed} screened, {len(failed)} failed, "
        f"{rows} rows, {elapsed:.1f}s wall, exit {exit_code}",
        flush=True,
    )
    return exit_code


def _record(
    clip_dir: Path,
    error_dir: Path,
    key: str,
    payload: dict[str, Any] | None,
    error: str | None,
    failed: list[str],
) -> None:
    if payload is not None:
        _write_json(clip_dir / f"{key}.json", payload)
        (error_dir / f"{key}.json").unlink(missing_ok=True)
        return
    failed.append(key)
    _write_json(error_dir / f"{key}.json", {"motion_key": key, "traceback": error})


def _progress(index: int, total: int, started: float) -> None:
    if index == total or index % 25 == 0:
        elapsed = time.time() - started
        rate = index / elapsed if elapsed > 0 else 0.0
        print(
            f"[hygiene-screen] {index}/{total}  {elapsed:.1f}s  {rate:.2f} clip/s",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
