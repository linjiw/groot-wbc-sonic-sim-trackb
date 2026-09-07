#!/usr/bin/env python3
"""Resume the registered ICRA pipeline under existing serial GPU and budget gates."""

import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "research-data/groot-wbc"
STUDY = DATA / "m2s-icra-v1"
LEARNING = DATA / "m2s-icra-learning-v1"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(stage, **extra):
    path = STUDY / "pipeline_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"stage": stage, "updated_at": now(), **extra}, indent=2) + "\n"
    )
    temporary.replace(path)


def verify_refs():
    for folder in (STUDY, LEARNING):
        reg = json.loads((folder / "registration.json").read_text())
        for ref in reg["references"]:
            actual = "sha256:" + hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()
            if actual != ref["sha256"]:
                raise RuntimeError("Frozen dependency changed: " + ref["path"])


def pending(index):
    master = json.loads(index.read_text())
    for block in master["batches"]:
        folder = Path(block["directory"])
        admission = folder / "admission.json"
        if admission.exists():
            if not json.loads(admission.read_text())["admitted"]:
                raise RuntimeError("Measurement audit rejected " + str(folder))
            continue
        record = folder / "run_record.json"
        if record.exists():
            r = json.loads(record.read_text())
            if any(c["status"] not in ("completed", "not_started") for c in r["cells"].values()):
                raise RuntimeError("Unresolved running/failed cell in " + str(folder))
        return folder
    return None


def call(script, command, out):
    log = STUDY / "pipeline.log"
    with log.open("a") as stream:
        stream.write(f"\n{now()} {script} {command}\n")
        stream.flush()
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/research" / script), command, "--out", str(out)],
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wall-hours", type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.max_wall_hours <= 24:
        raise ValueError("bounded supervisor duration required")
    with (STUDY / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_refs()
        receipt = STUDY / f"pipeline_launch_{time.time_ns()}.json"
        receipt.write_text(
            json.dumps(
                {
                    "started_at": now(),
                    "max_wall_hours": args.max_wall_hours,
                    "script": str(Path(__file__)),
                    "sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "gpu_floor_mib": 7500,
                    "automatic_retries": "only never-started cells after resource yield",
                    "budgets": "unchanged rolling 8 contended GPU h/day and 24/week",
                    "stops": "any changed reference, failed cell, audit rejection, unexpected process failure",
                },
                indent=2,
            )
            + "\n"
        )
        started = time.monotonic()
        while time.monotonic() - started < args.max_wall_hours * 3600:
            verify_refs()
            label_pending = pending(STUDY / "prepared.json")
            if label_pending:
                stage, script, folder, index = (
                    "labels",
                    "motion2scene_icra_study.py",
                    STUDY,
                    STUDY / "prepared.json",
                )
            elif not (LEARNING / "fit.json").exists():
                status("fitting")
                if call("motion2scene_icra_learning.py", "fit", LEARNING):
                    raise RuntimeError("Fitting failed; no automatic restart")
                continue
            elif not (LEARNING / "evaluation_master.json").exists():
                status("preparing_evaluation")
                if call("motion2scene_icra_learning.py", "prepare", LEARNING):
                    raise RuntimeError("Evaluation preparation failed; no automatic restart")
                continue
            elif pending(LEARNING / "evaluation_master.json"):
                stage, script, folder, index = (
                    "evaluation",
                    "motion2scene_icra_learning.py",
                    LEARNING,
                    LEARNING / "evaluation_master.json",
                )
            else:
                if not (LEARNING / "comparison_540.json").exists():
                    if call("motion2scene_icra_learning.py", "summarize", LEARNING):
                        raise RuntimeError("Final analysis failed")
                status("complete", labels_assigned=82, evaluation_assigned=540)
                return
            free = int(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True,
                )
                .strip()
                .splitlines()[0]
            )
            if free < 7500:
                status(
                    "waiting_gpu",
                    next_stage=stage,
                    free_gpu_mib=free,
                    required_gpu_mib=7500,
                    pending_batch=str(pending(index)),
                )
                time.sleep(30)
                continue
            status("running_" + stage, free_gpu_mib=free)
            code = call(script, "run", folder)
            blocked = pending(index)
            if code:
                record = blocked / "run_record.json" if blocked else None
                if (
                    record is None
                    or not record.exists()
                    or json.loads(record.read_text())["status"] != "yielded_gpu_contention"
                ):
                    raise RuntimeError(f"Unexpected {script} exit {code}; no retry")
            if blocked:
                status("resource_yield", next_stage=stage, pending_batch=str(blocked))
                time.sleep(30)
        status("supervisor_window_ended", complete=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("stopped_for_review", error=str(error))
        raise
