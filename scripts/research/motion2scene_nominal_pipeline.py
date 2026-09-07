#!/usr/bin/env python3
"""Run the nominal-contract labels after M2S-ICRA-v1 finishes, under the same gates.

The registered protocol requires this study never to compete with M2S-ICRA-v1 for the
card, so this supervisor refuses to start until the parent pipeline reports `complete`.
It changes no threshold: the 7500 MiB free-memory floor, the 375 s cell timeout, the
rolling budget review and the measurement audits are the ones the label runner already
applies. It only decides when to call the runner.

Any failed cell, changed reference hash or rejected audit stops it without a retry.
"""

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
PARENT = DATA / "m2s-icra-v1"
STUDY = DATA / "m2s-icra-nominal-v1"
LEARNING = DATA / "m2s-icra-nominal-learning-v1"
FLOOR_MIB = 7500


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
    reg = json.loads((STUDY / "registration.json").read_text())
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
                raise RuntimeError("Unresolved running or failed cell in " + str(folder))
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


def parent_complete():
    path = PARENT / "pipeline_status.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("stage") == "complete"


def free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return int(out.strip().splitlines()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wall-hours", type=float, default=12)
    args = parser.parse_args()
    if not 0 < args.max_wall_hours <= 24:
        raise ValueError("bounded supervisor duration required")
    with (STUDY / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_refs()
        (STUDY / f"pipeline_launch_{time.time_ns()}.json").write_text(
            json.dumps(
                {
                    "started_at": now(),
                    "max_wall_hours": args.max_wall_hours,
                    "script": str(Path(__file__)),
                    "sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "gpu_floor_mib": FLOOR_MIB,
                    "waits_for": "M2S-ICRA-v1 pipeline stage == complete",
                    "stages": "labels -> register -> fit -> prepare -> evaluation -> summarize",
                    "budgets": "unchanged rolling 8 contended GPU h/day and 24/week",
                    "stops": "changed reference, failed cell, audit rejection, runner failure",
                },
                indent=2,
            )
            + "\n"
        )
        started = time.monotonic()
        while time.monotonic() - started < args.max_wall_hours * 3600:
            verify_refs()
            if not parent_complete():
                status("waiting_for_parent", parent=str(PARENT))
                time.sleep(60)
                continue
            label_pending = pending(STUDY / "prepared.json")
            if label_pending:
                stage, script, folder, index = (
                    "labels",
                    "motion2scene_icra_nominal.py",
                    STUDY,
                    STUDY / "prepared.json",
                )
            elif not (LEARNING / "registration.json").exists():
                status("registering_learning")
                if call("motion2scene_icra_nominal_learning.py", "register", LEARNING):
                    raise RuntimeError("Learning registration failed; no automatic restart")
                continue
            elif not (LEARNING / "fit.json").exists():
                status("fitting")
                if call("motion2scene_icra_nominal_learning.py", "fit", LEARNING):
                    raise RuntimeError("Fitting failed; no automatic restart")
                continue
            elif not (LEARNING / "evaluation_master.json").exists():
                status("preparing_evaluation")
                if call("motion2scene_icra_nominal_learning.py", "prepare", LEARNING):
                    raise RuntimeError("Evaluation preparation failed; no automatic restart")
                continue
            elif pending(LEARNING / "evaluation_master.json"):
                stage, script, folder, index = (
                    "evaluation",
                    "motion2scene_icra_nominal_learning.py",
                    LEARNING,
                    LEARNING / "evaluation_master.json",
                )
            else:
                if not list(LEARNING.glob("comparison_*.json")):
                    status("summarizing")
                    if call("motion2scene_icra_nominal_learning.py", "summarize", LEARNING):
                        raise RuntimeError("Final analysis failed")
                status("complete", labels_assigned=72, evaluation_assigned=144)
                return
            free = free_mib()
            if free < FLOOR_MIB:
                status(
                    "waiting_gpu",
                    next_stage=stage,
                    free_gpu_mib=free,
                    required_gpu_mib=FLOOR_MIB,
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
                time.sleep(30)
        status("stopped_wall_clock")


if __name__ == "__main__":
    main()
