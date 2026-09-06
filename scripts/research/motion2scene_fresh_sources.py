#!/usr/bin/env python3
"""Registered CPU acquisition with a complete, non-replacing source funnel."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from generate_kimodo_motions import read_prompts, slugify
from motion2scene_beam_teacher import capsules
from motion2scene_carrier_learning import RECORDS, gate
from motion2scene_inverse_learning import inputs
from motion2scene_station_study import load as load_station
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import make_clouds
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import shared_origin_pair
from gear_sonic.dataset_generation.local_adaptation import local_crouch
from gear_sonic.dataset_generation.reference_payload import payload_from_reference

DATA = ROOT.parent / "research-data/groot-wbc"
SIBLING = ROOT.parent / "motion2scene"
KIMODO = ROOT.parent / "kimodo"
SOURCE_IDS = list(range(43001, 43009))
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots"
    / "3020ad8c419c244e0429d360163730c63c4ed011"
)
sys.path.insert(0, str(SIBLING / "src"))
from motion2scene.motion.route_semantics import classify_route  # noqa: E402


def large_artifact(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.absolute()), "sha256": "sha256:" + digest}


def seed_mentions(value, seeds):
    """Conservative integer scan, plus generation-report implied ranges."""
    hits = set()
    if isinstance(value, dict):
        base = value.get("seed_base")
        count = value.get("seeds_per_prompt")
        prompts = value.get("prompts", 1)
        if type(base) is int and type(count) is int and type(prompts) is int:
            for p in range(prompts):
                start = base + (
                    1000 * p if value.get("seed_design") != "shared_across_prompts" else 0
                )
                hits.update(s for s in seeds if start <= s < start + count)
        for v in value.values():
            hits.update(seed_mentions(v, seeds))
    elif isinstance(value, list):
        for v in value:
            hits.update(seed_mentions(v, seeds))
    elif type(value) is int and value in seeds:
        hits.add(value)
    return hits


def inventory():
    roots = [ROOT, SIBLING, DATA]
    rows, collisions, unparsed, excluded = [], [], [], []
    for root in roots:
        for directory, dirs, files in os.walk(root):
            for d in list(dirs):
                if d in {".git", "__pycache__", "node_modules"} or d.startswith(".venv"):
                    excluded.append(str(Path(directory) / d))
                    dirs.remove(d)
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                path = Path(directory) / name
                ref = large_artifact(path)
                rows.append(ref)
                try:
                    value = json.loads(path.read_text())
                except (ValueError, UnicodeError) as exc:
                    unparsed.append({**ref, "error": str(exc)})
                    continue
                found = seed_mentions(value, SOURCE_IDS)
                if found:
                    collisions.append({**ref, "seeds": sorted(found)})
    return {
        "roots": [str(p) for p in roots],
        "files": rows,
        "collisions": collisions,
        "unparsed": unparsed,
        "excluded_dependency_directories": sorted(excluded),
        "scope": "local JSON metadata only; no claim about unrecorded or remote generation",
    }


def register(args):
    _, parent, _, _ = load_station(DATA / "m2s-station-search-v1/registration.json")
    inv = inventory()
    if inv["collisions"] or inv["unparsed"]:
        raise ValueError(
            f"freshness inventory incomplete or colliding: {inv['collisions']}, {inv['unparsed']}"
        )
    prompt = SIBLING / "configs/prompts/kimodo_route_retention_neutral_v1.txt"
    cache = DATA / "cg-wbc-v2-shared-seed-confirmatory/taxonomy/prompt_cache.npz"
    if read_prompts(prompt) != ["A person walks at a steady pace in a straight line"]:
        raise ValueError("unexpected acquisition prompt")
    previous = json.loads((DATA / "m2s-carrier-learning-v1/registration.json").read_text())
    paths = [
        Path(__file__),
        ROOT / "scripts/research/motion2scene_fresh_audit.py",
        ROOT / "scripts/research/generate_kimodo_motions.py",
        ROOT / "gear_sonic/dataset_generation/kimodo_prompt_cache.py",
        SIBLING / "src/motion2scene/motion/route_semantics.py",
    ]
    paths += sorted((KIMODO / "kimodo").rglob("*.py"))
    paths += sorted(p for p in SNAPSHOT.rglob("*") if p.is_file())
    # Pin all referenced skeleton assets, not just the top-level XML.
    paths += sorted(
        p
        for p in (KIMODO / "kimodo/assets/skeletons/g1skel34").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    refs = [large_artifact(p) for p in dict.fromkeys(paths)]
    runtime = subprocess.check_output(
        [
            str(ROOT / ".venv_kimodo/bin/python"),
            "-c",
            "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,"
            "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))",
        ],
        text=True,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "inventory.json", inv)
    write_new(
        args.out / "registration.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": artifact(ROOT / "docs/motion2scene/FRESH_SOURCE_V1.md"),
            "station_parent": artifact(DATA / "m2s-station-search-v1/registration.json"),
            "inventory": artifact(args.out / "inventory.json"),
            "implementations": refs,
            "prompt": artifact(prompt),
            "cache": artifact(cache),
            "mjcf": previous["mjcf"],
            "snapshot": str(SNAPSHOT),
            "source_ids": SOURCE_IDS,
            "event_stations": [0.35, 0.65],
            "seeds": parent["seeds"],
            "cells": parent["cells"],
            "arms": [
                "raw",
                "rank",
                "gradient",
                "probe_refine",
                "multistart",
                "pattern",
                "uniform_pattern",
            ],
            "main_draw_seed_offset": 80000,
            "uniform_seed_offset": 90000,
            "max_generation_seconds": 1800,
            "max_build_seconds": 600,
            "max_search_seconds": 1800,
            "max_audit_seconds": 1200,
            "generation_runtime": json.loads(runtime),
            "generation_python": large_artifact(ROOT / ".venv_kimodo/bin/python"),
            "generation_device": "cpu",
            "generation_threads": 4,
            "gpu_preflight": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.free,memory.used,utilization.gpu",
                    "--format=csv",
                ],
                text=True,
            ),
            "gpu_seconds": 0,
            "training_eligible": False,
            "execution_eligible": False,
            "source_role": "fresh_audit_permanently_excluded_from_training_and_selection",
            "gate_failure_policy": "stop_entire_audit_no_replacement",
        },
    )


def registration(path):
    reg = json.loads(path.read_text())
    for key in [
        "protocol",
        "inventory",
        "station_parent",
        "prompt",
        "cache",
        "mjcf",
        "generation_python",
    ]:
        checked(Path(reg[key]["path"]), reg[key]["sha256"])
    for ref in reg["implementations"]:
        if large_artifact(Path(ref["path"])) != ref:
            raise ValueError(f"implementation changed: {ref['path']}")
    _, parent, _, _ = load_station(Path(reg["station_parent"]["path"]))
    if reg["source_ids"] != SOURCE_IDS or reg["cells"] != parent["cells"]:
        raise ValueError("source/checkpoint registration changed")
    return reg, parent


def validate_generation(folder, reg):
    prompt = read_prompts(Path(reg["prompt"]["path"]))[0]
    expected = {f"000_{slugify(prompt)}_s{i}" for i in range(8)}
    if {p.stem for p in folder.glob("*.csv")} != expected:
        raise ValueError("incomplete or unexpected generated CSV set")
    if {p.stem for p in folder.glob("*.json")} != expected | {"generation_report"}:
        raise ValueError("incomplete or unexpected sidecar set")
    report = json.loads((folder / "generation_report.json").read_text())
    if report["generated"] != 8 or report["failed"] != 0:
        raise ValueError("generation failure retained; audit stopped")
    rows = []
    for i, seed in enumerate(reg["source_ids"]):
        stem = f"000_{slugify(prompt)}_s{i}"
        side = folder / f"{stem}.json"
        value = json.loads(side.read_text())
        checks = {
            "seed": seed,
            "generation_seed": seed,
            "prompt": prompt,
            "num_frames": 120,
            "fps": 30,
            "denoising_steps": 100,
            "model": "kimodo-g1-rp",
            "seed_design": "independent_per_prompt",
            "prompt_design_version": "motion2scene_fresh_source_v1",
            "csv": f"{stem}.csv",
            "qpos_convention": "mujoco_z_up_x_forward_wxyz_root",
        }
        if any(value[k] != v for k, v in checks.items()):
            raise ValueError(f"generation metadata mismatch: {stem}")
        csv = folder / value["csv"]
        qpos = np.loadtxt(csv, delimiter=",")
        if qpos.shape != (120, 36) or not np.isfinite(qpos).all():
            raise ValueError("invalid generated qpos")
        rows.append({"generation_seed": seed, "csv": artifact(csv), "sidecar": artifact(side)})
    return rows


def generate(args):
    reg, _ = registration(args.registration)
    root = args.registration.parent
    folder = root / "generated"
    folder.mkdir(exist_ok=False)
    cache = root / "model-cache"
    cache.mkdir(exist_ok=False)
    (cache / "Kimodo-G1-RP-v1").symlink_to(reg["snapshot"], target_is_directory=True)
    env = {
        "CHECKPOINT_DIR": str(cache),
        "HF_HUB_OFFLINE": "1",
        "LOCAL_CACHE": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
    }
    argv = [
        str(ROOT / ".venv_kimodo/bin/python"),
        str(ROOT / "scripts/research/generate_kimodo_motions.py"),
        "--prompts",
        reg["prompt"]["path"],
        "--cache",
        reg["cache"]["path"],
        "--out",
        str(folder),
        "--model",
        "kimodo-g1-rp",
        "--duration",
        "4.0",
        "--seeds",
        "8",
        "--seed-base",
        "43001",
        "--seed-design",
        "independent_per_prompt",
        "--prompt-design-version",
        "motion2scene_fresh_source_v1",
        "--steps",
        "100",
        "--device",
        "cpu",
    ]
    write_new(
        root / "generation_launch.json",
        {
            "registration": artifact(args.registration),
            "argv": argv,
            "environment_overrides": env,
            "timeout_seconds": reg["max_generation_seconds"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    start = time.monotonic()
    failure, code, rows = None, None, []
    try:
        with (root / "generation.log").open("x") as handle:
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env={**os.environ, **env},
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=reg["max_generation_seconds"],
                check=False,
            )
        code = result.returncode
        if code:
            raise RuntimeError(f"generator exit status {code}")
        rows = validate_generation(folder, reg)
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "category": "generation_or_integrity_failure",
        }
    write_new(
        root / "generation_result.json",
        {
            "launch": artifact(root / "generation_launch.json"),
            "log": artifact(root / "generation.log"),
            "elapsed_seconds": time.monotonic() - start,
            "returncode": code,
            "failure": failure,
            "rows": rows,
            "complete": failure is None,
            "retained_outputs": [artifact(p) for p in sorted(folder.iterdir()) if p.is_file()],
            "candidate_source_ids": reg["source_ids"],
            "gpu_seconds": 0,
        },
    )
    if failure:
        raise RuntimeError(str(failure))


def build(args):
    reg, _ = registration(args.registration)
    root = args.registration.parent
    generated = json.loads((root / "generation_result.json").read_text())
    if not generated["complete"]:
        raise ValueError("incomplete generation: no subset construction")
    carriers = validate_generation(root / "generated", reg)
    if carriers != generated["rows"]:
        raise ValueError("generated artifacts changed")
    out = root / "references"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    rows, sources, values = [], [], {}
    for carrier in carriers:
        seed = carrier["generation_seed"]
        neutral = np.loadtxt(carrier["csv"]["path"], delimiter=",")
        ng = gate(neutral, f"{seed}_neutral", reg["mjcf"]["path"])
        route = classify_route(neutral[:, :2], expected_route="straight", fps=30).to_dict()
        sources.append({**carrier, "neutral_gate": ng, "route": route})
        for j, station in enumerate(reg["event_stations"]):
            row = {
                "case_id": f"{seed}_event{j}",
                "carrier_seed": seed,
                "event_station": station,
                "split": "fresh_audit",
                "parent": carrier["csv"],
                "neutral_gate": ng,
                "neutral_route": route,
                "q4_admitted": False,
                "execution_qualified": False,
            }
            try:
                if time.monotonic() - start > reg["max_build_seconds"]:
                    raise TimeoutError("construction cap")
                target, report = local_crouch(
                    neutral,
                    station,
                    target_drop_m=0.055,
                    window=0.18,
                    mjcf_path=reg["mjcf"]["path"],
                )
                ref = out / f"{row['case_id']}.csv"
                with ref.open("x") as handle:
                    np.savetxt(handle, target, delimiter=",", fmt="%.10f")
                target = np.loadtxt(ref, delimiter=",")
                row.update(target=artifact(ref), operator=asdict(report))
                endpoint_ok = bool(
                    np.allclose(target[[0, -1]], neutral[[0, -1]], atol=1e-8, rtol=0)
                )
                first, second = shared_origin_pair(neutral, target)
                tg = gate(target, row["case_id"], reg["mjcf"]["path"])
                tr = classify_route(target[:, :2], expected_route="straight", fps=30).to_dict()
                states = [
                    capsules(payload_from_reference(q, fps=30, mjcf_path=reg["mjcf"]["path"]))
                    for q in [first, second]
                ]
                values[f"{row['case_id']}_route"] = first[:, :2]
                for k, state in enumerate(states):
                    for key in ["starts", "ends", "radii"]:
                        values[f"{row['case_id']}_{k}_{key}"] = state[key]
                row.update(
                    target_gate=tg,
                    target_route=tr,
                    endpoints_preserved=endpoint_ok,
                    owners=list(states[0]["owners"]),
                )
                row["qualified"] = (
                    endpoint_ok
                    and all(g[k] for g in [ng, tg] for k in ["q0_pass", "q1_pass"])
                    and route["validity_class"] == tr["validity_class"] == "valid_straight"
                )
            except Exception as exc:
                row.update(
                    qualified=False, failure={"type": type(exc).__name__, "message": str(exc)}
                )
            rows.append(row)
            print(json.dumps({"case": row["case_id"], "qualified": row["qualified"]}), flush=True)
    with (root / "bank.npz").open("xb") as handle:
        np.savez_compressed(handle, **values)
    write_new(
        root / "carrier_registry.json",
        {
            "rows": rows,
            "sources": sources,
            "construction_seconds": time.monotonic() - start,
            "source_candidates": 8,
            "target_candidates": 16,
            "qualified_targets": sum(r["qualified"] for r in rows),
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    passed = len(rows) == 16 and all(r["qualified"] for r in rows)
    write_new(
        root / "manifest.json",
        {
            "registration": artifact(args.registration),
            "generation": artifact(root / "generation_result.json"),
            "bank": artifact(root / "bank.npz"),
            "registry": artifact(root / "carrier_registry.json"),
            "reference_gates_pass": passed,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    if not passed:
        raise ValueError("reference gate failure: entire inference audit stopped, no replacements")


def load_cases(path):
    reg, parent = registration(path)
    root = path.parent
    manifest = json.loads((root / "manifest.json").read_text())
    for ref in manifest.values():
        if isinstance(ref, dict) and "sha256" in ref:
            checked(Path(ref["path"]), ref["sha256"])
    if not manifest["reference_gates_pass"]:
        raise ValueError("source bank failed qualification")
    registry = json.loads((root / "carrier_registry.json").read_text())
    expected = {f"{s}_event{j}" for s in reg["source_ids"] for j in range(2)}
    if len(registry["rows"]) != 16 or {r["case_id"] for r in registry["rows"]} != expected:
        raise ValueError("incomplete or duplicated source cases")
    cases = {}
    with np.load(root / "bank.npz", allow_pickle=False) as bank:
        for row in registry["rows"]:
            for key in ["parent", "target"]:
                checked(Path(row[key]["path"]), row[key]["sha256"])
            if not row["qualified"] or row["split"] != "fresh_audit":
                raise ValueError("source qualification or role changed")
            name = row["case_id"]
            states = {
                record["name"]: {
                    **{k: bank[f"{name}_{i}_{k}"] for k in ["starts", "ends", "radii"]},
                    "label": record["label"],
                    "owners": row["owners"],
                }
                for i, record in enumerate(RECORDS)
            }
            route = bank[f"{name}_route"]
            feature, _, rt, progress, yaw, mask, costs = inputs(states, route, RECORDS)
            cases[name] = {
                "metadata": row,
                "states": states,
                "route": route,
                "feature": feature,
                "clouds": make_clouds(states),
                "rt": rt,
                "progress": progress,
                "yaw": yaw,
                "mask": mask,
                "costs": costs,
            }
    return reg, parent, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "generate", "build"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if args.registration:
            failure = args.registration.parent / f"{args.command}_failure.json"
            if not failure.exists():
                write_new(failure, {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
