#!/usr/bin/env python3
"""Reserve test candidates against named source-acquisition provenance records."""

import hashlib
import json
import os
from pathlib import Path
import time

from motion2scene_fresh_sources import DATA, SIBLING
from motion2scene_reserve_transfer_sources_v4 import mentions
from motion2scene_timing_diagnostic import ROOT, artifact, write_new
import numpy as np

IDS = list(range(9490001, 9490009))
NAMES = {"generation_report.json", "carrier_registry.json", "candidates.json"}


def canonical_motion_hash(qpos):
    q = np.asarray(qpos, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("finite multi-frame G1 qpos required")
    q[:, :2] -= q[0, :2].copy()
    norms = np.linalg.norm(q[:, 3:7], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("invalid root quaternion")
    q[:, 3:7] /= norms[:, None]
    # The first nonzero quaternion coordinate fixes antipodal sign even at w=0.
    for row in q:
        quat = row[3:7]
        pivot = next(v for v in quat if abs(v) > 1e-12)
        if pivot < 0:
            row[3:7] *= -1
    q = np.rint(q * 1e6).astype("<i8")
    return "sha256:" + hashlib.sha256(str(q.shape).encode() + q.tobytes()).hexdigest()


def csv_paths(value):
    result = set()
    if isinstance(value, dict):
        for item in value.values():
            result.update(csv_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.update(csv_paths(item))
    elif isinstance(value, str) and value.endswith(".csv") and Path(value).is_absolute():
        result.add(Path(value))
    return result


def main():
    start = time.monotonic()
    paths = set(DATA.glob("m2s-final-transfer*/reservation.json"))
    for root in (ROOT, SIBLING, DATA):
        for directory, dirs, files in os.walk(root):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "__pycache__", "node_modules"} and not d.startswith(".venv")
            ]
            paths.update(Path(directory) / name for name in files if name in NAMES)
    refs, collisions, references, missing = [], [], set(), []
    for path in sorted(paths):
        if time.monotonic() - start > 180:
            raise TimeoutError("180 s provenance inventory ceiling")
        raw = path.read_bytes()
        value = json.loads(raw)
        ref = {"path": str(path), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
        refs.append(ref)
        found = mentions(value, IDS)
        if found:
            collisions.append({**ref, "seeds": sorted(found)})
        references.update(csv_paths(value))
    fingerprints = []
    for path in sorted(references):
        if not path.exists():
            missing.append(str(path))
            continue
        qpos = np.loadtxt(path, delimiter=",")
        fingerprints.append(
            {**artifact(path), "canonical_motion_hash": canonical_motion_hash(qpos)}
        )
    out = DATA / "m2s-final-transfer-provenance-v1"
    out.mkdir(exist_ok=False)
    write_new(
        out / "ledger.json",
        {
            "protocol": artifact(
                ROOT / "docs/motion2scene/FINAL_SOURCE_PROVENANCE_RESERVATION_V1.md"
            ),
            "driver": artifact(Path(__file__)),
            "range_checker": artifact(
                ROOT / "scripts/research/motion2scene_reserve_transfer_sources_v4.py"
            ),
            "roots": [str(p) for p in (ROOT, SIBLING, DATA)],
            "selected_names": sorted(NAMES),
            "provenance_files": refs,
            "collisions": collisions,
            "csv_fingerprints": fingerprints,
            "missing_referenced_csvs": missing,
            "seconds": time.monotonic() - start,
            "scope": "explicit named acquisition provenance only; earlier whole-JSON attempts remain unresolved",
        },
    )
    if collisions or missing:
        raise ValueError("reservation refused: collision or unresolved prior source fingerprint")
    write_new(
        out / "reservation.json",
        {
            "source_ids": IDS,
            "ledger": artifact(out / "ledger.json"),
            "split": "final_transfer_test_only",
            "generation_started": False,
            "training_eligible": False,
            "candidate_count": 8,
            "replacement_allowed": False,
            "grouping": "all descendants, scene variants, physics repeats and sensor traces by ancestor",
            "post_generation_gate": "reject fingerprint duplicates against all ledger CSVs and new siblings",
            "scope": "IDs reserved against explicit provenance; no global metadata or external-history claim",
            "qualification": "unstarted; requires separate frozen acquisition and same-switch contract",
        },
    )
    print(
        json.dumps(
            {
                "reserved": IDS,
                "provenance_files": len(refs),
                "fingerprinted_csvs": len(fingerprints),
                "seconds": time.monotonic() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
