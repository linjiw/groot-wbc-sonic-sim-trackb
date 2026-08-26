#!/usr/bin/env python3
"""Index every executed LFH rollout as one delivery-model training row.

The row joins what was *commanded* (a reference qpos clip, plus the operator that produced it)
to what the frozen controller *delivered* (the recorded Isaac verdict and tracking diagnostics).

This is the supervision the design plan calls D_phi and has never validated: given a reference
motion alone, will SONIC track it, and how much of the commanded edit survives execution.
Nothing here re-grades a cell; outcomes are copied verbatim from the immutable run records.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
RUN_RECORDS = DATA_ROOT / "hallucination/run_records"


def _reference_csv(motion_pkl: Path) -> Path | None:
    """Locate the reference qpos CSV a motion pickle was built from."""
    manifest = motion_pkl.with_suffix(motion_pkl.suffix + ".manifest.json")
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        candidate = payload.get("input", {}).get("path")
        if candidate and Path(candidate).exists():
            return Path(candidate)
    sibling = motion_pkl.with_suffix(".csv")
    return sibling if sibling.exists() else None


def _baseline_csv(reference: Path) -> Path | None:
    """The nominal reference the adapted clip was edited from, if one is stored beside it."""
    stem = reference.stem
    if stem in {"nominal", "w_nominal"} or stem.endswith("__nominal"):
        return None
    for candidate in (
        reference.with_name("nominal.csv"),
        reference.with_name("w_nominal.csv"),
        reference.with_name(stem.split("__")[0] + "__nominal.csv"),
    ):
        if candidate.exists():
            return candidate
    return None


_SILHOUETTE_CACHE: dict[str, np.ndarray] = {}
_HALFWIDTH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _silhouette_of(path: Path, qpos: np.ndarray) -> np.ndarray:
    from gear_sonic.dataset_generation.local_adaptation import _silhouette
    from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

    key = str(path)
    if key not in _SILHOUETTE_CACHE:
        _SILHOUETTE_CACHE[key] = _silhouette(qpos, DEFAULT_G1_MJCF)
    return _SILHOUETTE_CACHE[key]


def _half_widths_of(path: Path, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from gear_sonic.dataset_generation.local_adaptation import _signed_half_widths
    from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

    key = str(path)
    if key not in _HALFWIDTH_CACHE:
        _HALFWIDTH_CACHE[key] = _signed_half_widths(qpos, DEFAULT_G1_MJCF)
    return _HALFWIDTH_CACHE[key]


def _commanded_edit(reference: Path, qpos: np.ndarray) -> dict:
    """Measure the commanded edit from the stored clip pair, not from a manifest.

    Most calibration clips predate the manifest ``operator`` block, so the amplitude is
    recovered geometrically: it is a property of the two reference clips and needs no
    provenance beyond them.
    """
    from gear_sonic.dataset_generation.local_adaptation import active_frames

    blank = {
        "role": "nominal",
        "baseline_csv": "",
        "commanded_drop_m": None,
        "commanded_left_reduction_m": None,
        "commanded_right_reduction_m": None,
        "commanded_max_joint_change_rad": None,
        "commanded_active_fraction": None,
    }
    baseline_path = _baseline_csv(reference)
    if baseline_path is None:
        return blank
    baseline = np.loadtxt(baseline_path, delimiter=",")
    if baseline.shape != qpos.shape:
        return blank
    mask = active_frames(baseline, qpos)
    if not mask.any():
        mask = np.ones(len(qpos), dtype=bool)
    drop = (_silhouette_of(baseline_path, baseline) - _silhouette_of(reference, qpos))[mask]
    base_left, base_right = _half_widths_of(baseline_path, baseline)
    edit_left, edit_right = _half_widths_of(reference, qpos)
    return {
        "role": "adapted",
        "baseline_csv": str(baseline_path),
        "commanded_drop_m": float(drop.min()),
        "commanded_left_reduction_m": float((base_left - edit_left)[mask].min()),
        "commanded_right_reduction_m": float((base_right - edit_right)[mask].min()),
        "commanded_max_joint_change_rad": float(np.abs(qpos[:, 7:] - baseline[:, 7:]).max()),
        "commanded_active_fraction": float(mask.mean()),
    }


def _route_features(qpos: np.ndarray) -> dict:
    root = np.asarray(qpos[:, :2], dtype=np.float64)
    steps = np.linalg.norm(np.diff(root, axis=0), axis=1)
    path_length = float(steps.sum())
    net = float(np.linalg.norm(root[-1] - root[0]))
    heading = np.unwrap(np.arctan2(np.diff(root[:, 1]), np.diff(root[:, 0])))
    frames = int(qpos.shape[0])
    return {
        "frames": frames,
        "path_length_m": path_length,
        "net_displacement_m": net,
        "straightness": net / path_length if path_length > 1e-9 else 0.0,
        "mean_speed_mps": path_length / (frames / 30.0) if frames else 0.0,
        "heading_change_rad": float(abs(heading[-1] - heading[0])) if len(heading) > 1 else 0.0,
        "root_height_mean_m": float(np.mean(qpos[:, 2])),
        "root_height_min_m": float(np.min(qpos[:, 2])),
    }


def rows_from_records(records_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text())
        for cell_id, cell in (record.get("cells") or {}).items():
            scientific = cell.get("scientific") or {}
            outcome = scientific.get("outcome")
            if outcome is None:
                continue
            command = cell.get("command") or []
            motion = scene = None
            for index, token in enumerate(command):
                if token == "--motion" and index + 1 < len(command):
                    motion = Path(command[index + 1])
                if token == "--scene" and index + 1 < len(command):
                    scene = command[index + 1]
            if motion is None or not motion.exists():
                continue
            reference = _reference_csv(motion)
            if reference is None:
                continue
            qpos = np.loadtxt(reference, delimiter=",")
            if qpos.ndim != 2:
                continue
            diagnostics = scientific.get("diagnostics") or {}
            contact = diagnostics.get("contact_decomposition") or {}
            row = {
                "experiment": record_path.stem,
                "cell_id": cell_id,
                "motion_pkl": str(motion),
                "reference_csv": str(reference),
                "scene": scene or "",
                "outcome": outcome,
                "accepted": int(outcome == "accepted"),
                "rejection_reasons": "|".join(scientific.get("rejection_reasons") or []),
                "endpoint_error_m": diagnostics.get("endpoint_error_m"),
                "schedule_error_p95_m": diagnostics.get("schedule_error_p95_m"),
                "drift_rate_mps": diagnostics.get("drift_rate_mps"),
                "max_external_contact_force_n": contact.get("max_external_contact_force_n"),
                "max_self_contact_force_n": contact.get("max_self_contact_force_n"),
                "executed_frames": scientific.get("frames"),
            }
            row.update(_route_features(qpos))
            row.update(_commanded_edit(reference, qpos))
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RUN_RECORDS)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/delivery_corpus.csv"
    )
    args = parser.parse_args()

    rows = rows_from_records(args.records)
    if not rows:
        raise SystemExit("no delivery rows recovered")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    accepted = sum(row["accepted"] for row in rows)
    print(f"{len(rows)} executed cells -> {args.out}")
    print(f"accepted {accepted}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
