#!/usr/bin/env python
"""Assemble everything that exists into one browsable release folder.

Two views, kept separate because they answer different questions and because collapsing them is
how a corpus comes to look larger than its evidence:

* **motions/** -- the qualification view. One row per reference clip: what was asked for, what the
  reference actually does, whether the embodiment can reach it, whether SONIC tracked it, and why
  it was rejected if it was. ``prompt_intent`` is recorded as an intent, never as a verified label,
  because an audit found only 24 of 75 predicate-bearing prompts actually contained their target
  behaviour.
* **families/** -- the counterfactual view. One directory per family, holding the exact scene
  geometry physics loaded, the per-cell verdict with its contact attribution, the raw trajectory,
  and the renders.

Raw trajectories are authoritative and are copied, not summarised. Rejected cells are kept: a
family's negative *is* the evidence, and a corpus that stores only its successes cannot support a
claim about what fails.

Everything is fingerprinted. Numbers quoted anywhere else should be reproducible from MANIFEST.json
alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import build_counterfactual_family as cf  # noqa: E402

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.scene_route_check import (  # noqa: E402
    check_route_meets_obstacle,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
SPLITS = REPO_ROOT / "gear_sonic/data/splits"

#: Which cells belong to a family, and what role each plays. Role is read from the name because it
#: determines what the cell is *evidence for*, and a physics-first classification loses that.
ROLES = ("nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_of(cell: Path) -> str | None:
    """The scene id physics loaded, read from whichever log name the builder of the day used.

    Matched families write ``<cell>.runner.log``; the earlier mined families write ``<cell>.log``.
    Checking only one silently dropped every mined family's scene geometry from the release, which
    is the single most important artifact a family has -- the claim is about that geometry.
    """
    for name in (f"{cell.name}.runner.log", f"{cell.name}.log"):
        log = cell.parent / "logs" / name
        if log.exists():
            match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
            if match:
                return match.group(1)
    return None


def grade(cell: Path) -> dict:
    paths = sorted(cell.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return {"status": "missing"}
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        return {"status": "unevaluable", "trajectory": str(paths[0])}
    outcome = classify_episode(cell.name, payload)
    diagnostics = outcome.diagnostics or {}
    decomposition = diagnostics.get("contact_decomposition") or {}
    record = {
        "status": outcome.outcome,
        "rejection_reasons": list(outcome.rejection_reasons),
        "frames": int(len(payload["root_pos_w"])),
        "drift_rate_mps": round(float(diagnostics.get("drift_rate_mps", 0.0)), 4),
        "contact": {
            "external_n": round(float(decomposition.get("max_external_contact_force_n", 0.0)), 1),
            "overhead_n": round(float(decomposition.get("max_overhead_contact_force_n", 0.0)), 1),
            "lateral_n": round(float(decomposition.get("max_lateral_contact_force_n", 0.0)), 1),
            "self_n": round(float(decomposition.get("max_self_contact_force_n", 0.0)), 1),
            "bodies": decomposition.get("external_contact_bodies") or [],
            "frame": diagnostics.get("max_nonfoot_contact_frame"),
        },
        "trajectory": str(paths[0]),
    }
    scene_id = scene_of(cell)
    if scene_id and (SCENES / f"{scene_id}.usda").exists():
        record["scene"] = scene_id
        try:
            box = cf.rendered_shelf_box(SCENES / f"{scene_id}.usda")
            import numpy as np

            check = check_route_meets_obstacle(
                np.asarray(payload["root_pos_w"])[:, :2], (box[0], box[1], box[3], box[4])
            )
            record["obstacle"] = {
                "underside_m": round(box[2], 4),
                "route_reaches_it": bool(check.passes),
                "frames_inside": int(check.frames_inside),
            }
        except Exception:  # noqa: BLE001 - a scene without a shelf is legitimate
            pass
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families-root", type=Path, nargs="+", required=True)
    ap.add_argument("--screen", type=Path, help="taxonomy screen.json, the qualification view")
    ap.add_argument("--clips", type=Path, help="directory of reference motion CSVs")
    ap.add_argument("--renders", type=Path, help="directory of prebuilt mp4s")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    for sub in ("families", "motions", "splits", "schema"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    family_rows = []
    manifests = []
    for root in args.families_root:
        # A matched family carries its manifest beside the work dir; a mined one carries
        # family.json inside it. Both shapes are read rather than one being normalised away,
        # because the difference is real: mined families pair two separately generated clips.
        manifests += [(p, root / p.stem) for p in sorted(root.glob("*.json"))]
        manifests += [
            (d / "family.json", d)
            for d in sorted(root.iterdir())
            if d.is_dir() and (d / "family.json").exists()
        ]
    seen = set()
    for manifest_path, work in manifests:
        if not work.is_dir() or work in seen:
            continue
        seen.add(work)
        meta = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        family_id = work.name
        target = out / "families" / family_id
        (target / "scenes").mkdir(parents=True, exist_ok=True)
        (target / "cells").mkdir(parents=True, exist_ok=True)

        cells, scenes_used = {}, set()
        for cell_dir in sorted(work.iterdir()):
            if not cell_dir.is_dir() or cell_dir.name in ("logs", "motions"):
                continue
            record = grade(cell_dir)
            if record.get("status") == "missing":
                continue
            cells[cell_dir.name] = record
            if record.get("scene"):
                scenes_used.add(record["scene"])
            slot = target / "cells" / cell_dir.name
            slot.mkdir(parents=True, exist_ok=True)
            (slot / "outcome.json").write_text(json.dumps(record, indent=2) + "\n")
            raw = record.get("trajectory")
            if raw and Path(raw).exists():
                shutil.copy2(raw, slot / "trajectory.pkl")

        for scene_id in sorted(scenes_used):
            source = SCENES / f"{scene_id}.usda"
            if source.exists():
                shutil.copy2(source, target / "scenes" / source.name)

        if args.renders:
            rendered = target / "renders"
            for pattern, label in (
                ("room_{}.mp4", "room"),
                ("{}_third_person.mp4", "side"),
                ("{}_ego.mp4", "ego"),
            ):
                for cell_name in cells:
                    source = args.renders / pattern.format(cell_name)
                    if source.exists():
                        rendered.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, rendered / f"{cell_name}__{label}.mp4")

        # Whether the family is *verified* is decided by the release page's role mapping, which
        # handles both cell-naming conventions; deciding it here from one convention was how a
        # count of four survived beside a correct count of three.
        record = {
            "family_id": family_id,
            "geometry": {k: meta[k] for k in meta if k.endswith("_m") or k.endswith("_m_")},
            "window_m": meta.get("window_m"),
            "cells": {name: c["status"] for name, c in cells.items()},
            "cell_count": len(cells),
            "scenes": sorted(scenes_used),
        }
        (target / "family.json").write_text(
            json.dumps({**record, "cell_detail": cells, "source_manifest": meta}, indent=2) + "\n"
        )
        family_rows.append(record)
        print(
            f"{family_id:>16s}  {len(cells)} cells, {len(scenes_used)} scenes"
            f"{'  (renders)' if args.renders else ''}"
        )

    (out / "families" / "index.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in family_rows)
    )

    # --- the motion qualification view ---------------------------------------------------
    motion_rows = []
    if args.screen and args.screen.exists():
        screen = json.loads(args.screen.read_text())
        for record in screen.get("records", []):
            name = record.get("csv", "")
            # The prompt is an *intent*. An audit found only 24 of 75 predicate-bearing prompts
            # actually contained their target behaviour, so it is never stored as a verified
            # label and never as the same field as one.
            motion_rows.append(
                {
                    "clip": name,
                    "prompt_intent": re.sub(r"^\d+_|_s\d+\.csv$", "", name).replace("_", " "),
                    "frames": record.get("frames"),
                    "embodiment_feasible": bool(record.get("passed")),
                    "rejection_reasons": record.get("reasons", []),
                    "root_height_min_m": record.get("root_height_min_m"),
                    "saturated_frame_fraction": record.get("saturated_frame_fraction"),
                    "reference_semantic_valid": None,
                    "executed_semantic_valid": None,
                    "sonic_tracking_outcome": None,
                }
            )
    if motion_rows:
        (out / "motions" / "index.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in motion_rows)
        )
    if args.clips and args.clips.exists():
        clip_dir = out / "motions" / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for row in motion_rows:
            source = args.clips / row["clip"]
            if source.exists():
                shutil.copy2(source, clip_dir / source.name)
    print(
        f"{len(motion_rows)} motions in the qualification view, "
        f"{sum(1 for r in motion_rows if r['embodiment_feasible'])} embodiment-feasible"
    )
    for split in SPLITS.glob("*.json"):
        shutil.copy2(split, out / "splits" / split.name)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "generated_from": str(args.families_root),
        "families": len(family_rows),
        "cells": sum(r["cell_count"] for r in family_rows),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "fingerprint_sha256": hashlib.sha256(
            "\n".join(f"{p.relative_to(out)}:{sha256(p)}" for p in files).encode()
        ).hexdigest(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\n{manifest['families']} families, {manifest['cells']} cells, "
        f"{manifest['bytes']/1024/1024:.1f} MB"
    )
    print(f"fingerprint {manifest['fingerprint_sha256'][:16]}...")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
