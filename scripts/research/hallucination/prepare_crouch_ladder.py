#!/usr/bin/env python3
"""Materialize an E12 amplitude-ladder cohort from the crouch ladder screen.

One motion contributes one nominal clip and its usable ladder rungs.  The operator block is
written into every motion manifest, which the older calibration clips lack -- that omission is
why `build_delivery_corpus.py` could recover a commanded amplitude for only one of 52 adapted
cells without re-measuring the clip pair.

Reference-side artifacts only.  Physics decides which rungs deliver.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    active_frames,
    local_crouch,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from scripts.research.hallucination.prepare_arm_tuck_strong_calibration import (  # noqa: E402
    artifact_set,
)
from scripts.research.hallucination.prepare_probe_candidates import (  # noqa: E402
    DATA_ROOT,
    _save_motion,
)
from scripts.research.hallucination.screen_crouch_ladder import (  # noqa: E402
    STATION_FRACTION,
    WINDOW_FRACTION,
)


def _write_provenance(path: Path, key: str, source_csv: Path, operator: dict | None) -> Path:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    payload = {
        "schema_version": "lfh_ladder_motion_v1",
        "converter": "gear_sonic.dataset_generation.kimodo_motion_adapter",
        "motion_key": key,
        "source_fps": 30.0,
        "canonicalize_horizontal_origin": True,
        "scene_start_xyz": [0.0, 0.0, 0.0],
        "scene_yaw": 0.0,
        "operator": operator,
        "input": {"path": str(source_csv), "sha256": sha256_file(source_csv)},
        "output": {"path": str(path), "sha256": sha256_file(path)},
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest.chmod(0o664)
    return manifest


def select(screen: dict, *, cohort: int, exclude_sourced: bool) -> list[dict]:
    """Prefer body-mode diversity, then depth of usable ladder."""
    eligible = [
        clip
        for clip in screen["clips"]
        if clip["usable_rungs"] >= 2 and not (exclude_sourced and clip["already_sourced"])
    ]
    eligible.sort(key=lambda clip: (-clip["usable_rungs"], -clip["deepest_usable_mm"]))
    chosen: list[dict] = []
    taken: set[int] = set()
    seen_modes: set[str] = set()
    # First pass takes the best clip of each body mode, so the cohort cannot collapse onto one
    # gait; the second pass fills the remaining slots by ladder depth regardless of mode.
    for mode_pass in (True, False):
        for clip in eligible:
            if len(chosen) >= cohort:
                break
            index = int(clip["motion_index"])
            if index in taken:
                continue
            if mode_pass and clip["body_mode"] in seen_modes:
                continue
            chosen.append(clip)
            taken.add(index)
            seen_modes.add(clip["body_mode"])
    return chosen[:cohort]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen", type=Path, default=REPO_ROOT / "docs/hallucination/crouch_ladder_screen.json"
    )
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "lfh_crouch_ladder")
    parser.add_argument("--cohort", type=int, default=12)
    parser.add_argument("--max-rungs", type=int, default=3)
    parser.add_argument("--include-sourced", action="store_true")
    args = parser.parse_args()

    screen = json.loads(args.screen.read_text())
    chosen = select(screen, cohort=args.cohort, exclude_sourced=not args.include_sourced)
    if not chosen:
        raise SystemExit("the ladder screen yielded no eligible clip")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for clip in chosen:
        index = int(clip["motion_index"])
        source = Path(clip["source_csv"])
        nominal = np.loadtxt(source, delimiter=",")
        pair_id = f"ladder_{index:03d}"
        pair_dir = args.out / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        nominal_csv = pair_dir / "nominal.csv"
        np.savetxt(nominal_csv, nominal, delimiter=",", fmt="%.10f")
        nominal_motion = pair_dir / "nominal.pkl"
        nominal_key = f"{pair_id}__nominal"
        _save_motion(nominal_motion, nominal_key, nominal)
        nominal_manifest = _write_provenance(nominal_motion, nominal_key, nominal_csv, None)

        usable = [
            rung
            for rung in clip["rungs"]
            if rung["worth_a_rollout"]
            and rung["root_path_preserved"]
            and not rung["excursion_capped"]
            and rung["reference_drop_mm"] >= 0.9 * rung["target_drop_mm"]
        ]
        usable.sort(key=lambda rung: rung["target_drop_mm"])
        rungs = []
        for rung in usable[: args.max_rungs]:
            target = rung["target_drop_mm"] / 1000.0
            adapted, report = local_crouch(
                nominal, STATION_FRACTION, target_drop_m=target, window=WINDOW_FRACTION
            )
            if abs(1000 * report.silhouette_drop_m - rung["reference_drop_mm"]) > 1e-6:
                raise SystemExit(f"{pair_id}: operator output changed since the screen")
            gate = screen_reference(adapted, f"{pair_id}_{int(1000 * target):03d}", "walk")
            if not gate.worth_a_rollout or not report.root_path_preserved:
                raise SystemExit(f"{pair_id}: adapted reference gate changed since the screen")
            label = f"d{int(round(1000 * target)):03d}"
            adapted_csv = pair_dir / f"{label}.csv"
            np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
            adapted_motion = pair_dir / f"{label}.pkl"
            adapted_key = f"{pair_id}__{label}"
            _save_motion(adapted_motion, adapted_key, adapted)
            operator = {
                "name": "local_crouch",
                "station_fraction": STATION_FRACTION,
                "window_fraction": WINDOW_FRACTION,
                "target_drop_m": target,
                "report": asdict(report),
            }
            adapted_manifest = _write_provenance(adapted_motion, adapted_key, adapted_csv, operator)
            rungs.append(
                {
                    "label": label,
                    "target_drop_m": target,
                    "reference_silhouette_drop_m": report.silhouette_drop_m,
                    "lateral_coupling_mm": rung["lateral_coupling_mm"],
                    "max_joint_change_rad": report.max_joint_change_rad,
                    "active_frames": np.flatnonzero(active_frames(nominal, adapted)).tolist(),
                    "operator_report": asdict(report),
                    "artifacts": artifact_set(adapted_csv, adapted_motion, adapted_manifest),
                }
            )

        pairs.append(
            {
                "pair_id": pair_id,
                "motion_index": index,
                "body_mode": clip["body_mode"],
                "route": clip["route"],
                "source_csv": str(source),
                "source_sha256": sha256_file(source),
                "nominal_artifacts": artifact_set(nominal_csv, nominal_motion, nominal_manifest),
                "rungs": rungs,
            }
        )

    payload = {
        "schema_version": "lfh_crouch_ladder_candidates_v1",
        "operator": "local_crouch",
        "station_fraction": STATION_FRACTION,
        "window_fraction": WINDOW_FRACTION,
        "screen": str(args.screen.relative_to(REPO_ROOT)),
        "screen_sha256": sha256_file(args.screen),
        "selection_basis": (
            "body-mode diversity first, then usable ladder depth; motions already anchoring "
            "verified critical support are excluded unless --include-sourced"
        ),
        "pairs": pairs,
    }
    manifest = args.out / "candidates.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rung_total = sum(len(pair["rungs"]) for pair in pairs)
    print(f"PASS: {len(pairs)} ladder pairs, {rung_total} rungs -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
