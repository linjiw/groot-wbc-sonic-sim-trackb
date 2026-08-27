#!/usr/bin/env python3
"""Build LfLH candidate sets from real clips: a nominal plus crouches and one-sided arm tucks.

The inverse problem is only interesting when the scene has something to choose *between*. Each
clip contributes one candidate set: the nominal at zero edit cost, crouches at several depths, and
left/right arm tucks at several amplitudes. The observed motion is whichever edit we are asking the
scene to explain.

Edit cost is the operator's own reference-side magnitude, normalised per operator so a crouch and
a tuck are comparable: the decoder's lexicographic rule needs a single ordering over edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.motion_envelope import (  # noqa: E402
    extract_envelope,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_arm_tuck,
    local_crouch,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402

STATION_FRACTION = 0.55
WINDOW_FRACTION = 0.30
STATIONS = 16
#: Edits offered to the decoder. Cost is in normalised edit units, monotone within each operator.
CROUCH_DROPS_M = (0.040, 0.055, 0.070)
TUCK_REDUCTIONS_M = (0.040, 0.070)


def build_candidate_set(index: int, path: Path) -> dict | None:
    qpos = np.loadtxt(path, delimiter=",")
    if not screen_reference(qpos, f"cand_{index:03d}", "walk").worth_a_rollout:
        return None
    fractions = np.linspace(0.25, 0.85, STATIONS)

    def envelope(clip: np.ndarray, motion_id: str):
        tracks = extract_keypoints(payload_from_reference(clip))
        return extract_envelope(tracks, motion_id, fractions=fractions)

    labels = ["nominal"]
    costs = [0.0]
    envelopes = [envelope(qpos, f"{index:03d}_nominal")]

    for drop in CROUCH_DROPS_M:
        adapted, report = local_crouch(
            qpos, STATION_FRACTION, target_drop_m=drop, window=WINDOW_FRACTION
        )
        if report.excursion_capped or not report.root_path_preserved:
            continue
        labels.append(f"crouch_{int(1000 * drop):03d}")
        # Normalised so a 70 mm crouch and a 70 mm tuck cost the same to a first approximation.
        costs.append(report.silhouette_drop_m / 0.070)
        envelopes.append(envelope(adapted, f"{index:03d}_crouch{int(1000 * drop):03d}"))

    for side in ("left", "right"):
        for reduction in TUCK_REDUCTIONS_M:
            adapted, report = local_arm_tuck(
                qpos,
                STATION_FRACTION,
                target_reduction_m=reduction,
                window=WINDOW_FRACTION,
                side=side,
            )
            if not report.root_path_preserved:
                continue
            labels.append(f"tuck_{side}_{int(1000 * reduction):03d}")
            costs.append(reduction / 0.070)
            envelopes.append(
                envelope(adapted, f"{index:03d}_tuck{side}{int(1000 * reduction):03d}")
            )

    if len(envelopes) < 4:
        return None
    return {
        "motion_index": index,
        "source_csv": str(path),
        "labels": labels,
        "costs": costs,
        "stations": STATIONS,
        "fractions": fractions.tolist(),
        "station_xy_m": envelopes[0].station_xy_m.tolist(),
        "yaw_rad": envelopes[0].yaw_rad.tolist(),
        "extents": np.stack([item.stack() for item in envelopes]).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument(
        "--reference-gate",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/coverage/reference_gate.json",
    )
    parser.add_argument("--clips", type=int, default=24)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/lflh_candidates.json"
    )
    args = parser.parse_args()

    gate = json.loads(args.reference_gate.read_text())
    built = []
    for row in gate:
        if len(built) >= args.clips:
            break
        if not row.get("worth_a_rollout"):
            continue
        matches = sorted(args.source_dir.glob(f"{row['index']:03d}_*.csv"))
        if len(matches) != 1:
            continue
        record = build_candidate_set(row["index"], matches[0])
        if record is None:
            continue
        record["body_mode"] = row.get("body_mode", "")
        built.append(record)
        print(
            f"  [{len(built)}/{args.clips}] {row['index']:03d} {record['body_mode']:<14} "
            f"{len(record['labels'])} candidates",
            flush=True,
        )

    args.out.write_text(
        json.dumps(
            {
                "schema_version": "lflh_candidates_v1",
                "stations": STATIONS,
                "station_fraction": STATION_FRACTION,
                "clips": len(built),
                "candidate_sets": built,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(built)} candidate sets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
