#!/usr/bin/env python3
"""Infer a critical-scene proposal from a single reference motion, before spending physics.

This is the motion-to-scene inference step. Given one generated clip it applies the crouch
amplitude ladder, predicts the executed overhead reach of the nominal and of each rung with the
cross-validated ``D_phi`` fit, solves the predicted window, and reports which amplitude -- if any
-- is worth a rollout.

Three things it deliberately does not do. It does not author geometry: a predicted window is a
reason to spend two rollouts, never a scene. It does not predict a physics verdict. And it does
not hide its uncertainty: every window is reported both raw and after subtracting the model's
leave-one-motion-out q90 band from each side, and the refusal is decided on the conservative one.
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
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from scripts.research.hallucination.build_reach_response_corpus import (  # noqa: E402
    FACE_ACROSS_M,
    FACE_ALONG_M,
    STATION_FRACTION,
    _route_axis,
    _station,
)
from scripts.research.hallucination.screen_crouch_ladder import (  # noqa: E402
    LADDER_M,
    WINDOW_FRACTION,
)

#: A window narrower than this is not worth a rollout; it matches the corpus-wide floor.
MIN_WINDOW_MM = 20.0


def _reach(qpos: np.ndarray, station, axis) -> float:
    tracks = extract_keypoints(payload_from_reference(qpos))
    face = overhead_face_reach(
        tracks, station, axis, FACE_ALONG_M, FACE_ACROSS_M, require_all_groups=False
    )
    return float(face.reach_m)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "docs/hallucination/reach_delivery_model.json"
    )
    parser.add_argument("--station-fraction", type=float, default=STATION_FRACTION)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model = json.loads(args.model.read_text())
    slope = model["fitted_linear"]["slope"]
    intercept = model["fitted_linear"]["intercept_m"]
    band_mm = model["cross_validation"]["scores"][model["cross_validation"]["best_model"]][
        "q90_abs_mm"
    ]

    def predict(commanded_m: float) -> float:
        return slope * commanded_m + intercept

    qpos = np.loadtxt(args.reference_csv, delimiter=",")
    gate = screen_reference(qpos, args.reference_csv.stem, "walk")
    root_xy = qpos[:, :2]
    axis = _route_axis(root_xy)
    station = _station(root_xy, args.station_fraction)
    progress = route_progress(np.asarray(root_xy, dtype=np.float64))
    straightness = float(
        np.linalg.norm(root_xy[-1] - root_xy[0])
        / max(float(np.linalg.norm(np.diff(root_xy, axis=0), axis=1).sum()), 1e-9)
    )

    nominal_commanded = _reach(qpos, station, axis)
    nominal_predicted = predict(nominal_commanded)

    rungs = []
    for target in LADDER_M:
        adapted, report = local_crouch(
            qpos, args.station_fraction, target_drop_m=target, window=WINDOW_FRACTION
        )
        rung_gate = screen_reference(adapted, f"{args.reference_csv.stem}_rung", "walk")
        usable = bool(
            rung_gate.worth_a_rollout
            and report.root_path_preserved
            and not report.excursion_capped
            and report.silhouette_drop_m >= 0.9 * target
        )
        adapted_commanded = _reach(adapted, station, axis)
        adapted_predicted = predict(adapted_commanded)
        raw_mm = 1000 * (nominal_predicted - adapted_predicted)
        # Both endpoints carry the same model, so the band is subtracted from each side.
        conservative_mm = raw_mm - 2 * band_mm
        rungs.append(
            {
                "target_drop_mm": 1000 * target,
                "reference_drop_mm": 1000 * report.silhouette_drop_m,
                "reference_gate_usable": usable,
                "commanded_reach_nominal_m": nominal_commanded,
                "commanded_reach_adapted_m": adapted_commanded,
                "predicted_reach_nominal_m": nominal_predicted,
                "predicted_reach_adapted_m": adapted_predicted,
                "predicted_window_mm": raw_mm,
                "predicted_window_after_band_mm": conservative_mm,
                "worth_a_rollout": bool(usable and conservative_mm >= MIN_WINDOW_MM),
            }
        )

    promising = [rung for rung in rungs if rung["worth_a_rollout"]]
    best = max(promising, key=lambda rung: rung["predicted_window_after_band_mm"], default=None)
    decision = "propose_empty_scene_calibration" if best else "refuse_no_predicted_window"
    report_payload = {
        "schema_version": "lfh_motion_scene_proposal_v1",
        "reference_csv": str(args.reference_csv),
        "route": {
            "axis": axis,
            "station_xy_m": list(station),
            "straightness": straightness,
            "progress_at_station": float(
                progress[int(np.argmin(np.abs(progress - args.station_fraction)))]
            ),
        },
        "nominal_reference_gate": {
            "worth_a_rollout": gate.worth_a_rollout,
            "diagnosis": gate.diagnosis,
        },
        "delivery_model": {
            "path": str(args.model),
            "best_model": model["cross_validation"]["best_model"],
            "leave_one_motion_out_q90_mm": band_mm,
            "rmse_reduction_vs_identity": model["cross_validation"]["rmse_reduction_vs_identity"],
        },
        "minimum_window_mm": MIN_WINDOW_MM,
        "ladder": rungs,
        "recommended_rung_mm": best["target_drop_mm"] if best else None,
        "decision": decision,
        "contract": (
            "proposal only; a recommended rung licenses empty-scene physics for the pair and "
            "nothing else. Geometry is authored only from accepted executed reaches."
        ),
        "does_not_predict": [
            "whether the adapted clip will track: D_phi is fitted on accepted rollouts and "
            "estimates reach conditional on acceptance, never acceptance itself. E9a failed on "
            "tracking, not on reach, and this tool would not have caught it.",
            "any physics verdict, in any scene",
        ],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")

    print(f"{args.reference_csv.name}")
    print(
        f"  route axis {axis}  straightness {straightness:.3f}  "
        f"nominal gate {'pass' if gate.worth_a_rollout else 'REFUSED'}"
    )
    print(f"  predicted nominal reach {nominal_predicted:.4f} m  (band +/-{band_mm:.1f} mm)")
    for rung in rungs:
        flag = (
            "propose"
            if rung["worth_a_rollout"]
            else ("thin" if rung["reference_gate_usable"] else "refused")
        )
        print(
            f"    {rung['target_drop_mm']:5.0f} mm -> window {rung['predicted_window_mm']:6.1f} mm "
            f"(after band {rung['predicted_window_after_band_mm']:6.1f})  {flag}"
        )
    print(f"  DECISION: {decision}" + (f" at {best['target_drop_mm']:.0f} mm" if best else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
