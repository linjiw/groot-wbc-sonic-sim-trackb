#!/usr/bin/env python3
"""Collect a browsable video gallery and compute the corpus distribution.

Rollouts accumulate across many working directories -- placement batches, clutter
batches, density sweeps, review renders. This gathers the ones worth looking at into
one folder with names that say what they are, and writes a distribution report
alongside so the numbers and the footage come from the same pass.

Videos are copied, not moved: the working directories stay intact so a rollout can
still be re-analysed or re-exported from where it was produced.

Usage::

    python scripts/research/build_dataset_gallery.py --work /data/.../groot-wbc-kimodo-m0 \\
        --out /data/.../gallery --json distribution.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.latent_parity import check_latent_parity  # noqa: E402
from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)

#: Where rollouts live, and the shelf each belongs on in the gallery.
SOURCES = [
    ("clean/rollouts", "01_placed_scenes", "Hand-authored scenes, planned placement"),
    ("clutter_rollouts", "02_generated_clutter", "Clutter generated around the motion path"),
    ("clutter3d_rollouts", "03_clutter_3d", "Vertical bands and cantilevered geometry"),
    ("density_rollouts", "04_density_ladder", "Sparse to tight density sweep"),
]
#: Review renders, which are alternate views of an episode rather than new episodes.
REVIEW = [("thirdperson", "05_third_person"), ("multiview", "06_multiview")]


def _tilt(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.maximum(np.abs(roll), np.abs(pitch))


def describe(trajectory_path: Path) -> dict:
    with trajectory_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)
    action = np.asarray(payload["action_motion_token"], dtype=np.float64)
    contact = decompose_payload_contacts(payload)
    report = evaluate_locomotion_trajectory(payload)
    parity = check_latent_parity(action)
    gates = {gate.name: gate for gate in report.gates}
    step = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    return {
        "frames": int(payload["total_frames"]),
        "fps": float(payload["fps"]),
        "duration_s": float(payload["total_frames"]) / float(payload["fps"]),
        "path_length_m": float(step.sum()),
        "net_displacement_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        "mean_speed_mps": float(step.sum() / (len(step) / float(payload["fps"]))),
        "root_height_min_m": float(root[:, 2].min()),
        "root_height_max_m": float(root[:, 2].max()),
        "max_tilt_rad": float(_tilt(payload["root_quat_w"]).max()),
        "reference_min_height_m": float(reference[:, 2].min()),
        "endpoint_error_m": float(gates["endpoint_error"].value) if "endpoint_error" in gates else None,
        "path_error_p95_m": float(gates["path_error_p95"].value) if "path_error_p95" in gates else None,
        "foot_support_fraction": float(gates["foot_support"].value) if "foot_support" in gates else None,
        "self_contact_n": contact.max_self_contact,
        "lateral_scene_contact_n": contact.max_lateral_contact,
        "support_contact_n": contact.max_support_contact,
        "accepted": bool(report.accepted),
        "rejection_reasons": list(report.rejection_reasons),
        "latent_verdict": parity.verdict,
        "latent_distinct_values": int(np.unique(action).size),
        "action_effective_rank": float(
            np.linalg.eigvalsh(np.cov(action.T) + 1e-12 * np.eye(action.shape[1])).clip(0).sum() ** 2
            / (np.linalg.eigvalsh(np.cov(action.T) + 1e-12 * np.eye(action.shape[1])).clip(0) ** 2).sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, dict] = {}
    copied = 0

    for relative, shelf, blurb in SOURCES:
        source = args.work / relative
        if not source.is_dir():
            continue
        shelf_dir = args.out / shelf
        shelf_dir.mkdir(exist_ok=True)
        (shelf_dir / "README.txt").write_text(f"{shelf}\n{blurb}\n", encoding="utf-8")
        for rollout in sorted(p for p in source.iterdir() if p.is_dir()):
            videos = sorted(rollout.glob("renders/*.mp4"))
            trajectories = sorted(rollout.glob("trajectories/*.trajectory.pkl"))
            if not videos or not trajectories:
                continue
            record = describe(trajectories[0])
            record["shelf"] = shelf
            record["source"] = str(rollout)
            status = "accepted" if record["accepted"] else "rejected"
            target = shelf_dir / f"{rollout.name}__{status}__ego.mp4"
            shutil.copy2(videos[0], target)
            record["video"] = str(target.relative_to(args.out))
            episodes[f"{shelf}/{rollout.name}"] = record
            copied += 1

    for relative, shelf in REVIEW:
        source = args.work / relative
        if not source.is_dir():
            continue
        shelf_dir = args.out / shelf
        shelf_dir.mkdir(exist_ok=True)
        for video in sorted(source.rglob("*.mp4")):
            if video.stat().st_size < 4096:  # never-finalised writer
                continue
            label = "__".join(video.relative_to(source).parts[:-1]) or video.stem
            shutil.copy2(video, shelf_dir / f"{label}.mp4")
            copied += 1

    # ---- distribution ----
    accepted = {k: v for k, v in episodes.items() if v["accepted"]}
    by_shelf = Counter(v["shelf"] for v in episodes.values())
    by_shelf_accepted = Counter(v["shelf"] for v in accepted.values())
    reasons = Counter(r for v in episodes.values() for r in v["rejection_reasons"])
    verdicts = Counter(v["latent_verdict"] for v in episodes.values())

    def spread(key: str, source: dict) -> dict:
        values = [v[key] for v in source.values() if v.get(key) is not None]
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }

    distribution = {
        "episodes_total": len(episodes),
        "episodes_accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(episodes) if episodes else 0.0,
        "videos_copied": copied,
        "by_shelf": dict(by_shelf),
        "by_shelf_accepted": dict(by_shelf_accepted),
        "rejection_reasons": dict(reasons),
        "latent_verdicts": dict(verdicts),
        "accepted_frames": int(sum(v["frames"] for v in accepted.values())),
        "accepted_duration_s": float(sum(v["duration_s"] for v in accepted.values())),
        "spread_all": {
            key: spread(key, episodes)
            for key in ("path_length_m", "mean_speed_mps", "max_tilt_rad", "root_height_min_m")
        },
        "spread_accepted": {
            key: spread(key, accepted)
            for key in (
                "path_length_m",
                "net_displacement_m",
                "mean_speed_mps",
                "endpoint_error_m",
                "path_error_p95_m",
                "foot_support_fraction",
                "self_contact_n",
                "lateral_scene_contact_n",
                "action_effective_rank",
            )
        },
        "episodes": episodes,
    }

    print(f"copied {copied} videos into {args.out}")
    print(f"episodes: {len(accepted)}/{len(episodes)} accepted "
          f"({distribution['acceptance_rate']:.0%}), "
          f"{distribution['accepted_duration_s']:.0f} s of accepted footage")
    for shelf, count in sorted(by_shelf.items()):
        print(f"  {shelf:22s} {by_shelf_accepted.get(shelf, 0):2d}/{count:2d} accepted")
    if reasons:
        print("rejection reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
    for key, stats in distribution["spread_accepted"].items():
        if stats:
            print(f"  {key:24s} min {stats['min']:8.3f}  median {stats['median']:8.3f}  max {stats['max']:8.3f}")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(distribution, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
