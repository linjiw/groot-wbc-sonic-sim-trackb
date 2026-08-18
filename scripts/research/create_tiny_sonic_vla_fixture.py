#!/usr/bin/env python3
"""Create a tiny schema-valid SONIC/VLA LeRobot-style fixture dataset.

This fixture is for validator/manifest plumbing only. It contains synthetic
numeric rows and placeholder videos; it is not suitable for model training or
performance evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

PROMPT = "pick up the red cup and place it on the tray"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def modality_json(profile: str = "live_vr") -> dict:
    state = {
        "observation.state": {
            "start": 0,
            "end": 43,
            "original_key": "observation.state",
        },
        "observation.eef_state": {
            "start": 43,
            "end": 55,
            "original_key": "observation.eef_state",
        },
        "observation.root_orientation": {
            "start": 55,
            "end": 59,
            "original_key": "observation.root_orientation",
        },
        "observation.projected_gravity": {
            "start": 59,
            "end": 62,
            "original_key": "observation.projected_gravity",
        },
    }
    if profile == "synthetic_g1":
        state = {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "right_arm": {"start": 29, "end": 36},
            "left_hand": {"start": 22, "end": 29},
            "right_hand": {"start": 36, "end": 43},
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        }
    return {
        "state": state,
        "action": {
            "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.right_hand_joints",
            },
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view",
                "path": "videos/observation.images.ego_view",
            }
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def make_rows(episodes: int, frames: int, fps: int, profile: str = "live_vr") -> list[dict]:
    rows: list[dict] = []
    for episode_index in range(episodes):
        for frame_index in range(frames):
            t = (episode_index * frames + frame_index) / fps
            base = episode_index + frame_index * 0.01
            row = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": t,
                "task_index": 0,
                "observation.state": (np.full(43, base, dtype=np.float32)).tolist(),
                "observation.projected_gravity": [0.0, 0.0, -1.0],
                "action.motion_token": (
                    np.linspace(0.0, 1.0, 64, dtype=np.float32) + base
                ).tolist(),
                "teleop.left_hand_joints": (
                    np.linspace(0.0, 0.2, 7, dtype=np.float32) + base
                ).tolist(),
                "teleop.right_hand_joints": (
                    np.linspace(0.0, 0.2, 7, dtype=np.float32) + base
                ).tolist(),
            }
            if profile == "synthetic_g1":
                qpos = np.zeros(36, dtype=np.float32)
                qpos[:3] = [base, 0.0, 0.8]
                qpos[3] = 1.0
                row["reference.g1_qpos"] = qpos.tolist()
            else:
                row.update(
                    {
                        "observation.eef_state": np.full(12, base + 0.1, dtype=np.float32).tolist(),
                        "observation.root_orientation": [0.0, 0.0, 0.0, 1.0],
                        "teleop.smpl_pose": (
                            np.linspace(0.01, 0.63, 63, dtype=np.float32) + base
                        ).tolist(),
                    }
                )
            rows.append(row)
    return rows


def create_fixture(
    output: Path, episodes: int, frames: int, fps: int, profile: str = "live_vr"
) -> None:
    if output.exists():
        shutil.rmtree(output)
    (output / "meta").mkdir(parents=True)
    (output / "data").mkdir(parents=True)
    video_dir = output / "videos" / "observation.images.ego_view"
    video_dir.mkdir(parents=True)

    write_json(output / "meta" / "modality.json", modality_json(profile))
    write_json(
        output / "meta" / "info.json",
        {
            "fps": fps,
            "total_episodes": episodes,
            "total_frames": episodes * frames,
            "discarded_episode_indices": [],
            "description": f"Schema-only SONIC/VLA {profile} fixture; not for training.",
        },
    )
    write_jsonl(output / "meta" / "tasks.jsonl", [{"task_index": 0, "task": PROMPT}])
    write_jsonl(
        output / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": episode_index,
                "length": frames,
                "tasks": [PROMPT],
            }
            for episode_index in range(episodes)
        ],
    )

    df = pd.DataFrame(make_rows(episodes, frames, fps, profile))
    df.to_parquet(output / "data" / "train-00000-of-00001.parquet", index=False)

    # The validator only requires an ego-view mp4 to be present. These placeholder
    # files deliberately avoid pretending to be real camera recordings.
    for episode_index in range(episodes):
        (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--profile", choices=("live_vr", "synthetic_g1"), default="live_vr")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.frames < 2:
        parser.error("--frames must be >= 2")
    create_fixture(args.output, args.episodes, args.frames, args.fps, args.profile)
    print(f"wrote tiny SONIC/VLA fixture to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
