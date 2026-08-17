#!/usr/bin/env python3
"""Freeze the corpus into one manifest every report must read from.

Walks the trajectory store once, grades each episode under the current policy, hashes the
trajectory and its clip, and writes a named snapshot. Reports built from it cannot disagree
with each other, because there is only one pass.

Usage::

    python scripts/research/build_corpus_snapshot.py --work /data/.../groot-wbc-kimodo-m0 \\
        --name g1_motionbank_v0.3 --out /data/.../snapshots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.corpus_snapshot import (  # noqa: E402
    EpisodeRecord,
    build_snapshot,
    file_digest,
    trajectory_fingerprint,
)
from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Must match the version recorded by audit_gate_regrade, so a snapshot says which rules
#: produced its verdicts.
GATE_VERSION = "2026-08-17.policy-v1"

#: Directories holding *review renders* of episodes that already exist elsewhere -- an
#: alternate camera of the same rollout, not a new episode. Counting them inflates the
#: corpus with duplicates of itself; the snapshot's duplicate count was how they surfaced.
REVIEW_RENDER_DIRS = frozenset({"multiview", "thirdperson", "renders_review"})

#: Camera-view names that appear as leaf directories under a multi-view render.
CAMERA_VIEW_NAMES = frozenset({"ego", "chase", "overhead", "wrist"})


def is_review_render(path: Path) -> bool:
    """Whether this trajectory is a re-render of an episode counted elsewhere."""
    parts = set(path.parts)
    return bool(parts & REVIEW_RENDER_DIRS) or path.parent.parent.name in CAMERA_VIEW_NAMES


def infer_behaviour(episode_id: str, specs: list[dict]) -> tuple[str, str]:
    match = re.search(r"(?:clutter_)?(\d{3})_", episode_id)
    if match:
        index = int(match.group(1))
        if index < len(specs):
            return specs[index]["body_mode"], specs[index]["speed"]
    return "walk", "steady"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--name", required=True, help="snapshot name, e.g. g1_motionbank_v0.3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument(
        "--hash-videos",
        action="store_true",
        help="hash clips too; slower, but makes the snapshot reproducible end to end",
    )
    args = parser.parse_args()

    specs: list[dict] = []
    taxonomy_path = args.taxonomy or (args.work / "taxonomy/taxonomy.json")
    if taxonomy_path.exists():
        specs = json.loads(taxonomy_path.read_text(encoding="utf-8"))["specs"]

    records: list[EpisodeRecord] = []
    skipped_renders = 0
    for path in sorted(args.work.rglob("*.trajectory.pkl")):
        if is_review_render(path):
            skipped_renders += 1
            continue
        rollout_dir = path.parent.parent
        episode_id = rollout_dir.name
        behaviour, speed = infer_behaviour(episode_id, specs)

        videos = sorted(p for p in rollout_dir.glob("renders/*.mp4") if p.stat().st_size > 4096)
        video = videos[0] if videos else None

        try:
            with path.open("rb") as handle:
                raw = pickle.load(handle)  # noqa: S301 - local recorder artifact
        except Exception as error:  # noqa: BLE001 - a bad file is a record, not a crash
            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    trajectory_path=str(path),
                    trajectory_sha256=file_digest(path),
                    trajectory_fingerprint="unreadable",
                    frames=0,
                    fps=0.0,
                    outcome="unevaluable",
                    rejection_reasons=(),
                    demoted_failures=(),
                    errors=(f"{type(error).__name__}: {error}",),
                    recovered_from_split=False,
                    policy="n/a",
                    behaviour=behaviour,
                    speed_style=speed,
                    video_path=str(video) if video else None,
                    video_sha256=file_digest(video) if (video and args.hash_videos) else None,
                )
            )
            continue

        outcome = classify_episode(episode_id, raw)
        try:
            payload, _ = best_evaluable_payload(raw)
            fingerprint = trajectory_fingerprint(payload)
            frames = int(payload.get("total_frames", 0))
            fps = float(payload.get("fps", 0.0))
        except Exception:  # noqa: BLE001 - unevaluable captures still get a record
            fingerprint = "unevaluable"
            frames, fps = 0, 0.0

        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                trajectory_path=str(path),
                trajectory_sha256=file_digest(path),
                trajectory_fingerprint=fingerprint,
                frames=frames,
                fps=fps,
                outcome=outcome.outcome,
                rejection_reasons=outcome.rejection_reasons,
                demoted_failures=outcome.demoted_failures,
                errors=outcome.errors,
                recovered_from_split=outcome.recovered_from_split,
                policy=outcome.policy,
                behaviour=behaviour,
                speed_style=speed,
                video_path=str(video) if video else None,
                video_sha256=file_digest(video) if (video and args.hash_videos) else None,
                diagnostics={
                    k: v for k, v in outcome.diagnostics.items()
                    if isinstance(v, (int, float, str, bool))
                },
            )
        )

    snapshot = build_snapshot(args.name, GATE_VERSION, records)
    payload = snapshot.to_dict()

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.name}.json"
    if destination.exists():
        raise SystemExit(
            f"{destination} already exists. A snapshot is frozen: pick a new --name rather "
            "than overwriting, so verdicts stay comparable across gate changes."
        )
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    counts = payload["counts"]
    print(f"snapshot {args.name}  (gate {GATE_VERSION})")
    if skipped_renders:
        print(f"  skipped {skipped_renders} review re-render(s) of episodes counted elsewhere")
    print(f"  episodes {counts['episodes']}   evaluable {counts['evaluable']}   "
          f"accepted {counts['accepted']}")
    print(f"  distinct executed trajectories {counts['distinct_executed_trajectories']}"
          f"   duplicates {counts['duplicate_episodes']}")
    print(f"  clips {counts['with_video']}   missing {counts['missing_video']}")
    for record in payload["missing_clips"]:
        print(f"     no clip: {record['episode_id']}  ({record['outcome']})")
    if payload["uninformative_families"]:
        print("\n  families whose rate spans more than half the unit interval "
              "(too few episodes to quote):")
        for name in payload["uninformative_families"]:
            stats = payload["families"][name]
            print(f"     {name:18s} {stats['accepted']}/{stats['episodes']}  "
                  f"[{stats['wilson_low']:.2f}, {stats['wilson_high']:.2f}]")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
