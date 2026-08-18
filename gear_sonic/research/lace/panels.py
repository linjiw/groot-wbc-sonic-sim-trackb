"""Representation-blind source panels for causal LACE transfer interventions."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Mapping

from gear_sonic.research.lace.schema import canonical_sha256, validate_split_manifest

PANEL_KIND = "lace_representation_blind_source_panels"
PANEL_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tie_break(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_representation_blind_panels(
    split_manifest: Mapping[str, Any],
    *,
    partition: str = "D_curriculum",
    panel_count: int = 8,
    seed: int = 20260814,
) -> dict[str, Any]:
    """Partition source groups without using policy or representation features.

    Groups are greedily balanced by pre-rollout reference duration. Seeded hashes
    break ties, making the result reproducible without consulting semantics,
    kinematics, difficulty, failure probes, or downstream transfer outcomes.
    """

    validate_split_manifest(split_manifest)
    _require(isinstance(partition, str) and partition, "partition must be non-empty")
    _require(isinstance(panel_count, int) and panel_count >= 2, "panel_count must be >= 2")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")

    records = [record for record in split_manifest["motions"] if record["partition"] == partition]
    _require(records, f"split contains no motions in {partition}")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        duration = record.get("duration_source_frames")
        _require(
            isinstance(duration, int) and not isinstance(duration, bool) and duration > 0,
            f"motion {record['motion_key']!r} lacks positive duration_source_frames",
        )
        grouped[record["source_group_id"]].append(record)
    _require(
        len(grouped) >= panel_count,
        f"partition has {len(grouped)} source groups, fewer than panel_count={panel_count}",
    )

    groups = []
    for source_group_id, members in grouped.items():
        groups.append(
            {
                "source_group_id": source_group_id,
                "motion_count": len(members),
                "duration_source_frames": sum(
                    int(member["duration_source_frames"]) for member in members
                ),
                "motion_keys": sorted(member["motion_key"] for member in members),
            }
        )
    groups.sort(
        key=lambda group: (
            -group["duration_source_frames"],
            -group["motion_count"],
            _tie_break(seed, group["source_group_id"]),
        )
    )

    assignments: list[list[dict[str, Any]]] = [[] for _ in range(panel_count)]
    stats = [
        {"source_group_count": 0, "motion_count": 0, "duration_source_frames": 0}
        for _ in range(panel_count)
    ]
    seeded_panel_order = sorted(
        range(panel_count),
        key=lambda index: _tie_break(seed, f"panel:{index}"),
    )
    panel_rank = {panel_index: rank for rank, panel_index in enumerate(seeded_panel_order)}

    for group in groups:
        panel_index = min(
            range(panel_count),
            key=lambda index: (
                stats[index]["duration_source_frames"],
                stats[index]["motion_count"],
                stats[index]["source_group_count"],
                panel_rank[index],
            ),
        )
        assignments[panel_index].append(group)
        stats[panel_index]["source_group_count"] += 1
        stats[panel_index]["motion_count"] += group["motion_count"]
        stats[panel_index]["duration_source_frames"] += group["duration_source_frames"]

    targets = {
        "source_group_count": len(groups) / panel_count,
        "motion_count": len(records) / panel_count,
        "duration_source_frames": sum(group["duration_source_frames"] for group in groups)
        / panel_count,
    }

    def objective(candidate_stats: list[dict[str, int]]) -> float:
        return sum(
            (
                (panel["duration_source_frames"] - targets["duration_source_frames"])
                / targets["duration_source_frames"]
            )
            ** 2
            + ((panel["motion_count"] - targets["motion_count"]) / targets["motion_count"]) ** 2
            + 0.2
            * (
                (panel["source_group_count"] - targets["source_group_count"])
                / targets["source_group_count"]
            )
            ** 2
            for panel in candidate_stats
        )

    # Deterministic local search removes the large motion-count imbalance that
    # duration-only LPT can leave when a source group contains many short clips.
    # It still uses only the pre-rollout quantities declared above.
    for _ in range(10_000):
        baseline = objective(stats)
        best: tuple[tuple[Any, ...], tuple[Any, ...], list[dict[str, int]]] | None = None
        for left in range(panel_count):
            for right in range(left + 1, panel_count):
                for left_index, left_group in enumerate(assignments[left]):
                    for right_index, right_group in enumerate(assignments[right]):
                        candidate = [dict(panel) for panel in stats]
                        for field in ("motion_count", "duration_source_frames"):
                            candidate[left][field] += right_group[field] - left_group[field]
                            candidate[right][field] += left_group[field] - right_group[field]
                        candidate_objective = objective(candidate)
                        sort_key = (
                            candidate_objective,
                            "swap",
                            left,
                            right,
                            left_group["source_group_id"],
                            right_group["source_group_id"],
                        )
                        if candidate_objective < baseline - 1e-12 and (
                            best is None or sort_key < best[0]
                        ):
                            best = (
                                sort_key,
                                ("swap", left, right, left_index, right_index),
                                candidate,
                            )
        for left in range(panel_count):
            if len(assignments[left]) <= 1:
                continue
            for right in range(panel_count):
                if left == right:
                    continue
                for left_index, group in enumerate(assignments[left]):
                    candidate = [dict(panel) for panel in stats]
                    for field in ("motion_count", "duration_source_frames"):
                        candidate[left][field] -= group[field]
                        candidate[right][field] += group[field]
                    candidate[left]["source_group_count"] -= 1
                    candidate[right]["source_group_count"] += 1
                    candidate_objective = objective(candidate)
                    sort_key = (
                        candidate_objective,
                        "move",
                        left,
                        right,
                        group["source_group_id"],
                    )
                    if candidate_objective < baseline - 1e-12 and (
                        best is None or sort_key < best[0]
                    ):
                        best = (
                            sort_key,
                            ("move", left, right, left_index),
                            candidate,
                        )
        if best is None:
            break
        operation = best[1]
        stats = best[2]
        if operation[0] == "swap":
            _, left, right, left_index, right_index = operation
            assignments[left][left_index], assignments[right][right_index] = (
                assignments[right][right_index],
                assignments[left][left_index],
            )
        else:
            _, left, right, left_index = operation
            assignments[right].append(assignments[left].pop(left_index))
    else:  # pragma: no cover - defensive guard for future objective changes
        raise RuntimeError("panel balancing did not converge")

    panels = []
    for index, (assigned_groups, panel_stats) in enumerate(zip(assignments, stats, strict=True)):
        panels.append(
            {
                "panel_id": f"source_panel_{index:02d}",
                "source_group_ids": sorted(group["source_group_id"] for group in assigned_groups),
                "motion_keys": sorted(
                    motion_key for group in assigned_groups for motion_key in group["motion_keys"]
                ),
                **panel_stats,
            }
        )

    manifest: dict[str, Any] = {
        "kind": PANEL_KIND,
        "schema_version": PANEL_SCHEMA_VERSION,
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "partition": partition,
        "panel_count": panel_count,
        "seed": seed,
        "assignment_inputs": [
            "source_group_id",
            "duration_source_frames",
            "motion_count",
            "seeded_hash_tie_break",
            "deterministic_duration_motion_local_search",
        ],
        "balance_objective": {
            "duration_weight": 1.0,
            "motion_count_weight": 1.0,
            "source_group_count_weight": 0.2,
            "value": objective(stats),
        },
        "representation_blind": True,
        "panels": panels,
    }
    manifest["panel_sha256"] = canonical_sha256(manifest)
    validate_panel_manifest(manifest, split_manifest=split_manifest)
    return manifest


def validate_panel_manifest(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    verify_digest: bool = True,
) -> None:
    """Validate panel coverage, source grouping, and optional split binding."""

    _require(manifest.get("kind") == PANEL_KIND, f"kind must be {PANEL_KIND!r}")
    _require(
        manifest.get("schema_version") == PANEL_SCHEMA_VERSION,
        "unsupported panel schema version",
    )
    _require(manifest.get("representation_blind") is True, "representation_blind must be true")
    panels = manifest.get("panels")
    panel_count = manifest.get("panel_count")
    _require(isinstance(panels, list) and len(panels) == panel_count, "panel_count mismatch")

    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_motions: set[str] = set()
    for index, panel in enumerate(panels):
        _require(isinstance(panel, Mapping), f"panels[{index}] must be a mapping")
        panel_id = panel.get("panel_id")
        groups = panel.get("source_group_ids")
        motions = panel.get("motion_keys")
        _require(isinstance(panel_id, str) and panel_id, f"panels[{index}].panel_id missing")
        _require(panel_id not in seen_ids, f"duplicate panel_id: {panel_id}")
        seen_ids.add(panel_id)
        _require(isinstance(groups, list) and groups, f"panels[{index}] source groups missing")
        _require(isinstance(motions, list) and motions, f"panels[{index}] motions missing")
        _require(not (seen_groups & set(groups)), "source group appears in multiple panels")
        _require(not (seen_motions & set(motions)), "motion appears in multiple panels")
        seen_groups.update(groups)
        seen_motions.update(motions)
        _require(panel.get("source_group_count") == len(groups), "source_group_count mismatch")
        _require(panel.get("motion_count") == len(motions), "motion_count mismatch")
        _require(panel.get("duration_source_frames", 0) > 0, "panel duration must be positive")

    if split_manifest is not None:
        validate_split_manifest(split_manifest)
        _require(
            manifest.get("split_sha256") == split_manifest["split_sha256"],
            "panel split_sha256 mismatch",
        )
        _require(
            manifest.get("split_selection_sha256") == split_manifest["selection_sha256"],
            "panel split_selection_sha256 mismatch",
        )
        partition = manifest.get("partition")
        expected_records = [
            record for record in split_manifest["motions"] if record["partition"] == partition
        ]
        expected_motions = {record["motion_key"] for record in expected_records}
        expected_groups = {record["source_group_id"] for record in expected_records}
        _require(seen_motions == expected_motions, "panels do not exactly cover partition motions")
        _require(seen_groups == expected_groups, "panels do not exactly cover partition groups")
        motion_to_group = {
            record["motion_key"]: record["source_group_id"] for record in expected_records
        }
        for panel in panels:
            panel_groups = set(panel["source_group_ids"])
            _require(
                all(motion_to_group[motion] in panel_groups for motion in panel["motion_keys"]),
                f"panel {panel['panel_id']} separates a motion from its source group",
            )

    if verify_digest:
        expected_digest = manifest.get("panel_sha256")
        _require(
            isinstance(expected_digest, str) and len(expected_digest) == 64,
            "panel_sha256 must be a SHA-256",
        )
        actual_digest = canonical_sha256(manifest, digest_field="panel_sha256")
        _require(expected_digest == actual_digest, "panel_sha256 mismatch")
