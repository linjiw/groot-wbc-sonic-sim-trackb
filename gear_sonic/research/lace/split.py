"""Deterministic source-disjoint partitioning for BONES motion cohorts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

from gear_sonic.research.lace.schema import (
    SPLIT_KIND,
    SPLIT_NAMES,
    SPLIT_SCHEMA_VERSION,
    canonical_sha256,
    split_selection_sha256,
)

DEFAULT_SPLIT_RATIOS = {
    "D_atlas": 0.20,
    "D_curriculum": 0.45,
    "D_geometry": 0.15,
    "D_controller": 0.10,
    "D_test": 0.10,
}
_ACTOR_PATTERN = re.compile(r"__(A\d+)(?:_M)?(?:\.[^.]+)?$")


def derive_source_group_id(record: Mapping[str, Any]) -> str:
    """Return a conservative recording-source identity from cohort metadata.

    Explicit source/subject fields win. BONES metadata otherwise exposes a stable
    actor identifier (``A###``) in its provenance key. This identifier is not a
    semantic motion label and groups every mirror, crop, and variant from the same
    actor. If a future dataset lacks such provenance, construction fails rather
    than falling back to motion-name similarity.
    """

    for field in ("source_group_id", "source_recording_id", "subject_id", "actor_id"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field in ("release_filter_key", "g1_archive_member", "motion_key"):
        value = record.get(field)
        if not isinstance(value, str):
            continue
        match = _ACTOR_PATTERN.search(value)
        if match:
            return f"bones_actor:{match.group(1)}"
    raise ValueError(f"cannot derive source_group_id for motion {record.get('motion_key')!r}")


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"lace-split-v1:{seed}:{value}".encode()).hexdigest()


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must have exactly these keys: {list(SPLIT_NAMES)}")
    result = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(value <= 0.0 for value in result.values()):
        raise ValueError("all split ratios must be positive")
    total = sum(result.values())
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"split ratios must sum to one, got {total}")
    return result


def _assignment_cost(
    partition: str,
    group_records: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float],
    total_motions: int,
    total_groups: int,
    stratum_totals: Mapping[str, int],
    motion_counts: Mapping[str, int],
    group_counts: Mapping[str, int],
    stratum_counts: Mapping[str, Counter[str]],
) -> float:
    """Incremental normalized imbalance after assigning a complete source group."""

    ratio = ratios[partition]
    group_strata = Counter(
        str(record.get("stratum", record.get("category", "unknown"))) for record in group_records
    )

    def squared_error(actual: float, target: float) -> float:
        return ((actual - target) / max(target, 1.0)) ** 2

    before = squared_error(motion_counts[partition], ratio * total_motions)
    after = squared_error(motion_counts[partition] + len(group_records), ratio * total_motions)
    before += 0.25 * squared_error(group_counts[partition], ratio * total_groups)
    after += 0.25 * squared_error(group_counts[partition] + 1, ratio * total_groups)
    for stratum, increment in group_strata.items():
        target = ratio * stratum_totals[stratum]
        before += 0.5 * squared_error(stratum_counts[partition][stratum], target)
        after += 0.5 * squared_error(stratum_counts[partition][stratum] + increment, target)
    return after - before


def build_source_disjoint_split(
    motions: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, pre-rollout-stratified, source-disjoint manifest."""

    split_ratios = _validate_ratios(ratios)
    records = [dict(record) for record in motions]
    if not records:
        raise ValueError("motions must be non-empty")
    motion_keys = [record.get("motion_key") for record in records]
    if any(not isinstance(key, str) or not key for key in motion_keys):
        raise ValueError("every motion requires a non-empty motion_key")
    if len(set(motion_keys)) != len(motion_keys):
        raise ValueError("motion_key values must be unique")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_group_id = derive_source_group_id(record)
        record["source_group_id"] = source_group_id
        groups[source_group_id].append(record)
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError(f"at least {len(SPLIT_NAMES)} source groups are required")

    stratum_totals: Counter[str] = Counter(
        str(record.get("stratum", record.get("category", "unknown"))) for record in records
    )
    motion_counts = {name: 0 for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    stratum_counts = {name: Counter() for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    ordered_groups = sorted(
        groups, key=lambda group: (-len(groups[group]), _stable_rank(seed, group))
    )

    for group_index, source_group_id in enumerate(ordered_groups):
        group_records = groups[source_group_id]
        unfilled = [name for name in SPLIT_NAMES if group_counts[name] == 0]
        groups_remaining = len(ordered_groups) - group_index
        candidates = unfilled if len(unfilled) >= groups_remaining else list(SPLIT_NAMES)
        partition = min(
            candidates,
            key=lambda name: (
                _assignment_cost(
                    name,
                    group_records,
                    ratios=split_ratios,
                    total_motions=len(records),
                    total_groups=len(groups),
                    stratum_totals=stratum_totals,
                    motion_counts=motion_counts,
                    group_counts=group_counts,
                    stratum_counts=stratum_counts,
                ),
                _stable_rank(seed, f"{source_group_id}:{name}"),
            ),
        )
        assignment[source_group_id] = partition
        motion_counts[partition] += len(group_records)
        group_counts[partition] += 1
        stratum_counts[partition].update(
            str(record.get("stratum", record.get("category", "unknown")))
            for record in group_records
        )

    output_records: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda value: str(value["motion_key"])):
        selected = {
            key: record[key]
            for key in (
                "motion_key",
                "source_group_id",
                "category",
                "package",
                "stratum",
                "duration_source_frames",
                "is_mirror",
                "g1_archive_member",
                "smpl_archive_member",
                "robot_path",
                "smpl_path",
                "available_modalities",
            )
            if key in record
        }
        selected["partition"] = assignment[record["source_group_id"]]
        output_records.append(selected)

    summary = {
        name: {
            "motion_count": motion_counts[name],
            "source_group_count": group_counts[name],
            "stratum_counts": dict(sorted(stratum_counts[name].items())),
        }
        for name in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "kind": SPLIT_KIND,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "ratios": split_ratios,
        "grouping_rule": "explicit source id, else BONES actor provenance identifier",
        "stratification_fields": ["stratum", "motion_count", "source_group_count"],
        "dataset": dict(dataset or {}),
        "partition_summary": summary,
        "motions": output_records,
    }
    manifest["selection_sha256"] = split_selection_sha256(manifest)
    manifest["split_sha256"] = canonical_sha256(manifest)
    return manifest
