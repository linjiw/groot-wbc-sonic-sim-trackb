#!/usr/bin/env python3
"""Freeze a bounded, metadata-stratified cohort from official BONES-SEED.

Selection uses only published metadata and the release filename filter. It does
not inspect policy metrics, which keeps the cohort independent of either sampler
arm. The emitted member lists support selective extraction from the archive-only
Hugging Face releases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data_process.filter_and_copy_bones_data import (  # noqa: E402
    DEFAULT_FILTER_KEYWORDS,
    should_filter_out,
)

REQUIRED_COLUMNS = {
    "move_name",
    "filename",
    "move_duration_frames",
    "package",
    "category",
    "is_mirror",
    "move_g1_path",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(seed: int, motion_key: str) -> str:
    return hashlib.sha256(f"{seed}\0{motion_key}".encode()).hexdigest()


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _duration_thresholds(durations: list[int], bins: int) -> list[int]:
    ordered = sorted(durations)
    return [ordered[min(len(ordered) - 1, (len(ordered) * index) // bins)] for index in range(1, bins)]


def _duration_bin(frames: int, thresholds: list[int]) -> int:
    return sum(frames > threshold for threshold in thresholds)


def _allocate_quotas(capacities: dict[str, int], size: int) -> dict[str, int]:
    if size > sum(capacities.values()):
        raise ValueError(f"requested {size} motions but only {sum(capacities.values())} are eligible")
    nonempty = sorted(key for key, count in capacities.items() if count)
    if size < len(nonempty):
        raise ValueError(f"cohort size {size} cannot cover all {len(nonempty)} category-duration strata")

    # Guarantee metadata coverage first, then allocate remaining slots in
    # proportion to each stratum's remaining population via largest remainder.
    quotas = {key: 1 for key in nonempty}
    remaining = size - len(nonempty)
    residual_capacity = {key: capacities[key] - 1 for key in nonempty}
    residual_total = sum(residual_capacity.values())
    if remaining and residual_total:
        exact = {key: remaining * residual_capacity[key] / residual_total for key in nonempty}
        for key in nonempty:
            quotas[key] += min(residual_capacity[key], int(exact[key]))
        unfilled = size - sum(quotas.values())
        order = sorted(
            nonempty,
            key=lambda key: (-(exact[key] - int(exact[key])), key),
        )
        while unfilled:
            progressed = False
            for key in order:
                if quotas[key] < capacities[key]:
                    quotas[key] += 1
                    unfilled -= 1
                    progressed = True
                    if not unfilled:
                        break
            if not progressed:
                raise RuntimeError("quota allocation exhausted capacity unexpectedly")
    return quotas


def build_cohort(metadata_csv: Path, *, size: int, seed: int, duration_bins: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    filtered = 0
    with metadata_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata CSV is missing required columns: {sorted(missing)}")
        for source in reader:
            move_name = source["move_name"].strip()
            filename = source["filename"].strip()
            g1_member = source["move_g1_path"].strip()
            path = Path(g1_member)
            if not move_name or not filename or not g1_member or path.stem != filename:
                raise ValueError(
                    "metadata filename/path mismatch: "
                    f"move_name={move_name!r}, filename={filename!r}, path={g1_member!r}"
                )
            release_filter_key = f"{path.parent.name}/{path.stem}.pkl"
            if should_filter_out(release_filter_key, DEFAULT_FILTER_KEYWORDS):
                filtered += 1
                continue
            rows.append(
                {
                    # The released robot converter and SMPL pairing both key by
                    # archive filename stem. ``move_name`` occasionally contains
                    # a human-readable spelling that differs from that stem.
                    "motion_key": filename,
                    "metadata_move_name": move_name,
                    "duration_source_frames": int(source["move_duration_frames"]),
                    "package": source["package"].strip() or "UNKNOWN",
                    "category": source["category"].strip() or "UNKNOWN",
                    "is_mirror": _as_bool(source["is_mirror"]),
                    "g1_archive_member": g1_member,
                    "smpl_archive_member": f"smpl_filtered/{filename}.pkl",
                    "release_filter_key": release_filter_key,
                }
            )

    if len({row["motion_key"] for row in rows}) != len(rows):
        raise ValueError("eligible metadata contains duplicate motion keys")
    thresholds = _duration_thresholds([row["duration_source_frames"] for row in rows], duration_bins)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bin_index = _duration_bin(row["duration_source_frames"], thresholds)
        row["duration_bin"] = bin_index
        row["stratum"] = f"{row['category']}|duration_q{bin_index}"
        strata[row["stratum"]].append(row)

    quotas = _allocate_quotas({key: len(value) for key, value in strata.items()}, size)
    selected: list[dict[str, Any]] = []
    for stratum, candidates in strata.items():
        candidates.sort(key=lambda row: (_stable_rank(seed, row["motion_key"]), row["motion_key"]))
        for row in candidates[: quotas[stratum]]:
            row["selection_rank_sha256"] = _stable_rank(seed, row["motion_key"])
            selected.append(row)
    selected.sort(key=lambda row: row["motion_key"])

    selection_digest = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "bones_seed_official_metadata_cohort",
        "source": {
            "repo_id": "bones-studio/seed",
            "repo_type": "dataset",
            "revision": "2f59b2077b9da34dd4e43618e705c7cb962c9a66",
            "metadata_filename": "metadata/seed_metadata_v004.csv",
            "metadata_sha256": _sha256_file(metadata_csv),
            "published_motion_count": len(rows) + filtered,
        },
        "eligibility": {
            "rule": "official_release_filename_filter",
            "filter_keywords": list(DEFAULT_FILTER_KEYWORDS),
            "eligible_motion_count": len(rows),
            "filtered_motion_count": filtered,
        },
        "selection": {
            "seed": seed,
            "size": size,
            "stratification": "category_x_eligible_duration_quantile",
            "duration_bins": duration_bins,
            "duration_thresholds_source_frames": thresholds,
            "minimum_per_nonempty_stratum": 1,
            "tie_breaker": "sha256(seed + NUL + motion_key)",
            "selection_sha256": selection_digest,
            "selected_category_counts": dict(sorted(Counter(row["category"] for row in selected).items())),
            "selected_package_counts": dict(sorted(Counter(row["package"] for row in selected).items())),
            "selected_mirror_counts": {
                str(key).lower(): value
                for key, value in sorted(Counter(row["is_mirror"] for row in selected).items())
            },
        },
        "motions": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1-members-output", type=Path)
    parser.add_argument("--smpl-members-output", type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--duration-bins", type=int, default=4)
    args = parser.parse_args()
    if args.size <= 0 or args.duration_bins <= 0:
        parser.error("--size and --duration-bins must be positive")

    cohort = build_cohort(
        args.metadata_csv.resolve(),
        size=args.size,
        seed=args.seed,
        duration_bins=args.duration_bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for output, field in (
        (args.g1_members_output, "g1_archive_member"),
        (args.smpl_members_output, "smpl_archive_member"),
    ):
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(f"{row[field]}\n" for row in cohort["motions"]), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": len(cohort["motions"]),
                "eligible": cohort["eligibility"]["eligible_motion_count"],
                "selection_sha256": cohort["selection"]["selection_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
