from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.research.build_bones_seed_official_cohort import build_cohort

FIELDS = [
    "move_name",
    "filename",
    "move_duration_frames",
    "package",
    "category",
    "is_mirror",
    "move_g1_path",
]


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, *, category: str, frames: int, name: str | None = None) -> dict[str, str]:
    filename = name or f"motion_{index:02d}"
    return {
        "move_name": f"semantic_{index:02d}",
        "filename": filename,
        "move_duration_frames": str(frames),
        "package": "Locomotion",
        "category": category,
        "is_mirror": str(index % 2 == 0),
        "move_g1_path": f"g1/csv/session/{filename}.csv",
    }


def test_cohort_is_deterministic_stratified_and_release_filtered(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    rows = [
        *[_row(index, category="A", frames=100 + index * 10) for index in range(3)],
        *[_row(index + 3, category="B", frames=130 + index * 10) for index in range(3)],
        *[_row(index + 6, category="A", frames=500 + index * 10) for index in range(3)],
        *[_row(index + 9, category="B", frames=530 + index * 10) for index in range(3)],
        _row(99, category="B", frames=700, name="sit_filtered_motion"),
    ]
    _write_metadata(metadata, rows)

    first = build_cohort(metadata, size=4, seed=17, duration_bins=2)
    second = build_cohort(metadata, size=4, seed=17, duration_bins=2)

    assert first == second
    assert first["source"]["published_motion_count"] == 13
    assert first["eligibility"]["eligible_motion_count"] == 12
    assert first["eligibility"]["filtered_motion_count"] == 1
    assert len(first["motions"]) == 4
    assert len({row["stratum"] for row in first["motions"]}) == 4
    assert all(row["motion_key"] == Path(row["g1_archive_member"]).stem for row in first["motions"])
    assert all("sit_filtered_motion" not in row["motion_key"] for row in first["motions"])


def test_cohort_rejects_filename_path_mismatch(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    row = _row(0, category="A", frames=100)
    row["move_g1_path"] = "g1/csv/session/different.csv"
    _write_metadata(metadata, [row])

    with pytest.raises(ValueError, match="filename/path mismatch"):
        build_cohort(metadata, size=1, seed=0, duration_bins=1)
