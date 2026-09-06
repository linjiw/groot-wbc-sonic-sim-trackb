"""Leakage and aggregation checks for the source/phase experiment."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_source_phase import ARMS, selected_ids, study_split, summarize  # noqa: E402


def test_group_and_phase_exclusion():
    cases = {}
    for seed in list(range(41001, 41009)) + list(range(42001, 42009)):
        for index, station in enumerate([0.25, 0.5, 0.75, 0.35, 0.65]):
            cases[f"{seed}_{index}"] = {
                "metadata": {
                    "carrier_seed": seed,
                    "split": study_split(seed),
                    "event_station": station,
                }
            }
    for arm in ARMS.values():
        ids = selected_ids(cases, arm["parents"])
        assert len(ids) == 3 * len(arm["parents"])
        assert all(cases[k]["metadata"]["split"] == "train" for k in ids)
        assert all(cases[k]["metadata"]["event_station"] not in [0.35, 0.65] for k in ids)
    for invalid in [[41007], [42005], [41001, 41001]]:
        with pytest.raises(ValueError):
            selected_ids(cases, invalid)
    with pytest.raises(ValueError):
        study_split(43001)
    cases.pop("41001_0")
    with pytest.raises(ValueError):
        selected_ids(cases, [41001])


def test_summary_keeps_parent_phase_and_seed():
    base = dict(
        arm="base4",
        budget=600,
        carrier_seed=41007,
        phase="unseen",
        split="test",
        seed=8421,
        valid=2,
        proposals=8,
        target_failures=3,
        neutral_failures=4,
    )
    result = summarize(
        [
            base,
            {**base, "valid": 3},
            {**base, "seed": 8422},
            {**base, "carrier_seed": 41008},
            {**base, "phase": "seen"},
        ]
    )
    assert len(result) == 3
    first = result[0]
    assert (first["valid"], first["proposals"]) == (7, 24)
    assert first["by_seed"] == {
        "8421": {"valid": 5, "proposals": 16},
        "8422": {"valid": 2, "proposals": 8},
    }
    assert first["target_failures"] == 9 and first["neutral_failures"] == 12
