"""Repeated layouts cannot silently reweight the carrier-averaged endpoint."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_icra_compare import compare  # noqa: E402


def test_carrier_mean_and_discordance_denominator():
    rows = []
    for i, source in enumerate([1, 1, 1, 2]):
        rows += [
            {"arm": "a", "group_id": str(i), "source": source, "pass": source == 1},
            {"arm": "b", "group_id": str(i), "source": source, "pass": source == 2},
        ]
    result = compare(rows, "a", "b")
    assert result["carrier_averaged_difference"] == 0
    assert result["counts"] == {"both_pass": 0, "only_first": 3, "only_second": 1, "both_fail": 0}
    assert result["paired_conditions"] == 4
    with pytest.raises(AssertionError):
        compare(rows[:-1], "a", "b")
