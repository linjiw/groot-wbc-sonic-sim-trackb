from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_d040_beam_execution import SEEDS, summarize  # noqa: E402


def panel():
    return [
        {
            "seed": seed,
            "label": label,
            "condition": condition,
            "pass": True,
            "tracker_outcome": "accepted",
        }
        for seed in SEEDS
        for label in ("neutral", "d040", "d055")
        for condition in ("absent", "present")
    ]


def test_intermediate_passage_falsifies_deep_crouch_necessity():
    rows = panel()
    assert summarize(rows)["predictions"] == {"p1": True, "p2": True}
    for row in rows:
        if row["label"] == "d040" and row["condition"] == "present":
            row["pass"] = False
    assert not summarize(rows)["predictions"]["p2"]
    next(r for r in rows if r["label"] == "d040" and r["condition"] == "absent")[
        "tracker_outcome"
    ] = "rejected"
    assert not summarize(rows)["predictions"]["p1"]


@pytest.mark.parametrize("error", ["missing", "duplicate", "wrong_seed"])
def test_all_seed_motion_condition_cells_required(error):
    rows = panel()
    if error == "missing":
        rows.pop()
    elif error == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["seed"] = 9999
    with pytest.raises(ValueError):
        summarize(rows)
