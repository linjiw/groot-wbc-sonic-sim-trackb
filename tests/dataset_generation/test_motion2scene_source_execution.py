"""Protect source denominators and the beam-only scene intervention."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import motion2scene_source_execution as study


def panel():
    return [
        {"source": s, "label": label, "pass": True, "tracker_outcome": "accepted"}
        for s in study.SOURCES
        for label in ("neutral", "d055")
    ]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_source"])
def test_qualification_rejects_incomplete_denominators(kind):
    rows = panel()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["source"] = 43001
    with pytest.raises(ValueError):
        study.qualification_map(rows)


def test_qualification_requires_both_passage_and_tracking():
    rows = panel()
    rows[0]["pass"] = False
    rows[3]["tracker_outcome"] = "rejected"
    verdicts = study.qualification_map(rows)
    assert sum(verdicts.values()) == 6
    assert not verdicts["41005"] and not verdicts["41006"]


def test_distinct_outputs_do_not_refill_using_acceptance():
    scenes = np.array([[0.2, 1.2], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4]])
    assert study.distinct_indices(scenes) == [0, 2, 3]
    assert study.distinct_indices(scenes[:2]) == [0]


def test_scene_edit_preserves_everything_outside_beam(tmp_path):
    original = (study.PRIOR / "beam_present.usda").read_text()
    path = tmp_path / "beam.usda"
    study.scene_file(path, {"center_xy_m": [1.0, 2.0], "underside_m": 1.2, "yaw_rad": 0.0})
    result = path.read_text()
    marker = '    def Cube "CounterfactualBeam"'
    assert result.split(marker)[0] == original.split(marker)[0]
    assert "xformOp:translate = (1.0, 2.0, 1.25)" in result.split(marker)[1]
    assert "xformOp:rotateZ = 0.0" in result.split(marker)[1]
    assert "physics:collisionEnabled = true" in result.split(marker)[1]
