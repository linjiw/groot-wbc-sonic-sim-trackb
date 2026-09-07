"""Common validity must not restore the contrast objective to the ablation."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "icra_study_test", SCRIPTS / "motion2scene_icra_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_ablation_and_uniform_do_not_require_future_walk_interference():
    point = np.array([0.6, 1.3])
    for arm in ("uniform", "no_contrast"):
        assert study.eligibility(arm, point, 0.02, False, True, False, []) == []
    for arm in ("analytic", "motion2scene"):
        assert study.eligibility(arm, point, 0.02, False, True, False, []) == ["contrast_audit"]
    assert study.eligibility("uniform", point, 0.02, False, False, False, []) == []
    assert study.eligibility("no_contrast", point, 0.02, False, False, False, []) == [
        "target_audit"
    ]


def test_matched_prefix_validity_is_common():
    for arm in study.ARMS:
        assert "predecision_clearance" in study.eligibility(
            arm, np.array([0.6, 1.3]), 0.0, True, True, False, []
        )


def test_absent_and_blocked_authored_boxes(tmp_path, monkeypatch):
    def template(path, beam):
        path.write_text(
            'room\n    def Cube "CounterfactualBeam"\n'
            "bool physics:collisionEnabled = true\n"
            "double3 xformOp:translate = (0, 0, 0)\n"
            "double3 xformOp:scale = (0.1, 1.2, 0.1)\n"
        )

    monkeypatch.setattr(study, "scene_file", template)
    beam = dict(center_xy_m=[2.0, 0.1], underside_m=0.0, thickness_m=1.4, length_m=0.1, width_m=1.2)
    absent = tmp_path / "absent.usda"
    blocked = tmp_path / "blocked.usda"
    study.physical_scene(absent, beam, absent=True)
    study.physical_scene(blocked, beam)
    assert "collisionEnabled = false" in absent.read_text()
    assert "collisionEnabled = true" in blocked.read_text()
    assert "translate = (2.0, 0.1, 0.7)" in blocked.read_text()
    assert "scale = (0.1, 1.2, 1.4)" in blocked.read_text()
