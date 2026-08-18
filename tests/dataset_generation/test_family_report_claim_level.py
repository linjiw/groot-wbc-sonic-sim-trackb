"""The claim level must be read from artefacts, never asserted.

A family page that says "perturbation robust" because the author believed it, rather than
because robustness.json says so, is the single most misleading thing this pipeline could
emit. The rule is that a stronger claim requires a file, and its absence downgrades rather
than defaults.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/research/build_family_report.py"
spec = importlib.util.spec_from_file_location("build_family_report", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
claim_level = module.claim_level


def write(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_no_artefacts_means_only_a_geometric_prediction(tmp_path):
    level, strength = claim_level(tmp_path)
    assert level == "geometrically predicted only"
    assert strength == "weak"


def test_pure_attribution_alone_is_not_robustness(tmp_path):
    write(tmp_path, "attribution.json", {"attribution_pure": True})
    level, strength = claim_level(tmp_path)
    assert "nominally physics verified" in level
    assert "not yet perturbed" in level
    assert strength == "medium"


def test_impure_attribution_does_not_reach_the_middle_rung(tmp_path):
    write(tmp_path, "attribution.json", {"attribution_pure": False})
    assert claim_level(tmp_path)[0] == "geometrically predicted only"


def test_only_a_passing_robustness_file_earns_the_strong_claim(tmp_path):
    write(tmp_path, "attribution.json", {"attribution_pure": True})
    write(tmp_path, "robustness.json", {"perturbation_robust": True})
    level, strength = claim_level(tmp_path)
    assert level == "perturbation robust"
    assert strength == "strong"


def test_a_failed_perturbation_says_so_rather_than_going_quiet(tmp_path):
    """Silence would leave the middle claim standing while the evidence against it exists."""
    write(tmp_path, "attribution.json", {"attribution_pure": True})
    write(tmp_path, "robustness.json", {"perturbation_robust": False})
    level, strength = claim_level(tmp_path)
    assert "did not hold" in level
    assert strength == "weak"


@pytest.mark.parametrize("robust", [True, False])
def test_robustness_outranks_attribution_either_way(tmp_path, robust):
    write(tmp_path, "robustness.json", {"perturbation_robust": robust})
    assert "geometrically predicted only" != claim_level(tmp_path)[0]
