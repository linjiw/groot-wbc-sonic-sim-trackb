"""The nominal contract must differ from M2S-ICRA-v1 in exactly one place."""

from pathlib import Path
import sys

import numpy as np
import pytest

# The motion2scene research scripts import each other by bare module name and are run
# with scripts/research on PYTHONPATH; mirror that here rather than rewriting them.
RESEARCH = Path(__file__).resolve().parents[2] / "scripts/research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from motion2scene_envelope_tradeoff import margins  # noqa: E402
from motion2scene_icra_nominal import (  # noqa: E402
    CLEARANCE_M,
    eligibility as nominal_eligibility,
)
from motion2scene_icra_study import eligibility as inherited_eligibility  # noqa: E402

LAYOUTS = []
POINT = np.array([0.5, 1.27])


def _reasons(fn, arm, critical=True, target_clear=True, preclear=0.1):
    return fn(arm, POINT, preclear, critical, target_clear, False, LAYOUTS)


@pytest.mark.parametrize("arm", ["uniform", "analytic", "no_contrast", "motion2scene"])
def test_contracts_agree_whenever_geometry_agrees(arm):
    """With the same geometry verdicts both contracts must reach the same decision.

    The contracts differ only in *how* criticality is measured (nominal pose versus
    the inherited 113-offset envelope), never in what is done with the answer.
    """
    for critical in (True, False):
        for target_clear in (True, False):
            inherited = _reasons(inherited_eligibility, arm, critical, target_clear)
            nominal = _reasons(nominal_eligibility, arm, critical, target_clear)
            assert [r.replace("nominal_contrast_audit", "contrast_audit") for r in nominal] == (
                inherited
            )


def test_contrast_arms_still_require_a_contrast():
    for arm in ("analytic", "motion2scene"):
        assert _reasons(nominal_eligibility, arm, critical=False) == ["nominal_contrast_audit"]
    assert _reasons(nominal_eligibility, "no_contrast", target_clear=False) == ["target_audit"]
    assert _reasons(nominal_eligibility, "uniform", critical=False, target_clear=False) == []


def test_domain_and_clearance_bounds_are_unchanged():
    outside = np.array([0.95, 1.27])
    assert "domain" in nominal_eligibility("uniform", outside, 0.1, True, True, False, LAYOUTS)
    assert "predecision_clearance" in nominal_eligibility(
        "uniform", POINT, CLEARANCE_M / 2, True, True, False, LAYOUTS
    )
    assert "duplicate_within_arm" in nominal_eligibility(
        "uniform", POINT, 0.1, True, True, True, LAYOUTS
    )


def test_margins_take_the_worst_case_over_offsets():
    # values[scene, offset, channel]; channel 1 is the target, channel 0 the walk.
    values = np.array([[[-0.05, 0.03], [-0.02, 0.012]], [[-0.05, 0.03], [0.004, 0.02]]])
    mask = np.array([False, True])
    target, walk = margins(values, mask)
    assert target == pytest.approx([0.012, 0.02])
    assert walk == pytest.approx([-0.02, 0.004])
    # Only the first scene keeps a two-sided 10 mm contrast at every offset.
    critical = (target >= 0.01) & (walk <= -0.01)
    assert critical.tolist() == [True, False]
