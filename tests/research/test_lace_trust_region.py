from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gear_sonic.research.lace.trust_region import TrustRegionConstraints, apply_lace_tilt


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    base = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    membership = np.array(
        [
            [1.0, 0.0],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return base, membership


def test_disabled_lace_is_bitwise_identical_to_native_distribution() -> None:
    base, membership = _fixture()
    result = apply_lace_tilt(base, membership, [100.0, -100.0], enabled=False)

    assert np.array_equal(result.probabilities, base)
    assert result.applied_scale == 0.0
    assert not result.enabled


def test_default_research_and_release_configs_keep_lace_disabled() -> None:
    root = Path(__file__).resolve().parents[2]
    research = yaml.safe_load((root / "configs/research/lace/curriculum.yaml").read_text())
    motion = yaml.safe_load(
        (root / "gear_sonic/config/manager_env/commands/terms/motion.yaml").read_text()
    )

    assert research["enable"] is False
    assert motion["motion"]["motion_lib_cfg"]["adaptive_sampling"]["lace"]["enable"] is False


def test_projection_enforces_kl_ratio_and_normalization() -> None:
    base, membership = _fixture()
    constraints = TrustRegionConstraints(max_kl=0.005, max_probability_ratio=1.15)
    result = apply_lace_tilt(
        base,
        membership,
        [10.0, -10.0],
        strength=2.0,
        constraints=constraints,
    )

    assert result.projected
    assert result.applied_scale < 2.0
    assert result.probabilities.sum() == pytest.approx(1.0)
    assert result.diagnostics.legal
    assert result.diagnostics.kl_from_base <= constraints.max_kl + 1e-12
    assert result.diagnostics.max_probability_ratio <= 1.15 + 1e-12


def test_soft_memberships_produce_expected_mechanism_exposure() -> None:
    base, membership = _fixture()
    result = apply_lace_tilt(base, membership, [0.0, 0.0])

    assert result.diagnostics.mechanism_exposure == pytest.approx((0.5, 0.5))


def test_gate_rejects_an_illegal_native_baseline() -> None:
    base, membership = _fixture()
    constraints = TrustRegionConstraints(min_mechanism_exposure=(0.6, 0.6))

    with pytest.raises(ValueError, match="native baseline"):
        apply_lace_tilt(base, membership, [1.0, 0.0], constraints=constraints)
