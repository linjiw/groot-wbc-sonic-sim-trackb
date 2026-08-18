from __future__ import annotations

import pytest

from gear_sonic.research.lace.signatures import build_factorized_signatures

MECHANISMS = ("contact", "slip", "drift")


def _episode(*, failed: bool, scores: tuple[float, float, float]) -> dict[str, object]:
    return {
        "motion_key": "jump__A001",
        "probe_policy_id": "lite_early",
        "failed": failed,
        "mechanism_scores": dict(zip(MECHANISMS, scores, strict=True)),
    }


def test_signature_separates_difficulty_from_conditional_mechanism() -> None:
    signatures = build_factorized_signatures(
        [
            _episode(failed=True, scores=(3.0, 1.0, 0.0)),
            _episode(failed=True, scores=(0.0, 2.0, 0.0)),
            _episode(failed=False, scores=(9.0, 9.0, 9.0)),
            _episode(failed=False, scores=(0.0, 0.0, 0.0)),
        ],
        mechanism_names=MECHANISMS,
        minimum_resolved_failures=1,
    )

    signature = signatures[0]
    assert signature["difficulty"] == pytest.approx(0.5)
    assert signature["q"] == pytest.approx([0.375, 0.625, 0.0])
    assert signature["f"] == pytest.approx([0.1875, 0.3125, 0.0])
    assert sum(signature["q"]) == pytest.approx(1.0)
    assert sum(signature["f"]) == pytest.approx(signature["difficulty"])
    assert sum(signature["q_posterior_mean"]) == pytest.approx(1.0)
    assert len(signature["q_credible_interval"]) == len(MECHANISMS)


def test_no_failure_has_missing_mechanism_instead_of_uniform_label() -> None:
    signature = build_factorized_signatures(
        [_episode(failed=False, scores=(0.0, 0.0, 0.0))],
        mechanism_names=MECHANISMS,
    )[0]

    assert signature["difficulty"] == 0.0
    assert signature["q"] is None
    assert signature["f"] is None


def test_unresolved_failure_affects_difficulty_but_not_mechanism_evidence() -> None:
    signature = build_factorized_signatures(
        [
            _episode(failed=True, scores=(0.0, 0.0, 0.0)),
            _episode(failed=True, scores=(0.0, 2.0, 0.0)),
        ],
        mechanism_names=MECHANISMS,
        minimum_resolved_failures=1,
    )[0]

    assert signature["difficulty"] == 1.0
    assert signature["attributed_failure_rate"] == pytest.approx(0.5)
    assert signature["unresolved_failure_probability"] == pytest.approx(0.5)
    assert signature["q"] == pytest.approx([0.0, 1.0, 0.0])
    assert signature["f"] == pytest.approx([0.0, 0.5, 0.0])
    assert signature["f_all_failure_mar_sensitivity"] == pytest.approx([0.0, 1.0, 0.0])
    assert signature["unresolved_failure_rate"] == pytest.approx(0.5)


def test_missing_or_nonfinite_probe_channel_fails_loudly() -> None:
    bad = _episode(failed=True, scores=(1.0, 0.0, 0.0))
    del bad["mechanism_scores"]["drift"]
    with pytest.raises(ValueError, match="channel mismatch"):
        build_factorized_signatures([bad], mechanism_names=MECHANISMS)

    bad = _episode(failed=True, scores=(1.0, float("nan"), 0.0))
    with pytest.raises(ValueError, match="invalid"):
        build_factorized_signatures([bad], mechanism_names=MECHANISMS)


def test_frozen_channel_scales_are_applied_before_episode_normalization() -> None:
    signature = build_factorized_signatures(
        [
            _episode(failed=True, scores=(3.0, 1.0, 0.0)),
            _episode(failed=True, scores=(0.0, 2.0, 0.0)),
        ],
        mechanism_names=MECHANISMS,
        mechanism_scales={"contact": 3.0, "slip": 1.0, "drift": 1.0},
        minimum_resolved_failures=1,
    )[0]

    assert signature["q"] == pytest.approx([0.25, 0.75, 0.0])
    assert signature["mechanism_scales"] == [3.0, 1.0, 1.0]
    assert signature["mechanism_raw_evidence"] == [3.0, 3.0, 0.0]


def test_channel_scale_contract_fails_loudly() -> None:
    episode = _episode(failed=True, scores=(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="exactly match"):
        build_factorized_signatures(
            [episode],
            mechanism_names=MECHANISMS,
            mechanism_scales={"contact": 1.0, "slip": 1.0},
        )
    with pytest.raises(ValueError, match="finite and positive"):
        build_factorized_signatures(
            [episode],
            mechanism_names=MECHANISMS,
            mechanism_scales={"contact": 1.0, "slip": 0.0, "drift": 1.0},
        )
