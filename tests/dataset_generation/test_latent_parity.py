"""Offline gates for SONIC 64D latent parity with the C++ deployment path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.latent_parity import (
    DEPLOY_RELEASE_OBSERVATION_CONFIG,
    SONIC_RELEASE_TOKEN_CONTRACT,
    LatentParityReport,
    TokenContract,
    check_latent_parity,
    deploy_observation_enabled,
    describe_contract,
    flatten_tokens,
    fsq_level_code_values,
    latent_parity_provenance,
    read_deploy_encoder_dimension,
    summarize_payload_latent,
    unflatten_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _on_lattice(frames: int, rng: np.random.Generator) -> np.ndarray:
    codes = fsq_level_code_values(32)
    return rng.choice(codes, size=(frames, 64)).astype(np.float32)


# --------------------------------------------------------------------------
# FSQ lattice derivation
# --------------------------------------------------------------------------


def test_even_level_lattice_is_biased_one_step_negative():
    values = fsq_level_code_values(32)
    assert values.size == 32
    assert values[0] == pytest.approx(-1.0)
    assert values[-1] == pytest.approx(15.0 / 16.0)
    assert np.allclose(np.diff(values), 1.0 / 16.0)


def test_odd_level_lattice_is_symmetric():
    values = fsq_level_code_values(5)
    assert values.size == 5
    assert values[0] == pytest.approx(-1.0)
    assert values[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("level", [2, 3, 4, 5, 8, 16, 17, 32])
def test_lattice_size_always_equals_level(level):
    assert fsq_level_code_values(level).size == level


@pytest.mark.parametrize("level", [0, 1, -3, True, 2.5])
def test_invalid_levels_are_rejected(level):
    with pytest.raises(ValueError):
        fsq_level_code_values(level)


def test_lattice_matches_the_installed_fsq_implementation():
    """Cross-check the closed form against ``vector_quantize_pytorch.FSQ`` itself."""
    torch = pytest.importorskip("torch")
    fsq_module = pytest.importorskip("vector_quantize_pytorch")

    quantizer = fsq_module.FSQ(levels=[32] * 32)
    generator = torch.Generator().manual_seed(0)
    # Wide inputs so the tanh bound saturates at both extremes.
    latent = torch.randn(4096, 2, 32, generator=generator) * 8.0
    with torch.no_grad():
        codes, _ = quantizer(latent)

    observed = np.unique(codes.numpy().astype(np.float64))
    expected = fsq_level_code_values(32)
    assert np.allclose(observed, expected[np.isin(expected, observed)])
    assert observed.min() >= expected.min() - 1e-9
    assert observed.max() <= expected.max() + 1e-9

    report = check_latent_parity(codes.reshape(codes.shape[0], -1).numpy())
    assert report.ok
    assert report.verdict == LatentParityReport.ENCODER_EQUIVALENT
    assert report.max_lattice_deviation == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Token contract and flatten order
# --------------------------------------------------------------------------


def test_release_contract_matches_all_mlp_v1_config():
    """The contract must track ``actor_critic/universal_token/all_mlp_v1.yaml``."""
    config = (
        REPO_ROOT / "gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "num_fsq_levels: 32" in config
    assert "fsq_level_list: 32" in config
    assert "max_num_tokens: 2" in config

    assert SONIC_RELEASE_TOKEN_CONTRACT.token_dim == 32
    assert SONIC_RELEASE_TOKEN_CONTRACT.max_num_tokens == 2
    assert SONIC_RELEASE_TOKEN_CONTRACT.total_dim == 64


def test_release_contract_matches_the_cpp_deploy_encoder_dimension():
    """Binds the Python dataset contract to the C++ runtime observation config."""
    config_path = REPO_ROOT / DEPLOY_RELEASE_OBSERVATION_CONFIG
    assert config_path.is_file()
    assert read_deploy_encoder_dimension(config_path) == SONIC_RELEASE_TOKEN_CONTRACT.total_dim
    assert deploy_observation_enabled(config_path, "token_state") is True


def test_deploy_encoder_dimension_rejects_a_config_without_one(tmp_path):
    path = tmp_path / "observation_config.yaml"
    path.write_text('observations:\n  - name: "token_state"\n    enabled: true\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no encoder.dimension"):
        read_deploy_encoder_dimension(path)


def test_deploy_encoder_dimension_ignores_a_dimension_outside_the_encoder_block(tmp_path):
    path = tmp_path / "observation_config.yaml"
    path.write_text(
        "planner:\n  dimension: 7\nencoder:\n  dimension: 64\n",
        encoding="utf-8",
    )
    assert read_deploy_encoder_dimension(path) == 64


def test_deploy_observation_enabled_reads_the_named_entry(tmp_path):
    path = tmp_path / "observation_config.yaml"
    path.write_text(
        "observations:\n"
        '  - name: "token_state"\n'
        "    enabled: false\n"
        '  - name: "projected_gravity"\n'
        "    enabled: true\n"
        "encoder:\n  dimension: 64\n",
        encoding="utf-8",
    )
    assert deploy_observation_enabled(path, "token_state") is False
    assert deploy_observation_enabled(path, "projected_gravity") is True
    assert deploy_observation_enabled(path, "absent") is False


def test_flatten_order_is_row_major_over_the_trailing_two_axes():
    tokens = np.arange(2 * 2 * 32, dtype=np.float64).reshape(2, 2, 32)
    flat = flatten_tokens(tokens, SONIC_RELEASE_TOKEN_CONTRACT)
    assert flat.shape == (2, 64)
    # Token 0 occupies [0:32] and token 1 occupies [32:64], matching both
    # ``view(*shape[:-2], -1)`` and ``flatten(start_dim=-2)``.
    assert np.array_equal(flat[0, :32], tokens[0, 0])
    assert np.array_equal(flat[0, 32:], tokens[0, 1])
    assert np.array_equal(unflatten_tokens(flat, SONIC_RELEASE_TOKEN_CONTRACT), tokens)


def test_flatten_order_matches_both_torch_call_sites():
    torch = pytest.importorskip("torch")
    tokens = torch.arange(3 * 1 * 2 * 32, dtype=torch.float64).reshape(3, 1, 2, 32)
    python_wrapper = tokens.view(*tokens.shape[:-2], -1)  # universal_token_modules.py
    onnx_export = tokens.flatten(start_dim=-2)  # inference_helpers.py
    assert torch.equal(python_wrapper, onnx_export)
    numpy_equivalent = flatten_tokens(tokens.numpy(), SONIC_RELEASE_TOKEN_CONTRACT)
    assert np.array_equal(numpy_equivalent, python_wrapper.numpy())


def test_flatten_rejects_a_mismatched_trailing_shape():
    with pytest.raises(ValueError):
        flatten_tokens(np.zeros((4, 3, 32)), SONIC_RELEASE_TOKEN_CONTRACT)
    with pytest.raises(ValueError):
        unflatten_tokens(np.zeros((4, 63)), SONIC_RELEASE_TOKEN_CONTRACT)


def test_non_uniform_contract_maps_levels_per_column():
    contract = TokenContract(max_num_tokens=2, fsq_level_list=(4, 5))
    assert contract.total_dim == 4
    assert contract.flat_levels().tolist() == [4, 5, 4, 5]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_num_tokens": 0, "fsq_level_list": (32,)},
        {"max_num_tokens": 2, "fsq_level_list": ()},
        {"max_num_tokens": 2, "fsq_level_list": (1,)},
    ],
)
def test_invalid_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TokenContract(**kwargs)


# --------------------------------------------------------------------------
# Representation classification
# --------------------------------------------------------------------------


def test_pure_encoder_tokens_are_deployment_equivalent():
    rng = np.random.default_rng(0)
    report = check_latent_parity(_on_lattice(64, rng))
    assert report.ok
    assert report.deployment_encoder_equivalent
    assert report.on_lattice_fraction == pytest.approx(1.0)
    assert report.residual_rms == pytest.approx(0.0, abs=1e-9)
    assert report.frame_count == 64
    assert report.out_of_encoder_range_count == 0


def test_residual_perturbed_tokens_are_classified_not_silently_accepted():
    rng = np.random.default_rng(1)
    tokens = _on_lattice(32, rng).astype(np.float64)
    tokens += rng.normal(scale=0.01, size=tokens.shape)
    tokens = np.clip(tokens, -1.0, 15.0 / 16.0)

    report = check_latent_parity(tokens)
    assert report.ok  # not an error by default
    assert report.verdict == LatentParityReport.RESIDUAL_PERTURBED
    assert not report.deployment_encoder_equivalent
    assert report.on_lattice_fraction < 0.5
    assert report.residual_rms > 0.0


def test_residual_perturbed_tokens_fail_when_encoder_equivalence_is_required():
    rng = np.random.default_rng(2)
    tokens = np.clip(
        _on_lattice(8, rng).astype(np.float64) + rng.normal(scale=0.02, size=(8, 64)),
        -1.0,
        15.0 / 16.0,
    )
    report = check_latent_parity(tokens, require_encoder_equivalent=True)
    assert not report.ok
    assert any("encoder-equivalent" in error for error in report.errors)


def test_values_beyond_the_fsq_range_are_counted_but_still_read_as_a_residual():
    """An unbounded post-quantization residual routinely exceeds the code range."""
    rng = np.random.default_rng(3)
    tokens = _on_lattice(4, rng).astype(np.float64)
    tokens[0, 0] = 3.5
    report = check_latent_parity(tokens)
    assert report.verdict == LatentParityReport.RESIDUAL_PERTURBED
    assert report.out_of_encoder_range_count == 1
    assert any("never produce" in note for note in report.notes)


def test_a_single_frame_is_accepted():
    rng = np.random.default_rng(4)
    report = check_latent_parity(_on_lattice(1, rng)[0])
    assert report.ok
    assert report.frame_count == 1


def test_wrong_width_is_an_error():
    report = check_latent_parity(np.zeros((4, 63), dtype=np.float32))
    assert not report.ok
    assert any("token_state size 64" in error for error in report.errors)


def test_non_finite_values_are_an_error():
    rng = np.random.default_rng(5)
    tokens = _on_lattice(4, rng).astype(np.float64)
    tokens[2, 5] = np.nan
    report = check_latent_parity(tokens)
    assert not report.ok
    assert any("NaN" in error for error in report.errors)


def test_higher_rank_input_is_rejected():
    report = check_latent_parity(np.zeros((2, 3, 64)))
    assert not report.ok
    assert report.verdict == LatentParityReport.OUT_OF_CONTRACT


def test_float32_round_tripping_stays_on_lattice():
    """The default tolerance must absorb float32 storage, not a real residual."""
    rng = np.random.default_rng(6)
    tokens = _on_lattice(128, rng)
    assert tokens.dtype == np.float32
    report = check_latent_parity(tokens)
    assert report.deployment_encoder_equivalent


def test_smallest_representable_residual_is_still_detected():
    """One FSQ step is 1/16; a residual an order of magnitude smaller must show."""
    rng = np.random.default_rng(7)
    tokens = _on_lattice(16, rng).astype(np.float64)
    # Perturb a strictly interior code so the probe cannot leave the code range.
    tokens[:, 0] = 0.0
    tokens[:, 0] += 1.0 / 160.0
    report = check_latent_parity(tokens)
    assert report.verdict == LatentParityReport.RESIDUAL_PERTURBED
    assert report.out_of_encoder_range_count == 0
    assert report.max_lattice_deviation == pytest.approx(1.0 / 160.0)


# --------------------------------------------------------------------------
# Payload and provenance helpers
# --------------------------------------------------------------------------


def test_summarize_payload_reads_the_recorder_field():
    rng = np.random.default_rng(8)
    payload = {"action_motion_token": _on_lattice(10, rng)}
    report = summarize_payload_latent(payload)
    assert report.deployment_encoder_equivalent


def test_summarize_payload_reports_a_missing_field():
    report = summarize_payload_latent({})
    assert not report.ok
    assert any("action_motion_token" in error for error in report.errors)


def test_provenance_block_records_both_source_call_sites():
    rng = np.random.default_rng(9)
    report = check_latent_parity(_on_lattice(4, rng))
    provenance = latent_parity_provenance(report)
    assert provenance["total_dim"] == 64
    assert provenance["representation"] == "post_quantization_fsq_codes"
    assert "_last_full_latent_flat" in provenance["python_source"]
    assert "encoded_tokens" in provenance["cpp_source"]
    assert provenance["parity_report"]["verdict"] == LatentParityReport.ENCODER_EQUIVALENT


def test_describe_contract_is_json_safe():
    import json

    assert json.loads(json.dumps(describe_contract()))["fsq_levels"] == [32]
