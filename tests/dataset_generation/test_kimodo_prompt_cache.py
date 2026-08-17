"""Tests for the precomputed prompt-embedding cache."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.kimodo_prompt_cache import (
    CachedTextEncoder,
    PromptCacheError,
    PromptEmbedding,
    cached_text_encoder_from_path,
    load_prompt_cache,
    prompt_key,
    save_prompt_cache,
)

LLM_DIM = 8


def embedding(prompt: str, fill: float) -> PromptEmbedding:
    # LLM2Vec pools each prompt to one vector, so the block is (1, dim) and length is 1.
    return PromptEmbedding(
        prompt=prompt, features=np.full((1, LLM_DIM), fill, dtype=np.float32), length=1
    )


@pytest.fixture()
def cache_file(tmp_path):
    path = tmp_path / "cache.npz"
    save_prompt_cache(path, [embedding("walk forward", 0.25), embedding("turn left", -0.5)])
    return path


def test_round_trip_preserves_prompts_and_features(cache_file):
    cache = load_prompt_cache(cache_file)
    assert set(cache) == {"walk forward", "turn left"}
    assert cache["walk forward"].features == pytest.approx(np.full((1, LLM_DIM), 0.25))
    assert cache["turn left"].length == 1


def test_prompt_key_is_whitespace_sensitive():
    """Two prompts differing only in spacing are different conditioning, not the same."""
    assert prompt_key("walk forward") != prompt_key("walk  forward")
    assert prompt_key("walk forward") == prompt_key("walk forward")


def test_refuses_to_write_an_empty_cache(tmp_path):
    with pytest.raises(PromptCacheError, match="empty prompt cache"):
        save_prompt_cache(tmp_path / "empty.npz", [])


def test_rejects_mixed_embedding_dimensions(tmp_path):
    mixed = [
        embedding("a", 0.0),
        PromptEmbedding("b", np.zeros((1, LLM_DIM + 1), dtype=np.float32), 1),
    ]
    with pytest.raises(PromptCacheError, match="inconsistent embedding dimensions"):
        save_prompt_cache(tmp_path / "mixed.npz", mixed)


def test_rejects_length_outside_the_feature_block():
    with pytest.raises(PromptCacheError, match="outside"):
        PromptEmbedding("a", np.zeros((1, LLM_DIM), dtype=np.float32), 5)


def test_list_input_returns_batched_features_and_a_list_of_lengths(cache_file):
    encoder = CachedTextEncoder(load_prompt_cache(cache_file))
    features, lengths = encoder(["walk forward", "turn left"])
    assert features.shape == (2, 1, LLM_DIM)
    assert lengths == [1, 1]


def test_string_input_is_unbatched_matching_the_real_encoder(cache_file):
    """LLM2VecEncoder returns ``features[0]`` and an ``int`` for a bare string.

    Batching a string input instead would silently change the model's sample shape.
    """
    encoder = CachedTextEncoder(load_prompt_cache(cache_file))
    features, length = encoder("walk forward")
    assert features.shape == (1, LLM_DIM)
    assert isinstance(length, int) and length == 1


def test_missing_prompt_raises_rather_than_substituting(cache_file):
    encoder = CachedTextEncoder(load_prompt_cache(cache_file))
    with pytest.raises(PromptCacheError, match="not in the cache"):
        encoder(["walk forward", "do a backflip"])


def test_to_sets_device_and_dtype_and_chains(cache_file):
    encoder = CachedTextEncoder(load_prompt_cache(cache_file))
    assert encoder.to(dtype=torch.float16) is encoder
    features, _ = encoder(["walk forward"])
    assert features.dtype == torch.float16


def test_factory_resolves_a_string_dtype_for_the_preset_registry(cache_file):
    # Kimodo's preset table carries plain JSON-ish kwargs, so dtype arrives as a string.
    encoder = cached_text_encoder_from_path(str(cache_file), device="cpu", dtype="float32")
    features, _ = encoder(["turn left"])
    assert features.dtype == torch.float32
    assert features == pytest.approx(np.full((1, 1, LLM_DIM), -0.5))
