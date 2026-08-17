"""Precomputed prompt embeddings, so motion generation does not need a 15 GB encoder.

Kimodo conditions on LLM2Vec (Llama-3-8B) embeddings of the prompt. In bfloat16 that
encoder occupies about 15 GB, while the motion diffusion model itself needs roughly 2 GB.
On a shared single GPU the encoder is what fails to fit -- and it is also pure waste to
reload it for every generation, because a prompt's embedding never changes.

So the two stages are split. Encoding runs once over the whole prompt taxonomy and writes
a cache; generation loads that cache instead of the encoder. Beyond fitting in memory,
this is what makes a 100-200 prompt sweep practical: one encoder load for the corpus
rather than one per motion, and the encode stage can run on CPU while the GPU is busy
generating.

The cache is keyed by the SHA-256 of the exact prompt string. A prompt that was never
encoded raises rather than silently falling back to some other embedding -- a generation
conditioned on the wrong text would be indistinguishable from a bad sample downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
CACHE_SCHEMA_VERSION = 1


class PromptCacheError(KeyError):
    """Raised when a prompt is missing from the cache or the cache is malformed."""


def prompt_key(prompt: str) -> str:
    """Stable per-prompt key. Exact string match -- whitespace is significant."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PromptEmbedding:
    """One encoded prompt: a ``(length, llm_dim)`` feature block and its true length."""

    prompt: str
    features: np.ndarray
    length: int

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise PromptCacheError(
                f"prompt features must be 2-D (tokens, dim), got {self.features.shape}"
            )
        if not 0 < self.length <= self.features.shape[0]:
            raise PromptCacheError(
                f"length {self.length} outside the {self.features.shape[0]}-token feature block"
            )


def save_prompt_cache(path: str | Path, embeddings: Sequence[PromptEmbedding]) -> Path:
    """Write embeddings to a single ``.npz``.

    One file rather than one per prompt: the sweep reads all of them at startup, and a
    few hundred small files is a worse thing to move between machines than one blob.
    """
    if not embeddings:
        raise PromptCacheError("refusing to write an empty prompt cache")
    dims = {embedding.features.shape[1] for embedding in embeddings}
    if len(dims) != 1:
        raise PromptCacheError(f"inconsistent embedding dimensions in one cache: {sorted(dims)}")

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "prompts": np.asarray([e.prompt for e in embeddings], dtype=object),
        "keys": np.asarray([prompt_key(e.prompt) for e in embeddings]),
        "lengths": np.asarray([e.length for e in embeddings], dtype=np.int64),
    }
    for embedding in embeddings:
        payload[f"features/{prompt_key(embedding.prompt)}"] = embedding.features
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def load_prompt_cache(path: str | Path) -> dict[str, PromptEmbedding]:
    """Read a cache back, keyed by prompt string."""
    with np.load(Path(path), allow_pickle=True) as handle:
        version = int(handle["schema_version"])
        if version != CACHE_SCHEMA_VERSION:
            raise PromptCacheError(
                f"prompt cache schema {version} != supported {CACHE_SCHEMA_VERSION}"
            )
        prompts = [str(p) for p in handle["prompts"]]
        lengths = [int(v) for v in handle["lengths"]]
        cache: dict[str, PromptEmbedding] = {}
        for prompt, length in zip(prompts, lengths):
            features = np.asarray(handle[f"features/{prompt_key(prompt)}"])
            cache[prompt] = PromptEmbedding(prompt=prompt, features=features, length=length)
    return cache


class CachedTextEncoder:
    """Drop-in replacement for Kimodo's text encoder, backed by precomputed embeddings.

    Reproduces ``LLM2VecEncoder.__call__`` exactly, including two details that are easy to
    miss and expensive to get wrong: LLM2Vec pools each prompt to a single vector, so the
    feature block is ``(1, llm_dim)`` and the length is always 1; and a bare ``str`` input
    returns an unbatched ``(1, llm_dim)`` tensor with an ``int`` length, while a list input
    returns ``(B, 1, llm_dim)`` with a list of lengths. Silently batching a string input
    would change the model's sample shape rather than raise.
    """

    def __init__(self, cache: dict[str, PromptEmbedding], device: str = "cpu", dtype=None):
        import torch

        self._cache = cache
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

    def to(self, device=None, dtype=None) -> "CachedTextEncoder":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def _lookup(self, texts: Sequence[str]) -> list[PromptEmbedding]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            raise PromptCacheError(
                f"{len(missing)} prompt(s) not in the cache; encode them first. "
                f"First missing: {missing[0]!r}"
            )
        return [self._cache[t] for t in texts]

    def __call__(self, texts):
        import torch

        is_string = isinstance(texts, str)
        entries = self._lookup([texts] if is_string else list(texts))

        width = max(e.features.shape[0] for e in entries)
        dim = entries[0].features.shape[1]
        batch = np.zeros((len(entries), width, dim), dtype=np.float32)
        for index, entry in enumerate(entries):
            batch[index, : entry.features.shape[0]] = entry.features
        features = torch.as_tensor(batch, dtype=self.dtype, device=self.device)

        if is_string:
            return features[0], entries[0].length
        return features, [e.length for e in entries]


def cached_text_encoder_from_path(
    cache_path: str, device: str = "cpu", dtype: str | None = None
) -> CachedTextEncoder:
    """Factory for Kimodo's ``TEXT_ENCODER_PRESETS`` registry.

    Kimodo picks its text encoder by name from a preset table, so registering a preset
    that targets this function is enough to swap in the cache -- no patching of the model
    or of ``load_model`` is needed, and the substitution is visible in configuration
    rather than hidden in an import side effect.
    """
    import torch

    resolved = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return CachedTextEncoder(load_prompt_cache(cache_path), device=device, dtype=resolved)
