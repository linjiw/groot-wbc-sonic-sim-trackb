from __future__ import annotations

import pytest

from scripts.research.generate_kimodo_motions import generation_seed


def test_legacy_seed_design_keeps_prompt_spacing() -> None:
    assert generation_seed(1000, 0, 0, "independent_per_prompt") == 1000
    assert generation_seed(1000, 3, 2, "independent_per_prompt") == 4002


def test_shared_seed_design_pairs_every_prompt_at_one_replicate() -> None:
    seeds = {
        generation_seed(41001, prompt_index, 4, "shared_across_prompts")
        for prompt_index in range(18)
    }
    assert seeds == {41005}


def test_unknown_seed_design_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown seed design"):
        generation_seed(1, 0, 0, "interleaved")
