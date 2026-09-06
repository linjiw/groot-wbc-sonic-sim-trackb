"""Negative byte filtering preserves the original integer/range freshness test."""

import importlib.util
import json
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "scripts/research"


def test_prefilter_matches_original_on_nested_metadata_and_escaped_keys(monkeypatch):
    monkeypatch.syspath_prepend(str(RESEARCH))
    spec = importlib.util.spec_from_file_location(
        "reservation_v3", RESEARCH / "motion2scene_reserve_transfer_sources_v3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [9470001, 9470002]
    cases = [
        b'{"values":[1,2,3]}',
        b'{"value":9470001}',
        b'{"value":94700010}',
        b'{"seed_base":9470000,"seeds_per_prompt":3}',
        b'{"seed_\\u0062ase":9470000,"seeds_per_prompt":3}',
        b'{"seed_base":9469001,"seeds_per_prompt":1,"prompts":2}',
        b'{"seed_base":9469001,"seeds_per_prompt":1,"prompts":2,"seed_design":"shared_across_prompts"}',
    ]
    for offset in range(-20, 20):
        cases.append(
            json.dumps(
                {"nested": [{"seed_base": 9470001 + offset, "seeds_per_prompt": 4}]}
            ).encode()
        )
    for raw in cases:
        assert module.scan(raw, ids) == module.seed_mentions(json.loads(raw), ids)
    with pytest.raises(ValueError):
        module.scan(b'{"value":9470001,}', ids)
