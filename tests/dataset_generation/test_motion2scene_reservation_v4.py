"""Check optimized freshness semantics against the retained original implementation."""

import importlib
import json
from pathlib import Path


def test_iterative_scan_preserves_integer_and_implied_range_semantics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    fast = importlib.import_module("motion2scene_reserve_transfer_sources_v4")
    old = importlib.import_module("motion2scene_fresh_sources").seed_mentions
    seeds = [9480001, 9480002]
    for base in (9477999, 9479001, 9480001, 9480003):
        for count in (-1, 0, 1, 3, 1001, 4000):
            for prompts in (-1, 0, 1, 3):
                for design in ("independent", "shared_across_prompts"):
                    value = {
                        "nested": [
                            {
                                "seed_base": base,
                                "seeds_per_prompt": count,
                                "prompts": prompts,
                                "seed_design": design,
                            },
                            [True, 1.2],
                        ]
                    }
                    assert fast.mentions(value, seeds) == old(value, seeds)
                    assert fast.scan(json.dumps(value).encode(), seeds) == old(value, seeds)
    for raw in (
        b'{"seed_\\u0062ase":9480000,"seeds_per_prompt":3}',
        '{"seed_base":9480000,"seeds_per_prompt":3}'.encode("utf-16"),
        '{"value":9480001}'.encode("utf-32"),
        b'{"value":"9480001", "number":94800010}',
        b'{"text":"line\\nline", "value":[1,2,3]}',
    ):
        assert fast.scan(raw, seeds) == old(json.loads(raw), seeds)
