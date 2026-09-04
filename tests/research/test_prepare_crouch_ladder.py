from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/research/hallucination/prepare_crouch_ladder.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_crouch_ladder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_can_restrict_materialization_to_q3_motion() -> None:
    module = _module()
    screen = {
        "clips": [
            {
                "motion_index": 0,
                "body_mode": "walk",
                "usable_rungs": 2,
                "deepest_usable_mm": 55.0,
                "already_sourced": False,
            },
            {
                "motion_index": 6,
                "body_mode": "arm_tuck",
                "usable_rungs": 4,
                "deepest_usable_mm": 85.0,
                "already_sourced": False,
            },
        ]
    }

    chosen = module.select(
        screen,
        cohort=4,
        exclude_sourced=True,
        motion_indices={0},
    )

    assert [clip["motion_index"] for clip in chosen] == [0]
