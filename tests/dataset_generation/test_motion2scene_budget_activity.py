import datetime
import importlib
import json
from pathlib import Path


def test_delayed_rollout_is_not_charged_only_to_preflight_date(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    module = importlib.import_module("motion2scene_layout_budget_review")
    now = datetime.datetime(2026, 9, 8, tzinfo=datetime.timezone.utc)
    folder = tmp_path / "m2s-test-b00"
    folder.mkdir()
    (folder / "run_record.json").write_text(
        json.dumps(
            {
                "started_at": (now - datetime.timedelta(days=2)).isoformat(),
                "budget": {"actual_contended_gpu_hours": 6.0},
                "cells": {"c0": {"started_at": now.isoformat(), "ended_at": now.isoformat()}},
            }
        )
    )
    result = module.review(tmp_path, remaining=20, now=now)
    assert result["daily_hours"] == result["weekly_hours"] == 6.0
    assert not result["admitted"]
