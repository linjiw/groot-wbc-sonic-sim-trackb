import datetime
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "today,older,admitted", [(6.0, 0.0, False), (1.0, 21.0, False), (4.0, 5.0, True)]
)
def test_layout_block_budget_counts_all_chunk_records(
    tmp_path, monkeypatch, today, older, admitted
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    module = importlib.import_module("motion2scene_independent_layouts")
    monkeypatch.setattr(module, "DATA", tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    for name, hours, days in [("m2s-layout-b00", today, 0), ("m2s-prior", older, 2)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "run_record.json").write_text(
            json.dumps(
                {
                    "started_at": (now - datetime.timedelta(days=days)).isoformat(),
                    "budget": {"actual_contended_gpu_hours": hours},
                }
            )
        )
    launch = tmp_path / "m2s-layout-b01"
    launch.mkdir()
    assert module.budget_gate(launch, remaining=20) is admitted
    receipt = json.loads((launch / "launch_gate_000.json").read_text())
    assert receipt["daily_hours"] == today
    assert receipt["weekly_hours"] == today + older
    assert receipt["reserved_hours"] == pytest.approx(20 * 375 / 3600)
