from __future__ import annotations

import json
from pathlib import Path

from scripts.research.summarize_sonic_logs import (
    parse_eval_log,
    parse_training_log,
    summarize_logs,
    write_markdown,
)


def test_parse_training_log_extracts_final_iteration_metrics() -> None:
    text = """
│                          Learning iteration 99                         │
│                       Mean rewards: 0.85612                                  │
│                        Mean length: 10.43000                                 │
│ Env/Metrics/motion/error_anchor_pos: 0.1102                                  │
│  Env/Metrics/motion/error_body_pos: 0.0893                                   │
│                    Total timesteps: 38016                                    │
│                         Total time: 575.28s                                  │
│                         Learning iteration 100                         │
│                       Mean rewards: 0.85156                                  │
│                        Mean length: 10.52000                                 │
│ Env/Metrics/motion/error_anchor_pos: 0.0912                                  │
│  Env/Metrics/motion/error_body_pos: 0.0898                                   │
│                     Total episodes: 1600                                     │
│                    Total timesteps: 38400                                    │
│                     Iteration time: 5.85s                                    │
│                         Total time: 581.13s                                  │
"""

    summary = parse_training_log(text)

    assert summary["ok"] is True
    assert summary["learning_iteration"] == 100
    assert summary["mean_rewards"] == 0.85156
    assert summary["mean_length"] == 10.52
    assert summary["error_anchor_pos"] == 0.0912
    assert summary["error_body_pos"] == 0.0898
    assert summary["total_episodes"] == 1600
    assert summary["total_timesteps"] == 38400
    assert summary["iteration_time_s"] == 5.85
    assert summary["total_time_s"] == 581.13
    assert summary["traceback_count"] == 0


def test_parse_eval_log_extracts_metric_groups_and_error_state() -> None:
    text = """
Terminated: 1 | max frames: 2002 | steps 2001 | env_loop: 0 | eval_time: 3.9m | Start: 0 | Succ rate: 0.000 | Mpjpe: nan
All:  mpjpe_g: 130.802 \tmpjpe_l: 18.728 \tmpjpe_pa: 11.824 \taccel_dist: 0.983 \tvel_dist: 2.634
Succ:  mpjpe_g: 199.754 \tmpjpe_l: 20.975 \tmpjpe_pa: 12.139 \taccel_dist: 0.868 \tvel_dist: 2.555
"""

    summary = parse_eval_log(text)

    assert summary["ok"] is True
    assert summary["terminated_final"] == 1
    assert summary["success_rate_final"] == 0.0
    assert summary["all"]["mpjpe_g"] == 130.802
    assert summary["all"]["mpjpe_l"] == 18.728
    assert summary["all"]["vel_dist"] == 2.634
    assert summary["succ"]["mpjpe_pa"] == 12.139
    assert summary["traceback_count"] == 0


def test_summarize_logs_writes_json_ready_schema(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    eval_log = tmp_path / "eval.log"
    train_log.write_text("Learning iteration 1\nMean rewards: 0.5\nTotal timesteps: 384\n", encoding="utf-8")
    eval_log.write_text("All:  mpjpe_g: 1.0 \tmpjpe_l: 2.0\n", encoding="utf-8")

    summary = summarize_logs(train_log=train_log, eval_log=eval_log)

    assert summary["train"]["log_path"] == str(train_log)
    assert summary["eval"]["log_path"] == str(eval_log)
    json.dumps(summary)


def test_write_markdown_includes_key_tables(tmp_path: Path) -> None:
    output = tmp_path / "summary.md"
    write_markdown(
        output,
        {
            "train": {"ok": True, "learning_iteration": 100, "mean_rewards": 0.85156},
            "eval": {"ok": True, "all": {"mpjpe_g": 130.802, "mpjpe_l": 18.728}},
        },
    )

    text = output.read_text(encoding="utf-8")
    assert "# SONIC Log Summary" in text
    assert "learning_iteration" in text
    assert "mpjpe_g" in text
