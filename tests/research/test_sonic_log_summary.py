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


def test_parse_eval_log_prefers_final_success_rate_over_tqdm_status() -> None:
    text = """
Terminated: 0 | max frames: 200 | steps 199 | Succ rate: 0.000 | Mpjpe: nan
Success Rate: 1.0000000000
Progress Rate: 1.0000000000
All:  mpjpe_g: 16.493 \tmpjpe_l: 13.255 \tmpjpe_pa: 9.168
Succ:  mpjpe_g: 16.493 \tmpjpe_l: 13.255 \tmpjpe_pa: 9.168
"""

    summary = parse_eval_log(text)

    assert summary["ok"] is True
    assert summary["terminated_final"] == 0
    assert summary["success_rate_final"] == 1.0
    assert summary["progress_rate_final"] == 1.0
    assert summary["all"]["mpjpe_g"] == 16.493


def test_parse_training_log_extracts_final_adp_samp_telemetry() -> None:
    text = """
 \033[1m Learning iteration 49  \033[0m
                      Mean rewards: 0.93542
  Env/adp_samp/num_episodes_mean: 1.9000
  Env/adp_samp/failure_rate_mean: 0.9000
Env/adp_samp/prob_max_over_uniform: 2.9000
 \033[1m Learning iteration 50  \033[0m
                      Mean rewards: 0.93600
  Env/adp_samp/num_episodes_mean: 2.1000
  Env/adp_samp/failure_rate_mean: 0.8500
Env/adp_samp/prob_max_over_uniform: 3.0500
Env/adp_samp/num_concentrated_bins: 0.0000
Env/adp_samp/effective_num_bins: 69.7000
Env/adp_samp/episodes_max_over_mean: 1.8000
"""

    summary = parse_training_log(text)

    assert summary["adp_samp_num_episodes_mean"] == 2.1
    assert summary["adp_samp_failure_rate_mean"] == 0.85
    assert summary["adp_samp_prob_max_over_uniform"] == 3.05
    assert summary["adp_samp_num_concentrated_bins"] == 0.0
    assert summary["adp_samp_effective_num_bins"] == 69.7
    assert summary["adp_samp_episodes_max_over_mean"] == 1.8


def test_parse_training_log_reports_nonfinite_final_adp_samp_value_as_absent() -> None:
    # %.4f prints nan for non-finite tensors; the final value must not silently
    # fall back to an earlier finite iteration.
    text = (
        "Learning iteration 49\nEnv/adp_samp/failure_rate_mean: 0.9000\n"
        "Learning iteration 50\nEnv/adp_samp/failure_rate_mean: nan\n"
    )

    summary = parse_training_log(text)

    assert "adp_samp_failure_rate_mean" not in summary


def test_parse_training_log_omits_adp_samp_keys_for_uniform_logs() -> None:
    text = "Learning iteration 50\nMean rewards: 0.98755\nTotal timesteps: 9600\n"

    summary = parse_training_log(text)

    assert not any(key.startswith("adp_samp_") for key in summary)


def test_summarize_logs_writes_json_ready_schema(tmp_path: Path) -> None:
    train_log = tmp_path / "train.log"
    eval_log = tmp_path / "eval.log"
    train_log.write_text(
        "Learning iteration 1\nMean rewards: 0.5\nTotal timesteps: 384\n", encoding="utf-8"
    )
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
