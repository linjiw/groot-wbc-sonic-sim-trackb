from pathlib import Path

from gear_sonic.scripts.launch_data_collection import (
    DataCollectionLaunchConfig,
    _build_exporter_cmd,
)


def test_exporter_command_passes_root_output_dir_for_raw_dataset_collection():
    config = DataCollectionLaunchConfig(
        task_prompt="pick up the red cup and place it on the tray",
        dataset_name="g1_fetch_place_tiny_raw",
        root_output_dir="/data",
        camera_host="192.168.123.164",
        camera_port=5555,
        data_exporter_frequency=50,
        record_wrist_cameras=False,
        text_to_speech=False,
    )

    cmd = _build_exporter_cmd(config, Path("/repo"))

    assert "--root-output-dir /data" in cmd
    assert "--dataset-name 'g1_fetch_place_tiny_raw'" in cmd
    assert "--task-prompt 'pick up the red cup and place it on the tray'" in cmd
    assert "--camera-host 192.168.123.164" in cmd
    assert "--no-text-to-speech" in cmd
