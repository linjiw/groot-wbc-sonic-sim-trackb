from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = REPO_ROOT / "gear_sonic/envs/manager_env/mdp/recorders.py"


class _FakeRecorderTerm:
    def __init__(self, _cfg: object, _env: object) -> None:
        pass


class _FakeRecorderTermCfg:
    pass


class _FakeRecorderManagerBaseCfg:
    pass


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


class _FakeCamera:
    def __init__(self) -> None:
        self._is_outdated = torch.zeros(1, dtype=torch.bool)
        self._view = SimpleNamespace(_sync_usd_on_fabric_write=False)
        self.data = SimpleNamespace(output={"rgb": torch.full((1, 3, 4, 3), 17, dtype=torch.uint8)})
        self.pose_updates: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.sensor_updates: list[tuple[float, bool]] = []

    def set_world_poses_from_view(self, eye: torch.Tensor, target: torch.Tensor) -> None:
        self.pose_updates.append((eye.clone(), target.clone()))

    def update(self, *, dt: float, force_recompute: bool) -> None:
        self.sensor_updates.append((dt, force_recompute))


class _FakeScene:
    def __init__(self, camera: _FakeCamera) -> None:
        self.camera = camera
        self.lookups: list[str] = []

    def __getitem__(self, name: str) -> _FakeCamera:
        self.lookups.append(name)
        if name != "ego_camera":
            raise KeyError(name)
        return self.camera


class _FakeSim:
    def __init__(self) -> None:
        self.render_calls = 0

    def render(self) -> None:
        self.render_calls += 1


def _load_recorder_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")
    managers.manager_term_cfg = SimpleNamespace(RecorderTermCfg=_FakeRecorderTermCfg)
    managers.recorder_manager = SimpleNamespace(
        RecorderManagerBaseCfg=_FakeRecorderManagerBaseCfg,
        RecorderTerm=_FakeRecorderTerm,
    )
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda value: value
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)

    module_name = "_camera_recorder_under_test"
    spec = importlib.util.spec_from_file_location(module_name, RECORDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_dataset_camera_records_frame_zero_without_moving_attached_sensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_recorder_module(monkeypatch)
    writer = _FakeWriter()
    monkeypatch.setattr(module.imageio, "get_writer", lambda *args, **kwargs: writer)

    camera = _FakeCamera()
    scene = _FakeScene(camera)
    simulation = _FakeSim()
    environment = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene=scene,
        sim=simulation,
        wrapper=SimpleNamespace(
            start_idx=0,
            config={"render_frame_skip": 2, "max_render_envs": 1},
        ),
    )
    config = SimpleNamespace(
        video_save_path=str(tmp_path),
        video_quality=5,
        camera_name="ego_camera",
        track_root=False,
    )
    recorder = module.RenderEnvsRecorderTerm(config, environment)

    recorder.record_post_step()
    recorder.record_post_step()
    recorder.record_post_step()

    assert scene.lookups == ["ego_camera", "ego_camera"]
    assert len(writer.frames) == 2
    assert writer.frames[0].shape == (3, 4, 3)
    assert writer.frames[0].tolist() == np.full((3, 4, 3), 17, dtype=np.uint8).tolist()
    assert camera.pose_updates == []
    assert camera.sensor_updates == [(0.0, True), (0.0, True)]
    assert camera._is_outdated.tolist() == [True]
    assert simulation.render_calls == 4

    recorder.close_writers()
    assert writer.closed


def test_tracking_camera_preserves_free_camera_follow_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_recorder_module(monkeypatch)
    writer = _FakeWriter()
    monkeypatch.setattr(module.imageio, "get_writer", lambda *args, **kwargs: writer)

    camera = _FakeCamera()
    motion = SimpleNamespace(robot_body_pos_w=torch.tensor([[[1.0, 2.0, 3.0]]]))
    environment = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene=_FakeScene(camera),
        sim=_FakeSim(),
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        wrapper=SimpleNamespace(
            start_idx=0,
            config={
                "render_frame_skip": 1,
                "max_render_envs": 1,
                "eval_camera_offset": [2.0, -1.0, 0.5],
            },
        ),
    )
    config = SimpleNamespace(
        video_save_path=str(tmp_path),
        video_quality=5,
        camera_name="ego_camera",
        track_root=True,
    )
    recorder = module.RenderEnvsRecorderTerm(config, environment)

    recorder.record_post_step()

    assert len(camera.pose_updates) == 1
    eye, target = camera.pose_updates[0]
    assert eye.tolist() == [[3.0, 1.0, 3.5]]
    assert target.tolist() == [[1.0, 2.0, 3.0]]
    assert camera._view._sync_usd_on_fabric_write is True
    recorder.close_writers()


def test_dataset_overlay_selects_attached_ego_camera_and_matching_calibration() -> None:
    dataset = yaml.safe_load(
        (REPO_ROOT / "gear_sonic/config/manager_env/recorders/dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    base_environment = yaml.safe_load(
        (REPO_ROOT / "gear_sonic/config/manager_env/base_env.yaml").read_text(encoding="utf-8")
    )
    base_evaluation = yaml.safe_load(
        (REPO_ROOT / "gear_sonic/config/base_eval.yaml").read_text(encoding="utf-8")
    )

    render_config = dataset["render_envs"]
    assert dataset["dataset_export_mode"] == 0
    assert render_config["camera_name"] == "ego_camera"
    assert render_config["track_root"] is False
    assert dataset["trajectory"]["save_path"] == "${manager_env.config.save_trajectory_dir}"

    environment_config = base_environment["config"]
    evaluation_config = base_evaluation["manager_env"]["config"]
    assert environment_config["render_ego"] is False
    assert evaluation_config["render_ego"] is False
    assert evaluation_config["cameras"] == environment_config["cameras"]
    assert environment_config["cameras"]["camera_attached_link"] == "torso_link/head_link"
    assert environment_config["cameras"]["camera_resolution"] == [480, 640]


def test_eval_enables_camera_extension_before_launch_for_ego_rendering() -> None:
    source = (REPO_ROOT / "gear_sonic/eval_agent_trl.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    camera_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "enable_cameras"
            for target in node.targets
        )
    ]
    launcher_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AppLauncher"
    ]

    assert len(camera_assignments) == 1
    flag_reads = {
        call.args[0].value
        for call in ast.walk(camera_assignments[0].value)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value is False
    }
    assert {"render_results", "enable_cameras", "render_ego"} <= flag_reads
    assert launcher_calls
    assert camera_assignments[0].lineno < min(call.lineno for call in launcher_calls)
