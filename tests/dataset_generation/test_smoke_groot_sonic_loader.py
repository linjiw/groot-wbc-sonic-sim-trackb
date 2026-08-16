from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research import smoke_groot_sonic_loader as smoke


class FakeTag:
    value = "unitree_g1_sonic"


class FakeEmbodimentTag:
    UNITREE_G1_SONIC = FakeTag()


class FakeEpisodeLoader:
    def __init__(self) -> None:
        self.episode = [object()] * 40

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> list[object]:
        assert index == 0
        return self.episode


class FakeShardedSingleStepDataset:
    init_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs = kwargs
        self.episode_loader = FakeEpisodeLoader()

    def __len__(self) -> int:
        return 1


MODALITY_CONFIG = {
    "state": SimpleNamespace(delta_indices=[0]),
    "action": SimpleNamespace(delta_indices=list(range(40))),
    "video": SimpleNamespace(delta_indices=[0]),
    "language": SimpleNamespace(delta_indices=[0]),
}


def fake_extract_step_data(
    episode,
    step_index,
    modality_config,
    tag,
    *,
    allow_padding,
):
    assert len(episode) == 40
    assert step_index == 0
    assert modality_config is MODALITY_CONFIG
    assert tag is FakeEmbodimentTag.UNITREE_G1_SONIC
    assert allow_padding is False
    return SimpleNamespace(
        states={
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in smoke.EXPECTED_STATE_SHAPES.items()
        },
        actions={
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in smoke.EXPECTED_ACTION_SHAPES.items()
        },
        images={
            key: np.zeros(shape, dtype=np.uint8)
            for key, shape in smoke.EXPECTED_VIDEO_SHAPES.items()
        },
        text="walk to the workstation",
    )


def fake_api() -> smoke.GrootApi:
    return smoke.GrootApi(
        embodiment_tag=FakeEmbodimentTag,
        modality_configs={FakeTag.value: MODALITY_CONFIG},
        dataset_class=FakeShardedSingleStepDataset,
        extract_step_data=fake_extract_step_data,
        module_file="/fake/gr00t/__init__.py",
    )


def test_runtime_smoke_uses_official_dataset_path_and_reports_shapes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(smoke, "load_groot_api", fake_api)
    monkeypatch.setattr(
        smoke,
        "get_groot_provenance",
        lambda module_file: {
            "module_file": module_file,
            "package_version": "0.1.0",
            "vcs_revision": smoke.PINNED_GROOT_REVISION,
        },
    )

    report = smoke.run_loader_smoke(tmp_path)

    assert report["ok"] is True
    assert report["gate"] == "isaac_groot_runtime_loader"
    assert report["action_horizon"] == 40
    assert report["action_shapes"]["motion_token"] == [40, 64]
    assert report["video_shapes"]["ego_view"] == [1, 480, 640, 3]
    assert FakeShardedSingleStepDataset.init_kwargs == {
        "dataset_path": tmp_path,
        "embodiment_tag": FakeEmbodimentTag.UNITREE_G1_SONIC,
        "modality_configs": MODALITY_CONFIG,
        "shard_size": 64,
        "episode_sampling_rate": 1.0,
        "seed": 0,
        "allow_padding": False,
    }


def test_cli_reports_missing_groot_as_unavailable(tmp_path: Path, monkeypatch, capsys) -> None:
    def unavailable(*args, **kwargs):
        raise smoke.GrootUnavailableError(
            "NVIDIA Isaac-GR00T is not installed; this is not the file-only gate"
        )

    monkeypatch.setattr(smoke, "run_loader_smoke", unavailable)

    return_code = smoke.main([str(tmp_path)])

    assert return_code == 2
    assert "UNAVAILABLE: NVIDIA Isaac-GR00T is not installed" in capsys.readouterr().err


def test_loader_api_reports_absent_package_clearly(monkeypatch) -> None:
    def missing_package(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name="gr00t")

    monkeypatch.setattr(smoke, "import_module", missing_package)

    with pytest.raises(smoke.GrootUnavailableError, match="is not installed"):
        smoke.load_groot_api()


def test_provenance_rejects_an_installed_unpinned_revision(monkeypatch) -> None:
    class UnpinnedDistribution:
        version = "0.1.0"

        @staticmethod
        def read_text(name: str) -> str:
            assert name == "direct_url.json"
            return '{"vcs_info": {"commit_id": "deadbeef"}}'

    monkeypatch.setattr(smoke, "distribution", lambda name: UnpinnedDistribution())

    with pytest.raises(smoke.GrootUnavailableError, match="does not match"):
        smoke.get_groot_provenance("/fake/gr00t/__init__.py")


def test_runtime_smoke_rejects_wrong_action_shape(tmp_path: Path, monkeypatch) -> None:
    def wrong_shape_extract(*args, **kwargs):
        sample = fake_extract_step_data(*args, **kwargs)
        sample.actions["motion_token"] = np.zeros((39, 64), dtype=np.float32)
        return sample

    api = fake_api()
    monkeypatch.setattr(
        smoke,
        "load_groot_api",
        lambda: smoke.GrootApi(
            embodiment_tag=api.embodiment_tag,
            modality_configs=api.modality_configs,
            dataset_class=api.dataset_class,
            extract_step_data=wrong_shape_extract,
            module_file=api.module_file,
        ),
    )
    monkeypatch.setattr(smoke, "get_groot_provenance", lambda module_file: {})

    try:
        smoke.run_loader_smoke(tmp_path)
    except smoke.GrootLoaderSmokeError as exc:
        assert "action shapes do not match UNITREE_G1_SONIC" in str(exc)
    else:
        raise AssertionError("wrong action horizon unexpectedly passed")
