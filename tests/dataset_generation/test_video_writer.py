from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest


class _FakeStream:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.frames: list[np.ndarray] = []

    def encode(self, frame: np.ndarray | None = None) -> list:
        if frame is not None:
            self.frames.append(frame)
        return []


class _FakeContainer:
    def __init__(self, output_path: str) -> None:
        Path(output_path).touch()
        self.stream = _FakeStream()
        self.closed = False

    def add_stream(self, _codec: str, *, rate: float) -> _FakeStream:  # noqa: ARG002
        return self.stream

    def mux(self, _packet: object) -> None:
        pass

    def close(self) -> None:
        self.closed = True


fake_av = types.ModuleType("av")
fake_av.open = lambda output_path, mode: _FakeContainer(output_path)  # type: ignore[attr-defined]
fake_av.VideoFrame = types.SimpleNamespace(  # type: ignore[attr-defined]
    from_ndarray=lambda frame, format: frame
)
sys.modules.setdefault("av", fake_av)

from gear_sonic.data.video_writer import VideoWriter  # noqa: E402


def test_video_writer_drains_and_joins_worker(tmp_path: Path) -> None:
    output = tmp_path / "episode.mp4"
    writer = VideoWriter(str(output), width=64, height=48, fps=50)

    for value in range(4):
        writer.add_frame(np.full((48, 64, 3), value * 40, dtype=np.uint8))

    assert writer.stop() == str(output)
    assert writer.stop() == str(output)
    assert not writer._thread.is_alive()  # noqa: SLF001
    assert len(writer.stream.frames) == 4


def test_video_writer_cancel_joins_worker_and_removes_file(tmp_path: Path) -> None:
    output = tmp_path / "cancelled.mp4"
    writer = VideoWriter(str(output), width=64, height=48, fps=50)
    writer.add_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    writer.cancel()

    assert not writer._thread.is_alive()  # noqa: SLF001
    assert not output.exists()


def test_video_writer_failed_worker_stop_removes_partial_file(tmp_path: Path) -> None:
    output = tmp_path / "failed-worker.mp4"
    writer = VideoWriter(str(output), width=64, height=48, fps=50)

    def fail_encode(_frame: np.ndarray | None = None) -> list:
        raise RuntimeError("encode failed")

    writer.stream.encode = fail_encode
    writer.add_frame(np.zeros((48, 64, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="Video encoder worker failed"):
        writer.stop()

    assert not writer._thread.is_alive()  # noqa: SLF001
    assert not output.exists()


def test_video_writer_failed_flush_removes_partial_file(tmp_path: Path) -> None:
    output = tmp_path / "failed-flush.mp4"
    writer = VideoWriter(str(output), width=64, height=48, fps=50)
    writer.add_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.queue.join()

    def fail_flush(frame: np.ndarray | None = None) -> list:
        if frame is None:
            raise RuntimeError("flush failed")
        return []

    writer.stream.encode = fail_flush

    with pytest.raises(RuntimeError, match="flush failed"):
        writer.stop()

    assert not writer._thread.is_alive()  # noqa: SLF001
    assert not output.exists()
