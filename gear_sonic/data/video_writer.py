import os
import queue
import sys
import threading
from typing import Final

import av
import numpy as np


class VideoWriter:
    _STOP: Final = object()

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True
        self._closed = False
        self._worker_error: BaseException | None = None

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height, (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot add a frame after the video writer has closed")
        self._assert_dimensions(frame)
        self._enqueue(frame)

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Video encoder worker failed") from self._worker_error

    def _enqueue(self, item: np.ndarray | object) -> None:
        """Enqueue without deadlocking if the encoder worker has failed."""
        while True:
            self._raise_worker_error()
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if not self._thread.is_alive():
                    self._raise_worker_error()
                    raise RuntimeError("Video encoder worker stopped unexpectedly")

    def _writer_worker(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                try:
                    if frame is self._STOP:
                        return
                    self._assert_dimensions(frame)
                    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                    if self._first_frame:
                        stderr_fd = sys.stderr.fileno()
                        old_stderr = os.dup(stderr_fd)
                        devnull = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull, stderr_fd)
                        try:
                            packets = self.stream.encode(video_frame)
                            for packet in packets:
                                self.container.mux(packet)
                        finally:
                            os.dup2(old_stderr, stderr_fd)
                            os.close(old_stderr)
                            os.close(devnull)
                            self._first_frame = False
                    else:
                        packets = self.stream.encode(video_frame)
                        for packet in packets:
                            self.container.mux(packet)
                finally:
                    self.queue.task_done()
        except BaseException as exc:  # propagate asynchronous encoder failures on stop/add
            self._worker_error = exc

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Drain frames, terminate the worker, flush, and close the container."""
        if self._closed:
            return self.output_path
        try:
            self._enqueue(self._STOP)
            self._thread.join()
            self._raise_worker_error()
            self._flush_stream()
            self.container.close()
        except BaseException:
            # A failed encoder must never leave a partial file that looks like a
            # committed episode. The worker has normally stopped by this point;
            # join it explicitly when it reported the failure before touching
            # the container or output path.
            if self._worker_error is not None:
                self._thread.join()
            try:
                self.container.close()
            except BaseException:
                pass
            self._closed = True
            try:
                os.remove(self.output_path)
            except FileNotFoundError:
                pass
            raise
        self._closed = True
        return self.output_path

    def cancel(self) -> None:
        """Stop without flushing queued frames and delete the output file."""
        if self._closed:
            return
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        if self._thread.is_alive():
            try:
                self._enqueue(self._STOP)
            except RuntimeError:
                pass
            self._thread.join()
        self.container.close()
        self._closed = True
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def __del__(self) -> None:
        try:
            self.cancel()
        except Exception:
            pass
