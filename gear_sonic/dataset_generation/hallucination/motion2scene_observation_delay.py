"""Causal, immutable observation packets at a declared capture cadence."""

import copy
import math


class ObservationDelay:
    def __init__(self, delay_s, fps=50):
        if not math.isfinite(delay_s) or delay_s < 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("invalid delay or cadence")
        self.frames = math.ceil(delay_s * fps - 1e-9)
        self.fps = fps
        self.packets = []

    def push(self, packet):
        index = len(self.packets)
        captured = copy.deepcopy(packet)
        captured["capture_frame"] = index
        captured["capture_elapsed_s"] = index / self.fps
        self.packets.append(captured)
        eligible = index - self.frames
        return copy.deepcopy(self.packets[eligible]) if eligible >= 0 else None
