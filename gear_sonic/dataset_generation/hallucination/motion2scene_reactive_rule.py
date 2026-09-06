"""Frozen development rule for the phase-aligned sensing/interface pilot."""

import math


def select_skill(mode, time_s, occupied, active):
    """Latch an early overhead observation, then return at a fixed legal phase."""
    if mode not in ("reactive", "blind", "oracle"):
        raise ValueError("unknown interface mode")
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("invalid motion clock")
    if time_s >= 3.3 or mode == "blind":
        return 0
    if active or (time_s >= 0.2 and (occupied or mode == "oracle")):
        return 1
    return 0
