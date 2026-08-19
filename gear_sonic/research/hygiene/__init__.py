"""Motion hygiene for SONIC reference banks: screen, repair, and diagnose.

The package answers one question that SONIC's training loop cannot: *is this
reference clip something a controller could ever have tracked?*  Retargeted
motion banks carry clips whose feet never touch the floor, whose root
accelerates with nothing pushing on it, or whose joints would need more torque
than the G1 has.  Training on them spends exposure on impossible targets and
poisons the tracking reward.

Three pieces, each usable on its own:

``motion_io``
    the on-disk SONIC clip format, plus SONIC's exact load-time 30 -> 50 Hz
    resample rule.
``screen``
    per-clip dynamic feasibility: contact-free inverse dynamics against the
    contacts the reference offers, solved as a friction-cone + torque-limited LP.
``repair``
    the contact-projection operator that pulls a salvageable clip back onto the
    floor, and its budget/accept criteria.
``sampler_diagnostics``
    what the training sampler does with clips of each hygiene class.

Positioning: ``gear_sonic.research.lace`` carries a frozen hash cascade and a
*kinematic* reference-feasibility proxy.  This package is the dynamic
counterpart and is deliberately independent of it - nothing here imports LACE at
module import time, and nothing here writes into LACE's artifacts.

Everything in this package is CPU-only (MuJoCo, NumPy, SciPy).  Imports are kept
lazy where they are expensive so that multiprocessing workers start fast.
"""

from __future__ import annotations

from gear_sonic.research.hygiene.motion_io import (
    MOTION_KEYS,
    Motion,
    load_motion,
    motion_sha256,
    resample_to,
    save_motion,
    validate_motion,
)
from gear_sonic.research.hygiene.screen import (
    DEFAULT_G1_MJCF,
    SCREEN_SCHEMA_VERSION,
    ClipScreen,
    ScreenThresholds,
    load_model,
    screen_motion,
)

__all__ = [
    "DEFAULT_G1_MJCF",
    "MOTION_KEYS",
    "SCREEN_SCHEMA_VERSION",
    "ClipScreen",
    "Motion",
    "ScreenThresholds",
    "load_model",
    "load_motion",
    "motion_sha256",
    "resample_to",
    "save_motion",
    "screen_motion",
    "validate_motion",
]
