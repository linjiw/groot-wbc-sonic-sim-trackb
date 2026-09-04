#!/usr/bin/env python3
"""Launch a minimal camera-free Isaac Lab simulation and exit after fixed steps."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Lab requires the application to exist before simulation modules are imported.
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    simulation = SimulationContext(SimulationCfg(dt=0.01))
    simulation.reset()
    for _ in range(args.steps):
        simulation.step()
    print(f"ISAACLAB_RUNTIME_SMOKE_SUCCESS steps={args.steps}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
