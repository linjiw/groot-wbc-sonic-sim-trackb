"""Wait for room on a shared GPU rather than failing rollouts against a full one.

This card is shared, and a neighbour holding 22 GB of a 32 GB device is normal here. A
rollout started under those conditions does not run slowly -- it dies, because PhysX asks
for a 256 MiB block up front and gets nothing:

    PxgCudaDeviceMemoryAllocator failed to allocate memory 268435456 bytes!
    PhysX error: Unable to create scene.

The damage is not the lost rollout, it is what the lost rollout looks like downstream. A
counterfactual family verified by twelve empty cells reports itself as not robust, and a
batch reports a low acceptance rate; both blame the science for a neighbour's memory. That
is the same confusion the accepted/unevaluable split exists to prevent, one layer down.

So capacity is checked before a rollout starts, and its absence is a reason to wait rather
than a result.
"""

from __future__ import annotations

import subprocess
import time

#: Free memory a single-environment Isaac rollout needs before it is worth starting, in MiB.
#: Measured against rollouts that succeeded on this card; PhysX's own first allocation is
#: 256 MiB and the scene, robot and renderer follow it.
DEFAULT_REQUIRED_MIB = 6000


class GpuUnavailable(RuntimeError):
    """The card never freed up. Never a scientific result."""


def free_gpu_mib() -> int:
    """Free memory on the first CUDA device, or -1 if it cannot be read.

    -1 rather than an exception: a machine without ``nvidia-smi`` should not be prevented
    from running, and the caller treats an unreadable card as "go ahead and try".
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return -1


def wait_for_gpu(
    required_mib: int = DEFAULT_REQUIRED_MIB,
    *,
    timeout_s: float = 7200.0,
    poll_s: float = 60.0,
    announce=print,
) -> None:
    """Block until the card has ``required_mib`` free, or raise ``GpuUnavailable``."""
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        free = free_gpu_mib()
        if free < 0 or free >= required_mib:
            if announced:
                announce(f"    GPU has {free} MiB free; continuing")
            return
        if time.monotonic() > deadline:
            raise GpuUnavailable(
                f"only {free} MiB free after {timeout_s / 60:.0f} min; need {required_mib}"
            )
        announce(f"    waiting for GPU: {free} MiB free, need {required_mib} MiB")
        announced = True
        time.sleep(poll_s)
