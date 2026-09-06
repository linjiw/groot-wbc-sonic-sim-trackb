"""Check runtime root routes against source clips, detecting frozen augmentation."""

import numpy as np


def audit_root_route(root, raw, source_fps):
    root, raw = np.asarray(root), np.asarray(raw)
    if (
        root.shape != (199, 3)
        or raw.ndim != 2
        or raw.shape[1] != 3
        or len(raw) < 2
        or not np.isfinite(source_fps)
        or source_fps <= 0
        or not np.isfinite(root).all()
        or not np.isfinite(raw).all()
    ):
        raise ValueError("invalid loaded/source root route")
    expected = np.stack(
        [
            np.interp(np.arange(199) / 50, np.arange(len(raw)) / source_fps, raw[:, j])
            for j in range(2)
        ],
        axis=1,
    )
    error = float(np.max(np.abs(root[:, :2] - expected)))
    if error > 1e-4:
        raise ValueError(f"loaded reference bank differs from source route: {error} m")
    return error
