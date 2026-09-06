"""Action-data admission independent of scientific success or generator arm."""

import numpy as np


def reserved_layout(station, height, layouts):
    if not np.isfinite([station, height]).all():
        raise ValueError("nonfinite layout")
    return any(
        abs(station - row["station"]) <= 0.01 + 1e-12
        and abs(height - row["underside_m"]) <= 0.005 + 1e-12
        for row in layouts
    )


def pair_decisions(first, second):
    """Require exact causal state and sensing agreement before either command."""
    keys = (
        "phase_s",
        "capture_frame",
        "capture_elapsed_s",
        "active_before",
        "packet",
        "state",
        "features",
        "joint_names",
        "root_pos_w",
        "root_quat_w",
    )
    differences = [key for key in keys if first[key] != second[key]]
    valid = (
        not differences
        and first["phase_s"] == 0.3
        and first["active_before"] == 0
        and first["requested_action"] == 0
        and second["requested_action"] == 1
        and len(first["features"]) == 214
        and np.isfinite(first["features"]).all()
    )
    return {"valid": bool(valid), "differing_fields": differences}
