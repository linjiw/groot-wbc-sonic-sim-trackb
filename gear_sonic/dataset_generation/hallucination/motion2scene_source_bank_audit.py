"""Source-specific bounds for a loaded neutral/d040 bank."""

import numpy as np

from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES

from .motion2scene_reference_bank import audit_root_route


def source_bank_errors(bank, entries, joint_names):
    if len(joint_names) != 29 or set(joint_names) != set(KIMODO_G1_JOINT_NAMES):
        raise ValueError("source bank requires all named G1 joints")
    if bank["root_xyz"].shape != (2, 199, 3) or bank["joint_pos"].shape != (2, 199, 29):
        raise ValueError("expected two complete 199-frame banks")
    if float(bank["fps"]) != 50 or len(entries) != 2:
        raise ValueError("two 50 Hz banks required")
    order = [KIMODO_G1_JOINT_NAMES.index(j) for j in joint_names]
    rows = []
    for i, entry in enumerate(entries):
        root = np.asarray(entry["root_trans_offset"])
        raw = np.asarray(entry["dof"])
        if raw.shape != (len(root), 29) or not np.isfinite(raw).all():
            raise ValueError("invalid source joints")
        times = np.arange(len(root)) / float(entry["fps"])
        dof = np.stack([np.interp(np.arange(199) / 50, times, raw[:, j]) for j in order], 1)
        errors = {
            "root_xy_m": audit_root_route(bank["root_xyz"][i], root, float(entry["fps"])),
            "joint_scalar_interpolation_rad": float(abs(bank["joint_pos"][i] - dof).max()),
            "root_z_first_offset_m": float(bank["root_xyz"][i, 0, 2] - root[0, 2]),
        }
        if not np.isfinite(list(errors.values())).all():
            raise ValueError("nonfinite source bank")
        rows.append(errors)
    return rows
