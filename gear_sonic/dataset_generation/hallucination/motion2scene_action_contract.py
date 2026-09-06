"""One encounter decision, with scene-independent transition permission."""

import numpy as np


def requested_skill(action, decision_time_s, time_s, active, issued):
    """Walk commits for the encounter; crouch requests once, then returns at 3.3 s."""
    if action not in (0, 1) or active not in (0, 1):
        raise ValueError("binary supported skills required")
    if not np.isfinite([decision_time_s, time_s]).all() or time_s < 0:
        raise ValueError("invalid clock")
    if (
        not 0.2 <= decision_time_s <= 0.4
        or abs(decision_time_s * 50 - round(decision_time_s * 50)) > 1e-8
    ):
        raise ValueError("decision must lie on the legal 50 Hz grid")
    if active and time_s >= 3.3:
        return 0, issued
    if not issued and time_s >= decision_time_s:
        # A missed decision tick is refused, never silently shifted later.
        return (action if abs(time_s - decision_time_s) < 1e-8 else active), True
    return active, issued


def command_permission(requested, active, time_s, joint_jump, root_jump):
    if requested not in (0, 1) or active not in (0, 1):
        raise ValueError("binary supported skills required")
    if (
        not np.isfinite([time_s, joint_jump, root_jump]).all()
        or min(time_s, joint_jump, root_jump) < 0
    ):
        raise ValueError("invalid guard input")
    if requested == active:
        return True, []
    legal = 0.2 <= time_s <= 0.4 if requested else 3.3 <= time_s <= 3.5
    reasons = []
    if not legal:
        reasons.append("outside_legal_phase")
    if joint_jump > 0.05:
        reasons.append("joint_reference_jump")
    if root_jump > 0.01:
        reasons.append("root_reference_jump")
    return not reasons, reasons


def paired_prefix(a, b, decision_time_s):
    """Exact recorded state/action/token history agreement, not a hidden-state snapshot."""
    times = np.asarray(a["motion_time_s"])
    n = int(round(decision_time_s * 50))
    if not np.array_equal(times[:n], np.arange(n) / 50):
        raise ValueError("incomplete pre-decision clock")
    keys = (
        "dof_pos",
        "dof_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "applied_joint_action",
        "action_motion_token",
        "reference_g1_qpos",
        "motion_time_s",
    )
    errors = {}
    for key in keys:
        x, y = np.asarray(a[key])[:n], np.asarray(b[key])[:n]
        if (
            x.shape != y.shape
            or len(x) != n
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("malformed paired prefix")
        errors[key] = float(np.max(np.abs(x - y)))
    return {
        "frames": n,
        "max_abs_errors": errors,
        "exact_match": all(v == 0 for v in errors.values()),
    }
