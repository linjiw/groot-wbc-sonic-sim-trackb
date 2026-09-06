"""Upper occupancy with measured lower free space; no obstacle identity input."""

import numpy as np

from .motion2scene_ray_observer import normalize_hit, observe_overhead


def nearest_hit(query, origin, direction, distance):
    hits, errors = [], []

    def collect(raw):
        try:
            hit = normalize_hit(raw)
            if not hit["path"].startswith("/World/envs/env_0/Robot"):
                hits.append(hit)
            return True
        except Exception as error:
            errors.append(error)
            return False

    query.raycast_all(tuple(float(x) for x in origin), tuple(direction), float(distance), collect)
    if errors:
        raise RuntimeError("lower free-space ray callback failed") from errors[0]
    return min(hits, key=lambda h: h["distance"]) if hits else None


def observe_overhang(root, quat, query):
    _, origin, upper_rays = observe_overhead(root, quat, query)
    origin = np.asarray(origin)
    for ray in upper_rays:
        hit = ray["hit"]
        ray["upper_candidate"] = bool(hit and 1.15 <= hit["position"][2] <= 1.5)
        ray["lower_rays"] = []
        ray["overhang"] = False
        if not ray["upper_candidate"]:
            continue
        offset = np.asarray(hit["position"])[:2] - origin[:2]
        horizontal_range = float(np.linalg.norm(offset))
        if horizontal_range < 1e-6:
            continue
        direction = (*list(offset / horizontal_range), 0.0)
        # Extend through the upper hit's near surface, without exceeding 3 m.
        distance = min(3.0, horizontal_range + 0.1)
        for height in (0.35, 0.75, 1.05):
            lower_origin = np.array([*origin[:2], height])
            lower = nearest_hit(query, lower_origin, direction, distance)
            ray["lower_rays"].append(
                {
                    "origin": lower_origin.tolist(),
                    "direction": direction,
                    "range_m": distance,
                    "hit": lower,
                }
            )
        ray["overhang"] = all(r["hit"] is None for r in ray["lower_rays"])
    return any(r["overhang"] for r in upper_rays), origin.tolist(), upper_rays


def transition_decision(mode, time_s, occupied, active, joint_jump, root_jump):
    """Return the requested skill and explicit permission; never silently reschedule."""
    if mode not in ("reactive", "blind", "oracle", "late_oracle"):
        raise ValueError("unknown guarded mode")
    if (
        not np.isfinite([time_s, joint_jump, root_jump]).all()
        or min(time_s, joint_jump, root_jump) < 0
    ):
        raise ValueError("invalid transition inputs")
    requested = active
    if active and time_s >= 3.3:
        requested = 0
    elif not active and time_s < 3.3:
        trigger = occupied if mode == "reactive" else mode == "oracle"
        if mode == "late_oracle":
            trigger = time_s >= 2.6
        if mode != "blind" and trigger and time_s >= 0.2:
            requested = 1
    if requested == active:
        return requested, True, []
    legal_phase = (0.2 <= time_s <= 0.4) if requested == 1 else (3.3 <= time_s <= 3.5)
    reasons = []
    if not legal_phase:
        reasons.append("outside_legal_phase")
    if joint_jump > 0.05:
        reasons.append("joint_reference_jump")
    if root_jump > 0.01:
        reasons.append("root_reference_jump")
    return requested, not reasons, reasons
