"""Box-only proposal visibility, separate from physical sensor observations."""

import numpy as np


def beam_ray_hits(packet, beam):
    yaw = beam["yaw_rad"]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    centre = np.array([*beam["center_xy_m"], beam["underside_m"] + beam["thickness_m"] / 2])
    origin = np.array(packet["origin"], dtype=float)
    local = (origin - centre) @ rotation
    half = np.array([beam["length_m"], beam["width_m"], beam["thickness_m"]]) / 2
    if not np.isfinite(np.r_[origin, centre, half, yaw]).all() or (half <= 0).any():
        raise ValueError("finite positive box required")
    hits = []
    for index, ray in enumerate(packet["rays"]):
        direction = np.array(ray["direction"], dtype=float)
        if not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1):
            raise ValueError("unit finite ray required")
        v = direction @ rotation
        lo, hi = 0.0, 3.0
        for j in range(3):
            if abs(v[j]) < 1e-12:
                if abs(local[j]) > half[j]:
                    lo, hi = 1.0, 0.0
                    break
            else:
                t = sorted([(-half[j] - local[j]) / v[j], (half[j] - local[j]) / v[j]])
                lo, hi = max(lo, t[0]), min(hi, t[1])
        if lo <= hi:
            height = origin[2] + lo * direction[2]
            if 1.15 <= height <= 1.5:
                hits.append({"ray": index, "distance_m": float(lo), "height_m": float(height)})
    return hits
