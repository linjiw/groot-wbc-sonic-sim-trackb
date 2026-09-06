"""Sparse overhead ray observation with typed PhysX hits and fail-fast callbacks."""

from collections.abc import Mapping
import math

import numpy as np


def normalize_hit(hit):
    if isinstance(hit, Mapping):
        path = hit.get("collision", hit.get("rigidBody", ""))
        distance, position = hit["distance"], hit["position"]
    else:
        path, distance, position = hit.collision, hit.distance, hit.position
    values = np.array([distance, *position], dtype=float)
    if not str(path) or values.shape != (4,) or not np.isfinite(values).all() or distance < 0:
        raise ValueError("invalid ray hit")
    return {"distance": float(distance), "position": values[1:].tolist(), "path": str(path)}


def observe_overhead(root, quat, query):
    w, x, y, z = quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    origin = root + np.array([0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4])
    rays = []
    for elevation in (-5, 0, 5, 10):
        for azimuth in (-10, 0, 10):
            a, e = yaw + math.radians(azimuth), math.radians(elevation)
            direction = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
            hits, errors = [], []

            def collect(hit):
                # PyBind may swallow callback exceptions. Re-raise after the query.
                try:
                    normalized = normalize_hit(hit)
                    if not normalized["path"].startswith("/World/envs/env_0/Robot"):
                        hits.append(normalized)
                    return True
                except Exception as error:
                    errors.append(error)
                    return False

            query.raycast_all(tuple(float(v) for v in origin), direction, 3.0, collect)
            if errors:
                raise RuntimeError("scene-ray callback failed; observation is invalid") from errors[
                    0
                ]
            nearest = min(hits, key=lambda hit: hit["distance"]) if hits else None
            rays.append({"direction": direction, "hit": nearest})
    occupied = any(r["hit"] and 1.15 <= r["hit"]["position"][2] <= 1.5 for r in rays)
    return bool(occupied), origin.tolist(), rays
