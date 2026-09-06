"""Finite reference-grid coverage and geometry-only beam proposals.

Passing grid centres are witnesses, never certified cells or feasible-volume estimates.
Envelope proposals require the finite capsule checker before any acceptance label.
"""

import numpy as np


def station_bins(scenes, count=20):
    scenes = np.asarray(scenes, dtype=float)
    if scenes.ndim != 2 or scenes.shape[1] != 2 or not np.isfinite(scenes).all():
        raise ValueError("expected finite station/height pairs")
    if count < 1 or np.any(scenes < [0.1, 1.1]) or np.any(scenes > [0.9, 1.45]):
        raise ValueError("invalid bin count or scene outside domain")
    return np.minimum(np.floor((scenes[:, 0] - 0.1) / (0.8 / count)).astype(int), count - 1)


def reference_coverage(reference, reference_pass, scenes, accepted):
    reference, scenes = np.asarray(reference), np.asarray(scenes)
    reference_pass, accepted = np.asarray(reference_pass), np.asarray(accepted)
    if reference_pass.dtype != bool or accepted.dtype != bool:
        raise ValueError("acceptance masks must be boolean")
    if reference_pass.shape != (len(reference),) or accepted.shape != (len(scenes),):
        raise ValueError("acceptance masks must match scene counts")
    support = set(station_bins(reference)[reference_pass].tolist())
    occupied = set(station_bins(scenes)[accepted].tolist())
    return {
        "reference_station_bins": sorted(support),
        "accepted_station_bins": sorted(occupied),
        "covered_reference_bins": sorted(support & occupied),
        "accepted_bins_outside_reference_support": sorted(occupied - support),
        "reference_relative_station_coverage": (
            len(support & occupied) / len(support) if support else None
        ),
        "empty_reference_map": not support,
        "accepted": int(accepted.sum()),
        "requested": len(scenes),
    }


def analytic_proposals(states, route, yaw, *, depth, width, count=8):
    """Rank 20 route stations by target/upright top-envelope separation.

    No model, event metadata, achieved rollout, grid map or final-audit feedback is used.
    A horizontal capsule AABB overlap can overestimate actual beam interference.
    """
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or not np.isfinite(route).all():
        raise ValueError("expected finite route (T,2)")
    if not 1 <= count <= 20 or depth <= 0 or width <= 0:
        raise ValueError("invalid output count or beam dimensions")
    progress = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    if progress[-1] <= 0:
        raise ValueError("stationary route")
    progress /= progress[-1]
    stations = 0.1 + (np.arange(20) + 0.5) * 0.04
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    labels = [s["label"] for s in states.values()]
    if sorted(labels) != ["d055", "neutral"]:
        raise ValueError("requires exactly one neutral and one d055 target")
    rows = []
    for station in stations:
        centre = np.array([np.interp(station, progress, route[:, a]) for a in range(2)])
        tops = {}
        for state in states.values():
            starts, ends, radii = (np.asarray(state[k]) for k in ("starts", "ends", "radii"))
            if not all(np.isfinite(v).all() for v in (starts, ends, radii)):
                raise ValueError("nonfinite capsule geometry")
            a, b = (starts[..., :2] - centre) @ rotation, (ends[..., :2] - centre) @ rotation
            half = np.array([depth, width]) / 2
            keep = np.all(np.minimum(a, b) - radii[..., None] <= half, axis=-1)
            keep &= np.all(np.maximum(a, b) + radii[..., None] >= -half, axis=-1)
            top = np.maximum(starts[..., 2], ends[..., 2]) + radii
            tops[state["label"]] = float(top[keep].max()) if keep.any() else None
        usable = all(v is not None for v in tops.values())
        low = tops["d055"] + 0.02 if usable else 1.1
        high = tops["neutral"] - 0.02 if usable else 1.1
        rows.append(
            {
                "station": float(station),
                "tops_m": tops,
                "height": float(np.clip((low + high) / 2, 1.1, 1.45)),
                "separation_m": high - low if usable else None,
            }
        )
    order = sorted(
        range(20),
        key=lambda i: (
            -(rows[i]["separation_m"] if rows[i]["separation_m"] is not None else -np.inf),
            i,
        ),
    )[:count]
    return np.array([[rows[i]["station"], rows[i]["height"]] for i in order]), rows
