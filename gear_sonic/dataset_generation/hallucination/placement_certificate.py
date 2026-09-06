"""Conservative subdivision of a static box's translation/yaw uncertainty domain.

Certificates are conditional on the query's absolute-error bound and geometry
contract. They cover placement only, not unrecorded motion or missing body geometry.
"""

import math

import numpy as np


def box_displacement_bound(half_widths, horizontal_radius):
    """Hausdorff bound for translation and yaw about the box's own center.

    half_widths are dx,dy,dz,dyaw radii about a pose-cell center. The angular
    radius is restricted to [0,pi], where this chord bound is monotone.
    """
    widths = np.asarray(half_widths, dtype=float)
    if widths.shape != (4,) or not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("cell half widths must be four finite nonnegative values")
    if not math.isfinite(horizontal_radius) or horizontal_radius < 0 or widths[3] > math.pi:
        raise ValueError("invalid box radius or angular domain")
    return float(np.linalg.norm(widths[:3]) + 2 * horizontal_radius * np.sin(widths[3] / 2))


def certify_placement(
    query, lower, upper, *, horizontal_radius, margin=0.01, query_error=1e-8, max_queries=4095
):
    """query(offset) -> (minimum target clearance, maximum upright clearance).

    Every aggregate must be 1-Lipschitz under the box displacement bound. The
    callback's declared query_error is an assumed absolute bound, not estimated
    here. All domain cells must be discharged before returning conditional_pass.
    Any certified point violation returns counterexample; exhaustion is unresolved.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != (4,) or upper.shape != (4,) or not np.isfinite([lower, upper]).all():
        raise ValueError("domain must have finite four-dimensional bounds")
    if np.any(lower > upper):
        raise ValueError("domain bounds must be ordered")
    if (
        not math.isfinite(margin)
        or margin <= 0
        or not math.isfinite(query_error)
        or query_error < 0
    ):
        raise ValueError("invalid margin or query error")
    if not isinstance(max_queries, int) or max_queries < 1:
        raise ValueError("query budget must be a positive integer")
    box_displacement_bound((upper - lower) / 2, horizontal_radius)
    pending = [(lower, upper, 0)]
    rows, certified = [], 0
    worst_bound = math.inf
    while pending and len(rows) < max_queries:
        lo, hi, depth = pending.pop()
        center, half = (lo + hi) / 2, (hi - lo) / 2
        values = np.asarray(query(center.copy()), dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("query must return two finite clearance values")
        target, upright = values
        bound = box_displacement_bound(half, horizontal_radius) + query_error
        slack_bound = min(target - margin, -margin - upright) - bound
        row = {
            "lower": lo.tolist(),
            "upper": hi.tolist(),
            "center": center.tolist(),
            "depth": depth,
            "target_clearance_m": float(target),
            "upright_clearance_m": float(upright),
            "distance_bound_m": bound,
            "margin_slack_lower_bound_m": float(slack_bound),
        }
        rows.append(row)
        if target + query_error < margin or upright - query_error > -margin:
            row["decision"] = "counterexample"
            return {
                "status": "counterexample",
                "queries": len(rows),
                "certified_cells": certified,
                "counterexample": row,
                "trace": rows,
            }
        if slack_bound >= 0:
            row["decision"] = "certified_cell"
            certified += 1
            worst_bound = min(worst_bound, slack_bound)
            continue
        influence = half.copy()
        influence[3] = 2 * horizontal_radius * np.sin(half[3] / 2)
        axis = int(np.argmax(influence))
        midpoint = center[axis]
        if not lo[axis] < midpoint < hi[axis]:
            row["decision"] = "unresolved_precision"
            pending.append((lo, hi, depth))
            break
        row["decision"] = "split"
        first_hi, second_lo = hi.copy(), lo.copy()
        first_hi[axis], second_lo[axis] = midpoint, midpoint
        pending.extend(((second_lo, hi, depth + 1), (lo, first_hi, depth + 1)))
    return {
        "status": "unresolved" if pending else "conditional_pass",
        "queries": len(rows),
        "certified_cells": certified,
        "unresolved_cells": len(pending),
        "certified_slack_lower_bound_m": None if certified == 0 else float(worst_bound),
        "remaining_domains": [
            {"lower": lo.tolist(), "upper": hi.tolist(), "depth": depth}
            for lo, hi, depth in pending
        ],
        "trace": rows,
    }
