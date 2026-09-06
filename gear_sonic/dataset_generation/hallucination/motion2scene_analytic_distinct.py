"""Select distinct beam candidates while balancing search-passing stations."""

import numpy as np


def distinct_indices(trace, count=8):
    candidates, slack, stations = (
        np.asarray(trace[k]) for k in ("candidates", "slack", "station_ids")
    )
    if candidates.ndim != 2 or candidates.shape[1] != 2 or slack.shape != (len(candidates),):
        raise ValueError("invalid search trace")
    if (
        stations.shape != slack.shape
        or not np.isfinite(candidates).all()
        or not np.isfinite(slack).all()
    ):
        raise ValueError("nonfinite or misaligned search trace")
    # All ranking uses the search trace, never an independent audit or reference map.
    station_order = sorted(set(stations.tolist()), key=lambda s: (-slack[stations == s].max(), s))
    queues = {
        s: sorted(
            np.flatnonzero((stations == s) & (slack >= 0)).tolist(), key=lambda i: (-slack[i], i)
        )
        for s in station_order
    }
    chosen, seen = [], set()
    while any(queues.values()) and len(chosen) < count:
        for station in station_order:
            queue = queues[station]
            while queue and tuple(candidates[queue[0]]) in seen:
                queue.pop(0)
            if queue and len(chosen) < count:
                index = queue.pop(0)
                chosen.append(index)
                seen.add(tuple(candidates[index]))
    # Refusals remain requested outputs if search found fewer than count unique passes.
    for index in sorted(range(len(candidates)), key=lambda i: (-slack[i], i)):
        if len(chosen) == count:
            break
        if tuple(candidates[index]) not in seen:
            chosen.append(index)
            seen.add(tuple(candidates[index]))
    if len(chosen) != count:
        raise ValueError("candidate bank contains fewer than requested distinct placements")
    return np.asarray(chosen)
