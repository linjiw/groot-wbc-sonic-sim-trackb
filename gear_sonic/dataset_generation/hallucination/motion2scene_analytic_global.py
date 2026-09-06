"""Budget-matched global beam proposals; selection sees search offsets only."""

import numpy as np


def global_search(nominal, ranking, query):
    nominal, ranking = np.asarray(nominal), np.asarray(ranking)
    if nominal.shape != (20, 2) or not np.isfinite(nominal).all():
        raise ValueError("requires twenty finite nominal station/height proposals")
    if np.any(nominal < [0.1, 1.1]) or np.any(nominal > [0.9, 1.45]):
        raise ValueError("nominal outside beam domain")
    if ranking.shape != (20,) or sorted(ranking.tolist()) != list(range(20)):
        raise ValueError("ranking must permute all station indices")
    candidates = np.repeat(nominal, 6, axis=0)
    candidates[:, 1] = np.clip(
        candidates[:, 1] + np.tile([-0.025, -0.015, -0.005, 0.005, 0.015, 0.025], 20), 1.1, 1.45
    )
    candidates = np.concatenate([candidates, nominal[ranking[:16]]])
    station_ids = np.r_[np.repeat(np.arange(20), 6), ranking[:16]]
    values = np.concatenate([query(chunk) for chunk in np.array_split(candidates, 17)])
    if values.shape != (136, 17, 2) or not np.isfinite(values).all():
        raise ValueError("expected finite search clearances (136,17,2)")
    slack = np.minimum(values[:, :, 1].min(1) - 0.01, -values[:, :, 0].max(1) - 0.01)
    winners = [
        np.flatnonzero(station_ids == i)[np.argmax(slack[station_ids == i])] for i in range(20)
    ]
    winners.sort(key=lambda i: (-slack[i], station_ids[i]))
    passing = [i for i in winners if slack[i] >= 0]
    pool = passing if passing else winners
    selected = np.array([pool[i % len(pool)] for i in range(8)])
    return candidates[selected], {
        "candidates": candidates,
        "station_ids": station_ids,
        "search_clearances": values,
        "slack": slack,
        "station_winners": np.array(winners),
        "selected": selected,
        "passing_station_count": np.array(len(passing)),
    }
