"""Shared-world, first-episode passage scoring for the beam development intervention."""

import numpy as np

from gear_sonic.dataset_generation.trajectory_segments import find_reset_boundaries


def score_passage(payload, forces, beam):
    positions = np.asarray(payload["body_pos_w"])
    root = np.asarray(payload["root_pos_w"])
    gravity = np.asarray(payload["projected_gravity_b"])
    times = np.asarray(payload["motion_time_s"]).reshape(-1)
    forces = np.asarray(forces)
    fps = float(payload["fps"])
    n = len(times)
    if (
        n == 0
        or fps <= 0
        or not np.isfinite(fps)
        or positions.ndim != 3
        or positions.shape[0] != n
        or positions.shape[2] != 3
        or positions.shape[1] == 0
        or root.shape != (n, 3)
        or gravity.shape != (n, 3)
        or forces.ndim != 3
        or forces.shape[0] != n
        or forces.shape[2] != 3
        or forces.shape[1] == 0
        or not all(np.isfinite(a).all() for a in (positions, root, gravity, times, forces))
    ):
        raise ValueError("requires complete finite synchronized body/force/clock arrays")
    resets = find_reset_boundaries(times)
    end = resets[0] if resets else n
    normal = np.array([np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"])])
    downstream = ((positions[:end, :, :2] - beam["center_xy_m"]) @ normal).min(1)
    crossed = downstream >= beam["length_m"] / 2 + 0.1
    upright = (root[:end, 2] >= 0.5) & (-gravity[:end, 2] >= 0.5)
    window = int(np.ceil(0.3 * fps)) + 1
    finish = next(
        (
            i + window
            for i in range(max(0, end - window + 1))
            if crossed[i : i + window].all() and upright[i : i + window].all()
        ),
        None,
    )
    horizon = finish if finish is not None else end
    norms = np.linalg.norm(forces[:horizon], axis=-1)
    maximum = float(norms.max())
    contact = maximum > 1.0
    return {
        "pass": finish is not None and not contact,
        "observed_beam_contact": contact,
        "incomplete_crossing": not bool(crossed.any()),
        "stabilization_failed": finish is None,
        "fall_observed": bool((~upright).any()),
        "first_episode_frames": end,
        "capture_frames": n,
        "reset_count": len(resets),
        "passage_finish_frame_exclusive": finish,
        "stabilization_seconds": (window - 1) / fps,
        "maximum_beam_normal_force_n_through_passage": maximum,
        "force_threshold_frame_counts": {
            str(t): int((norms.max(1) > t).sum()) for t in (0.1, 1, 10)
        },
        "minimum_body_downstream_max_m": float(downstream.max()),
        "crossing_geometry": "all recorded body origins; not all native collider extents",
    }
