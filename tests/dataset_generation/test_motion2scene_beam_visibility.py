import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_beam_visibility import beam_ray_hits


def test_box_visibility_handles_parallel_miss_range_and_rotation():
    packet = {"origin": [0, 0, 1.25], "rays": [{"direction": [1, 0, 0]}]}
    beam = {
        "yaw_rad": 0.0,
        "center_xy_m": [2.0, 0.0],
        "underside_m": 1.2,
        "thickness_m": 0.1,
        "length_m": 0.1,
        "width_m": 1.2,
    }
    assert np.isclose(beam_ray_hits(packet, beam)[0]["distance_m"], 1.95)
    assert not beam_ray_hits(packet, {**beam, "center_xy_m": [4.0, 0.0]})
    assert not beam_ray_hits(packet, {**beam, "underside_m": 1.4})
    rotated = {**beam, "yaw_rad": np.pi / 2, "center_xy_m": [0.0, 2.0]}
    assert np.isclose(
        beam_ray_hits({**packet, "rays": [{"direction": [0, 1, 0]}]}, rotated)[0]["distance_m"],
        1.95,
    )
