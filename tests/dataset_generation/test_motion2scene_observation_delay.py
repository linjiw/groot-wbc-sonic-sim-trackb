import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_observation_delay import (
    ObservationDelay,
)


@pytest.mark.parametrize("delay,frames", [(0, 0), (0.1, 5), (0.25, 13), (0.5, 25)])
def test_causal_packets_and_quantization(delay, frames):
    queue = ObservationDelay(delay)
    for i in range(40):
        packet = {"time_s": (i % 20) / 50, "occupied": i == 2, "rays": [i]}
        delivered = queue.push(packet)
        packet["rays"][0] = -1
        if i < frames:
            assert delivered is None
        else:
            assert delivered["capture_frame"] == i - frames
            assert delivered["rays"] == [i - frames]
            assert delivered["occupied"] == (i - frames == 2)
            assert (i - delivered["capture_frame"]) / 50 >= delay - 1e-9
            delivered["rays"][0] = -2
            assert queue.packets[i - frames]["rays"] == [i - frames]


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_invalid_delay(delay):
    with pytest.raises(ValueError):
        ObservationDelay(delay)
