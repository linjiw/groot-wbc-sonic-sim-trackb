import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_reference_bank import audit_root_route


def routes():
    raw = np.stack([np.linspace(0, 4.4, 120), np.zeros(120), np.full(120, 0.8)], axis=1)
    loaded = np.stack(
        [np.interp(np.arange(199) / 50, np.arange(120) / 30, raw[:, i]) for i in range(3)], axis=1
    )
    return raw, loaded


def test_interpolated_unaugmented_route_passes():
    raw, loaded = routes()
    assert audit_root_route(loaded, raw, 30) == 0


def test_freeze_frame_regression_fails():
    raw, loaded = routes()
    loaded[24:] = loaded[23]
    with pytest.raises(ValueError, match="differs from source route"):
        audit_root_route(loaded, raw, 30)


def test_invalid_loaded_shapes_and_nonfinite_routes_fail():
    raw, loaded = routes()
    with pytest.raises(ValueError, match="invalid"):
        audit_root_route(loaded[:-1], raw, 30)
    loaded[0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        audit_root_route(loaded, raw, 30)
