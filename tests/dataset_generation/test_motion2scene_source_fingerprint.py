import importlib
from pathlib import Path

import numpy as np
import pytest


def test_fingerprint_ignores_horizontal_translation_and_quaternion_sign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    fingerprint = importlib.import_module(
        "motion2scene_reserve_provenance_sources"
    ).canonical_motion_hash
    q = np.zeros((3, 36))
    q[:, 3] = 1
    q[:, 0] = [0, 1, 2]
    transformed = q.copy()
    transformed[:, :2] += [10, -4]
    transformed[:, 3:7] *= -1
    assert fingerprint(q) == fingerprint(transformed)
    transformed[1, 10] += 2e-5
    assert fingerprint(q) != fingerprint(transformed)
    with pytest.raises(ValueError):
        fingerprint(np.zeros((3, 36)))
