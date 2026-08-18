"""The scene-first test set must be frozen, reproducible, and unfitted.

Fitted counterfactual families place their obstacle where a particular executed trajectory's
swept volume separates from another's, which makes them training material and nothing else.
Evaluating on them measures whether a model can recover geometry the generator derived from
the very trajectories it saw. These scenes exist to avoid that, and they are only worth
anything if nobody can quietly resample them after seeing a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[2] / "data/splits/scene_first_v1.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="test set not frozen yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_the_fingerprint_still_matches_the_scenes(manifest):
    """The one check that catches an edit to the parameters after the fact."""
    import hashlib

    payload = json.dumps(manifest["scenes"], sort_keys=True).encode()
    assert hashlib.sha256(payload).hexdigest() == manifest["fingerprint"]


def test_all_three_regimes_are_present_with_ten_scenes_each(manifest):
    counts: dict[str, int] = {}
    for scene in manifest["scenes"]:
        counts[scene["regime"]] = counts.get(scene["regime"], 0) + 1
    assert counts == {"overhead": 10, "lateral": 10, "floor": 10}


def test_no_scene_references_a_motion(manifest):
    """The definition of scene-first. A parameter derived from a trajectory would make this
    a fitted scene wearing an evaluation label."""
    for scene in manifest["scenes"]:
        assert "motion" not in json.dumps(scene).lower()
        assert set(scene) == {"scene_id", "regime", "parameters", "route_length_m"}


def test_the_ranges_include_scenes_where_nothing_fits_and_scenes_where_everything_does(manifest):
    """A test set of only discriminative scenes measures an easier task than the real one,
    where most geometry is uninformative and the model must notice which is not.

    A plain walk's silhouette peak is 1.248 m and a retargeted crouch reaches 1.036 m.
    """
    shelves = [
        s["parameters"]["shelf_underside_m"]
        for s in manifest["scenes"] if s["regime"] == "overhead"
    ]
    assert min(shelves) < 1.036, "no scene stops every behaviour"
    assert max(shelves) > 1.248, "no scene admits every behaviour"


def test_the_rules_are_recorded_alongside_the_scenes(manifest):
    joined = " ".join(manifest["rules"]).lower()
    assert "never be used for training" in joined
    assert "not permitted" in joined


def test_the_set_is_reproducible_from_its_seed(manifest):
    """Anyone must be able to regenerate it exactly, or "frozen" is just a filename."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts/research/freeze_scene_first_testset.py"
    spec = importlib.util.spec_from_file_location("freeze_scene_first_testset", path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves its own module through sys.modules, and
    # an unregistered module makes the decorator itself raise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    import numpy as np

    rng = np.random.default_rng(manifest["seed"])
    scenes = []
    for regime in manifest["regimes"]:
        scenes.extend(module.sample(regime, manifest["per_regime"], rng))
    assert module.fingerprint(scenes) == manifest["fingerprint"]
