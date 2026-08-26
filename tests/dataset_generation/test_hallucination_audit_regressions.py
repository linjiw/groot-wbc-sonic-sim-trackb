"""Regressions for the 2026-08-26 LFH geometry/model audit.

Each test fails on the pre-audit code and passes after the fix.  They are grouped here rather
than spread across the existing files because they share one origin: instruments that failed
*open* -- returning a plausible number for a question they had not actually answered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.delivery import (
    DeliveryModel,
    DeliveryOutOfSupport,
)
from gear_sonic.dataset_generation.hallucination.keypoints import SemanticCapsuleTracks
from gear_sonic.dataset_generation.hallucination.reach import (
    lateral_face_reach,
    overhead_face_reach,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import _records
from gear_sonic.dataset_generation.hallucination.window import solve_window


def _one_capsule(start, end, radius=0.05, group="wrist_left", owner="left_wrist_yaw_link"):
    return SemanticCapsuleTracks(
        starts=np.asarray([[start]], dtype=np.float64),
        ends=np.asarray([[end]], dtype=np.float64),
        radii=np.asarray([radius], dtype=np.float64),
        owners=(owner,),
        groups=(group,),
        root_pos_w=np.asarray([[0.0, 0.0, 0.8]], dtype=np.float64),
        root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_lateral_reach_measures_only_the_part_inside_the_face():
    """A capsule that merely clips the slab must not report its far end's lateral extent."""
    tracks = _one_capsule((0.0, 0.10, 1.0), (1.0, 2.0, 1.0))
    reach = lateral_face_reach(tracks, (0.0, 0.0), "x", 0.10, (0.5, 1.5), "left")
    # Whole-capsule endpoints would give 2.0 m + radius; the in-slab surface is ~0.30 m.
    assert reach.per_keypoint_reach_m["wrist_left"] == pytest.approx(0.3024, abs=5e-3)


def test_lateral_reach_refuses_a_capsule_whose_in_slab_part_misses_the_band():
    """Overlapping the slab and the band separately is not occupying the face."""
    tracks = _one_capsule((0.0, 0.10, 0.2), (1.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="no capsule occupies"):
        lateral_face_reach(tracks, (0.0, 0.0), "x", 0.02, (1.5, 1.6), "left")


def test_window_margins_never_carry_infinity():
    """A group that never occupies the face has no margin, rather than an infinite one."""

    class _Face:
        axis_type = "lateral_one_sided"
        binding_keypoint = "wrist_left"
        critical_frame = 3

        def __init__(self, bound: float):
            self.reach_m = bound
            self.per_keypoint_reach_m = {
                "wrist_left": bound,
                "foot_left": -math.inf,
            }

    solution = solve_window(_Face(1.30), _Face(1.20), min_window_m=0.02)
    assert "foot_left" not in solution.per_keypoint_margins_m
    assert all(
        math.isfinite(value)
        for margins in solution.per_keypoint_margins_m.values()
        for value in margins.values()
    )


def test_root_tracks_must_be_finite():
    """NaN propagates through every argmin station selector, so it is refused at the door."""
    with pytest.raises(ValueError, match="must be finite"):
        SemanticCapsuleTracks(
            starts=np.zeros((2, 1, 3)),
            ends=np.ones((2, 1, 3)),
            radii=np.asarray([0.05]),
            owners=("left_wrist_yaw_link",),
            groups=("wrist_left",),
            root_pos_w=np.asarray([[0.0, 0.0, 0.8], [np.nan, 0.0, 0.8]]),
            root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        )


def test_overhead_face_reach_can_report_a_missing_group_for_profiles():
    """The single-face API still raises; profile mode keeps -inf at that station."""
    tracks = _one_capsule((5.0, 5.0, 1.0), (5.1, 5.0, 1.0))
    with pytest.raises(ValueError, match="misses semantic groups"):
        overhead_face_reach(tracks, (0.0, 0.0), "x", 0.1, 3.0)
    lenient = overhead_face_reach(tracks, (0.0, 0.0), "x", 0.1, 3.0, require_all_groups=False)
    assert lenient.per_keypoint_reach_m["wrist_left"] == -math.inf


def test_stage_reader_attributes_xform_ops_authored_after_a_child():
    """USD composes ops regardless of authoring order; so must the reader."""
    text = """#usda 1.0
def Xform "Frame"
{
    def Cube "Binding"
    {
        double size = 1
    }
    double3 xformOp:translate = (0, 0, 0.5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"""
    frame = next(record for record in _records(text) if record.name == "Frame")
    assert "xformOp:translate" in frame.direct_body


def test_stage_reader_does_not_absorb_a_sibling_block_into_a_prim():
    """Attributes of a following sibling must not be folded into the preceding prim."""
    text = """#usda 1.0
def Xform "Frame"
{
    def Cube "Binding"
    {
        double size = 1
    }
    def Xform "Other"
    {
        double3 xformOp:translate = (0, 0, 0.5)
    }
}
"""
    binding = next(record for record in _records(text) if record.name == "Binding")
    assert "xformOp:translate" not in binding.direct_body


def test_delivery_model_refuses_to_extrapolate():
    """Flat extrapolation returned the 30 mm response for a 300 mm command, band unchanged."""
    model = DeliveryModel(
        operator="local_arm_tuck",
        keypoint="wrist_left",
        axis="lateral",
        commanded_mm=(10.0, 20.0, 30.0),
        fitted_mm=(8.0, 15.0, 20.0),
        residual_q90_mm=2.0,
        training_motions=("a", "b", "c"),
        alpha_levels=(0.33, 0.67, 1.0),
        samples_per_level=(3, 3, 3),
        unsupported_alpha_levels=(),
    )
    assert model.predict(20.0)[0] == pytest.approx(15.0)
    with pytest.raises(DeliveryOutOfSupport):
        model.predict(120.0)


def test_stage_reader_scopes_attributes_at_depth_with_siblings():
    """Attributes belong to the prim that authored them, wherever they sit among its children."""
    text = """#usda 1.0
def Xform "World"
{
    double3 xformOp:translate = (1, 0, 0)
    def Xform "Frame"
    {
        double A = 1
        def Cube "Binding"
        {
            double size = 2
            def Cube "Inner" { double size = 3 }
            double afterInner = 9
        }
        double B = 2
        def Cube "Ctx" { double size = 4 }
        double C = 3
    }
    double3 xformOp:scale = (2, 2, 2)
}
"""
    records = {record.path: record.direct_body for record in _records(text)}
    assert set(records) == {
        "/World",
        "/World/Frame",
        "/World/Frame/Binding",
        "/World/Frame/Binding/Inner",
        "/World/Frame/Ctx",
    }
    # Ops authored before *and* after a child both belong to the parent.
    assert "xformOp:translate" in records["/World"]
    assert "xformOp:scale" in records["/World"]
    assert "double A" not in records["/World"]
    # Attributes interleaved between children all belong to the prim that authored them.
    for name in ("double A", "double B", "double C"):
        assert name in records["/World/Frame"]
    assert "double size" not in records["/World/Frame"]
    # A prim keeps its own attributes, including those after its child, and none of the child's.
    assert "double size = 2" in records["/World/Frame/Binding"]
    assert "afterInner" in records["/World/Frame/Binding"]
    assert "double size = 3" not in records["/World/Frame/Binding"]
    assert "double B" not in records["/World/Frame/Binding"]
    assert "double size = 4" in records["/World/Frame/Ctx"]
    assert "double C" not in records["/World/Frame/Ctx"]


def test_oriented_face_matches_axis_aligned_when_the_route_is_axis_aligned():
    """A zero-yaw route frame must reproduce the world-axis result exactly."""
    starts = np.asarray([[[0.0, 0.0, 1.20], [0.30, 0.10, 1.10]]])
    ends = np.asarray([[[0.05, 0.0, 1.22], [0.34, 0.10, 1.12]]])
    tracks = SemanticCapsuleTracks(
        starts=starts,
        ends=ends,
        radii=np.asarray([0.06, 0.05]),
        owners=("torso_link", "left_wrist_yaw_link"),
        groups=("head_torso", "wrist_left"),
        root_pos_w=np.asarray([[0.0, 0.0, 0.8]]),
        root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    axis_aligned = overhead_face_reach(tracks, (0.0, 0.0), "x", 0.4, 3.0, require_all_groups=False)
    oriented = overhead_face_reach(
        tracks, (0.0, 0.0), "x", 0.4, 3.0, require_all_groups=False, route_yaw_rad=0.0
    )
    assert oriented.reach_m == pytest.approx(axis_aligned.reach_m, abs=1e-12)
    assert oriented.binding_keypoint == axis_aligned.binding_keypoint


def test_oriented_face_is_invariant_to_rotating_the_whole_problem():
    """Rotating body and face together must not change the measured reach."""
    base_start = np.asarray([0.0, 0.0, 1.20])
    base_end = np.asarray([0.30, 0.02, 1.24])
    yaw = 0.9

    def _rotate(point, angle):
        cos, sin = math.cos(angle), math.sin(angle)
        return np.asarray(
            (cos * point[0] - sin * point[1], sin * point[0] + cos * point[1], point[2])
        )

    def _tracks(start, end):
        return SemanticCapsuleTracks(
            starts=np.asarray([[start]]),
            ends=np.asarray([[end]]),
            radii=np.asarray([0.06]),
            owners=("torso_link",),
            groups=("head_torso",),
            root_pos_w=np.asarray([[0.0, 0.0, 0.8]]),
            root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        )

    straight = overhead_face_reach(
        _tracks(base_start, base_end),
        (0.0, 0.0),
        "x",
        0.20,
        1.0,
        require_all_groups=False,
        route_yaw_rad=0.0,
    )
    turned = overhead_face_reach(
        _tracks(_rotate(base_start, yaw), _rotate(base_end, yaw)),
        (0.0, 0.0),
        "x",
        0.20,
        1.0,
        require_all_groups=False,
        route_yaw_rad=yaw,
    )
    assert turned.reach_m == pytest.approx(straight.reach_m, abs=1e-9)


def test_axis_aligned_face_mismeasures_a_turned_route():
    """The reason turning motions are excluded from the corpus.

    The body sits 0.25 m *along the executed route* from the station, so a correctly oriented
    0.20 m face excludes it. The world-axis face of identical nominal extent still catches it,
    because 0.25 m along a route yawed 50 degrees is only 0.161 m along world x, and the capsule
    radius then reaches back inside the world slab. The two instruments answer different
    questions about the same body, and only one is the question the window solver asks.
    """
    yaw = math.radians(50.0)
    cos, sin = math.cos(yaw), math.sin(yaw)
    offset = 0.25
    centre = np.asarray([cos * offset, sin * offset, 1.25])
    along = np.asarray([cos * 0.03, sin * 0.03, 0.0])
    tracks = SemanticCapsuleTracks(
        starts=np.asarray([[centre - along]]),
        ends=np.asarray([[centre + along]]),
        radii=np.asarray([0.05]),
        owners=("torso_link",),
        groups=("head_torso",),
        root_pos_w=np.asarray([[float(centre[0]), float(centre[1]), 0.8]]),
        root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    oriented = overhead_face_reach(
        tracks, (0.0, 0.0), "x", 0.20, 3.0, require_all_groups=False, route_yaw_rad=yaw
    )
    axis_aligned = overhead_face_reach(tracks, (0.0, 0.0), "x", 0.20, 3.0, require_all_groups=False)
    # Correctly oriented: the body is past the face, so it does not occupy it at all.
    assert oriented.per_keypoint_reach_m["head_torso"] == -math.inf
    # World-axis: the same body reports a finite reach against a face it never crosses.
    assert math.isfinite(axis_aligned.per_keypoint_reach_m["head_torso"])
