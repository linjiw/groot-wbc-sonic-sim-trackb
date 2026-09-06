import numpy as np
import pytest

pytest.importorskip("pxr", reason="Requires USD libraries; see NATIVE_BEAM_AUDIT_V1_RESULT.md")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_native_geometry import (  # noqa: E402
    enclosing_sphere,
    extract_native_geometry,
)


def body_stage():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Robot/torso")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddTranslateOp().Set((7, 8, 9))
    return stage


def test_native_capsule_is_body_local_despite_world_translation():
    stage = body_stage()
    capsule = UsdGeom.Capsule.Define(stage, "/Robot/torso/capsule")
    capsule.CreateRadiusAttr(0.07)
    capsule.CreateHeightAttr(0.1)
    capsule.CreateAxisAttr("Y")
    UsdGeom.Xformable(capsule).AddTranslateOp().Set((0.1, 0, 0.4))
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    inner, outer, rows = extract_native_geometry(stage)
    assert np.allclose(inner["torso"][0].start, [0.1, -0.05, 0.4])
    assert np.allclose(inner["torso"][0].end, [0.1, 0.05, 0.4])
    assert inner == outer
    assert rows[0]["role"] == "native_primitive_subset"


def test_collision_transform_mesh_is_outer_only():
    stage = body_stage()
    sphere = UsdGeom.Sphere.Define(stage, "/Robot/torso/sphere")
    sphere.CreateRadiusAttr(0.03)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    parent = UsdGeom.Xform.Define(stage, "/Robot/torso/hand")
    UsdPhysics.CollisionAPI.Apply(parent.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Robot/torso/hand/mesh")
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [0.1, 0.5, 0.3]])
    mesh.CreatePointsAttr(points)
    inner, outer, rows = extract_native_geometry(stage)
    assert len(inner["torso"]) == 1 and len(outer["torso"]) == 2
    bound = next(r for r in rows if r["role"] == "outer_mesh_sphere")
    assert np.all(np.linalg.norm(points - bound["start"], axis=1) <= bound["radius"] + 1e-8)


def test_nonuniform_native_primitive_cannot_be_treated_as_capsule():
    stage = body_stage()
    sphere = UsdGeom.Sphere.Define(stage, "/Robot/torso/sphere")
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdGeom.Xformable(sphere).AddScaleOp().Set((1, 2, 1))
    with pytest.raises(ValueError, match="nonuniform"):
        extract_native_geometry(stage)


def test_enclosing_sphere_contains_vertices_and_convex_combinations():
    points = np.array([[-1, 2, 3], [5, -2, 0], [-2, 0, 4]])
    center, radius = enclosing_sphere(points)
    assert np.all(np.linalg.norm(points - center, axis=1) <= radius)
    assert np.linalg.norm(points.mean(0) - center) <= radius
