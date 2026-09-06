"""Cached USD primitive subset and enclosing mesh spheres in rigid-body coordinates."""

import hashlib

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from gear_sonic.dataset_generation.swept_volume import CollisionCapsule


def collision_owner(prim):
    """Find the nearest authored collision API and the owning rigid body."""
    collision, body = None, None
    current = prim
    while current and not current.IsPseudoRoot():
        if collision is None and current.HasAPI(UsdPhysics.CollisionAPI):
            collision = current
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            body = current
            break
        current = current.GetParent()
    return collision, body


def enclosing_sphere(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("requires finite nonempty mesh vertices")
    center = (points.min(0) + points.max(0)) / 2
    radius = np.nextafter(np.linalg.norm(points - center, axis=1).max(), np.inf)
    return center, float(radius)


def extract_native_geometry(stage):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    inner, outer, rows = {}, {}, []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        collision, body = collision_owner(prim)
        if (
            collision is None
            or not UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        ):
            continue
        if body is None:
            raise ValueError(f"collision shape lacks rigid owner: {prim.GetPath()}")
        local = np.asarray(
            cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
        )
        if not np.isfinite(local).all():
            raise ValueError("nonfinite USD transform")
        owner = body.GetName()
        kind = prim.GetTypeName()
        row = {
            "path": str(prim.GetPath()),
            "owner": owner,
            "type": kind,
            "collision_api_path": str(collision.GetPath()),
            "local_transform": local.tolist(),
        }

        def transform(points):
            return np.asarray(points) @ local[:3, :3] + local[3, :3]

        if kind in ("Capsule", "Sphere"):
            scale = np.linalg.svd(local[:3, :3], compute_uv=False)
            if scale.min() <= 0 or not np.allclose(scale, scale[0], rtol=1e-6, atol=1e-9):
                raise ValueError(f"nonuniform primitive scale: {prim.GetPath()}")
            radius = float(prim.GetAttribute("radius").Get()) * float(scale[0])
            points = np.zeros((2, 3))
            if kind == "Capsule":
                axis = str(prim.GetAttribute("axis").Get())
                height = float(prim.GetAttribute("height").Get())
                points[:, "XYZ".index(axis)] = [-height / 2, height / 2]
            points = transform(points)
            capsule = CollisionCapsule(tuple(points[0]), tuple(points[1]), radius)
            inner.setdefault(owner, []).append(capsule)
            row["role"] = "native_primitive_subset"
        elif kind == "Mesh":
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            center, radius = enclosing_sphere(transform(vertices))
            capsule = CollisionCapsule(tuple(center), tuple(center), radius)
            row.update(
                role="outer_mesh_sphere",
                vertices=len(vertices),
                vertex_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
            )
        else:
            raise ValueError(f"unsupported collision shape: {kind} {prim.GetPath()}")
        outer.setdefault(owner, []).append(capsule)
        row.update(start=list(capsule.start), end=list(capsule.end), radius=capsule.radius)
        rows.append(row)
    if not inner or not outer:
        raise ValueError("empty cached collision geometry")
    return inner, outer, rows
