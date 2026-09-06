"""Read composed USD collision declarations, including meshes below collision Xforms.

This is an asset inventory; cooked PhysX shape agreement still requires a running audit.
"""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def collision_inventory(stage):
    """Snapshot authored/composed native collision shapes and settings at capture start."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        scene = prim.IsA(UsdPhysics.Scene)
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ancestor = prim.GetParent()
        collision_ancestor = None
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                collision_ancestor = str(ancestor.GetPath())
                break
            ancestor = ancestor.GetParent()
        geometry_child = bool(collision_ancestor) and prim.IsA(UsdGeom.Gprim)
        filtering = prim.HasAPI(UsdPhysics.FilteredPairsAPI) or prim.IsA(UsdPhysics.CollisionGroup)
        if not (collider or scene or rigid or geometry_child or filtering):
            continue
        attributes = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(("physics:", "physx")) or name in (
                "size",
                "radius",
                "height",
                "axis",
                "points",
                "faceVertexCounts",
                "faceVertexIndices",
            ):
                value = attr.Get()
                if value is not None and name in (
                    "points",
                    "faceVertexCounts",
                    "faceVertexIndices",
                ):
                    attributes[name] = np.asarray(value).tolist()
                else:
                    attributes[name] = str(value) if value is not None else None
        rows.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "schemas": list(prim.GetAppliedSchemas()),
                "collision": collider,
                "collision_ancestor": collision_ancestor,
                "instance_proxy": prim.IsInstanceProxy(),
                "rigid_body": rigid,
                "attributes": attributes,
                "relationships": {
                    r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
                },
                "local_to_world_at_capture_start": np.array(
                    cache.GetLocalToWorldTransform(prim)
                ).tolist(),
            }
        )
    return rows
