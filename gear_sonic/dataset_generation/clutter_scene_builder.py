"""Build a cluttered indoor scene *around* a known-good motion path.

The inversion that makes this work
----------------------------------

The first approach was: fix a scene, then search for a placement where the motion
fits. That is placement-limited -- most references collide, and the surviving
episodes walk through mostly empty space, which is not the data anyone wants.

Turn it around. The motion is known first, so its swept corridor is known first.
Generate the furniture *around* that corridor: everything is placed outside the
corridor by at least the required clearance, and preferentially just outside it.
The result is a densely cluttered room in which the recorded episode is
traversable **by construction**, with the robot threading between real obstacles
rather than crossing an empty floor.

Guarantees
----------

* Every placed solid clears the executed corridor by ``clearance_m``.
* Solids do not overlap each other or the walls.
* Everything sits inside the room footprint.
* Output is the same primitive USDA subset (``Plane`` floor + axis-aligned
  ``Cube`` solids) that ``scene_asset_preflight`` already validates, so a
  generated scene passes exactly the same gate as a hand-authored one.

Clutter density is reported, not assumed: :class:`ClutterSceneSpec` records how
many solids were placed, how close the nearest one comes to the path, and what
fraction of the free floor they occupy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "FURNITURE_CATALOG",
    "ClutterSceneSpec",
    "FurniturePiece",
    "build_clutter_scene",
    "render_scene_usda",
]


@dataclass(frozen=True)
class FurnitureKind:
    """A furniture archetype with plausible dimensions and colour."""

    name: str
    #: (x, y, z) extent ranges in metres.
    size_min: tuple[float, float, float]
    size_max: tuple[float, float, float]
    color: tuple[float, float, float]
    #: Whether the piece may be rotated 90 degrees about +Z (axis-aligned only).
    allow_rotation: bool = True


#: Household/office archetypes. Heights matter: anything reaching above ~0.3 m is a
#: real obstacle for a walking G1, and the preflight only counts solids whose
#: vertical extent overlaps the robot band, so low rugs would not register.
FURNITURE_CATALOG: tuple[FurnitureKind, ...] = (
    FurnitureKind("Sofa", (1.8, 0.8, 0.75), (2.3, 0.95, 0.85), (0.35, 0.38, 0.45)),
    FurnitureKind("Armchair", (0.75, 0.75, 0.75), (0.95, 0.95, 0.95), (0.42, 0.32, 0.28)),
    FurnitureKind("CoffeeTable", (0.9, 0.5, 0.40), (1.3, 0.7, 0.48), (0.45, 0.33, 0.22)),
    FurnitureKind("DiningTable", (1.4, 0.85, 0.72), (1.9, 1.05, 0.78), (0.40, 0.29, 0.20)),
    FurnitureKind("Chair", (0.45, 0.45, 0.85), (0.55, 0.55, 1.0), (0.38, 0.27, 0.19)),
    FurnitureKind("Bookshelf", (0.9, 0.32, 1.7), (1.4, 0.42, 2.1), (0.36, 0.26, 0.18)),
    FurnitureKind("Cabinet", (0.8, 0.45, 0.9), (1.2, 0.6, 1.2), (0.50, 0.44, 0.36)),
    FurnitureKind("Counter", (1.6, 0.62, 0.9), (2.4, 0.7, 0.95), (0.55, 0.52, 0.48)),
    FurnitureKind("Fridge", (0.7, 0.7, 1.7), (0.85, 0.8, 1.9), (0.72, 0.73, 0.75)),
    FurnitureKind("StorageBox", (0.4, 0.4, 0.35), (0.7, 0.6, 0.55), (0.58, 0.45, 0.28)),
    FurnitureKind("PlantPot", (0.35, 0.35, 0.8), (0.5, 0.5, 1.2), (0.24, 0.36, 0.24)),
    FurnitureKind("FloorLamp", (0.3, 0.3, 1.4), (0.4, 0.4, 1.7), (0.30, 0.30, 0.33)),
)


@dataclass(frozen=True)
class FurniturePiece:
    """One placed solid."""

    name: str
    kind: str
    center_xy: tuple[float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]

    @property
    def rect(self) -> tuple[float, float, float, float]:
        half_x, half_y = self.size[0] / 2.0, self.size[1] / 2.0
        return (
            self.center_xy[0] - half_x,
            self.center_xy[1] - half_y,
            self.center_xy[0] + half_x,
            self.center_xy[1] + half_y,
        )


@dataclass
class ClutterSceneSpec:
    """A generated scene plus the density evidence a reviewer needs."""

    scene_id: str
    split_group: str
    room_size_xy: tuple[float, float]
    wall_height: float
    pieces: list[FurniturePiece]
    path_xy: np.ndarray
    clearance_m: float
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def walkable_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half_x, half_y = self.room_size_xy[0] / 2.0, self.room_size_xy[1] / 2.0
        return (-half_x, -half_y), (half_x, half_y)


def _segment_point_distance(
    point: tuple[float, float], start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq <= 1e-12:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))
    t = max(0.0, min(1.0, float((np.asarray(point) - start) @ segment) / length_sq))
    closest = start + t * segment
    return float(np.hypot(point[0] - closest[0], point[1] - closest[1]))


def _rect_to_path_distance(rect: tuple[float, float, float, float], path: np.ndarray) -> float:
    """Minimum distance from an axis-aligned rectangle to a polyline.

    Exact for this case: the minimum is attained either at a rectangle corner
    against a path segment, or at a path vertex against the rectangle.
    """
    min_x, min_y, max_x, max_y = rect
    corners = ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
    best = math.inf
    for start, end in zip(path[:-1], path[1:]):
        for corner in corners:
            best = min(best, _segment_point_distance(corner, start, end))
    for point in path:
        inside_x = min_x <= point[0] <= max_x
        inside_y = min_y <= point[1] <= max_y
        if inside_x and inside_y:
            return 0.0
        dx = max(min_x - point[0], 0.0, point[0] - max_x)
        dy = max(min_y - point[1], 0.0, point[1] - max_y)
        best = min(best, float(math.hypot(dx, dy)))
    return best


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float
) -> bool:
    return not (
        a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1]
    )


def build_clutter_scene(
    path_xy: np.ndarray,
    *,
    scene_id: str,
    seed: int = 0,
    clearance_m: float = 0.75,
    room_margin_m: float = 1.5,
    min_room_size_m: tuple[float, float] = (7.0, 6.0),
    wall_height: float = 2.8,
    target_pieces: int = 22,
    max_distance_from_path_m: float = 3.0,
    piece_gap_m: float = 0.15,
    attempts_per_piece: int = 400,
    catalog: Sequence[FurnitureKind] = FURNITURE_CATALOG,
) -> ClutterSceneSpec:
    """Populate a room around ``path_xy`` with furniture that never blocks it.

    Args:
        path_xy: ``(T, 2)`` scene-frame path the robot will actually walk.
        clearance_m: Minimum distance from any solid to the path. Should be the
            body radius plus the tracking margin.
        room_margin_m: How far the walls sit beyond the path extent.
        target_pieces: How many solids to try to place.
        max_distance_from_path_m: Solids are placed within this band of the path,
            so the corridor is genuinely enclosed rather than merely non-blocking.
        piece_gap_m: Minimum gap between solids.
    """
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError(f"path_xy must be (T>=2, 2); got {path.shape}")
    if clearance_m <= 0:
        raise ValueError("clearance_m must be positive")

    centre = (path.max(axis=0) + path.min(axis=0)) / 2.0
    span = path.max(axis=0) - path.min(axis=0)
    room_x = max(float(span[0]) + 2.0 * room_margin_m, min_room_size_m[0])
    room_y = max(float(span[1]) + 2.0 * room_margin_m, min_room_size_m[1])
    # Work in a scene frame centred on the path so the room is symmetric about it.
    path = path - centre
    half_x, half_y = room_x / 2.0, room_y / 2.0

    rng = np.random.default_rng(seed)
    wall_rects = [
        (-half_x - 0.2, -half_y - 0.2, -half_x, half_y + 0.2),
        (half_x, -half_y - 0.2, half_x + 0.2, half_y + 0.2),
        (-half_x - 0.2, -half_y - 0.2, half_x + 0.2, -half_y),
        (-half_x - 0.2, half_y, half_x + 0.2, half_y + 0.2),
    ]
    placed: list[FurniturePiece] = []
    placed_rects: list[tuple[float, float, float, float]] = list(wall_rects)

    for index in range(target_pieces):
        kind = catalog[int(rng.integers(len(catalog)))]
        for _ in range(attempts_per_piece):
            size = tuple(
                float(rng.uniform(low, high))
                for low, high in zip(kind.size_min, kind.size_max)
            )
            if kind.allow_rotation and rng.random() < 0.5:
                size = (size[1], size[0], size[2])
            # Sample near the path, then verify, rather than sampling the whole room:
            # the goal is an enclosed corridor, not a sparsely furnished hall.
            anchor = path[int(rng.integers(len(path)))]
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(clearance_m, max_distance_from_path_m)
            centre_xy = (
                float(anchor[0] + radius * math.cos(angle)),
                float(anchor[1] + radius * math.sin(angle)),
            )
            candidate = FurniturePiece(
                name=f"{kind.name}_{index:02d}",
                kind=kind.name,
                center_xy=centre_xy,
                size=size,  # type: ignore[arg-type]
                color=kind.color,
            )
            rect = candidate.rect
            if not (
                -half_x + 0.05 <= rect[0]
                and rect[2] <= half_x - 0.05
                and -half_y + 0.05 <= rect[1]
                and rect[3] <= half_y - 0.05
            ):
                continue
            if _rect_to_path_distance(rect, path) < clearance_m:
                continue
            if any(_rects_overlap(rect, other, piece_gap_m) for other in placed_rects):
                continue
            placed.append(candidate)
            placed_rects.append(rect)
            break

    distances = [_rect_to_path_distance(piece.rect, path) for piece in placed]
    floor_area = room_x * room_y
    occupied = sum(piece.size[0] * piece.size[1] for piece in placed)
    spec = ClutterSceneSpec(
        scene_id=scene_id,
        split_group=f"{scene_id}_layout_v1",
        room_size_xy=(room_x, room_y),
        wall_height=wall_height,
        pieces=placed,
        path_xy=path,
        clearance_m=clearance_m,
        seed=seed,
        metrics={
            "requested_pieces": target_pieces,
            "placed_pieces": len(placed),
            "min_distance_to_path_m": float(min(distances)) if distances else None,
            "median_distance_to_path_m": float(np.median(distances)) if distances else None,
            "floor_area_m2": floor_area,
            "occupied_area_m2": occupied,
            "clutter_occupancy": occupied / floor_area if floor_area else 0.0,
            "path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "path_offset_applied_xy": [float(-centre[0]), float(-centre[1])],
        },
    )
    return spec


def _cube(name: str, role: str, centre_xyz, size_xyz, color) -> str:
    return f"""
        def Cube "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            custom string g1Dataset:role = "{role}"
            double size = 1
            bool physics:collisionEnabled = 1
            color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]
            double3 xformOp:scale = ({size_xyz[0]:.4f}, {size_xyz[1]:.4f}, {size_xyz[2]:.4f})
            double3 xformOp:translate = ({centre_xyz[0]:.4f}, {centre_xyz[1]:.4f}, {centre_xyz[2]:.4f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
"""


def render_scene_usda(spec: ClutterSceneSpec) -> str:
    """Serialise a spec to the primitive USDA subset the preflight validates."""
    half_x, half_y = spec.room_size_xy[0] / 2.0, spec.room_size_xy[1] / 2.0
    height = spec.wall_height
    body = [
        f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    timeCodesPerSecond = 60
    upAxis = "Z"
)

# Procedurally generated by gear_sonic.dataset_generation.clutter_scene_builder.
# Furniture is placed AROUND a known-good motion path: every solid clears the
# executed corridor by at least {spec.clearance_m:.2f} m, so the recorded episode is
# traversable by construction while the room stays densely furnished.
# seed={spec.seed}  pieces={len(spec.pieces)}  occupancy={spec.metrics['clutter_occupancy']:.3f}
def Xform "World" (
    kind = "assembly"
)
{{
    custom string g1Dataset:sceneId = "{spec.scene_id}"
    custom string g1Dataset:splitGroup = "{spec.split_group}"

    def Xform "Structure"
    {{
        def Plane "Floor" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            custom string g1Dataset:role = "support_floor"
            uniform token axis = "Z"
            bool doubleSided = true
            double width = {spec.room_size_xy[0]:.3f}
            double length = {spec.room_size_xy[1]:.3f}
            bool physics:collisionEnabled = 1
            color3f[] primvars:displayColor = [(0.42, 0.36, 0.29)]
        }}
"""
    ]
    wall_color = (0.82, 0.80, 0.74)
    body.append(
        _cube("WallWest", "wall", (-half_x - 0.05, 0.0, height / 2), (0.1, 2 * half_y + 0.2, height), wall_color)
    )
    body.append(
        _cube("WallEast", "wall", (half_x + 0.05, 0.0, height / 2), (0.1, 2 * half_y + 0.2, height), wall_color)
    )
    body.append(
        _cube("WallSouth", "wall", (0.0, -half_y - 0.05, height / 2), (2 * half_x + 0.2, 0.1, height), wall_color)
    )
    body.append(
        _cube("WallNorth", "wall", (0.0, half_y + 0.05, height / 2), (2 * half_x + 0.2, 0.1, height), wall_color)
    )
    body.append("    }\n\n    def Xform \"Clutter\"\n    {\n")
    for piece in spec.pieces:
        body.append(
            _cube(
                piece.name,
                "clutter",
                (piece.center_xy[0], piece.center_xy[1], piece.size[2] / 2.0),
                piece.size,
                piece.color,
            )
        )
    body.append("    }\n}\n")
    return "".join(body)
