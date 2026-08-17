"""Turn a released scene into instructions for rebuilding it on a lab floor.

The corpus's scenes are axis-aligned boxes with recorded dimensions, which means a
scene package is not merely a description of a room -- it is sufficient to reconstruct
that room physically. This module is what makes that claim checkable: it reads the
released ``.usda`` (the artifact others download, not an internal object) and emits a
tape-measure layout plus a per-piece placement table.

Two coordinate frames appear here, and confusing them is the obvious way to build the
wrong room:

* **scene frame** -- what the USDA and the trajectories use. Origin at the room centre,
  so coordinates are signed.
* **tape frame** -- origin at the room's minimum corner, +x and +y running along two
  walls. Every coordinate is positive, which is what a person with a tape measure and a
  roll of masking tape can actually lay out.

The per-piece ``placement_tolerance_m`` is the honest part: it says how far a piece may
drift *toward the walking corridor* before it eats into the clearance the simulated
episode relied on. A builder who stays inside that number has rebuilt a room the recorded
motion still fits through; one who does not has built a different room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Sequence

import numpy as np

#: Roles the USDA writers stamp onto prims via ``custom string g1Dataset:role``.
FLOOR_ROLE = "support_floor"
WALL_ROLE = "wall"
CLUTTER_ROLE = "clutter"

#: Roles that are the room itself rather than something placed inside it. Everything else
#: is a build item. Selecting by exclusion rather than by an allow-list is deliberate:
#: generated scenes label every solid "clutter", while the hand-authored scenes use
#: semantic roles (furniture, storage_rack, pallet, freight, interior_wall...). An
#: allow-list silently emits an empty sheet for the scenes it does not know about, which
#: reads exactly like a room with no furniture in it.
STRUCTURAL_ROLES = frozenset({FLOOR_ROLE, WALL_ROLE})

_PRIM_PATTERN = re.compile(
    r'def\s+(?P<type>Cube|Plane)\s+"(?P<name>[^"]+)"\s*\((?:[^)]*)\)\s*\{(?P<body>[^}]*)\}',
    re.DOTALL,
)
_ROLE_PATTERN = re.compile(r'g1Dataset:role\s*=\s*"([^"]+)"')
_SCALE_PATTERN = re.compile(r"xformOp:scale\s*=\s*\(([^)]*)\)")
_TRANSLATE_PATTERN = re.compile(r"xformOp:translate\s*=\s*\(([^)]*)\)")
_WIDTH_PATTERN = re.compile(r"double\s+width\s*=\s*([-\d.eE]+)")
_LENGTH_PATTERN = re.compile(r"double\s+length\s*=\s*([-\d.eE]+)")
_SCENE_ID_PATTERN = re.compile(r'g1Dataset:sceneId\s*=\s*"([^"]+)"')


class BuildSheetError(ValueError):
    """Raised when a scene cannot be turned into a physical build sheet."""


@dataclass(frozen=True)
class ScenePrim:
    """One box parsed out of the released USDA, in the scene frame."""

    name: str
    role: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """Footprint ``(min_x, min_y, max_x, max_y)``."""
        return (
            self.center[0] - self.size[0] / 2.0,
            self.center[1] - self.size[1] / 2.0,
            self.center[0] + self.size[0] / 2.0,
            self.center[1] + self.size[1] / 2.0,
        )

    @property
    def z_base(self) -> float:
        """Height of the underside. Non-zero means the piece is off the floor."""
        return self.center[2] - self.size[2] / 2.0

    @property
    def is_raised(self) -> bool:
        # A millimetre of float noise should not turn a floor-standing crate into a
        # piece that a builder thinks needs a stand underneath it.
        return self.z_base > 1e-3


@dataclass(frozen=True)
class BuildItem:
    """One piece as a person building the room needs it: sizes, tape marks, tolerance."""

    name: str
    #: The scene's own label for this solid ("clutter", "furniture", "storage_rack"...).
    role: str
    footprint_xy_m: tuple[float, float]
    height_m: float
    #: Height of the underside above the floor. Non-zero needs a stand or a wall mount.
    stand_height_m: float
    #: Footprint corner nearest the tape origin, and the centre, both in the tape frame.
    tape_corner_m: tuple[float, float]
    tape_center_m: tuple[float, float]
    #: Clear horizontal distance from this footprint to the walking route.
    distance_to_route_m: float
    #: How far this piece may drift toward the route before eating the clearance the
    #: simulated episode relied on. Negative means the piece overhangs the corridor and
    #: is only passable because the robot goes underneath it.
    placement_tolerance_m: float

    @property
    def is_over_corridor(self) -> bool:
        return self.placement_tolerance_m < 0.0


@dataclass
class BuildSheet:
    """Everything needed to lay out one scene on a floor, with its own provenance."""

    scene_id: str
    package_id: str
    source_file: str
    room_size_xy_m: tuple[float, float]
    #: Scene-frame coordinates of the tape origin, i.e. the room's minimum corner.
    tape_origin_scene_m: tuple[float, float]
    items: list[BuildItem]
    route_tape_m: np.ndarray
    route_clearance_radius_m: float
    robot_clearance_height_m: float
    #: Lowest underside among raised pieces -- the headroom a walker actually gets.
    lowest_overhead_m: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def floor_items(self) -> list[BuildItem]:
        return [item for item in self.items if item.stand_height_m <= 1e-3]

    @property
    def raised_items(self) -> list[BuildItem]:
        return [item for item in self.items if item.stand_height_m > 1e-3]

    @property
    def tightest_tolerance_m(self) -> float | None:
        tolerances = [i.placement_tolerance_m for i in self.items if not i.is_over_corridor]
        return min(tolerances) if tolerances else None


def parse_scene_usda(text: str) -> list[ScenePrim]:
    """Parse the primitive subset our own writer emits.

    This is deliberately a parser for *our* USDA rather than a general USD reader: the
    build sheet must be derivable from the released file with no Isaac Sim installed,
    which is the whole point of shipping primitive geometry.
    """
    prims: list[ScenePrim] = []
    for match in _PRIM_PATTERN.finditer(text):
        body = match.group("body")
        role_match = _ROLE_PATTERN.search(body)
        if role_match is None:
            continue
        role = role_match.group(1)
        if match.group("type") == "Plane":
            width = _WIDTH_PATTERN.search(body)
            length = _LENGTH_PATTERN.search(body)
            if width is None or length is None:
                raise BuildSheetError(f"plane {match.group('name')!r} is missing width/length")
            size = (float(width.group(1)), float(length.group(1)), 0.0)
            center = (0.0, 0.0, 0.0)
        else:
            scale = _SCALE_PATTERN.search(body)
            translate = _TRANSLATE_PATTERN.search(body)
            if scale is None or translate is None:
                raise BuildSheetError(f"cube {match.group('name')!r} is missing scale/translate")
            size = tuple(float(v) for v in scale.group(1).split(","))  # type: ignore[assignment]
            center = tuple(float(v) for v in translate.group(1).split(","))  # type: ignore[assignment]
            if len(size) != 3 or len(center) != 3:
                raise BuildSheetError(f"cube {match.group('name')!r} has a non-3D transform")
        prims.append(ScenePrim(name=match.group("name"), role=role, center=center, size=size))
    if not prims:
        raise BuildSheetError("no prims with a g1Dataset:role found -- is this our USDA?")
    return prims


def scene_id_from_usda(text: str) -> str | None:
    match = _SCENE_ID_PATTERN.search(text)
    return match.group(1) if match else None


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    length_squared = float(span @ span)
    if length_squared <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ span / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _segment_intersects_rect(
    start: np.ndarray, end: np.ndarray, rect: Sequence[float]
) -> bool:
    """Liang-Barsky slab clip: does the segment touch the axis-aligned rectangle?

    Needed because neither of the two cheap distance families detects a crossing. A route
    that runs straight through a footprint has every vertex outside the rectangle and
    every rectangle corner off the line, so both families report the offset to the nearest
    edge instead of zero -- which would quietly report clearance for a piece standing in
    the robot's path.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in rect)
    delta = end - start
    t_enter, t_exit = 0.0, 1.0
    for direction, origin, low, high in (
        (float(delta[0]), float(start[0]), min_x, max_x),
        (float(delta[1]), float(start[1]), min_y, max_y),
    ):
        if abs(direction) < 1e-15:
            if origin < low or origin > high:
                return False
            continue
        t_low = (low - origin) / direction
        t_high = (high - origin) / direction
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        t_enter = max(t_enter, t_low)
        t_exit = min(t_exit, t_high)
        if t_enter > t_exit:
            return False
    return True


def rect_distance_to_route(rect: Sequence[float], route_xy: np.ndarray) -> float:
    """Clear horizontal distance from an axis-aligned footprint to a polyline.

    Zero when the route touches or crosses the footprint. Three cases must all be
    covered: a route vertex inside or beside the rectangle, a rectangle corner beside a
    route segment, and a segment passing clean through the rectangle without either
    endpoint or corner being close to anything.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in rect)
    route = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    if route.shape[0] == 0:
        raise BuildSheetError("route is empty")

    best = math.inf
    for point in route:
        clamped_x = min(max(point[0], min_x), max_x)
        clamped_y = min(max(point[1], min_y), max_y)
        best = min(best, math.hypot(point[0] - clamped_x, point[1] - clamped_y))
        if best <= 0.0:
            return 0.0

    corners = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=np.float64
    )
    if route.shape[0] == 1:
        for corner in corners:
            best = min(best, float(np.linalg.norm(corner - route[0])))
        return best

    for start, end in zip(route[:-1], route[1:]):
        if _segment_intersects_rect(start, end, rect):
            return 0.0
        for corner in corners:
            best = min(best, _segment_distance(corner, start, end))
    return best


def build_sheet_from_scene(
    usda_text: str,
    *,
    scene_entry: dict,
    package_id: str,
    exclude_roles: Iterable[str] = STRUCTURAL_ROLES,
) -> BuildSheet:
    """Assemble the build sheet for one scene of a released package.

    ``scene_entry`` is the scene's record from the package ``manifest.json``; the route
    and clearance come from there rather than being re-derived, so the sheet describes
    the corridor the episode was actually validated against.
    """
    prims = parse_scene_usda(usda_text)
    skipped = set(exclude_roles)

    floors = [p for p in prims if p.role == FLOOR_ROLE]
    if not floors:
        raise BuildSheetError("scene has no support_floor prim")
    floor = floors[0]
    room_x, room_y = float(floor.size[0]), float(floor.size[1])
    origin = (-room_x / 2.0, -room_y / 2.0)

    route_scene = np.asarray(scene_entry["route_xy"], dtype=np.float64).reshape(-1, 2)
    clearance = float(scene_entry.get("route_clearance_radius_m", 0.0))

    items: list[BuildItem] = []
    for prim in prims:
        if prim.role in skipped:
            continue
        distance = rect_distance_to_route(prim.rect, route_scene)
        min_x, min_y, _, _ = prim.rect
        items.append(
            BuildItem(
                name=prim.name,
                role=prim.role,
                footprint_xy_m=(prim.size[0], prim.size[1]),
                height_m=prim.size[2],
                stand_height_m=max(prim.z_base, 0.0),
                tape_corner_m=(min_x - origin[0], min_y - origin[1]),
                tape_center_m=(prim.center[0] - origin[0], prim.center[1] - origin[1]),
                distance_to_route_m=distance,
                placement_tolerance_m=distance - clearance,
            )
        )
    if not items:
        found = sorted({p.role for p in prims})
        raise BuildSheetError(
            f"scene has no buildable solids after excluding {sorted(skipped)}; roles present: "
            f"{found}. An empty build sheet would read as an empty room."
        )
    items.sort(key=lambda i: (i.stand_height_m > 1e-3, i.tape_corner_m[1], i.tape_corner_m[0]))

    raised = [i for i in items if i.stand_height_m > 1e-3]
    route_tape = route_scene - np.asarray(origin, dtype=np.float64)

    notes: list[str] = []
    over_corridor = [i for i in items if i.is_over_corridor]
    if over_corridor:
        notes.append(
            f"{len(over_corridor)} piece(s) overhang the walking corridor. They are passable "
            "only because the robot goes underneath: their underside height is the safety "
            "dimension, not their footprint."
        )
    tightest = min((i.placement_tolerance_m for i in items if not i.is_over_corridor), default=None)
    if tightest is not None:
        notes.append(
            f"Tightest floor-standing placement tolerance is {tightest:.3f} m. Build to that "
            "or better, measuring toward the corridor."
        )
    return BuildSheet(
        scene_id=scene_entry.get("scene_id") or scene_id_from_usda(usda_text) or "unknown",
        package_id=package_id,
        source_file=scene_entry.get("file", "unknown.usda"),
        room_size_xy_m=(room_x, room_y),
        tape_origin_scene_m=origin,
        items=items,
        route_tape_m=route_tape,
        route_clearance_radius_m=clearance,
        robot_clearance_height_m=float(scene_entry.get("robot_clearance_height_m", 0.0)),
        lowest_overhead_m=min((i.stand_height_m for i in raised), default=None),
        notes=notes,
    )


def render_build_sheet_markdown(sheet: BuildSheet) -> str:
    """Render the sheet as text a person can carry onto the floor."""
    lines: list[str] = []
    lines.append(f"# Build sheet — {sheet.scene_id}")
    lines.append("")
    lines.append(f"Package `{sheet.package_id}` · source `{sheet.source_file}`")
    lines.append("")
    lines.append(
        f"Room **{sheet.room_size_xy_m[0]:.3f} m × {sheet.room_size_xy_m[1]:.3f} m**. "
        f"Tape origin is the minimum corner; in scene coordinates that is "
        f"({sheet.tape_origin_scene_m[0]:.3f}, {sheet.tape_origin_scene_m[1]:.3f}). "
        "All coordinates below are in the tape frame and are positive."
    )
    lines.append("")
    lines.append("## Lay the corridor first")
    lines.append("")
    lines.append(
        f"Tape the route, then the {sheet.route_clearance_radius_m:.3f} m clearance band on "
        "either side of it. Every floor-standing piece goes outside that band."
    )
    lines.append("")
    lines.append("| # | x (m) | y (m) |")
    lines.append("|---|---|---|")
    for index, (x, y) in enumerate(sheet.route_tape_m):
        lines.append(f"| {index} | {x:.3f} | {y:.3f} |")
    lines.append("")

    if sheet.floor_items:
        lines.append("## Floor-standing pieces")
        lines.append("")
        lines.append("| Piece | Role | Footprint (m) | Height (m) | Corner x,y (m) | Tolerance (m) |")
        lines.append("|---|---|---|---|---|---|")
        for item in sheet.floor_items:
            lines.append(
                f"| {item.name} | {item.role} "
                f"| {item.footprint_xy_m[0]:.3f} × {item.footprint_xy_m[1]:.3f} "
                f"| {item.height_m:.3f} | {item.tape_corner_m[0]:.3f}, {item.tape_corner_m[1]:.3f} "
                f"| {item.placement_tolerance_m:.3f} |"
            )
        lines.append("")

    if sheet.raised_items:
        lines.append("## Raised pieces (need a stand or a wall mount)")
        lines.append("")
        lines.append(
            "`Underside` is the dimension that matters: it is the headroom the robot walks "
            "through. Build these to the underside height, not to the footprint."
        )
        lines.append("")
        lines.append("| Piece | Role | Footprint (m) | Height (m) | Underside (m) | Corner x,y (m) |")
        lines.append("|---|---|---|---|---|---|")
        for item in sheet.raised_items:
            lines.append(
                f"| {item.name} | {item.role} "
                f"| {item.footprint_xy_m[0]:.3f} × {item.footprint_xy_m[1]:.3f} "
                f"| {item.height_m:.3f} | **{item.stand_height_m:.3f}** "
                f"| {item.tape_corner_m[0]:.3f}, {item.tape_corner_m[1]:.3f} |"
            )
        lines.append("")

    lines.append("## Before you run the robot")
    lines.append("")
    for note in sheet.notes:
        lines.append(f"- {note}")
    if sheet.lowest_overhead_m is not None:
        lines.append(
            f"- Lowest underside in the room is {sheet.lowest_overhead_m:.3f} m. The simulated "
            f"episode assumed {sheet.robot_clearance_height_m:.3f} m of clearance height overall."
        )
    lines.append(
        "- Walk the taped corridor yourself before powering the robot. If you cannot walk it "
        "without stepping over something, the room is not built to the sheet."
    )
    return "\n".join(lines) + "\n"


def render_build_sheet_svg(sheet: BuildSheet, *, scale_px_per_m: float = 90.0) -> str:
    """Render a printable floor plan: footprints, route, clearance band, tape grid."""
    room_x, room_y = sheet.room_size_xy_m
    pad = 46.0
    width = room_x * scale_px_per_m + 2 * pad
    height = room_y * scale_px_per_m + 2 * pad

    def to_px(x: float, y: float) -> tuple[float, float]:
        # SVG y grows downward; the plan reads with +y up, as a floor plan should.
        return (pad + x * scale_px_per_m, pad + (room_y - y) * scale_px_per_m)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="monospace">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
    ]

    # One-metre tape grid, because the builder measures in metres from the origin.
    for metre in range(0, int(math.floor(room_x)) + 1):
        x0, y0 = to_px(metre, 0.0)
        _, y1 = to_px(metre, room_y)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" '
            'stroke="#dcdcdc" stroke-width="1"/>'
        )
        parts.append(f'<text x="{x0:.1f}" y="{y0 + 16:.1f}" font-size="10" fill="#888" '
                     f'text-anchor="middle">{metre}</text>')
    for metre in range(0, int(math.floor(room_y)) + 1):
        x0, y0 = to_px(0.0, metre)
        x1, _ = to_px(room_x, metre)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" '
            'stroke="#dcdcdc" stroke-width="1"/>'
        )
        parts.append(f'<text x="{x0 - 8:.1f}" y="{y0 + 4:.1f}" font-size="10" fill="#888" '
                     f'text-anchor="end">{metre}</text>')

    x0, y0 = to_px(0.0, room_y)
    parts.append(
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{room_x * scale_px_per_m:.1f}" '
        f'height="{room_y * scale_px_per_m:.1f}" fill="none" stroke="#222" stroke-width="2"/>'
    )

    route = sheet.route_tape_m
    if len(route) >= 2:
        points = " ".join(f"{px:.1f},{py:.1f}" for px, py in (to_px(x, y) for x, y in route))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#2f6fd0" stroke-width="'
            f'{2 * sheet.route_clearance_radius_m * scale_px_per_m:.1f}" stroke-opacity="0.16" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#2f6fd0" stroke-width="2.5" '
            'stroke-dasharray="7 5"/>'
        )

    for item in sheet.items:
        w = item.footprint_xy_m[0] * scale_px_per_m
        h = item.footprint_xy_m[1] * scale_px_per_m
        px, py = to_px(item.tape_corner_m[0], item.tape_corner_m[1] + item.footprint_xy_m[1])
        raised = item.stand_height_m > 1e-3
        fill = "none" if raised else "#e8e2d6"
        dash = ' stroke-dasharray="6 4"' if raised else ""
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
            f'stroke="{"#b06a2c" if raised else "#6b6257"}" stroke-width="1.6"{dash}/>'
        )
        label = f"{item.stand_height_m:.2f}↑" if raised else f"{item.height_m:.2f}"
        parts.append(
            f'<text x="{px + w / 2:.1f}" y="{py + h / 2 + 4:.1f}" font-size="10" '
            f'fill="{"#b06a2c" if raised else "#4a443c"}" text-anchor="middle">{label}</text>'
        )

    ox, oy = to_px(0.0, 0.0)
    parts.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="5" fill="#c0392b"/>')
    parts.append(
        f'<text x="{ox + 9:.1f}" y="{oy - 8:.1f}" font-size="11" fill="#c0392b">tape origin (0,0)</text>'
    )
    parts.append(
        f'<text x="{pad:.1f}" y="{height - 12:.1f}" font-size="11" fill="#333">'
        f'{sheet.scene_id} — solid = floor-standing (height in m), dashed = raised '
        f'(underside height ↑)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
