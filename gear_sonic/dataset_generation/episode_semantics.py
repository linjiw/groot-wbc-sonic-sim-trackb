"""Turn an episode's geometry into a goal object and a task instruction.

A VLA benchmark needs episodes conditioned on language that refers to the scene. Today the
corpus's task strings are scene-level and hand-written ("walk forward through the household
room"), which a policy can satisfy without looking at anything. This module derives the
language from the episode's own geometry instead: which object the route ends at, which
objects it passes between, and what the walk did.

Three decisions worth stating, because each one is a way this could have been wrong:

* **The goal object is the nearest piece to the endpoint, not the nearest to the path.**
  A route brushes many objects; only one is where it stopped. Nearest-to-endpoint is the
  only reading under which "stop at the X" is true of the episode.
* **A goal is refused when nothing is close enough.** An episode ending in open floor has
  no goal object, and inventing one would put a false referent in the training language.
  The instruction falls back to a form that makes no object claim.
* **Furniture names are mapped to words people use.** The catalog says ``LowCrate`` and
  ``PetBowl``; instructions say "low crate" and "pet bowl". The class name is kept
  alongside so the semantic label stays machine-readable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np

#: Beyond this the endpoint is in open floor and no object is the goal.
DEFAULT_GOAL_RADIUS_M = 1.5

#: How close a piece must come to the route to count as something passed.
DEFAULT_PASSED_RADIUS_M = 1.0

#: Catalog names whose plain-English form is not just the split camel case.
_SPOKEN_NAMES = {
    "PlantPot": "potted plant",
    "PetBowl": "pet bowl",
    "LowCrate": "low crate",
    "StorageBox": "storage box",
    "WallShelf": "wall shelf",
    "WallCabinet": "wall cabinet",
    "CeilingLamp": "ceiling lamp",
    "FloorLamp": "floor lamp",
    "CoffeeTable": "coffee table",
    "DiningTable": "dining table",
}


class SemanticsError(ValueError):
    """Raised when episode semantics cannot be derived from the inputs given."""


@dataclass(frozen=True)
class SceneObject:
    """One placed solid, as the semantics layer needs it."""

    name: str
    rect: tuple[float, float, float, float]
    z_base: float = 0.0

    @property
    def kind(self) -> str:
        """Catalog class, recovered by dropping the ``_NN`` placement suffix."""
        return re.sub(r"_\d+$", "", self.name)

    @property
    def spoken(self) -> str:
        kind = self.kind
        if kind in _SPOKEN_NAMES:
            return _SPOKEN_NAMES[kind]
        # CamelCase -> spaced lower case, so an unlisted catalog entry still reads.
        return re.sub(r"(?<!^)(?=[A-Z])", " ", kind).lower()

    @property
    def center(self) -> tuple[float, float]:
        left, bottom, right, top = self.rect
        return ((left + right) / 2.0, (bottom + top) / 2.0)

    def distance_to(self, point: Sequence[float]) -> float:
        """Distance from a point to this footprint; zero if the point is inside it."""
        left, bottom, right, top = self.rect
        dx = max(left - point[0], point[0] - right, 0.0)
        dy = max(bottom - point[1], point[1] - top, 0.0)
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class EpisodeSemantics:
    """The language annotation for one episode, plus the facts behind it."""

    task: str
    goal_object: str | None
    goal_object_kind: str | None
    goal_distance_m: float | None
    passed_objects: tuple[str, ...]
    #: Named motion style, from the prompt taxonomy's body mode where known.
    motion_style: str

    @property
    def has_goal(self) -> bool:
        return self.goal_object is not None


def _describe_motion(path_xy: np.ndarray, root_z: np.ndarray | None) -> str:
    """A short phrase for what the walk did, from the trajectory itself."""
    steps = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    length = float(steps.sum())
    displacement = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    if length <= 1e-6:
        return "stand still"

    deltas = np.diff(path_xy, axis=0)
    moving = deltas[np.linalg.norm(deltas, axis=1) > 1e-6]
    turned = False
    if len(moving) >= 2:
        headings = np.unwrap(np.arctan2(moving[:, 1], moving[:, 0]))
        turned = abs(float(headings[-1] - headings[0])) > math.radians(35)

    crouched = root_z is not None and len(root_z) and float(np.min(root_z)) < 0.55

    if crouched:
        return "crouch low and move forward"
    if turned:
        return "walk forward and turn"
    if displacement < 0.5 * length:
        return "walk a winding path"
    return "walk forward"


def derive_semantics(
    path_xy: np.ndarray,
    objects: Sequence[SceneObject],
    *,
    root_z: np.ndarray | None = None,
    motion_style: str | None = None,
    goal_radius_m: float = DEFAULT_GOAL_RADIUS_M,
    passed_radius_m: float = DEFAULT_PASSED_RADIUS_M,
    max_passed: int = 2,
) -> EpisodeSemantics:
    """Derive the goal object and a task instruction from the episode's geometry."""
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < 2:
        raise SemanticsError("need at least two path points to describe an episode")

    endpoint = path[-1]
    ranked = sorted(objects, key=lambda o: o.distance_to(endpoint))
    goal = ranked[0] if ranked and ranked[0].distance_to(endpoint) <= goal_radius_m else None

    # Objects near the route but not the goal, ordered by how close they came.
    passed: list[SceneObject] = []
    for obstacle in objects:
        if goal is not None and obstacle.name == goal.name:
            continue
        nearest = min(obstacle.distance_to(point) for point in path)
        if nearest <= passed_radius_m:
            passed.append(obstacle)
    passed.sort(key=lambda o: min(o.distance_to(point) for point in path))
    passed = passed[:max_passed]

    verb = motion_style_phrase(motion_style) if motion_style else _describe_motion(path, root_z)

    if goal is not None:
        task = f"{verb} and stop at the {goal.spoken}"
    elif passed:
        task = f"{verb} past the {passed[0].spoken}"
    else:
        # No object claim at all rather than a fabricated referent.
        task = f"{verb} across the room"

    if len(passed) >= 2 and goal is not None:
        task = f"{verb} between the {passed[0].spoken} and the {passed[1].spoken}, " \
               f"then stop at the {goal.spoken}"

    return EpisodeSemantics(
        task=task,
        goal_object=goal.name if goal else None,
        goal_object_kind=goal.kind if goal else None,
        goal_distance_m=goal.distance_to(endpoint) if goal else None,
        passed_objects=tuple(o.name for o in passed),
        motion_style=motion_style or "derived",
    )


#: Phrasing for the prompt taxonomy's body modes, so the instruction matches what was asked
#: for rather than only what the trajectory happens to look like.
_STYLE_PHRASES = {
    "walk": "walk forward",
    "walk_pause": "walk forward, pausing on the way",
    "walk_look": "walk forward, looking around on the way",
    "crouch_walk": "lower down slightly and move forward",
    "crouch_deep": "crouch low and move forward",
    "duck_under": "walk forward, ducking low to pass underneath",
    "side_step": "step sideways",
    "arm_tuck": "walk forward, arms held in close to squeeze through the gap",
    "shoulder_turn": "walk forward, turning sideways to slip through the opening",
    "backward": "walk backwards",
    "stand_to_walk": "start walking from a standstill",
    "walk_to_stop": "walk forward and come to a stop",
    "step_over": "walk forward, stepping over what is in the way",
    "turn_in_place": "turn around, then walk forward",
    "carry_walk": "walk forward carrying something",
    "reach_walk": "walk forward, then reach out",
    "squat_pick": "walk forward, then squat down to pick something up",
}


def motion_style_phrase(style: str) -> str:
    """Instruction phrasing for a taxonomy body mode."""
    if style not in _STYLE_PHRASES:
        raise SemanticsError(
            f"unknown motion style {style!r}; known: {sorted(_STYLE_PHRASES)}"
        )
    return _STYLE_PHRASES[style]
