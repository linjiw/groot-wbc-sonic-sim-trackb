"""Keep policy-specific geometry out of evaluation scenes.

Most scenes in this corpus were generated *around* an executed trajectory: the room is
built so that one recorded motion clears everything, with the margin measured against that
motion's own swept volume. That is exactly right for training data -- the tracking error is
already baked into the clearance, which is why the tight preset passes at 0.12 m. It is
disqualifying for evaluation. A room fitted to one policy's path is collision-free for that
policy and nothing else, so a benchmark built on such scenes measures whether the evaluated
policy reproduces the trajectory the room was cut from, not whether it can navigate.

The rule is therefore: **evaluation scenes must not use executed-swept-volume geometry.**
This module enforces it two ways, because the declarative check and the empirical one fail
differently:

* **Declared provenance.** A scene whose manifest names a ``source_motion`` was generated
  around that motion. Cheap, and catches everything correctly labelled.
* **Independent routability.** A scene that admits no route other than the one it was built
  around is policy-specific whatever its manifest says. This catches mislabelled scenes and
  quantifies *how* specific a scene is.

The second check is not hypothetical. Measured on the density ladder, ``density_dense`` and
``density_tight`` admit no independent 2 m route at 0.60 m clearance after 200 attempts,
while ``density_sparse`` and ``density_moderate`` do. Those two scenes are traversable only
by the trajectory they were cut from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

#: Manifests that were hand-authored rather than fitted to any trajectory.
AUTHORED_PROVENANCE = "repo_authored_primitive_geometry"

#: Generated, but budgeted from the *commanded* path plus an allowance for policy drift, so
#: no executed trajectory enters the geometry. Eligible, and labelled distinctly from
#: hand-authored: calling a generated room "authored" would misdescribe how it was made to
#: anyone reading a split report.
REFERENCE_CLEARANCE_PROVENANCE = "repo_generated_reference_clearance_geometry"


class EvalSceneGateError(ValueError):
    """Raised when a scene set violates the evaluation-geometry rule."""


@dataclass(frozen=True)
class SceneEligibility:
    """Whether one scene may appear in an evaluation split, and why."""

    scene_id: str
    #: "authored", "executed_swept_volume", or "reference_clearance".
    geometry_origin: str
    source_motion: str | None
    #: Independent routes found at the required clearance; None when not tested.
    independent_routes: int | None
    eligible: bool
    reasons: tuple[str, ...]

    @property
    def is_policy_specific(self) -> bool:
        return self.geometry_origin == "executed_swept_volume"


def classify_scene(
    scene_entry: dict, *, independent_routes: int | None = None
) -> SceneEligibility:
    """Classify one manifest entry against the evaluation-geometry rule."""
    scene_id = scene_entry.get("scene_id", "unknown")
    source_motion = scene_entry.get("source_motion")
    provenance = scene_entry.get("provenance", "")

    if provenance == REFERENCE_CLEARANCE_PROVENANCE:
        origin = "reference_clearance"
    elif provenance == AUTHORED_PROVENANCE or source_motion is None:
        origin = "authored"
    else:
        origin = "executed_swept_volume"

    reasons: list[str] = []
    if origin == "executed_swept_volume":
        reasons.append(
            f"geometry was generated around executed motion {source_motion!r}; its "
            "clearance encodes that motion's tracking error"
        )
    # Routability is checked even for authored scenes: a hand-authored room can still be
    # too tight to admit an alternative path, and that matters for the same reason.
    if independent_routes is not None and independent_routes <= 0:
        reasons.append(
            "no independent route exists at the required clearance, so the scene is "
            "traversable only by the trajectory it was built around"
        )

    return SceneEligibility(
        scene_id=scene_id,
        geometry_origin=origin,
        source_motion=source_motion,
        independent_routes=independent_routes,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def assert_eval_scenes_are_eligible(
    scene_entries: Sequence[dict],
    *,
    routes_by_scene: dict[str, int] | None = None,
) -> list[SceneEligibility]:
    """Raise unless every scene may be used for evaluation.

    Meant to be called where an evaluation split is assembled, so the rule is enforced by
    the code that builds the split rather than by a paragraph someone has to remember.
    """
    if not scene_entries:
        raise EvalSceneGateError("refusing to certify an empty evaluation scene set")

    routes = routes_by_scene or {}
    verdicts = [
        classify_scene(entry, independent_routes=routes.get(entry.get("scene_id", "")))
        for entry in scene_entries
    ]
    rejected = [v for v in verdicts if not v.eligible]
    if rejected:
        detail = "; ".join(f"{v.scene_id}: {v.reasons[0]}" for v in rejected)
        raise EvalSceneGateError(
            f"{len(rejected)} of {len(verdicts)} evaluation scene(s) use policy-specific "
            f"geometry -- {detail}"
        )
    return verdicts


def partition_scenes(
    scene_entries: Iterable[dict], *, routes_by_scene: dict[str, int] | None = None
) -> tuple[list[SceneEligibility], list[SceneEligibility]]:
    """Split scenes into (eval-eligible, training-only). Never raises.

    Training-only is not a demotion: scenes fitted to a motion are the corpus's densest and
    most useful *training* contexts. They simply cannot grade a policy.
    """
    routes = routes_by_scene or {}
    verdicts = [
        classify_scene(entry, independent_routes=routes.get(entry.get("scene_id", "")))
        for entry in scene_entries
    ]
    return [v for v in verdicts if v.eligible], [v for v in verdicts if not v.eligible]
