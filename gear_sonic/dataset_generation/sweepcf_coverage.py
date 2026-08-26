"""Identity and coverage primitives for the SweepCF LFH index.

The functions in this module are deliberately geometry-free.  They enrich facts that are
already present in manifests or episode rows and retain ``unknown`` when the artifact does not
carry enough evidence.  In particular, a directory name is an instance identifier, not proof of
an independent causal family.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

UNKNOWN = "unknown"

SEMANTIC_GROUP_BY_BODY = {
    "pelvis": "pelvis",
    "left_hip_roll_link": "knee_left",
    "left_knee_link": "knee_left",
    "left_ankle_roll_link": "foot_left",
    "right_hip_roll_link": "knee_right",
    "right_knee_link": "knee_right",
    "right_ankle_roll_link": "foot_right",
    "torso_link": "head_torso",
    "left_shoulder_yaw_link": "shoulder_left",
    "left_elbow_link": "shoulder_left",
    "left_wrist_yaw_link": "wrist_left",
    "right_shoulder_yaw_link": "shoulder_right",
    "right_elbow_link": "shoulder_right",
    "right_wrist_yaw_link": "wrist_right",
}

AXIS_ALIASES = {
    "overhead": "overhead",
    "ceiling": "overhead",
    "lateral": "lateral_gap",
    "lateral_gap": "lateral_gap",
    "wall": "lateral_gap",
    "floor": "ground_support",
    "ground_support": "ground_support",
}

OPERATOR_BEHAVIOUR = {
    "local_crouch": "crouch",
    "local_arm_tuck": "arms",
}

CANONICAL_CELL_ROLES = frozenset(
    {"nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard"}
)


@dataclass(frozen=True)
class FamilyIdentity:
    """Causal source and scene-instance identity kept as separate facts."""

    source_family_id: str
    variant_id: str
    source_identity_status: str


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _number(value: object) -> float | None:
    try:
        text = _text(value)
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def family_identity(meta: Mapping[str, object], artifact_id: str) -> FamilyIdentity:
    """Resolve identity from explicit manifest fields, never from naming resemblance.

    ``artifact_id`` remains a valid variant identifier because it names the stored instance.
    It is not promoted to ``source_family_id`` unless the manifest says that it is the source.
    """

    source = _text(meta.get("source_family_id") or meta.get("family_id"))
    variant = _text(meta.get("variant_id")) or artifact_id
    return FamilyIdentity(
        source_family_id=source or UNKNOWN,
        variant_id=variant or UNKNOWN,
        source_identity_status="manifest" if source else UNKNOWN,
    )


def constraint_axis(meta: Mapping[str, object], obstacle_coordinate: object = None) -> str:
    """Return the explicit constraint axis, or the narrow shelf-only fallback.

    A measured ``obstacle_underside_m`` is definitionally an overhead face.  No analogous
    fallback is made from family names or operators for lateral/support constraints.
    """

    for key in ("constraint_axis", "axis_type", "regime", "obstacle"):
        value = _text(meta.get(key)).lower()
        if value in AXIS_ALIASES:
            return AXIS_ALIASES[value]
    return "overhead" if _number(obstacle_coordinate) is not None else UNKNOWN


def binding_station_offset_mm(meta: Mapping[str, object], box: tuple[float, ...]) -> float | None:
    """Maximum XY offset between a manifest station and the loaded obstacle AABB.

    Older matched-family assets carried a two-metre frame error that a broad footprint route
    check could still pass. Returning ``None`` when no station was recorded keeps that absence
    explicit; callers must not invent a station from the family name.
    """

    expected = meta.get("shelf_center_xy_m")
    if not isinstance(expected, (list, tuple)) or len(expected) < 2:
        return None
    measured = ((float(box[0]) + float(box[3])) / 2, (float(box[1]) + float(box[4])) / 2)
    return 1000 * max(
        abs(float(expected[0]) - measured[0]),
        abs(float(expected[1]) - measured[1]),
    )


def semantic_keypoint(body_name: object) -> str:
    """Map one authoritative capsule owner to its accepted semantic capsule group."""

    body = _text(body_name).split("/")[-1].lower()
    return SEMANTIC_GROUP_BY_BODY.get(body, UNKNOWN)


def edit_behaviour(operator: object) -> str:
    """Behaviour introduced by an edit operator, distinct from prompt intent."""

    return OPERATOR_BEHAVIOUR.get(_text(operator), UNKNOWN)


def coordinate_bucket(axis: object, coordinate: object, config: Mapping[str, object]) -> str:
    """Bucket one measured binding coordinate with the versioned DCS configuration."""

    axis_name = _text(axis) or UNKNOWN
    value = _number(coordinate)
    spec = (config.get("coordinate_buckets") or {}).get(axis_name)  # type: ignore[union-attr]
    if value is None or not isinstance(spec, Mapping):
        return UNKNOWN
    edges = [float(v) for v in spec.get("edges_m", [])]
    labels = [str(v) for v in spec.get("labels", [])]
    if len(labels) != len(edges) + 1 or not edges:
        raise ValueError(f"invalid coordinate bucket config for {axis_name}")
    for index, edge in enumerate(edges):
        if value < edge:
            return labels[index]
    return labels[-1]


def margin_bucket(clearance_mm: object, config: Mapping[str, object]) -> str:
    """Bucket signed capsule clearance; non-numeric values remain unknown."""

    value = _number(clearance_mm)
    if value is None:
        return UNKNOWN
    edges = [float(v) for v in config.get("margin_edges_mm", [0, 10, 25, 50, 100])]
    labels = [str(v) for v in config.get("margin_labels", [])]
    if len(labels) != len(edges) + 1:
        raise ValueError("margin_labels must contain len(margin_edges_mm) + 1 values")
    # Contact includes exact zero, matching the registered SweepCF convention.
    if value <= edges[0]:
        return labels[0]
    for index, edge in enumerate(edges[1:], start=1):
        if value <= edge:
            return labels[index]
    return labels[-1]


def motion_key(name: object) -> str:
    """Stable clip stem shared by screen, gate, and rollout artifacts."""

    stem = Path(_text(name)).stem
    stem = re.sub(r"^(clutter|placed|batch)_", "", stem)
    return re.sub(r"(_s\d+)+$", "", stem)


def load_reference_gate(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    return {int(row["index"]): row for row in rows}


def load_trackability(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    return {motion_key(row["clip"]): row for row in rows}


def motion_gate_fields(
    screen_record: Mapping[str, object],
    reference_gate: Mapping[int, Mapping[str, object]],
    trackability: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Join CPU gate and executed empty-room trackability onto one motion row."""

    csv_name = _text(screen_record.get("csv"))
    match = re.match(r"(\d{3})_", Path(csv_name).name)
    index = int(match.group(1)) if match else -1
    gate = reference_gate.get(index, {})
    tracked = trackability.get(motion_key(csv_name), {})

    semantic = gate.get("reference_semantic_valid")
    collision_free = gate.get("self_collision_free")
    worth = gate.get("worth_a_rollout")
    empty_trackable = tracked.get("empty_room_trackable")
    empty_rollouts = int(tracked.get("empty_room_rollouts") or 0)

    if empty_trackable is True:
        evidence = "accepted_empty_room"
    elif empty_trackable is False:
        evidence = "rejected_empty_room"
    elif tracked.get("tracked") is True:
        evidence = "accepted_nonempty_only"
    else:
        evidence = "no_empty_room_evidence"

    def encoded(value: object) -> object:
        return int(value) if isinstance(value, bool) else ""

    return {
        "reference_semantic_valid": encoded(semantic),
        "self_collision_free": encoded(collision_free),
        "worth_a_rollout": encoded(worth),
        "controller_trackable": encoded(empty_trackable),
        "controller_trackability_evidence": evidence,
        "controller_trackability_rollouts": empty_rollouts,
    }


def target_bins(config: Mapping[str, object]) -> list[dict[str, str]]:
    """Expand the scientifically admissible target templates in one DCS config."""

    bins: list[dict[str, str]] = []
    for template in config.get("target_templates", []):
        if not isinstance(template, Mapping):
            continue
        varying = {
            "binding_keypoint": list(template.get("binding_keypoints", [])),
            "constraint_coordinate_bucket": list(template.get("coordinate_buckets", [])),
            "margin_bucket": list(template.get("margin_buckets", [])),
        }
        for keypoint, coordinate, margin in itertools.product(*varying.values()):
            bins.append(
                {
                    "edit_behaviour_class": str(template["edit_behaviour_class"]),
                    "operator": str(template["operator"]),
                    "constraint_axis": str(template["constraint_axis"]),
                    "binding_keypoint": str(keypoint),
                    "constraint_coordinate_bucket": str(coordinate),
                    "margin_bucket": str(margin),
                }
            )
    return bins


def independent_verified_sources(families: Iterable[Mapping[str, object]]) -> set[str]:
    """Unique manifest-backed causal sources with at least one verified variant."""

    def evidence_valid(row: Mapping[str, object]) -> bool:
        if "evidence_valid" not in row:
            return True
        value = row.get("evidence_valid")
        return value is True or _text(value).lower() in ("1", "true")

    return {
        _text(row.get("source_family_id"))
        for row in families
        if _text(row.get("status")) == "verified"
        and evidence_valid(row)
        and _text(row.get("source_family_id")) not in ("", UNKNOWN)
    }


def verified_variant_keys(
    families: Iterable[Mapping[str, object]],
) -> set[tuple[str, str]]:
    """Return exact source/variant identities with complete verified evidence.

    Source-level verification is intentionally insufficient here.  A source may have both
    verified and refused variants, and calibration ``probe`` rows may share the verified source
    identity.  Neither should fill a causal-family coverage target.
    """

    def evidence_valid(row: Mapping[str, object]) -> bool:
        if "evidence_valid" not in row:
            return True
        value = row.get("evidence_valid")
        return value is True or _text(value).lower() in ("1", "true")

    return {
        (_text(row.get("source_family_id")), _text(row.get("variant_id")))
        for row in families
        if _text(row.get("status")) == "verified"
        and evidence_valid(row)
        and _text(row.get("source_family_id")) not in ("", UNKNOWN)
        and _text(row.get("variant_id")) not in ("", UNKNOWN)
    }


def target_occupancy_episodes(
    episodes: Iterable[Mapping[str, object]],
    families: Iterable[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Select the evidence allowed to fill generation targets.

    ``all_episodes`` preserves the original SweepCF-DCS v1 accounting.  The v2 policy admits
    only canonical cells from an exact verified variant, preventing a probe or a refused sibling
    variant from blocking a useful independent-family proposal.
    """

    policy = _text(config.get("target_occupancy_policy")) or "all_episodes"
    rows = list(episodes)
    if policy == "all_episodes":
        return rows
    if policy != "verified_canonical_family_cells":
        raise ValueError(f"unknown target_occupancy_policy {policy!r}")
    verified = verified_variant_keys(families)
    return [
        row
        for row in rows
        if (_text(row.get("source_family_id")), _text(row.get("variant_id"))) in verified
        and _text(row.get("cell_role")) in CANONICAL_CELL_ROLES
    ]
