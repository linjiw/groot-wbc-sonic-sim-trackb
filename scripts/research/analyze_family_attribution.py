#!/usr/bin/env python3
"""Ask *why* each cell of a counterfactual family passed or failed.

A 2x2 whose bottom-left cell is merely ``rejected`` proves nothing. The claim is that a
named obstacle stopped a named body at a predicted moment, and a rejection that happens to
arrive by some other route -- the robot drifted into a wall, fell over, tangled its own
legs -- would support the same table while meaning something entirely different. That
happened once already: an adapted motion failed both scenes at an identical 710.9 N, which
looked like a shelf result until the contact turned out to be legs against a far wall.

So this measures **failure attribution purity**:

* is the first disqualifying event the intended obstacle contact, or something earlier?
* is the contacting body in the group the regime targets (overhead -> head/torso)?
* does the observed contact frame agree with the one geometry predicted from the probe?
* did the reference drift begin before the contact, or is it the contact's consequence?

The last question matters because ``nominal_hard`` is rejected for both
``disallowed_robot_contact`` and ``unstable_reference_drift``. If the drift came first, the
episode failed for tracking reasons and the shelf is incidental. If it followed the contact,
it *is* the collision's downstream effect and the attribution is clean.

Usage::

    python scripts/research/analyze_family_attribution.py --family /data/.../duck_002
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.counterfactual_family import (  # noqa: E402
    swept_clearance_profile,
)
from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Body groups each regime is entitled to hit. A contact outside its group means the scene
#: interfered with the robot somewhere the family was not designed to test.
REGIME_BODIES = {
    "overhead": ("torso_link", "head", "waist", "shoulder", "pelvis", "elbow", "wrist"),
    "lateral": ("shoulder", "elbow", "wrist", "hip", "torso_link", "hand"),
    "floor": ("ankle", "knee", "foot", "hip"),
}

#: Newtons above which a non-foot contact counts as having happened. The acceptance gate
#: uses 1.0 N; this is the same number, named here so the report can say what it means.
CONTACT_THRESHOLD_N = 1.0

#: Metres of root-to-reference error taken as the drift having begun.
DRIFT_ONSET_M = 0.15


def load(directory: Path) -> dict:
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectory in {directory}")
    with paths[0].open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def shelf_box(usda: Path) -> tuple[float, ...]:
    """Recover the shelf's world box from the scene file rather than recomputing it.

    Recomputing the mid-path point would re-derive what the builder *intended*; reading the
    USDA reports what was actually rendered, which is the thing the physics saw.
    """
    text = usda.read_text(encoding="utf-8")
    block = text[text.index("LowShelf_00"):]
    size = [float(v) for v in re.search(r"double3 xformOp:scale = \(([^)]+)\)", block).group(1).split(",")]
    trans = [float(v) for v in re.search(r"double3 xformOp:translate = \(([^)]+)\)", block).group(1).split(",")]
    half = [s / 2.0 for s in size]
    return (
        trans[0] - half[0], trans[1] - half[1], trans[2] - half[2],
        trans[0] + half[0], trans[1] + half[1], trans[2] + half[2],
    )


def first_contact(payload: dict) -> tuple[int | None, str, float]:
    """Earliest frame carrying a *lateral* external contact, and the body carrying it.

    Reading the raw force array instead gets frame 0 of every episode, because the robot is
    dropped into the scene and its hips carry a settling load -- 237.4 N in one cell here,
    on an episode with no collision at all. The acceptance gate has always decomposed
    contact by Newton's third law into self versus external and lateral versus support, and
    a collision is the lateral part. Using anything else re-answers a question that was
    already answered correctly, and gets it wrong.
    """
    decomposition = decompose_payload_contacts(payload)
    lateral = decomposition.external_lateral_by_frame
    hits = np.argwhere(lateral > CONTACT_THRESHOLD_N)
    if hits.size == 0:
        return None, "", 0.0
    frame = int(hits[0, 0])

    # Attribute the frame to a body by its horizontal force, over the same non-foot set the
    # decomposition considered external.
    forces = np.asarray(payload["robot_contact_force_w"], dtype=np.float64)[frame]
    names = list(payload["contact_body_names"])
    external = set(decomposition.external_contact_bodies)
    horizontal = np.linalg.norm(forces[:, :2], axis=1)
    candidates = [i for i, n in enumerate(names) if n in external]
    if not candidates:
        return frame, "", float(lateral[frame])
    best = max(candidates, key=lambda i: horizontal[i])
    return frame, names[best], float(lateral[frame])


def drift_onset(payload: dict) -> int | None:
    """First frame whose root has left the reference by more than ``DRIFT_ONSET_M``."""
    reference = np.asarray(payload.get("reference_g1_qpos"), dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] == 0:
        return None
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    error = np.linalg.norm(root[:, :2] - reference[:, :2], axis=1)
    over = np.argwhere(error > DRIFT_ONSET_M)
    return int(over[0, 0]) if over.size else None


def predicted_frame(probe: dict, box) -> tuple[float, int | None, int | None]:
    """Minimum clearance, the frame it first goes negative, and the frame it is deepest.

    The first-negative frame is what an observed first contact should be compared against.
    Comparing against the deepest frame instead flatters or damns the predictor for no
    reason: the robot touches the near face of a 0.5 m shelf well before it reaches the
    point of greatest penetration.
    """
    profile = swept_clearance_profile(
        np.asarray(probe["body_pos_w"], dtype=np.float64),
        np.asarray(probe["body_quat_w"], dtype=np.float64),
        list(probe["body_names"]),
        box,
    )
    negative = np.argwhere(profile < 0.0)
    first = int(negative[0, 0]) if negative.size else None
    deepest = int(np.argmin(profile)) if profile.size else None
    return float(profile.min()), first, (deepest if first is not None else None)


def analyse(family_dir: Path, scenes_root: Path) -> dict:
    family = json.loads((family_dir / "family.json").read_text())
    regime = family.get("regime", "overhead")
    allowed = REGIME_BODIES.get(regime, ())
    probes = {
        "nominal": load(family_dir / "probe_nominal"),
        "adapted": load(family_dir / "probe_adapted"),
    }
    # Prefer the run-tagged scene, falling back to the bare family_id for runs made before
    # scenes were namespaced. A missing scene is an error rather than a silent skip: reading
    # another run's geometry is exactly the failure this naming exists to prevent.
    def scene_for(difficulty: str) -> Path:
        candidates = [
            scenes_root / f"{family['family_id']}_{family_dir.name}_{difficulty}.usda",
            scenes_root / f"{family['family_id']}_{difficulty}.usda",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SystemExit(
            f"no scene file for {difficulty}; looked for "
            + " and ".join(str(c) for c in candidates)
        )

    boxes = {difficulty: shelf_box(scene_for(difficulty)) for difficulty in ("easy", "hard")}

    cells = {}
    for motion in ("nominal", "adapted"):
        for difficulty in ("easy", "hard"):
            key = f"{motion}_{difficulty}"
            payload = load(family_dir / key)
            outcome = classify_episode(key, payload)
            frame, body, force = first_contact(payload)
            onset = drift_onset(payload)
            clearance, pframe, pdeep = predicted_frame(probes[motion], boxes[difficulty])
            expect_fail = motion == "nominal" and difficulty == "hard"

            problems = []
            if expect_fail:
                if outcome.outcome != "rejected":
                    problems.append(f"expected a rejection, got {outcome.outcome}")
                if frame is None:
                    problems.append("no contact at all, so the shelf did not stop it")
                elif allowed and not any(token in body for token in allowed):
                    problems.append(f"contact on {body}, outside the {regime} body group")
                if frame is not None and onset is not None and onset < frame:
                    problems.append(
                        f"drift began at frame {onset}, before the contact at {frame}"
                    )
                if clearance >= 0:
                    problems.append("geometry predicted no interference in the hard scene")
            else:
                if outcome.outcome != "accepted":
                    problems.append(f"expected acceptance, got {outcome.outcome}")
                if frame is not None:
                    problems.append(f"unexpected {force:.1f} N contact on {body}")

            cells[key] = {
                "outcome": outcome.outcome,
                "rejection_reasons": list(outcome.rejection_reasons),
                "frames": int(payload["total_frames"]),
                "first_contact_frame": frame,
                "first_contact_body": body,
                "first_contact_force_n": round(force, 2),
                "drift_onset_frame": onset,
                "predicted_clearance_m": round(clearance, 4),
                "predicted_first_interference_frame": pframe,
                "predicted_deepest_frame": pdeep,
                "attribution_problems": problems,
            }

    # Penetration depth, which the clearance metric cannot report. A capsule wholly inside
    # the box has point-to-box distance zero, so clearance saturates at minus the capsule
    # radius -- 0.068 m for the torso -- however far past the surface the body actually is.
    # The boundary height does not saturate: the shelf would have to rise by exactly
    # (boundary - underside) to stop touching. That is the number that tracks contact force,
    # and it is what a margin has to be tuned against.
    penetration = None
    if family.get("nominal_clears_to_m") and family.get("hard_shelf_underside_m"):
        penetration = family["nominal_clears_to_m"] - family["hard_shelf_underside_m"]

    pure = all(not cell["attribution_problems"] for cell in cells.values())
    return {
        "nominal_penetration_depth_m": penetration,
        "family_id": family["family_id"],
        "regime": regime,
        "window_m": family.get("window_m"),
        "cells": cells,
        "attribution_pure": pure,
        "claim_level": "nominally physics verified" if pure else "unverified",
    }


def render(report: dict) -> str:
    lines = [
        f"family {report['family_id']}   regime {report['regime']}   "
        f"window {report['window_m']:.3f} m",
        "",
        f"{'cell':16s}{'outcome':10s}{'contact':>26s}{'drift':>8s}"
        f"{'predicted':>12s}{'obs/pred frame':>16s}",
    ]
    for key, cell in report["cells"].items():
        frame = cell["first_contact_frame"]
        contact = (
            f"{cell['first_contact_body']} @{frame} ({cell['first_contact_force_n']:.1f} N)"
            if frame is not None else "none"
        )
        onset = cell["drift_onset_frame"]
        pframe = cell["predicted_first_interference_frame"]
        lines.append(
            f"{key:16s}{cell['outcome']:10s}{contact:>26s}"
            f"{('-' if onset is None else str(onset)):>8s}"
            f"{cell['predicted_clearance_m']:>12.4f}"
            f"{f'{frame}/{pframe}' if frame is not None and pframe is not None else '-':>16s}"
        )
    lines.append("")
    for key, cell in report["cells"].items():
        for problem in cell["attribution_problems"]:
            lines.append(f"  IMPURE  {key}: {problem}")
    lines.append("")
    depth = report.get("nominal_penetration_depth_m")
    if depth is not None:
        force = report["cells"].get("nominal_hard", {}).get("first_contact_force_n", 0.0)
        lines.append(
            f"intended failure: the shelf sits {depth * 1000:.0f} mm below the height that "
            f"would clear the nominal motion, and produced {force:.1f} N"
        )
        lines.append(
            "  (clearance saturates at -0.068 m once the torso capsule is engulfed; this "
            "does not)"
        )
    lines.append(f"attribution pure: {report['attribution_pure']}")
    lines.append(f"claim level:      {report['claim_level']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = analyse(args.family, args.scenes)
    print(render(report))
    destination = args.json or args.family / "attribution.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0 if report["attribution_pure"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
