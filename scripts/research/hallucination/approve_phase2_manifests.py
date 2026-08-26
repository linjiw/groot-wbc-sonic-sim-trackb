#!/usr/bin/env python3
"""Create immutable, self-approved LFH Phase-2 manifests after a strict preflight."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_ROOT = REPO_ROOT / "docs/hallucination/manifests"
GOVERNANCE = REPO_ROOT / "docs/hallucination/GOVERNANCE.md"
REGISTER = REPO_ROOT / "docs/prediction_register.md"
DAILY_BUDGET_GPU_H = 8.0

SOURCES = {
    "E16B_ACCEPTANCE_REPEATABILITY_PROPOSED.json": (
        "4812c77cf3a1fad84605a997a14450b279b8046dd08c7801d89acf7ec7490e5a"
    ),
    "E18_ARCHETYPE_FREEDOM_PROPOSED.json": (
        "e571ef8f8850e3bceaa24cdc3629d15dea098a68d90b037a9eb32ce7dde6b91f"
    ),
    "E17_LADDER_FAMILY_PROPOSED.json": (
        "1eb55b30ff1bf2d6c7fb776dd8fa399a7c623c158435f84bc0a5be7eadd442c0"
    ),
    "E16_WINDOW_REPEATABILITY_PROPOSED.json": (
        "294be95150e5919781dc2dfd1524fd7a3ad5971f246bc723066fcdbf3ab82861"
    ),
    "E12_CROUCH_LADDER_PROPOSED.json": (
        "29decfdf22c3d753f9e6f5f2513280fc0e36609a799c170d7a850d2f2122b063"
    ),
    "E1A_REPEATABILITY_PROPOSED.json": (
        "cdb718992587554cb1628160dc0a8050bc1b768a5c0646b89c8cebe74d52cbac"
    ),
    "E1B_DUCK003_PHYSICS_PROPOSED_V2.json": (
        "7809e36bec81f4f0b9b3211725300e7fc8c6d1fe3357716a23b8ada9baf48856"
    ),
    "MINIMAL_PLANE_PROBES_PROPOSED_V2.json": (
        "0cb332765361d57e934a698d4954b4c571d20eb79b972bbee3c998b248d0c0c6"
    ),
    "MINIMAL_PLANE_PROBES_PROPOSED_V3.json": (
        "10dcd17454b913c0eb6be73ace580db21efe2a4d2b904129320b77a6364e0ed7"
    ),
    "MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V4.json": (
        "a028f08ab7ca1ad6b3c81a9be3baa638e0f5f1c98c3afc1720a815be73756bdf"
    ),
    "MINIMAL_EMPTY_ROOM_PROBES_PROPOSED_V5.json": (
        "1bae4625fee43821676d49bc78f40152fdee2fa592aff6ddb7942342eb06e560"
    ),
    "E2_VARIANT_TRANSFER_PROPOSED.json": (
        "cba8997fdd0f041e20be8c325e4da661c8e407be035ab91f183ddd914331b2d7"
    ),
    "ARM_TUCK_CALIBRATION_PROPOSED.json": (
        "51202f748bf81d33791383a2c4b8fc962537f765dca88c54e3b5c01b4774b830"
    ),
    "ARM_TUCK_STRONG_CALIBRATION_PROPOSED.json": (
        "d3b0dcf8a370c51405b79cdc88205a71a246b9cdc281811af7ca7764a721ef65"
    ),
    "E3_LATERAL_PILOT_PROPOSED.json": (
        "eab420e271953952eb1aa64c6a7b5f7c522c15a4eb2fde6748ae493360bd28f6"
    ),
    "CROUCH_CALIBRATION_PROPOSED.json": (
        "a58019a4d3acb4c23a183b6f35f190bd57e22e69d3d8a86d75cd658aafcd9a2e"
    ),
    "E6A_CROUCH_CONTEXT_PROPOSED.json": (
        "0969a48cb83f00594034f234b68d71169818b1e5526eeb3c4c1fa90a22c4835a"
    ),
    "E6B_CROUCH_HARD_PROPOSED.json": (
        "145a8bd14c794c018a6b2e003e57c55c2721196fda09966ffe4c4ec84c642224"
    ),
    "E6C_CROUCH_EXPOSURE_EASY_PROPOSED.json": (
        "c1fcb5229a73b1d6f10cc8c94111590d3f895dbd8dd3f273187547b1577fefe9"
    ),
    "E6C_CROUCH_EXPOSURE_HARD_PROPOSED.json": (
        "89dee203edeebd8aebcf35de6874842eac4625b186f5e19f6d6d603477fdcab9"
    ),
    "E7A_ARCHETYPE_CONTEXT_PROPOSED.json": (
        "f95bfb63f1a403e6bf85b8fef97bb056709f65f589101a23ee75c2228f3c3d45"
    ),
    "E7B_ARCHETYPE_HARD_PROPOSED.json": (
        "58243a8a0c8ec7d570d64aa80b33fd77d53d2c2870bbce331cd5bfb387cff4be"
    ),
    "E7C_REPLACEMENT_EASY_PROPOSED.json": (
        "49f2ad83142aa8b88f9f0327ecabd1c3481f2504e1b72a7a2bd32d78332eb16e"
    ),
    "E7C_REPLACEMENT_HARD_PROPOSED.json": (
        "6d477ec4bb2108d0c33c1d754ed47fba38a95b0003c441979ff0598ee7e8f125"
    ),
    "E9A_MOTION_SCENE_CALIBRATION_PROPOSED.json": (
        "c1722d3cb72d2622c1e83aed0c97b2243bb9993cd508dd4b94f45310192b5f4d"
    ),
    "E10_CONTEXT_RICH_PROPOSED.json": (
        "deb3ceba0c99b52a7559544dc18c44b54e40089f054236acf1d35fdc953e936c"
    ),
    "E10B_CONTEXT_RICH_SEED_CONTROL_PROPOSED.json": (
        "962af6a87de018cad19c6b040c764d4b49694cd1c7b238cfd00d162e5b875447"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_pinned(path: Path, expected: str, label: str) -> None:
    actual = "sha256:" + sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: hash drift: {actual} != {expected}")


def preflight(payload: dict) -> None:
    policy = payload["execution_policy"]
    if policy.get("not_authorized") is not True:
        raise ValueError("source manifest is not an unmodified proposal")
    if policy.get("claim5_inputs_untouched") is not True:
        raise ValueError("source manifest does not preserve claim-5 inputs")
    if policy["runtime"]["free_gpu_mib_required"] != 9000:
        raise ValueError("free-memory gate drifted")
    if policy["runtime"]["hang_timeout_seconds"] != 1800:
        raise ValueError("hang timeout drifted")

    for cell in payload["cells"]:
        motion = cell["motion"]
        motion_path = Path(motion["path"])
        provenance_path = Path(motion["conversion_provenance"])
        _check_pinned(motion_path, motion["sha256"], f"{cell['cell_id']} motion")
        _check_pinned(
            provenance_path,
            motion["conversion_provenance_sha256"],
            f"{cell['cell_id']} provenance",
        )
        if motion["scene_start_xyz"] != cell["scene_start_xyz_expected"]:
            raise ValueError(f"{cell['cell_id']}: scene-start mismatch")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("scene_start_xyz") != cell["scene_start_xyz_expected"]:
            raise ValueError(f"{cell['cell_id']}: provenance scene-start is absent or mismatched")

        scene = cell["scene"]
        if scene["scene_id"] == "plane":
            if scene["path"] is not None or scene["sha256"] is not None:
                raise ValueError(f"{cell['cell_id']}: plane must not carry an authored scene")
        else:
            scene_path = REPO_ROOT / scene["path"]
            _check_pinned(scene_path, scene["sha256"], f"{cell['cell_id']} scene")

        output = Path(cell["output"])
        protected_tokens = ("scene_first_v1", "scene_first_v2", "claim5", "sweepcf_release")
        if any(token in output.parts for token in protected_tokens):
            raise ValueError(f"{cell['cell_id']}: protected output path {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--only", action="append", choices=tuple(SOURCES))
    args = parser.parse_args()

    if not GOVERNANCE.exists() or "Autonomous Research Charter" not in GOVERNANCE.read_text():
        raise SystemExit("LFH governance charter is unavailable")
    if "### LFH-E1a" not in REGISTER.read_text() or "### LFH-PROBES-1" not in REGISTER.read_text():
        raise SystemExit("LFH predictions must be registered before approval")
    if args.only and "CROUCH_CALIBRATION_PROPOSED.json" in args.only:
        if "### LFH-CAL3" not in REGISTER.read_text():
            raise SystemExit("LFH-CAL3 prediction must be registered before approval")
    if args.only and "E6A_CROUCH_CONTEXT_PROPOSED.json" in args.only:
        if "### LFH-E6" not in REGISTER.read_text():
            raise SystemExit("LFH-E6 prediction must be registered before approval")
    if args.only and "E6B_CROUCH_HARD_PROPOSED.json" in args.only:
        if "### LFH-E6a reconciliation and E6b registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E6b prediction must be registered before approval")
    if args.only and "E6C_CROUCH_EXPOSURE_EASY_PROPOSED.json" in args.only:
        if "### LFH-E6c" not in REGISTER.read_text():
            raise SystemExit("LFH-E6c prediction must be registered before approval")
    if args.only and "E6C_CROUCH_EXPOSURE_HARD_PROPOSED.json" in args.only:
        if "### LFH-E6c-easy reconciliation and hard registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E6c hard prediction must be registered before approval")
    if args.only and "E7A_ARCHETYPE_CONTEXT_PROPOSED.json" in args.only:
        if "### LFH-E7" not in REGISTER.read_text():
            raise SystemExit("LFH-E7 prediction must be registered before approval")
    if args.only and "E7B_ARCHETYPE_HARD_PROPOSED.json" in args.only:
        if "### LFH-E7a reconciliation and E7b registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E7b prediction must be registered before approval")
    if args.only and "E7C_REPLACEMENT_EASY_PROPOSED.json" in args.only:
        if "### LFH-E7 reconciliation and E7c registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E7c prediction must be registered before approval")
    if args.only and "E7C_REPLACEMENT_HARD_PROPOSED.json" in args.only:
        if "### LFH-E7c-easy reconciliation and hard registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E7c hard prediction must be registered before approval")
    if args.only and "E9A_MOTION_SCENE_CALIBRATION_PROPOSED.json" in args.only:
        if "### LFH-E9a" not in REGISTER.read_text():
            raise SystemExit("LFH-E9a prediction must be registered before approval")
    if args.only and "E10_CONTEXT_RICH_PROPOSED.json" in args.only:
        if "### LFH-E9a reconciliation and E10 registration" not in REGISTER.read_text():
            raise SystemExit("LFH-E10 prediction must be registered before approval")
    if args.only and "E10B_CONTEXT_RICH_SEED_CONTROL_PROPOSED.json" in args.only:
        if "### LFH-E10 reconciliation and E10b seed control" not in REGISTER.read_text():
            raise SystemExit("LFH-E10b prediction must be registered before approval")

    if args.only and "E16B_ACCEPTANCE_REPEATABILITY_PROPOSED.json" in args.only:
        if "### LFH-E16b" not in REGISTER.read_text():
            raise SystemExit("LFH-E16b prediction must be registered before approval")
    if args.only and "E18_ARCHETYPE_FREEDOM_PROPOSED.json" in args.only:
        if "### LFH-E18" not in REGISTER.read_text():
            raise SystemExit("LFH-E18 prediction must be registered before approval")
    if args.only and "E17_LADDER_FAMILY_PROPOSED.json" in args.only:
        if "### LFH-E17" not in REGISTER.read_text():
            raise SystemExit("LFH-E17 prediction must be registered before approval")
    if args.only and "E16_WINDOW_REPEATABILITY_PROPOSED.json" in args.only:
        if "### LFH-E16" not in REGISTER.read_text():
            raise SystemExit("LFH-E16 prediction must be registered before approval")
    if args.only and "E12_CROUCH_LADDER_PROPOSED.json" in args.only:
        if "### LFH-E12" not in REGISTER.read_text():
            raise SystemExit("LFH-E12 prediction must be registered before approval")

    loaded: list[tuple[Path, dict, str]] = []
    projected = 0.0
    selected = args.only or list(SOURCES)
    for name in selected:
        expected_hash = SOURCES[name]
        source = MANIFEST_ROOT / name
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise SystemExit(f"{name}: reviewed hash drift: {actual_hash} != {expected_hash}")
        payload = json.loads(source.read_text())
        preflight(payload)
        projected += float(payload["execution_policy"]["cost_ceiling"]["gpu_hours_contended"])
        loaded.append((source, payload, actual_hash))
    if projected > DAILY_BUDGET_GPU_H:
        raise SystemExit(f"projected {projected:.3f} GPU-h exceeds daily budget")

    register_hash = "sha256:" + sha256(REGISTER)
    for source, proposal, source_hash in loaded:
        destination = source.with_name(source.name.replace("_PROPOSED", "_APPROVED"))
        if destination.exists():
            raise SystemExit(f"refusing to overwrite immutable approval: {destination}")
        approved = deepcopy(proposal)
        policy = approved["execution_policy"]
        policy["not_authorized"] = False
        policy["requires_explicit_user_approval"] = False
        policy["prediction_register_untouched"] = False
        policy["user_owned_prediction_register_entries_untouched"] = True
        policy["timing_override"] = {
            "default_not_before_overridden": True,
            "authority": "docs/hallucination/GOVERNANCE.md §1",
            "reason": (
                "outputs are isolated from claim-5 inputs and execution yields under GPU "
                "contention"
            ),
        }
        approved["authorization"] = {
            "status": "self_approved",
            "approved_at": args.timestamp,
            "approved_by": "Codex LFH autonomous researcher",
            "authority": "docs/hallucination/GOVERNANCE.md",
            "source_manifest": str(source.relative_to(REPO_ROOT)),
            "source_manifest_sha256": "sha256:" + source_hash,
            "prediction_register_sha256_at_approval": register_hash,
            "claim5_isolation_verified": True,
            "daily_projected_gpu_hours_approval_set": round(projected, 3),
        }
        destination.write_text(json.dumps(approved, indent=2, sort_keys=True) + "\n")
        destination.chmod(0o444)
        print(f"APPROVED {destination.name} sha256:{sha256(destination)}")
    print(f"projected Phase-2 spend: {projected:.3f}/{DAILY_BUDGET_GPU_H:.3f} GPU-h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
