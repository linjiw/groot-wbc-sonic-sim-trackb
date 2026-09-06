#!/usr/bin/env python3
"""Explicit reset-aware loader adaptation; preserve the registered physics/scoring."""

import argparse
import json
from pathlib import Path

import motion2scene_action_label_completion as original
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    revision = args.out / "analysis_loader_revision_v2.json"
    if not revision.exists():
        write_new(
            revision,
            {
                "reason": (
                    "pre-execution CPU input audit exposed strict single-episode loader refusal "
                    "on reset-spanning failures"
                ),
                "references": [
                    artifact(Path(__file__)),
                    artifact(Path(original.__file__)),
                    artifact(
                        ROOT
                        / "gear_sonic/dataset_generation/hallucination/motion2scene_reset_capture.py"
                    ),
                    artifact(args.out / "manifest.json"),
                ],
                "change": (
                    "validate all reset-separated segments, retain every raw frame; "
                    "no changes to pairing thresholds, commands, scene, passage, contact horizon or predictions"
                ),
                "original_physics_manifest_unchanged": True,
            },
        )
    for ref in json.loads(revision.read_text())["references"]:
        checked(Path(ref["path"]), ref["sha256"])

    if args.register_only:
        return

    # Deliberate compatibility adapter for the one loader dependency. The original
    # registered analysis source and every physical result remain byte-identical.
    def payload(row):
        ref = row["trajectory"]
        return load_reset_capture(checked(Path(ref["path"]), ref["sha256"]))

    original.payload = payload
    original.analyze(args.out)


if __name__ == "__main__":
    main()
