#!/usr/bin/env python3
"""Reproduce the four diagnostic controls from public files, without simulator data."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from motion2scene_selector_diagnosis_v2 import low_capacity
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle, diagnostic, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    archive = bundle / "selectors.tar.gz"
    manifest = json.loads((bundle / "selectors-manifest.json").read_text())
    assert sha(archive) == manifest["archive"]["sha256"]
    with tarfile.open(archive) as stream:
        training = json.load(stream.extractfile("training.json"))
    pairs = {p["group_id"]: p for p in training["pairs"]}
    exports = json.loads((diagnostic / "exports.json").read_text())
    for ref in exports:
        assert sha(diagnostic / ref["public"]) == ref["public_sha256"]
    physical = json.loads((diagnostic / "p3-result.json").read_text())["rows"]
    p1 = json.loads((diagnostic / "p1-result.json").read_text())["pairs"]
    features = {p["block"]: p["features"] for p in p1}
    rows = []
    for arm, ids in training["fit_ids"].items():
        x = np.array([pairs[k]["features"] for k in ids], np.float32)
        y = np.array([pairs[k]["outcomes"] for k in ids], np.float32)
        e = np.array([features[k] for k in range(6)], np.float32)
        fitted, _, _ = low_capacity(x, y, e)
        with np.load(diagnostic / f"p0-linear_{arm}.npz") as saved:
            exact = all(np.array_equal(fitted[k], saved[k]) for k in fitted)
        policy_path = diagnostic / f"{arm}.pt"
        policy = load_readout(policy_path, sha(policy_path))
        errors, matches = [], []
        for r in [r for r in physical if r["arm"] == arm]:
            actual = readout(policy, features[r["original_block"]])
            errors.append(
                float(abs(np.array(actual["probabilities"]) - r["readout"]["probabilities"]).max())
            )
            matches.append(
                all(
                    actual[k] == r["readout"][k]
                    for k in ("selected_action", "requested_action", "refusal")
                )
            )
        assert exact and all(matches) and max(errors) <= 1e-6
        rows.append(
            {
                "arm": arm,
                "fit_exact": exact,
                "deployed_requests_exact": all(matches),
                "max_probability_error": max(errors),
            }
        )
    with output.open("x") as f:
        json.dump(
            {"rows": rows, "scope": "CPU reproduction, no new physics or evaluation outcomes"},
            f,
            indent=2,
        )
        f.write("\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.bundle, args.diagnostic, args.output)
