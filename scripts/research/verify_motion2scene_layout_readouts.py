#!/usr/bin/env python3
"""Reproduce public evaluation readouts using only small released files and CPU."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)


def verify(bundle, decisions):
    manifest = json.loads((bundle / "selectors-manifest.json").read_text())
    raw = (bundle / "selectors.tar.gz").read_bytes()
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == manifest["archive"]["sha256"]
    panel = json.loads(decisions.read_text())
    models, errors = {}, []
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            for i, row in enumerate(manifest["models"]):
                path = Path(directory) / f"model_{i}.pt"
                path.write_bytes(archive.extractfile(row["file"]).read())
                models[row["arm"], row["optimizer_seed"]] = load_readout(path, row["sha256"])
        for row in panel["rows"]:
            actual = readout(models[row["arm"], row["optimizer_seed"]], row["features"])
            expected = row["readout"]
            error = float(
                np.max(np.abs(np.array(actual["probabilities"]) - expected["probabilities"]))
            )
            assert error <= 1e-6, row["cell_id"]
            assert all(
                actual[k] == expected[k] for k in ("requested_action", "selected_action", "refusal")
            ), row["cell_id"]
            errors.append(error)
    return {
        "decision_file_sha256": "sha256:" + hashlib.sha256(decisions.read_bytes()).hexdigest(),
        "selector_archive_sha256": manifest["archive"]["sha256"],
        "readouts_checked": len(errors),
        "max_probability_error": max(errors),
        "all_requests_and_refusals_match": True,
        "scope": "CPU readout reproduction; no new fitting, physics or alternative-action labels",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle, args.decisions)
    if args.receipt:
        with args.receipt.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result))
