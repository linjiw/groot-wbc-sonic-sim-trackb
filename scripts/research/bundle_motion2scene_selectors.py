#!/usr/bin/env python3
"""Release the small fixed selectors and verify reproduction from their training inputs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import fit

CORPUS = ROOT.parent / "research-data/groot-wbc/m2s-comparative-corpus-completion-v1"


def sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def bundle(out):
    out.mkdir(exist_ok=False)
    fits = json.loads((CORPUS / "fits/result.json").read_text())
    registration = json.loads((CORPUS / "registration.json").read_text())
    admission = json.loads((CORPUS / "admission.json").read_text())
    assert fits["fits"] == 20 and admission["all_inputs_admitted"]
    training = {
        "fit_ids": registration["fit_ids"],
        "pairs": [
            {k: p[k] for k in ("group_id", "features", "outcomes", "mask")}
            for p in admission["pairs"]
        ],
        "scope": "38 measured scenes; eleven fixed fitting encounters per arm; shared backgrounds reused",
    }
    files = {"training.json": (json.dumps(training, indent=2) + "\n").encode()}
    rows = []
    for row in fits["rows"]:
        ref = row["checkpoint"]
        raw = checked(Path(ref["path"]), ref["sha256"]).read_bytes()
        name = f"checkpoints/{row['arm']}_{row['seed']}.pt"
        files[name] = raw
        rows.append(
            {
                "arm": row["arm"],
                "optimizer_seed": row["seed"],
                "file": name,
                "sha256": sha(raw),
                "final_training_loss": row["final_training_loss"],
            }
        )
    manifest = {
        "models": rows,
        "training_sha256": sha(files["training.json"]),
        "implementation": artifact(Path(__file__)),
        "torch_version": torch.__version__,
        "primary_fits": 20,
        "inputs": 214,
        "outputs": 2,
        "steps": 1000,
        "scope": "fixed supervised outcome predictors, not SONIC or source motion banks",
        "provenance": "Original checkpoint bytes are preserved, including their provenance paths.",
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files["README.txt"] = (
        "Fixed Motion2Scene comparison selectors, September 6.\n"
        "Architecture 214-64-32-2; training counts eleven encounters per arm.\n"
        "Contains original twenty small selector checkpoints and exact training values.\n"
        "SONIC, motion banks and simulator assets are not included.\n"
        "Reproduce with the matching research source snapshot and CPU PyTorch:\n"
        "PYTHONPATH=. python scripts/research/bundle_motion2scene_selectors.py verify --out <this-bundle-folder>\n"
        "The verify command checks every parameter and normalization tensor bitwise; it performs no physics.\n"
        "Reuse the original files for evaluation; verification fits are not additional primary models.\n"
    ).encode()
    with tarfile.open(out / "selectors.tar.gz", "w:gz") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(raw), 0, 0o644
            archive.addfile(info, io.BytesIO(raw))
    write_new(
        out / "selectors-manifest.json",
        {
            **manifest,
            "archive": {
                "file": "selectors.tar.gz",
                "sha256": sha((out / "selectors.tar.gz").read_bytes()),
            },
        },
    )
    print(json.dumps({"models": 20, "bytes": (out / "selectors.tar.gz").stat().st_size}))


def verify(out):
    manifest = json.loads((out / "selectors-manifest.json").read_text())
    raw = (out / manifest["archive"]["file"]).read_bytes()
    assert sha(raw) == manifest["archive"]["sha256"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start, rows = time.monotonic(), []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        training_raw = archive.extractfile("training.json").read()
        assert sha(training_raw) == manifest["training_sha256"]
        training = json.loads(training_raw)
        pairs = {p["group_id"]: p for p in training["pairs"]}
        for row in manifest["models"]:
            if time.monotonic() - start > 180:
                raise TimeoutError("bounded CPU fit reproduction")
            checkpoint = archive.extractfile(row["file"]).read()
            assert sha(checkpoint) == row["sha256"]
            saved = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=False)
            ids = training["fit_ids"][row["arm"]]
            selected = [pairs[k] for k in ids]
            x = np.array([p["features"] for p in selected], dtype=np.float32)
            y = np.array([p["outcomes"] for p in selected], dtype=np.float32)
            mask = np.array([p["mask"] for p in selected], dtype=bool)
            model, mean, std, loss = fit(x, y, mask, row["optimizer_seed"], steps=1000)
            exact = all(
                torch.equal(value, saved["model"][key]) for key, value in model.state_dict().items()
            )
            normalization = np.array_equal(mean, saved["mean"]) and np.array_equal(
                std, saved["std"]
            )
            rows.append(
                {
                    "arm": row["arm"],
                    "optimizer_seed": row["optimizer_seed"],
                    "model_exact": exact,
                    "normalization_exact": normalization,
                    "loss_exact": loss == row["final_training_loss"],
                }
            )
    write_new(
        out / "selectors-reproduction.json",
        {
            "archive_sha256": manifest["archive"]["sha256"],
            "rows": rows,
            "all_exact": all(
                r["model_exact"] and r["normalization_exact"] and r["loss_exact"] for r in rows
            ),
            "seconds": time.monotonic() - start,
            "primary_models_changed": False,
            "scope": "CPU reproducibility check of twenty existing fits; no new data or physics",
        },
    )
    print(
        json.dumps(
            {
                "models": len(rows),
                "all_exact": all(
                    r["model_exact"] and r["normalization_exact"] and r["loss_exact"] for r in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bundle", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.out)
