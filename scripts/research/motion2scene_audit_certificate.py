#!/usr/bin/env python3
"""Independently check certificate partition topology and every recorded query in PyTorch."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_inverse_learning import DOMAIN, inputs
from motion2scene_margin_learning import load
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import make_clouds, tensor
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)


def key(lower, upper):
    return tuple(lower) + tuple(upper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    for ref in [
        manifest["source_result"],
        manifest["source_manifest"],
        manifest["protocol"],
        *manifest["implementations"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    _, source, states, route = load(Path(manifest["source_manifest"]["path"]))
    torch.set_num_threads(2)
    _, _, rt, progress, yaw, _, _ = inputs(states, route, source["records"])
    clouds = make_clouds(states)
    mask = np.array([r["label"] == "d055" for r in source["records"]])
    started = time.monotonic()
    rows = []
    for candidate in result["rows"]:
        ref = candidate["artifact"]
        detail = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        certificate, queries = detail["certificate"], detail["queries"]
        if certificate["status"] != "conditional_pass":
            rows.append({"arm": candidate["arm"], "status": "no_complete_certificate_to_audit"})
            continue
        trace = certificate["trace"]
        assert len(trace) == len(queries) == certificate["queries"]
        domains = {key(r["lower"], r["upper"]): r for r in trace}
        assert len(domains) == len(trace)
        parents = {k: 0 for k in domains}
        leaf_volume = 0.0
        for row, query in zip(trace, queries):
            lo, hi = np.array(row["lower"]), np.array(row["upper"])
            np.testing.assert_array_equal((lo + hi) / 2, query["offset"])
            values = np.array(query["per_recording_clearance_m"])
            assert values[mask].min() == row["target_clearance_m"]
            assert values[~mask].max() == row["upright_clearance_m"]
            if row["decision"] == "split":
                valid_splits = []
                for axis in range(4):
                    midpoint = (lo[axis] + hi[axis]) / 2
                    if not lo[axis] < midpoint < hi[axis]:
                        continue
                    child_hi, child_lo = hi.copy(), lo.copy()
                    child_hi[axis], child_lo[axis] = midpoint, midpoint
                    children = key(lo, child_hi), key(child_lo, hi)
                    if all(k in domains for k in children):
                        valid_splits.append(children)
                assert len(valid_splits) == 1
                for child in valid_splits[0]:
                    parents[child] += 1
            else:
                assert row["decision"] == "certified_cell"
                half = (hi - lo) / 2
                bound = np.linalg.norm(half[:3]) + 2 * manifest["horizontal_radius_m"] * np.sin(
                    half[3] / 2
                )
                bound += manifest["assumed_absolute_numerical_error_m"]
                assert row["target_clearance_m"] - bound >= manifest["margin_m"]
                assert row["upright_clearance_m"] + bound <= -manifest["margin_m"]
                leaf_volume += float(np.prod(hi - lo))
        root_key = key(manifest["lower"], manifest["upper"])
        assert parents[root_key] == 0
        assert all(count == 1 for k, count in parents.items() if k != root_key)
        domain_volume = float(np.prod(np.array(manifest["upper"]) - manifest["lower"]))
        assert abs(leaf_volume / domain_volume - 1) < 1e-12
        offsets = tensor([q["offset"] for q in queries])
        with torch.no_grad():
            actual = torch.cat(
                [
                    perturbed_clearances(
                        tensor([candidate["scene"]]), chunk, clouds, rt, progress, yaw, DOMAIN
                    )
                    for chunk in offsets.split(16)
                ],
                dim=1,
            )[0].numpy()
        expected = np.array([q["per_recording_clearance_m"] for q in queries])
        error = float(np.abs(actual - expected).max())
        assert error <= manifest["assumed_absolute_numerical_error_m"]
        rows.append(
            {
                "arm": candidate["arm"],
                "queries": len(queries),
                "partition_topology_verified": True,
                "leaf_volume_fraction": leaf_volume / domain_volume,
                "all_leaf_inequalities_verified": True,
                "torch_numpy_max_error_m": error,
            }
        )
    write_new(
        args.out,
        {
            "source": artifact(args.result),
            "auditor": artifact(Path(__file__)),
            "rows": rows,
            "elapsed_s": time.monotonic() - started,
            "numerical_error_bound_formally_validated": False,
            "scope": "Finite trace consistency and query agreement; not a formal floating-point proof",
        },
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
