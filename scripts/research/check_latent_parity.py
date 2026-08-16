#!/usr/bin/env python3
"""Classify a recorded SONIC 64D latent against the C++ deployment token contract.

``action.motion_token`` is the one dataset feature whose meaning is owned by the
C++ deploy runtime rather than by this repository. The Python recorder writes
``env._full_latent`` and the deploy runtime reads a ``token_state`` observation;
both occupy the decoder's flattened token input, both are post-quantization FSQ
codes, and both flatten ``(num_tokens, token_dim)`` row-major.

The C++ runtime has no residual path at all. The Python wrapper has one, but it
is unreachable while `action_transform_module_cfg` is null, which it is in every
checked-in config -- rollouts on 2026-08-15 measured residual RMS exactly 0.
Attaching a wrapper-level ATM, enabling `use_latent_residual`, or switching to a
student `direct_latent` policy would each change the recorded convention while
every shape, dtype, and schema check kept passing. Post-quantization FSQ output
lies on an exact rational lattice, so this checker tells the two apart from the
recorded numbers alone.

Usage::

    python scripts/research/check_latent_parity.py --trajectory trajectory.pkl
    python scripts/research/check_latent_parity.py --dataset /path/to/lerobot_export
    python scripts/research/check_latent_parity.py --dataset ... --require-encoder-equivalent
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.latent_parity import (  # noqa: E402
    DEPLOY_RELEASE_OBSERVATION_CONFIG,
    SONIC_RELEASE_TOKEN_CONTRACT,
    LatentParityReport,
    check_latent_parity,
    deploy_observation_enabled,
    describe_contract,
    read_deploy_encoder_dimension,
)

MOTION_TOKEN_COLUMN = "action.motion_token"
TRAJECTORY_FIELD = "action_motion_token"


def load_from_trajectory(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
    if not isinstance(payload, dict) or TRAJECTORY_FIELD not in payload:
        raise ValueError(f"{path} does not contain '{TRAJECTORY_FIELD}'")
    return np.asarray(payload[TRAJECTORY_FIELD])


def load_from_dataset(path: Path) -> np.ndarray:
    import pandas as pd

    parquet_files = sorted((path / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no parquet files found under {path / 'data'}")

    rows: list[np.ndarray] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        if MOTION_TOKEN_COLUMN not in frame.columns:
            raise ValueError(f"{parquet_path} is missing column '{MOTION_TOKEN_COLUMN}'")
        for value in frame[MOTION_TOKEN_COLUMN]:
            array = np.asarray(value)
            if array.dtype == object and array.size == 1:
                array = np.asarray(array.reshape(-1)[0])
            rows.append(np.asarray(array, dtype=np.float64).reshape(-1))
    widths = {row.shape[0] for row in rows}
    if len(widths) != 1:
        raise ValueError(f"inconsistent motion token widths across rows: {sorted(widths)}")
    return np.stack(rows, axis=0)


def check_deploy_config_binding(config_path: Path) -> tuple[list[str], list[str]]:
    """Confirm the C++ runtime still declares the token width this gate assumes."""
    errors: list[str] = []
    notes: list[str] = []
    if not config_path.is_file():
        errors.append(f"deploy observation config not found: {config_path}")
        return errors, notes

    dimension = read_deploy_encoder_dimension(config_path)
    if dimension != SONIC_RELEASE_TOKEN_CONTRACT.total_dim:
        errors.append(
            f"deploy encoder.dimension={dimension} does not match the token contract "
            f"total_dim={SONIC_RELEASE_TOKEN_CONTRACT.total_dim}"
        )
    else:
        notes.append(f"deploy encoder.dimension={dimension} matches the token contract")

    if not deploy_observation_enabled(config_path, "token_state"):
        errors.append(f"'token_state' observation is not enabled in {config_path}")
    else:
        notes.append("deploy config enables the 'token_state' observation")
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path, help="raw recorder pickle")
    source.add_argument("--dataset", type=Path, help="exported LeRobot dataset directory")
    parser.add_argument(
        "--require-encoder-equivalent",
        action="store_true",
        help=(
            "fail unless every token lies on the FSQ lattice. Use this for datasets that "
            "will be mixed with real teleop episodes recorded from the deploy encoder."
        ),
    )
    parser.add_argument(
        "--lattice-atol",
        type=float,
        default=1e-6,
        help="absolute tolerance for calling a value on-lattice (default: 1e-6)",
    )
    parser.add_argument(
        "--deploy-observation-config",
        type=Path,
        default=REPO_ROOT / DEPLOY_RELEASE_OBSERVATION_CONFIG,
        help="C++ deploy observation config to cross-check the token width against",
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    source_path = args.trajectory if args.trajectory is not None else args.dataset
    try:
        if args.trajectory is not None:
            tokens = load_from_trajectory(args.trajectory)
        else:
            tokens = load_from_dataset(args.dataset)
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        print(f"ERROR: could not load motion tokens: {exc}", file=sys.stderr)
        return 2

    binding_errors, binding_notes = check_deploy_config_binding(args.deploy_observation_config)

    report = check_latent_parity(
        tokens,
        lattice_atol=args.lattice_atol,
        require_encoder_equivalent=args.require_encoder_equivalent,
    )

    result = {
        "source": str(source_path.resolve()),
        "contract": describe_contract(),
        "deploy_binding": {
            "config": str(args.deploy_observation_config),
            "errors": binding_errors,
            "notes": binding_notes,
        },
        **report.to_dict(),
    }
    if binding_errors:
        result["errors"] = list(result["errors"]) + binding_errors

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for note in binding_notes:
        print(f"NOTE: {note}")
    for note in report.notes:
        print(f"NOTE: {note}")
    print(f"VERDICT: {report.verdict}")
    if report.verdict == LatentParityReport.RESIDUAL_PERTURBED:
        print(
            "NOTE: this latent is the correct decoder input for a VLA that drives the "
            "decoder directly, but it is NOT interchangeable with encoder-sourced "
            "token_state recorded from a real teleop session."
        )
    for error in binding_errors:
        print(f"ERROR: {error}")
    for error in report.errors:
        print(f"ERROR: {error}")

    ok = report.ok and not binding_errors
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
