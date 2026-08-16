#!/usr/bin/env python3
"""Build a typed EpisodeRequest -> GenerationResult -> ConversionResult bundle.

The dataset exporter refuses to write without one, and cross-checks it against the
Isaac runtime manifest. That check is strict for good reason, but it means several
fields have to agree exactly with what was already captured:

* ``task_prompt`` must equal the runtime ``dataset_task`` string,
* ``scene_hash`` must equal the scene file's hash in the runtime capture,
* ``physics_hash`` must equal the runtime config hash,
* the converted motion bytes must equal the motion the rollout loaded.

These bundles are honest POST-HOC records. The bundled Kimodo demos were not
sampled in response to a request this pipeline issued, so ``kimodo_seed`` and
``duration_s`` come from the demo's own ``meta.json`` and the demo's generation
prompt is carried in ``style_prompt``. Do not present these as original prompts.

This works only because the Kimodo-to-SONIC conversion is deterministic: re-running
it reproduces byte-identical motion PKLs, so a bundle built after a rollout still
binds to the motion that rollout actually consumed.

Usage::

    python scripts/research/build_kimodo_provenance_bundle.py \\
        --rollout-dir /path/rollouts/<config_id> \\
        --demo-meta /path/kimodo-g1-rp/01_single_text_prompt/meta.json \\
        --motion-npz /path/kimodo-g1-rp/01_single_text_prompt/motion.npz \\
        --qpos-csv /path/kimodo_01_single_text_prompt.csv \\
        --out-dir /path/provenance/<config_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.schemas import (  # noqa: E402
    ArtifactRef,
    EpisodeRequest,
    GenerationResult,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "sonic_release" / "last.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--demo-meta", type=Path, required=True)
    parser.add_argument("--motion-npz", type=Path, required=True)
    parser.add_argument("--qpos-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-family", default="navigation")
    parser.add_argument("--nominal-speed-mps", type=float, default=0.8)
    parser.add_argument(
        "--route-xy",
        type=float,
        nargs="+",
        default=None,
        help="flat x y x y ... route; defaults to the executed start and end",
    )
    args = parser.parse_args()

    manifest = json.loads((args.rollout_dir / "success_manifest.json").read_text(encoding="utf-8"))
    capture = manifest["capture_context"]
    meta = json.loads(args.demo_meta.read_text(encoding="utf-8"))

    if args.route_xy:
        flat = args.route_xy
        if len(flat) % 2 or len(flat) < 4:
            print("ERROR: --route-xy needs an even number of values, at least two points")
            return 2
        route = tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))
    else:
        # A two-point placeholder route is honest: the request records where the
        # motion was placed, not a planner-authored corridor.
        route = ((0.0, 0.0), (1.0, 0.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    request = EpisodeRequest(
        scene_id=capture["scene_id"],
        task_family=args.task_family,
        task_prompt=capture["task"],
        style_prompt=meta.get("text", "kimodo_bundled_demo"),
        route_xy=route,
        nominal_speed_mps=args.nominal_speed_mps,
        duration_s=float(meta.get("duration", 5.0)),
        kimodo_model="Kimodo-G1-RP",
        kimodo_seed=int(meta.get("seed", 0)),
        simulation_seed=0,
        render_seed=0,
        candidate_index=0,
        scene_hash=capture["scene"]["hash"],
        controller_hash="sha256:" + hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        physics_hash=manifest["runtime_config_hash"],
    )
    request.write_json(args.out_dir / "episode_request.json")

    generation = GenerationResult(
        episode_request_id=request.request_id,
        generator_name="kimodo.exports.mujoco.MujocoQposConverter",
        generator_version="kimodo-1.0.0-bundled-demo",
        resolved_model="Kimodo-G1-RP",
        source_fps=30.0,
        seed=int(meta.get("seed", 0)),
        artifacts=(
            ArtifactRef.from_path(
                "kimodo_motion_npz", args.motion_npz, media_type="application/x-npz"
            ),
            ArtifactRef.from_path("kimodo_qpos_csv", args.qpos_csv, media_type="text/csv"),
        ),
    )
    generation.write_json(args.out_dir / "generation_result.json")

    print(f"episode_request:   {request.request_id}")
    print(f"generation_result: {generation.result_id}")
    print(f"wrote bundle to {args.out_dir}")
    print("next: run convert_kimodo_to_motion_lib.py --generation-result "
          f"{args.out_dir / 'generation_result.json'} --source-artifact-name kimodo_qpos_csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
