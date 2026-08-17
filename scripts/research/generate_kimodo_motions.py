#!/usr/bin/env python3
"""Generate G1 motions from prompts, writing the 36-column qpos CSVs the corpus ingests.

Uses the prompt cache from ``encode_kimodo_prompts.py`` instead of loading LLM2Vec, which
is what lets generation run on a shared GPU: the diffusion model needs about 2 GB, the
encoder it replaces needed about 15 GB.

The CSV convention is Kimodo's own ``MujocoQposConverter`` default -- z-up, +x forward,
w-first root quaternion, 7 root + 29 joints, 30 fps -- which is exactly the contract
``gear_sonic.dataset_generation.kimodo_motion_adapter`` validates on the way in. Nothing
here reimplements that conversion.

Every motion is written alongside a JSON sidecar recording the prompt, seed, model
snapshot and denoising steps, so a clip can always be traced back to what produced it.

Must run in the Kimodo sm_120 environment, not the Isaac Lab environment.

Usage::

    python scripts/research/generate_kimodo_motions.py \\
        --prompts prompts.txt --cache prompt_cache.npz --out /data/.../motions \\
        --seeds 3 --duration 5.0
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Kimodo generates at 30 fps; the model card caps a single segment at 10 s / 300 frames.
KIMODO_FPS = 30
KIMODO_MAX_FRAMES = 300


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return list(dict.fromkeys(prompts))


def slugify(prompt: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug[:limit].rstrip("_") or "prompt"


def register_cached_encoder(cache_path: Path, device: str) -> None:
    """Point Kimodo's text-encoder registry at the prompt cache."""
    # ``kimodo.model.__init__`` re-exports the ``load_model`` *function* under the same
    # name as its submodule, so attribute access -- including ``import a.b.c as x`` --
    # returns the function. import_module goes to sys.modules and gets the module, which
    # is where the preset table lives.
    load_model_module = importlib.import_module("kimodo.model.load_model")

    load_model_module.TEXT_ENCODER_PRESETS["cached"] = {
        "target": "gear_sonic.dataset_generation.kimodo_prompt_cache.cached_text_encoder_from_path",
        "kwargs": {"cache_path": str(cache_path), "device": device, "dtype": "float32"},
    }
    # "local" means "build from the preset table" -- it is the branch that reads
    # TEXT_ENCODER, and it never contacts the network.
    os.environ["TEXT_ENCODER_MODE"] = "local"
    os.environ["TEXT_ENCODER"] = "cached"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="kimodo-g1-rp")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds, max 10")
    parser.add_argument("--seeds", type=int, default=1, help="samples per prompt")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100, help="DDIM denoising steps")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    num_frames = int(round(args.duration * KIMODO_FPS))
    if not 0 < num_frames <= KIMODO_MAX_FRAMES:
        raise SystemExit(
            f"duration {args.duration}s is {num_frames} frames; the model card caps a "
            f"single segment at {KIMODO_MAX_FRAMES} frames (10 s)"
        )

    prompts = read_prompts(args.prompts)
    if not args.cache.exists():
        raise SystemExit(f"prompt cache not found: {args.cache} (run encode_kimodo_prompts.py)")
    register_cached_encoder(args.cache, args.device)

    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    import torch

    started = time.monotonic()
    model, resolved = load_model(args.model, device=args.device, return_resolved_name=True)
    converter = MujocoQposConverter(model.skeleton)
    print(f"loaded {resolved} in {time.monotonic() - started:.1f} s")
    print(f"generating {len(prompts)} prompt(s) x {args.seeds} seed(s) at {num_frames} frames")

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    failures: list[dict] = []

    for prompt_index, prompt in enumerate(prompts):
        for seed_index in range(args.seeds):
            seed = args.seed_base + prompt_index * 1000 + seed_index
            stem = f"{prompt_index:03d}_{slugify(prompt)}_s{seed_index}"
            csv_path = args.out / f"{stem}.csv"
            if csv_path.exists():
                print(f"  skip {stem} (already generated)")
                continue

            tick = time.monotonic()
            torch.manual_seed(seed)
            try:
                output = model(
                    prompts=prompt,
                    num_frames=num_frames,
                    num_denoising_steps=args.steps,
                    constraint_lst=[],
                    num_samples=1,
                    return_numpy=True,
                )
                qpos = converter.dict_to_qpos(output, args.device)
                converter.save_csv(qpos, str(csv_path))
            except Exception as error:  # noqa: BLE001 - one bad prompt must not end the sweep
                print(f"  FAIL {stem}: {type(error).__name__}: {error}")
                failures.append({"stem": stem, "prompt": prompt, "error": repr(error)})
                continue

            elapsed = time.monotonic() - tick
            sidecar = {
                "prompt": prompt,
                "seed": seed,
                "model": resolved,
                "num_frames": num_frames,
                "fps": KIMODO_FPS,
                "denoising_steps": args.steps,
                "generation_seconds": round(elapsed, 3),
                "csv": csv_path.name,
                "qpos_convention": "mujoco_z_up_x_forward_wxyz_root",
            }
            (args.out / f"{stem}.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            written.append(sidecar)
            print(f"  {stem}  {elapsed:.1f} s")

    report = {
        "generated": len(written),
        "failed": len(failures),
        "failures": failures,
        "prompts": len(prompts),
        "seeds_per_prompt": args.seeds,
        "num_frames": num_frames,
        "total_seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n{len(written)} motion(s) written, {len(failures)} failed, "
          f"{report['total_seconds']:.0f} s total -> {args.out}")
    return 1 if failures and not written else 0


if __name__ == "__main__":
    sys.exit(main())
