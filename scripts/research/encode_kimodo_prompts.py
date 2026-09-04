#!/usr/bin/env python3
"""Encode motion prompts once into a cache that generation can reuse.

Loads LLM2Vec (Llama-3-8B, ~15 GB in bfloat16), encodes every prompt, writes the cache,
and exits. Nothing else should ever pay that load again: ``generate_kimodo_motions.py``
reads the cache instead.

Runs on CPU by default. That is slower per prompt but leaves the whole GPU to motion
generation, which is the resource the corpus is actually bottlenecked on -- and it means
the encode stage can run alongside a generation batch rather than waiting for it.

Must run in the Kimodo sm_120 environment (see install_scripts/install_kimodo_sm120.sh),
not the Isaac Lab environment.

Usage::

    python scripts/research/encode_kimodo_prompts.py \\
        --prompts prompts.txt --out /data/.../prompt_cache.npz [--device cuda]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.kimodo_prompt_cache import (  # noqa: E402
    PromptEmbedding,
    load_prompt_cache,
    save_prompt_cache,
)


def read_prompts(path: Path) -> list[str]:
    """One prompt per line. Blank lines and ``#`` comments are skipped."""
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    # Preserve order but drop duplicates: encoding the same string twice wastes minutes.
    return list(dict.fromkeys(prompts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
        help="encoder dtype; auto uses float32 on CPU and bfloat16 on CUDA",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep prompts already in --out and only encode the new ones",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        # This must happen before the first ``kimodo`` import below. Importing
        # ``kimodo.sanitize`` initializes torch through kimodo.__init__, after which changing
        # CUDA_VISIBLE_DEVICES is too late to prevent the encoder loader from seeing the GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Kimodo sanitizes prompts (capitalisation, trailing period) before handing them to
    # the encoder, so the model looks the sanitized string up. Caching the raw string
    # would miss on every prompt. Using Kimodo's own function rather than reimplementing
    # the rules keeps the key correct if those rules ever change.
    from kimodo.sanitize import sanitize_texts

    prompts = sanitize_texts(read_prompts(args.prompts))
    existing: dict[str, PromptEmbedding] = {}
    if args.append and args.out.exists():
        existing = load_prompt_cache(args.out)
        print(f"loaded {len(existing)} cached prompt(s) from {args.out}")

    todo = [p for p in prompts if p not in existing]
    if not todo:
        print(f"all {len(prompts)} prompt(s) already cached; nothing to do")
        return 0
    print(f"encoding {len(todo)} of {len(prompts)} prompt(s) on {args.device}")

    from kimodo.model import LLM2VecEncoder

    started = time.monotonic()
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if args.device == "cuda" else "float32"
    encoder = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype=dtype,
        llm_dim=4096,
    )
    encoder.to(args.device)
    print(f"encoder ready in {time.monotonic() - started:.1f} s")

    embeddings = list(existing.values())
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        tick = time.monotonic()
        features, lengths = encoder(chunk)
        features = features.float().cpu().numpy()
        for prompt, block, length in zip(chunk, features, lengths):
            embeddings.append(
                PromptEmbedding(
                    prompt=prompt,
                    features=np.asarray(block, dtype=np.float32),
                    length=int(length),
                )
            )
        done = min(start + args.batch_size, len(todo))
        print(f"  {done}/{len(todo)}  ({time.monotonic() - tick:.1f} s for {len(chunk)})")

    path = save_prompt_cache(args.out, embeddings)
    dim = embeddings[0].features.shape[-1]
    print(f"wrote {len(embeddings)} embedding(s) of dim {dim} to {path}")
    print(f"total {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
