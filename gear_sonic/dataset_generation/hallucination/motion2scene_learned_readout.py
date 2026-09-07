"""Hash-pinned CPU outcome readout without changing the controller's random stream."""

import hashlib
from pathlib import Path
import time

import torch

from .motion2scene_outcome_learner import FEATURE_COUNT, OutcomePredictor, select_action


def load_readout(path, expected_sha256):
    path = Path(path)
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("policy checkpoint hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # Initialization is overwritten by the checkpoint and must not perturb SONIC RNG.
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = OutcomePredictor()
        model.load_state_dict(saved["model"], strict=True)
    model.eval()
    mean = torch.as_tensor(saved["mean"], dtype=torch.float32, device="cpu")
    std = torch.as_tensor(saved["std"], dtype=torch.float32, device="cpu")
    if mean.shape != (FEATURE_COUNT,) or std.shape != mean.shape:
        raise ValueError("policy normalization dimensions differ")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid policy normalization")
    return model, mean, std


def readout(policy, features):
    model, mean, std = policy
    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if x.shape != (FEATURE_COUNT,) or not torch.isfinite(x).all():
        raise ValueError("214 finite causal values required")
    started = time.perf_counter()
    with torch.no_grad():
        p = model(((x - mean) / std).unsqueeze(0)).sigmoid()[0].numpy()
    seconds = time.perf_counter() - started
    selected = int(select_action(p))
    return {
        "probabilities": p.tolist(),
        "selected_action": selected,
        "requested_action": max(selected, 0),
        "refusal": selected == -1,
        "inference_seconds": seconds,
        "fallback": "neutral_commitment_not_stop" if selected == -1 else None,
    }
