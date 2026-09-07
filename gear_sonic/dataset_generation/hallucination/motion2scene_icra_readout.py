"""Shared readout adapter for frozen learned controls and explicitly privileged baselines."""

import hashlib
import json
from pathlib import Path

from .motion2scene_learned_readout import load_readout as load_learned, readout as read_learned


def load_readout(path, expected_sha256):
    path = Path(path)
    if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("readout hash mismatch")
    if path.suffix == ".pt":
        return {"kind": "learned", "value": load_learned(path, expected_sha256)}
    value = json.loads(path.read_text())
    if value["kind"] not in ("scripted_rays", "privileged_geometry"):
        raise ValueError("unsupported comparator")
    return value


def readout(policy, features, packet=None):
    if policy["kind"] == "learned":
        return read_learned(policy["value"], features)
    if policy["kind"] == "scripted_rays":
        if packet is None or not isinstance(packet["occupied"], bool):
            raise ValueError("scripted comparator requires the actual causal packet")
        action = int(packet["occupied"])
        selected = action
    else:
        selected = int(policy["selected_action"])
        if selected not in (-1, 0, 1):
            raise ValueError("invalid privileged action")
        action = max(selected, 0)
    return {
        "probabilities": None,
        "selected_action": selected,
        "requested_action": action,
        "refusal": selected == -1,
        "inference_seconds": 0.0,
        "fallback": "neutral_commitment_not_stop" if selected == -1 else None,
        "comparator_kind": policy["kind"],
    }
