#!/usr/bin/env python3
"""Change B launch precondition: dry-run the schedule_dict termination path (plan §3.2).

The M5-T threshold curriculum drives Isaac Lab termination thresholds through the
trainer's ``schedule_dict`` engine (``trl/utils/scheduler.py``). The engine's path
root is the TRAINER (``update_scheduled_params(self, ...)`` at
``ppo_trainer.py:1702-1706``), and the exact attribute chain from the trainer to
``termination_manager`` depends on the wrapper stack — it MUST be verified against
the live object before any M5-T run (hard launch precondition; a wrong path either
throws at iteration 0 or, worse, silently schedules nothing).

What this tool does, given a live trainer (or any root object):
1. resolves each candidate path with the scheduler's own ``_navigate_object_path``
   (same code that will run in training — no reimplementation drift);
2. round-trips a value: applies a one-step ``update_scheduled_params`` schedule,
   asserts the value visible through the termination manager changed, restores
   the original;
3. reports the first fully-verified path per term, formatted as ready-to-paste
   ``schedule_dict`` keys.

Isaac Lab facts this relies on (verified in source):
- ``TerminationManager.get_term_cfg(name)`` returns the live term cfg
  (``termination_manager.py:231``);
- ``compute()`` re-reads ``term_cfg.params`` every step (``:168``), so a
  scheduled ``params['threshold']`` takes effect immediately.

Usage on robotixx (env_isaaclab), one-off after trainer construction — e.g. add
a temporary two-liner right before ``trainer.train()`` in a scratch copy of the
train entry, or run in an interactive session::

    from scripts.research.verify_schedule_path import verify_candidate_paths
    print(verify_candidate_paths(
        trainer,
        terms=("anchor_pos", "ee_body_pos"),
        parameters=("threshold", "down_threshold"),
    ))

Offline self-test (this checkout, no Isaac Lab): ``--self-test`` builds a mock
object chain shaped like trainer->wrapper->gym->unwrapped->termination_manager
and asserts resolution + round-trip + restore all work through the real
scheduler engine. Tests in tests/research/test_verify_schedule_path.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from gear_sonic.trl.utils import scheduler as _scheduler  # noqa: E402

# Candidate chains from the trainer root to the Isaac Lab env holding
# termination_manager. Ordered most-likely first (manager_env_wrapper.py:35-38
# shows the wrapper reaches the base env as either self.env or
# self.env.env.unwrapped depending on the stack).
_CANDIDATE_CHAINS = (
    "env@env@unwrapped",
    "env@env",
    "env@env@env@unwrapped",
    "env@unwrapped",
    "env",
)


def _path_for(chain: str, term: str, parameter: str = "threshold") -> str:
    # NOTE the bracket syntax on the last segment: term_cfg.params is a dict, and
    # the scheduler engine resolves bare '@name' segments with getattr — a plain
    # '@params@threshold' tail raises AttributeError on the dict. This exact
    # trap is why the plan (§3.2) demands a live dry-run before any M5-T launch.
    return f"{chain}@termination_manager@get_term_cfg('{term}')@params['{parameter}']"


def _read_parameter(root: Any, chain: str, term: str, parameter: str) -> float:
    manager = _scheduler._navigate_object_path(root, f"{chain}@termination_manager")
    return float(manager.get_term_cfg(term).params[parameter])


def verify_one_path(
    root: Any,
    chain: str,
    term: str,
    *,
    parameter: str = "threshold",
    probe_value: float = 0.987,
) -> dict[str, Any]:
    """Resolve + round-trip one candidate path. Never leaves the value mutated."""
    path = _path_for(chain, term, parameter)
    record: dict[str, Any] = {
        "path": path,
        "parameter": parameter,
        "resolved": False,
        "roundtrip": False,
    }
    try:
        current = _scheduler._navigate_object_path(root, path)
    except Exception as err:  # noqa: BLE001 — any failure means "not this chain"
        record["error"] = f"{type(err).__name__}: {err}"
        return record
    record["resolved"] = True
    record["current_value"] = float(current)

    original = _read_parameter(root, chain, term, parameter)
    schedule = {path: {"type": "segment", "seg_steps": [0], "seg_vals": [probe_value]}}
    try:
        _scheduler.update_scheduled_params(root, schedule, step=0)
        seen = _read_parameter(root, chain, term, parameter)
        record["roundtrip"] = abs(seen - probe_value) < 1e-9
        if not record["roundtrip"]:
            record["error"] = (
                f"set through the scheduler but the termination manager still reads {seen} "
                "(path resolves to a copy, not the live cfg)"
            )
    finally:
        restore = {path: {"type": "segment", "seg_steps": [0], "seg_vals": [original]}}
        _scheduler.update_scheduled_params(root, restore, step=0)
    record["restored_value"] = _read_parameter(root, chain, term, parameter)
    return record


def verify_candidate_paths(
    root: Any,
    *,
    terms: tuple[str, ...] = ("anchor_pos", "ee_body_pos"),
    parameters: tuple[str, ...] = ("threshold",),
) -> dict[str, Any]:
    """Try every candidate chain per term; return verified paths + full trace."""
    result: dict[str, Any] = {"verified_paths": {}, "trace": {}}
    for term in terms:
        for parameter in parameters:
            label = term if parameters == ("threshold",) else f"{term}.{parameter}"
            records = []
            for chain in _CANDIDATE_CHAINS:
                record = verify_one_path(
                    root,
                    chain,
                    term,
                    parameter=parameter,
                )
                records.append(record)
                if record["roundtrip"]:
                    result["verified_paths"][label] = record["path"]
                    break
            result["trace"][label] = records
    expected = {
        term if parameters == ("threshold",) else f"{term}.{parameter}"
        for term in terms
        for parameter in parameters
    }
    result["ok"] = set(result["verified_paths"]) == expected
    if result["ok"]:
        result["schedule_dict_keys"] = {term: path for term, path in result["verified_paths"].items()}
    return result


# ---------------------------------------------------------------------------
# Offline self-test scaffolding (no Isaac Lab)
# ---------------------------------------------------------------------------


class _MockTermCfg:
    def __init__(self, threshold: float):
        self.params = {"threshold": threshold, "down_threshold": 5.0 * threshold}


class _MockTerminationManager:
    def __init__(self):
        self._cfgs = {"anchor_pos": _MockTermCfg(0.15), "ee_body_pos": _MockTermCfg(0.15)}

    def get_term_cfg(self, name: str) -> _MockTermCfg:
        if name not in self._cfgs:
            raise ValueError(f"Termination term '{name}' not found.")
        return self._cfgs[name]


class _Obj:
    """Attribute bag for building arbitrary wrapper chains."""

    def __init__(self, **attrs: Any):
        for key, value in attrs.items():
            setattr(self, key, value)


def build_mock_trainer() -> Any:
    """Trainer-shaped mock: trainer.env.env.unwrapped.termination_manager
    (the most-likely live chain per manager_env_wrapper.py:35-38)."""
    unwrapped = _Obj(termination_manager=_MockTerminationManager())
    gym_env = _Obj(unwrapped=unwrapped)
    wrapper = _Obj(env=gym_env)
    return _Obj(env=wrapper)


def _self_test() -> dict[str, Any]:
    return verify_candidate_paths(build_mock_trainer(), parameters=("threshold", "down_threshold"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not args.self_test:
        parser.error(
            "this CLI only runs --self-test; on robotixx import verify_candidate_paths "
            "and call it with the live trainer (see module docstring)"
        )
    result = _self_test()
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
