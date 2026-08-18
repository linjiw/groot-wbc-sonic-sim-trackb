from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gear_sonic.research.lace.atlas import _validate_scientific_instrument
from gear_sonic.research.lace.instrument import (
    INSTRUMENT_DIGEST_FIELD,
    RUNTIME_RNG_SEED_SEMANTICS,
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    SOURCE_BUNDLE_DIGEST_FIELD,
    TERMINATION_FRESHNESS_SEMANTICS,
    TERMINATION_LEGACY_DIAGNOSTIC_SEMANTICS,
    TERMINATION_MULTI_HOT_SEMANTICS,
    TERMINATION_RAW_TRACE_SEMANTICS,
    TERMINATION_TRACE_ALGORITHM,
    build_measurement_family,
    build_scientific_instrument,
    build_source_bundle,
    episode_instrument_sha256,
    validate_measurement_family,
    validate_scientific_instrument,
    validate_source_bundle,
)
from gear_sonic.research.lace.schedule import DOMAIN_RANDOMIZATION_SEED_SEMANTICS
from gear_sonic.research.lace.schema import (
    TERMINATION_TRACE_KIND,
    TERMINATION_TRACE_SCHEMA_VERSION,
    canonical_sha256,
)
from scripts.research.build_lace_instrument import main as instrument_cli_main


def _termination_contract() -> dict:
    term_configs = [
        {
            "term_name": "anchor_pos",
            "callable": "gear_sonic.envs.manager_env.mdp.terminations:bad_anchor_pos",
            "time_out": False,
            "params": {"threshold": 0.5},
        },
        {
            "term_name": "time_out",
            "callable": "isaaclab.envs.mdp.terminations:time_out",
            "time_out": True,
            "params": {},
        },
    ]
    return {
        "kind": TERMINATION_TRACE_KIND,
        "schema_version": TERMINATION_TRACE_SCHEMA_VERSION,
        "algorithm": TERMINATION_TRACE_ALGORITHM,
        "manager_type": "isaaclab.managers.termination_manager:TerminationManager",
        "manager_compute_source_sha256": "a" * 64,
        "instrument_compute_source_sha256": "b" * 64,
        "term_names": ["anchor_pos", "time_out"],
        "time_out_flags": [False, True],
        "term_config_sha256": canonical_sha256({"terms": term_configs}),
        "term_configs": term_configs,
        "legacy_term_dones_semantics": TERMINATION_LEGACY_DIAGNOSTIC_SEMANTICS,
        "raw_trace_semantics": TERMINATION_RAW_TRACE_SEMANTICS,
        "freshness_semantics": TERMINATION_FRESHNESS_SEMANTICS,
    }


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "gear_sonic" / "research" / "lace").mkdir(parents=True)
    (root / "gear_sonic" / "research" / "lace" / "probes.py").write_bytes(b"probe-v1\n")
    (root / "gear_sonic" / "research" / "lace" / "isaac_recorder.py").write_bytes(b"recorder-v1\n")
    return root


def _build(root: Path) -> dict:
    cache_suffixes = (
        "tmp",
        "xdg-cache",
        "isaaclab-usd-cache",
        "cuda-cache",
        "torch-home",
        "omni-user-cache",
    )
    return build_scientific_instrument(
        schedule_sha256="0" * 64,
        probe_thresholds={"slip_speed_threshold": 0.15, "score_window_seconds": 2.0},
        recorder_config={"enabled": True, "max_episode_steps": 10000},
        resolved_hydra_config={"env": {"num_envs": 512}, "headless": True},
        environment_fingerprint={
            "python": "3.11.9",
            "isaaclab": "2.3.2",
            "isaacsim": "5.0.0",
            "torch": "2.7.0",
            "scientific_cache_environment": {
                key: f"/data/lace/instrument-test/{suffix}"
                for key, suffix in zip(
                    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
                    cache_suffixes,
                    strict=True,
                )
            },
            "scientific_cache_environment_semantics": (SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS),
            "process_argv": ["gear_sonic/eval_agent_trl.py", "++cell=first"],
        },
        sensor_semantics={"contact": "net_forces_w_history", "velocity": "link_origin_w"},
        termination_predicates=_termination_contract(),
        score_window_config={
            "rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
            "seconds": 2.0,
        },
        domain_randomization_config={
            "mode_order": ["startup", "reset"],
            "interval_events_instrumented": False,
        },
        git_commit="9" * 40,
        repo_root=root,
        source_paths=[
            "gear_sonic/research/lace/probes.py",
            "gear_sonic/research/lace/isaac_recorder.py",
        ],
    )


def _refresh_instrument_digest(instrument: dict) -> None:
    instrument[INSTRUMENT_DIGEST_FIELD] = canonical_sha256(
        instrument,
        digest_field=INSTRUMENT_DIGEST_FIELD,
    )


def test_builder_is_deterministic_source_bound_and_atlas_compatible(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = _build(root)
    second = _build(root)

    assert first == second
    assert [row["path"] for row in first["source_bundle"]["files"]] == [
        "gear_sonic/research/lace/isaac_recorder.py",
        "gear_sonic/research/lace/probes.py",
    ]
    assert first["termination_semantics"] == TERMINATION_MULTI_HOT_SEMANTICS
    assert first["runtime_rng_seed_semantics"] == RUNTIME_RNG_SEED_SEMANTICS
    assert first["domain_randomization_seed_semantics"] == DOMAIN_RANDOMIZATION_SEED_SEMANTICS
    validate_scientific_instrument(first, repo_root=root)

    atlas_instrument, atlas_digest = _validate_scientific_instrument(
        first,
        episodes=[],
        expected_schedule_sha256="0" * 64,
    )
    assert atlas_instrument == first
    assert atlas_digest == episode_instrument_sha256(first)
    assert atlas_digest == canonical_sha256(first)


def test_payload_and_self_hash_tampering_are_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    instrument = _build(root)

    tampered = deepcopy(instrument)
    tampered["probe_thresholds"]["slip_speed_threshold"] = 0.99
    _refresh_instrument_digest(tampered)
    with pytest.raises(ValueError, match="does not bind probe_thresholds"):
        validate_scientific_instrument(tampered, repo_root=root)

    tampered = deepcopy(instrument)
    tampered[INSTRUMENT_DIGEST_FIELD] = "f" * 64
    with pytest.raises(ValueError, match=INSTRUMENT_DIGEST_FIELD):
        validate_scientific_instrument(tampered, repo_root=root)


def test_source_bundle_rejects_duplicates_escape_symlink_and_byte_drift(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    source = "gear_sonic/research/lace/probes.py"

    with pytest.raises(ValueError, match="duplicate source path"):
        build_source_bundle(root, [source, source])
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside repository"):
        build_source_bundle(root, [outside])

    link = root / "linked.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        build_source_bundle(root, [link])

    bundle = build_source_bundle(root, [source])
    (root / source).write_text("probe-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest drifted"):
        validate_source_bundle(bundle, repo_root=root)


def test_source_bundle_rejects_noncanonical_order_even_if_rehashed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    bundle = build_source_bundle(
        root,
        [
            "gear_sonic/research/lace/probes.py",
            "gear_sonic/research/lace/isaac_recorder.py",
        ],
    )
    bundle["files"].reverse()
    bundle[SOURCE_BUNDLE_DIGEST_FIELD] = canonical_sha256(
        bundle,
        digest_field=SOURCE_BUNDLE_DIGEST_FIELD,
    )

    with pytest.raises(ValueError, match="canonical order"):
        validate_source_bundle(bundle, repo_root=root)


def test_git_commit_without_source_bundle_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    instrument = _build(root)
    instrument.pop("source_bundle")
    instrument.pop("source_bundle_sha256")
    _refresh_instrument_digest(instrument)

    with pytest.raises(ValueError, match="fields do not match"):
        validate_scientific_instrument(instrument, repo_root=root)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("termination_multi_hot_available", False, "requires verified independent"),
        ("termination_semantics", "legacy_last_trigger", "ordered raw boolean matrix"),
        (
            "domain_randomization_seed_semantics",
            "declared_seed_only",
            "do not match the schedule schema",
        ),
        ("runtime_rng_seed_semantics", "seed_requested_not_read_back", "semantics are invalid"),
    ],
)
def test_inconsistent_scientific_semantics_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    root = _repo(tmp_path)
    instrument = _build(root)
    instrument[field] = value
    _refresh_instrument_digest(instrument)

    with pytest.raises(ValueError, match=match):
        validate_scientific_instrument(instrument, repo_root=root)


def test_unverified_termination_contract_cannot_claim_multi_hot(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    kwargs = {
        "schedule_sha256": "0" * 64,
        "probe_thresholds": {"threshold": 1.0},
        "recorder_config": {"enabled": True},
        "resolved_hydra_config": {"env": {"num_envs": 1}},
        "environment_fingerprint": {"isaaclab": "2.3.2"},
        "sensor_semantics": {"contact": "forces"},
        "termination_predicates": _termination_contract(),
        "score_window_config": {"seconds": 2.0},
        "domain_randomization_config": {"events": []},
        "git_commit": "9" * 40,
        "repo_root": root,
        "source_paths": ["gear_sonic/research/lace/probes.py"],
    }
    kwargs["termination_predicates"]["raw_trace_semantics"] = "cached_last_trigger"

    with pytest.raises(ValueError, match="single-evaluation raw trace"):
        build_scientific_instrument(**kwargs)


def test_resolved_hydra_payload_rejects_interpolation(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    instrument = _build(root)
    instrument["resolved_hydra_config"]["checkpoint"] = "${oc.env:CHECKPOINT}"
    instrument["resolved_hydra_config_sha256"] = canonical_sha256(
        instrument["resolved_hydra_config"]
    )
    _refresh_instrument_digest(instrument)

    with pytest.raises(ValueError, match="still contains"):
        validate_scientific_instrument(instrument, repo_root=root)


def test_measurement_family_masks_explicit_timestamp_bookkeeping_only(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = _build(root)
    second = deepcopy(first)
    second["environment_fingerprint"]["scientific_cache_environment"] = {
        key: value.replace("/instrument-test/", "/instrument-test-second/")
        for key, value in second["environment_fingerprint"]["scientific_cache_environment"].items()
    }
    second["environment_fingerprint"]["process_argv"] = [
        "gear_sonic/eval_agent_trl.py",
        "++cell=second",
    ]
    second["environment_sha256"] = canonical_sha256(second["environment_fingerprint"])
    for instrument, timestamp in ((first, "20260814_010203"), (second, "20260814_040506")):
        instrument["resolved_hydra_config"]["timestamp"] = timestamp
        instrument["resolved_hydra_config"]["eval_timestamp"] = timestamp
        instrument["resolved_hydra_config_sha256"] = canonical_sha256(
            instrument["resolved_hydra_config"]
        )
        _refresh_instrument_digest(instrument)
    first_family = build_measurement_family(first)
    second_family = build_measurement_family(second)
    assert first_family == second_family
    validate_measurement_family(first_family, instruments=[first, second])

    causal_drift = deepcopy(second)
    causal_drift["resolved_hydra_config"]["headless"] = False
    causal_drift["resolved_hydra_config_sha256"] = canonical_sha256(
        causal_drift["resolved_hydra_config"]
    )
    _refresh_instrument_digest(causal_drift)
    assert build_measurement_family(causal_drift) != first_family


def test_cli_writes_deterministic_pretty_json_and_refuses_overwrite(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    inputs = {
        "probe-thresholds": {"threshold": 1.0},
        "recorder-config": {"enabled": True},
        "resolved-hydra-config": {"env": {"num_envs": 1}},
        "environment-fingerprint": {"isaaclab": "2.3.2"},
        "sensor-semantics": {"contact": "forces"},
        "termination-predicates": _termination_contract(),
        "score-window-config": {"seconds": 2.0},
        "domain-randomization-config": {"events": []},
    }
    args = [
        "--schedule-sha256",
        "0" * 64,
        "--git-commit",
        "9" * 40,
        "--repo-root",
        str(root),
        "--source",
        "gear_sonic/research/lace/probes.py",
    ]
    for option, payload in inputs.items():
        path = tmp_path / f"{option}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        args.extend([f"--{option}", str(path)])
    output = tmp_path / "nested" / "instrument.json"
    args.extend(["--output", str(output)])

    assert instrument_cli_main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_scientific_instrument(payload, repo_root=root)
    expected = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert output.read_text(encoding="utf-8") == expected
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        instrument_cli_main(args)
