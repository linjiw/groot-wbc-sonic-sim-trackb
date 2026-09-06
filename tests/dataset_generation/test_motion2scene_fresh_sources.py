"""Acquisition lineage must fail closed before evaluating a selected subset."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from generate_kimodo_motions import slugify  # noqa: E402
from motion2scene_fresh_sources import seed_mentions, validate_generation  # noqa: E402


def test_inventory_checks_nested_seeds_and_implicit_ranges():
    assert seed_mentions({"rows": [{"generation_seed": 43001}]}, [43001]) == {43001}
    assert seed_mentions({"seed_base": 42001, "seeds_per_prompt": 8, "prompts": 2}, [43007]) == {
        43007
    }
    assert not seed_mentions(
        {
            "seed_base": 42001,
            "seeds_per_prompt": 8,
            "prompts": 2,
            "seed_design": "shared_across_prompts",
        },
        [43007],
    )
    assert not seed_mentions({"time": 0.43001, "label": "43001"}, [43001])


@pytest.fixture
def bank(tmp_path):
    prompt = "A person walks at a steady pace in a straight line"
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(prompt)
    folder = tmp_path / "generated"
    folder.mkdir()
    ids = list(range(43001, 43009))
    for i, seed in enumerate(ids):
        stem = f"000_{slugify(prompt)}_s{i}"
        np.savetxt(folder / f"{stem}.csv", np.zeros((120, 36)), delimiter=",")
        (folder / f"{stem}.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "generation_seed": seed,
                    "prompt": prompt,
                    "num_frames": 120,
                    "fps": 30,
                    "denoising_steps": 100,
                    "model": "kimodo-g1-rp",
                    "seed_design": "independent_per_prompt",
                    "prompt_design_version": "motion2scene_fresh_source_v1",
                    "csv": f"{stem}.csv",
                    "qpos_convention": "mujoco_z_up_x_forward_wxyz_root",
                }
            )
        )
    (folder / "generation_report.json").write_text(json.dumps({"generated": 8, "failed": 0}))
    return folder, {"prompt": {"path": str(prompt_path)}, "source_ids": ids}


def test_partial_csv_set_rejected_even_if_success_report(bank):
    folder, reg = bank
    assert len(validate_generation(folder, reg)) == 8
    next(folder.glob("*.csv")).unlink()
    with pytest.raises(ValueError, match="CSV set"):
        validate_generation(folder, reg)


def test_stale_source_identity_rejected(bank):
    folder, reg = bank
    side = next(p for p in folder.glob("*.json") if p.name != "generation_report.json")
    value = json.loads(side.read_text())
    value["generation_seed"] = 41001
    side.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_generation(folder, reg)


def test_nonfinite_output_rejected(bank):
    folder, reg = bank
    csv = next(folder.glob("*.csv"))
    qpos = np.loadtxt(csv, delimiter=",")
    qpos[4, 7] = np.nan
    np.savetxt(csv, qpos, delimiter=",")
    with pytest.raises(ValueError, match="qpos"):
        validate_generation(folder, reg)
