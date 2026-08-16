from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from gear_sonic.dataset_generation.scene_asset_preflight import (
    DEFAULT_SCENE_PACKAGE_DIR,
    preflight_scene_package,
)
from scripts.research.check_g1_dataset_scenes import main as checker_main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "g1_dataset"
    shutil.copytree(DEFAULT_SCENE_PACKAGE_DIR, package_dir)
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["license"]["source"] = str(REPO_ROOT / "LICENSE")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return package_dir


def test_repo_scene_package_passes_integrity_and_clearance_gate() -> None:
    report = preflight_scene_package()

    assert report.ok, report.to_dict()
    assert report.package_id == "g1_dataset_m0_scenes"
    assert {scene.scene_id for scene in report.scenes} == {
        "factory_aisle",
        "household_room",
    }
    by_id = {scene.scene_id: scene for scene in report.scenes}
    assert by_id["household_room"].collision_prim_count == 13
    assert by_id["household_room"].minimum_route_clearance_m == pytest.approx(0.8)
    assert by_id["factory_aisle"].collision_prim_count == 16
    assert by_id["factory_aisle"].minimum_route_clearance_m == pytest.approx(0.7)
    for scene in report.scenes:
        assert scene.minimum_route_clearance_m is not None
        assert scene.declared_route_clearance_m is not None
        assert scene.minimum_route_clearance_m >= scene.declared_route_clearance_m


def test_manifest_records_redistributable_repo_authored_provenance() -> None:
    manifest = json.loads((DEFAULT_SCENE_PACKAGE_DIR / "manifest.json").read_text())

    assert manifest["license"]["spdx"] == "Apache-2.0"
    assert manifest["license"]["redistribution_allowed"] is True
    assert manifest["provenance"]["kind"] == "repo_authored_primitive_geometry"
    assert manifest["provenance"]["third_party_assets"] == []
    assert all(scene["split_group"] for scene in manifest["scenes"])
    assert all(scene["sha256"].startswith("sha256:") for scene in manifest["scenes"])


def test_preflight_rejects_tampered_stage_axis_and_hash(tmp_path: Path) -> None:
    package_dir = _copy_package(tmp_path)
    scene_path = package_dir / "household_room.usda"
    text = scene_path.read_text(encoding="utf-8")
    scene_path.write_text(text.replace('upAxis = "Z"', 'upAxis = "Y"'), encoding="utf-8")

    report = preflight_scene_package(package_dir)
    household = next(scene for scene in report.scenes if scene.scene_id == "household_room")
    errors = "\n".join(household.errors)

    assert not report.ok
    assert "sha256 mismatch" in errors
    assert "upAxis must be Z" in errors


def test_preflight_rejects_manifest_route_through_furniture(tmp_path: Path) -> None:
    package_dir = _copy_package(tmp_path)
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    household = next(scene for scene in manifest["scenes"] if scene["scene_id"] == "household_room")
    household["route_xy"] = [[-3.2, -3.2], [-3.2, -2.0]]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = preflight_scene_package(package_dir)
    result = next(scene for scene in report.scenes if scene.scene_id == "household_room")

    assert not report.ok
    assert result.minimum_route_clearance_m == 0.0
    assert any("below declared radius" in error for error in result.errors)


def test_research_checker_reports_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert checker_main([str(DEFAULT_SCENE_PACKAGE_DIR)]) == 0

    output = capsys.readouterr().out
    assert "PASS: household_room: 13 collision prims" in output
    assert "PASS: factory_aisle: 16 collision prims" in output
    assert output.rstrip().endswith("PASS")
