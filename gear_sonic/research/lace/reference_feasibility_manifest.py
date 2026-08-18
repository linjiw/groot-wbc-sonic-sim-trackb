"""CPU-only builder and validator for frozen LACE feasibility controls.

The builder parses the pinned SONIC G1 Python configuration with :mod:`ast`
instead of importing Isaac Lab, parses the configured URDF, reproduces the
release reference timeline and foot-contact heuristic, and loads only motions
selected by both the split and reference-length inventory.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
import yaml

from gear_sonic.research.lace.reference_feasibility import (
    REFERENCE_FEASIBILITY_KIND,
    REFERENCE_FEASIBILITY_SCHEMA_VERSION,
    REFERENCE_FEASIBILITY_SCOPE,
    ReferenceFeasibilityThresholds,
    compute_reference_feasibility_proxy,
)
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schema import (
    ROBOT_CONTRACT_READBACK_KIND,
    ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
    RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
    canonical_sha256,
    validate_split_manifest,
)

FloatArray = NDArray[np.float64]

REFERENCE_FEASIBILITY_MANIFEST_KIND = "lace_reference_feasibility_manifest"
REFERENCE_FEASIBILITY_MANIFEST_SCHEMA_VERSION = 1
REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD = "manifest_sha256"
G1_REFERENCE_CONTRACT_KIND = "lace_g1_reference_proxy_contract"
G1_REFERENCE_CONTRACT_SCHEMA_VERSION = 1
G1_REFERENCE_CONTRACT_DIGEST_FIELD = "robot_contract_sha256"
LIVE_ROBOT_EVIDENCE_BINDING_KIND = "lace_live_robot_contract_evidence"
LIVE_ROBOT_EVIDENCE_BINDING_SCHEMA_VERSION = 1

_G1_CFG_ASSIGNMENT = "G1_CYLINDER_MODEL_12_DEX_CFG"
_G1_BODY_NAMES_ASSIGNMENT = "G1_ISAACLAB_JOINTS"
_IL_TO_MJ_ASSIGNMENT = "G1_ISAACLAB_TO_MUJOCO_DOF"
_MJ_TO_IL_ASSIGNMENT = "G1_MUJOCO_TO_ISAACLAB_DOF"
_EXPECTED_FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
_LIVE_ROBOT_DATA_FIELDS = {
    "joint_names",
    "joint_pos_limits",
    "soft_joint_pos_limits",
    "joint_vel_limits",
    "soft_joint_vel_limits",
}
_HARD_CORRUPTION_CHECK_FIELDS = {
    "source_schema_and_file_hash_verified",
    "source_quaternion_norm_violation",
    "root_rotation_representation_mismatch",
    "root_orientation_step_discontinuity",
    "reference_contact_kinematic_rule_mismatch",
}
_REFERENCE_CONSTRAINT_FLAG_FIELDS = {
    "joint_hard_limit_violation",
    "joint_soft_limit_violation",
    "joint_velocity_limit_violation",
}
_IMPLEMENTATION_SOURCE_ROLES = {
    "proxy_implementation": Path(__file__).with_name("reference_feasibility.py"),
    "manifest_builder_validator": Path(__file__),
    "sonic_reference_resampling_source": Path(__file__).resolve().parents[3]
    / "gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
    "sonic_contact_rule_source": Path(__file__).resolve().parents[3]
    / "gear_sonic/utils/motion_lib/motion_lib_base.py",
    "sonic_rotation_math_source": Path(__file__).resolve().parents[3]
    / "gear_sonic/isaac_utils/rotations.py",
    "sonic_torch_transform_source": Path(__file__).resolve().parents[3]
    / "gear_sonic/trl/utils/torch_transform.py",
    "sonic_kornia_transform_source": Path(__file__).resolve().parents[3]
    / "gear_sonic/trl/utils/kornia_transform.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path, name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash {name} {path}: {error}") from error
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be lowercase hexadecimal") from error
    _require(value == value.lower(), f"{name} must be lowercase hexadecimal")
    return value


def _strict_json_loads(text: str, *, context: str) -> Any:
    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def _reject_nonfinite(value: str) -> None:
        raise ValueError(f"{context} contains non-standard numeric value {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is malformed JSON: {error}") from error


def _canonical_robot_contract_pair(payload: Mapping[str, Any], *, context: str) -> bytes:
    required = {"robot_contract_readback", "robot_contract_readback_sha256"}
    missing = sorted(required - set(payload))
    _require(not missing, f"{context} is missing required fields: {missing}")
    readback = payload["robot_contract_readback"]
    digest = payload["robot_contract_readback_sha256"]
    _require(
        isinstance(readback, Mapping),
        f"{context} robot_contract_readback must be a JSON object",
    )
    digest = _sha256(digest, f"{context} robot_contract_readback_sha256")
    try:
        realized_digest = canonical_sha256(readback)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} robot_contract_readback is not canonical JSON: {error}"
        ) from error
    _require(
        realized_digest == digest,
        f"{context} robot_contract_readback_sha256 mismatch",
    )
    pair = {
        "robot_contract_readback": readback,
        "robot_contract_readback_sha256": digest,
    }
    return json.dumps(
        pair,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_recorder_evidence_binding(
    binding: Any,
    *,
    verify_source_file: bool,
) -> dict[str, Any]:
    fields = {
        "format",
        "kind",
        "path",
        "row_count",
        "schema_version",
        "scientific_runtime_ready_all_rows",
        "sha256",
    }
    _require(
        isinstance(binding, Mapping) and set(binding) == fields,
        "recorder_evidence_binding fields are invalid",
    )
    _require(
        binding.get("kind") == LIVE_ROBOT_EVIDENCE_BINDING_KIND
        and binding.get("schema_version") == LIVE_ROBOT_EVIDENCE_BINDING_SCHEMA_VERSION,
        "unsupported recorder_evidence_binding schema",
    )
    evidence_format = binding.get("format")
    row_count = binding.get("row_count")
    _require(
        evidence_format in {"json_object", "jsonl"},
        "recorder_evidence_binding format is invalid",
    )
    _require(
        isinstance(row_count, Integral) and not isinstance(row_count, bool) and row_count > 0,
        "recorder_evidence_binding row_count must be a positive integer",
    )
    _require(
        evidence_format == "jsonl" or row_count == 1,
        "JSON-object recorder evidence must have exactly one row",
    )
    _require(
        isinstance(binding.get("scientific_runtime_ready_all_rows"), bool),
        "recorder evidence readiness summary must be boolean",
    )
    digest = _sha256(binding.get("sha256"), "recorder_evidence_binding.sha256")
    path = Path(str(binding.get("path")))
    _require(
        str(path) == str(path.resolve()),
        "recorder_evidence_binding path must be absolute and canonical",
    )
    if verify_source_file:
        _require(path.is_file(), f"recorder evidence source is missing: {path}")
        _require(
            _sha256_file(path, "recorder evidence source") == digest,
            "recorder evidence source SHA-256 mismatch",
        )
    return {
        "format": evidence_format,
        "kind": LIVE_ROBOT_EVIDENCE_BINDING_KIND,
        "path": str(path),
        "row_count": int(row_count),
        "schema_version": LIVE_ROBOT_EVIDENCE_BINDING_SCHEMA_VERSION,
        "scientific_runtime_ready_all_rows": binding["scientific_runtime_ready_all_rows"],
        "sha256": digest,
    }


def load_live_robot_evidence(path: str | Path) -> dict[str, Any]:
    """Load one recorder JSON object or strict homogeneous non-empty JSONL.

    JSONL rows may carry unrelated episode data, but every row must carry a
    canonically byte-equivalent robot-contract readback and matching digest.
    """

    resolved = Path(path).resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read live robot.data evidence {resolved}: {error}") from error
    _require(text.strip(), f"live robot.data evidence {resolved} is empty")

    payloads: list[dict[str, Any]]
    evidence_format: str
    try:
        whole_payload = _strict_json_loads(text, context=f"live robot.data evidence {resolved}")
    except ValueError:
        lines = text.splitlines()
        _require(
            lines and all(line.strip() for line in lines),
            f"live robot.data JSONL {resolved} must be non-empty and contain no blank rows",
        )
        payloads = []
        for line_number, line in enumerate(lines, start=1):
            row = _strict_json_loads(
                line,
                context=f"live robot.data JSONL {resolved} line {line_number}",
            )
            _require(
                isinstance(row, dict),
                f"live robot.data JSONL {resolved} line {line_number} must be a JSON object",
            )
            payloads.append(row)
        evidence_format = "jsonl"
    else:
        _require(
            isinstance(whole_payload, dict),
            f"live robot.data evidence {resolved} must be one JSON object or JSONL",
        )
        payloads = [whole_payload]
        evidence_format = "json_object"

    canonical_pair: bytes | None = None
    for row_index, payload in enumerate(payloads, start=1):
        context = (
            f"live robot.data evidence {resolved}"
            if evidence_format == "json_object"
            else f"live robot.data JSONL {resolved} line {row_index}"
        )
        row_pair = _canonical_robot_contract_pair(payload, context=context)
        if canonical_pair is None:
            canonical_pair = row_pair
        else:
            _require(
                row_pair == canonical_pair,
                f"live robot.data JSONL {resolved} contains mixed, non-byte-equivalent "
                "robot contract readback evidence",
            )

    first = payloads[0]
    evidence_binding = {
        "format": evidence_format,
        "kind": LIVE_ROBOT_EVIDENCE_BINDING_KIND,
        "path": str(resolved),
        "row_count": len(payloads),
        "schema_version": LIVE_ROBOT_EVIDENCE_BINDING_SCHEMA_VERSION,
        "scientific_runtime_ready_all_rows": all(
            payload.get("scientific_runtime_ready") is True for payload in payloads
        ),
        "sha256": _sha256_file(resolved, "recorder evidence source"),
    }
    return {
        "robot_contract_readback": first["robot_contract_readback"],
        "robot_contract_readback_sha256": first["robot_contract_readback_sha256"],
        "recorder_evidence_binding": _validate_recorder_evidence_binding(
            evidence_binding,
            verify_source_file=True,
        ),
    }


def _implementation_bindings() -> list[dict[str, str]]:
    bindings = []
    for role, raw_path in sorted(_IMPLEMENTATION_SOURCE_ROLES.items()):
        path = raw_path.resolve()
        _require(path.is_file(), f"implementation source does not exist: {path}")
        bindings.append(
            {
                "role": role,
                "path": str(path),
                "sha256": _sha256_file(path, f"implementation source {role}"),
            }
        )
    return bindings


def _runtime_dependency_versions() -> dict[str, str]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - SONIC runtime dependency
        raise ValueError("torch is required for the SONIC reference contract") from error
    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_device": "cpu",
        "torch_default_dtype": str(torch.get_default_dtype()),
    }


def _load_json_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{name} must contain a JSON object")
    return payload


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    matches = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            matches.append(node.value)
    _require(len(matches) == 1, f"G1 config must define {name} exactly once")
    return matches[0]


def _safe_ast_value(node: ast.AST, constants: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        _require(node.id in constants, f"unsupported G1 config name reference: {node.id}")
        return constants[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_safe_ast_value(element, constants) for element in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        return {
            _safe_ast_value(key, constants): _safe_ast_value(value, constants)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_ast_value(node.operand, constants)
        _require(isinstance(value, (int, float)), "unary G1 config value must be numeric")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_ast_value(node.left, constants) + _safe_ast_value(node.right, constants)
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                pieces.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                pieces.append(str(_safe_ast_value(value.value, constants)))
            else:  # pragma: no cover - AST grammar is defensive
                raise ValueError("unsupported formatted G1 asset path")
        return "".join(pieces)
    raise ValueError(f"unsupported G1 config expression: {ast.dump(node)}")


def _call_keywords(node: ast.AST, context: str) -> dict[str, ast.AST]:
    _require(isinstance(node, ast.Call), f"{context} must be a constructor call")
    keywords: dict[str, ast.AST] = {}
    for keyword in node.keywords:
        _require(keyword.arg is not None, f"{context} may not use **kwargs")
        _require(keyword.arg not in keywords, f"{context} repeats keyword {keyword.arg}")
        keywords[str(keyword.arg)] = keyword.value
    return keywords


def _resolve_repo_asset(
    config_path: Path,
    configured_path: str,
    *,
    description: str,
) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured.resolve()
    matches = [
        (ancestor / configured).resolve()
        for ancestor in config_path.parents
        if (ancestor / configured).is_file()
    ]
    _require(
        len(matches) == 1,
        f"configured {description} path {configured_path!r} did not resolve uniquely from "
        f"{config_path}",
    )
    return matches[0]


def _vector(
    text: str | None, name: str, *, default: tuple[float, ...] | None = None
) -> list[float]:
    if text is None:
        _require(default is not None, f"{name} is required")
        return list(default)
    try:
        values = [float(item) for item in text.split()]
    except ValueError as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    _require(
        len(values) == 3 and all(math.isfinite(value) for value in values),
        f"{name} must contain exactly three finite values",
    )
    return values


def _parse_urdf(urdf_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot parse G1 URDF {urdf_path}: {error}") from error
    _require(root.tag == "robot", "G1 URDF root element must be <robot>")
    links = {element.get("name") for element in root.findall("link")}
    _require(None not in links and len(links) == len(root.findall("link")), "URDF links invalid")
    child_to_joint: dict[str, dict[str, Any]] = {}
    joint_names: set[str] = set()
    for index, element in enumerate(root.findall("joint")):
        name = element.get("name")
        joint_type = element.get("type")
        parent = element.find("parent")
        child = element.find("child")
        _require(
            isinstance(name, str)
            and name
            and name not in joint_names
            and isinstance(joint_type, str)
            and parent is not None
            and child is not None,
            f"URDF joint {index} has invalid identity fields",
        )
        parent_link = parent.get("link")
        child_link = child.get("link")
        _require(
            parent_link in links and child_link in links and child_link not in child_to_joint,
            f"URDF joint {name} has invalid parent/child links",
        )
        origin = element.find("origin")
        axis = element.find("axis")
        record: dict[str, Any] = {
            "name": name,
            "type": joint_type,
            "parent_link": parent_link,
            "child_link": child_link,
            "origin_xyz": _vector(
                None if origin is None else origin.get("xyz"),
                f"URDF joint {name} origin xyz",
                default=(0.0, 0.0, 0.0),
            ),
            "origin_rpy": _vector(
                None if origin is None else origin.get("rpy"),
                f"URDF joint {name} origin rpy",
                default=(0.0, 0.0, 0.0),
            ),
            "axis": _vector(
                None if axis is None else axis.get("xyz"),
                f"URDF joint {name} axis",
                default=(1.0, 0.0, 0.0),
            ),
        }
        if joint_type in {"revolute", "continuous"}:
            limit = element.find("limit")
            _require(limit is not None, f"URDF actuated joint {name} lacks a limit")
            try:
                record["lower"] = float(limit.attrib["lower"])
                record["upper"] = float(limit.attrib["upper"])
                record["velocity"] = float(limit.attrib["velocity"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"URDF actuated joint {name} has invalid limits") from error
            _require(
                all(math.isfinite(record[field]) for field in ("lower", "upper", "velocity"))
                and record["upper"] > record["lower"]
                and record["velocity"] > 0.0,
                f"URDF actuated joint {name} limits are invalid",
            )
            axis_array = np.asarray(record["axis"], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis_array))
            _require(axis_norm > 0.0, f"URDF actuated joint {name} has zero axis")
            record["axis"] = [float(value) for value in axis_array / axis_norm]
        child_to_joint[str(child_link)] = record
        joint_names.add(name)
    return {"robot_name": root.get("name"), "links": sorted(links)}, child_to_joint


def _parse_mjcf_reference_tree(mjcf_path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(mjcf_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot parse motion-library MJCF {mjcf_path}: {error}") from error
    _require(root.tag == "mujoco", "motion-library MJCF root element must be <mujoco>")
    compiler = root.find("compiler")
    _require(
        compiler is not None and compiler.get("angle", "degree") == "radian",
        "motion-library MJCF must use radian joint angles",
    )
    worldbody = root.find("worldbody")
    _require(worldbody is not None, "motion-library MJCF lacks worldbody")
    root_bodies = worldbody.findall("body")
    _require(len(root_bodies) == 1, "motion-library MJCF must have one root body")
    records: list[dict[str, Any]] = []

    def visit(element: ET.Element, parent_name: str | None) -> None:
        name = element.get("name")
        _require(
            isinstance(name, str)
            and name
            and name not in {record["body_name"] for record in records},
            "motion-library MJCF body names must be unique and non-empty",
        )
        position = _vector(
            element.get("pos"),
            f"MJCF body {name} position",
            default=(0.0, 0.0, 0.0),
        )
        quaternion = _vector4(
            element.get("quat"),
            f"MJCF body {name} quaternion",
            default=(1.0, 0.0, 0.0, 0.0),
        )
        quaternion_array = np.asarray(quaternion, dtype=np.float64)
        quaternion_norm = float(np.linalg.norm(quaternion_array))
        _require(quaternion_norm > 1e-12, f"MJCF body {name} has zero quaternion")
        quaternion = [float(value) for value in quaternion_array / quaternion_norm]
        joints = element.findall("joint")
        if parent_name is None:
            _require(
                len(joints) == 1 and joints[0].get("type") == "free",
                "motion-library MJCF root body must have one free joint",
            )
            joint_name = str(joints[0].get("name"))
            joint_axis = None
        else:
            _require(
                len(joints) == 1 and joints[0].get("type", "hinge") == "hinge",
                f"motion-library MJCF body {name} must have one hinge joint",
            )
            joint_name = joints[0].get("name")
            _require(
                isinstance(joint_name, str) and joint_name,
                f"motion-library MJCF body {name} joint name is invalid",
            )
            joint_axis = _vector(
                joints[0].get("axis"),
                f"MJCF joint {joint_name} axis",
                default=(0.0, 0.0, 1.0),
            )
            axis_array = np.asarray(joint_axis, dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis_array))
            _require(axis_norm > 1e-12, f"MJCF joint {joint_name} has zero axis")
            joint_axis = [float(value) for value in axis_array / axis_norm]
        records.append(
            {
                "body_name": name,
                "parent_body_name": parent_name,
                "local_position": position,
                "local_quaternion_wxyz": quaternion,
                "joint_name": joint_name,
                "joint_axis": joint_axis,
            }
        )
        for child in element.findall("body"):
            visit(child, name)

    visit(root_bodies[0], None)
    return records


def _vector4(
    text: str | None,
    name: str,
    *,
    default: tuple[float, float, float, float],
) -> list[float]:
    if text is None:
        return list(default)
    try:
        values = [float(item) for item in text.split()]
    except ValueError as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    _require(
        len(values) == 4 and all(math.isfinite(value) for value in values),
        f"{name} must contain exactly four finite values",
    )
    return values


def _derive_motion_reference_contract(
    motion_command_config_path: Path,
    motion_mjcf_path: Path,
    *,
    robot_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with motion_command_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"cannot parse motion command config {motion_command_config_path}: {error}"
        ) from error
    _require(isinstance(config, Mapping), "motion command config must be a mapping")
    motion = config.get("motion")
    _require(isinstance(motion, Mapping), "motion command config lacks motion")
    motion_lib_cfg = motion.get("motion_lib_cfg")
    _require(isinstance(motion_lib_cfg, Mapping), "motion command config lacks motion_lib_cfg")
    asset = motion_lib_cfg.get("asset")
    _require(isinstance(asset, Mapping), "motion_lib_cfg lacks asset configuration")
    asset_root = asset.get("assetRoot")
    asset_file = asset.get("assetFileName")
    _require(
        isinstance(asset_root, str) and asset_root and isinstance(asset_file, str) and asset_file,
        "motion-library MJCF asset fields are invalid",
    )
    configured_path = str(Path(asset_root) / asset_file)
    configured_mjcf = _resolve_repo_asset(
        motion_command_config_path,
        configured_path,
        description="motion-library MJCF",
    )
    _require(
        configured_mjcf == motion_mjcf_path,
        f"supplied MJCF is not the motion-library asset: {motion_mjcf_path} != {configured_mjcf}",
    )
    _require(
        motion_lib_cfg.get("target_fps") == 50,
        "motion-library reference contract requires target_fps=50",
    )
    _require(
        motion_lib_cfg.get("extend_config") == [],
        "motion-library reference contract requires no extended bodies",
    )
    _require(
        "fix_height" not in motion_lib_cfg,
        "motion-library reference contract requires the default no-fix height mode",
    )
    body_records = _parse_mjcf_reference_tree(motion_mjcf_path)
    _require(
        len(body_records) == 30 and body_records[0]["body_name"] == "pelvis",
        "motion-library MJCF must contain the frozen 30-body G1 tree",
    )
    mjcf_joint_names = [str(record["joint_name"]) for record in body_records[1:]]
    _require(
        mjcf_joint_names == robot_contract["mujoco_source_joint_names"],
        "motion-library MJCF joint order disagrees with the frozen G1 source order",
    )
    body_names = [str(record["body_name"]) for record in body_records]
    _require(
        all(name in body_names for name in _EXPECTED_FOOT_LINKS),
        "motion-library MJCF lacks the frozen foot bodies",
    )
    contract = {
        "motion_command_config": {
            "path": str(motion_command_config_path),
            "sha256": _sha256_file(motion_command_config_path, "motion command config"),
        },
        "motion_mjcf": {
            "path": str(motion_mjcf_path),
            "sha256": _sha256_file(motion_mjcf_path, "motion-library MJCF"),
            "configured_asset_root": asset_root,
            "configured_asset_file": asset_file,
        },
        "target_fps": 50,
        "extend_config": [],
        "fix_height_mode": "MotionLibBase_default_FixHeightMode.no_fix",
        "body_names_mjcf_order": body_names,
        "joint_names_mjcf_order": mjcf_joint_names,
        "foot_body_names": list(_EXPECTED_FOOT_LINKS),
        "reference_interpolation_source": (
            "gear_sonic.utils.motion_lib.torch_humanoid_batch:Humanoid_Batch.interploate_pose"
        ),
        "reference_fk_source": (
            "gear_sonic.utils.motion_lib.torch_humanoid_batch:Humanoid_Batch.fk_batch"
        ),
        "contact_source": ("gear_sonic.utils.motion_lib.motion_lib_base:MotionLibBase.foot_detect"),
    }
    contract["motion_reference_contract_sha256"] = canonical_sha256(contract)
    return contract, body_records


def validate_motion_reference_contract(
    contract: Mapping[str, Any],
    *,
    robot_contract: Mapping[str, Any],
    verify_source_files: bool = False,
) -> None:
    """Validate the exact SONIC reference FK/contact source contract."""

    command_binding = contract.get("motion_command_config")
    mjcf_binding = contract.get("motion_mjcf")
    _require(isinstance(command_binding, Mapping), "motion command binding is missing")
    _require(isinstance(mjcf_binding, Mapping), "motion MJCF binding is missing")
    for binding, name in (
        (command_binding, "motion command config"),
        (mjcf_binding, "motion-library MJCF"),
    ):
        _sha256(binding.get("sha256"), f"{name} SHA-256")
        _require(
            isinstance(binding.get("path"), str) and binding["path"],
            f"{name} path is invalid",
        )
    _require(contract.get("target_fps") == 50, "motion reference target FPS must be 50")
    _require(contract.get("extend_config") == [], "motion reference extended bodies are invalid")
    _require(
        contract.get("fix_height_mode") == "MotionLibBase_default_FixHeightMode.no_fix",
        "motion reference height-fix mode is invalid",
    )
    body_names = contract.get("body_names_mjcf_order")
    joint_names = contract.get("joint_names_mjcf_order")
    _require(
        isinstance(body_names, list)
        and len(body_names) == 30
        and body_names[0] == "pelvis"
        and len(set(body_names)) == 30,
        "motion reference body order is invalid",
    )
    _require(
        joint_names == robot_contract.get("mujoco_source_joint_names"),
        "motion reference joint order disagrees with the robot contract",
    )
    _require(
        contract.get("foot_body_names") == list(_EXPECTED_FOOT_LINKS),
        "motion reference foot body order is invalid",
    )
    expected_digest = _sha256(
        contract.get("motion_reference_contract_sha256"),
        "motion_reference_contract_sha256",
    )
    _require(
        expected_digest
        == canonical_sha256(contract, digest_field="motion_reference_contract_sha256"),
        "motion_reference_contract_sha256 mismatch",
    )
    if verify_source_files:
        command_path = Path(str(command_binding["path"])).resolve()
        mjcf_path = Path(str(mjcf_binding["path"])).resolve()
        _require(command_path.is_file(), f"motion command config is missing: {command_path}")
        _require(mjcf_path.is_file(), f"motion-library MJCF is missing: {mjcf_path}")
        _require(
            _sha256_file(command_path, "motion command config") == command_binding["sha256"],
            "motion command config SHA-256 mismatch",
        )
        _require(
            _sha256_file(mjcf_path, "motion-library MJCF") == mjcf_binding["sha256"],
            "motion-library MJCF SHA-256 mismatch",
        )
        reconstructed, _ = _derive_motion_reference_contract(
            command_path,
            mjcf_path,
            robot_contract=robot_contract,
        )
        _require(reconstructed == contract, "motion reference contract disagrees with sources")


def _resolve_actuator_velocity_limits(
    actuator_node: ast.AST,
    *,
    joint_names: Sequence[str],
    constants: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, float]]]:
    _require(isinstance(actuator_node, ast.Dict), "G1 actuators must be a literal mapping")
    groups = []
    resolved: dict[str, tuple[str, float]] = {}
    for key_node, value_node in zip(actuator_node.keys, actuator_node.values, strict=True):
        group_name = _safe_ast_value(key_node, constants)
        _require(isinstance(group_name, str) and group_name, "actuator group name is invalid")
        keywords = _call_keywords(value_node, f"actuator group {group_name}")
        _require(
            "joint_names_expr" in keywords and "velocity_limit_sim" in keywords,
            f"actuator group {group_name} must declare joints and velocity_limit_sim",
        )
        expressions = _safe_ast_value(keywords["joint_names_expr"], constants)
        velocity_spec = _safe_ast_value(keywords["velocity_limit_sim"], constants)
        _require(
            isinstance(expressions, list)
            and expressions
            and all(isinstance(item, str) and item for item in expressions),
            f"actuator group {group_name} joint expressions are invalid",
        )
        group_joints = [
            name for name in joint_names if any(re.fullmatch(expr, name) for expr in expressions)
        ]
        _require(group_joints, f"actuator group {group_name} resolves no G1 joints")
        velocities: dict[str, float] = {}
        for joint_name in group_joints:
            if isinstance(velocity_spec, Mapping):
                matches = [
                    value
                    for expression, value in velocity_spec.items()
                    if isinstance(expression, str) and re.fullmatch(expression, joint_name)
                ]
                _require(
                    len(matches) == 1,
                    f"actuator group {group_name} velocity mapping is ambiguous for {joint_name}",
                )
                value = matches[0]
            else:
                value = velocity_spec
            _require(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0,
                f"actuator velocity for {joint_name} must be positive",
            )
            _require(
                joint_name not in resolved, f"joint {joint_name} belongs to multiple actuators"
            )
            velocities[joint_name] = float(value)
            resolved[joint_name] = (group_name, float(value))
        groups.append(
            {
                "actuator_group": group_name,
                "joint_names_expr": expressions,
                "velocity_limit_sim_spec": velocity_spec,
                "resolved_joint_names": group_joints,
                "resolved_velocity_limits": velocities,
            }
        )
    _require(set(resolved) == set(joint_names), "actuator groups do not cover every G1 joint")
    return groups, resolved


def _live_validation(
    expected: Mapping[str, Any],
    live_robot_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_properties = {
        "joint_names": "robot.joint_names",
        "joint_pos_limits": "robot.data.joint_pos_limits[0]",
        "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
        "joint_vel_limits": "robot.data.joint_vel_limits[0]",
        "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
    }
    if live_robot_data is None:
        return {
            "verified": False,
            "status": "unverified_no_isaac_launch",
            "source_properties": source_properties,
            "required_follow_up": (
                "launch the pinned G1 articulation and compare all ordered robot.data tensors"
            ),
        }
    _require(isinstance(live_robot_data, Mapping), "live_robot_data must be a mapping")
    _require(
        set(live_robot_data) >= {"robot_contract_readback", "robot_contract_readback_sha256"},
        "live_robot_data must be a recorder export with readback and digest",
    )
    recorder_evidence_binding = live_robot_data.get("recorder_evidence_binding")
    if recorder_evidence_binding is not None:
        recorder_evidence_binding = _validate_recorder_evidence_binding(
            recorder_evidence_binding,
            verify_source_file=True,
        )
    readback = live_robot_data.get("robot_contract_readback")
    readback_digest = _sha256(
        live_robot_data.get("robot_contract_readback_sha256"),
        "robot_contract_readback_sha256",
    )
    _require(isinstance(readback, Mapping), "robot_contract_readback must be a mapping")
    _require(
        canonical_sha256(readback) == readback_digest,
        "robot_contract_readback_sha256 mismatch",
    )
    _require(
        readback.get("kind") == ROBOT_CONTRACT_READBACK_KIND
        and readback.get("schema_version") == ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
        "unsupported robot contract readback schema",
    )
    _require(
        readback.get("capture_lifecycle") == RUNTIME_REALIZATION_CAPTURE_LIFECYCLE
        and readback.get("environment_invariance_verified") is True,
        "robot contract readback lacks live lifecycle/invariance evidence",
    )
    _require(
        readback.get("source_properties") == source_properties,
        "robot contract readback source properties mismatch",
    )
    limits = readback.get("limits")
    _require(
        isinstance(limits, Mapping) and set(limits) == _LIVE_ROBOT_DATA_FIELDS - {"joint_names"},
        "robot contract readback limits are invalid",
    )
    normalized_live_robot_data: dict[str, Any] = {
        "joint_names": readback.get("ordered_joint_names")
    }
    for field in sorted(_LIVE_ROBOT_DATA_FIELDS - {"joint_names"}):
        tensor_record = limits[field]
        _require(
            isinstance(tensor_record, Mapping)
            and set(tensor_record) == {"dtype", "shape", "values"}
            and isinstance(tensor_record.get("dtype"), str)
            and tensor_record["dtype"],
            f"robot contract readback {field} tensor record is invalid",
        )
        values = np.asarray(tensor_record["values"])
        _require(
            list(values.shape) == tensor_record["shape"],
            f"robot contract readback {field} shape metadata mismatch",
        )
        normalized_live_robot_data[field] = tensor_record["values"]
    live_robot_data = normalized_live_robot_data
    _require(
        set(live_robot_data) == _LIVE_ROBOT_DATA_FIELDS,
        "live_robot_data fields are invalid",
    )
    joint_names = list(live_robot_data["joint_names"])
    _require(joint_names == expected["joint_names"], "live robot.data.joint_names mismatch")
    for field in sorted(_LIVE_ROBOT_DATA_FIELDS - {"joint_names"}):
        actual = np.asarray(live_robot_data[field], dtype=np.float64)
        target = np.asarray(expected[field], dtype=np.float64)
        _require(actual.shape == target.shape, f"live robot.data.{field} shape mismatch")
        _require(
            np.all(np.isfinite(actual)) and np.allclose(actual, target, rtol=0.0, atol=1e-6),
            f"live robot.data.{field} does not match the frozen G1 contract",
        )
    result = {
        "verified": True,
        "status": "verified_against_supplied_live_robot_data",
        "source_properties": source_properties,
        "required_follow_up": None,
        "recorder_readback_sha256": readback_digest,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "live_values_sha256": canonical_sha256(
            {
                "joint_names": joint_names,
                **{
                    field: np.asarray(live_robot_data[field], dtype=np.float64).tolist()
                    for field in sorted(_LIVE_ROBOT_DATA_FIELDS - {"joint_names"})
                },
            }
        ),
    }
    if recorder_evidence_binding is not None:
        result["recorder_evidence_binding"] = recorder_evidence_binding
    return result


def derive_g1_reference_contract(
    g1_config_path: str | Path,
    urdf_path: str | Path,
    *,
    live_robot_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive ordered hard/soft/velocity limits without importing Isaac Lab."""

    config_path = Path(g1_config_path).expanduser().resolve()
    resolved_urdf_path = Path(urdf_path).expanduser().resolve()
    _require(config_path.is_file(), f"G1 config does not exist: {config_path}")
    _require(resolved_urdf_path.is_file(), f"G1 URDF does not exist: {resolved_urdf_path}")
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(config_path))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot parse G1 config {config_path}: {error}") from error

    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = _safe_ast_value(node.value, constants)
        except ValueError:
            continue
    body_names = _safe_ast_value(_assignment(tree, _G1_BODY_NAMES_ASSIGNMENT), constants)
    il_to_mj = _safe_ast_value(_assignment(tree, _IL_TO_MJ_ASSIGNMENT), constants)
    mj_to_il = _safe_ast_value(_assignment(tree, _MJ_TO_IL_ASSIGNMENT), constants)
    _require(
        isinstance(body_names, list)
        and len(body_names) == 30
        and body_names[0] == "pelvis"
        and len(set(body_names)) == len(body_names),
        "G1_ISAACLAB_JOINTS must contain pelvis plus 29 unique body names",
    )
    for values, name in ((il_to_mj, _IL_TO_MJ_ASSIGNMENT), (mj_to_il, _MJ_TO_IL_ASSIGNMENT)):
        _require(
            isinstance(values, list)
            and values == [int(value) for value in values]
            and sorted(values) == list(range(29)),
            f"{name} must be a permutation of range(29)",
        )
    _require(
        all(il_to_mj[mj_to_il[index]] == index for index in range(29))
        and all(mj_to_il[il_to_mj[index]] == index for index in range(29)),
        "G1 DOF mappings are not exact inverses",
    )

    cfg_keywords = _call_keywords(_assignment(tree, _G1_CFG_ASSIGNMENT), _G1_CFG_ASSIGNMENT)
    _require(
        {"spawn", "soft_joint_pos_limit_factor", "actuators"} <= set(cfg_keywords),
        "G1 articulation config lacks required robot-contract fields",
    )
    spawn_keywords = _call_keywords(cfg_keywords["spawn"], "G1 spawn config")
    configured_asset_path = _safe_ast_value(spawn_keywords["asset_path"], constants)
    _require(isinstance(configured_asset_path, str), "G1 configured asset path must be a string")
    configured_urdf = _resolve_repo_asset(
        config_path,
        configured_asset_path,
        description="G1 URDF",
    )
    _require(
        configured_urdf == resolved_urdf_path,
        f"supplied URDF is not the G1 config asset: {resolved_urdf_path} != {configured_urdf}",
    )
    soft_factor = _safe_ast_value(cfg_keywords["soft_joint_pos_limit_factor"], constants)
    _require(
        isinstance(soft_factor, Real)
        and not isinstance(soft_factor, bool)
        and 0.0 < float(soft_factor) <= 1.0,
        "soft_joint_pos_limit_factor must be in (0, 1]",
    )

    urdf_metadata, child_to_joint = _parse_urdf(resolved_urdf_path)
    il_joint_records = []
    for body_name in body_names[1:]:
        _require(body_name in child_to_joint, f"G1 body {body_name} is not a URDF joint child")
        record = child_to_joint[body_name]
        _require(
            record["type"] in {"revolute", "continuous"},
            f"G1 body {body_name} does not resolve to an actuated URDF joint",
        )
        il_joint_records.append(record)
    il_joint_names = [record["name"] for record in il_joint_records]
    actuator_groups, actuator_limits = _resolve_actuator_velocity_limits(
        cfg_keywords["actuators"],
        joint_names=il_joint_names,
        constants=constants,
    )
    lower = np.asarray([record["lower"] for record in il_joint_records], dtype=np.float64)
    upper = np.asarray([record["upper"] for record in il_joint_records], dtype=np.float64)
    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower) * float(soft_factor)
    soft_lower = midpoint - half_range
    soft_upper = midpoint + half_range
    velocity = np.asarray(
        [actuator_limits[name][1] for name in il_joint_names],
        dtype=np.float64,
    )
    expected_live = {
        "joint_names": il_joint_names,
        "joint_pos_limits": np.column_stack((lower, upper)).tolist(),
        "soft_joint_pos_limits": np.column_stack((soft_lower, soft_upper)).tolist(),
        "joint_vel_limits": velocity.tolist(),
        "soft_joint_vel_limits": velocity.tolist(),
    }
    source_order_records = [il_joint_records[index] for index in il_to_mj]
    source_joint_names = [record["name"] for record in source_order_records]

    result: dict[str, Any] = {
        "kind": G1_REFERENCE_CONTRACT_KIND,
        "schema_version": G1_REFERENCE_CONTRACT_SCHEMA_VERSION,
        "g1_config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path, "G1 config"),
            "articulation_assignment": _G1_CFG_ASSIGNMENT,
            "configured_urdf_asset_path": configured_asset_path,
        },
        "urdf": {
            "path": str(resolved_urdf_path),
            "sha256": _sha256_file(resolved_urdf_path, "G1 URDF"),
            **urdf_metadata,
        },
        "dof_count": 29,
        "isaaclab_body_names_with_root": body_names,
        "isaaclab_joint_names": il_joint_names,
        "mujoco_source_joint_names": source_joint_names,
        "isaaclab_to_mujoco_dof": il_to_mj,
        "mujoco_to_isaaclab_dof": mj_to_il,
        "mapping_semantics": {
            "motion_payload_dof_order": "mujoco_source_joint_names",
            "runtime_reorder": "motion_dof[:, mujoco_to_isaaclab_dof]",
        },
        "soft_joint_pos_limit_factor": float(soft_factor),
        "actuator_velocity_groups": actuator_groups,
        "velocity_limit_source_contract": {
            "proxy_denominator": "g1_config.actuators.velocity_limit_sim",
            "urdf_limits_retained_as_provenance_not_substituted": True,
            "config_urdf_mismatch_is_allowed_and_reported": True,
            "config_urdf_mismatch_joint_names": [
                record["name"]
                for record in il_joint_records
                if not math.isclose(
                    actuator_limits[record["name"]][1],
                    record["velocity"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ],
        },
        "joint_contract_isaaclab_order": [
            {
                "joint_name": record["name"],
                "child_link": record["child_link"],
                "axis": record["axis"],
                "hard_lower_radians": float(lower[index]),
                "hard_upper_radians": float(upper[index]),
                "soft_lower_radians": float(soft_lower[index]),
                "soft_upper_radians": float(soft_upper[index]),
                "urdf_velocity_limit_radians_per_second": float(record["velocity"]),
                "actuator_velocity_limit_radians_per_second": float(velocity[index]),
                "velocity_limit_sources_equal": math.isclose(
                    float(record["velocity"]),
                    float(velocity[index]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "actuator_group": actuator_limits[record["name"]][0],
            }
            for index, record in enumerate(il_joint_records)
        ],
        "expected_live_robot_data": expected_live,
        "live_robot_data_validation": _live_validation(expected_live, live_robot_data),
    }
    result[G1_REFERENCE_CONTRACT_DIGEST_FIELD] = canonical_sha256(result)
    validate_g1_reference_contract(result)
    return result


def validate_g1_reference_contract(
    contract: Mapping[str, Any],
    *,
    verify_source_files: bool = False,
) -> None:
    """Validate a frozen CPU-derived G1 limit and ordering contract."""

    _require(contract.get("kind") == G1_REFERENCE_CONTRACT_KIND, "invalid G1 contract kind")
    _require(
        contract.get("schema_version") == G1_REFERENCE_CONTRACT_SCHEMA_VERSION,
        "unsupported G1 contract schema version",
    )
    _require(contract.get("dof_count") == 29, "G1 contract dof_count must be 29")
    body_names = contract.get("isaaclab_body_names_with_root")
    il_names = contract.get("isaaclab_joint_names")
    source_names = contract.get("mujoco_source_joint_names")
    _require(
        isinstance(body_names, list)
        and len(body_names) == 30
        and body_names[0] == "pelvis"
        and len(set(body_names)) == 30,
        "invalid G1 IsaacLab body order",
    )
    _require(
        isinstance(il_names, list)
        and isinstance(source_names, list)
        and len(il_names) == len(source_names) == 29
        and len(set(il_names)) == len(set(source_names)) == 29
        and set(il_names) == set(source_names),
        "invalid G1 joint orders",
    )
    il_to_mj = contract.get("isaaclab_to_mujoco_dof")
    mj_to_il = contract.get("mujoco_to_isaaclab_dof")
    _require(
        isinstance(il_to_mj, list)
        and isinstance(mj_to_il, list)
        and sorted(il_to_mj) == sorted(mj_to_il) == list(range(29)),
        "invalid G1 DOF permutations",
    )
    _require(
        source_names == [il_names[index] for index in il_to_mj],
        "mujoco source names do not follow the frozen mapping",
    )
    records = contract.get("joint_contract_isaaclab_order")
    _require(isinstance(records, list) and len(records) == 29, "G1 joint contract is incomplete")
    expected_position_limits: list[list[float]] = []
    expected_soft_position_limits: list[list[float]] = []
    expected_velocity_limits: list[float] = []
    record_groups: dict[str, set[str]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"G1 joint contract {index} must be a mapping")
        _require(record.get("joint_name") == il_names[index], "G1 joint contract order mismatch")
        lower = record.get("hard_lower_radians")
        upper = record.get("hard_upper_radians")
        soft_lower = record.get("soft_lower_radians")
        soft_upper = record.get("soft_upper_radians")
        velocity = record.get("actuator_velocity_limit_radians_per_second")
        urdf_velocity = record.get("urdf_velocity_limit_radians_per_second")
        _require(
            all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (
                    lower,
                    upper,
                    soft_lower,
                    soft_upper,
                    velocity,
                    urdf_velocity,
                )
            )
            and lower < soft_lower < soft_upper < upper
            and velocity > 0.0
            and urdf_velocity > 0.0,
            f"G1 joint contract {index} limits are invalid",
        )
        _require(
            record.get("velocity_limit_sources_equal")
            is math.isclose(
                float(urdf_velocity),
                float(velocity),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"G1 joint contract {index} velocity comparison is inconsistent",
        )
        group_name = record.get("actuator_group")
        _require(
            isinstance(group_name, str) and group_name,
            f"G1 joint contract {index} actuator group is invalid",
        )
        record_groups.setdefault(group_name, set()).add(str(record["joint_name"]))
        expected_position_limits.append([float(lower), float(upper)])
        expected_soft_position_limits.append([float(soft_lower), float(soft_upper)])
        expected_velocity_limits.append(float(velocity))
    groups = contract.get("actuator_velocity_groups")
    _require(isinstance(groups, list) and groups, "actuator velocity groups are missing")
    declared_groups: dict[str, set[str]] = {}
    for index, group in enumerate(groups):
        _require(isinstance(group, Mapping), f"actuator velocity group {index} is invalid")
        group_name = group.get("actuator_group")
        resolved_names = group.get("resolved_joint_names")
        resolved_limits = group.get("resolved_velocity_limits")
        _require(
            isinstance(group_name, str)
            and group_name
            and group_name not in declared_groups
            and isinstance(resolved_names, list)
            and resolved_names
            and len(resolved_names) == len(set(resolved_names))
            and isinstance(resolved_limits, Mapping)
            and set(resolved_limits) == set(resolved_names),
            f"actuator velocity group {index} resolution is invalid",
        )
        declared_groups[group_name] = set(resolved_names)
        for joint_name in resolved_names:
            _require(
                joint_name in il_names
                and resolved_limits[joint_name]
                == records[il_names.index(joint_name)][
                    "actuator_velocity_limit_radians_per_second"
                ],
                f"actuator velocity group {group_name} disagrees for {joint_name}",
            )
    _require(
        declared_groups == record_groups
        and set().union(*declared_groups.values()) == set(il_names),
        "actuator velocity groups disagree with the ordered joint contract",
    )
    velocity_source_contract = contract.get("velocity_limit_source_contract")
    _require(
        isinstance(velocity_source_contract, Mapping)
        and velocity_source_contract.get("proxy_denominator")
        == "g1_config.actuators.velocity_limit_sim"
        and velocity_source_contract.get("urdf_limits_retained_as_provenance_not_substituted")
        is True
        and velocity_source_contract.get("config_urdf_mismatch_is_allowed_and_reported") is True,
        "velocity limit source contract is invalid",
    )
    expected_mismatches = [
        str(record["joint_name"])
        for record in records
        if record["velocity_limit_sources_equal"] is False
    ]
    _require(
        velocity_source_contract.get("config_urdf_mismatch_joint_names") == expected_mismatches,
        "velocity limit source mismatch summary is inconsistent",
    )
    expected_live = contract.get("expected_live_robot_data")
    _require(isinstance(expected_live, Mapping), "expected_live_robot_data is missing")
    _require(
        set(expected_live) == _LIVE_ROBOT_DATA_FIELDS,
        "expected_live_robot_data fields are invalid",
    )
    _require(expected_live.get("joint_names") == il_names, "expected live joint order mismatch")
    reconstructed_live = {
        "joint_pos_limits": expected_position_limits,
        "soft_joint_pos_limits": expected_soft_position_limits,
        "joint_vel_limits": expected_velocity_limits,
        "soft_joint_vel_limits": expected_velocity_limits,
    }
    for field, reconstructed in reconstructed_live.items():
        observed = np.asarray(expected_live.get(field), dtype=np.float64)
        target = np.asarray(reconstructed, dtype=np.float64)
        _require(
            observed.shape == target.shape
            and np.all(np.isfinite(observed))
            and np.array_equal(observed, target),
            f"expected live {field} disagrees with the ordered joint contract",
        )
    live_validation = contract.get("live_robot_data_validation")
    _require(isinstance(live_validation, Mapping), "live_robot_data_validation is missing")
    _require(
        isinstance(live_validation.get("verified"), bool), "live verified flag must be boolean"
    )
    if not live_validation["verified"]:
        _require(
            live_validation.get("status") == "unverified_no_isaac_launch",
            "unverified live validation must state no Isaac launch",
        )
    else:
        _require(
            live_validation.get("status") == "verified_against_supplied_live_robot_data"
            and live_validation.get("capture_lifecycle") == RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
            "verified live validation lacks recorder lifecycle evidence",
        )
        _sha256(
            live_validation.get("recorder_readback_sha256"),
            "live recorder_readback_sha256",
        )
        _sha256(live_validation.get("live_values_sha256"), "live_values_sha256")
        recorder_evidence_binding = live_validation.get("recorder_evidence_binding")
        if recorder_evidence_binding is not None:
            _validate_recorder_evidence_binding(
                recorder_evidence_binding,
                verify_source_file=verify_source_files,
            )
    for source_name, description in (("g1_config", "G1 config"), ("urdf", "G1 URDF")):
        source = contract.get(source_name)
        _require(isinstance(source, Mapping), f"{source_name} binding is missing")
        digest = _sha256(source.get("sha256"), f"{source_name}.sha256")
        path = Path(str(source.get("path")))
        if verify_source_files:
            _require(path.is_file(), f"{description} source is missing: {path}")
            _require(_sha256_file(path, description) == digest, f"{description} SHA-256 mismatch")
    expected_digest = _sha256(
        contract.get(G1_REFERENCE_CONTRACT_DIGEST_FIELD),
        G1_REFERENCE_CONTRACT_DIGEST_FIELD,
    )
    _require(
        expected_digest
        == canonical_sha256(contract, digest_field=G1_REFERENCE_CONTRACT_DIGEST_FIELD),
        f"{G1_REFERENCE_CONTRACT_DIGEST_FIELD} mismatch",
    )


def _numeric_array(value: Any, name: str, shape_tail: tuple[int, ...]) -> FloatArray:
    raw = np.asarray(value)
    _require(
        raw.ndim == len(shape_tail) + 1 and raw.shape[0] >= 2 and raw.shape[1:] == shape_tail,
        f"{name} must have shape [frames >= 2, {shape_tail}]",
    )
    _require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must be real numeric",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(np.all(np.isfinite(result)), f"{name} must contain only finite values")
    return result


def _positive_fraction(value: Any, name: str) -> Fraction:
    _require(not isinstance(value, bool), f"{name} must be numeric, not boolean")
    if isinstance(value, Integral):
        result = Fraction(int(value), 1)
    elif isinstance(value, Real):
        numeric = float(value)
        _require(math.isfinite(numeric), f"{name} must be finite")
        result = Fraction(str(numeric))
    elif isinstance(value, Mapping):
        _require(set(value) == {"numerator", "denominator"}, f"{name} fields are invalid")
        numerator = value["numerator"]
        denominator = value["denominator"]
        _require(
            isinstance(numerator, int)
            and not isinstance(numerator, bool)
            and isinstance(denominator, int)
            and not isinstance(denominator, bool)
            and denominator > 0,
            f"{name} must be a valid rational mapping",
        )
        result = Fraction(numerator, denominator)
        _require(
            (result.numerator, result.denominator) == (numerator, denominator),
            f"{name} rational must be reduced",
        )
    else:
        raise ValueError(f"{name} must be a positive rational")
    _require(result > 0, f"{name} must be positive")
    return result


def _load_motion_payload(path: Path, motion_key: str) -> Mapping[str, Any]:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - core runtime dependency
        raise ValueError("joblib is required for feasibility-manifest construction") from error
    try:
        container = joblib.load(path)
    except Exception as error:
        raise ValueError(f"cannot load selected motion {path}: {error}") from error
    _require(isinstance(container, Mapping), f"{path} must contain a keyed mapping")
    _require(
        list(container) == [motion_key],
        f"{path} must contain exactly the selected motion key {motion_key!r}",
    )
    payload = container[motion_key]
    _require(isinstance(payload, Mapping), f"{path}[{motion_key!r}] must be a mapping")
    return payload


def _quaternion_xyzw_from_rotation_vector(rotation_vectors: FloatArray) -> FloatArray:
    angles = np.linalg.norm(rotation_vectors, axis=-1)
    half = 0.5 * angles
    scale = np.empty_like(angles)
    nonzero = angles > 1e-12
    scale[nonzero] = np.sin(half[nonzero]) / angles[nonzero]
    scale[~nonzero] = 0.5
    return np.concatenate(
        (rotation_vectors * scale[..., None], np.cos(half)[..., None]),
        axis=-1,
    )


def _runtime_reference(
    *,
    pose_rotation_vector: FloatArray,
    root_position: FloatArray,
    source_fps: Fraction,
    target_fps: int,
    target_num_frames: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Reproduce SONIC's float32 quaternion interpolation on CPU."""

    source_num_frames = pose_rotation_vector.shape[0]
    _require(
        root_position.shape == (source_num_frames, 3)
        and pose_rotation_vector.shape == (source_num_frames, 30, 3),
        "runtime reference source arrays have inconsistent frame axes",
    )
    _require(
        isinstance(target_fps, int)
        and not isinstance(target_fps, bool)
        and target_fps > 0
        and isinstance(target_num_frames, int)
        and not isinstance(target_num_frames, bool)
        and target_num_frames >= 3,
        "runtime target FPS/frame count are invalid",
    )
    try:
        import torch

        from gear_sonic.isaac_utils.rotations import (
            axis_angle_to_quaternion,
            matrix_to_quaternion,
            quaternion_to_matrix,
            slerp,
        )
        from gear_sonic.trl.utils.torch_transform import quaternion_to_angle_axis
    except ImportError as error:  # pragma: no cover - SONIC runtime dependency
        raise ValueError(
            "torch rotation utilities are required for runtime-equivalent resampling"
        ) from error

    pose = torch.from_numpy(np.asarray(pose_rotation_vector, dtype=np.float32))
    translation = torch.from_numpy(np.asarray(root_position, dtype=np.float32))
    with torch.no_grad():
        pose_quaternion = axis_angle_to_quaternion(pose)
        if source_fps == target_fps:
            _require(
                target_num_frames == source_num_frames,
                "equal-FPS runtime reference must retain the raw source frame count",
            )
            runtime_pose_quaternion = pose_quaternion
            runtime_root_position = translation
        else:
            duration = (source_num_frames - 1) * 1.0 / float(source_fps)
            times = torch.arange(
                0,
                duration,
                1.0 / target_fps,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            _require(
                len(times) == target_num_frames,
                "torch float32 runtime timeline disagrees with the frozen inventory",
            )
            phase = times / duration
            coordinates = phase * (source_num_frames - 1)
            index_0 = torch.floor(coordinates).to(dtype=torch.long)
            index_1 = torch.minimum(
                index_0 + 1,
                torch.tensor(source_num_frames - 1, dtype=torch.long),
            )
            blend = coordinates - index_0
            runtime_pose_quaternion = slerp(
                pose_quaternion[index_0],
                pose_quaternion[index_1],
                blend[:, None, None],
            )
            runtime_root_position = (
                translation[index_0] * (1.0 - blend[:, None])
                + translation[index_1] * blend[:, None]
            )
        runtime_pose_rotation_vector = quaternion_to_angle_axis(runtime_pose_quaternion)
        runtime_joint_position = runtime_pose_rotation_vector[:, 1:, :].sum(dim=-1)
        runtime_local_rotation_matrices = quaternion_to_matrix(runtime_pose_quaternion)
        runtime_root_quaternion_wxyz = matrix_to_quaternion(runtime_local_rotation_matrices[:, 0])
        runtime_root_quaternion_xyzw = runtime_root_quaternion_wxyz[:, [1, 2, 3, 0]]
    return (
        np.asarray(runtime_joint_position.numpy(), dtype=np.float64),
        np.asarray(runtime_root_position.numpy(), dtype=np.float64),
        np.asarray(runtime_root_quaternion_xyzw.numpy(), dtype=np.float64),
        np.asarray(runtime_local_rotation_matrices.numpy(), dtype=np.float64),
    )


def _foot_positions_from_mjcf(
    *,
    body_records: Sequence[Mapping[str, Any]],
    local_rotation_matrices: FloatArray,
    root_position: FloatArray,
    root_quaternion_xyzw: FloatArray,
) -> FloatArray:
    frame_count = local_rotation_matrices.shape[0]
    _require(
        local_rotation_matrices.shape == (frame_count, 30, 3, 3)
        and root_position.shape == (frame_count, 3)
        and root_quaternion_xyzw.shape == (frame_count, 4),
        "MJCF forward-kinematics arrays have inconsistent shapes",
    )
    _require(
        isinstance(body_records, Sequence)
        and len(body_records) == 30
        and body_records[0].get("body_name") == "pelvis",
        "MJCF body records do not contain the frozen G1 tree",
    )
    try:
        import torch

        from gear_sonic.isaac_utils.rotations import quaternion_to_matrix
    except ImportError as error:  # pragma: no cover - SONIC runtime dependency
        raise ValueError("torch rotation utilities are required for motion-library FK") from error
    local_rotations = torch.from_numpy(np.asarray(local_rotation_matrices, dtype=np.float32))
    positions: dict[str, Any] = {
        "pelvis": torch.from_numpy(np.asarray(root_position, dtype=np.float32))
    }
    rotations: dict[str, Any] = {"pelvis": local_rotations[:, 0]}
    with torch.no_grad():
        for body_index, body in enumerate(body_records[1:], start=1):
            body_name = str(body["body_name"])
            parent_name = str(body["parent_body_name"])
            _require(
                parent_name in positions,
                f"MJCF parent {parent_name} precedes no transform",
            )
            parent_position = positions[parent_name]
            parent_rotation = rotations[parent_name]
            offset = torch.tensor(body["local_position"], dtype=torch.float32)
            fixed_quaternion = torch.tensor(
                body["local_quaternion_wxyz"],
                dtype=torch.float32,
            )[None, :]
            fixed_rotation = quaternion_to_matrix(fixed_quaternion)[0]
            positions[body_name] = parent_position + torch.matmul(
                parent_rotation,
                offset[None, :, None],
            ).squeeze(-1)
            rotations[body_name] = parent_rotation @ (
                fixed_rotation @ local_rotations[:, body_index]
            )
    return np.asarray(
        torch.stack([positions[name] for name in _EXPECTED_FOOT_LINKS], dim=1).numpy(),
        dtype=np.float64,
    )


def _sonic_contact_schedule(
    foot_positions: FloatArray,
    thresholds: ReferenceFeasibilityThresholds,
) -> NDArray[np.bool_]:
    displacement_squared = np.sum(np.square(np.diff(foot_positions, axis=0)), axis=2)
    displacement_squared = np.concatenate(
        (displacement_squared, displacement_squared[[-1]]),
        axis=0,
    )
    return np.asarray(
        (displacement_squared < thresholds.contact_displacement_squared_threshold_m2)
        & (foot_positions[:, :, 2] < thresholds.contact_height_threshold_meters),
        dtype=np.bool_,
    )


def _cohort_release_contract(
    cohort_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    selected_motion_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, bool]]:
    dataset = split_manifest.get("dataset")
    _require(isinstance(dataset, Mapping), "split dataset provenance must be a mapping")
    expected_digest = _sha256(
        dataset.get("cohort_manifest_sha256"),
        "split dataset.cohort_manifest_sha256",
    )
    actual_digest = _sha256_file(cohort_manifest_path, "cohort manifest")
    _require(actual_digest == expected_digest, "cohort manifest SHA-256 does not match split")
    cohort = _load_json_mapping(cohort_manifest_path, "cohort manifest")
    _require(
        cohort.get("kind") == "bones_seed_official_metadata_cohort",
        "invalid official cohort kind",
    )
    eligibility = cohort.get("eligibility")
    _require(isinstance(eligibility, Mapping), "cohort eligibility contract is missing")
    _require(
        eligibility.get("rule") == "official_release_filename_filter",
        "cohort eligibility rule mismatch",
    )
    keywords = eligibility.get("filter_keywords")
    _require(
        isinstance(keywords, list)
        and keywords
        and all(isinstance(value, str) and value for value in keywords),
        "cohort filter keywords are invalid",
    )
    motions = cohort.get("motions")
    _require(isinstance(motions, list), "cohort motions must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(motions):
        _require(isinstance(record, Mapping), f"cohort motions[{index}] must be a mapping")
        motion_key = record.get("motion_key")
        _require(
            isinstance(motion_key, str) and motion_key and motion_key not in indexed,
            f"cohort motions[{index}] has invalid motion_key",
        )
        indexed[motion_key] = record
    release_pass: dict[str, bool] = {}
    for motion_key in selected_motion_keys:
        _require(motion_key in indexed, f"selected motion {motion_key!r} is absent from cohort")
        filter_key = indexed[motion_key].get("release_filter_key")
        _require(
            isinstance(filter_key, str) and filter_key,
            f"cohort motion {motion_key!r} lacks release_filter_key",
        )
        release_pass[motion_key] = not any(
            keyword.lower() in filter_key.lower() for keyword in keywords
        )
        _require(
            release_pass[motion_key],
            f"split-selected motion {motion_key!r} fails the frozen release filename filter",
        )
    return (
        {
            "path": str(cohort_manifest_path),
            "sha256": actual_digest,
            "declared_split_path": dataset.get("cohort_manifest"),
            "rule": "official_release_filename_filter",
            "filter_keywords": keywords,
        },
        release_pass,
    )


def _motion_source_contract(
    payload: Mapping[str, Any],
    *,
    motion_key: str,
    inventory_record: Mapping[str, Any],
    source_joint_axes: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    required_fields = {"root_trans_offset", "pose_aa", "dof", "root_rot", "fps"}
    _require(
        required_fields <= set(payload),
        f"{motion_key} source payload lacks fields {sorted(required_fields - set(payload))}",
    )
    root = _numeric_array(payload["root_trans_offset"], f"{motion_key}.root_trans_offset", (3,))
    pose = _numeric_array(payload["pose_aa"], f"{motion_key}.pose_aa", (30, 3))
    joints = _numeric_array(payload["dof"], f"{motion_key}.dof", (29,))
    root_quaternion = _numeric_array(payload["root_rot"], f"{motion_key}.root_rot", (4,))
    frame_count = root.shape[0]
    _require(
        pose.shape[0] == joints.shape[0] == root_quaternion.shape[0] == frame_count,
        f"{motion_key} source arrays do not share one frame axis",
    )
    _require(
        frame_count == inventory_record.get("source_num_frames"),
        f"{motion_key} source frame count disagrees with inventory",
    )
    source_fps = _positive_fraction(payload["fps"], f"{motion_key}.fps")
    _require(
        source_fps == _positive_fraction(inventory_record.get("source_fps"), "inventory FPS"),
        f"{motion_key} source FPS disagrees with inventory",
    )
    projected = np.sum(pose[:, 1:, :] * source_joint_axes[None, :, :], axis=2)
    off_axis = pose[:, 1:, :] - projected[:, :, None] * source_joint_axes[None, :, :]
    max_off_axis = float(np.max(np.linalg.norm(off_axis, axis=2)))
    max_dof_disagreement = float(np.max(np.abs(projected - joints)))
    _require(
        max_off_axis <= 1e-6 and max_dof_disagreement <= 1e-6,
        f"{motion_key} pose_aa and dof representations disagree",
    )
    return (
        root,
        root_quaternion,
        pose,
        max_off_axis,
        max_dof_disagreement,
    )


def _proxy_set_sha256(motions: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "proxies": [
                {
                    "motion_key": record["motion_key"],
                    "reference_feasibility_sha256": record["proxy"]["reference_feasibility_sha256"],
                }
                for record in motions
            ]
        }
    )


def _manifest_summary(motions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hard_corruption_keys = sorted(
        record["motion_key"] for record in motions if record["proxy"]["hard_corruption_exclusion"]
    )
    hard_limit_keys = sorted(
        record["motion_key"]
        for record in motions
        if record["proxy"]["reference_constraint_flags"]["joint_hard_limit_violation"]
    )
    soft_limit_keys = sorted(
        record["motion_key"]
        for record in motions
        if record["proxy"]["reference_constraint_flags"]["joint_soft_limit_violation"]
    )
    velocity_limit_keys = sorted(
        record["motion_key"]
        for record in motions
        if record["proxy"]["reference_constraint_flags"]["joint_velocity_limit_violation"]
    )
    release_failure_keys = sorted(
        record["motion_key"]
        for record in motions
        if not record["proxy"]["release_eligibility"]["pass"]
    )
    return {
        "motion_count": len(motions),
        "hard_corruption_exclusion_count": len(hard_corruption_keys),
        "hard_corruption_exclusion_motion_keys": hard_corruption_keys,
        "joint_hard_limit_violation_count": len(hard_limit_keys),
        "joint_hard_limit_violation_motion_keys": hard_limit_keys,
        "joint_soft_limit_violation_count": len(soft_limit_keys),
        "joint_soft_limit_violation_motion_keys": soft_limit_keys,
        "joint_velocity_limit_violation_count": len(velocity_limit_keys),
        "joint_velocity_limit_violation_motion_keys": velocity_limit_keys,
        "release_filename_filter_failure_count": len(release_failure_keys),
        "release_filename_filter_failure_motion_keys": release_failure_keys,
    }


def build_reference_feasibility_manifest(
    *,
    split_manifest_path: str | Path,
    reference_inventory_path: str | Path,
    cohort_manifest_path: str | Path,
    g1_config_path: str | Path,
    urdf_path: str | Path,
    motion_command_config_path: str | Path,
    motion_mjcf_path: str | Path,
    thresholds: ReferenceFeasibilityThresholds | None = None,
    live_robot_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a split-, inventory-, source-, and robot-bound feasibility artifact."""

    split_path = Path(split_manifest_path).expanduser().resolve()
    inventory_path = Path(reference_inventory_path).expanduser().resolve()
    cohort_path = Path(cohort_manifest_path).expanduser().resolve()
    config_path = Path(g1_config_path).expanduser().resolve()
    resolved_urdf_path = Path(urdf_path).expanduser().resolve()
    command_config_path = Path(motion_command_config_path).expanduser().resolve()
    resolved_mjcf_path = Path(motion_mjcf_path).expanduser().resolve()
    for path, name in (
        (split_path, "split manifest"),
        (inventory_path, "reference-length inventory"),
        (cohort_path, "cohort manifest"),
        (config_path, "G1 config"),
        (resolved_urdf_path, "G1 URDF"),
        (command_config_path, "motion command config"),
        (resolved_mjcf_path, "motion-library MJCF"),
    ):
        _require(path.is_file(), f"{name} does not exist: {path}")

    split = _load_json_mapping(split_path, "split manifest")
    inventory = _load_json_mapping(inventory_path, "reference-length inventory")
    validate_split_manifest(split, verify_digest=True)
    validate_reference_length_inventory(
        inventory,
        split_manifest=split,
        verify_digest=True,
        verify_source_files=True,
    )
    selected = list(inventory["selected_motion_keys"])
    _require(
        selected == sorted(selected) and len(selected) == len(set(selected)),
        "reference inventory selection must be canonical",
    )
    split_records = {
        record["motion_key"]: record
        for record in split["motions"]
        if record["partition"] == "D_atlas"
    }
    _require(
        set(selected) <= set(split_records),
        "reference inventory contains a motion outside split D_atlas",
    )
    if inventory["artifact_mode"] == "scientific":
        _require(
            selected == sorted(split_records),
            "scientific feasibility manifest must cover the full D_atlas",
        )

    config = thresholds if thresholds is not None else ReferenceFeasibilityThresholds()
    _require(
        isinstance(config, ReferenceFeasibilityThresholds),
        "thresholds must be ReferenceFeasibilityThresholds",
    )
    config.validate()
    robot_contract = derive_g1_reference_contract(
        config_path,
        resolved_urdf_path,
        live_robot_data=live_robot_data,
    )
    validate_g1_reference_contract(robot_contract, verify_source_files=True)
    motion_reference_contract, mjcf_body_records = _derive_motion_reference_contract(
        command_config_path,
        resolved_mjcf_path,
        robot_contract=robot_contract,
    )
    validate_motion_reference_contract(
        motion_reference_contract,
        robot_contract=robot_contract,
        verify_source_files=True,
    )
    _require(
        motion_reference_contract["target_fps"] == int(inventory["target_fps"]),
        "motion command target FPS disagrees with the reference inventory",
    )
    cohort_binding, release_pass = _cohort_release_contract(cohort_path, split, selected)

    _, child_to_joint = _parse_urdf(resolved_urdf_path)
    urdf_by_name = {record["name"]: record for record in child_to_joint.values()}
    source_joint_names = list(robot_contract["mujoco_source_joint_names"])
    source_joint_records = [urdf_by_name[name] for name in source_joint_names]
    source_joint_axes = np.asarray(
        [record["axis"] for record in source_joint_records],
        dtype=np.float64,
    )
    il_records = {
        record["joint_name"]: record for record in robot_contract["joint_contract_isaaclab_order"]
    }
    source_lower = np.asarray(
        [il_records[name]["hard_lower_radians"] for name in source_joint_names],
        dtype=np.float64,
    )
    source_upper = np.asarray(
        [il_records[name]["hard_upper_radians"] for name in source_joint_names],
        dtype=np.float64,
    )
    source_soft_lower = np.asarray(
        [il_records[name]["soft_lower_radians"] for name in source_joint_names],
        dtype=np.float64,
    )
    source_soft_upper = np.asarray(
        [il_records[name]["soft_upper_radians"] for name in source_joint_names],
        dtype=np.float64,
    )
    source_velocity = np.asarray(
        [
            il_records[name]["actuator_velocity_limit_radians_per_second"]
            for name in source_joint_names
        ],
        dtype=np.float64,
    )

    inventory_records = {record["motion_key"]: record for record in inventory["motions"]}
    target_fps = int(inventory["target_fps"])
    motion_records: list[dict[str, Any]] = []
    for motion_key in selected:
        inventory_record = inventory_records[motion_key]
        source_path = Path(inventory_record["source_path"]).expanduser().resolve()
        _require(
            source_path == Path(split_records[motion_key]["robot_path"]).expanduser().resolve(),
            f"{motion_key} inventory source path does not match split robot_path",
        )
        source_digest = _sha256_file(source_path, f"motion {motion_key}")
        _require(
            source_digest == inventory_record["source_file_sha256"],
            f"{motion_key} source SHA-256 does not match inventory",
        )
        payload = _load_motion_payload(source_path, motion_key)
        (
            source_root,
            source_root_quaternion,
            source_pose_rotation_vector,
            pose_off_axis_max,
            dof_pose_disagreement_max,
        ) = _motion_source_contract(
            payload,
            motion_key=motion_key,
            inventory_record=inventory_record,
            source_joint_axes=source_joint_axes,
        )
        source_fps = _positive_fraction(inventory_record["source_fps"], "inventory FPS")
        runtime_joints, runtime_root, runtime_quaternion, runtime_local_rotations = (
            _runtime_reference(
                pose_rotation_vector=source_pose_rotation_vector,
                root_position=source_root,
                source_fps=source_fps,
                target_fps=target_fps,
                target_num_frames=int(inventory_record["target_num_frames"]),
            )
        )
        foot_positions = _foot_positions_from_mjcf(
            body_records=mjcf_body_records,
            local_rotation_matrices=runtime_local_rotations,
            root_position=runtime_root,
            root_quaternion_xyzw=runtime_quaternion,
        )
        contacts = _sonic_contact_schedule(foot_positions, config)
        proxy = compute_reference_feasibility_proxy(
            motion_key=motion_key,
            joint_position=runtime_joints,
            root_position=runtime_root,
            root_quaternion_xyzw=runtime_quaternion,
            source_root_quaternion_xyzw=source_root_quaternion,
            source_root_rotation_vector=source_pose_rotation_vector[:, 0, :],
            reference_foot_position=foot_positions,
            reference_foot_contact=contacts,
            reference_fps=target_fps,
            joint_lower_limits=source_lower,
            joint_upper_limits=source_upper,
            soft_joint_lower_limits=source_soft_lower,
            soft_joint_upper_limits=source_soft_upper,
            joint_velocity_limits=source_velocity,
            joint_names=source_joint_names,
            robot_contract_sha256=robot_contract[G1_REFERENCE_CONTRACT_DIGEST_FIELD],
            source_file_sha256=source_digest,
            source_schema_and_file_hash_verified=True,
            release_filename_filter_pass=release_pass[motion_key],
            thresholds=config,
        )
        motion_records.append(
            {
                "motion_key": motion_key,
                "source_group_id": split_records[motion_key]["source_group_id"],
                "source_binding": {
                    "path": str(source_path),
                    "sha256": source_digest,
                    "source_num_frames": int(inventory_record["source_num_frames"]),
                    "source_fps": {
                        "numerator": source_fps.numerator,
                        "denominator": source_fps.denominator,
                    },
                },
                "source_integrity_checks": {
                    "inventory_file_hash_verified": True,
                    "singleton_motion_key_verified": True,
                    "required_array_schema_verified": True,
                    "shared_source_frame_axis_verified": True,
                    "pose_joint_axis_max_residual_radians": pose_off_axis_max,
                    "pose_dof_max_disagreement_radians": dof_pose_disagreement_max,
                    "pose_dof_contract_verified": True,
                },
                "runtime_reference": {
                    "target_fps": target_fps,
                    "target_num_frames": int(inventory_record["target_num_frames"]),
                    "motion_reference_contract_sha256": motion_reference_contract[
                        "motion_reference_contract_sha256"
                    ],
                    "timeline_rule": "SONIC_torch_float32_arange_phase_blend",
                    "joint_rule": "SONIC_quaternion_slerp_then_angle_axis_sum",
                    "root_rotation_rule": "SONIC_quaternion_slerp",
                    "root_translation_rule": "SONIC_float32_linear_interpolation",
                    "foot_positions_rule": "pinned_motion_library_MJCF_forward_kinematics",
                    "contact_rule": "sonic_motion_lib_foot_detect_v1",
                    "cpu_reconstruction_uses_pinned_runtime_math": True,
                    "runtime_readback_verified": False,
                    "runtime_readback_status": "unverified_no_isaac_launch",
                },
                "proxy": proxy,
            }
        )

    _require(bool(motion_records), "feasibility manifest selection is empty")
    feature_names = motion_records[0]["proxy"]["continuous_feature_names"]
    _require(
        all(
            record["proxy"]["continuous_feature_names"] == feature_names
            for record in motion_records
        ),
        "motion proxy feature schemas differ",
    )
    live_verified = bool(robot_contract["live_robot_data_validation"]["verified"])
    artifact_mode = str(inventory["artifact_mode"])
    scientific_ready = artifact_mode == "scientific" and live_verified
    result: dict[str, Any] = {
        "kind": REFERENCE_FEASIBILITY_MANIFEST_KIND,
        "schema_version": REFERENCE_FEASIBILITY_MANIFEST_SCHEMA_VERSION,
        "artifact_mode": artifact_mode,
        "partition": "D_atlas",
        "selection_complete_for_d_atlas": bool(inventory["selection_complete_for_d_atlas"]),
        "scientific_use": scientific_ready,
        "scientific_use_status": (
            "ready"
            if scientific_ready
            else (
                "provisional_pending_live_robot_data_velocity_and_order_validation"
                if artifact_mode == "scientific"
                else "non_scientific_pilot"
            )
        ),
        "scope": REFERENCE_FEASIBILITY_SCOPE,
        "dynamic_feasibility_proof": False,
        "split_binding": {
            "path": str(split_path),
            "file_sha256": _sha256_file(split_path, "split manifest"),
            "split_sha256": split["split_sha256"],
            "selection_sha256": split["selection_sha256"],
        },
        "reference_inventory_binding": {
            "path": str(inventory_path),
            "file_sha256": _sha256_file(inventory_path, "reference-length inventory"),
            "inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
            "source_file_set_sha256": inventory["source_file_set_sha256"],
        },
        "cohort_binding": cohort_binding,
        "robot_contract": robot_contract,
        "motion_reference_contract": motion_reference_contract,
        "implementation_bindings": _implementation_bindings(),
        "runtime_dependency_versions": _runtime_dependency_versions(),
        "thresholds": asdict(config),
        "continuous_feature_schema": {
            "names": feature_names,
            "dimension": len(feature_names),
            "outcome_independent": True,
            "fit_partition": None,
            "scaler_or_outlier_threshold_fit_here": False,
        },
        "hard_corruption_exclusions_are_separate_from_continuous_features": True,
        "selected_motion_keys": selected,
        "motion_count": len(motion_records),
        "proxy_set_sha256": _proxy_set_sha256(motion_records),
        "motions": motion_records,
        "summary": _manifest_summary(motion_records),
    }
    result[REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD] = canonical_sha256(result)
    validate_reference_feasibility_manifest(
        result,
        split_manifest=split,
        reference_inventory=inventory,
        verify_source_files=True,
        verify_robot_sources=True,
        verify_motion_sources=True,
        verify_implementation_sources=True,
    )
    return result


def validate_reference_feasibility_manifest(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    reference_inventory: Mapping[str, Any] | None = None,
    verify_digest: bool = True,
    verify_source_files: bool = False,
    verify_robot_sources: bool = False,
    verify_motion_sources: bool = False,
    verify_implementation_sources: bool = False,
) -> None:
    """Validate selection, provenance, proxy schemas, summaries, and self-digests."""

    _require(
        manifest.get("kind") == REFERENCE_FEASIBILITY_MANIFEST_KIND,
        "invalid feasibility manifest kind",
    )
    _require(
        manifest.get("schema_version") == REFERENCE_FEASIBILITY_MANIFEST_SCHEMA_VERSION,
        "unsupported feasibility manifest schema version",
    )
    artifact_mode = manifest.get("artifact_mode")
    _require(artifact_mode in {"scientific", "pilot"}, "invalid feasibility artifact_mode")
    _require(manifest.get("partition") == "D_atlas", "feasibility partition must be D_atlas")
    _require(
        manifest.get("scope") == REFERENCE_FEASIBILITY_SCOPE
        and manifest.get("dynamic_feasibility_proof") is False,
        "feasibility scope must remain a kinematic proxy, not a dynamics claim",
    )
    _require(
        manifest.get("hard_corruption_exclusions_are_separate_from_continuous_features") is True,
        "hard corruption exclusions must be separate from continuous features",
    )
    selected = manifest.get("selected_motion_keys")
    _require(
        isinstance(selected, list)
        and selected
        and selected == sorted(set(selected))
        and all(isinstance(value, str) and value for value in selected),
        "selected_motion_keys must be non-empty and canonical",
    )
    _require(manifest.get("motion_count") == len(selected), "motion_count mismatch")
    _require(
        manifest.get("selection_complete_for_d_atlas") is (artifact_mode == "scientific"),
        "selection completeness must agree with artifact_mode",
    )

    robot_contract = manifest.get("robot_contract")
    _require(isinstance(robot_contract, Mapping), "robot_contract must be a mapping")
    validate_g1_reference_contract(
        robot_contract,
        verify_source_files=verify_robot_sources,
    )
    motion_reference_contract = manifest.get("motion_reference_contract")
    _require(
        isinstance(motion_reference_contract, Mapping),
        "motion_reference_contract must be a mapping",
    )
    validate_motion_reference_contract(
        motion_reference_contract,
        robot_contract=robot_contract,
        verify_source_files=verify_motion_sources,
    )
    live_verified = robot_contract["live_robot_data_validation"]["verified"]
    expected_scientific_use = artifact_mode == "scientific" and live_verified
    _require(
        manifest.get("scientific_use") is expected_scientific_use,
        "scientific_use must remain false until live robot.data validation passes",
    )
    expected_status = (
        "ready"
        if expected_scientific_use
        else (
            "provisional_pending_live_robot_data_velocity_and_order_validation"
            if artifact_mode == "scientific"
            else "non_scientific_pilot"
        )
    )
    _require(manifest.get("scientific_use_status") == expected_status, "scientific status mismatch")

    implementation_bindings = manifest.get("implementation_bindings")
    _require(
        isinstance(implementation_bindings, list)
        and len(implementation_bindings) == len(_IMPLEMENTATION_SOURCE_ROLES),
        "implementation_bindings are incomplete",
    )
    indexed_implementation_bindings: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(implementation_bindings):
        _require(
            isinstance(binding, Mapping),
            f"implementation_bindings[{index}] must be a mapping",
        )
        role = binding.get("role")
        _require(
            role in _IMPLEMENTATION_SOURCE_ROLES and role not in indexed_implementation_bindings,
            f"implementation_bindings[{index}] role is invalid",
        )
        _sha256(binding.get("sha256"), f"implementation_bindings[{index}].sha256")
        path = Path(str(binding.get("path")))
        _require(
            path == _IMPLEMENTATION_SOURCE_ROLES[str(role)].resolve(),
            f"implementation_bindings[{index}] path is not canonical",
        )
        if verify_implementation_sources:
            _require(path.is_file(), f"implementation source is missing: {path}")
            _require(
                _sha256_file(path, f"implementation source {role}") == binding["sha256"],
                f"implementation source {role} SHA-256 mismatch",
            )
        indexed_implementation_bindings[str(role)] = binding
    _require(
        list(indexed_implementation_bindings) == sorted(_IMPLEMENTATION_SOURCE_ROLES),
        "implementation_bindings must be in canonical role order",
    )
    dependency_versions = manifest.get("runtime_dependency_versions")
    _require(
        isinstance(dependency_versions, Mapping)
        and set(dependency_versions)
        == {"numpy", "python", "torch", "torch_device", "torch_default_dtype"}
        and all(isinstance(value, str) and value for value in dependency_versions.values())
        and dependency_versions.get("torch_device") == "cpu",
        "runtime dependency version contract is invalid",
    )
    if verify_implementation_sources:
        _require(
            dict(dependency_versions) == _runtime_dependency_versions(),
            "runtime dependency versions disagree with the frozen artifact",
        )

    split_binding = manifest.get("split_binding")
    inventory_binding = manifest.get("reference_inventory_binding")
    cohort_binding = manifest.get("cohort_binding")
    for binding, name in (
        (split_binding, "split_binding"),
        (inventory_binding, "reference_inventory_binding"),
        (cohort_binding, "cohort_binding"),
    ):
        _require(isinstance(binding, Mapping), f"{name} must be a mapping")
        _sha256(binding.get("file_sha256", binding.get("sha256")), f"{name} file digest")
        path = Path(str(binding.get("path")))
        if verify_source_files:
            _require(path.is_file(), f"{name} source is missing: {path}")
            _require(
                _sha256_file(path, name) == binding.get("file_sha256", binding.get("sha256")),
                f"{name} source SHA-256 mismatch",
            )

    feature_schema = manifest.get("continuous_feature_schema")
    _require(isinstance(feature_schema, Mapping), "continuous_feature_schema is missing")
    feature_names = feature_schema.get("names")
    _require(
        isinstance(feature_names, list)
        and feature_names
        and len(feature_names) == feature_schema.get("dimension")
        and len(set(feature_names)) == len(feature_names),
        "continuous feature schema is invalid",
    )
    _require(
        feature_schema.get("outcome_independent") is True
        and feature_schema.get("fit_partition") is None
        and feature_schema.get("scaler_or_outlier_threshold_fit_here") is False,
        "feasibility manifest may not fit outcome-dependent preprocessing",
    )
    thresholds = manifest.get("thresholds")
    _require(isinstance(thresholds, Mapping), "thresholds must be a mapping")
    try:
        threshold_config = ReferenceFeasibilityThresholds(**dict(thresholds))
        threshold_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid frozen feasibility thresholds: {error}") from error
    _require(
        dict(thresholds) == asdict(threshold_config),
        "thresholds must contain exactly the frozen schema fields",
    )

    motions = manifest.get("motions")
    _require(isinstance(motions, list) and len(motions) == len(selected), "motions mismatch")
    robot_digest = robot_contract[G1_REFERENCE_CONTRACT_DIGEST_FIELD]
    normalized_motions: list[Mapping[str, Any]] = []
    for index, record in enumerate(motions):
        _require(isinstance(record, Mapping), f"motions[{index}] must be a mapping")
        _require(record.get("motion_key") == selected[index], "motions must follow selected order")
        source_binding = record.get("source_binding")
        integrity = record.get("source_integrity_checks")
        runtime = record.get("runtime_reference")
        proxy = record.get("proxy")
        _require(isinstance(source_binding, Mapping), f"motions[{index}] source binding missing")
        _require(isinstance(integrity, Mapping), f"motions[{index}] integrity checks missing")
        _require(isinstance(runtime, Mapping), f"motions[{index}] runtime reference missing")
        _require(isinstance(proxy, Mapping), f"motions[{index}] proxy missing")
        _require(
            isinstance(record.get("source_group_id"), str) and record["source_group_id"],
            f"motions[{index}] source_group_id missing",
        )
        source_digest = _sha256(
            source_binding.get("sha256"),
            f"motions[{index}].source_binding.sha256",
        )
        _require(
            all(
                integrity.get(name) is True
                for name in (
                    "inventory_file_hash_verified",
                    "singleton_motion_key_verified",
                    "required_array_schema_verified",
                    "shared_source_frame_axis_verified",
                    "pose_dof_contract_verified",
                )
            ),
            f"motions[{index}] source integrity verification incomplete",
        )
        _require(
            integrity.get("pose_joint_axis_max_residual_radians", math.inf) <= 1e-6
            and integrity.get("pose_dof_max_disagreement_radians", math.inf) <= 1e-6,
            f"motions[{index}] pose/dof integrity tolerance exceeded",
        )
        _require(runtime.get("runtime_readback_verified") is False, "unexpected runtime readback")
        _require(
            runtime.get("runtime_readback_status") == "unverified_no_isaac_launch",
            f"motions[{index}] must state unverified runtime readback",
        )
        _require(
            runtime.get("timeline_rule") == "SONIC_torch_float32_arange_phase_blend"
            and runtime.get("joint_rule") == "SONIC_quaternion_slerp_then_angle_axis_sum"
            and runtime.get("root_rotation_rule") == "SONIC_quaternion_slerp"
            and runtime.get("root_translation_rule") == "SONIC_float32_linear_interpolation"
            and runtime.get("foot_positions_rule")
            == "pinned_motion_library_MJCF_forward_kinematics"
            and runtime.get("contact_rule") == "sonic_motion_lib_foot_detect_v1"
            and runtime.get("cpu_reconstruction_uses_pinned_runtime_math") is True,
            f"motions[{index}] runtime reference contract mismatch",
        )
        _require(
            runtime.get("motion_reference_contract_sha256")
            == motion_reference_contract["motion_reference_contract_sha256"],
            f"motions[{index}] motion reference digest mismatch",
        )
        _require(
            proxy.get("kind") == REFERENCE_FEASIBILITY_KIND
            and proxy.get("schema_version") == REFERENCE_FEASIBILITY_SCHEMA_VERSION,
            f"motions[{index}] proxy schema mismatch",
        )
        _require(proxy.get("motion_key") == selected[index], f"motions[{index}] proxy key mismatch")
        _require(proxy.get("source_file_sha256") == source_digest, "proxy source digest mismatch")
        _require(proxy.get("robot_contract_sha256") == robot_digest, "proxy robot digest mismatch")
        _require(
            proxy.get("scope") == REFERENCE_FEASIBILITY_SCOPE
            and proxy.get("dynamic_feasibility_proof") is False
            and proxy.get("inverse_dynamics_used") is False
            and proxy.get("policy_rollout_used") is False,
            f"motions[{index}] proxy overstates reference feasibility",
        )
        _require(
            proxy.get("thresholds") == dict(thresholds),
            f"motions[{index}] threshold contract mismatch",
        )
        _require(
            proxy.get("reference_fps") == runtime.get("target_fps")
            and proxy.get("reference_num_frames") == runtime.get("target_num_frames")
            and proxy.get("source_file_num_frames") == source_binding.get("source_num_frames"),
            f"motions[{index}] reference timeline binding mismatch",
        )
        _require(
            proxy.get("continuous_feature_names") == feature_names
            and len(proxy.get("continuous_feature_vector", [])) == len(feature_names),
            f"motions[{index}] feature schema mismatch",
        )
        feature_values = proxy.get("continuous_feature_vector")
        _require(
            isinstance(feature_values, list)
            and all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in feature_values
            ),
            f"motions[{index}] continuous features must be finite numeric values",
        )
        corruption_checks = proxy.get("hard_corruption_checks")
        _require(
            isinstance(corruption_checks, Mapping)
            and set(corruption_checks) == _HARD_CORRUPTION_CHECK_FIELDS
            and all(isinstance(value, bool) for value in corruption_checks.values()),
            f"motions[{index}] hard corruption checks are invalid",
        )
        _require(
            corruption_checks["source_schema_and_file_hash_verified"] is True,
            f"motions[{index}] source integrity flag mismatch",
        )
        expected_corruption_exclusion = not corruption_checks[
            "source_schema_and_file_hash_verified"
        ] or any(
            value
            for name, value in corruption_checks.items()
            if name != "source_schema_and_file_hash_verified"
        )
        _require(
            proxy.get("hard_corruption_exclusion") is expected_corruption_exclusion,
            f"motions[{index}] hard corruption exclusion is inconsistent",
        )
        constraint_flags = proxy.get("reference_constraint_flags")
        _require(
            isinstance(constraint_flags, Mapping)
            and set(constraint_flags) == _REFERENCE_CONSTRAINT_FLAG_FIELDS
            and all(isinstance(value, bool) for value in constraint_flags.values()),
            f"motions[{index}] reference constraint flags are invalid",
        )
        expected_constraint_screen_pass = not (
            constraint_flags["joint_hard_limit_violation"]
            or constraint_flags["joint_velocity_limit_violation"]
        )
        _require(
            proxy.get("reference_constraint_screen_pass") is expected_constraint_screen_pass,
            f"motions[{index}] reference constraint screen is inconsistent",
        )
        contact_contract = proxy.get("reference_contact_contract")
        _require(
            isinstance(contact_contract, Mapping)
            and contact_contract.get("contact_labels_are_kinematically_derived") is True
            and contact_contract.get("independent_physical_contact_labels_available") is False,
            f"motions[{index}] contact-label scope is invalid",
        )
        expected_proxy_digest = _sha256(
            proxy.get("reference_feasibility_sha256"),
            f"motions[{index}].reference_feasibility_sha256",
        )
        _require(
            expected_proxy_digest
            == canonical_sha256(proxy, digest_field="reference_feasibility_sha256"),
            f"motions[{index}] reference feasibility digest mismatch",
        )
        if verify_source_files:
            source_path = Path(str(source_binding.get("path")))
            _require(source_path.is_file(), f"motion source is missing: {source_path}")
            _require(
                _sha256_file(source_path, f"motion {selected[index]}") == source_digest,
                f"motions[{index}] source file SHA-256 mismatch",
            )
        normalized_motions.append(record)
    _require(
        manifest.get("proxy_set_sha256") == _proxy_set_sha256(normalized_motions),
        "proxy_set_sha256 mismatch",
    )
    _require(manifest.get("summary") == _manifest_summary(normalized_motions), "summary mismatch")

    if split_manifest is not None:
        validate_split_manifest(split_manifest, verify_digest=True)
        _require(
            split_binding.get("split_sha256") == split_manifest["split_sha256"]
            and split_binding.get("selection_sha256") == split_manifest["selection_sha256"],
            "split binding mismatch",
        )
        atlas_keys = sorted(
            record["motion_key"]
            for record in split_manifest["motions"]
            if record["partition"] == "D_atlas"
        )
        if artifact_mode == "scientific":
            _require(selected == atlas_keys, "scientific manifest does not cover full D_atlas")
        else:
            _require(
                set(selected) < set(atlas_keys), "pilot manifest is not a strict D_atlas subset"
            )
        split_records = {record["motion_key"]: record for record in split_manifest["motions"]}
        for index, record in enumerate(normalized_motions):
            _require(
                record["source_group_id"] == split_records[selected[index]]["source_group_id"],
                f"motions[{index}] source_group_id disagrees with split",
            )
    if reference_inventory is not None:
        validate_reference_length_inventory(
            reference_inventory,
            split_manifest=split_manifest,
            verify_digest=True,
            verify_source_files=verify_source_files,
        )
        _require(
            inventory_binding.get("inventory_sha256")
            == reference_inventory[REFERENCE_LENGTH_DIGEST_FIELD]
            and inventory_binding.get("source_file_set_sha256")
            == reference_inventory["source_file_set_sha256"]
            and selected == reference_inventory["selected_motion_keys"],
            "reference inventory binding mismatch",
        )
        inventory_records = {
            record["motion_key"]: record for record in reference_inventory["motions"]
        }
        for index, record in enumerate(normalized_motions):
            expected = inventory_records[selected[index]]
            _require(
                record["source_binding"]["sha256"] == expected["source_file_sha256"]
                and record["source_binding"]["source_num_frames"] == expected["source_num_frames"]
                and record["runtime_reference"]["target_num_frames"]
                == expected["target_num_frames"],
                f"motions[{index}] inventory binding mismatch",
            )
    if verify_digest:
        expected_digest = _sha256(
            manifest.get(REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD),
            REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD,
        )
        _require(
            expected_digest
            == canonical_sha256(
                manifest,
                digest_field=REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD,
            ),
            f"{REFERENCE_FEASIBILITY_MANIFEST_DIGEST_FIELD} mismatch",
        )
