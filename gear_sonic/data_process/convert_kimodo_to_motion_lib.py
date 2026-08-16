#!/usr/bin/env python3
"""Convert a Kimodo-G1 qpos CSV into a SONIC motion-library PKL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gear_sonic.dataset_generation.kimodo_motion_adapter import (
    load_kimodo_qpos_csv,
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.dataset_generation.schemas import (
    ArtifactRef,
    ConversionResult,
    GenerationResult,
)

CONVERTER_NAME = "gear_sonic.dataset_generation.kimodo_motion_adapter"
CONVERTER_VERSION = "1"
LEGACY_DEFAULT_SOURCE_FPS = 30.0


def _select_parent_source_artifact(
    parent: GenerationResult,
    input_artifact: ArtifactRef,
    *,
    requested_name: str | None,
) -> str:
    """Bind the CLI input bytes to exactly one artifact in a verified parent result."""
    if requested_name is not None:
        matches = [artifact for artifact in parent.artifacts if artifact.name == requested_name]
        if not matches:
            raise ValueError(f"parent GenerationResult has no artifact named {requested_name!r}")
        candidates = matches
    else:
        candidates = list(parent.artifacts)

    content_matches = [
        artifact
        for artifact in candidates
        if artifact.sha256 == input_artifact.sha256
        and artifact.size_bytes == input_artifact.size_bytes
        and artifact.media_type == input_artifact.media_type
    ]
    if not content_matches:
        qualifier = f" named {requested_name!r}" if requested_name is not None else ""
        raise ValueError(
            f"converter input does not match any parent GenerationResult artifact{qualifier}"
        )
    if len(content_matches) > 1:
        names = sorted(artifact.name for artifact in content_matches)
        raise ValueError(
            "converter input matches multiple parent artifacts; pass --source-artifact-name "
            f"to select one of {names}"
        )
    return content_matches[0].name


def _reject_path_collisions(paths: dict[str, Path]) -> None:
    """Reject configurations that could overwrite an input or provenance record."""
    resolved: dict[Path, list[str]] = {}
    for label, path in paths.items():
        resolved.setdefault(path.resolve(), []).append(label)
    collisions = {path: labels for path, labels in resolved.items() if len(labels) > 1}
    if collisions:
        details = ", ".join(
            f"{path} ({'/'.join(labels)})" for path, labels in sorted(collisions.items())
        )
        raise ValueError(f"input/output/manifest paths must be distinct: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Kimodo G1 36-column CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output SONIC joblib PKL")
    parser.add_argument("--motion-key", help="Motion key; defaults to the input filename stem")
    parser.add_argument(
        "--generation-result",
        type=Path,
        help=(
            "verified parent GenerationResult JSON; enables a typed, hash-linked "
            "ConversionResult manifest"
        ),
    )
    parser.add_argument(
        "--source-artifact-name",
        help="parent artifact name for --input; required only when identical artifacts are ambiguous",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        help="source FPS; derived from --generation-result, otherwise defaults to 30",
    )
    parser.add_argument("--scene-start", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--scene-yaw", type=float, default=0.0, help="Scene-local yaw in radians")
    parser.add_argument(
        "--keep-source-origin",
        action="store_true",
        help="Do not subtract the first root XY before applying the scene transform",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Conversion manifest path; defaults to <output>.manifest.json",
    )
    args = parser.parse_args()

    if args.source_artifact_name is not None and args.generation_result is None:
        raise ValueError("--source-artifact-name requires --generation-result")
    if args.generation_result is not None and (
        args.motion_key is None or not args.motion_key.strip()
    ):
        raise ValueError("--motion-key is required with --generation-result")

    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    protected_paths = {
        "input": args.input,
        "output": args.output,
        "manifest": manifest_path,
    }
    if args.generation_result is not None:
        protected_paths["generation_result"] = args.generation_result
    _reject_path_collisions(protected_paths)

    input_artifact = ArtifactRef.from_path("kimodo_qpos_csv", args.input, media_type="text/csv")
    parent = None
    source_artifact_name = None
    if args.generation_result is not None:
        parent = GenerationResult.read_json(args.generation_result)
        output_paths = {args.output.resolve(), manifest_path.resolve()}
        conflicting_parent_artifacts = [
            artifact.name
            for artifact in parent.artifacts
            if Path(artifact.path).resolve() in output_paths
        ]
        if conflicting_parent_artifacts:
            raise ValueError(
                "output or manifest would overwrite parent artifact(s): "
                f"{sorted(conflicting_parent_artifacts)}"
            )
        source_artifact_name = _select_parent_source_artifact(
            parent,
            input_artifact,
            requested_name=args.source_artifact_name,
        )
        if args.source_fps is not None and not math.isclose(
            float(args.source_fps),
            float(parent.source_fps),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"--source-fps={args.source_fps:g} does not match parent "
                f"source_fps={parent.source_fps:g}"
            )
        source_fps = float(parent.source_fps)
    else:
        source_fps = (
            LEGACY_DEFAULT_SOURCE_FPS if args.source_fps is None else float(args.source_fps)
        )

    qpos = load_kimodo_qpos_csv(args.input)
    entry = qpos_to_sonic_motion_entry(
        qpos,
        source_fps=source_fps,
        scene_start_xyz=args.scene_start,
        scene_yaw=args.scene_yaw,
        canonicalize_horizontal_origin=not args.keep_source_origin,
    )
    motion_key = args.motion_key or args.input.stem
    save_sonic_motion_file(args.output, motion_key=motion_key, motion_entry=entry)

    output_artifact = ArtifactRef.from_path(
        "sonic_motion_lib", args.output, media_type="application/x-joblib"
    )
    if parent is not None:
        conversion = ConversionResult(
            episode_request_id=parent.episode_request_id,
            source_generation_result_id=parent.result_id,
            source_artifact_name=source_artifact_name,
            converter_name=CONVERTER_NAME,
            converter_version=CONVERTER_VERSION,
            source_fps=source_fps,
            motion_key=motion_key,
            frame_count=int(qpos.shape[0]),
            scene_start_xyz=tuple(args.scene_start),
            scene_yaw=args.scene_yaw,
            canonicalize_horizontal_origin=not args.keep_source_origin,
            artifacts=(input_artifact, output_artifact),
        )
        conversion.verify_parent(parent)
        conversion.write_json(manifest_path, include_legacy_aliases=True)
    else:
        # Preserve the original ad-hoc manifest when no typed parent is supplied.
        manifest = {
            "schema_version": 1,
            "converter": CONVERTER_NAME,
            "motion_key": motion_key,
            "source_fps": source_fps,
            "frame_count": int(qpos.shape[0]),
            "scene_start_xyz": list(args.scene_start),
            "scene_yaw": args.scene_yaw,
            "canonicalize_horizontal_origin": not args.keep_source_origin,
            "input": input_artifact.to_dict(),
            "output": output_artifact.to_dict(),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")

    print(f"wrote SONIC motion '{motion_key}' to {args.output}")
    print(f"wrote conversion manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
