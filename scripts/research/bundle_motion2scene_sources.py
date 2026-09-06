#!/usr/bin/env python3
"""Publish the local Python dependency closure, without models or motion data."""

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def candidate(module, directory):
    stem = directory.joinpath(*module.split(".")) if module else directory
    for path in [stem.with_suffix(".py"), stem / "__init__.py"]:
        if path.is_file() and path.resolve().is_relative_to(ROOT):
            return path.resolve()
    return None


def closure(seeds):
    done, pending = set(), list(seeds)
    while pending:
        path = pending.pop().resolve()
        if path in done:
            continue
        done.add(path)
        for parent in path.parents:
            if not parent.is_relative_to(ROOT):
                break
            init = parent / "__init__.py"
            if init.is_file() and init not in done:
                pending.append(init)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules, bases = [], [ROOT, ROOT / "scripts/research", path.parent]
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.level:
                    bases = [path.parents[node.level - 1]]
                modules += [
                    ((node.module + ".") if node.module else "") + a.name for a in node.names
                ]
            for module in modules:
                for base in bases:
                    found = candidate(module, base)
                    if found and found not in done:
                        pending.append(found)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene/evidence")
    args = parser.parse_args()
    seeds = list((ROOT / "scripts/research").glob("*motion2scene*.py"))
    # Hydra targets are string imports; AST import closure alone cannot discover them.
    seeds += list((ROOT / "gear_sonic/dataset_generation/hallucination").glob("motion2scene*.py"))
    seeds += list((ROOT / "tests/dataset_generation").glob("test_motion2scene*.py"))
    seeds += list((ROOT / "tests/dataset_generation").glob("test_capsule_box*.py"))
    seeds += [ROOT / "tests/dataset_generation/test_placement_certificate.py", Path(__file__)]
    paths = closure(seeds) | set((ROOT / "docs/motion2scene").glob("*.md"))
    paths |= {ROOT / "pyproject.toml", ROOT / "AGENTS.md", ROOT / "LICENSE"}
    paths |= set((ROOT / "legal").glob("*.txt"))
    contents = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(paths)}
    manifest = {
        "role": "source snapshot; external motion banks, models, Kimodo and environments are not bundled",
        "python": sys.version,
        "installed_packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "files": [
            {"path": p, "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
            for p, data in contents.items()
        ],
    }
    with (args.out / "research-source.tar.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, data in {
                    **contents,
                    "SOURCE_MANIFEST.json": (
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                    ).encode(),
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
    path = args.out / "research-source.tar.gz"
    manifest["archive"] = {
        "path": path.name,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with (args.out / "research-source-manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"source_files": len(contents), "archive_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
