#!/usr/bin/env python3
"""Create a deterministic archive of the published diagnostic records and controls."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def bundle(folder):
    archive = folder / "breakpoint-evidence.tar.gz"
    manifest = folder / "breakpoint-evidence-manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("keep the existing published archive immutable")
    files = {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}
    with archive.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as stream:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(content), 0o644, 0
                    stream.addfile(info, io.BytesIO(content))
    payload = {
        "archive": {
            "name": archive.name,
            "sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        "files": [
            {"path": p, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}
            for p, raw in files.items()
        ],
        "scope": "diagnostic records and four controls; external raw physics and motion banks excluded",
    }
    with manifest.open("x") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
    print(json.dumps({"files": len(files), "bytes": archive.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle(args.out)
