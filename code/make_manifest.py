#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate FILE_MANIFEST.json (path -> size_bytes + sha256) for this release.

Run from the repository root:
    python code/make_manifest.py
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "FILE_MANIFEST.json")
SKIP = {"FILE_MANIFEST.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    files = {}
    for root, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            if rel in SKIP:
                continue
            files[rel] = {"size_bytes": os.path.getsize(full), "sha256": sha256(full)}
    payload = {
        "status": "PASS",
        "algorithm": "sha256",
        "file_count": len(files),
        "files": dict(sorted(files.items())),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
