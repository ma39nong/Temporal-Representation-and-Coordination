"""Stable fingerprints and report writing without touching configured inputs."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_report(path, report, *, protected=()):
    path = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        if path == root or path.is_relative_to(root):
            raise ValueError('Report output must be outside read-only input paths')
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
