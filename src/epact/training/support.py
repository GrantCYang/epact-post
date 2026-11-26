"""Training integrity, gradient accumulation and adapter identity."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"configuration changed: {path}; use a new run directory")
    atomic(path, value)


def accumulation_window(index, total, accumulation):
    """Return the actual window size and whether this row ends the window."""
    if accumulation < 1 or not 0 <= index < total:
        raise ValueError("invalid accumulation window")
    start = index // accumulation * accumulation
    size = min(accumulation, total - start)
    return (size, index + 1 == start + size)


def row_key(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def adapter_identity(path):
    if path is None:
        return None
    path = Path(path).resolve()
    files = sorted(
        (
            p
            for p in path.iterdir()
            if p.is_file()
            and (
                p.name == "adapter_config.json"
                or p.suffix in (".safetensors", ".bin")
                or p.name.endswith(".index.json")
            )
        )
    )
    if not (path / "adapter_config.json").is_file() or not any(
        (p.suffix in (".safetensors", ".bin") for p in files)
    ):
        raise ValueError(f"incomplete adapter: {path}")
    return {"path": str(path), "files": {p.name: digest(p) for p in files}}
