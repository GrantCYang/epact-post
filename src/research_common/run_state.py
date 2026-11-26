"""Immutable run identities and explicit completion receipts."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_identity(directory):
    directory = Path(directory).resolve()
    if not (directory / "config.json").is_file():
        raise ValueError("Model directory must contain config.json")
    files = sorted(
        (
            p
            for p in directory.rglob("*")
            if p.is_file()
            and ".cache" not in p.relative_to(directory).parts
            and (
                p.suffix
                in {
                    ".json",
                    ".safetensors",
                    ".bin",
                    ".model",
                    ".txt",
                    ".jinja",
                    ".tiktoken",
                }
            )
        )
    )
    if not any((p.suffix in {".safetensors", ".bin"} for p in files)):
        raise ValueError("Model directory contains no staged weights")
    return {p.relative_to(directory).as_posix(): sha256(p) for p in files}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare_run(directory, identity):
    """Refuse changed identities and unverified legacy directories before execution."""
    directory = Path(directory)
    manifest = directory / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run configuration differs; choose a new output directory")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Nonempty output has no run identity; choose a new output directory"
        )
    else:
        atomic_json(manifest, identity)


def reusable_episode(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return "ERROR" not in record and record.get("parsed_choice") is not None


def completion(directory, expected_ids):
    expected = set(expected_ids)
    paths = {p.stem: p for p in (Path(directory) / "episodes").glob("*.json")}
    if set(paths) - expected:
        raise ValueError("Output contains episodes outside the frozen task selection")
    completed = sum((reusable_episode(paths[i]) for i in expected if i in paths))
    failed = sum((not reusable_episode(paths[i]) for i in expected if i in paths))
    missing = len(expected - set(paths))
    return {
        "expected_episodes": len(expected),
        "completed_episodes": completed,
        "failed_episodes": failed,
        "missing_episodes": missing,
        "status": "complete" if completed == len(expected) else "incomplete",
    }
