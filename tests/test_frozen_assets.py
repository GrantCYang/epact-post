import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from epact.data.frozen import check, restore_part, sha


def part(tmp_path, name="frames/a.png", raw=b"pixels"):
    archive = tmp_path / "part.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("NOTICE.txt", "notice")
        z.writestr(name, raw)
    manifest = {
        "parts": [
            {"sha256": sha(archive), "bytes": archive.stat().st_size, "members": [name]}
        ],
        "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
    }
    return archive, manifest


def test_archive_restores_bound_bytes_only(tmp_path):
    archive, manifest = part(tmp_path)
    destination = tmp_path / "out"
    assert restore_part(archive, destination, manifest) == 1
    assert (destination / "frames/a.png").read_bytes() == b"pixels"
    with pytest.raises(ValueError, match="new output"):
        restore_part(archive, destination, manifest)


def test_archive_traversal_rejected_before_writing(tmp_path):
    archive, manifest = part(tmp_path, "frames/../../escape.png")
    with pytest.raises(ValueError, match="Unsafe"):
        restore_part(archive, tmp_path / "out", manifest)
    assert not (tmp_path / "out").exists()


def test_image_hash_corruption_rejected(tmp_path):
    archive, manifest = part(tmp_path)
    manifest["files"]["frames/a.png"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Image bytes"):
        restore_part(archive, tmp_path / "out", manifest)
    assert not (tmp_path / "out").exists()


def test_bundled_frozen_task_membership():
    root = Path(__file__).resolve().parents[1]
    result = check(root)
    assert result["unique_views"] == 352
    assert [x["pairs"] for x in result["suites"].values()] == [72, 32, 23]
