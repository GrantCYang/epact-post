"""Resolve packaged inputs without depending on the original checkout location."""

import os
from pathlib import Path
from research_common.paths import ROOT


def repo_path(value, data_root=None):
    """Manifest paths are relative to the restored attachment root."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(data_root or ROOT / "data/epact") / path


def house_dataset(value=None, split="val"):
    """An explicit CLI path wins over the environment and bundled-assets default."""
    value = value or os.environ.get("EPACT_HOUSE_DATASET")
    return (
        Path(os.path.expandvars(str(value))).expanduser()
        if value
        else ROOT / "data" / "external" / "procthor-10k" / f"{split}.jsonl.gz"
    )


def resolve_observations(capture, data_root=None):
    for views in capture.get("observations", {}).values():
        for view in views.values():
            if isinstance(view, dict) and view.get("image"):
                view["image"] = str(repo_path(view["image"], data_root))
    return capture
