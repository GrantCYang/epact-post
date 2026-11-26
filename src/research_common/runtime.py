"""Explicit staged AI2-THOR executable shared by both studies."""

import os
from pathlib import Path

BUILD_NAME = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917"


def require_staged_build() -> Path:
    override = os.environ.get("EPACT_SIMULATOR_PATH")
    releases = Path(
        os.environ.get("EPACT_AI2THOR_RELEASES", "~/.ai2thor/releases")
    ).expanduser()
    cached = (
        Path(os.path.expandvars(override)).expanduser()
        if override
        else releases / BUILD_NAME / BUILD_NAME
    )
    if not cached.is_file():
        raise FileNotFoundError(
            f"staged AI2-THOR release missing: {cached}; refusing implicit download. Set EPACT_SIMULATOR_PATH to the staged CloudRendering executable."
        )
    return cached
