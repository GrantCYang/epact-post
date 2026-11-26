"""Checkout resources; EVIDENCE_STUDIES_ROOT supports noneditable installs."""

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("EVIDENCE_STUDIES_ROOT", Path(__file__).resolve().parents[2])
).resolve()
EPACT_DATA = ROOT / "data" / "epact"
POST_INPUTS = ROOT / "configs" / "post" / "frozen"
