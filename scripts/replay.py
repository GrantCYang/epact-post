"""Replay saved scoring fields and every POST action prefix, then draw figures."""

from pathlib import Path
import os, runpy, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from epact.data.frozen import check
print(check(ROOT))
sys.argv = ["replay"]
runpy.run_module("research_common.results", run_name="__main__")
for name in ("check_information.py", "make_figures.py", "make_summary_figures.py"):
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")
