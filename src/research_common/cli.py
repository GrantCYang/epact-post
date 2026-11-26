"""Invoke a command module without changing working directory or losing exit status."""

import importlib
import sys


def run_module(name, argv):
    module = importlib.import_module(name)
    saved = sys.argv
    try:
        sys.argv = [name, *map(str, argv)]
        return module.main() or 0
    finally:
        sys.argv = saved
