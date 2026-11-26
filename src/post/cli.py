"""POST commands, independent of the EPACT study interface."""

import argparse
from research_common.cli import run_module
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "run": "client",
        "server": "server",
        "report": "report",
        "build": "build",
        "validate-replay": "validate_replay",
    }
    if not args or args[0] in ("-h", "--help"):
        print("Usage: post {run,server,report,build,validate-replay} [options]")
        print("Run command-specific --help for arguments. No model is loaded by help.")
        return 0
    if args[0] not in commands:
        raise SystemExit("Unknown POST command: " + args[0])
    return run_module("post." + commands[args[0]], args[1:])
