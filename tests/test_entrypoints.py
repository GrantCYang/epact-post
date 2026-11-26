"""Installed entry points must remain usable from outside the checkout."""

import subprocess
import sys
import tempfile
import unittest


class EntrypointTests(unittest.TestCase):
    def test_documented_commands_have_cpu_only_help(self):
        commands = [
            ("epact", name)
            for name in (
                "train",
                "evaluate",
                "data-check",
                "results",
                "server",
                "serve",
                "train-rows",
                "evaluate-reason",
            )
        ]
        commands += [
            ("epact", "data", name)
            for name in (
                "generate",
                "pool",
                "audit",
                "witnesses",
                "select",
                "demonstrations",
            )
        ]
        commands += [
            ("epact", "diagnose", name)
            for name in ("pilot", "readability", "witness", "procedure", "report")
        ]
        commands += [
            ("epact", "original", name)
            for name in ("data-check", "results", "train", "serve", "evaluate")
        ]
        commands += [
            ("epact", "policy", name) for name in ("serve", "evaluate", "train", "plan")
        ]
        commands += [
            ("research_common.evidence", name) for name in ("list", "show", "replay")
        ]
        commands += [
            ("post", name)
            for name in ("run", "server", "build", "report", "validate-replay")
        ]
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        [sys.executable, "-X", "utf8", "-m", *command, "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout.lower())
