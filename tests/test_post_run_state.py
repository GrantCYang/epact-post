import importlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from research_common.run_state import (
    prepare_run,
    reusable_episode,
    completion,
    atomic_json,
)


class PostRunStateTests(unittest.TestCase):
    def test_changed_cell_budget_model_or_pairs_cannot_resume(self):
        original = {
            "cell": "checkup_reasoned",
            "k_turns": 12,
            "model": "one",
            "pairs": "one",
        }
        with tempfile.TemporaryDirectory() as directory:
            prepare_run(directory, original)
            prepare_run(directory, original)
            for key, value in [
                ("cell", "checkup_watch_reasoned"),
                ("k_turns", 40),
                ("model", "two"),
                ("pairs", "two"),
            ]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    prepare_run(directory, {**original, key: value})

    def test_legacy_directory_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            atomic_json(Path(directory) / "episodes/one.json", {"cell": "old"})
            with self.assertRaises(ValueError):
                prepare_run(directory, {"cell": "new"})

    def test_failed_and_truncated_records_are_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes/one.json"
            atomic_json(path, {"ERROR": "transport failure", "parsed_choice": None})
            self.assertFalse(reusable_episode(path))
            path.write_text("{", encoding="utf-8")
            self.assertFalse(reusable_episode(path))
            atomic_json(path, {"parsed_choice": {"choice": None}})
            self.assertTrue(
                reusable_episode(path)
            )  # A valid model parse failure is a research outcome.
            self.assertEqual(
                completion(directory, ["one", "two"])["status"], "incomplete"
            )

    def test_all_episode_errors_return_nonzero_and_retry(self):
        client = importlib.import_module("post.client")

        class Loader:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def eval(self):
                return self

        class Link:
            calls = 0

            def __init__(self, port):
                pass

            def call(self, msg):
                Link.calls += 1
                raise RuntimeError("simulated transport failure")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            pair = {
                "pair_id": "p",
                "dim": "mu",
                "candidates": [{"name": "cup", "mass": 1}],
                "members": [
                    {"member": m, "theta": {}, "label_choice": "cup"}
                    for m in ("A", "B")
                ],
            }
            pairs.write_text(json.dumps(pair) + "\n", encoding="utf-8")
            argv = [
                "post",
                "--pairs",
                str(pairs),
                "--port",
                "1",
                "--model-path",
                str(root),
                "--cell",
                "checkup_reasoned",
                "--out",
                str(root / "run"),
            ]
            stubs = {
                "torch": types.SimpleNamespace(bfloat16="stub"),
                "transformers": types.SimpleNamespace(
                    AutoModelForImageTextToText=Loader, AutoProcessor=Loader
                ),
            }
            with (
                patch.dict("sys.modules", stubs),
                patch("sys.argv", argv),
                patch.object(client, "Client", Link),
                patch.object(client, "model_identity", return_value={"config": "same"}),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(client.main(), 1)
                self.assertEqual(client.main(), 1)
            self.assertEqual(Link.calls, 4)
            report = json.loads((root / "run/report.json").read_text())
            self.assertEqual(report["failed_episodes"], 2)
            self.assertEqual(report["completed_episodes"], 0)
