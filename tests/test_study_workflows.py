"""Reader-facing evidence selection and frozen evaluation recipe contracts."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research_common.evidence import catalog, replay, display
from research_common.paths import ROOT
from epact.evaluation.recipe import recipes, plan


class StudyWorkflowTests(unittest.TestCase):
    def test_catalog_covers_every_included_dataset_once(self):
        manifest = json.loads((ROOT / "data/records/manifest.json").read_text())
        covered = {}
        for study in catalog().values():
            for field in (
                "input_manifest",
                "input_records",
                "configuration",
                "documentation",
                "paper",
            ):
                if field in study:
                    self.assertTrue((ROOT / study[field]).is_file(), study[field])
            for key, count in study.get("datasets", {}).items():
                self.assertNotIn(key, covered)
                covered[key] = count
        self.assertEqual(covered, manifest["datasets"])

    def test_replay_separates_studies_and_names_conditions(self):
        epact = replay("epact")
        post = replay("post")
        self.assertNotIn("post", epact)
        self.assertNotIn("target_combined", post)
        self.assertEqual(epact["target_combined"]["a0"]["pair_success"], 19)
        self.assertIn("No-Open SFT, seed 2", display(epact))
        self.assertIn("Watched checkup, reasoned | 9/12", display(post))
        self.assertFalse(post["new_model_inference"])

    def test_recipe_plan_retains_protocol_and_separates_segments(self):
        a = plan(
            "target-a0-sega",
            Path("archive"),
            Path("val.jsonl.gz"),
            Path("model"),
            Path("output"),
        )
        b = plan(
            "target-a0-segb",
            Path("archive"),
            Path("val.jsonl.gz"),
            Path("model"),
            Path("output-b"),
        )
        self.assertFalse(a["executed"])
        self.assertEqual((a["pairs"], b["pairs"]), (32, 23))
        self.assertNotEqual(a["tasks_sha256"], b["tasks_sha256"])
        self.assertIn("--adapter-path", a["evaluate_argv"])
        base = plan(
            "step0-base-reason",
            Path("archive"),
            Path("train.jsonl.gz"),
            Path("model"),
            Path("out"),
        )
        self.assertNotIn("--adapter-path", base["evaluate_argv"])
        args = base["evaluate_argv"]
        self.assertEqual(args[args.index("--temperature") + 1], "1.0")
        self.assertEqual(args[args.index("--max-new-tokens") + 1], "256")

    def test_all_recipe_pair_files_have_expected_hash_and_pair_count(self):
        self.assertEqual(len(recipes()), 19)
        for name, recipe in recipes().items():
            payload = (ROOT / recipe["pair_ids"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), recipe["pair_ids_sha256"], name
            )
            pairs = json.loads(payload)
            self.assertEqual(len(pairs), recipe["pairs"])
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_check_inputs_rejects_changed_tasks_before_any_model_loading(self):
        recipe = recipes()["target-base-sega"]
        with TemporaryDirectory() as directory:
            task = Path(directory) / recipe["tasks"]
            task.parent.mkdir(parents=True)
            task.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "task identity"):
                plan(
                    "target-base-sega",
                    directory,
                    "val",
                    "model",
                    "output",
                    check_inputs=True,
                )
