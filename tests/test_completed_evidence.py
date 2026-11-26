"""Checks for paired scoring and the independently stored research tables."""

import sys
import unittest
from pathlib import Path

from research_common.results import (
    paired,
    opened,
    episode_summary,
    recompute,
    verify_report,
)


class CompletedEvidenceTests(unittest.TestCase):
    def test_frozen_report_matches_episode_records(self):
        self.assertGreaterEqual(verify_report(recompute()), 119)

    def test_pair_cannot_pass_with_missing_or_duplicate_member(self):
        a = {"pair_id": "p", "gold": "yes", "correct": True}
        with self.assertRaises(ValueError):
            paired([a])
        with self.assertRaises(ValueError):
            paired([a, a])

    def test_open_requires_success_on_queried_object(self):
        row = {
            "receptacle_id": "fridge",
            "open_steps": [
                {
                    "action": "Open",
                    "args": {"target": "fridge"},
                    "ok": True,
                    "native_steps": [
                        {
                            "action": "OpenObject",
                            "objectId": "cupboard",
                            "success": True,
                        }
                    ],
                }
            ],
        }
        self.assertFalse(opened(row))
        row["open_steps"][0]["native_steps"][0]["objectId"] = "fridge"
        self.assertTrue(opened(row))

    def test_forced_correct_guess_does_not_make_pair_success(self):
        a = {
            "pair_id": "p",
            "gold": "yes",
            "correct": True,
            "scored_correct": True,
            "forced_answer": None,
            "status": "answered",
            "answer": "yes",
            "receptacle_id": "x",
            "open_steps": [],
        }
        b = {
            **a,
            "gold": "no",
            "answer": "no",
            "forced_answer": "step_limit",
            "scored_correct": False,
        }
        self.assertEqual(episode_summary([a, b])["pair_success"], 0)
        self.assertEqual(episode_summary([a, b])["answer_counts"]["no"], 1)
