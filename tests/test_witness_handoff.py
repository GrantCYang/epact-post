import copy
import sys
import unittest
from pathlib import Path

from epact.diagnostics.witness import (
    finish_score,
    policy_prompt,
    summarize,
    validate_twins,
    replay_prefix,
)


class HandoffTest(unittest.TestCase):
    def twins(self):
        action = {
            "action": "Open",
            "args": {"target": "query_location"},
            "ok": True,
            "error": "",
        }
        return [
            {"answer": a, "actual_actions": [copy.deepcopy(action)]}
            for a in ("yes", "no")
        ]

    def test_answer_or_route_cannot_signal_label(self):
        twins = self.twins()
        self.assertEqual(len(validate_twins(twins)), 2)
        twins[1]["actual_actions"][0]["args"]["target"] = "Cup"
        with self.assertRaisesRegex(ValueError, "pair-dependent"):
            validate_twins(twins)
        twins = self.twins()
        for m in twins:
            m["actual_actions"][0]["action"] = "Answer"
        with self.assertRaisesRegex(ValueError, "answer in prefix"):
            validate_twins(twins)

    def test_privileged_fields_and_cost_do_not_enter_policy(self):
        state = {
            "query": "Is a cup in the cabinet?",
            "budget": 20,
            "protocol": "inspection-v4-dev",
            "reference_help": "Public references only.",
        }
        history = self.twins()[0]["actual_actions"]
        original = policy_prompt(state, history, False, True)
        changed = {
            **state,
            "answer": "yes",
            "privileged_determinable": True,
            "privileged_evidence_proxy": {"ready": True},
            "remaining_budget": 1,
        }
        self.assertEqual(
            original, policy_prompt(changed, [{**history[0], "cost": 99}], False, True)
        )
        self.assertNotIn("You can now determine", original)

    def test_harness_answer_never_credited(self):
        answer = {"action": "Answer", "args": {"answer": "no"}, "parse_failed": False}
        self.assertTrue(
            finish_score([answer], {"correct": True}, None, "no")["scored_correct"]
        )
        self.assertFalse(
            finish_score([answer], {"correct": True}, "step_limit", "no")[
                "scored_correct"
            ]
        )
        self.assertFalse(
            finish_score(
                [{**answer, "parse_failed": True}],
                {"correct": True},
                "parse_failure",
                "no",
            )["scored_correct"]
        )
        action = {"action": "LookAt", "args": {}, "parse_failed": False}
        self.assertFalse(
            finish_score([action], {"correct": True}, None, "no")["scored_correct"]
        )

    def test_prefix_mismatch_aborts(self):
        class Link:
            def call(self, **msg):
                if msg["cmd"] == "start":
                    return {"protocol": "inspection-v4-dev", "query": "q"}
                return {
                    "terminated": False,
                    "result": {"ok": False, "error": "changed", "cost": 1},
                }

        with self.assertRaisesRegex(RuntimeError, "outcome changed"):
            replay_prefix(
                Link(),
                {
                    "task_id": "t",
                    "query": "q",
                    "prefix": self.twins()[0]["actual_actions"],
                },
            )

    def test_missing_pair_not_scored_as_complete(self):
        pairs = {"p": {"family": "C", "ids": ["a", "b"]}}
        row = {
            "task_id": "a",
            "first_answer": True,
            "forced_answer": None,
            "decided": [{}],
            "repeated_actions": 0,
            "witness_image_mae": 0,
            **{
                k: True
                for k in (
                    "baseline_correct",
                    "old_witness_correct",
                    "fresh_qa_correct",
                    "first_answer_correct",
                    "scored_correct",
                )
            },
        }
        result = summarize([row], pairs)
        self.assertEqual(result["completed_pairs"], 0)
        self.assertEqual(result["missing_or_incomplete_pairs"], 1)
        self.assertIsNone(result["pair_scores"]["scored_correct"]["accuracy"])
        self.assertIsNone(result["check_macro"]["scored_correct"])

    def test_conditional_pair_requires_both_qa_answers(self):
        pairs = {"p": {"family": "N", "ids": ["a", "b"]}}
        rows = [
            {
                "task_id": tid,
                "first_answer": True,
                "forced_answer": None,
                "decided": [{}],
                "repeated_actions": 0,
                "witness_image_mae": 0,
                **{
                    k: True
                    for k in (
                        "baseline_correct",
                        "old_witness_correct",
                        "first_answer_correct",
                        "scored_correct",
                    )
                },
                "fresh_qa_correct": tid == "a",
            }
            for tid in ("a", "b")
        ]
        result = summarize(rows, pairs)
        self.assertEqual(result["pair_scores"]["scored_correct"]["correct"], 1)
        self.assertEqual(
            result["fresh_qa_correct_pair_subset"]["scored_correct"]["pairs"], 0
        )


if __name__ == "__main__":
    unittest.main()
