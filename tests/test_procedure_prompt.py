import copy
import sys
import unittest
from pathlib import Path

from epact.diagnostics.procedure import (
    PROCEDURE,
    procedure_prompt,
    policy_prompt,
    initial_state,
    decision_order,
    cell_summary,
    action_stats,
    finish_score,
)


class PromptTest(unittest.TestCase):
    def state(self):
        return {
            "query": "Is a cup in the cabinet?",
            "protocol": "inspection-v4-dev",
            "reference_help": "Public reference instructions.",
            "budget": 20,
        }

    def test_only_fixed_public_text_is_added(self):
        state = self.state()
        history = []
        self.assertEqual(
            procedure_prompt(state, history),
            policy_prompt(state, history, False, True) + PROCEDURE,
        )
        other = {
            **state,
            "answer": "yes",
            "privileged_determinable": True,
            "privileged_evidence_proxy": {"ready": True},
            "remaining_budget": 0,
            "oracle_plan": ["Answer"],
        }
        self.assertEqual(
            procedure_prompt(state, history), procedure_prompt(other, history)
        )
        self.assertNotIn("You can now determine", procedure_prompt(state, history))

    def test_prompt_does_not_block_repeated_history(self):
        history = [
            {
                "action": "LookAt",
                "args": {"target": "query_location"},
                "ok": True,
                "error": "",
            }
        ] * 10
        before = copy.deepcopy(history)
        prompt = procedure_prompt(self.state(), history)
        self.assertEqual(history, before)
        self.assertEqual(prompt.count("LookAt({'target': 'query_location'})"), 6)

    def test_both_start_conditions_use_same_server_protocol(self):
        outer = self

        class Link:
            def __init__(self):
                self.calls = []

            def call(self, **kw):
                self.calls.append(kw)
                if kw["cmd"] == "start":
                    return outer.state()
                return {
                    "terminated": False,
                    "result": {"ok": True, "error": "", "cost": 1},
                }

        member = {
            "task_id": "t",
            "query": self.state()["query"],
            "prefix": [
                {
                    "action": "LookAt",
                    "args": {"target": "query_location"},
                    "ok": True,
                    "error": "",
                }
            ],
        }
        link = Link()
        _, history = initial_state(link, member, "start")
        self.assertEqual(len(link.calls), 1)
        self.assertEqual(history, [])
        link = Link()
        _, history = initial_state(link, member, "handoff")
        self.assertEqual(len(link.calls), 2)
        self.assertEqual(len(history), 1)
        self.assertEqual(len(member["prefix"]), 1)

    def test_order_is_paired_and_interleaved(self):
        plan = {"pairs": {"p": {"ids": ["yes", "no"]}, "q": {"ids": ["yes2", "no2"]}}}
        self.assertEqual(
            decision_order(plan)[:4],
            [("start", "yes"), ("start", "no"), ("handoff", "yes"), ("handoff", "no")],
        )
        self.assertEqual(len(set(decision_order(plan))), 8)

    def test_repeat_definition_and_partial_pair_denominator(self):
        action = {"action": "LookAt", "args": {"target": "query_location"}}
        stats = action_stats(
            [action, action, {"action": "Answer", "args": {"answer": "yes"}}]
        )
        self.assertEqual(stats["repeated_actions"], 1)
        row = {
            "task_id": "a",
            "scored_correct": True,
            "first_answer": False,
            "forced_answer": None,
            "decided": [action, action],
            **stats,
        }
        summary = cell_summary([row], {"p": {"ids": ["a", "b"], "family": "C"}})
        self.assertEqual(summary["completed_pairs"], 0)
        self.assertIsNone(summary["pair_accuracy"])
        self.assertIsNone(summary["check_macro"])

    def test_forced_and_non_answer_termination_not_credited(self):
        d = {"action": "Answer", "args": {"answer": "no"}, "parse_failed": False}
        self.assertFalse(
            finish_score([d], {"correct": True}, "step_limit", "no")["scored_correct"]
        )
        self.assertTrue(
            finish_score([d], {"correct": True}, None, "no")["scored_correct"]
        )


if __name__ == "__main__":
    unittest.main()
