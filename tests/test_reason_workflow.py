"""Protocol preservation, twin integrity and terminal scoring without model/sim loads."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from epact.evaluation.reason import (
    completed,
    rollout_seed,
    run_episode,
    select_tasks,
    summary,
)
from epact.evaluation.protocol import parse_action_reason
from epact.runtime.server import Handler


def twins():
    base = dict(
        pair_id="pair",
        query="Is it visible?",
        house_index=0,
        start_pose={},
        split="val",
        receptacle_id="box",
    )
    return [
        {**base, "task_id": "yes", "answer": "yes"},
        {**base, "task_id": "no", "answer": "no"},
    ]


class Sim:
    def __init__(self, terminate=True):
        self.actions = []
        self.terminate = terminate

    def call(self, **message):
        if message["cmd"] == "start":
            return {
                "protocol": "inspection-v4-dev",
                "query": "Is it visible?",
                "reference_help": "Public rules",
            }
        self.actions.append(message["action"])
        if message["action"] == "Answer" or self.terminate:
            return {
                "terminated": True,
                "outcome": {
                    "correct": True,
                    "answer": "yes",
                    "status": "answered",
                    "step_log": [],
                },
            }
        return {"terminated": False, "result": {"ok": True, "error": ""}}


class ReasonWorkflowTests(unittest.TestCase):
    def test_twin_selection_rejects_missing_duplicates_and_context_changes(self):
        self.assertEqual(set(select_tasks(twins(), ["pair"])), {"yes", "no"})
        for rows, selected in [
            (twins()[:1], ["pair"]),
            (twins(), ["missing"]),
            (twins(), ["pair", "pair"]),
            (twins() + twins()[:1], ["pair"]),
        ]:
            with self.assertRaises(ValueError):
                select_tasks(rows, selected)
        rows = twins()
        rows[1]["query"] = "Different query"
        with self.assertRaises(ValueError):
            select_tasks(rows, ["pair"])

    def test_last_action_parser_retains_original_reason_protocol(self):
        reply = '{"action":"Open","args":{"target":"x"}}\nI have evidence.\n{"action":"Answer","args":{"answer":"yes"}}'
        self.assertEqual(parse_action_reason(reply), ("Answer", {"answer": "yes"}))
        self.assertIsNone(parse_action_reason("I cannot decide."))

    def test_forced_correct_guess_never_scores_as_success(self):
        sim = Sim()
        row = run_episode(
            sim, lambda *args: "unparseable", twins()[0], 0, 7, "reason", 0.0
        )
        self.assertTrue(row["correct"])
        self.assertFalse(row["scored_correct"])
        self.assertEqual(row["forced_answer"], "parse_failure")

    def test_reason_and_immediate_are_distinct_and_hide_budget(self):
        prompts = []

        def model(state, prompt, seed):
            prompts.append(prompt)
            return '{"action":"Answer","args":{"answer":"yes"}}'

        for mode in ("action", "reason"):
            row = run_episode(Sim(), model, twins()[0], 0, 7, mode, 0.0)
            self.assertTrue(row["scored_correct"])
        self.assertIn("Reply with JSON only", prompts[0])
        self.assertIn("First think briefly", prompts[1])
        self.assertTrue(all("Remaining budget: not disclosed" in p for p in prompts))

    def test_step_limit_retains_query_and_forces_unscored_answer(self):
        sim, prompts = Sim(False), []

        def model(state, prompt, seed):
            prompts.append(prompt)
            return '{"action":"LookAt","args":{"target":"box"}}'

        row = run_episode(sim, model, twins()[0], 0, 7, "reason", 0.0)
        self.assertEqual(len(prompts), 14)
        self.assertTrue(all("Is it visible?" in p for p in prompts))
        self.assertEqual(sim.actions[-1], "Answer")
        self.assertEqual(row["forced_answer"], "step_limit")
        self.assertFalse(row["scored_correct"])

    def test_resume_rejects_truncated_wrong_pair_or_failed_records(self):
        row = run_episode(
            Sim(),
            lambda *args: '{"action":"Answer","args":{"answer":"yes"}}',
            twins()[0],
            0,
            7,
            "reason",
            0.0,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.json"
            path.write_text(json.dumps(row))
            self.assertTrue(completed(path, twins()[0], 0))
            self.assertFalse(completed(path, twins()[1], 0))
            for bad in [
                {**row, "pair_id": "other"},
                {**row, "ERROR": "failed"},
                {**row, "scored_correct": False},
            ]:
                path.write_text(json.dumps(bad))
                self.assertFalse(completed(path, twins()[0], 0))
            path.write_text("{")
            self.assertFalse(completed(path, twins()[0], 0))

    def test_missing_twin_cannot_be_counted_as_complete_pair(self):
        row = run_episode(
            Sim(),
            lambda *args: '{"action":"Answer","args":{"answer":"yes"}}',
            twins()[0],
            0,
            7,
            "reason",
            0.0,
        )
        result = summary([row], {r["task_id"]: r for r in twins()}, 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["correct_pair_rollouts"], 0)

    def test_seed_has_fixed_cross_process_value(self):
        import subprocess, sys

        value = subprocess.check_output(
            [
                sys.executable,
                "-c",
                'from epact.evaluation.reason import rollout_seed; print(rollout_seed("yes",0,7))',
            ],
            text=True,
        )
        self.assertEqual(rollout_seed("yes", 0, 7), int(value))
        self.assertNotEqual(rollout_seed("yes", 0, 7), rollout_seed("yes", 1, 7))

    def test_server_exposes_bound_input_identity(self):
        from types import SimpleNamespace

        server = SimpleNamespace(
            protocol="inspection-v4-dev",
            input_identity={"tasks_sha256": "tasks", "dataset_sha256": "houses"},
        )
        reply = Handler.dispatch(None, server, {"cmd": "describe"})
        self.assertEqual(reply["tasks_sha256"], "tasks")
        self.assertEqual(reply["dataset_sha256"], "houses")
