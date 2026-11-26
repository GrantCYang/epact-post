import json
from pathlib import Path
import sys
import tempfile
import unittest

from epact.training.support import accumulation_window, adapter_identity, freeze
from epact.evaluation.behavior import behavior


class TailFixTests(unittest.TestCase):
    def test_all_rows_updated_for_actual_arm_sizes(self):
        for total, steps, tail in [(174, 22, 6), (724, 91, 4), (1258, 158, 2)]:
            pending = []
            applied = []
            sizes = []
            for i in range(total):
                size, end = accumulation_window(i, total, 8)
                pending.append(i)
                if end:
                    self.assertEqual(len(pending), size)
                    sizes.append(size)
                    applied.extend(pending)
                    pending = []
            self.assertEqual(applied, list(range(total)))
            self.assertEqual(pending, [])
            self.assertEqual(len(sizes), steps)
            self.assertEqual(sizes[-1], tail)

    def test_tail_gradient_is_actual_mean(self):
        values = list(range(1, 11))
        acc = 0
        updates = []
        for i, value in enumerate(values):
            size, end = accumulation_window(i, len(values), 8)
            acc += value / size
            if end:
                updates.append(acc)
                acc = 0
        self.assertEqual(updates, [4.5, 9.5])

    def test_single_short_and_exact_windows(self):
        self.assertEqual(accumulation_window(0, 1, 8), (1, True))
        self.assertEqual(accumulation_window(7, 16, 8), (8, True))
        self.assertEqual(accumulation_window(15, 16, 8), (8, True))
        with self.assertRaises(ValueError):
            accumulation_window(0, 1, 0)

    def test_repeat_count_comes_from_decisions(self):
        actions = [{"action": "LookAt", "args": {"target": "query_location"}}] * 3
        actions += [{"action": "Answer", "args": {"answer": "yes"}}]
        result = behavior(actions)
        self.assertEqual(result["repeated_actions"], 2)
        self.assertFalse(result["no_action"])
        self.assertEqual(result["action_counts"]["LookAt"], 3)

    def test_adapter_change_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights-one")
            before = adapter_identity(adapter)
            freeze(root / "run.json", before)
            (adapter / "adapter_model.safetensors").write_bytes(b"weights-two")
            with self.assertRaises(ValueError):
                freeze(root / "run.json", adapter_identity(adapter))

    def test_missing_adapter_weights_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                adapter_identity(root)


if __name__ == "__main__":
    unittest.main()
