import sys
import unittest
from pathlib import Path

from epact.evaluation.protocol import policy_prompt
from epact.runtime.client import advance_state


class InspectionPromptDispatch(unittest.TestCase):
    def test_current_protocol_delivers_public_reference_rules_across_steps(self):
        state = {
            "protocol": "inspection-v4-dev",
            "reference_help": "CHECK targets use query_location; queried items cannot be referenced.",
            "query": "Is there a cup in the cabinet?",
            "budget": 20,
        }
        for actual in [
            state,
            advance_state(state, {"remaining_budget": 19, "frame_png_b64": "frame"}),
        ]:
            text = policy_prompt(actual, [], hide_budget=True)
            self.assertIn(state["reference_help"], text)
            self.assertIn('"target": "query_location"', text)
            self.assertIn("stance: stand/crouch", text)
            self.assertNotIn("Fridge|2|1", text)
            self.assertIn("not disclosed", text)
            self.assertNotIn("Remaining budget: 19", text)


if __name__ == "__main__":
    unittest.main()
