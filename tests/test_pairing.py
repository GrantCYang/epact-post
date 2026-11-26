import unittest
from copy import deepcopy
from epact.environment.pairing import construct_pair


class PairingTests(unittest.TestCase):
    def seed(self, kind):
        return {
            "task_id": "v-L-CHECK-0",
            "query_type": "CHECK",
            "query": "Is there a Cup on the Desk?",
            "hiding_kinds": kind,
            "answer": "yes",
            "target_object_id": "cup",
            "target_type": "Cup",
            "receptacle_id": "desk",
            "visible_from_start": ["cup", "desk"],
            "house_mod": {"removed_lights": ["light"], "darkness": {"scale": 0.04}},
            "oracle_plan": [{"action": "LookAt", "target": "cup"}],
        }

    def objects(self):
        return [
            {"objectId": "cup", "objectType": "Cup", "parentReceptacles": ["desk"]},
            {"objectId": "cup2", "objectType": "Cup", "parentReceptacles": ["desk"]},
            {
                "objectId": "cup_elsewhere",
                "objectType": "Cup",
                "parentReceptacles": ["table"],
            },
            {"objectId": "desk", "objectType": "Desk", "openable": False},
        ]

    def test_light_and_composition_preserve_branch_setup_and_remove_all_in_region(self):
        for kinds in (["L"], ["L", "D"], ["L", "C"], ["D", "door"]):
            t = self.seed(kinds)
            before = deepcopy(t)
            pair, info = construct_pair(t, self.objects())
            self.assertEqual(info["status"], "constructed")
            self.assertEqual(t, before)
            self.assertEqual(pair[1]["house_mod"]["removed_objects"], ["cup", "cup2"])
            self.assertEqual(pair[1]["house_mod"]["removed_lights"], ["light"])
            self.assertEqual(pair[0]["query"], pair[1]["query"])
            self.assertEqual(pair[1]["oracle_plan"], [])

    def test_visible_controls_are_not_rejected_as_hidden_start_leaks(self):
        pair, _ = construct_pair(self.seed(["N"]), self.objects())
        self.assertEqual(pair[1]["benchmark_condition"], "already_answerable")

    def test_incorrect_positive_relation_is_invalid(self):
        objects = self.objects()
        objects[0]["parentReceptacles"] = ["elsewhere"]
        pair, info = construct_pair(self.seed(["C"]), objects)
        self.assertEqual(pair, [])
        self.assertEqual(info["status"], "invalid")

    def test_state_flip_preserves_object_and_prep(self):
        t = {
            "task_id": "s",
            "query_type": "STATE",
            "query": "Is the Fridge open?",
            "hiding_kinds": ["D"],
            "target_object_id": "fridge",
            "answer": "no",
            "state_field": "isOpen",
            "state_value": False,
        }
        pair, _ = construct_pair(
            t, [{"objectId": "fridge", "objectType": "Fridge", "isOpen": False}]
        )
        self.assertEqual(pair[1]["target_object_id"], "fridge")
        self.assertEqual(pair[1]["house_mod"]["prep"][0]["action"], "OpenObject")
        self.assertEqual(pair[1]["answer"], "yes")


if __name__ == "__main__":
    unittest.main()
