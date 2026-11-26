"""S2 baseline policies (plan §6): random floor, frontier-explore, oracle.

Scripted controls use the Episode's privileged helpers deliberately — they
calibrate the instrument. The procedure wrapper (frozen model + memory) is a
model policy and lives with probe A.
"""

from __future__ import annotations
import random
from epact.environment.episode import Episode
from epact.environment.generator import type_name


class OraclePolicy:
    """Ceiling: replay the stored plan, look until determinable, answer GT.

    Plans end with navigation; the camera does not necessarily hold the
    evidence in frame at answer time (v1.2 appended LookAt only for C
    kinds). To keep every oracle answer evidence-grounded — pilot audits
    require determinable-at-answer — the policy adds up to two gaze
    corrections (target, then receptacle) before answering.
    """

    def reset(self, episode: Episode) -> None:
        self.plan = list(episode.task["oracle_plan"])
        t = episode.task
        self.gaze = [
            x for x in (t.get("receptacle_id"), t.get("target_object_id")) if x
        ]

    def decide(self, episode: Episode):
        if episode.determinable():
            return ("Answer", {"answer": episode.task["answer"], "confidence": 1.0})
        if self.plan:
            step = self.plan.pop(0)
            return (step["action"], {"target": step["target"]})
        if self.gaze:
            return ("LookAt", {"target": self.gaze.pop(0)})
        return ("Answer", {"answer": episode.task["answer"], "confidence": 1.0})


class RandomPolicy:
    """Floor: uniform over currently valid-looking actions until the budget
    frays, then a coin-flip answer."""

    def __init__(self, seed: int = 20260908) -> None:
        self.seed = seed

    def reset(self, episode: Episode) -> None:
        self.rng = random.Random(f"{self.seed}:{episode.task['task_id']}")

    def decide(self, episode: Episode):
        if episode.view().remaining_budget < 1.5:
            return (
                "Answer",
                {"answer": self.rng.choice(("yes", "no")), "confidence": 0.5},
            )
        objs = episode.sim.objects()
        visible = episode.visible_ids()
        candidates = []
        for o in objs:
            oid = o["objectId"]
            if o["openable"] and oid in visible and (not oid.startswith("door")):
                candidates.append(("Close" if o["isOpen"] else "Open", oid))
            if o["toggleable"] and oid in visible:
                candidates.append(("Toggle", oid))
            if o["receptacle"] or o["openable"] or o["toggleable"]:
                candidates.append(("GoTo", oid))
        candidates.extend((("GoTo", r["id"]) for r in episode.sim.rooms()))
        candidates.sort()
        action, target = self.rng.choice(candidates)
        return (action, {"target": target})


class FrontierExplorePolicy:
    """Strong non-semantic baseline: room by room, light the dark, open every
    closed openable near the path, answer the moment the answer is readable.
    Uses privileged determinability — that is its role as a control."""

    def reset(self, episode: Episode) -> None:
        task = episode.task
        self.queue: list[tuple[str, str]] = []
        rooms = sorted((r["id"] for r in episode.sim.rooms()))
        start_room = task.get("start_room")
        ordered = ([start_room] if start_room in rooms else []) + [
            r for r in rooms if r != start_room
        ]
        objs = {o["objectId"]: o for o in episode.sim.objects()}
        lamps = [
            oid
            for oid, o in sorted(objs.items())
            if o["toggleable"] and "Lamp" in type_name(o)
        ]
        for lamp in lamps:
            self.queue.append(("GoTo", lamp))
            self.queue.append(("Toggle", lamp))
        mod = task.get("house_mod") or {}
        for door_id in mod.get("closed_doors") or []:
            self.queue.append(("GoTo", door_id))
            self.queue.append(("Open", door_id))
        for room in ordered:
            self.queue.append(("GoTo", room))
            for oid, o in sorted(objs.items()):
                if (
                    o["openable"]
                    and (not o["isOpen"])
                    and (not oid.startswith("door"))
                    and (
                        episode.sim.room_of_point(
                            o["position"]["x"], o["position"]["z"]
                        )
                        == room
                    )
                ):
                    self.queue.append(("GoTo", oid))
                    self.queue.append(("Open", oid))

    def decide(self, episode: Episode):
        if episode.determinable():
            return ("Answer", {"answer": episode.task["answer"], "confidence": 1.0})
        if episode.view().remaining_budget < 1.5 or not self.queue:
            return ("Answer", {"answer": "no", "confidence": 0.5})
        action, target = self.queue.pop(0)
        return (action, {"target": target})
