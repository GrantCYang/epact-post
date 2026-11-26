"""Episode protocol (plan §4) over one task: budget, invalid-action rule,
darkness transform, termination, scoring, and §11-style logging.

Policies receive an EpisodeView (frame, query, history, remaining budget).
Scripted baselines may additionally use the privileged helpers on Episode —
they are controls, not contestants; models never see metadata.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
from epact.environment.actions import ActionAdapter
from epact.environment.generator import darken
from epact.environment.sim import EpactSim, frame_hash, load_house


def prune_objects(objects: list, drop: set) -> list:
    """Remove objects (and their subtrees) by id from a house-JSON object list.

    Used by the v3 paired design: a task's twin is the identical scene with
    the target object deleted, so the two members share a byte-identical
    start observation and query string and differ only in the hidden answer.
    Deleting from the house JSON is pure data — no physics, unlike placing.
    """
    kept = []
    for obj in objects:
        if obj.get("id") in drop:
            continue
        if obj.get("children"):
            obj = {**obj, "children": prune_objects(obj["children"], drop)}
        kept.append(obj)
    return kept


@dataclass
class EpisodeView:
    query: str
    frame: np.ndarray
    remaining_budget: float
    history: list = field(default_factory=list)


class Episode:
    def __init__(
        self, sim: EpactSim, task: dict, dataset_path, save_frames_dir=None
    ) -> None:
        self.sim = sim
        self.task = task
        self.dataset_path = dataset_path
        self.adapter = ActionAdapter(sim)
        if task.get("query_type") == "STATE":
            import re as _re

            tid = task.get("target_object_id")
            if tid:
                self.adapter.referenceable.add(tid)
            ttype = task.get("target_type")
            if ttype:
                self.adapter.referenceable.add(_re.sub("[^a-z0-9]", "", ttype.lower()))
        self.cost = 0.0
        self.invalid_count = 0
        self.steps: list[dict] = []
        self.outcome: dict | None = None
        self.save_frames_dir = save_frames_dir

    def setup(self, house_cache: dict | None = None) -> None:
        key = (self.task["split"], self.task["house_index"])
        if house_cache is not None and key in house_cache:
            house = house_cache[key]
        else:
            house = load_house(self.dataset_path, self.task["house_index"])
            if house_cache is not None:
                house_cache[key] = house
        mod = self.task.get("house_mod")
        if mod and (
            mod.get("removed_lights")
            or mod.get("closed_doors")
            or mod.get("removed_objects")
        ):
            house = copy.deepcopy(house)
            if mod.get("removed_lights"):
                house["proceduralParameters"]["lights"] = [
                    l
                    for l in house["proceduralParameters"]["lights"]
                    if l["id"] not in mod["removed_lights"]
                ]
            for door in house.get("doors", []):
                if door["id"] in (mod.get("closed_doors") or []):
                    door["openness"] = 0
            if mod.get("removed_objects"):
                house["objects"] = prune_objects(
                    house.get("objects", []), set(mod["removed_objects"])
                )
        if self.sim.controller is None:
            self.sim.start(house)
        else:
            self.sim.reset(house)
        for prep in (mod or {}).get("prep", []):
            prepared = self.sim.controller.step(**prep)
            if not prepared.metadata.get("lastActionSuccess"):
                raise RuntimeError(
                    "scene preparation failed: "
                    + str(prep.get("action"))
                    + ": "
                    + str(prepared.metadata.get("errorMessage", ""))[:200]
                )
        pose = self.task["start_pose"]
        event = self.sim.controller.step(
            action="Teleport",
            position=pose["position"],
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose.get("horizon", 0.0),
            standing=True,
        )
        if not event.metadata["lastActionSuccess"]:
            raise RuntimeError(
                f"start teleport failed for {self.task['task_id']}: {event.metadata.get('errorMessage', '')[:80]}"
            )

    def darkness_active(self) -> bool:
        mod = self.task.get("house_mod") or {}
        darkness = mod.get("darkness")
        if not darkness:
            return False
        for lamp_id in darkness.get("lit_when_any_on", []):
            lamp = self.sim.get_object(lamp_id)
            if lamp is not None and lamp["isToggled"]:
                return False
        agent = self.sim.agent_pose()["position"]
        return self.sim.room_of_point(agent["x"], agent["z"]) == mod.get("dark_room")

    def frame(self) -> np.ndarray:
        raw = self.sim.frame
        return darken(raw) if self.darkness_active() else raw

    def visible_ids(self) -> set:
        return {o["objectId"] for o in self.sim.objects() if o["visible"]}

    def determinable(self) -> bool:
        t = self.task
        condition = t["reveal_condition"]
        visible = self.visible_ids()
        if condition == "already_visible":
            return True
        if condition == "target_visible":
            return t.get("target_object_id") in visible
        if condition == "receptacle_open_visible":
            rec = self.sim.get_object(t["receptacle_id"])
            return rec is not None and rec["isOpen"] and rec["visible"]
        if condition == "receptacle_visible":
            rec = self.sim.get_object(t["receptacle_id"])
            return rec is not None and rec["visible"]
        if condition == "lit_and_visible":
            rec = self.sim.get_object(t["receptacle_id"])
            return not self.darkness_active() and rec is not None and rec["visible"]
        if condition == "occluder_moved":
            occ = self.sim.get_object(t["occluder_id"])
            rec = self.sim.get_object(t["receptacle_id"])
            if occ is None or rec is None:
                return False
            orig = t["occluder_position"]
            moved = (
                abs(occ["position"]["x"] - orig["x"]) > 0.3
                or abs(occ["position"]["z"] - orig["z"]) > 0.3
            )
            return moved and rec["visible"]
        raise ValueError(f"unknown reveal_condition {condition!r}")

    def view(self) -> EpisodeView:
        return EpisodeView(
            query=self.task["query"],
            frame=self.frame(),
            remaining_budget=round(self.task["budget"] - self.cost, 3),
            history=[
                {k: s[k] for k in ("action", "args", "ok", "error", "cost")}
                for s in self.steps
            ],
        )

    def act(self, name: str, **args) -> dict:
        if not self.outcome is None:
            raise ValueError("episode already terminated")
        if name == "Answer":
            answer = str(args.get("answer", "")).lower()
            self.steps.append(
                {
                    "action": "Answer",
                    "args": {
                        "answer": answer,
                        "confidence": float(args.get("confidence", 0.5)),
                    },
                    "ok": True,
                    "error": "",
                    "determinable": self.determinable(),
                }
            )
            self._finish(
                status="answered",
                answer=answer,
                confidence=float(args.get("confidence", 0.5)),
                correct=answer == self.task["answer"],
            )
            return {"terminated": True}
        if name in ("Open", "Close") and isinstance(args.get("target"), str):
            resolved = self.adapter.resolve_target(args["target"])
            allowed = (self.task.get("house_mod") or {}).get("closed_doors") or []
            if resolved.startswith("door") and resolved not in allowed:
                from epact.environment.actions import ActionResult

                refuse = ActionResult(
                    name,
                    dict(args),
                    ok=False,
                    error="precondition: that door is already open",
                )
                record = refuse.to_json()
                record["frame_sha256"] = frame_hash(self.frame())
                record["determinable"] = self.determinable()
                self.steps.append(record)
                self.invalid_count += 1
                if self.invalid_count >= 2:
                    self._finish(status="invalid_action", answer=None, correct=False)
                    return {"terminated": True}
                return {"terminated": False, "result": record}
        if (
            self.task.get("query_type") == "STATE"
            and name in ("Toggle", "Open", "Close", "MoveAside")
            and isinstance(args.get("target"), str)
        ):
            resolved = self.adapter.resolve_target(args["target"])
            if resolved == self.task.get("target_object_id"):
                from epact.environment.actions import ActionResult

                refuse = ActionResult(
                    name,
                    dict(args),
                    ok=False,
                    error="precondition: the queried object may be inspected but not altered",
                )
                record = refuse.to_json()
                record["frame_sha256"] = frame_hash(self.frame())
                record["determinable"] = self.determinable()
                self.steps.append(record)
                self.invalid_count += 1
                if self.invalid_count >= 2:
                    self._finish(status="invalid_action", answer=None, correct=False)
                    return {"terminated": True}
                return {"terminated": False, "result": record}
        result = self.adapter.execute(name, **args)
        record = result.to_json()
        frame = self.frame()
        record["frame_sha256"] = frame_hash(frame)
        if self.save_frames_dir is not None:
            from PIL import Image as _Image

            path = self.save_frames_dir / f"{record['frame_sha256']}.png"
            if not path.exists():
                _Image.fromarray(frame).save(path)
        record["determinable"] = self.determinable()
        self.steps.append(record)
        if result.ok:
            self.cost += result.cost
            if self.cost > self.task["budget"]:
                self._finish(status="no_termination", answer=None, correct=False)
                return {"terminated": True}
        else:
            self.invalid_count += 1
            if self.invalid_count >= 2:
                self._finish(status="invalid_action", answer=None, correct=False)
                return {"terminated": True}
        return {"terminated": False, "result": record}

    def _finish(self, **outcome) -> None:
        self.outcome = {
            "task_id": self.task["task_id"],
            "trap": self.task.get("trap"),
            "family": f"{self.task['query_type']}|{'+'.join(self.task['hiding_kinds'])}|d{self.task['depth']}",
            "cost": round(self.cost, 3),
            "oracle_cost": self.task["oracle_cost"],
            "budget": self.task["budget"],
            "steps": len([s for s in self.steps if s.get("action") != "Answer"]),
            "lure_taken": self._lure_taken(),
            **outcome,
        }

    def _lure_taken(self) -> bool | None:
        lure = self.task.get("trap_lure")
        if not lure:
            return None
        return any(
            (
                s["action"] == lure["action"]
                and s["args"].get("target") == lure["target"]
                and s["ok"]
                for s in self.steps
            )
        )


def run_episode(episode: Episode, policy, max_steps: int = 40) -> dict:
    episode.setup(getattr(policy, "house_cache", None))
    policy.reset(episode)
    for _ in range(max_steps):
        name, args = policy.decide(episode)
        out = episode.act(name, **args)
        if out["terminated"]:
            break
    if episode.outcome is None:
        episode.act("Answer", answer="no", confidence=0.0)
    return {**episode.outcome, "step_log": episode.steps}
