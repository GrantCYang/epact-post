"""EPACT task generator (plan §3).

Per house: choose start poses, take a 4-frame panorama to fix the
"visible from start" set (union of metadata `visible` over rotations, wide
visibilityDistance), enumerate candidate tasks from metadata, then verify
reveals with amortized simulator work (one interact-pose + open check per
unique receptacle, shared by every task that uses it).

v0 scope: query types CHECK + STATE (FIND deferred, Grant §12.3 default);
hiding kinds N, C, D, C+D (depth ≤ 2, Grant §12.5 default), L in a second
pass (house-JSON light removal); O deferred. Traps: T-visible, T-open-empty,
T-wrong-container (T-lit and T-dark-irrelevant land with kind L).

Every task stores its full spec, seed, oracle plan, oracle cost, and frame
hashes so it regenerates and scores without re-running generation.
"""

from __future__ import annotations
import copy as _copy
import random
from collections import Counter
from pathlib import Path
from epact.environment.actions import REACH_M
from epact.environment.sim import EpactSim, dist_xz, frame_hash, polyline_length

GEN_VERSION = "epact-gen-v4-dev"
PANORAMA_ROTATIONS = (0, 90, 180, 270)
PSEUDO_TYPES = {"Floor", "Wall", "Doorway", "Window", "Painting", "Room"}
DOOR_PREFIX = "door"
DEFAULT_QUOTAS = {"N": 2, "C": 4, "D": 2, "C+D": 2, "STATE": 2, "TRAP": 3}


def type_name(obj: dict) -> str:
    return obj["objectType"]


class HouseGenerator:
    def __init__(
        self,
        sim: EpactSim,
        split: str,
        house_index: int,
        seed: int,
        frames_dir: Path,
        quotas: dict | None = None,
    ) -> None:
        self.sim = sim
        self.split = split
        self.house_index = house_index
        self.rng = random.Random(f"{seed}:{split}:{house_index}")
        self.frames_dir = frames_dir
        self.quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
        self.remaining = Counter(
            {
                "C_yes": self.quotas["C"] - self.quotas["C"] // 2,
                "C_no": self.quotas["C"] // 2,
                "C+D": self.quotas["C+D"],
                "D": self.quotas["D"],
                "N": self.quotas["N"],
                "STATE": self.quotas["STATE"],
                "TRAP": self.quotas["TRAP"],
            }
        )
        self.exam_cap = 15
        self.tasks: list[dict] = []
        self.counter = Counter()
        self.answers = Counter()
        self._reveal_cache: dict[str, dict] = {}
        self._type_counts = Counter(
            (type_name(o) for o in sim.objects() if type_name(o) not in PSEUDO_TYPES)
        )
        self._room_types = {r["id"]: r.get("roomType", "room") for r in sim.rooms()}

    def _room_of(self, obj: dict) -> str | None:
        return self.sim.room_of_point(obj["position"]["x"], obj["position"]["z"])

    def display_name(self, obj: dict) -> str | None:
        """Unambiguous natural name, or None if the house makes it ambiguous."""
        t = type_name(obj)
        if self._type_counts[t] == 1:
            return f"the {t}"
        room = self._room_of(obj)
        if room is None:
            return None
        same_type_same_room = [
            o
            for o in self.sim.objects()
            if type_name(o) == t and self._room_of(o) == room
        ]
        if len(same_type_same_room) == 1:
            return f"the {t} in the {self._room_types.get(room, 'room')}"
        xs = [o["position"]["x"] for o in same_type_same_room]
        zs = [o["position"]["z"] for o in same_type_same_room]
        if (
            2 <= len(same_type_same_room) <= 4
            and max(xs) - min(xs) < 0.6
            and (max(zs) - min(zs) < 0.6)
        ):
            ordered = sorted(
                same_type_same_room, key=lambda o: (-o["position"]["y"], o["objectId"])
            )
            labels = {
                2: ("top", "bottom"),
                3: ("top", "middle", "bottom"),
                4: ("top", "second", "third", "bottom"),
            }[len(ordered)]
            for label, member in zip(labels, ordered):
                if member["objectId"] == obj["objectId"]:
                    return (
                        f"the {label} {t} in the {self._room_types.get(room, 'room')}"
                    )
        return None

    def teleport(self, pose: dict) -> bool:
        event = self.sim.controller.step(
            action="Teleport",
            position=pose["position"],
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose.get("horizon", 0.0),
            standing=True,
        )
        return bool(event.metadata["lastActionSuccess"])

    def panorama(
        self, pose: dict, save_frames: bool, transform=None
    ) -> tuple[set[str], list[str]]:
        visible: set[str] = set()
        hashes: list[str] = []
        for rot in PANORAMA_ROTATIONS:
            event = self.sim.controller.step(
                action="Teleport",
                position=pose["position"],
                rotation={"x": 0, "y": (pose["rotation"] + rot) % 360, "z": 0},
                horizon=pose.get("horizon", 0.0),
                standing=True,
            )
            if not event.metadata["lastActionSuccess"]:
                continue
            visible |= {
                o["objectId"] for o in event.metadata["objects"] if o["visible"]
            }
            frame = event.frame if transform is None else transform(event.frame)
            fhash = frame_hash(frame)
            hashes.append(fhash)
            if save_frames:
                path = self.frames_dir / f"{fhash}.png"
                if not path.exists():
                    from PIL import Image

                    Image.fromarray(frame).save(path)
        self.teleport(pose)
        return (visible, hashes)

    def start_poses(self, count: int = 2) -> list[dict]:
        event = self.sim.controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        if not positions:
            return []
        rooms = sorted(self._room_types)
        poses = []
        for room_id in rooms:
            centroid = self.sim.room_centroid(room_id)
            if centroid is None:
                continue
            in_room = [
                p
                for p in positions
                if self.sim.room_of_point(p["x"], p["z"]) == room_id
            ]
            if not in_room:
                continue
            cx, cz = centroid
            best = min(
                in_room,
                key=lambda p: (
                    round((p["x"] - cx) ** 2 + (p["z"] - cz) ** 2, 6),
                    p["x"],
                    p["z"],
                ),
            )
            poses.append(
                {
                    "position": dict(best),
                    "rotation": self.rng.choice((0, 90, 180, 270)),
                    "horizon": 0.0,
                }
            )
            if len(poses) >= count:
                break
        return poses

    def scene_summary(
        self, visible_start: set[str], dark_room: str | None = None
    ) -> dict:
        """Rooms → receptacles with states; visible contents from the start
        pose only (hidden objects omitted — knowledge, not truth)."""
        objs = {o["objectId"]: o for o in self.sim.objects()}
        rooms = []
        for room in sorted(self.sim.rooms(), key=lambda r: r["id"]):
            rid = room["id"]
            receptacles = []
            for oid, o in sorted(objs.items()):
                if (
                    not o["receptacle"]
                    or type_name(o) in PSEUDO_TYPES
                    or oid.startswith(DOOR_PREFIX)
                    or (self._room_of(o) != rid)
                ):
                    continue
                seen = oid in visible_start
                receptacles.append(
                    {
                        "id": oid,
                        "type": type_name(o),
                        "name": self.display_name(o),
                        "openable": o["openable"],
                        "is_open": o["isOpen"] if o["openable"] else None,
                        "visible": seen,
                        "visible_contents": sorted(
                            (
                                type_name(objs[c])
                                for c in o.get("receptacleObjectIds") or []
                                if c in objs and c in visible_start
                            )
                        )
                        if seen
                        else [],
                    }
                )
            rooms.append(
                {
                    "id": rid,
                    "type": room.get("roomType", "room"),
                    "lit": rid != dark_room,
                    "receptacles": receptacles,
                }
            )
        return {"rooms": rooms}

    def swept_visible(self) -> set[str]:
        """Union of `visible` over camera horizons 0/30/60 at the current
        pose. Receptacle interiors and surface items sit below a level
        camera; a task's oracle plan gains a LookAt step to match."""
        agent = self.sim.agent_pose()
        union: set[str] = set()
        for horizon in (0, 30, 60):
            event = self.sim.controller.step(
                action="Teleport",
                position=agent["position"],
                rotation=agent["rotation"],
                horizon=horizon,
                standing=True,
            )
            if event.metadata["lastActionSuccess"]:
                union |= {
                    o["objectId"] for o in event.metadata["objects"] if o["visible"]
                }
        return union

    def reveal_info(self, receptacle_id: str) -> dict | None:
        """Interact pose, path anchor, and (for openables) open-state facts.

        Openables retry up to three candidate poses, nearest first: opening a
        fridge or cabinet fails when the agent stands in the door swing.
        """
        if receptacle_id in self._reveal_cache:
            return self._reveal_cache[receptacle_id]
        obj = self.sim.get_object(receptacle_id)
        if obj is None:
            return None
        event = self.sim.controller.step(
            action="GetInteractablePoses", objectId=receptacle_id
        )
        poses = event.metadata.get("actionReturn") or []
        if not event.metadata["lastActionSuccess"] or not poses:
            self._reveal_cache[receptacle_id] = None
            return None
        near = [p for p in poses if dist_xz(p, obj["position"]) <= REACH_M] or poses
        ranked = sorted(
            near,
            key=lambda p: (
                round(dist_xz(p, obj["position"]), 4),
                round(p["x"], 4),
                round(p["z"], 4),
                p.get("rotation", 0),
                p.get("horizon", 0),
            ),
        )
        needs_open = obj["openable"] and (not obj["isOpen"])
        info: dict | None = None
        for pose in ranked[:3] if needs_open else ranked[:1]:
            pose_rec = {
                "position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
                "rotation": pose.get("rotation", 0),
                "horizon": pose.get("horizon", 0),
            }
            if not self.teleport(pose_rec):
                continue
            meta_obj = self.sim.get_object(receptacle_id)
            candidate: dict = {
                "pose": pose_rec,
                "receptacle_id": receptacle_id,
                "visible_at_pose": meta_obj["visible"],
                "visible_set_at_pose": self.swept_visible(),
            }
            if not needs_open:
                info = candidate
                break
            self.teleport(pose_rec)
            event = self.sim.controller.step(
                action="OpenObject", objectId=receptacle_id
            )
            if event.metadata["lastActionSuccess"]:
                opened = self.sim.get_object(receptacle_id)
                candidate["open_ok"] = True
                candidate["contents_after_open"] = list(
                    opened.get("receptacleObjectIds") or []
                )
                candidate["visible_after_open"] = self.swept_visible()
                self.sim.controller.step(action="CloseObject", objectId=receptacle_id)
                info = candidate
                break
            candidate["open_ok"] = False
            info = candidate
        self._reveal_cache[receptacle_id] = info
        return info

    def nav_cost(self, start: dict, end: dict) -> float | None:
        try:
            event = self.sim.controller.step(
                action="GetShortestPathToPoint",
                position=start["position"],
                target=end["position"],
            )
            ret = event.metadata.get("actionReturn")
            if event.metadata["lastActionSuccess"] and ret and ret.get("corners"):
                return polyline_length(ret["corners"])
        except Exception:
            pass
        return None

    def _base_task(self, kind: str, query_type: str) -> dict:
        n = self.counter[f"{kind}:{query_type}"]
        self.counter[f"{kind}:{query_type}"] += 1
        return {
            "task_id": f"{self.split}{self.house_index:05d}-{kind}-{query_type}-{n}",
            "generator": GEN_VERSION,
            "split": self.split,
            "house_index": self.house_index,
            "house_mod": None,
            "query_type": query_type,
            "hiding_kinds": kind.split("+") if kind not in ("N", "STATE") else [kind],
            "trap": None,
        }

    def _prefer_yes(self, family: str) -> bool:
        """Steer answers toward 50/50 *within each (query, kind) family* —
        global-only balance lets single families drift, and a drifted
        family rewards blind guessing."""
        return self.answers[f"{family}:yes"] <= self.answers[f"{family}:no"]

    def _finalize(
        self,
        task: dict,
        start_pose: dict,
        visible_start: set[str],
        frames: list[str],
        oracle_plan: list[dict],
        oracle_cost: float,
    ) -> dict:
        if task.get("answer") in ("yes", "no") and (not task.get("trap")):
            family = f"{task['query_type']}|{'+'.join(task['hiding_kinds'])}"
            self.answers[f"{family}:{task['answer']}"] += 1
        task["start_pose"] = start_pose
        task["start_room"] = self.sim.room_of_point(
            start_pose["position"]["x"], start_pose["position"]["z"]
        )
        task["start_frames"] = frames
        task["visible_from_start"] = sorted(visible_start)
        task["scene_summary"] = getattr(self, "_current_summary", None)
        task["oracle_plan"] = oracle_plan
        task["oracle_cost"] = round(oracle_cost, 3)
        task["budget"] = round(3 * task["oracle_cost"] + 3, 3)
        task["depth"] = sum(
            (1 for a in oracle_plan if a["action"] in ("Open", "Toggle", "MoveAside"))
        ) + (
            1
            if any((a["action"] == "GoTo" for a in oracle_plan))
            and "D" in task["hiding_kinds"]
            else 0
        )
        return task

    def generate_from_pose(
        self, start_pose: dict, visible_start: set[str], frames: list[str]
    ) -> None:
        self._current_summary = self.scene_summary(visible_start)
        objs = {o["objectId"]: o for o in self.sim.objects()}
        start_room = self.sim.room_of_point(
            start_pose["position"]["x"], start_pose["position"]["z"]
        )

        def room_of(oid: str) -> str | None:
            return self._room_of(objs[oid])

        closed_receptacles = sorted(
            (
                oid
                for oid, o in objs.items()
                if o["openable"]
                and o["receptacle"]
                and (not o["isOpen"])
                and (not oid.startswith(DOOR_PREFIX))
                and (type_name(o) not in PSEUDO_TYPES)
            )
        )
        open_surfaces = sorted(
            (
                oid
                for oid, o in objs.items()
                if o["receptacle"]
                and (not o["openable"])
                and (type_name(o) not in PSEUDO_TYPES)
            )
        )
        self._gen_kind_C(
            objs,
            closed_receptacles,
            start_pose,
            visible_start,
            frames,
            start_room,
            room_of,
        )
        self._gen_kind_D(
            objs, open_surfaces, start_pose, visible_start, frames, start_room, room_of
        )
        self._gen_kind_N(objs, open_surfaces, start_pose, visible_start, frames)
        self._gen_state(objs, start_pose, visible_start, frames, room_of)
        self._gen_traps(
            objs, closed_receptacles, open_surfaces, start_pose, visible_start, frames
        )

    def _plan_goto_open(self, start_pose: dict, info: dict, rid: str):
        nav = self.nav_cost(start_pose, info["pose"])
        if nav is None:
            return (None, None)
        plan = [{"action": "GoTo", "target": rid}, {"action": "Open", "target": rid}]
        return (plan, 1.0 + 0.1 * nav + 1.0)

    def _plan_goto(self, start_pose: dict, info: dict, rid: str):
        nav = self.nav_cost(start_pose, info["pose"])
        if nav is None:
            return (None, None)
        return ([{"action": "GoTo", "target": rid}], 1.0 + 0.1 * nav)

    def _gen_kind_C(
        self,
        objs,
        closed_receptacles,
        start_pose,
        visible_start,
        frames,
        start_room,
        room_of,
    ) -> None:
        rem = self.remaining
        self.rng.shuffle(closed_receptacles)
        examined = 0
        for rid in closed_receptacles:
            if (
                rem["C_yes"] <= 0
                and rem["C_no"] <= 0
                and (rem["C+D"] <= 0)
                or examined >= self.exam_cap
            ):
                break
            rec = objs[rid]
            rec_name = self.display_name(rec)
            if rec_name is None:
                continue
            examined += 1
            info = self.reveal_info(rid)
            if info is None or not info.get("open_ok"):
                continue
            contents = [
                cid
                for cid in info.get("contents_after_open") or []
                if cid in objs
                and objs[cid]["pickupable"]
                and (cid in info.get("visible_after_open", set()))
            ]
            same_room = room_of(rid) == start_room
            kind = "C" if same_room else "C+D"
            if kind == "C+D" and rem["C+D"] <= 0:
                continue
            if (
                contents
                and (rem["C_yes"] > 0 or kind == "C+D")
                and self._prefer_yes(f"CHECK|{kind}")
            ):
                cid = sorted(contents)[0]
                if cid not in visible_start:
                    target_type = type_name(objs[cid])
                    plan, cost = self._plan_goto_open(start_pose, info, rid)
                    if plan:
                        plan = plan + [{"action": "LookAt", "target": rid}]
                        cost += 0.5
                        task = self._base_task(kind, "CHECK")
                        task.update(
                            query=f"Is there a {target_type} in {rec_name}?",
                            answer="yes",
                            target_object_id=cid,
                            target_type=target_type,
                            receptacle_id=rid,
                            receptacle_name=rec_name,
                            reveal_condition="target_visible",
                            checks={
                                "target_hidden_at_start": cid not in visible_start,
                                "revealed_after_plan": True,
                            },
                        )
                        self.tasks.append(
                            self._finalize(
                                task, start_pose, visible_start, frames, plan, cost
                            )
                        )
                        rem["C_yes" if kind == "C" else "C+D"] -= 1
                        continue
            if (rem["C_no"] > 0 if kind == "C" else rem["C+D"] > 0) and (
                not self._prefer_yes(f"CHECK|{kind}")
            ):
                inside_types = {
                    type_name(objs[c])
                    for c in info.get("contents_after_open") or []
                    if c in objs
                }
                visible_types = {type_name(objs[v]) for v in visible_start if v in objs}
                candidates = sorted(
                    (
                        t
                        for t, c in self._type_counts.items()
                        if c > 0
                        and t not in inside_types
                        and (t not in visible_types)
                        and (t not in PSEUDO_TYPES)
                        and any(
                            (
                                objs[o]["pickupable"]
                                for o in objs
                                if type_name(objs[o]) == t
                            )
                        )
                    )
                )
                if candidates:
                    target_type = self.rng.choice(candidates)
                    plan, cost = self._plan_goto_open(start_pose, info, rid)
                    if plan:
                        task = self._base_task("C", "CHECK")
                        task.update(
                            query=f"Is there a {target_type} in {rec_name}?",
                            answer="no",
                            target_object_id=None,
                            target_type=target_type,
                            receptacle_id=rid,
                            receptacle_name=rec_name,
                            reveal_condition="receptacle_open_visible",
                            checks={
                                "type_absent_from_receptacle": True,
                                "revealed_after_plan": True,
                            },
                        )
                        self.tasks.append(
                            self._finalize(
                                task, start_pose, visible_start, frames, plan, cost
                            )
                        )
                        rem["C_no" if kind == "C" else "C+D"] -= 1

    def _gen_kind_D(
        self,
        objs,
        open_surfaces,
        start_pose,
        visible_start,
        frames,
        start_room,
        room_of,
    ) -> None:
        surfaces = [s for s in open_surfaces if room_of(s) != start_room]
        self.rng.shuffle(surfaces)
        examined = 0
        for rid in surfaces:
            if self.remaining["D"] <= 0 or examined >= self.exam_cap:
                break
            if rid in visible_start:
                continue
            rec = objs[rid]
            rec_name = self.display_name(rec)
            if rec_name is None:
                continue
            examined += 1
            info = self.reveal_info(rid)
            if info is None or not info.get("visible_at_pose"):
                continue
            on_it = [
                cid
                for cid in rec.get("receptacleObjectIds") or []
                if cid in objs
                and objs[cid]["pickupable"]
                and (cid in info.get("visible_set_at_pose", set()))
                and (cid not in visible_start)
            ]
            plan, cost = self._plan_goto(start_pose, info, rid)
            if plan is None:
                continue
            if on_it and self._prefer_yes("CHECK|D"):
                cid = sorted(on_it)[0]
                target_type = type_name(objs[cid])
                answer, tid = ("yes", cid)
                plan = plan + [{"action": "LookAt", "target": rid}]
                cost += 0.5
                check = {"target_hidden_at_start": True, "revealed_after_plan": True}
            elif self._prefer_yes("CHECK|D"):
                continue
            else:
                on_types = {
                    type_name(objs[c])
                    for c in rec.get("receptacleObjectIds") or []
                    if c in objs
                }
                visible_types = {type_name(objs[v]) for v in visible_start if v in objs}
                candidates = sorted(
                    (
                        t
                        for t, c in self._type_counts.items()
                        if c > 0
                        and t not in on_types
                        and (t not in visible_types)
                        and (t not in PSEUDO_TYPES)
                    )
                )
                if not candidates:
                    continue
                target_type = self.rng.choice(candidates)
                answer, tid = ("no", None)
                check = {
                    "type_absent_from_receptacle": True,
                    "revealed_after_plan": True,
                }
            task = self._base_task("D", "CHECK")
            task.update(
                query=f"Is there a {target_type} on {rec_name}?",
                answer=answer,
                target_object_id=tid,
                target_type=target_type,
                receptacle_id=rid,
                receptacle_name=rec_name,
                reveal_condition="target_visible"
                if answer == "yes"
                else "receptacle_visible",
                checks=check,
            )
            self.tasks.append(
                self._finalize(task, start_pose, visible_start, frames, plan, cost)
            )
            self.remaining["D"] -= 1

    def _gen_kind_N(
        self, objs, open_surfaces, start_pose, visible_start, frames
    ) -> None:
        surfaces = [s for s in open_surfaces if s in visible_start]
        self.rng.shuffle(surfaces)
        for rid in surfaces:
            if self.remaining["N"] <= 0:
                break
            rec = objs[rid]
            rec_name = self.display_name(rec)
            if rec_name is None:
                continue
            on_it_visible = [
                cid
                for cid in rec.get("receptacleObjectIds") or []
                if cid in objs and cid in visible_start and objs[cid]["pickupable"]
            ]
            if on_it_visible and self._prefer_yes("CHECK|N"):
                cid = sorted(on_it_visible)[0]
                target_type = type_name(objs[cid])
                answer, tid = ("yes", cid)
            elif self._prefer_yes("CHECK|N"):
                continue
            else:
                on_types = {
                    type_name(objs[c])
                    for c in rec.get("receptacleObjectIds") or []
                    if c in objs
                }
                candidates = sorted(
                    (
                        t
                        for t, c in self._type_counts.items()
                        if c > 0 and t not in on_types and (t not in PSEUDO_TYPES)
                    )
                )
                if not candidates:
                    continue
                target_type = self.rng.choice(candidates)
                answer, tid = ("no", None)
            task = self._base_task("N", "CHECK")
            task.update(
                query=f"Is there a {target_type} on {rec_name}?",
                answer=answer,
                target_object_id=tid,
                target_type=target_type,
                receptacle_id=rid,
                receptacle_name=rec_name,
                reveal_condition="already_visible",
                checks={"receptacle_visible_at_start": True},
            )
            self.tasks.append(
                self._finalize(task, start_pose, visible_start, frames, [], 0.0)
            )
            self.remaining["N"] -= 1

    def _gen_state(self, objs, start_pose, visible_start, frames, room_of) -> None:
        stateful = sorted(
            (
                oid
                for oid, o in objs.items()
                if oid not in visible_start
                and (not oid.startswith(DOOR_PREFIX))
                and (type_name(o) not in PSEUDO_TYPES)
                and (o["openable"] or o["toggleable"])
            )
        )
        self.rng.shuffle(stateful)
        examined = 0
        for oid in stateful:
            if self.remaining["STATE"] <= 0 or examined >= self.exam_cap:
                break
            obj = objs[oid]
            name = self.display_name(obj)
            if name is None:
                continue
            examined += 1
            info = self.reveal_info(oid)
            if info is None or not info.get("visible_at_pose"):
                continue
            plan, cost = self._plan_goto(start_pose, info, oid)
            if plan is None:
                continue
            if obj["openable"]:
                field, state = ("isOpen", obj["isOpen"])
                adjective = ("open", "closed")
            else:
                field, state = ("isToggled", obj["isToggled"])
                adjective = ("turned on", "turned off")
            want_yes = self._prefer_yes("STATE|D")
            adjective = adjective[0 if state == want_yes else 1]
            answer = "yes" if want_yes else "no"
            task = self._base_task("STATE", "STATE")
            task["hiding_kinds"] = ["D"]
            task.update(
                query=f"Is {name} {adjective}?",
                answer=answer,
                target_object_id=oid,
                target_type=type_name(obj),
                state_field=field,
                state_value=bool(state),
                reveal_condition="target_visible",
                checks={"target_hidden_at_start": True, "revealed_after_plan": True},
            )
            self.tasks.append(
                self._finalize(task, start_pose, visible_start, frames, plan, cost)
            )
            self.remaining["STATE"] -= 1

    def _gen_traps(
        self, objs, closed_receptacles, open_surfaces, start_pose, visible_start, frames
    ) -> None:
        budget = self.remaining["TRAP"]
        lure = sorted((r for r in closed_receptacles if r in visible_start))
        if budget > 0 and lure:
            visible_items = sorted(
                (
                    oid
                    for oid, o in objs.items()
                    if oid in visible_start
                    and o["pickupable"]
                    and (o.get("parentReceptacles") or [])
                    and (type_name(o) not in PSEUDO_TYPES)
                )
            )
            for oid in visible_items:
                parent = (objs[oid].get("parentReceptacles") or [None])[0]
                if parent not in objs or parent not in visible_start:
                    continue
                pname = self.display_name(objs[parent])
                if pname is None:
                    continue
                target_type = type_name(objs[oid])
                task = self._base_task("N", "CHECK")
                task.update(
                    trap="T-visible",
                    trap_lure={"action": "Open", "target": lure[0]},
                    query=f"Is there a {target_type} on {pname}?",
                    answer="yes",
                    target_object_id=oid,
                    target_type=target_type,
                    receptacle_id=parent,
                    receptacle_name=pname,
                    reveal_condition="already_visible",
                    checks={"target_visible_at_start": True},
                )
                self.tasks.append(
                    self._finalize(task, start_pose, visible_start, frames, [], 0.0)
                )
                budget -= 1
                break
        if budget > 0:
            lit_lamps = sorted(
                (
                    oid
                    for oid, o in objs.items()
                    if o["toggleable"]
                    and "Lamp" in type_name(o)
                    and (oid in visible_start)
                )
            )
            if lit_lamps:
                for oid, o in sorted(objs.items()):
                    if (
                        oid not in visible_start
                        or not o["pickupable"]
                        or type_name(o) in PSEUDO_TYPES
                    ):
                        continue
                    parents = o.get("parentReceptacles") or []
                    if not parents or parents[0] not in visible_start:
                        continue
                    pname = self.display_name(objs.get(parents[0], {}) or {})
                    if pname is None:
                        continue
                    task = self._base_task("N", "CHECK")
                    task.update(
                        trap="T-lit",
                        trap_lure={"action": "Toggle", "target": lit_lamps[0]},
                        query=f"Is there a {type_name(o)} on {pname}?",
                        answer="yes",
                        target_object_id=oid,
                        target_type=type_name(o),
                        receptacle_id=parents[0],
                        receptacle_name=pname,
                        reveal_condition="already_visible",
                        checks={"target_visible_at_start": True, "room_lit": True},
                    )
                    self.tasks.append(
                        self._finalize(task, start_pose, visible_start, frames, [], 0.0)
                    )
                    budget -= 1
                    break
        if budget > 0:
            empties = sorted(
                (
                    oid
                    for oid, o in objs.items()
                    if o["openable"]
                    and o["receptacle"]
                    and o["isOpen"]
                    and (oid in visible_start)
                    and (not (o.get("receptacleObjectIds") or []))
                    and (not oid.startswith(DOOR_PREFIX))
                )
            )
            for rid in empties:
                rec_name = self.display_name(objs[rid])
                if rec_name is None:
                    continue
                candidates = sorted(
                    (
                        t
                        for t, c in self._type_counts.items()
                        if c > 0 and t not in PSEUDO_TYPES
                    )
                )
                target_type = self.rng.choice(candidates)
                task = self._base_task("N", "CHECK")
                task.update(
                    trap="T-open-empty",
                    trap_lure={"action": "Open", "target": rid},
                    query=f"Is there a {target_type} in {rec_name}?",
                    answer="no",
                    target_object_id=None,
                    target_type=target_type,
                    receptacle_id=rid,
                    receptacle_name=rec_name,
                    reveal_condition="already_visible",
                    checks={"receptacle_open_and_empty": True},
                )
                self.tasks.append(
                    self._finalize(task, start_pose, visible_start, frames, [], 0.0)
                )
                budget -= 1
                break
        if budget > 0:
            visible_closed = sorted(
                (r for r in closed_receptacles if r in visible_start)
            )
            for rid in visible_closed:
                r1_name = self.display_name(objs[rid])
                if r1_name is None:
                    continue
                for oid, o in sorted(objs.items()):
                    if (
                        oid not in visible_start
                        or not o["pickupable"]
                        or type_name(o) in PSEUDO_TYPES
                    ):
                        continue
                    parents = o.get("parentReceptacles") or []
                    if not parents or parents[0] == rid:
                        continue
                    if parents[0] not in visible_start:
                        continue
                    target_type = type_name(o)
                    if self._type_counts[target_type] != 1:
                        continue
                    task = self._base_task("N", "CHECK")
                    task.update(
                        trap="T-wrong-container",
                        trap_lure={"action": "Open", "target": rid},
                        query=f"Is there a {target_type} in {r1_name}?",
                        answer="no",
                        target_object_id=oid,
                        target_type=target_type,
                        receptacle_id=rid,
                        receptacle_name=r1_name,
                        reveal_condition="already_visible",
                        checks={"target_visible_elsewhere": True},
                    )
                    self.tasks.append(
                        self._finalize(task, start_pose, visible_start, frames, [], 0.0)
                    )
                    budget -= 1
                    break
                if budget <= 0:
                    break
        self.remaining["TRAP"] = budget


def generate_house(
    sim: EpactSim,
    split: str,
    house_index: int,
    seed: int,
    frames_dir: Path,
    quotas: dict | None = None,
    save_frames: bool = True,
) -> list[dict]:
    gen = HouseGenerator(sim, split, house_index, seed, frames_dir, quotas)
    pose_data = []
    for pose in gen.start_poses(count=2):
        visible, frames = gen.panorama(pose, save_frames=save_frames)
        gen.generate_from_pose(pose, visible, frames)
        pose_data.append((pose, visible, frames))
    return gen.tasks


TWIN_REVEAL = {
    "target_visible": "receptacle_visible",
    "receptacle_open_visible": "receptacle_open_visible",
    "receptacle_visible": "receptacle_visible",
    "lit_and_visible": "lit_and_visible",
}
STATE_FLIP = {
    ("isOpen", True): "CloseObject",
    ("isOpen", False): "OpenObject",
    ("isToggled", True): "ToggleObjectOff",
    ("isToggled", False): "ToggleObjectOn",
}
EMISSIVE_TYPES = {
    "FloorLamp",
    "DeskLamp",
    "TableLamp",
    "LightSwitch",
    "Candle",
    "Television",
    "Laptop",
    "CellPhone",
}


def _state_twin(task: dict) -> dict | None:
    """STATE queries ask about a property, not presence: the twin keeps the
    query verbatim and flips the world state via a privileged setup action,
    so both members ask the identical question of an identical-looking start
    state. (Deleting the object, as for presence tasks, would be nonsense.)"""
    field = task.get("state_field")
    value = task.get("state_value")
    if field is None or value is None:
        return None
    if task.get("target_type") in EMISSIVE_TYPES:
        return None
    action = STATE_FLIP.get((field, bool(value)))
    if action is None:
        return None
    twin = dict(task)
    twin["task_id"] = f"{task['task_id']}-twin"
    twin["answer"] = "no" if task["answer"] == "yes" else "yes"
    twin["state_value"] = not bool(value)
    mod = dict(task.get("house_mod") or {})
    mod["prep"] = list(mod.get("prep") or []) + [
        {"action": action, "objectId": task["target_object_id"], "forceAction": True}
    ]
    twin["house_mod"] = mod
    twin["pair_id"] = task["task_id"]
    twin["checks"] = {
        **(task.get("checks") or {}),
        "paired_twin_of": task["task_id"],
        "state_flipped": f"{field}:{value}->{not bool(value)}",
    }
    task["pair_id"] = task["task_id"]
    return twin


def emit_paired_twins(gen: "HouseGenerator") -> list[dict]:
    """Emit one answer-flipped twin per eligible task.

    Presence (CHECK) tasks twin by deleting the target; STATE tasks twin by
    flipping the queried property. Either way the pair shares house, pose and
    query string, so the query text carries zero information about the answer
    and only acting can separate the members.
    """
    objs = {o["objectId"]: o for o in gen.sim.objects()}
    twins = []
    for task in gen.tasks:
        if task.get("trap") or task.get("pair_id"):
            continue
        if task.get("query_type") == "STATE":
            twin = _state_twin(task)
            if twin is not None:
                twins.append(twin)
            continue
        if task["answer"] != "yes":
            continue
        target_id = task.get("target_object_id")
        rid = task.get("receptacle_id")
        ttype = task.get("target_type")
        if not target_id or not ttype or target_id not in objs:
            continue
        if task["reveal_condition"] not in TWIN_REVEAL:
            continue
        if target_id in set(task.get("visible_from_start") or []):
            continue
        drop = {
            oid
            for oid, o in objs.items()
            if type_name(o) == ttype
            and (
                oid == target_id
                or (
                    o.get("parentReceptacles")
                    and rid in (o.get("parentReceptacles") or [])
                )
            )
        }
        visible = set(task.get("visible_from_start") or [])
        if any(
            (
                type_name(objs[v]) == ttype
                for v in visible
                if v in objs and v not in drop
            )
        ):
            continue
        twin = dict(task)
        twin["task_id"] = f"{task['task_id']}-twin"
        twin["answer"] = "no"
        twin["target_object_id"] = None
        rec_obj = objs.get(rid)
        if rec_obj is not None and rec_obj.get("openable"):
            twin["reveal_condition"] = "receptacle_open_visible"
        else:
            twin["reveal_condition"] = TWIN_REVEAL[task["reveal_condition"]]
        twin["oracle_plan"] = [
            {**s, "target": rid} if s.get("target") in drop else s
            for s in task.get("oracle_plan") or []
        ]
        mod = dict(task.get("house_mod") or {})
        mod["removed_objects"] = sorted(drop)
        twin["house_mod"] = mod
        twin["pair_id"] = task["task_id"]
        twin["checks"] = {
            **(task.get("checks") or {}),
            "paired_twin_of": task["task_id"],
            "removed_instances": len(drop),
        }
        task["pair_id"] = task["task_id"]
        twins.append(twin)
    return twins


def _gen_kind_O(
    gen: HouseGenerator, start_pose: dict, visible_start: set[str], frames: list[str]
) -> None:
    """Occlusion: a pickupable on the same surface hides the target; the
    reveal is MoveAside. Verified by executing the move and diffing the
    visible set (best-effort restore afterwards)."""
    from epact.environment.actions import ActionAdapter

    sim = gen.sim
    adapter = ActionAdapter(sim)
    objs = {o["objectId"]: o for o in sim.objects()}
    start_room = sim.room_of_point(
        start_pose["position"]["x"], start_pose["position"]["z"]
    )
    gen._current_summary = gen.scene_summary(visible_start)
    surfaces = sorted(
        (
            oid
            for oid, o in objs.items()
            if o["receptacle"]
            and (not o["openable"])
            and (type_name(o) not in PSEUDO_TYPES)
            and (gen._room_of(o) == start_room)
            and (len(o.get("receptacleObjectIds") or []) >= 2)
        )
    )
    gen.rng.shuffle(surfaces)
    examined = 0
    for rid in surfaces:
        if gen.remaining["O"] <= 0 or examined >= gen.exam_cap:
            break
        rec = objs[rid]
        rec_name = gen.display_name(rec)
        if rec_name is None:
            continue
        info = gen.reveal_info(rid)
        if info is None or not info.get("visible_at_pose"):
            continue
        examined += 1
        pose_visible = info["visible_set_at_pose"]
        contents = [
            c
            for c in rec.get("receptacleObjectIds") or []
            if c in objs and objs[c]["pickupable"]
        ]
        hidden = [
            c for c in contents if c not in pose_visible and c not in visible_start
        ]
        occluders = sorted((c for c in contents if c in pose_visible))
        if not hidden or not occluders:
            print(
                f"[O] {gen.split}{gen.house_index} {rid}: hidden={len(hidden)} occluders={len(occluders)} of {len(contents)}"
            )
            continue
        target_id, occluder_id = (sorted(hidden)[0], occluders[0])
        if not gen.teleport(info["pose"]):
            continue
        occluder_pos = dict(objs[occluder_id]["position"])
        result = adapter.execute("MoveAside", target=occluder_id)
        if not result.ok:
            continue
        revealed = gen.swept_visible()
        ev = sim.controller.step(action="PickupObject", objectId=occluder_id)
        if ev.metadata["lastActionSuccess"]:
            ev = sim.controller.step(action="PutObject", objectId=rid, forceAction=True)
            if not ev.metadata["lastActionSuccess"]:
                sim.controller.step(action="DropHandObject", forceAction=True)
        if target_id not in revealed:
            print(
                f"[O] {gen.split}{gen.house_index} {rid}: target not revealed by MoveAside"
            )
            continue
        nav = gen.nav_cost(start_pose, info["pose"])
        if nav is None:
            continue
        plan = [
            {"action": "GoTo", "target": occluder_id},
            {"action": "MoveAside", "target": occluder_id},
            {"action": "LookAt", "target": rid},
        ]
        cost = 1.0 + 0.1 * nav + 2.0 + 0.5
        if gen._prefer_yes("CHECK|O"):
            target_type = type_name(objs[target_id])
            answer, tid, condition = ("yes", target_id, "target_visible")
        else:
            visible_types = {type_name(objs[v]) for v in visible_start if v in objs}
            on_types = {type_name(objs[c]) for c in contents}
            pool = sorted(
                (
                    t
                    for t, c in gen._type_counts.items()
                    if c > 0
                    and t not in on_types
                    and (t not in visible_types)
                    and (t not in PSEUDO_TYPES)
                )
            )
            if not pool:
                continue
            target_type = gen.rng.choice(pool)
            answer, tid, condition = ("no", None, "occluder_moved")
        task = gen._base_task("O", "CHECK")
        task["hiding_kinds"] = ["O"]
        task.update(
            query=f"Is there a {target_type} on {rec_name}?",
            answer=answer,
            target_object_id=tid,
            target_type=target_type,
            receptacle_id=rid,
            receptacle_name=rec_name,
            occluder_id=occluder_id,
            occluder_position=occluder_pos,
            reveal_condition=condition,
            checks={"target_hidden_at_start": True, "revealed_after_moveaside": True},
        )
        gen.tasks.append(
            gen._finalize(task, start_pose, visible_start, frames, plan, cost)
        )
        gen.remaining["O"] -= 1


def generate_house_Ddoor(
    sim: EpactSim,
    split: str,
    house_index: int,
    seed: int,
    frames_dir: Path,
    house: dict,
    save_frames: bool = True,
) -> list[dict]:
    """D+door: the target room's only door starts closed (house-JSON
    openness=0). Poses and nav costs are computed in the PRISTINE scene —
    the navmesh is baked at load, so a closed-door scene has no path into
    the sealed room and every navmesh query there returns None. Geometry is
    identical with the door open. The modified scene is used only for the
    start panorama and the open-verification."""
    rng = random.Random(f"{seed}:door:{split}:{house_index}")
    links_by_room: dict[str, list[dict]] = {}
    for door in house.get("doors", []):
        sealed = door.get("openness") == 0 and (not door.get("openable"))
        if sealed:
            continue
        for room in (door["room0"], door["room1"]):
            links_by_room.setdefault(room, []).append(door)
    candidates = [
        (room, doors[0])
        for room, doors in sorted(links_by_room.items())
        if len(doors) == 1
        and doors[0].get("openable")
        and str(doors[0].get("assetId", "")).startswith("Doorway_Double")
    ]
    if not candidates:
        return []
    rng.shuffle(candidates)
    sim.reset(house)
    pristine = HouseGenerator(sim, split, house_index, seed, frames_dir)
    objs = {o["objectId"]: o for o in sim.objects()}
    event = sim.controller.step(action="GetReachablePositions")
    positions = event.metadata.get("actionReturn") or []
    prepared = None
    for room_b, door in candidates[:3]:
        door_obj = objs.get(door["id"])
        if door_obj is None:
            continue
        room_a = door["room0"] if door["room1"] == room_b else door["room1"]
        surfaces = sorted(
            (
                oid
                for oid, o in objs.items()
                if o["receptacle"]
                and type_name(o) not in PSEUDO_TYPES
                and (not oid.startswith(DOOR_PREFIX))
                and (pristine._room_of(o) == room_b)
                and (pristine.display_name(o) is not None)
            )
        )
        in_a = [p for p in positions if sim.room_of_point(p["x"], p["z"]) == room_a]
        if not surfaces or not in_a:
            continue
        ca = sim.room_centroid(room_a)
        best = min(
            in_a,
            key=lambda p: (
                round((p["x"] - ca[0]) ** 2 + (p["z"] - ca[1]) ** 2, 6),
                p["x"],
                p["z"],
            ),
        )
        start_pose = {
            "position": dict(best),
            "rotation": rng.choice((0, 90, 180, 270)),
            "horizon": 0.0,
        }
        event = sim.controller.step(action="GetInteractablePoses", objectId=door["id"])
        poses = event.metadata.get("actionReturn") or []
        near = sorted(
            (p for p in poses if sim.room_of_point(p["x"], p["z"]) == room_a),
            key=lambda p: (-round(dist_xz(p, door_obj["position"]), 4), p["x"]),
        )
        if not near:
            print(f"[Ddoor] {split}{house_index} {door['id']}: no near-side pose")
            continue
        door_poses = [
            {
                "position": {"x": p["x"], "y": p["y"], "z": p["z"]},
                "rotation": p.get("rotation", 0),
                "horizon": p.get("horizon", 0),
            }
            for p in near[:3]
        ]
        surface_infos = []
        for rid in surfaces[:2]:
            info = pristine.reveal_info(rid)
            if info is None or not info.get("visible_at_pose"):
                print(f"[Ddoor] {split}{house_index} {rid}: no reveal pose (pristine)")
                continue
            nav1 = pristine.nav_cost(start_pose, door_poses[0])
            nav2 = pristine.nav_cost(door_poses[0], info["pose"])
            if nav1 is None or nav2 is None:
                print(f"[Ddoor] {split}{house_index} {rid}: pristine nav failed")
                continue
            surface_infos.append((rid, info, nav1, nav2))
        if surface_infos:
            prepared = (room_b, door, start_pose, door_poses, surface_infos)
            break
    if prepared is None:
        return []
    room_b, door, start_pose, door_poses, surface_infos = prepared
    modified = _copy.deepcopy(house)
    for d in modified["doors"]:
        if d["id"] == door["id"]:
            d["openness"] = 0
    sim.reset(modified)
    gen = HouseGenerator(sim, split, house_index, seed, frames_dir)
    objs = {o["objectId"]: o for o in sim.objects()}
    door_obj = objs.get(door["id"])
    if door_obj is None or door_obj["isOpen"]:
        print(f"[Ddoor] {split}{house_index} {door['id']}: openness mod ineffective")
        return []
    if not gen.teleport(start_pose):
        return []
    visible_start, frames = gen.panorama(start_pose, save_frames=save_frames)
    gen._current_summary = gen.scene_summary(visible_start)
    print(
        f"[Ddoor] {split}{house_index} trying {door['id']} ({door.get('assetId')})",
        flush=True,
    )
    opened = False
    for door_pose in door_poses:
        if not gen.teleport(door_pose):
            continue
        event = sim.controller.step(action="OpenObject", objectId=door["id"])
        if event.metadata["lastActionSuccess"]:
            opened = True
            break
    if not opened:
        print(f"[Ddoor] {split}{house_index} {door['id']}: could not open")
        return []
    tasks: list[dict] = []
    for rid, info, nav1, nav2 in surface_infos:
        rec = objs[rid]
        rec_name = gen.display_name(rec)
        if rec_name is None or rid in visible_start:
            continue
        if not gen.teleport(info["pose"]):
            continue
        seen_now = gen.swept_visible()
        if rid not in seen_now:
            print(f"[Ddoor] {split}{house_index} {rid}: not visible after open")
            continue
        plan = [
            {"action": "GoTo", "target": door["id"]},
            {"action": "Open", "target": door["id"]},
            {"action": "GoTo", "target": rid},
            {"action": "LookAt", "target": rid},
        ]
        cost = 1.0 + 0.1 * nav1 + 1.0 + (1.0 + 0.1 * nav2) + 0.5
        contents = [
            c
            for c in rec.get("receptacleObjectIds") or []
            if c in objs
            and objs[c]["pickupable"]
            and (c not in visible_start)
            and (c in seen_now)
        ]
        if contents and gen._prefer_yes("CHECK|D+door"):
            cid = sorted(contents)[0]
            target_type = type_name(objs[cid])
            answer, tid, condition = ("yes", cid, "target_visible")
        elif gen._prefer_yes("CHECK|D+door"):
            continue
        else:
            visible_types = {type_name(objs[v]) for v in visible_start if v in objs}
            on_types = {
                type_name(objs[c])
                for c in rec.get("receptacleObjectIds") or []
                if c in objs
            }
            pool = sorted(
                (
                    t
                    for t, c in gen._type_counts.items()
                    if c > 0
                    and t not in on_types
                    and (t not in visible_types)
                    and (t not in PSEUDO_TYPES)
                )
            )
            if not pool:
                continue
            target_type = rng.choice(pool)
            answer, tid, condition = ("no", None, "receptacle_visible")
        preposition = "in" if rec["openable"] else "on"
        task = gen._base_task("DDOOR", "CHECK")
        task["hiding_kinds"] = ["D", "door"]
        task.update(
            house_mod={"closed_doors": [door["id"]]},
            query=f"Is there a {target_type} {preposition} {rec_name}?",
            answer=answer,
            target_object_id=tid,
            target_type=target_type,
            receptacle_id=rid,
            receptacle_name=rec_name,
            door_id=door["id"],
            reveal_condition=condition,
            checks={
                "door_closed_at_start": True,
                "target_hidden_at_start": True,
                "revealed_after_plan": True,
            },
        )
        tasks.append(gen._finalize(task, start_pose, visible_start, frames, plan, cost))
    return tasks


DARKNESS_SCALE = 0.04
LIT_ROOM_LUMINANCE_MIN = 60.0
L_QUOTA = 2


def darken(frame):
    import numpy as _np

    return (_np.asarray(frame, dtype=_np.float64) * DARKNESS_SCALE).astype("uint8")


def _mean_luminance(frame) -> float:
    import numpy as _np

    return float(_np.asarray(frame, dtype=_np.float64).mean())


def _yaw_to(src: dict, dst: dict) -> float:
    import math as _math

    return _math.degrees(_math.atan2(dst["x"] - src["x"], dst["z"] - src["z"])) % 360.0


def _room_lights(house: dict, sim: EpactSim, room_id: str) -> list[str]:
    out = []
    for light in house.get("proceduralParameters", {}).get("lights", []):
        if light.get("type") != "point":
            continue
        pos = light.get("position", {})
        if sim.room_of_point(pos.get("x", 0), pos.get("z", 0)) == room_id:
            out.append(light["id"])
    return out


def _toggle_pose(gen: HouseGenerator, lamp_id: str) -> dict | None:
    """A pose from which the lamp actually toggles; tries three candidates."""
    obj = gen.sim.get_object(lamp_id)
    if obj is None:
        return None
    event = gen.sim.controller.step(action="GetInteractablePoses", objectId=lamp_id)
    poses = event.metadata.get("actionReturn") or []
    if not event.metadata["lastActionSuccess"] or not poses:
        return None
    near = [p for p in poses if dist_xz(p, obj["position"]) <= REACH_M] or poses
    ranked = sorted(
        near,
        key=lambda p: (
            round(dist_xz(p, obj["position"]), 4),
            round(p["x"], 4),
            round(p["z"], 4),
            p.get("rotation", 0),
        ),
    )
    for pose in ranked[:5]:
        pose_rec = {
            "position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            "rotation": pose.get("rotation", 0),
            "horizon": pose.get("horizon", 0),
        }
        if not gen.teleport(pose_rec):
            continue
        event = gen.sim.controller.step(action="ToggleObjectOn", objectId=lamp_id)
        if event.metadata["lastActionSuccess"]:
            gen.sim.controller.step(action="ToggleObjectOff", objectId=lamp_id)
            return pose_rec
    return None


def generate_house_L(
    sim: EpactSim,
    split: str,
    house_index: int,
    seed: int,
    frames_dir: Path,
    house: dict,
    save_frames: bool = True,
) -> list[dict]:
    """Generate L tasks for one house. Resets the sim to a modified house;
    the caller must reset again before reusing the pristine one."""
    rng = random.Random(f"{seed}:L:{split}:{house_index}")
    objs = {o["objectId"]: o for o in sim.objects()}
    lamps_by_room: dict[str, list[str]] = {}
    for oid, o in objs.items():
        if o["toggleable"] and "Lamp" in type_name(o):
            room = sim.room_of_point(o["position"]["x"], o["position"]["z"])
            if room:
                lamps_by_room.setdefault(room, []).append(oid)
    candidates = []
    for room_id, lamps in sorted(lamps_by_room.items()):
        light_ids = _room_lights(house, sim, room_id)
        if not light_ids:
            continue
        surfaces = sorted(
            (
                oid
                for oid, o in objs.items()
                if o["receptacle"]
                and (not o["openable"])
                and (type_name(o) not in PSEUDO_TYPES)
                and (
                    sim.room_of_point(o["position"]["x"], o["position"]["z"]) == room_id
                )
                and (o.get("receptacleObjectIds") or [])
            )
        )
        if surfaces:
            candidates.append((room_id, sorted(lamps), light_ids, surfaces))
    if not candidates:
        return []
    for room_id, lamps, light_ids, surfaces in candidates:
        tasks = _try_L_room(
            sim,
            split,
            house_index,
            seed,
            frames_dir,
            house,
            rng,
            room_id,
            lamps,
            light_ids,
            surfaces,
            save_frames,
        )
        if tasks:
            return tasks
    return []


def _try_L_room(
    sim,
    split,
    house_index,
    seed,
    frames_dir,
    house,
    rng,
    room_id,
    lamps,
    light_ids,
    surfaces,
    save_frames,
) -> list[dict]:
    modified = _copy.deepcopy(house)
    modified["proceduralParameters"]["lights"] = [
        l
        for l in modified["proceduralParameters"]["lights"]
        if l["id"] not in light_ids
    ]
    sim.reset(modified)
    gen = HouseGenerator(sim, split, house_index, seed, frames_dir)
    objs = {o["objectId"]: o for o in sim.objects()}
    prep = []
    for lamp_id in lamps:
        lamp = objs.get(lamp_id)
        if lamp is not None and lamp["isToggled"]:
            event = sim.controller.step(action="ToggleObjectOff", objectId=lamp_id)
            if event.metadata["lastActionSuccess"]:
                prep.append({"action": "ToggleObjectOff", "objectId": lamp_id})
    house_mod = {
        "removed_lights": light_ids,
        "prep": prep,
        "dark_room": room_id,
        "darkness": {
            "scale": DARKNESS_SCALE,
            "lit_when_any_on": lamps,
            "note": "renderer has fixed ambient; harness darkens frames while all room light sources are off",
        },
    }
    tasks: list[dict] = []
    lamp_id, lamp_name, lamp_pose = (None, None, None)
    for candidate_lamp in lamps:
        if candidate_lamp not in objs:
            continue
        name = gen.display_name(objs[candidate_lamp])
        if name is None:
            continue
        pose = _toggle_pose(gen, candidate_lamp)
        if pose is None:
            print(f"[L] {split}{house_index} {room_id}: {candidate_lamp} untogglable")
            continue
        lamp_id, lamp_name, lamp_pose = (candidate_lamp, name, pose)
        break
    if lamp_id is None:
        print(f"[L] {split}{house_index} {room_id}: no usable lamp")
        return []
    nameable = [rid for rid in surfaces if gen.display_name(objs[rid]) is not None]
    if not nameable:
        print(f"[L] {split}{house_index} {room_id}: no nameable surface")
    for rid in nameable[: L_QUOTA + 2]:
        if len(tasks) >= L_QUOTA:
            break
        rec = objs[rid]
        rec_name = gen.display_name(rec)
        event = sim.controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        in_room = [
            p
            for p in positions
            if sim.room_of_point(p["x"], p["z"]) == room_id
            and dist_xz(p, rec["position"]) > 1.0
        ]
        if not in_room:
            print(f"[L] {split}{house_index} {rid}: no start position in room")
            continue
        best = min(
            in_room,
            key=lambda p: (round(dist_xz(p, rec["position"]), 4), p["x"], p["z"]),
        )
        yaw = _yaw_to(best, rec["position"])
        start_pose = {"position": dict(best), "rotation": round(yaw, 2), "horizon": 0.0}
        if not gen.teleport(start_pose):
            print(f"[L] {split}{house_index} {rid}: start teleport failed")
            continue
        raw_lum = _mean_luminance(sim.frame)
        dark_lum = _mean_luminance(darken(sim.frame))
        if not sim.get_object(rid)["visible"]:
            print(f"[L] {split}{house_index} {rid}: surface not in view from start")
            continue
        visible_start, frames = gen.panorama(
            start_pose, save_frames=save_frames, transform=darken
        )
        gen._current_summary = gen.scene_summary(visible_start, dark_room=room_id)
        back = gen.reveal_info(rid)
        if back is None or not back.get("visible_at_pose"):
            print(f"[L] {split}{house_index} {rid}: no reveal pose")
            continue
        nav1 = gen.nav_cost(start_pose, lamp_pose)
        nav2 = gen.nav_cost(lamp_pose, back["pose"])
        if nav1 is None or nav2 is None:
            print(f"[L] {split}{house_index} {rid}: nav cost failed")
            continue
        plan = [
            {"action": "GoTo", "target": lamp_id},
            {"action": "Toggle", "target": lamp_id},
            {"action": "GoTo", "target": rid},
        ]
        cost = 1.0 + 0.1 * nav1 + 1.0 + (1.0 + 0.1 * nav2)
        contents = [
            c
            for c in rec.get("receptacleObjectIds") or []
            if c in objs and objs[c]["pickupable"]
        ]
        if contents and gen._prefer_yes("CHECK|L"):
            cid = sorted(contents)[0]
            target_type = type_name(objs[cid])
            answer, tid = ("yes", cid)
        elif gen._prefer_yes("CHECK|L"):
            continue
        else:
            on_types = {
                type_name(objs[c])
                for c in rec.get("receptacleObjectIds") or []
                if c in objs
            }
            visible_types = {type_name(objs[v]) for v in visible_start if v in objs}
            type_pool = sorted(
                (
                    t
                    for t, c in gen._type_counts.items()
                    if c > 0
                    and t not in on_types
                    and (t not in visible_types)
                    and (t not in PSEUDO_TYPES)
                )
            )
            if not type_pool:
                continue
            target_type = rng.choice(type_pool)
            answer, tid = ("no", None)
        task = gen._base_task("L", "CHECK")
        task["hiding_kinds"] = ["L"]
        task.update(
            house_mod=house_mod,
            query=f"Is there a {target_type} on {rec_name}?",
            answer=answer,
            target_object_id=tid,
            target_type=target_type,
            receptacle_id=rid,
            receptacle_name=rec_name,
            lamp_id=lamp_id,
            reveal_condition="lit_and_visible",
            checks={
                "raw_luminance": round(raw_lum, 2),
                "darkened_luminance": round(dark_lum, 2),
                "surface_visible_when_lit": True,
                "revealed_after_plan": True,
            },
        )
        tasks.append(gen._finalize(task, start_pose, visible_start, frames, plan, cost))
    closed_in_room = sorted(
        (
            oid
            for oid, o in objs.items()
            if o["openable"]
            and o["receptacle"]
            and (not o["isOpen"])
            and (not oid.startswith(DOOR_PREFIX))
            and (type_name(o) not in PSEUDO_TYPES)
            and (
                gen.sim.room_of_point(o["position"]["x"], o["position"]["z"]) == room_id
            )
            and (gen.display_name(o) is not None)
        )
    )
    for rid in closed_in_room[:3]:
        if any((t["hiding_kinds"] == ["L", "C"] for t in tasks)):
            break
        rec = objs[rid]
        rec_name = gen.display_name(rec)
        info = gen.reveal_info(rid)
        if info is None or not info.get("open_ok"):
            continue
        rc_contents = [
            c
            for c in info.get("contents_after_open") or []
            if c in objs
            and objs[c]["pickupable"]
            and (c in info.get("visible_after_open", set()))
        ]
        in_room_positions = [
            p
            for p in gen.sim.controller.step(
                action="GetReachablePositions"
            ).metadata.get("actionReturn")
            or []
            if gen.sim.room_of_point(p["x"], p["z"]) == room_id
        ]
        if not in_room_positions:
            break
        best = min(
            in_room_positions,
            key=lambda p: (round(dist_xz(p, rec["position"]), 4), p["x"], p["z"]),
        )
        lc_start = {
            "position": dict(best),
            "rotation": round(_yaw_to(best, rec["position"]), 2),
            "horizon": 0.0,
        }
        if not gen.teleport(lc_start):
            continue
        lc_visible, lc_frames = gen.panorama(
            lc_start, save_frames=save_frames, transform=darken
        )
        gen._current_summary = gen.scene_summary(lc_visible, dark_room=room_id)
        nav1 = gen.nav_cost(lc_start, lamp_pose)
        nav2 = gen.nav_cost(lamp_pose, info["pose"])
        if nav1 is None or nav2 is None:
            continue
        plan = [
            {"action": "GoTo", "target": lamp_id},
            {"action": "Toggle", "target": lamp_id},
            {"action": "GoTo", "target": rid},
            {"action": "Open", "target": rid},
            {"action": "LookAt", "target": rid},
        ]
        cost = 1.0 + 0.1 * nav1 + 1.0 + (1.0 + 0.1 * nav2) + 1.0 + 0.5
        if rc_contents and gen._prefer_yes("CHECK|L+C"):
            cid = sorted(rc_contents)[0]
            target_type = type_name(objs[cid])
            answer, tid = ("yes", cid)
        elif gen._prefer_yes("CHECK|L+C"):
            continue
        else:
            visible_types = {type_name(objs[v]) for v in lc_visible if v in objs}
            inside_types = {
                type_name(objs[c])
                for c in info.get("contents_after_open") or []
                if c in objs
            }
            pool = sorted(
                (
                    t
                    for t, c in gen._type_counts.items()
                    if c > 0
                    and t not in inside_types
                    and (t not in visible_types)
                    and (t not in PSEUDO_TYPES)
                )
            )
            if not pool:
                continue
            target_type = rng.choice(pool)
            answer, tid = ("no", None)
        task = gen._base_task("LC", "CHECK")
        task["hiding_kinds"] = ["L", "C"]
        task.update(
            house_mod=house_mod,
            query=f"Is there a {target_type} in {rec_name}?",
            answer=answer,
            target_object_id=tid,
            target_type=target_type,
            receptacle_id=rid,
            receptacle_name=rec_name,
            lamp_id=lamp_id,
            reveal_condition="target_visible"
            if answer == "yes"
            else "receptacle_open_visible",
            checks={"dark_and_closed_at_start": True, "revealed_after_plan": True},
        )
        tasks.append(gen._finalize(task, lc_start, lc_visible, lc_frames, plan, cost))
    if tasks and nameable:
        outside = [
            p
            for p in gen.start_poses(count=4)
            if gen.sim.room_of_point(p["position"]["x"], p["position"]["z"]) != room_id
        ]
        for od_start in outside[:1]:
            rid = nameable[0]
            rec = objs[rid]
            rec_name = gen.display_name(rec)
            back = gen.reveal_info(rid)
            if back is None or rec_name is None:
                break
            if not gen.teleport(od_start):
                break
            od_visible, od_frames = gen.panorama(od_start, save_frames=save_frames)
            gen._current_summary = gen.scene_summary(od_visible, dark_room=room_id)
            if rid in od_visible:
                break
            nav1 = gen.nav_cost(od_start, lamp_pose)
            nav2 = gen.nav_cost(lamp_pose, back["pose"])
            if nav1 is None or nav2 is None:
                break
            plan = [
                {"action": "GoTo", "target": lamp_id},
                {"action": "Toggle", "target": lamp_id},
                {"action": "GoTo", "target": rid},
            ]
            cost = 1.0 + 0.1 * nav1 + 1.0 + (1.0 + 0.1 * nav2)
            ld_contents = [
                c
                for c in rec.get("receptacleObjectIds") or []
                if c in objs and objs[c]["pickupable"] and (c not in od_visible)
            ]
            if ld_contents and gen._prefer_yes("CHECK|L+D"):
                cid = sorted(ld_contents)[0]
                target_type = type_name(objs[cid])
                answer, tid = ("yes", cid)
            elif gen._prefer_yes("CHECK|L+D"):
                break
            else:
                visible_types = {type_name(objs[v]) for v in od_visible if v in objs}
                on_types = {
                    type_name(objs[c])
                    for c in rec.get("receptacleObjectIds") or []
                    if c in objs
                }
                pool = sorted(
                    (
                        t
                        for t, c in gen._type_counts.items()
                        if c > 0
                        and t not in on_types
                        and (t not in visible_types)
                        and (t not in PSEUDO_TYPES)
                    )
                )
                if not pool:
                    break
                target_type = rng.choice(pool)
                answer, tid = ("no", None)
            task = gen._base_task("LD", "CHECK")
            task["hiding_kinds"] = ["L", "D"]
            task.update(
                house_mod=house_mod,
                query=f"Is there a {target_type} on {rec_name}?",
                answer=answer,
                target_object_id=tid,
                target_type=target_type,
                receptacle_id=rid,
                receptacle_name=rec_name,
                lamp_id=lamp_id,
                reveal_condition="lit_and_visible"
                if answer == "no"
                else "target_visible",
                checks={"dark_room_elsewhere": True, "revealed_after_plan": True},
            )
            tasks.append(
                gen._finalize(task, od_start, od_visible, od_frames, plan, cost)
            )
    if tasks:
        lit_rooms = [r["id"] for r in sim.rooms() if r["id"] != room_id]
        for pose in gen.start_poses(count=3):
            p_room = sim.room_of_point(pose["position"]["x"], pose["position"]["z"])
            if p_room not in lit_rooms:
                continue
            if not gen.teleport(pose):
                continue
            if _mean_luminance(sim.frame) < LIT_ROOM_LUMINANCE_MIN:
                continue
            visible_start, frames = gen.panorama(pose, save_frames=save_frames)
            gen._current_summary = gen.scene_summary(visible_start, dark_room=room_id)
            objs2 = {o["objectId"]: o for o in sim.objects()}
            for oid in sorted(visible_start):
                o = objs2.get(oid)
                if o is None or not o["pickupable"] or type_name(o) in PSEUDO_TYPES:
                    continue
                parents = o.get("parentReceptacles") or []
                if not parents or parents[0] not in visible_start:
                    continue
                pname = gen.display_name(objs2.get(parents[0], {}) or {})
                if pname is None:
                    continue
                task = gen._base_task("N", "CHECK")
                task["task_id"] = task["task_id"].replace("-N-", "-NLdark-")
                task.update(
                    house_mod=house_mod,
                    trap="T-dark-irrelevant",
                    trap_lure={"action": "Toggle", "target": lamp_id},
                    query=f"Is there a {type_name(o)} on {pname}?",
                    answer="yes",
                    target_object_id=oid,
                    target_type=type_name(o),
                    receptacle_id=parents[0],
                    receptacle_name=pname,
                    reveal_condition="already_visible",
                    checks={"target_visible_at_start": True},
                )
                tasks.append(gen._finalize(task, pose, visible_start, frames, [], 0.0))
                break
            break
    return tasks
